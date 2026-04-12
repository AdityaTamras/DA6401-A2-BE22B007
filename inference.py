import argparse
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import wandb
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss

BREED_NAMES=["Abyssinian", "american_bulldog", "american_pit_bull_terrier", "basset_hound", "beagle", "Bengal", "Birman", "Bombay", "boxer", "British_Shorthair", "chihuahua", "Egyptian_Mau", "english_cocker_spaniel", "english_setter", "german_shorthaired", "great_pyrenees", "havanese", "japanese_chin", "keeshond", "leonberger", "Maine_Coon", "miniature_pinscher", "newfoundland", "Persian", "pomeranian", "pug", "Ragdoll", "Russian_Blue", "saint_bernard", "samoyed", "scottish_terrier", "shiba_inu", "Siamese", "Sphynx", "staffordshire_bull_terrier", "wheaten_terrier", "yorkshire_terrier"]

MASK_COLOURS=np.array([[0, 255, 0], [0, 0, 0], [255, 255, 0]], dtype=np.uint8)

def get_args():
    parser = argparse.ArgumentParser(description="DA6401 Assignment 2 Inference")

    parser.add_argument('--mode', type=str, choices=['evaluate', 'predict', 'feature_maps'], default='evaluate')
    parser.add_argument('--log_mode', type=str, choices=['all', 'detection', 'segmentation'], default='all')
    parser.add_argument('--classifier_path', type=str, default='checkpoints/classifier.pth')
    parser.add_argument('--localizer_path', type=str, default='checkpoints/localizer.pth')
    parser.add_argument('--unet_path', type=str, default='checkpoints/unet.pth')

    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--val_frac', type=float, default=0.15)
    parser.add_argument('--test_frac', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image_paths', type=str, nargs='+', default=[])
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--wandb_entity', type=str, default='be22b007-iit-madras')
    parser.add_argument('--wandb_project', type=str, default='DA6401-A2')
    parser.add_argument('--wandb_run_name', type=str, default='inference')

    return parser.parse_args()

def get_device():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def load_multitask_model(args, device):
    model = MultiTaskPerceptionModel(num_classes=37, seg_classes=3, in_channels=3, classifier_path=args.classifier_path, localizer_path=args.localizer_path, unet_path=args.unet_path).to(device)
    model.eval()
    return model

def load_classifier(classifier_path, device):
    model=VGG11Classifier(num_classes=37)
    state=torch.load(classifier_path, map_location='cpu')
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, img_size):
    original_pil=Image.open(image_path).convert('RGB')
    transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=OxfordIIITPetDataset.IMGNET_MEAN, std=OxfordIIITPetDataset.IMGNET_STD),
        ToTensorV2()
    ])
    np_image=np.array(original_pil.resize((img_size, img_size)))
    result=transform(image=np_image)
    tensor=result['image'].unsqueeze(0)
    return original_pil, tensor

def denormalise(tensor):
    mean=torch.tensor(OxfordIIITPetDataset.IMGNET_MEAN).view(3,1,1)
    std=torch.tensor(OxfordIIITPetDataset.IMGNET_STD).view(3,1,1)
    img=tensor.cpu()*std+mean
    img=img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img*255).astype(np.uint8)

def apply_colour_map(mask_np):
    H, W=mask_np.shape
    rgb=np.zeros((H, W, 3), dtype=np.uint8)
    for c, colour in enumerate(MASK_COLOURS):
        rgb[mask_np==c]=colour
    return rgb

def compute_macro_f1(all_labels, all_preds):
    return f1_score(all_labels, all_preds, average='macro', zero_division=0)

def compute_mean_iou_np(pred_boxes_np, target_boxes_np):
    pred=torch.tensor(pred_boxes_np, dtype=torch.float32)
    target=torch.tensor(target_boxes_np, dtype=torch.float32)
    loss=IoULoss(reduction='mean')(pred, target)
    return 1 - loss.item()

def compute_dice_score(pred_logits, target_masks):
    eps=1e-6
    preds=pred_logits.argmax(dim=1)
    num_classes=pred_logits.shape[1]
    scores=[]
    for c in range(num_classes):
        pred_c=(preds==c).float()
        target_c=(target_masks==c).float()
        dice_c=(2*(pred_c*target_c).sum())/(pred_c.sum()+target_c.sum()+eps)
        scores.append(dice_c.item())
    return float(np.mean(scores))

def compute_pixel_accuracy(pred_logits, target_masks):
    preds=pred_logits.argmax(dim=1)
    return (preds==target_masks).float().mean().item()

def run_evaluate(args):
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    device=get_device()
    model=load_multitask_model(args, device)

    test_ds = OxfordIIITPetDataset(root_dir=args.data_root, split='test', task='multitask', img_size=args.img_size, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Test set size: {len(test_ds)} samples")

    all_cls_labels=[]
    all_cls_preds=[]
    all_norm_boxes=[]
    all_target_boxes=[]
    all_seg_logits=[]
    all_seg_masks=[]
    table_images=[]
    table_pred_boxes=[]
    table_gt_boxes=[]
    table_ious=[]
    table_breeds=[]
    n_table=0
    seg_sample_imgs=[]
    seg_sample_gt=[]
    seg_sample_pred= []
    n_seg=0

    with torch.no_grad():
        for batch in test_loader:
            images=batch['image'].to(device)
            labels=batch['label'].to(device)
            target_boxes=batch['bbox'].to(device)
            masks=batch['mask'].long().to(device)
            out=model(images)

            bottleneck, _ = model.encoder(images, return_features=True)
            norm_boxes=model.reg_head(bottleneck)

            all_cls_labels.extend(labels.cpu().numpy())
            all_cls_preds.extend(out['classification'].argmax(dim=1).cpu().numpy())
            all_norm_boxes.extend(norm_boxes.cpu().numpy())
            all_target_boxes.extend(target_boxes.cpu().numpy())
            all_seg_logits.append(out['segmentation'].cpu())
            all_seg_masks.append(masks.cpu())

            if n_table<10:
                for i in range(min(images.size(0), 10-n_table)):
                    table_images.append(images[i].cpu())
                    table_pred_boxes.append(norm_boxes[i].cpu().numpy())
                    table_gt_boxes.append(target_boxes[i].cpu().numpy())
                    iou_val = 1-IoULoss(reduction='mean')(norm_boxes[i:i+1].cpu(), target_boxes[i:i+1].cpu()).item()
                    table_ious.append(iou_val)
                    table_breeds.append(out['classification'][i].argmax().item())
                    n_table+=1

            if n_seg<5:
                for i in range(min(images.size(0), 5-n_seg)):
                    seg_sample_imgs.append(images[i].cpu())
                    seg_sample_gt.append(masks[i].cpu().numpy())
                    seg_sample_pred.append(out['segmentation'][i].argmax(dim=0).cpu().numpy())
                    n_seg+=1
            
    all_seg_logits=torch.cat(all_seg_logits, dim=0)
    all_seg_masks=torch.cat(all_seg_masks,  dim=0)

    macro_f1=compute_macro_f1(all_cls_labels, all_cls_preds)
    accuracy=accuracy_score(all_cls_labels, all_cls_preds)
    precision=precision_score(all_cls_labels, all_cls_preds, average='macro', zero_division=0)
    recall=recall_score(all_cls_labels, all_cls_preds, average='macro', zero_division=0)
    mean_iou=compute_mean_iou_np(np.array(all_norm_boxes), np.array(all_target_boxes))
    dice_score=compute_dice_score(all_seg_logits, all_seg_masks)
    pixel_acc=compute_pixel_accuracy(all_seg_logits, all_seg_masks)
 
    print("FINAL TEST METRICS")
    print("-----------------------")
    print(f"Classification Accuracy : {accuracy:.4f}")
    print(f"Classification Precision : {precision:.4f}")
    print(f"Classification Recall : {recall:.4f}")
    print(f"Classification Macro F1 : {macro_f1:.4f}")
    print(f"Detection Mean IoU : {mean_iou:.4f}")
    print(f"Segmentation Dice : {dice_score:.4f}")
    print(f"Pixel Acc : {pixel_acc:.4f}")

    wandb.log({
        'test/accuracy'    : accuracy,
        'test/precision'   : precision,
        'test/recall'      : recall,
        'test/macro_f1'    : macro_f1,
        'test/mean_iou'    : mean_iou,
        'test/dice_score'  : dice_score,
        'test/pixel_acc'   : pixel_acc,
    })
    
    if args.log_mode in ('all', 'detection'):
        log_detection_table(table_images, table_pred_boxes, table_gt_boxes, table_ious, table_breeds, args.img_size)
    
    if args.log_mode in ('all', 'segmentation'):
        log_segmentation_samples(seg_sample_imgs, seg_sample_gt, seg_sample_pred)
 
    wandb.finish()

def log_detection_table(images, pred_boxes, gt_boxes, ious, breeds, img_size):
    columns=['image', 'breed', 'iou', 'confidence', 'result']
    table=wandb.Table(columns=columns)
 
    for i in range(len(images)):
        img_np=denormalise(images[i])               
        pil_img=Image.fromarray(img_np)
        draw_img=pil_img.copy()

        def to_pixel(box, size):
            cx, cy, w, h = box
            x1 = int((cx-w/2)*size)
            y1 = int((cy-h/2)*size)
            x2 = int((cx+w/2)*size)
            y2 = int((cy+h/2)*size)
            return x1, y1, x2, y2
 
        from PIL import ImageDraw
        draw=ImageDraw.Draw(draw_img)
        gt_px=to_pixel(gt_boxes[i],   img_size)
        pred_px=to_pixel(pred_boxes[i], img_size)
 
        draw.rectangle(gt_px, outline='green', width=3)   
        draw.rectangle(pred_px, outline='red', width=3)

        iou_val=ious[i]
        breed_name=BREED_NAMES[breeds[i]] if breeds[i] < len(BREED_NAMES) else str(breeds[i])
        result= 'Good' if iou_val > 0.5 else ('Partial' if iou_val > 0.2 else 'Poor')

        confidence = float(np.mean(pred_boxes[i]))
        table.add_data(
            wandb.Image(draw_img),
            breed_name,
            round(iou_val, 4),
            round(confidence, 4),
            result
        )

    wandb.log({'detection_table': table})
    print(f"Logged detection table with {len(images)} samples")
 
def log_segmentation_samples(images, gt_masks, pred_masks):
    panels = []
    for i in range(len(images)):
        img_np=denormalise(images[i])
        gt_colour=apply_colour_map(gt_masks[i])
        pred_colour=apply_colour_map(pred_masks[i])
 
        fig, axes=plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np); axes[0].set_title('Original'); axes[0].axis('off')
        axes[1].imshow(gt_colour); axes[1].set_title('GT Trimap'); axes[1].axis('off')
        axes[2].imshow(pred_colour); axes[2].set_title('Predicted Mask'); axes[2].axis('off')
        plt.tight_layout()

        panels.append(wandb.Image(fig, caption=f"Sample {i+1}"))
        plt.close(fig)
 
    wandb.log({'segmentation_samples': panels})
    print(f"Logged {len(images)} segmentation sample triplets")


def run_feature_maps(args):

    wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=f"{args.wandb_run_name}_feature_maps", config=vars(args))
    device = get_device()
 
    if not args.image_paths:
        raise ValueError("--image_paths required for feature_maps mode")
 
    model = load_classifier(args.classifier_path, device)
    model.eval()
 
    _, tensor = preprocess_image(args.image_paths[0], args.img_size)
    tensor    = tensor.to(device)
 
    captured = {}
 
    def make_hook(name):
        def hook_fn(module, input, output):
            captured[name] = output.detach().cpu()
        return hook_fn
 
    hook1 = model.encoder.block1[0].register_forward_hook(make_hook('block1'))
    hook5 = model.encoder.block5[-2].register_forward_hook(make_hook('block5'))
 
    with torch.no_grad():
        model(tensor)
 
    hook1.remove()
    hook5.remove()
 
    def make_feature_grid(feat_tensor, n_channels=16):
 
        feat = feat_tensor[0, :n_channels]
        f_min = feat.view(n_channels, -1).min(dim=1)[0].view(n_channels, 1, 1)
        f_max = feat.view(n_channels, -1).max(dim=1)[0].view(n_channels, 1, 1)
        feat = (feat - f_min) / (f_max - f_min + 1e-6)
        feat = feat.unsqueeze(1)
        grid = vutils.make_grid(feat, nrow=4, padding=2, normalize=False)
        return grid.permute(1, 2, 0).numpy()            # (H, W, 3)
 
    grid1 = make_feature_grid(captured['block1'], n_channels=16)
    grid5 = make_feature_grid(captured['block5'], n_channels=16)
 
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(grid1, cmap='viridis')
    axes[0].set_title('Block 1 — First Conv Layer\n(Low-level: edges, gradients)', fontsize=12)
    axes[0].axis('off')
    axes[1].imshow(grid5, cmap='viridis')
    axes[1].set_title('Block 5 — Last Conv Layer\n(High-level: semantic shapes)', fontsize=12)
    axes[1].axis('off')
    plt.suptitle('VGG11 Feature Maps', fontsize=14, fontweight='bold')
    plt.tight_layout()
 
    wandb.log({
        'feature_maps/block1_first_conv' : wandb.Image(fig),
        'feature_maps/block5_last_conv'  : wandb.Image(fig),
    })
    plt.close(fig)
    print("Feature maps logged to W&B")
    wandb.finish()
 
def run_predict(args):
    wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=f"{args.wandb_run_name}_predict", config=vars(args))
 
    device = get_device()
    model  = load_multitask_model(args, device)
    os.makedirs(args.output_dir, exist_ok=True)
 
    wandb_images = []
 
    for img_path in args.image_paths:
        print(f"Processing: {img_path}")
        original_pil, tensor = preprocess_image(img_path, args.img_size)
        tensor = tensor.to(device)
 
        with torch.no_grad():
            out = model(tensor)
 
        save_path = os.path.join(args.output_dir, f"pred_{Path(img_path).stem}.png")
        panel = visualise_prediction(original_pil, out, save_path, args.img_size)
        wandb_images.append(wandb.Image(panel, caption=f"{Path(img_path).stem} — " f"breed: {BREED_NAMES[out['classification'].argmax().item()]}"))
        print(f"Saved: {save_path}")
 
    wandb.log({'wild_image_predictions': wandb_images})
    wandb.finish()
    print(f"All predictions saved to {args.output_dir}/")
 
 
def visualise_prediction(original_pil, out, save_path, img_size):
    from PIL import ImageDraw

    pred_class  = out['classification'].argmax(dim=1).item()
    breed_name  = BREED_NAMES[pred_class] if pred_class < len(BREED_NAMES) else str(pred_class)
 
    cx, cy, w, h = out['localization'][0].tolist()
    x1 = max(0, int(cx - w/2))
    y1 = max(0, int(cy - h/2))
    x2 = min(img_size, int(cx + w/2))
    y2 = min(img_size, int(cy + h/2))

    seg_mask_np = out['segmentation'].argmax(dim=1)[0].cpu().numpy()
    seg_colour  = apply_colour_map(seg_mask_np)

    img_resized = original_pil.resize((img_size, img_size))
    bbox_img = img_resized.copy()
    draw = ImageDraw.Draw(bbox_img)
    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

    mask_pil = Image.fromarray(seg_colour)

    img_np = np.array(img_resized).astype(np.float32)
    mask_np = seg_colour.astype(np.float32)
    overlay_np = (0.6 * img_np + 0.4 * mask_np).clip(0, 255).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay_np)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
 
    axes[0].imshow(bbox_img)
    axes[0].set_title(f'Breed: {breed_name}\nPredicted BBox (red)', fontsize=10)
    axes[0].axis('off')
 
    axes[1].imshow(mask_pil)
    axes[1].set_title('Segmentation Mask\n' 'Green=Foreground Black=Background Yellow=Boundary', fontsize=9)
    axes[1].axis('off')
 
    axes[2].imshow(overlay_pil)
    axes[2].set_title('Overlay\n(image + mask blend)', fontsize=10)
    axes[2].axis('off')
 
    plt.suptitle(f'Pipeline Output — {Path(save_path).stem}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
 
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    panel_pil = Image.fromarray(buf)
    plt.close(fig)
 
    return panel_pil

def main():
    args = get_args()
 
    dispatch = {
        'evaluate'     : run_evaluate,
        'predict'      : run_predict,
        'feature_maps' : run_feature_maps,
    }
    if args.mode not in dispatch:
        raise ValueError(f"Unknown mode '{args.mode}'. " f"Choose from {list(dispatch.keys())}")
    dispatch[args.mode](args)

if __name__ == "__main__":
    main()




