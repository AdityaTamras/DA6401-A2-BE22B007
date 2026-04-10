import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import time
from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss

def get_args():
    parser=argparse.ArgumentParser(description="DA6401 Assignment 2 Training")

    parser.add_argument('--task', type=str, choices=['classification', 'detection', 'segmentation', 'multitask'], help='Choose Task')
    parser.add_argument('--data_root', type=str, default='data', help='Path to dataset root')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Path to checkpoints')
    parser.add_argument('--epochs', type=int, default=30, help='No. of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--dropout_p', type=float, default=0.5, help='CustomDropout probability')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for resizing')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes')
    parser.add_argument('--freeze_strategy', type=str, choices=['frozen', 'partial', 'full'], default='full', help='Backbone freeze strategy for detection and segmentation')
    parser.add_argument('--encoder_path', type=str, default='checkpoints/encoder_best.pth', help='Path to pretrained encoder weights')
    # parser.add_argument('--wandb_project', type=str, default='da6401_assignment2', help='Wandb Project Name')
    # parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')
    # parser.add_argument('--seed', type=int, default=42, help='Global random seed')
    # parser.add_argument('--lambda_cls', type=float, default=1.0, help='Multitask loss weight for classification')
    # parser.add_argument('--lambda_loc', type=float, default=1.0, help='Multitask loss weight for localization')
    # parser.add_argument('--lambda_seg', type=float, default=1.0, help='Multitask loss weight for segmentation')
    parser.add_argument('--val_frac', type=float, default=0.15)
    parser.add_argument('--test_frac', type=float, default=0.15)
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def get_device():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def save_checkpoint(state_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)
    print(f"Checkpoint saved {path}")

def apply_freeze_strategy(model, strategy):
    if strategy=='frozen':
        for p in model.encoder.parameters():
            p.requires_grad=False
    elif strategy=='partial':
        for name, param in model.encoder.named_parameters():
            if any(f'block{i}' in name for i in [1, 2, 3]):
                param.requires_grad = False
            elif any(f'block{i}' in name for i in [4, 5]):
                param.requires_grad = True
    elif strategy=='full':
        for p in model.encoder.parameters():
            p.requires_grad=True
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Freeze strategy '{strategy}': {trainable:,} trainable, {frozen:,} frozen params")
    

def make_dataloaders(args, task):
    common = dict(
        root_dir=args.data_root,
        task= task,
        img_size=args.img_size,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed
    )
    train_ds=OxfordIIITPetDataset(split='train', **common)
    val_ds=OxfordIIITPetDataset(split='val',   **common)
    test_ds=OxfordIIITPetDataset(split='test',  **common)
    use_cuda=torch.cuda.is_available()
    train_loader=DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=use_cuda, drop_last=True)
    val_loader=DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=use_cuda)
    test_loader=DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=use_cuda)
    return train_loader, val_loader, test_loader

def compute_accuracy(logits, labels):
    preds=logits.argmax(dim=1)
    return (preds==labels).float().mean().item()

def compute_mean_iou(pred_boxes, target_boxes):
    loss_fn=IoULoss(reduction='mean')
    mean_loss=loss_fn(pred_boxes, target_boxes)
    return 1-mean_loss.item()

def compute_dice_score(pred_logits, target_masks):
    eps=1e-6
    preds=pred_logits.argmax(dim=1)          # (B, H, W)
    num_classes=pred_logits.shape[1]
    dice_scores=[]
    for c in range(num_classes):
        pred_c= (preds==c).float()
        target_c=(target_masks==c).float()
        dice_c=(2*(pred_c*target_c).sum()) / (pred_c.sum() + target_c.sum() + eps)
        dice_scores.append(dice_c.item())
    return float(np.mean(dice_scores))

class SegmentationLoss(nn.Module):
    def __init__(self, num_classes=3, dice_weight=1.0):
        super().__init__()
        self.ce=nn.CrossEntropyLoss()
        self.dice_weight=dice_weight
        self.num_classes=num_classes
 
    def forward(self, pred_logits, target_masks):
        ce_loss=self.ce(pred_logits, target_masks)
        eps=1e-6
        preds=torch.softmax(pred_logits, dim=1)
        dice=0.0
        for c in range(self.num_classes):
            pred_c=preds[:, c]
            target_c=(target_masks == c).float()
            dice+=(2*(pred_c*target_c).sum()) / (pred_c.sum() + target_c.sum() + eps)
        dice_loss=1-dice/self.num_classes
        return ce_loss + self.dice_weight*dice_loss

def train_classification(args):
    # wandb.init(project=args.project, name=args.wandb_run_name, config=vars(args))
    set_seed(args.seed)
    device=get_device()
    train_loader, val_loader, _ = make_dataloaders(args, task='classification')
    model=VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device=device)
    criterion=nn.CrossEntropyLoss()
    optim=torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler=torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)
    best_val_acc=-1.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch_cls(model, train_loader, criterion, optim, device)
        val_loss, val_acc = evaluate_cls(model, val_loader, criterion, device)
        scheduler.step()
        # wandb.log({
        #     "train/cls_loss": train_loss,
        #     "train/accuracy": train_acc,
        #     "val/cls_loss": val_loss,
        #     "val/accuracy": val_acc,
        #     "lr": scheduler.get_last_lr()[0],
        #     "epoch": epoch
        #     })
        if val_acc>best_val_acc:
            best_val_acc=val_acc
            save_checkpoint(model.encoder.state_dict(), f'{args.checkpoint_dir}/encoder_best.pth')
            save_checkpoint(model.state_dict(), f'{args.checkpoint_dir}/classifier.pth')
            print(f"[Epoch {epoch+1}/{args.epochs}] " f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% " f"val_acc={val_acc:.2f}%")
    # wandb.finish()

def train_one_epoch_cls(model, loader, criterion, optimizer, device):
    model.train()
    running_loss=0.0
    total=0
    correct=0
    for i, batch in enumerate(loader):
        start=time.time()
        print(f"Batch {i} start")
        images=batch['image'].to(device)
        labels=batch['label'].to(device)
        print("images and labels are loaded")
        optimizer.zero_grad()
        logits=model(images)
        loss=criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f"Batch {i} done in {time.time() - start:.2f} sec")
        running_loss+=loss.item()
        preds=logits.argmax(dim=1)
        total+=labels.size(0)
        correct+=(preds==labels).sum().item()
    avg_loss=running_loss/len(loader)
    avg_accuracy=100*correct/total
    print("Hello world")
    return avg_loss, avg_accuracy

def evaluate_cls(model, loader, criterion, device):
    model.eval()
    running_loss=0.0
    total=0
    correct=0
    with torch.no_grad():
        for batch in loader:
            images=batch['image'].to(device)
            labels=batch['label'].to(device)
            logits=model(images)
            loss=criterion(logits, labels)
            running_loss+=loss.item()
            preds=logits.argmax(dim=1)
            total+=labels.size(0)
            correct+=(preds==labels).sum().item()
        avg_loss=running_loss/len(loader)
    avg_accuracy=100*correct/total
    return avg_accuracy

def train_detection(args):
    # wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    set_seed(args.seed)
    device=get_device()
    train_loader, val_loader, _ = make_dataloaders(args, task='detection')
    model=VGG11Localizer(in_channels=3, dropout_p=args.dropout_p).to(device)
    if os.path.exists(args.encoder_path):
        model.encoder.load_state_dict(torch.load(args.encoder_path, map_location='cpu'))
        print(f"Loaded encoder from {args.encoder_path}")
    apply_freeze_strategy(model, args.freeze_strategy)

    iou_loss=IoULoss(reduction='mean')
    optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    best_val_iou = -1.0
    for epoch in range(args.epochs):
        train_loss, train_iou=train_one_epoch_det(model, train_loader, iou_loss, optimizer, device)
        val_loss, val_iou= evaluate_det(model, val_loader, iou_loss, device)
        scheduler.step()
        # wandb.log({
        #     'train/iou_loss': train_loss,
        #     'train/mean_iou': train_iou,
        #     'val/iou_loss': val_loss,
        #     'val/mean_iou': val_iou,
        #     'lr': scheduler.get_last_lr()[0],
        #     'epoch': epoch
        # })
        if val_iou>best_val_iou:
            best_val_iou=val_iou
            save_checkpoint(model.state_dict(), f'{args.checkpoint_dir}/localizer.pth')
        print(f"[Epoch {epoch+1}/{args.epochs}]" f"train_loss={train_loss:.4f} val_iou={val_iou:.4f}")
    # wandb.finish()
 
 
def train_one_epoch_det(model, loader, iou_loss, optimizer, device):
    model.train()
    running_loss=0.0
    running_iou=0.0
    for batch in loader:
        images=batch['image'].to(device)
        target_boxes=batch['bbox'].to(device)
        optimizer.zero_grad()
        encoded=model.encoder(images, return_features=False)
        norm_boxes=model.head(encoded)
        loss=iou_loss(norm_boxes, target_boxes)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        with torch.no_grad():
            running_iou+=compute_mean_iou(norm_boxes.detach(), target_boxes)
    return running_loss/len(loader), running_iou/len(loader)
 
def evaluate_det(model, loader, iou_loss, device):
    model.eval()
    running_loss=0.0
    running_iou=0.0
    with torch.no_grad():
        for batch in loader:
            images=batch['image'].to(device)
            target_boxes=batch['bbox'].to(device)
            encoded=model.encoder(images, return_features=False)
            norm_boxes=model.head(encoded)
            loss=iou_loss(norm_boxes, target_boxes)
            running_loss+=loss.item()
            running_iou+=compute_mean_iou(norm_boxes, target_boxes)
    return running_loss/len(loader), running_iou/len(loader)


def train_segmentation(args):
    # wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    set_seed(args.seed)
    device = get_device()
    train_loader, val_loader, _ = make_dataloaders(args, task='segmentation')
    model=VGG11UNet(num_classes=3, in_channels=3).to(device)
    if os.path.exists(args.encoder_path):
        model.encoder.load_state_dict(torch.load(args.encoder_path, map_location='cpu'))
        print(f"Loaded encoder from {args.encoder_path}")
    apply_freeze_strategy(model, args.freeze_strategy)
 
    seg_loss_fn=SegmentationLoss(num_classes=3).to(device)
    optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_dice = -1.0

    for epoch in range(args.epochs):
        train_loss, train_dice=train_one_epoch_seg(model, train_loader, seg_loss_fn, optimizer, device)
        val_loss, val_dice=evaluate_seg(model, val_loader, seg_loss_fn, device)
        scheduler.step()

        # wandb.log({
        #     'train/seg_loss':   train_loss,
        #     'train/dice_score': train_dice,
        #     'val/seg_loss': val_loss,
        #     'val/dice_score': val_dice,
        #     'lr': scheduler.get_last_lr()[0],
        #     'epoch': epoch
        # })

        if val_dice>best_val_dice:
            best_val_dice=val_dice
            save_checkpoint(model.state_dict(), f'{args.checkpoint_dir}/unet.pth')
 
        print(f"[Epoch {epoch+1}/{args.epochs}] " f"train_loss={train_loss:.4f} val_dice={val_dice:.4f}")
    # wandb.finish()
 
def train_one_epoch_seg(model, loader, seg_loss_fn, optimizer, device):
    model.train()
    running_loss=0.0
    running_dice=0.0

    for batch in loader:
        images=batch['image'].to(device)
        masks=batch['mask'].to(device)
        optimizer.zero_grad()
        pred=model(images)     
        loss=seg_loss_fn(pred, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        with torch.no_grad():
            running_dice+=compute_dice_score(pred.detach(), masks)
    return running_loss/len(loader), running_dice/len(loader)
 
def evaluate_seg(model, loader, seg_loss_fn, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    with torch.no_grad():
        for batch in loader:
            images=batch['image'].to(device)
            masks=batch['mask'].to(device)
            pred=model(images)
            loss=seg_loss_fn(pred, masks)
            running_loss+=loss.item()
            running_dice+=compute_dice_score(pred, masks)
    return running_loss / len(loader), running_dice / len(loader)

def train_multitask(args):
    # wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    set_seed(args.seed)
    device = get_device()
 
    train_loader, val_loader, _ = make_dataloaders(args, task='multitask')
 
    model = MultiTaskPerceptionModel(
        num_breeds= 37,
        seg_classes=3,
        in_channels=3,
        classifier_path=f'{args.checkpoint_dir}/classifier.pth',
        localizer_path=f'{args.checkpoint_dir}/localizer.pth',
        unet_path=f'{args.checkpoint_dir}/unet.pth'
    ).to(device)
 
    cls_loss=nn.CrossEntropyLoss()
    iou_loss=IoULoss(reduction='mean')
    seg_loss_fn=SegmentationLoss(num_classes=3).to(device)
    lambdas=(args.lambda_cls, args.lambda_loc, args.lambda_seg)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_total = float('inf')
 
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch_multitask(model, train_loader, cls_loss, iou_loss, seg_loss_fn, lambdas, optimizer, device)
        val_metrics   = evaluate_multitask(model, val_loader, cls_loss, iou_loss, seg_loss_fn, lambdas, device)
        scheduler.step()
 
        # wandb.log({**{f'train/{k}': v for k, v in train_metrics.items()},
        #            **{f'val/{k}':   v for k, v in val_metrics.items()},
        #            'lr': scheduler.get_last_lr()[0], 'epoch': epoch})

        if val_metrics['total_loss'] < best_val_total:
            best_val_total=val_metrics['total_loss']
            save_checkpoint(model.state_dict(), f'{args.checkpoint_dir}/multitask.pth')
 
        print(f"[Epoch {epoch+1}/{args.epochs}] " f"val_total={val_metrics['total_loss']:.4f} " f"val_acc={val_metrics['accuracy']:.2f}% " f"val_iou={val_metrics['mean_iou']:.4f} " f"val_dice={val_metrics['dice_score']:.4f}")
 
    # wandb.finish()
 
 
def train_one_epoch_multitask(model, loader, cls_loss, iou_loss, seg_loss_fn, lambdas, optimizer, device):
    model.train()
    lc, ll, ls=lambdas
    totals=dict(total_loss=0, cls_loss=0, loc_loss=0, seg_loss=0, accuracy=0, mean_iou=0, dice_score=0)
 
    for batch in loader:
        images=batch['image'].to(device)
        labels=batch['label'].to(device)
        target_boxes=batch['bbox'].to(device)
        masks=batch['mask'].to(device)
        optimizer.zero_grad()
        out = model(images)
        encoded, _ = model.encoder(images, return_features=True)
        norm_boxes    = model.reg_head(encoded)
        l_cls=cls_loss(out['classification'], labels)
        l_loc=iou_loss(norm_boxes, target_boxes)
        l_seg=seg_loss_fn(out['segmentation'], masks)
        l_total=lc*l_cls + ll*l_loc + ls*l_seg
        l_total.backward()
        optimizer.step()
 
        totals['total_loss']+=l_total.item()
        totals['cls_loss']+=l_cls.item()
        totals['loc_loss']+=l_loc.item()
        totals['seg_loss']+=l_seg.item()
        with torch.no_grad():
            totals['accuracy']   += compute_accuracy(out['classification'], labels)
            totals['mean_iou']   += compute_mean_iou(norm_boxes.detach(), target_boxes)
            totals['dice_score'] += compute_dice_score(out['segmentation'].detach(), masks)
    n = len(loader)
    return {k: v / n for k, v in totals.items()}
 
 
def evaluate_multitask(model, loader, cls_loss, iou_loss, seg_loss_fn, lambdas, device):
    model.eval()
    lc, ll, ls=lambdas
    totals=dict(total_loss=0, cls_loss=0, loc_loss=0, seg_loss=0, accuracy=0, mean_iou=0, dice_score=0)
 
    with torch.no_grad():
        for batch in loader:
            images=batch['image'].to(device)
            labels=batch['label'].to(device)
            target_boxes=batch['bbox'].to(device)
            masks=batch['mask'].to(device)
            out = model(images)
            encoded, _ = model.encoder(images, return_features=True)
            norm_boxes=model.reg_head(encoded)
            l_cls=cls_loss(out['classification'], labels)
            l_loc=iou_loss(norm_boxes, target_boxes)
            l_seg=seg_loss_fn(out['segmentation'], masks)
            l_total = lc*l_cls + ll*l_loc + ls*l_seg
            totals['total_loss'] += l_total.item()
            totals['cls_loss']   += l_cls.item()
            totals['loc_loss']   += l_loc.item()
            totals['seg_loss']   += l_seg.item()
            totals['accuracy']   += compute_accuracy(out['classification'], labels)
            totals['mean_iou']   += compute_mean_iou(norm_boxes, target_boxes)
            totals['dice_score'] += compute_dice_score(out['segmentation'], masks)
    n = len(loader)
    return {k: v / n for k, v in totals.items()}
 
def main():
    args = get_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(args)
 
    dispatch = {
        'classification': train_classification,
        'detection':      train_detection,
        'segmentation':   train_segmentation,
        'multitask':      train_multitask,
    }
 
    if args.task not in dispatch:
        raise ValueError(f"Unknown task '{args.task}'. " f"Choose from {list(dispatch.keys())}")
    dispatch[args.task](args)
 
if __name__ == "__main__":
    main()



    

    





    




    
