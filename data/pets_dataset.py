import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OxfordIIITPetDataset(Dataset):
    NUM_CLASSES=37
    IMGNET_MEAN=(0.485, 0.456, 0.406)
    IMGNET_STD=(0.229, 0.224, 0.225)

    def __init__(self, root_dir, split, task, img_size, val_frac, test_frac, seed=42, transform=None):
        super(OxfordIIITPetDataset, self).__init__()

        self.root_dir=root_dir
        self.split=split
        self.task=task
        self.transform=transform
        self.img_size=img_size
        self.seed=seed
        self.test_frac=test_frac
        self.val_frac=val_frac

        self.images_dir=Path(root_dir)/'images'/'images'
        self.annotation_dir=Path(root_dir)/'annotations'/'annotations'
        self.xmls_dir=self.annotation_dir/'xmls'
        self.trimaps_dir=self.annotation_dir/'trimaps'

        self.samples=self._parse_list_txt()
        self.samples=self._filter_samples(self.samples)
        self.samples=self._split_samples(self.samples)

        if self.transform is None:
            self.transform=self._default_transform(split)

    def _parse_list_txt(self):
        final_ls=[]
        list_file_path=self.annotation_dir/'list.txt'
        with open(list_file_path) as file:
            for line in file:
                if not line or line.lstrip().startswith("#"):
                    continue
                img_name, class_id, species_id, breed_id = line.strip().split()
                final_ls.append({
                    "image_name": img_name,
                    "class_id": int(class_id)-1,
                    "species_id": int(species_id),
                    "breed_id": int(breed_id)
                })
        return final_ls
    
    def _filter_samples(self, samples):
        filtered_samples=[]
        if self.task=='classification':
            return samples
        for sample in samples:
            xmls_path=self.xmls_dir/f"{sample['image_name']}.xml"
            trimap_path=self.trimaps_dir/f"{sample['image_name']}.png"
            if self.task=='detection':
                if xmls_path.exists():
                    filtered_samples.append(sample)
            elif self.task=='segmentation':
                if trimap_path.exists():
                    filtered_samples.append(sample)
            elif self.task=='multitask':
                if xmls_path.exists() and trimap_path.exists():
                    filtered_samples.append(sample)
            else:
                raise ValueError(f"Incorrect task : {self.task}. Choose from 'classification', 'detection', 'segmentation', 'multitask'")
        
        return filtered_samples
    
    def _split_samples(self, samples):
        np.random.seed(self.seed)
        num_samples=len(samples)
        idxs=np.random.permutation(num_samples)
        n_test=int(num_samples*self.test_frac)
        n_val=int(num_samples*self.val_frac)
        n_train=num_samples-n_test-n_val
        samples_shuffled=[samples[i] for i in idxs]
        train_samples=samples_shuffled[:n_train]
        val_samples=samples_shuffled[n_train:n_train+n_val]
        test_samples=samples_shuffled[n_train+n_val:]
        if self.split=='train':
            return train_samples
        elif self.split=='test':
            return test_samples
        elif self.split=='val':
            return val_samples
        else:
            raise ValueError(f"Incorrect split : {self.split}. Choose from 'train', 'test' or 'val'")
        
    def _parse_bbox(self, image_name):
        xml_path=self.xmls_dir/f"{image_name}.xml"
        try:
            tree=ET.parse(xml_path)
            root=tree.getroot()
            W=float(root.find('size/width').text)
            H=float(root.find('size/height').text)
            obj=root.find("object")
            if obj is None:
                return None
            bbox=obj.find("bndbox")
            if bbox is None:
                return None
            xmin=float(bbox.find("xmin").text)
            ymin=float(bbox.find("ymin").text)
            xmax=float(bbox.find("xmax").text)
            ymax=float(bbox.find("ymax").text)

            cx=(xmin+xmax)/(2*W)
            cy=(ymin+ymax)/(2*H)
            w=(xmax-xmin)/W
            h=(ymax-ymin)/H
            return torch.tensor([cx, cy, w, h], dtype=torch.float32)
        except FileNotFoundError:
            return None

    def _load_trimap(self, image_name):
        try:
            trimap_path=self.trimaps_dir/f"{image_name}.png"
            img=Image.open(trimap_path).convert('L')
            img_arr=np.array(img).astype(np.int64)-1
            return torch.LongTensor(img_arr)
        except FileNotFoundError:
            return None
    
    def _default_transform(self, split):
        if split=='train':
            return A.Compose([
                A.RandomResizedCrop(size=(self.img_size, self.img_size)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(),
                A.Normalize(mean=self.IMGNET_MEAN, std=self.IMGNET_STD),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        elif split=='val' or split=='test':
            return A.Compose([
                A.Resize(height=self.img_size+32, width=self.img_size+32),
                A.CenterCrop(height=self.img_size, width=self.img_size),
                A.Normalize(mean=self.IMGNET_MEAN, std=self.IMGNET_STD),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        else:
            raise ValueError(f"Incorrect split : {self.split}. Choose from 'train', 'test' or 'val'")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample=self.samples[idx]
        img_name=sample['image_name']
        label=sample['class_id']

        path=self.images_dir/f"{img_name}.jpg"
        image=Image.open(path).convert('RGB')
        np_image=np.array(image).astype(np.uint8)

        bbox=self._parse_bbox(img_name)
        mask=self._load_trimap(img_name)

        kwargs={'image': np_image}
        if mask is not None:
            kwargs['mask']=np.array(mask)
        result=self.transform(**kwargs)
        image_tensor=result['image']
        mask_tensor=result.get('mask')

        out={'image': image_tensor, 'label': label}
        if self.task in ('detection', 'multitask'):
            out['bbox']=bbox
        if self.task in ('segmentation', 'multitask'):
            out['mask']=mask_tensor
        return out