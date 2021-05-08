import os
import random
from collections import defaultdict

import json
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.transforms import *
from albumentations.pytorch import ToTensorV2
import albumentations as A

import cv2
from pycocotools.coco import COCO




class BaseAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            ToTensorV2()
        ])
    def __call__(self, **kwargs):
        return self.transform(**kwargs)

class TestAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            ToTensorV2()
        ])
    def __call__(self, **kwargs):
        return self.transform(**kwargs)

class CustomAugmentation:
    def __init__(self):
        self.transform = A.Compose([
                                A.Rotate(limit=30, 
                                         p=0.5),
                                A.HorizontalFlip(p=0.5),
                                A.RandomResizedCrop(height=512, width=512,
                                                   scale=(0.3, 1.0), ratio=(0.75, 1.33),
                                                   p=0.5),
                                A.ElasticTransform(p=0.3),
                                A.OpticalDistortion(p=0.2),
                                A.MotionBlur(p=0.1),
                                A.RandomBrightness(limit=0.1,
                                                  p=0.2),
                                A.RandomContrast(limit=0.1,
                                                p=0.2),
                                A.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.1,
                                             p=0.2),
                              ToTensorV2()
                            ])
        
    def __call__(self, **kwargs):
        return self.transform(**kwargs)    
    
    


    
class CustomDataset(data.Dataset):
    """COCO format"""
    def __init__(self, 
                 data_dir='/opt/ml/input/data',
                 anotation_file = '/opt/ml/input/data/train.json',
                 mode='train', 
                 transform = None
                ):
        
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.data_dir = data_dir
        self.anotation_file = anotation_file
        self.coco = COCO(anotation_file)
 

        self.num_classes = 12
        
        with open(anotation_file, 'r') as f:
            dataset_info = json.loads(f.read())
        categories = dataset_info['categories']
        
        # Load categories and super categories
        cat_names = []
        for cat_it in categories:
            cat_names.append(cat_it['name'])

        
        cat_names = ['Backgroud'] + cat_names
        self.category_names = cat_names
        
        
        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.data_dir, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D 
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # Unknown = 1, General trash = 2, ... , Cigarette = 11
            for i in range(len(anns)):
                className = self.get_classname(anns[i]['category_id'], cats)
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
            masks = masks.astype(np.float32)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            
            return images, image_infos
    
    def get_classname(self, classID, cats):
        for i in range(len(cats)):
            if cats[i]['id']==classID:
                return cats[i]['name']
        return "None"

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


class TestDataset(data.Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
