
import os

import pandas as pd

from PIL import Image
from torch.torch_version import TorchVersion

from torch.utils.data import Dataset

import random
import torch
import numpy as np
from collections import defaultdict

import torch.nn.functional as F

from torchvision import transforms


class RadarDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, image_type='png'):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_type = image_type
        self.open = {
            'png': Image.open,
            'tensor': torch.load
            }
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = self.open[self.image_type](img_path)
        
        try:
            label = self.img_labels.iloc[idx, 1]
        except:
            label = 1

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label - 1

class SiammeseDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None,length = 100):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = 5
        self.datas = {}
        for i in range(5):
            self.datas[i] = [ img for img in  self.img_labels[self.img_labels.label == 1].id]
        self.length = length
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 0])
        # image = Image.open(img_path)
        
         # image1 = random.choice(self.dataset.imgs)
        label = None
        img1 = None
        img2 = None
        # get image from same class
        if index % 2 == 1:
            label = 1
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])
        # get image from different class
        else:
            label = 0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])
        image1, image2 = Image.open(os.path.join(self.img_dir,image1)), Image.open(os.path.join(self.img_dir,image2))

        gs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),
        ])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # label = torch.from_numpy(np.array([label], dtype=np.int32)).squeeze()
        label =torch.Tensor([label]).squeeze()
        return torch.cat([gs(image1), gs(image2)],dim=0), label
