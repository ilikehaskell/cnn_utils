
import os

import pandas as pd

from PIL import Image

from torch.utils.data import Dataset

from torchvision import transforms


class RadarDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        
        try:
            label = self.img_labels.iloc[idx, 1]
        except:
            label = 1

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label - 1


class ToFormat(object):
    def __init__(self, mode='RGB'):
        self.mode = mode
        
    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return pic.convert(self.mode)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    

val_transfrom = transforms.Compose([
    ToFormat('RGB'),
    transforms.ToTensor()
])

train_transform = transforms.Compose([
    ToFormat('RGB'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
#     transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.ToTensor()])