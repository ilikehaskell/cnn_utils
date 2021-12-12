from torchvision import transforms

import pickle
import torch
import numpy as np
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'pixel_to_value.pkl')

with open(file_path, 'rb') as f:
    pixel_to_value = pickle.load(f)

ptv = {a:torch.Tensor([b]) for a,b in pixel_to_value.items()}

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class ToTensorWithoutScaling(object):
    def __init__(self, mode='RGB'):
        self.mode = mode
    """H x W x C -> C x H x W"""
    def __call__(self, picture):
        return torch.ByteTensor(np.array(picture)).permute(2, 0, 1)

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
        return self.__class__.__name__ + f'({self.mode!r})'

class ToValue(object):
    def __init__(self):
        pass
        
    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        original_tensor = torch.Tensor(np.array(pic))
        original_tensor_shape = original_tensor.shape
        maped_flatten = [ptv[tuple(i.tolist())] for i in original_tensor.flatten(0,-2)]
        recreated = torch.cat(maped_flatten).reshape(original_tensor_shape[:2] + (1,))
        return recreated.permute(2,0,1)



    def __repr__(self):
        return self.__class__.__name__ + f'({self.mode!r})'


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