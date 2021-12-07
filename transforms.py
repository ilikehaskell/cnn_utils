from torchvision import transforms

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
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor()
])

train_transform = transforms.Compose([
    ToFormat('RGB'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
#     transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.ToTensor()])