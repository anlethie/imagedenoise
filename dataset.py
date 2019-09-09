from PIL import Image

from torch.utils.data import  Dataset
import glob
import numpy as np
import math
import torchvision
import random

def load_images_folder(path, extension='JPG'):
    images = []
    for filepath in sorted(glob.glob(path + '/*.' + extension)):
        im = Image.open(filepath)
        images.append(im)
    return images


ToTensor = torchvision.transforms.ToTensor()
ToImage = torchvision.transforms.ToPILImage()

class ImageDenoisingDataset(Dataset):
    def __init__(self, noisy_images, truth_images, random_crop=False, color_jitter=False, cropped_resolution = (64,64)):
        super(ImageDenoisingDataset, self).__init__()
        self.noisy_images = noisy_images
        self.truth_images = truth_images
        self.cropped_resolution = cropped_resolution
        self.toTensor = torchvision.transforms.ToTensor()
        self.CenterCrop = torchvision.transforms.CenterCrop(self.cropped_resolution)
        self.random_crop = random_crop
        self.color_jitter= color_jitter
        
        assert(len(self.noisy_images) == len(self.truth_images))
        
    def __len__(self):
        return len(self.noisy_images)
    
    def __getitem__(self, index):
        if self.color_jitter:
                brightness_max_prop = random.uniform(0, 1)
                if brightness_max_prop > 0.6:
                    brightness_range = (6.0, 10.0)
                else:
                    brightness_range = (0.5, 1.5)
                    
        if self.random_crop:
            i, j, h, w = torchvision.transforms.RandomCrop.get_params(self.noisy_images[index], self.cropped_resolution)            
            
            if self.color_jitter:
                colorjitter = torchvision.transforms.ColorJitter.get_params((0.5, 10.0),(0.5, 1.5),(0.5, 1.5),(0, 0))
                noisy_tensor = self.toTensor(colorjitter(
                    torchvision.transforms.functional.crop(self.noisy_images[index], i, j, h, w)))
                truth_tensor = self.toTensor(colorjitter(
                    torchvision.transforms.functional.crop(self.truth_images[index], i, j, h, w)))
            else:
                noisy_tensor = self.toTensor(
                    torchvision.transforms.functional.crop(self.noisy_images[index], i, j, h, w))
                truth_tensor = self.toTensor(
                torchvision.transforms.functional.crop(self.truth_images[index], i, j, h, w))
            
        elif self.color_jitter:
            colorjitter = torchvision.transforms.ColorJitter.get_params((0.5, 1.5),(0.5, 1.5),(0.5, 1.5),(0, 0))
            noisy_tensor = self.toTensor(colorjitter(self.CenterCrop(self.noisy_images[index])))
            truth_tensor = self.toTensor(colorjitter(self.CenterCrop(self.truth_images[index])))
        else:
            noisy_tensor = self.toTensor(self.CenterCrop(self.noisy_images[index]))
            truth_tensor = self.toTensor(self.CenterCrop(self.truth_images[index]))
            
        return noisy_tensor, truth_tensor