import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from torchvision import transforms
from deeplab3.dataloaders import custom_transforms as tr

class SampleLoader():
    def __init__(self, mode, split, base_size, crop_size):
        self.mode = mode
        self.split = split
        self.base_size = base_size
        self.crop_size = crop_size

        self.normalizationFactors()

    #Overload this function for custom normalization factors
    def normalizationFactors(self):
        if self.mode == "RGBD":
            print('Using RGB-D input')
            self.data_mean = (0.485, 0.456, 0.406, 0.213)  # TODO Cityscapes stats are [ 0.29089086,  0.32946742,  0.29078867,  0.29811586]
            self.data_std = (0.229, 0.224, 0.225, 0.111)  # TODO [ 0.19013525,  0.19000581,  0.18482447,  0.29437588]
        elif self.mode == "RGB":
            print('Using RGB input')
            self.data_mean = (0.485, 0.456, 0.406)
            self.data_std = (0.229, 0.224, 0.225)
        elif self.mode == "RGB_HHA":
            print('Using RGB HHA input')
            self.data_mean = (0.294, 0.332, 0.293, 0.425, 0.825, 0.648)
            self.data_std = (0.192, 0.193, 0.188, 0.374, 0.198, 0.187)

    def load_sample(self, img_path, depth_path, lbl_path, no_transforms=False):
        _img = Image.open(img_path).convert('RGB')

        if self.mode == "RGBD":
            _depth = self.loadDepth(depth_path)
            _img.putalpha(_depth)
        elif self.mode == "RGB_HHA":
            _depth = self.loadDepth(depth_path)
            _img = [_img, _depth]

        _target = self.getLabels(lbl_path)

        sample = {'image': _img, 'label': _target}
        if no_transforms:
            return sample

        if self.split in ['train', 'train_extra']:
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def loadDepth(self, depth_path):
        if self.mode == 'RGBD':
            _depth = Image.open(depth_path).convert('L')
        elif self.mode == 'RGB_HHA':
            _depth = Image.open(depth_path).convert('RGB')
        return _depth

    def getLabels(self, lbl_path):
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _target = Image.fromarray(_tmp)
        return _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=self.data_mean, std=self.data_std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.crop_size),
            tr.Normalize(mean=self.data_mean, std=self.data_std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(crop_size=self.crop_size),
            tr.Normalize(mean=self.data_mean, std=self.data_std),
            tr.ToTensor()])

        return composed_transforms(sample)