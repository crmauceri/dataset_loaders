import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from deeplab3.dataloaders import custom_transforms as tr
import scipy.stats

class SampleLoader():
    def __init__(self, cfg, mode, split, base_size, crop_size):
        self.cfg = cfg
        self.mode = mode
        self.split = split
        self.base_size = base_size
        self.crop_size = crop_size

        self.normalizationFactors()

    #Overload this function for custom normalization factors
    def normalizationFactors(self):
        if self.mode == "RGBD":
            print('Using RGB-D input')
            # Data mean and std empirically determined from Cityscapes
            self.data_mean = [0.291,  0.329,  0.291,  0.126] # TODO COCO are (0.485, 0.456, 0.406, 0.008)
            self.data_std = [0.190,  0.190,  0.185,  0.179] # TODO
        elif self.mode == "RGB":
            print('Using RGB input')
            self.data_mean = (0.291,  0.329,  0.291)
            self.data_std = (0.190,  0.190,  0.185)
        elif self.mode == "RGB_HHA":
            print('Using RGB HHA input')
            self.data_mean = (0.291,  0.329,  0.291, 0.080, 0.621, 0.370)
            self.data_std = (0.190,  0.190,  0.185, 0.061, 0.355, 0.196)

    def load_sample(self, img_path, depth_path, lbl_path, no_transforms=False):
        _img = Image.open(img_path).convert('RGB')

        if self.mode in ["RGB_HHA", "RGBD"]:
            _depth = self.loadDepth(depth_path)
        else:
            _depth = []

        _target = self.getLabels(lbl_path)

        sample = {'image': _img, 'label': _target, 'depth': _depth}

        if no_transforms:
            sample = tr.ToTensor()(sample)
        else:
            if self.split in ['train', 'train_extra']:
                sample = self.transform_tr(sample)
            elif self.split == 'val':
                sample = self.transform_val(sample)
            elif self.split == 'test':
                sample = self.transform_ts(sample)

            if self.cfg.DATASET.POWER_TRANSFORM and not isinstance(_depth, list):
                sample['depth'] = scipy.stats.boxcox(sample['depth'], self.cfg.DATASET.PT_LAMBDA)

        #Composite RGBD
        if self.mode == "RGBD":
            sample['image'] = torch.cat((sample['image'], sample['depth'].unsqueeze(0)), 0)
        elif self.mode == "RGB_HHA":
            sample['image'] = torch.cat((sample['image'], sample['depth']), 0)

        return sample

    def loadDepth(self, depth_path):
        if self.mode == 'RGBD':
            if self.cfg.DATASET.SYNTHETIC:
                _depth = self.loadSyntheticDepth(depth_path)
            else:
                _depth = Image.open(depth_path).convert('L')
        elif self.mode == 'RGB_HHA':
            _depth = Image.open(depth_path).convert('RGB')
        return _depth

    def loadSyntheticDepth(self, depth_path):
        _depth_arr = np.array(Image.open(depth_path), dtype=int)
        assert (np.max(_depth_arr) > 255)
        _depth_arr = _depth_arr.astype('float32') / 256.
        _depth = Image.fromarray(_depth_arr)

        return _depth

    def getLabels(self, lbl_path):
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _target = Image.fromarray(_tmp)
        return _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size, fill=255),
            #tr.RandomGaussianBlur(), #TODO Not working for depth channel
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

    def invert_normalization(self, img_tensor):
        img = img_tensor.numpy()
        img_tmp = np.transpose(img, axes=[1, 2, 0])
        img_tmp *= self.data_std
        img_tmp += self.data_mean
        img_tmp *= 255.0
        img_tmp = img_tmp.astype(np.uint8)

        return img_tmp