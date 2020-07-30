import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0., 0.), std=(1., 1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        img = sample['image']
        img = np.array(img).astype(np.float32)

        img /= 255.0
        img -= self.mean[:3]
        img /= self.std[:3]

        mask = sample['label']
        mask = np.array(mask).astype(np.float32)

        depth = sample['depth']
        if not isinstance(depth, list):
            depth = np.array(depth).astype(np.float32)
            depth /= 255.0
            depth -= self.mean[3:]
            depth /= self.std[3:]

        return {'image': img,
                'depth': depth,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        depth = sample['depth']
        if not isinstance(depth, list):
            depth = np.array(depth).astype(np.float32)
            if len(depth.shape) == 3:
                depth = depth.transpose((2, 0, 1))
            depth = torch.from_numpy(depth).float()

        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        coin_flip = random.random()
        mask = sample['label']
        depth = sample['depth']
        img = sample['image']

        if coin_flip < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if not isinstance(depth, list):
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        rotate_degree = random.uniform(-1*self.degree, self.degree)

        img = sample['image']
        img = img.rotate(rotate_degree, Image.BILINEAR)

        mask = sample['label']
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        depth = sample['depth']
        if not isinstance(depth, list):
             depth = depth.rotate(rotate_degree, Image.BILINEAR)

        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        coin_flip = random.random()
        rand_radius = random.random()
        mask = sample['label']

        img = sample['image']
        depth = sample['depth']
        if coin_flip < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=rand_radius))
            if not isinstance(depth, list):
                depth = depth.filter(ImageFilter.GaussianBlur(
                radius=rand_radius))

        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        depth = sample['depth']

        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w_resize, h_resize = mask.size
        x1 = random.randint(0, w_resize - self.crop_size)
        y1 = random.randint(0, h_resize - self.crop_size)

        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        img = img.resize((ow, oh), Image.BILINEAR)
        if short_size < self.crop_size:
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        if not isinstance(depth, list):
            depth = depth.resize((ow, oh), Image.BILINEAR)
            if short_size < self.crop_size:
                depth = ImageOps.expand(depth, border=(0, 0, padw, padh), fill=0)
            depth = depth.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'depth': depth,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):

        img = sample['image']
        mask = sample['label']
        depth = sample['depth']

        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)

        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w_resize, h_resize = mask.size
        x1 = int(round((w_resize - self.crop_size) / 2.))
        y1 = int(round((h_resize - self.crop_size) / 2.))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        img = img.resize((ow, oh), Image.BILINEAR)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        if not isinstance(depth, list):
            depth = depth.resize((ow, oh), Image.BILINEAR)
            depth = depth.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'depth': depth,
                'label': mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        mask = sample['label']
        depth = sample['depth']

        img = sample['image']
        assert img.size == mask.size
        img = img.resize(self.size, Image.BILINEAR)

        mask = mask.resize(self.size, Image.NEAREST)

        if not isinstance(depth, list):
            depth = depth.resize(self.size, Image.BILINEAR)

        return {'image': img,
                'depth': depth,
                'label': mask}