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
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        if isinstance(sample['image'], list):
           img = [np.array(img).astype(np.float32) for img in sample['image']]
           img = np.concatenate(img, axis=2)
        else:
           img = sample['image']
           img = np.array(img).astype(np.float32)

        mask = sample['label']
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
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

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        coin_flip = random.random()
        mask = sample['label']

        if isinstance(sample['image'], list):
            img = sample['image']
            if coin_flip < 0.5:
                img = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in sample['image']]
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = sample['image']
            if coin_flip < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        rotate_degree = random.uniform(-1*self.degree, self.degree)

        if isinstance(sample['image'], list):
            img = [img.rotate(rotate_degree, Image.BILINEAR) for img in sample['image']]
        else:
            img = sample['image']
            img = img.rotate(rotate_degree, Image.BILINEAR)

        mask = sample['label']
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        coin_flip = random.random()
        rand_radius = random.random()
        mask = sample['label']

        if isinstance(sample['image'], list):
            img = sample['image']
            if coin_flip < 0.5:
                img = [img.filter(ImageFilter.GaussianBlur(
                    radius=rand_radius)) for img in sample['image']]
        else:
            img = sample['image']
            if coin_flip < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(
                    radius=rand_radius))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):

        if not isinstance(sample['image'], list):
            sample['image'] = [sample['image']]

        img = sample['image'][0]
        mask = sample['label']
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

        img_out = []
        for img in sample['image']:
            img = img.resize((ow, oh), Image.BILINEAR)
            if short_size < self.crop_size:
                img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            img_out.append(img)

        if len(img_out) == 1:
            return {'image': img_out[0],
                    'label': mask}
        else:
            return {'image': img_out,
                    'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):

        if not isinstance(sample['image'], list):
            sample['image'] = [sample['image']]

        img = sample['image'][0]
        mask = sample['label']
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

        img_out = []
        for img in sample['image']:
            img = img.resize((ow, oh), Image.BILINEAR)
            img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            img_out.append(img)

        if len(img_out)==1:
            return {'image': img_out[0],
                    'label': mask}
        else:
            return {'image': img_out,
                    'label': mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        mask = sample['label']

        if isinstance(sample['image'], list):
            img_out = []
            for img in sample['image']:
                assert img.size == mask.size
                img_out.append(img.resize(self.size, Image.BILINEAR))

        else:
            img = sample['image']
            assert img.size == mask.size
            img_out = img.resize(self.size, Image.BILINEAR)

        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img_out,
                'label': mask}