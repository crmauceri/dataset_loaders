import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from torchvision import transforms
from deeplab3.dataloaders import custom_transforms as tr

class CityscapesSegmentation(data.Dataset):
    NUM_CLASSES = 19

    def __init__(self, cfg, split="train"):

        self.root = cfg.DATASET.ROOT
        self.split = split
        self.cfg = cfg
        self.use_depth = cfg.DATASET.USE_DEPTH
        if self.use_depth:
            print('Using RGB-D input')
            self.data_mean = (0.485, 0.456, 0.406, 0.300)
            self.data_std = (0.229, 0.224, 0.225, 0.295)
        else:
            print('Using RGB input')
            self.data_mean = (0.485, 0.456, 0.406)
            self.data_std = (0.229, 0.224, 0.225)


        self.files = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtCoarse', self.split)
        self.depth_base = os.path.join(self.root, 'disparity', self.split)  # {}{}'.format(split, year))

        # 'troisdorf_000000_000073' is corrupted
        self.files[split] = [x for x in self.recursive_glob(rootdir=self.images_base, suffix='.png') if 'troisdorf_000000_000073' not in x]

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index, no_transforms=False):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtCoarse_labelIds.png')
        depth_path = os.path.join(self.depth_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'disparity.png')

        _img = Image.open(img_path).convert('RGB')
        if self.use_depth:
            _depth_arr = np.asarray(Image.open(depth_path), dtype='float')
            # _depth_arr /= 25000 * 256 # Empirically determined normalization value (2.5 std)
            _depth = Image.fromarray(_depth_arr / 25000 * 256).convert('L')
            _img.putalpha(_depth)

        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}
        if no_transforms:
            return sample

        if self.split in ['train', 'train_extra']:
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.cfg.DATASET.BASE_SIZE, crop_size=self.cfg.DATASET.CROP_SIZE, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=self.data_mean, std=self.data_std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.cfg.DATASET.CROP_SIZE),
            tr.Normalize(mean=self.data_mean, std=self.data_std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(crop_size=self.cfg.DATASET.CROP_SIZE),
            tr.Normalize(mean=self.data_mean, std=self.data_std),
            tr.ToTensor()])

        return composed_transforms(sample)

if __name__ == '__main__':
    from deeplab3.config.defaults import get_cfg_defaults
    from deeplab3.dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description="Test cityscapes Loader")
    parser.add_argument('config_file', help='config file path')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)

    cityscapes_train = CityscapesSegmentation(cfg, split='train_extra')

    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='cityscapes')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= cityscapes_train.data_std
            img_tmp += cityscapes_train.data_mean
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)

