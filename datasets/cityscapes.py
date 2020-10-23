import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from torchvision import transforms
from deeplab3.dataloaders import custom_transforms as tr
from deeplab3.dataloaders.SampleLoader import SampleLoader


class CityscapesSegmentation(data.Dataset):
    def __init__(self, cfg, split="train"):

        self.root = cfg.DATASET.ROOT
        self.split = split
        self.cfg = cfg

        self.mode = cfg.DATASET.MODE

        self.loader = CityscapesSampleLoader(cfg, split)

        self.files = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        if split == "val" or split == "test":
            self.annotations_base = os.path.join(self.root, 'gtFine', self.split)
        else:
            self.annotations_base = os.path.join(self.root, cfg.DATASET.CITYSCAPES.GT_MODE, self.split)

        self.depth_base = os.path.join(self.root, cfg.DATASET.CITYSCAPES.DEPTH_DIR, self.split)  # {}{}'.format(split, year))

        # 'troisdorf_000000_000073' is corrupted
        self.files[split] = [x for x in self.recursive_glob(rootdir=self.images_base, suffix='.png') if 'troisdorf_000000_000073' not in x]

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index, no_transforms=False):
        img_path, depth_path, lbl_path = self.get_path(index)
        sample = self.loader.load_sample(img_path, depth_path, lbl_path, no_transforms)
        sample['id'] = img_path
        return sample

    def get_path(self, index):
        img_path = self.files[self.split][index].rstrip()
        gt_mode = 'gtFine' if self.split == 'val' else self.cfg.DATASET.CITYSCAPES.GT_MODE
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + '{}_labelIds.png'.format(gt_mode))

        depth_path = os.path.join(self.depth_base,
                                  img_path.split(os.sep)[-2],
                                  os.path.basename(img_path)[:-15] + '{}.png'.format(
                                      self.cfg.DATASET.CITYSCAPES.DEPTH_DIR))
        return img_path, depth_path, lbl_path

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]


class CityscapesSampleLoader(SampleLoader):
    def __init__(self, cfg, split="train"):
        super().__init__(cfg, cfg.DATASET.MODE, split,
                        cfg.DATASET.BASE_SIZE, cfg.DATASET.CROP_SIZE)

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, \
                              17, 19, 20, 21, 22, \
                              23, 24, 25, 26, 27, 28, 31, \
                              32, 33]
        self.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']
        self.NUM_CLASSES = len(self.valid_classes)

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

    def normalizationFactors(self):
        if self.mode == "RGBD":
            print('Using RGB-D input')
            # Data mean and std empirically determined from 1000 Cityscapes samples
            self.data_mean = [0.291,  0.329,  0.291,  0.126]
            self.data_std = [0.190,  0.190,  0.185,  0.179]
        elif self.mode == "RGB":
            print('Using RGB input')
            self.data_mean = [0.291,  0.329,  0.291]
            self.data_std = [0.190,  0.190,  0.185]
        elif self.mode == "RGB_HHA":
            print('Using RGB HHA input')
            self.data_mean =  [0.291,  0.329,  0.291, 0.080, 0.621, 0.370]
            self.data_std =  [0.190,  0.190,  0.185, 0.061, 0.355, 0.196]

    def getLabels(self, lbl_path):
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)
        return _target

    def loadDepth(self, depth_path):
        if self.cfg.DATASET.SYNTHETIC:
            _depth_arr = np.array(Image.open(depth_path), dtype=int)
            # if np.max(_depth_arr) > 10000:
            #     print("Large max depth: {} {}".format(np.max(_depth_arr), depth_path))
            _depth_arr = (_depth_arr.astype('float32') / 10000.) * 256
            np.clip(_depth_arr, 0, 255, out=_depth_arr)
            _depth_arr = _depth_arr.astype(np.uint8)
            _depth = Image.fromarray(_depth_arr).convert('L')
        elif self.mode == 'RGBD':
            _disparity_arr = np.array(Image.open(depth_path)).astype(np.float32)
            # Conversion from https://github.com/mcordts/cityscapesScripts see `disparity`
            # See https://github.com/mcordts/cityscapesScripts/issues/55#issuecomment-411486510
            _disparity_arr[_disparity_arr > 0] = (_disparity_arr[_disparity_arr > 0] - 1.0) / 256.
            _depth_arr = np.zeros(_disparity_arr.shape)
            _depth_arr[_disparity_arr > 0] = 0.2 * 2262 / _disparity_arr[_disparity_arr > 0]
            # _depth_arr[_depth_arr > 60] = 60
            # _depth_arr = _depth_arr / 60. * 255.
            # _depth_arr = _depth_arr.astype(np.uint8)
            # _depth = Image.fromarray(_depth_arr).convert('L')
            _depth = Image.fromarray(_depth_arr)
        elif self.mode == 'RGB_HHA':
            # Depth channel is inverse with 25600 / depth
            # Height channel is inverse with 2560 / height
            _depth = Image.open(depth_path).convert('RGB')
        return _depth

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

if __name__ == '__main__':
    from deeplab3.config.defaults import get_cfg_defaults
    from deeplab3.dataloaders.utils import decode_segmap, sample_distribution
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

    cityscapes_train = CityscapesSegmentation(cfg, split='val')

    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            try:
                img = sample['image']
                gt = sample['label'].numpy()
                tmp = np.array(gt[jj]).astype(np.uint8)
                segmap = decode_segmap(tmp, dataset='cityscapes')
                img_tmp = cityscapes_train.loader.invert_normalization(img[jj])
                plt.figure()
                plt.title('display')
                plt.subplot(131)
                plt.imshow(img_tmp[:, :, :3])
                plt.subplot(132)
                plt.imshow(img_tmp[:, :, 3:].squeeze())
                plt.subplot(133)
                plt.imshow(segmap)
            except SystemError as e:
                print(e)

        if ii == 1:
            break

    plt.show(block=True)

    # print(sample_distribution(cityscapes_train))

