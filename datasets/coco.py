import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange
import os
from pycocotools.coco import COCO
from pycocotools import mask as Mask
from torchvision import transforms
from deeplab3.dataloaders import custom_transforms as tr
from deeplab3.dataloaders.SampleLoader import SampleLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class COCOSegmentation(Dataset):
    def __init__(self,
                 cfg,
                 split='train',
                 year='2017',
                 #use_depth=True,
                 #categories = 'coco', #['pascal', 'sunrgbd']
                 ):
        super().__init__()
        base_dir = cfg.DATASET.ROOT
        ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format(split, year))
        ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}.pth'.format(split, year))
        self.img_dir = os.path.join(base_dir, 'images/{}{}'.format(split, year))
        self.depth_dir = os.path.join(base_dir, 'VNL_Monocular/') #{}{}'.format(split, year))
        self.split = split
        self.coco = COCO(ann_file)

        if cfg.DATASET.COCO.CATEGORIES == 'coco':
            self.CAT_LIST = [0]
            self.CAT_LIST.extend(list(self.coco.cats.keys()))
        elif cfg.DATASET.COCO.CATEGORIES == 'pascal':
            self.CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
        elif cfg.DATASET.COCO.CATEGORIES == 'sunrgbd:':
            # There is only partial overlap between these two category lists. This map indexes sunrgbd:coco
            self.CAT_MAP = {0:0, 4:65, 5:62, 6:63, 7:67, 23:84, 24:82, 25:72, 31:1, 33:70, 34:81, 37:31}
            self.CAT_LIST = list(self.CAT_MAP.values())
        elif cfg.DATASET.COCO.CATEGORIES == 'cityscapes:':
            # There is only partial overlap between these two category lists. This map indexes cityscapes:coco
            self.CAT_MAP = {19:10, 20:13, 24:0, 26:3, 27:8, 28:6, 31:7, 32:4, 33:2}
            self.CAT_LIST = list(self.CAT_MAP.values())
        else:
            raise ValueError('Category mapping to {} not implemented for COCOSegmentation'.format(cfg.DATASET.COCO.CATEGORIES))
        self.NUM_CLASSES = len(self.CAT_LIST)
        self.class_names = ['unknown'] + [self.coco.cats[i]['name'] for i in self.CAT_LIST[1:]]

        self.loader = COCOSegmentationSampleLoader(cfg, self.coco, split, self.CAT_LIST)

        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.cfg = cfg

    def __getitem__(self, index, no_transforms=False):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        img_path = os.path.join(self.img_dir, path)
        depth_path = os.path.join(self.depth_dir, path.replace('.jpg', '.png'))
        sample = self.loader.load_sample(img_path, depth_path, img_id, no_transforms)
        sample['id'] = index
        return sample

    def __len__(self):
        return len(self.ids)

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            try:
                img_metadata = self.coco.loadImgs(img_id)[0]
                path = img_metadata['file_name']
                img_path = os.path.join(self.img_dir, path)
                depth_path = os.path.join(self.depth_dir, path.replace('.jpg', '.png'))
                if os.path.exists(img_path) and os.path.exists(depth_path):
                    cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
                    img_metadata = self.coco.loadImgs(img_id)[0]
                    mask = self.loader.gen_seg_mask(cocotarget, img_metadata['height'],
                                                    img_metadata['width'])
                    # more than 1k pixels
                    if (mask > 0).sum() > 1000:
                        new_ids.append(img_id)

            except FileNotFoundError as e:
                print("{}:{}".format(e, img_id))

            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

class COCOSegmentationSampleLoader(SampleLoader):
    def __init__(self, cfg, coco, split, cat_list):
        super().__init__(cfg, cfg.DATASET.MODE, split,
                         cfg.DATASET.BASE_SIZE, cfg.DATASET.CROP_SIZE)

        self.coco = coco
        self.coco_mask = Mask
        self.CAT_LIST = cat_list

    def normalizationFactors(self):
        if self.mode == "RGBD":
            print('Using RGB-D input')
            # RGB mean and std are the standards used with ImageNet data
            # VNL Depth mean and std empirically determined from COCO
            self.data_mean = [0.485, 0.456, 0.406, .248]
            self.data_std = [0.229, 0.224, 0.225, .122]
        elif self.mode == "RGB":
            print('Using RGB input')
            self.data_mean = [0.485, 0.456, 0.406]
            self.data_std = [0.229, 0.224, 0.225]
        elif self.mode == "RGB_HHA":
            raise NotImplementedError("HHA normalization factors not implemented for COCO")

    def getLabels(self, img_id):
        img_metadata = self.coco.loadImgs(img_id)[0]
        cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        _target = Image.fromarray(self.gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width']))
        return _target

    def loadSyntheticDepth(self, depth_path):
        _depth_arr = np.array(Image.open(depth_path), dtype=int)
        # if np.max(_depth_arr) > 2000:
        #     print("Large max depth: {} {}".format(np.max(_depth_arr), depth_path))
        _depth_arr = (_depth_arr.astype('float32') / 2000.) * 256
        np.clip(_depth_arr, 0, 255, out=_depth_arr)
        _depth_arr = _depth_arr.astype(np.uint8)
        _depth = Image.fromarray(_depth_arr).convert('L')

        return _depth

    def gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

if __name__ == "__main__":
    from deeplab3.config.defaults import get_cfg_defaults
    from deeplab3.dataloaders import custom_transforms as tr
    from deeplab3.dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description="Test COCO Loader")
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

    coco_val = COCOSegmentation(cfg, split='val', year='2017')

    dataloader = DataLoader(coco_val, batch_size=4, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='coco')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= coco_val.loader.data_std
            img_tmp += coco_val.loader.data_mean
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(311)
            plt.imshow(img_tmp[:,:,:3])
            plt.subplot(312)
            plt.imshow(segmap)
            plt.subplot(313)
            plt.imshow(img_tmp[:,:,3])

        if ii == 1:
            break

    plt.show(block=True)
