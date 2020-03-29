import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange
import os
from pycocotools.coco import COCO
from pycocotools import mask
from torchvision import transforms
from deeplab3.dataloaders import custom_transforms as tr
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from deeplab3.dataloaders.datasets.coco import COCOSegmentationSampleLoader


class RGBDSegmentation(Dataset):
    NUM_CLASSES = 38
    CAT_LIST = list(range(38))

    def __init__(self,
                 cfg,
                 split='train'):
        super().__init__()
        base_dir = cfg.DATASET.ROOT
        ann_file = os.path.join(base_dir, 'annotations/instances_{}.json'.format(split))
        ids_file = os.path.join(base_dir, 'annotations/{}_ids.pth'.format(split))
        self.img_dir = os.path.join(base_dir, 'images')
        self.depth_dir = self.img_dir
        self.split = split
        self.coco = COCO(ann_file)
        self.mode = cfg.DATASET.MODE

        self.class_names = [self.coco.cats[i]['name'] for i in self.CAT_LIST]

        self.loader = RGBDSegmentationSampleLoader(cfg, self.coco, split, self.CAT_LIST)

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

        if self.mode == "RGBD":
            depth_path = os.path.join(self.depth_dir, img_metadata['depth_file_name'])

        elif self.mode == 'RGB_HHA':
            depth_path = os.path.join(self.depth_dir, path)

        return self.loader.load_sample(img_path, depth_path, img_id, no_transforms)

    def __len__(self):
        return len(self.ids)

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids


class RGBDSegmentationSampleLoader(COCOSegmentationSampleLoader):

    def gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = instance['segmentation']
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

    parser = argparse.ArgumentParser(description="Test SUNRGBD Loader")
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

    coco_val = RGBDSegmentation(cfg, split='val')

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
