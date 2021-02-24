from deeplab3.dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd, sunrgbd, scenenet
from deeplab3.dataloaders.utils import get_nyu13_labels, get_cityscapes_labels, get_pascal_labels, get_sunrgbd_labels
from torch.utils.data import DataLoader

def make_dataset(cfg, split):
    if cfg.DATASET.NAME == 'pascal':
        if cfg.DATASET.MODE != "RGB":
            raise ValueError('RGBD DataLoader not implemented')
        if cfg.DATASET.USE_SBD:
            sbd_train = sbd.SBDSegmentation(cfg, split=['train', 'val'])
            return combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        return pascal.VOCSegmentation(cfg, split=split)

    elif cfg.DATASET.NAME == 'cityscapes':
        if split == "train":
            return cityscapes.CityscapesSegmentation(cfg, split=cfg.DATASET.CITYSCAPES.TRAIN_SET)
        else:
            return cityscapes.CityscapesSegmentation(cfg, split=split)

    elif cfg.DATASET.NAME == 'scenenet':
        return scenenet.SceneNetSegmentation(cfg, split=split)

    elif cfg.DATASET.NAME == 'coco':
        if split == "test":
            return None
        else:
            return coco.COCOSegmentation(cfg, split=split)

    elif cfg.DATASET.NAME in ['sunrgbd', 'matterport3d']:
        if split == "test":
            return None
        else:
            return sunrgbd.RGBDSegmentation(cfg, split=split)

    else:
        raise NotImplementedError


def make_data_loader(cfg, **kwargs):
    train_set = make_dataset(cfg, 'train')
    val_set = make_dataset(cfg, 'val')
    test_set = make_dataset(cfg, 'test')

    num_class = cfg.DATASET.N_CLASSES
    train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, **kwargs)
    if test_set is not None:
        test_loader = DataLoader(test_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, **kwargs)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader, num_class

def get_label_colors(cfg):
    if cfg.DATASET.NAME == 'pascal' or cfg.DATASET.NAME == 'coco':
        return get_pascal_labels()

    elif cfg.DATASET.NAME == 'cityscapes':
        return get_cityscapes_labels()

    elif cfg.DATASET.NAME == 'scenenet':
        return get_nyu13_labels()

    elif cfg.DATASET.NAME in ['sunrgbd', 'matterport3d']:
        return get_sunrgbd_labels()

    else:
        raise NotImplementedError