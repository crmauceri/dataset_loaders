from deeplab3.dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd, sunrgbd, scenenet
from torch.utils.data import DataLoader

def make_data_loader(cfg, **kwargs):

    if cfg.DATASET.NAME == 'pascal':
        if cfg.DATASET.MODE != "RGB":
            raise ValueError('RGBD DataLoader not implemented')
        train_set = pascal.VOCSegmentation(cfg, split='train')
        val_set = pascal.VOCSegmentation(cfg, split='val')
        if cfg.DATASET.USE_SBD:
            sbd_train = sbd.SBDSegmentation(cfg, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

    elif cfg.DATASET.NAME == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(cfg, split=cfg.DATASET.CITYSCAPES.TRAIN_SET)
        val_set = cityscapes.CityscapesSegmentation(cfg, split='val')
        test_set = cityscapes.CityscapesSegmentation(cfg, split='test')

    elif cfg.DATASET.NAME == 'scenenet':
        train_set = scenenet.SceneNetSegmentation(cfg, split='train')
        val_set = scenenet.SceneNetSegmentation(cfg, split='val')
        test_set = scenenet.SceneNetSegmentation(cfg, split='test')

    elif cfg.DATASET.NAME == 'coco':
        train_set = coco.COCOSegmentation(cfg, split='train')
        val_set = coco.COCOSegmentation(cfg, split='val')
        test_set = None

    elif cfg.DATASET.NAME in ['sunrgbd', 'matterport3d']:
        train_set = sunrgbd.RGBDSegmentation(cfg, split='train')
        val_set = sunrgbd.RGBDSegmentation(cfg, split='val')
        test_set = None

    else:
        raise NotImplementedError

    num_class = cfg.DATASET.N_CLASSES
    train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, **kwargs)
    if test_set is not None:
        test_loader = DataLoader(test_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, **kwargs)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader, num_class
