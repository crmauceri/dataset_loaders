import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from torchvision import transforms
from warnings import warn
from deeplab3.dataloaders import custom_transforms as tr
import deeplab3.dataloaders.datasets.scenenet_pb2 as sn
from deeplab3.dataloaders.SampleLoader import SampleLoader

class SceneNetSegmentation(data.Dataset):
    NUM_CLASSES = 14

    def __init__(self, cfg, split="train"):

        self.split = split
        self.cfg = cfg
        self.mode = cfg.DATASET.MODE

        if split=="train":
            protobuf_paths = [os.path.join(cfg.DATASET.ROOT, 'train_protobufs/scenenet_rgbd_train_{}.pb.filtered'.format(1))] #TODO Use all trajectories # for i in range(1,2)]
            self.root = os.path.join(cfg.DATASET.ROOT, 'train')
        elif split=="val":
            protobuf_paths = [os.path.join(cfg.DATASET.ROOT, 'scenenet_rgbd_val.pb.filtered')]
            self.root = os.path.join(cfg.DATASET.ROOT, 'val')
        elif split=="test":
            # Reserve the 1st training partition as a test set
            protobuf_paths = [os.path.join(cfg.DATASET.ROOT, 'train_protobufs/scenenet_rgbd_train_{}.pb.filtered'.format(0))]
            self.root = os.path.join(cfg.DATASET.ROOT, 'train')
        else:
            raise ValueError("SceneNet split {} not implemented".format(split))

        self.dataset = []
        for protobuf_path in protobuf_paths:
            trajectories = sn.Trajectories()
            try:
                with open(protobuf_path, 'rb') as f:
                    trajectories.ParseFromString(f.read())
                for traj in trajectories.trajectories:
                    for view in traj.views:
                        self.dataset.append({'img_path': os.path.join(self.root, traj.render_path, 'photo', '{}.jpg'.format(view.frame_num)),
                                             'depth_path': os.path.join(self.root, traj.render_path, 'depth', '{}.png'.format(view.frame_num)),
                                             'lbl_path': os.path.join(self.root, traj.render_path, 'semantic', '{}.png'.format(view.frame_num))})
            except IOError:
                print('Scenenet protobuf data not found at location:{0}'.format(protobuf_path))
                print('Please ensure you have copied the pb file to the data directory')

        self.loader = ScenenetSampleLoader(cfg, split)

        print("Found %d %s images" % (len(self.dataset), split))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index, no_transforms=False):

        img_path = self.dataset[index]['img_path']
        lbl_path = self.dataset[index]['lbl_path']
        depth_path = self.dataset[index]['depth_path']

        try:
            sample = self.loader.load_sample(img_path, depth_path, lbl_path, no_transforms=no_transforms)

        except IOError as e:
            # Instead of raising error, warn and continue training, but this image should probably be added to the filter in __init__
            warn(e, category=RuntimeWarning)
            _img = Image.fromarray(np.zeros(()) if self.use_depth else np.zeros(()))
            _target = np.zeros((), dtype=np.uint8)
            sample = {'image': _img, 'label': _target}

        return sample

class ScenenetSampleLoader(SampleLoader):
    def __init__(self, cfg, split="train"):
        super().__init__(cfg, cfg.DATASET.MODE, split,
                        cfg.DATASET.BASE_SIZE, cfg.DATASET.CROP_SIZE)

        self.void_classes = [0]
        self.valid_classes = range(1, 13)
        self.class_names = ['bed', 'books', 'ceiling', 'chair', 'floor', 'furniture', 'objects', 'picture', 'sofa',
                            'table', 'tv', 'wall', 'window']

        self.NUM_CLASSES = len(self.valid_classes)
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

    def normalizationFactors(self):
        if self.mode == "RGBD":
            print('Using RGB-D input')
            # Data mean and std empirically determined from 1000 Scenenet samples
            self.data_mean = [0.291,  0.329,  0.291,  0.126]
            self.data_std = [0.190,  0.190,  0.185,  0.179]
        elif self.mode == "RGB":
            print('Using RGB input')
            self.data_mean = [0.291,  0.329,  0.291]
            self.data_std = [0.190,  0.190,  0.185]
        elif self.mode == "RGB_HHA":
            raise NotImplementedError("ScenenetSampleLoader: HHA images not implemented")

    def getLabels(self, lbl_path):
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint16)
        _target = Image.fromarray(_tmp).convert('L')
        return _target

    def loadDepth(self, depth_path):
        _depth_arr = np.array(Image.open(depth_path)).astype(np.uint16)
        _depth = Image.fromarray(_depth_arr).convert('L')
        return _depth

if __name__ == '__main__':
    from deeplab3.config.defaults import get_cfg_defaults
    from deeplab3.dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description="Test scenenet Loader")
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

    scenenet_train = SceneNetSegmentation(cfg, split='test')

    dataloader = DataLoader(scenenet_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='scenenet')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= scenenet_train.loader.data_std
            img_tmp += scenenet_train.loader.data_mean
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

