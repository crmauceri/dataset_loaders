from dataloaders import make_data_loader
import argparse, os.path, os
import numpy as np
from tqdm import tqdm

from dataloaders.config.defaults import get_cfg_defaults

## Use to convert dataset to YOLO format txt files

def main(cfg):
    datasets = make_data_loader(cfg)
    for dataset in datasets[:3]:
        img_list = []
        if dataset is not None:
            for ii, sample in enumerate(tqdm(dataset)):
                for jj in range(len(sample["id"])):
                    if cfg.DATASET.NAME == 'cityscapes':
                        filepath = sample['id'][jj].replace('leftImg8bit', 'bbox').replace('png', 'txt')
                        img_list.append(sample['id'][jj])
                    elif cfg.DATASET.NAME in ['sunrgbd', 'coco']:
                        id = dataset.dataset.coco_id_index[sample['id'][jj].item()]
                        img_path, depth_path, img_id = dataset.dataset.get_path(id)
                        assert img_id == sample['id'][jj].item()
                        filepath = 'bbox'.join(img_path.rsplit('image', 1))
                        filepath = os.path.splitext(filepath)[0] + '.txt'
                        img_list.append(dataset.dataset.coco.loadImgs(img_id)[0]['file_name'])

                    dir = os.path.dirname(filepath)
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                    #if not os.path.exists(filepath):
                    np.savetxt(filepath, sample['label'][jj], delimiter=",", fmt=['%d', '%10.8f', '%10.8f', '%10.8f', '%10.8f'])

            f = '{}/image_list_{}.txt'.format(cfg.DATASET.ROOT, dataset.dataset.split)
            with open(f, 'w') as fp:
                fp.write('\n'.join(img_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert instance segmentation annotation to yolo txt files")
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
    cfg.merge_from_list(['DATASET.ANNOTATION_TYPE', 'bbox', \
                         'DATASET.NO_TRANSFORMS', True, \
                         'TRAIN.BATCH_SIZE', 1, \
                         'TEST.BATCH_SIZE', 1])
    print(cfg)
    main(cfg)