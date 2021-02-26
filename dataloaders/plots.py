from PIL import Image, ImageDraw, ImageFont
import torch, random
from dataloaders.utils import xywh2xyxy

## Modified from https://github.com/ultralytics/yolov5/utils/plots.py under GNU License
def plot_bboxes(images, targets, fname='images.jpg', names=None, max_size=640, max_subplots=16, colors=None):
    # Plot image grid with labels
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.float32)  # init
    for i, sample in enumerate(zip(images, targets)):
        img, image_targets = sample
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = Image.fromarray(img.transpose(1, 2, 0)[:, :, :3].astype(np.uint8))
        if scale_factor < 1:
            img = img.resize((w, h))

        if image_targets.shape[0] > 0:
            boxes = xywh2xyxy(image_targets[:, 1:5]).T
            classes = image_targets[:, 0].astype('int')
            labels = image_targets.shape[1] == 5  # labels if no conf column
            conf = None if labels else image_targets[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale_factor < 1:  # absolute coords need scale if image scales
                    boxes *= scale_factor
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                if colors is not None:
                    color = colors[cls % len(colors)]
                else:
                    color = None
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = '%s' % cls if labels else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, img, label=label, color=color, line_thickness=tl)

        # Image border
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), img.size], outline=(255, 255, 255), width=3)
        mosaic[block_y:block_y + h, block_x:block_x + w, :] = np.array(img)

    if fname:
        r = min(1280. / max(h, w) / ns, 1.0)  # ratio to limit image size
        mosaic_img = Image.fromarray(mosaic[:, :, :3].astype(np.uint8))
        mosaic_img.resize((int(ns * w * r), int(ns * h * r)))
        mosaic_img.save(fname)  # PIL save
    return mosaic

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    draw = ImageDraw.Draw(img)
    draw.rectangle([c1, c2], outline=color, width=tl) # [(x0, y0), (x1, y1)]
    if label:
        tf = max(tl - 1, 1)  # font thickness
        fnt = ImageFont.truetype("/Library/Fonts/Arial.ttf") #, tl / 3)
        t_size = fnt.getsize(label)
        c2 = c1[0] + t_size[0]*1.5, c1[1] - t_size[1] - 3
        draw.rectangle([c1, c2], fill=color)  # filled
        draw.text((c1[0], c1[1] - t_size[1]), label, fnt=fnt)

if __name__ == '__main__':
    from dataloaders import make_dataset
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    from dataloaders.config.defaults import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/sunrgbd.yaml')
    cfg.merge_from_list(['DATASET.ANNOTATION_TYPE', 'bbox',
                         'DATASET.NO_TRANSFORMS', True,
                         'TRAIN.BATCH_SIZE', 1])

    # Same as main method of dataloaders.datasets.coco
    val = make_dataset(cfg, split='val')
    dataloader = DataLoader(val, batch_size=16, shuffle=False, num_workers=0)
    names = [x.replace('_', ' ') for x in val.loader.class_names]

    for ii, sample in enumerate(dataloader):
        mosaic = plot_bboxes(sample["image"], sample["label"], names=names)
        plt.figure()
        plt.imshow(mosaic.astype(np.uint8))
        plt.show()
        break