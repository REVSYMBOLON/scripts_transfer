import detectron2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
cfg = get_cfg()
import cv2 as cv
import torch
import os

import matplotlib.pyplot as plt
import numpy as np
from fastcore.all import *

# set config for all func
ths = [.15, .35, .55]
pixels_lows = [75, 150, 75]

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def rle_encode(img):
    pixels = np.concatenate([[0], img.flatten(), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def mask_op(fn, model):
    im = cv.imread(str(fn))
    pred = model(im)
    pred_class = torch.mode(pred['instances'].pred_classes)[0]
    take = pred['instances'].scores >= ths[pred_class]
    pred_masks = pred['instances'].pred_masks[take]
    pred_masks = pred_masks.cpu().numpy()
    res = []
    used = np.zeros(im.shape[:2], dtype=int)
    print(pred)
    for mask in pred_masks:
        mask = mask * (1-used)
        if mask.sum() >= pixels_lows[pred_class]:
            used += mask
            res.append(rle_encode(mask))
    return res

def infer():
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.INPUT.MASK_FORMAT='bitmask'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
    # set to your own model name
    cfg.MODEL.WEIGHTS = os.path.join('../input/sartorius-transfer-learning-model', "model_best.pth")  
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    model = DefaultPredictor(cfg)

    # choose your own test samples
    test_files = (data_path/'test').ls()
    encoded_masks = mask_op(test_files[1], model)

    _, axs = plt.subplots(1,2, figsize=(40,15))
    axs[1].imshow(cv.imread(str(test_files[1])))
    for encode in encoded_masks:
        decode = rle_decode(encode, (520, 704))
        axs[0].imshow(np.ma.masked_where(decode==0, decode))