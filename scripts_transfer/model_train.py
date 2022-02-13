import detectron2
import torch
from pathlib import Path
import random
import os
import numpy as np
import pycocotools.mask as mask_util
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
cfg = get_cfg()
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.evaluation.evaluator import DatasetEvaluator


setup_logger()


metadata = MetadataCatalog.get('sartorius_train')
train_ds = DatasetCatalog.get('sartorius_train')

# set seed
seed = 2021

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def precision_at(th, iou):
    matches = iou > th
    true_positives = np.sum(matches, axis=1) == 1
    false_positives = np.sum(matches, axis=0) == 0
    false_negatives = np.sum(matches, axis=1) == 0
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

def score(pred, target):
    pred_masks = pred['instances'].pred_masks.cpu().numpy()
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targets = list(map(lambda x:x['segmentation'], target))
    ious = mask_util.iou(enc_preds, enc_targets, [0]*len(enc_targets))
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)
    m = np.mean(prec)
    return m

class MAPIOUEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']:item['annotations'] for item in dataset_dicts}
            
    def reset(self):
        self.scores = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            if len(out['instances']) == 0:
                self.scores.append(0)    
            else:
                target = self.annotations_cache[inp['image_id']]
                self.scores.append(score(out, target))

    def evaluate(self):
        return {"MaP IoU": np.mean(self.scores)}

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)
    

def main():
    data_path=Path('../input/sartorius-cell-instance-segmentation/')
    cfg.INPUT.MASK_FORMAT='bitmask'
    register_coco_instances('sartorius_train',{}, '../input/sartorius-cell-instance-segmentation-coco/annotations_train.json', data_path)
    register_coco_instances('sartorius_val',{},'../input/sartorius-cell-instance-segmentation-coco/annotations_val.json', data_path)
    cfg.merge_from_file(detectron2.model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("sartorius_train",)
    cfg.DATASETS.TEST = ("sartorius_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    # load_pretrain_model_weights
    # transfer learning
    cfg.MODEL.WEIGHTS = '../input/pretrained-model-with-livecell/model_best.pth' 
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0005 
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []    
    cfg.SOLVER.CHECKPOINT_PERIOD = len(DatasetCatalog.get('sartorius_train'))
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
    cfg.TEST.EVAL_PERIOD = len(DatasetCatalog.get('sartorius_train'))

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()