import os
import numpy as np
import argparse
from scipy.optimize import linear_sum_assignment
import json
import sys
from utils import open_tif

class Logger(object):
    def __init__(self, logFile="Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def cal_dice(segmap1, segmap2):
    """Calculate Dice Similarity Coefficients (DSC) for a given label"""
    segmap1 = np.asarray(segmap1).astype(bool)
    segmap2 = np.asarray(segmap2).astype(bool)
    intersection = np.logical_and(segmap1, segmap2)
    return 2. * intersection.sum() / (segmap1.sum() + segmap2.sum())

def calculate_iou(segmentation_ref, segmentation_pred, label=1):
    """Calculate Intersection over Union (IoU) for a given label"""
    intersection = np.logical_and(segmentation_ref == label, segmentation_pred == label)
    union = np.logical_or(segmentation_ref == label, segmentation_pred == label)
    iou = np.sum(intersection) / np.sum(union)
    return iou

# idx_mapping = {'granules': [4], 'mitochondria': [], 'nucleus': [2, 3]}

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("--name", default='c4', type=str)
parser.add_argument("--dataset_path", default='/root/autodl-tmp/cell/dataset/betaseg', type=str, help="Path to the dataset directory.")
parser.add_argument("--pred_dir", default="./output", type=str, help="Path to the segmentation predictions directory")
parser.add_argument("--pred_name", default="<result>.tif", type=str, help="Path to the segmentation predictions file, numpy or tif file.")
parser.add_argument("--mapping_dir", default="./mappings", type=str)
parser.add_argument("--n_clusters", default=None, type=int, help="Number of clusters in the segmentation.")
parser.add_argument("--find_clusters", default=True, help="Flag to find clusters and generate a mapping if True.")
parser.add_argument("--cmd_log_dir", default="./metrics", type=str)
args = parser.parse_args()

# Prepare for config
name = args.name

dataset_path = args.dataset_path
n_clusters = args.n_clusters
pred_dir = args.pred_dir
pred_name = args.pred_name
mapping_name = pred_name.replace(".tif", ".json")

os.makedirs(args.mapping_dir, exist_ok=True)
mapping_path = os.path.join(args.mapping_dir, mapping_name)
pred_path = os.path.join(args.pred_dir, pred_name)

os.makedirs(args.cmd_log_dir, exist_ok=True)
cmd_log_file_name = os.path.join(args.cmd_log_dir, pred_name[:-4])
sys.stdout = Logger(cmd_log_file_name + '.txt')

# Paths to ground truth masks for different organelles
mask_gt_path = os.path.join(dataset_path, f'high_{name}_segs/high_{name}_membrane_full_mask.tif')
granules_path = os.path.join(dataset_path, f'high_{name}_segs/high_{name}_granules.tif')
mitochondria_path = os.path.join(dataset_path, f'high_{name}_segs/high_{name}_mitochondria_mask.tif')
nucleus_path = os.path.join(dataset_path, f'high_{name}_segs/high_{name}_nucleus_mask.tif')

# Load ground truth masks
mask_gt = open_tif(mask_gt_path, binary=True)
granules_gt = open_tif(granules_path, binary=True)
mitochondria_gt = open_tif(mitochondria_path, binary=True)
nucleus_gt = open_tif(nucleus_path, binary=True)

# Mapping of organelles to their corresponding ground truth masks
gt_mapping = {'nucleus': nucleus_gt, 'mitochondria': mitochondria_gt, 'granules': granules_gt}

# Load the prediction file
pred = open_tif(pred_path, binary=False)

# Apply mask to predictions
pred = pred * mask_gt

# Generate or load cluster-to-organ structure mapping
# or define it manually, e.g. idx_mapping = {'granules': [6, 7], 'mitochondria': [3], 'nucleus': [4]}
# Attention: We assume the background label is 0. If idx_mapping contains 0, you need to add pred[pred == 0] = n_clusters + 1.
# And do not forget to change to pred[pred == 0] = n_clusters + 1 in another file when using saved idx_mapping.
if args.find_clusters or not os.path.exists(mapping_path):
    idx_mapping = {'granules': [], 'mitochondria': [], 'nucleus': []}
    name_mapping = {0: "unrecognized", 1: "nucleus", 2: "mitochondria", 3: "granules"}
    unrecognized_gt = mask_gt.astype(np.int16) - ((granules_gt + mitochondria_gt + nucleus_gt) > 0)
    unrecognized_gt[unrecognized_gt < 0] = 0
    unrecognized_gt = unrecognized_gt.astype(np.uint8)

    gt_mapping_all = {'nucleus': nucleus_gt, 'mitochondria': mitochondria_gt,
                      'granules': granules_gt, 'unrecognized': unrecognized_gt}
    dice_matrix = np.zeros((4, n_clusters))
    for r in range(0, 4):
        for p in range(0, n_clusters):
            dice_matrix[r, p] = cal_dice(gt_mapping_all[name_mapping[r]], pred == p)
    row_ind, col_ind = linear_sum_assignment(-dice_matrix)
    used_p = set(col_ind)
    back_flag = False
    for r, c in zip(row_ind, col_ind):
        if r == 0:
            continue
        if r > 0 and np.sum(gt_mapping_all[name_mapping[r]]) == 0 or dice_matrix[r, c] < 0.1:
            continue
        if c == 0:
            pred[pred == 0] = n_clusters + 1
            c = n_clusters + 1
            used_p.add(c)
        idx_mapping[name_mapping[r]].append(int(c))
        for p, dice in enumerate(dice_matrix[r]):
                if p not in used_p and dice > 0.1 and dice_matrix[0, p] < 0.3:
                    if p == 0:
                        pred[pred == 0] = n_clusters + 1
                        p = n_clusters + 1
                        used_p.add(0)
                    idx_mapping[name_mapping[r]].append(int(p))
                    used_p.add(p)
    del gt_mapping_all
    with open(mapping_path, "a+") as f:
        json.dump(idx_mapping, f)
else:
    with open(mapping_path, "a+") as f:
        idx_mapping = json.load(mapping_path)

# Initialize metrics lists
dices = []
ious = []

# Evaluate performance for each organelle
for idx, organelle_name in enumerate(idx_mapping.keys()):
    organelle_gt = gt_mapping[organelle_name]
    organelle_pred = np.zeros_like(mask_gt)
    for label in idx_mapping[organelle_name]:
        organelle_pred[pred == label] = 1
    # Calculate metrics
    dice_organelle = cal_dice(organelle_gt, organelle_pred)
    iou_organelle = calculate_iou(organelle_gt, organelle_pred)
    print(f'{idx + 1}, dice, {organelle_name}:{dice_organelle}')
    print(f'{idx + 1}, iou, {organelle_name}:{iou_organelle}')
    dices.append(dice_organelle)
    ious.append(iou_organelle)

# Print mean metrics
print(f'mean dice: {np.mean(dices)}')
print(f'mean iou: {np.mean(ious)}')
