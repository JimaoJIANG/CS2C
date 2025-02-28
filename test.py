import os.path
from torchvision.transforms import ToTensor
from model import Graph_Learning
import numpy as np
from utils import read_yaml
import tifffile as tiff
import torch
import argparse
from datetime import datetime
from utils import open_tif

parser = argparse.ArgumentParser()
parser.add_argument("--chunk_size", default=1000, type=int)
parser.add_argument("--cmd_log_dir", default='./logs', type=str)
parser.add_argument("--output_dir", default='./output', type=str)
parser.add_argument("--model_config_name", default=None, type=str)
parser.add_argument("--suffix", default='default', type=str)
parser.add_argument("--epoch", default='4200', type=str)
parser.add_argument("--alpha", default=None, type=float)
parser.add_argument("--n_segments_test", default=None, type=int)
parser.add_argument("--skip_non", default=True, type=bool)
parser.add_argument("--device", default=None, type=int)

args = parser.parse_args()
if args.model_config_name is None:
    args.model_config_name = args.suffix + '.yaml'

pretrained = fr"./checkpoints/{args.suffix}/{args.suffix}_{args.epoch}.pt"

current_time = str(datetime.now()).split('.')[0].replace(' ', '_').replace(':', '-')
os.makedirs(args.cmd_log_dir, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)

cfg = read_yaml(os.path.join("./configs", args.model_config_name))
if args.device is not None:
    cfg["MODEL"]["device"] = args.device
    cfg["ENGINE"]["device"] = args.device
n_segments_test = cfg["MODEL"]["n_segments"]
chunk_size = args.chunk_size
n_clusters = cfg["MODEL"]["n_clusters"]
input_size = cfg["DATASET"]["input_size"]

if args.alpha is not None:
    cfg["MODEL"]["alpha"] = args.alpha
name = 'c4'
dataset_path = fr'/root/autodl-tmp/cell/dataset/betaseg'
mask_gt_path = os.path.join(dataset_path, f'high_{name}_segs/high_{name}_membrane_full_mask.tif')


device = torch.device(f"cuda:{cfg['ENGINE']['device']}" if torch.cuda.is_available() else "cpu")
model = Graph_Learning(cfg=cfg["MODEL"]).to(device)
model.load_state_dict(torch.load(pretrained, map_location=device))
model.eval()


tt = ToTensor()
img_array = np.load('/root/autodl-tmp/cell/dataset/betaseg/high_c4_source.npy')
orgD, orgH, orgW = img_array.shape

if args.n_segments_test is None:
    n_segments_test = int(orgH * orgW * cfg["MODEL"]["n_segments"] / (cfg["DATASET"]["input_size"] ** 2))
else:
    n_segments_test = args.n_segments_test

multi_seg_images_spec = list()
multi_seg_images_spa = list()

mask_gt = open_tif(mask_gt_path)

for idx in range(0, orgD):
    print(f"Processing {idx + 1}/{orgD}")
    img_array_tmp = img_array[idx]
    multi_seg_image_spec_tmp = np.zeros((orgH, orgW, 1))
    multi_seg_image_spa_tmp = np.zeros((orgH, orgW, 1))
    if args.skip_non and np.sum(mask_gt[idx]) == 0:
        print(f'No segmentation, skip {idx + 1}!')
        multi_seg_images_spec.append(np.transpose(multi_seg_image_spec_tmp, (2, 0, 1)))
        multi_seg_images_spa.append(np.transpose(multi_seg_image_spa_tmp, (2, 0, 1)))
        continue

    # Release memories
    if (idx + 1) % 50 == 0:
        torch.cuda.empty_cache()
        
    input = ToTensor()(img_array_tmp).to(device)
    # crop right
    col_diff = torch.diff(input, dim=1)
    is_uniform = (col_diff == 0).all(dim=1)
    first_non_padding_col_index = (is_uniform == 0)[0].nonzero(as_tuple=True)[0].min().item()
    last_non_padding_col_index = (is_uniform == 0)[0].nonzero(as_tuple=True)[0].max().item() + 1
    input = input[:, :, first_non_padding_col_index:last_non_padding_col_index]
    _, new_H, new_W = input.shape

    # Pre-calculate features for whole image
    feas = model.extract_feas(input)

    # No overlapped area
    multi_seg_image_spec, multi_seg_image_spa, segments, eigens = \
        model.test(input, feas, n_segments=n_segments_test, chunk_size=chunk_size)
    del input
    del feas
    del eigens

    ## Back to original size
    multi_seg_image_spec_tmp[:, first_non_padding_col_index:last_non_padding_col_index, :] = multi_seg_image_spec
    multi_seg_image_spec_tmp = np.transpose(multi_seg_image_spec_tmp, (2, 0, 1))
    multi_seg_image_spa_tmp[:, first_non_padding_col_index:last_non_padding_col_index, :] = multi_seg_image_spa
    multi_seg_image_spa_tmp = np.transpose(multi_seg_image_spa_tmp, (2, 0, 1))
    multi_seg_images_spec.append(multi_seg_image_spec_tmp)
    multi_seg_images_spa.append(multi_seg_image_spa_tmp)

multi_seg_images_spec = np.concatenate(multi_seg_images_spec, axis=0).astype(np.uint8)
multi_seg_images_spa = np.concatenate(multi_seg_images_spa, axis=0).astype(np.uint8)

tiff.imwrite(os.path.join(args.output_dir, f'high_c4_predict_n{n_segments_test}_{args.epoch}_{args.suffix}_spec.tif'),
             multi_seg_images_spec, compression='ADOBE_DEFLATE')
tiff.imwrite(os.path.join(args.output_dir, f'high_c4_predict_n{n_segments_test}_{args.epoch}_{args.suffix}_spa.tif'),
             multi_seg_images_spa, compression='ADOBE_DEFLATE')