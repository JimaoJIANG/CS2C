import copy
import os
import numpy as np
import torch
from model import Graph_Learning
from dataset import get_dataloader, get_slice
import argparse
from utils import read_yaml, save_checkpoint
import sys
from datetime import datetime
from PIL import Image

class Logger(object):
    def __init__(self, logFile="Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# Argument parser setup
parser = argparse.ArgumentParser(description="CS2C Training")
parser.add_argument("--model_config_dir", default="./configs", type=str, help='Directory to config files.')
parser.add_argument("--model_config_name", default="default.yaml", type=str, help='Directory to config file names.')
parser.add_argument("--save_dir", default="./checkpoints", type=str, help='Directory to save models.')
parser.add_argument("--cmd_log_dir", default="./logs", type=str, help='Directory to save logs.')
parser.add_argument("--plot_dir", default="./plots", type=str, help='Directory to save temp results.')
parser.add_argument("--set_kmeans_init", default=True, type=bool, help='Set initial KMeans anchors or not.')
parser.add_argument("--slice_path", default=r"/path/to/data/high_c3_source.pt", type=str, help='Path to anchor image.')
parser.add_argument("--slice_num", default=500, type=int,
                    help='Specify the slice for the volumetric image, we randomly choose an interlayer.')
parser.add_argument("--slice_size", default=[560, 560], type=list, help='Anchor slice size.')
parser.add_argument("--pretrained", default=None, type=str, help='Path to existing models to reload.')
args = parser.parse_args()


cfg = read_yaml(os.path.join(args.model_config_dir, args.model_config_name))
cmd_log_dir = os.path.join(args.cmd_log_dir, cfg['MODEL']['name'])
save_dir = os.path.join(args.save_dir, cfg['MODEL']['name'])
plot_dir = os.path.join(args.plot_dir, cfg['MODEL']['name'])

os.makedirs(cmd_log_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

current_time = str(datetime.now()).split('.')[0].replace(' ', '_').replace(':', '-')
sys.stdout = Logger(os.path.join(cmd_log_dir, f"log_{cfg['MODEL']['name']}_{current_time}.log"))

print("Starting...", flush=True)
print("model_config:", cfg, flush=True)
config_name = args.model_config_name
config_name = config_name.split('.')[0]


idx_mapping = {0: "background", 3: "granules", 6: "mitochondria", 7: "nuclei"}
device = torch.device(f"cuda:{cfg['ENGINE']['device']}" if torch.cuda.is_available() else "cpu")

os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.cmd_log_dir, exist_ok=True)
os.makedirs(args.plot_dir, exist_ok=True)

model = Graph_Learning(cfg=cfg["MODEL"]).to(device)
cfg_copied = copy.deepcopy(cfg)
if cfg["ENGINE"]["load_pretrained"]:
    print(f"==> Load Model..", flush=True)
    model.load_state_dict(torch.load(args.pretrained, map_location=device))
    print(f"Successful load models from {args.pretrained}")
    centers_path = args.pretrained.replace('.pt', '.txt')
    if cfg["ENGINE"]["stage_two_count"] <= 0:
        if os.path.exists(centers_path):
            model.centoids.data = torch.tensor(np.loadtxt(centers_path)).to(torch.float32).to(device)
            model.initialized = True
            print(f"Successful load centers from {centers_path}")
        else:
            print("Warning! Can't find centers!")

print(f"==> Model ready!", flush=True)
print(f"==> Preparing data..")
dataloader = get_dataloader(cfg)
optim_parameters = list(model.GNN_Eigens.parameters()) + list(model.Assignment_Spec.parameters()) + [model.centoids]
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, optim_parameters), lr=cfg["OPTIM"]["lr_initial"])
print(f"==> Data ready!")

count = 0
print_this_iter = False
update_kl = False

train_first_stage = True
train_second_stage = False
train_third_stage = False

slice_init = get_slice(args.slice_path, slice_num=args.slice_num, input_size=args.slice_size).to(device)
n_segments_test = int(args.slice_size[0] * args.slice_size[1] * cfg["MODEL"]["n_segments"] / (cfg["DATASET"]["input_size"] ** 2))

for params in model.Assignment_Spec.parameters():
    params.requires_grad = False
model.centoids.requires_grad = False

for epoch in range(cfg["ENGINE"]["epoch"]):
    print(f"=========================Starting Epoch {epoch + 1}/{cfg['ENGINE']['epoch']}============================")
    epoch_loss = []
    for img, seg in dataloader:
        img = img.cuda(cfg["ENGINE"]["device"])
        seg = 255 * seg[0].numpy()


        if not update_kl and (count + 1 - cfg["ENGINE"]["stage_two_start"]) > cfg["ENGINE"]["kl_centers_start"]:
            model.centoids.requires_grad = True
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg["OPTIM"]["lr_adjusted"]
            update_kl = True
            print("Starting updating KL centers!")

        if train_first_stage and (count + 1) > cfg["ENGINE"]["stage_two_start"]:
            print("====================Begin second stage training!============================")
            train_second_stage = True
            train_first_stage = False

            model.cfg["w_emb"] = 0

            if args.set_kmeans_init and not model.initialized:
                model.initialize_centers(slice_init, n_segments=n_segments_test, chunk_size=1200)
                model.initialized = True
            for params in model.Assignment_Spec.parameters():
                params.requires_grad = True
            for params in model.GNN_Eigens.parameters():
                params.requires_grad = False
            model.centoids.requires_grad = False


        if train_second_stage and (count + 1) > cfg["ENGINE"]["stage_three_start"]:
            print("====================Begin third stage training!============================")
            train_second_stage = False
            train_third_stage = True
            model.cfg["w_emb"] = cfg_copied["MODEL"]["w_emb"]
            for params in model.Assignment_Spec.parameters():
                params.requires_grad = True
            for params in model.GNN_Eigens.parameters():
                params.requires_grad = True

        optimizer.zero_grad()
        # First Stage: Training Eigens
        if train_first_stage:
            segments, loss, eigens = model.extract_eigens(img)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

            if print_this_iter:
                print(f"\n---------------------Now we have calculated {count + 1} image slices.----------------------")
                print(f"[Stage I, Overall Loss]: {loss.item()}")
                print(f"--------------------Now we have calculated {count + 1} image slices.----------------------\n")

            if (count + 1) % cfg["ENGINE"]["eigens_plot_count"] == 0:
                plot_count_dir = os.path.join(plot_dir, f"results_{count + 1}")
                os.makedirs(plot_count_dir, exist_ok=True)

                seg_image = (255 * (seg / (np.max(seg) + 1e-16))).astype(np.uint8)
                image = img[0].detach().cpu().numpy()
                image = (255 * (image / (np.max(image) + 1e-16))).astype(np.uint8)
                Image.fromarray(seg_image).save(os.path.join(plot_count_dir, f'gt_seg_{count + 1}.png'))
                Image.fromarray(image).save(os.path.join(plot_count_dir, f'image_seg_{count + 1}.png'))

                eigens = eigens.T
                eigens = eigens.detach().cpu().numpy()
                for i in range(eigens.shape[0]):
                    multi_eigens = np.zeros_like(seg)
                    for idx, label in enumerate(segments.flatten()):
                        multi_eigens.ravel()[idx] = eigens[i][label - 1]
                    multi_eigens = (255 * (multi_eigens / (np.max(multi_eigens) + 1e-16))).astype(np.uint8)
                    Image.fromarray(multi_eigens).save(os.path.join(plot_count_dir, f'eigens_{i}.png'))


        # Second Stage: Training Clustering
        if train_second_stage or train_third_stage:
            segments, clusters_spec, clusters_spa, loss, centroids, eigens = model(img)

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

            if print_this_iter:
                print(f"\n---------------------Now we have calculated {count + 1} image slices.----------------------")
                if train_second_stage:
                    print(f"[Stage II, Overall Loss]: {loss.item()}")
                if train_third_stage:
                    print(f"[Stage III, Overall Loss]: {loss.item()}")
                print(f"--------------------Now we have calculated {count + 1} image slices.----------------------\n")

            if (count + 1) % cfg["ENGINE"]["clusters_plot_count"] == 0:
                multi_seg_image_spec = np.zeros_like(seg)
                multi_seg_image_spa = np.zeros_like(seg)
                for idx, label in enumerate(segments.flatten()):
                    multi_seg_image_spec.ravel()[idx] = clusters_spec[label - 1]
                    multi_seg_image_spa.ravel()[idx] = clusters_spa[label - 1]

                plot_count_dir = os.path.join(plot_dir, f"results_{count + 1}")
                os.makedirs(plot_count_dir, exist_ok=True)

                multi_seg_image_spec = (255 * (multi_seg_image_spec / (np.max(multi_seg_image_spec) + 1e-16))).astype(np.uint8)
                multi_seg_image_spa = (255 * (multi_seg_image_spa / (np.max(multi_seg_image_spa) + 1e-16))).astype(np.uint8)
                seg_image = (255 * (seg / (np.max(seg) + 1e-16))).astype(np.uint8)
                image = img[0].detach().cpu().numpy()
                image = (255 * (image / (np.max(image) + 1e-16))).astype(np.uint8)
                Image.fromarray(multi_seg_image_spec).save(os.path.join(plot_count_dir, f'multi_seg_{count + 1}_spec.png'))
                Image.fromarray(multi_seg_image_spa).save(os.path.join(plot_count_dir, f'multi_seg_{count + 1}_spa.png'))
                Image.fromarray(seg_image).save(os.path.join(plot_count_dir, f'gt_seg_{count + 1}.png'))
                Image.fromarray(image).save(os.path.join(plot_count_dir, f'image_seg_{count + 1}.png'))

                if (count + 1) % cfg["ENGINE"]["eigens_plot_count"] == 0:
                    eigens = eigens.T
                    eigens = eigens.detach().cpu().numpy()
                    for i in range(eigens.shape[0]):
                        multi_eigens = np.zeros_like(seg)
                        for idx, label in enumerate(segments.flatten()):
                            multi_eigens.ravel()[idx] = eigens[i][label - 1]
                        multi_eigens = (255 * (multi_eigens / (np.max(multi_eigens) + 1e-16))).astype(np.uint8)
                        Image.fromarray(multi_eigens).save(os.path.join(plot_count_dir, f'eigens_{i}.png'))


        if (count + 1) % cfg["ENGINE"]["save_count"] == 0:
            state_dict = model.state_dict()
            save_checkpoint(save_dir, state_dict,
                            name=f"{cfg['MODEL']['name']}_{count + 1}.pt")

            if train_second_stage or train_third_stage:
                try:
                    centers_cpu = centroids.data.detach().cpu().numpy()
                    np.savetxt(os.path.join(save_dir, f"{cfg['MODEL']['name']}_{count + 1}.txt"),
                               centers_cpu, fmt='%f')
                except:
                    print('Centers are not well initialized')
        count += 1

    print(f"^^^^^^^^^^^^^^[{cfg['MODEL']['name']}] Overall Epoch Loss: {np.mean(epoch_loss)}^^^^^^^^^^^^^^^^")
    # Release memories
    torch.cuda.empty_cache()

print(f"==> Training finished!")
