# Basic
import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
# Deeep learning
import torch
import torchvision
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# My stuff
from data import *
from trainer import SWTrainEpoch, SWValidEpoch
from utils import *


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Train a model.')
parser.add_argument('--dataset',
                    help='The CTC Dataset (needs to be inside the `Data` directory.)')
parser.add_argument('--erosion', type=int, default=20,
                    help='The size of erosion needed to create the markers from the cell masks')
parser.add_argument('--use_gold_truth', type=bool, nargs='?', const=True, default=False,
                        help='Use Gold Truth when available?.')
parser.add_argument('--resize_to', type=int, nargs=2,
                    help='Resize datasets with non homogenous image size (e.g. Fluo-C2DL-MSC) to a similar size.')
parser.add_argument('--train_resolution', type=int, nargs=2, default=[512, 512],
                    help='Training patch resolution.')
parser.add_argument('--device', default='cuda',
                        help='Device to use for inference (`cpu` for CPU, `cuda` for GPU).')

args = parser.parse_args()

dataset = args.dataset
is_3d = is_ctc_dataset_3d(args.dataset) 

print("Is 3D: ")
print(is_3d)

basedir = os.path.join('../../Data/train/', dataset)
impaths, gtpaths, mrpaths = get_file_lists_st(
    basedir, use_gold_when_available=args.use_gold_truth)

data_df = pd.DataFrame({
    'image': impaths,
    'label': gtpaths,
    'marker': mrpaths
})

train_df, val_df = train_test_split(data_df, test_size=.2, random_state=42)

# print(train_df)
# print(val_df)

class_labels = {
    1: "cell",
}

backbone = 'se_resnet50'
pretrained = 'imagenet'
DEVICE = 'cuda'

train_resolution = args.train_resolution

# train
epochs = 120
batch_size = 16
learning_rate = 0.0001

prefix = 'gt_st' if args.use_gold_truth else 'st'
print(args.use_gold_truth)
now = datetime.now()
date_time = now.strftime("%m-%d-%Y_%H-%M-%S")

training_id = f"{prefix}_weighted_loss_{dataset}_unet_{train_resolution[0]}_{train_resolution[1]}_{backbone}_{pretrained if pretrained else 'nonpretrained'}_{date_time}"

# train from scratch
model = smp.Unet(backbone, in_channels=1, classes=2,
                  activation='sigmoid',
                  encoder_weights=pretrained)

save_dir = f'ckpt/{training_id}'

os.makedirs(save_dir, exist_ok=True)

print('Loading data...')
if is_3d:
    trainset = CTC3DDatasetWithMarkersSWMR(
        train_df, erosion=args.erosion, resolution=train_resolution, resize_to=args.resize_to)
    valset = CTC3DDatasetWithMarkersSWMR(
        val_df, erosion=args.erosion, augment=False, resize_to=args.resize_to)
else:
    trainset = CTC2DDatasetWithMarkersSWMR(
        train_df, erosion=args.erosion, resolution=train_resolution, resize_to=args.resize_to)
    valset = CTC2DDatasetWithMarkersSWMR(
        val_df, erosion=args.erosion, augment=False, resize_to=args.resize_to)

print('Preparing for training...')
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=8)
valloader = DataLoader(valset, batch_size=batch_size, num_workers=8)

loss = smp.utils.losses.BCELoss()

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Precision(threshold=0.5),
    smp.utils.metrics.Recall(threshold=0.5),
    smp.utils.metrics.Accuracy(threshold=0.5)
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=learning_rate),
])

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = SWTrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=False,
    weight_power=5
)

valid_epoch = SWValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=False,
    weight_power=5
)

writer = SummaryWriter(os.path.join('runs-markers', training_id))

max_score = 0
best_logs = {}

print('Training...')
for i in tqdm(range(epochs)):

    # print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(trainloader, i)
    valid_logs = valid_epoch.run(valloader, i)

    writer.add_scalar('IoU/Training', train_logs["iou_score"], i)
    writer.add_scalar('IoU/Validation', valid_logs["iou_score"], i)

    writer.add_scalar('Loss/Training', train_logs['loss'], i)
    writer.add_scalar('Loss/Validation', valid_logs['loss'], i)

    writer.add_scalar('Accuracy/Training', train_logs["accuracy"], i)
    writer.add_scalar('Accuracy/Validation', valid_logs["accuracy"], i)
    writer.add_scalar('Precision/Training', train_logs["precision"], i)
    writer.add_scalar('Precision/Validation', valid_logs["precision"], i)
    writer.add_scalar('Recall/Training', train_logs["recall"], i)
    writer.add_scalar('Recall/Validation', valid_logs["recall"], i)

    # Log image samples
    # VALIDATION LOG
    _, (img, lab, _) = next(enumerate(valloader))
    image = img[:2]
    output = model(image.to(DEVICE))
    grid_img = torchvision.utils.make_grid(
        (image - image.min()) / (image.max() - image.min()))

    grid_img = torch.cat([grid_img, grid_img], dim=1)

    grid_lab1 = torchvision.utils.make_grid(lab[:, [0]])
    grid_lab2 = torchvision.utils.make_grid(lab[:, [1]])
    grid_lab = torch.cat([grid_lab1, grid_lab2], dim=1)

    grid_out1 = torchvision.utils.make_grid(output[:, [0]])
    grid_out2 = torchvision.utils.make_grid(output[:, [1]])
    grid_out = torch.cat([grid_out1, grid_out2], dim=1)

    writer.add_image('Sample input for validation', grid_img, 0)
    writer.add_image('Sample label for validation', grid_lab, 0)
    writer.add_image('Sample output for validation', grid_out, i)

    # TRAIN LOG
    _, (img, lab, _) = next(enumerate(trainloader))
    image = img[:2]
    output = model(image.to(DEVICE))
    grid_img = torchvision.utils.make_grid(
        (image - image.min()) / (image.max() - image.min()))

    grid_img = torch.cat([grid_img, grid_img], dim=1)

    grid_lab1 = torchvision.utils.make_grid(lab[:, [0]])
    grid_lab2 = torchvision.utils.make_grid(lab[:, [1]])
    grid_lab = torch.cat([grid_lab1, grid_lab2], dim=1)

    grid_out1 = torchvision.utils.make_grid(output[:, [0]])
    grid_out2 = torchvision.utils.make_grid(output[:, [1]])
    grid_out = torch.cat([grid_out1, grid_out2], dim=1)

    writer.add_image('Sample input for training', grid_img, 0)
    writer.add_image('Sample label for training', grid_lab, 0)
    writer.add_image('Sample output for training', grid_out, i)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        best_logs = valid_logs
        torch.save(model, f'{save_dir}/{prefix}_all_best_model.pth')
        torch.save(model, f'../trained_models/{prefix}_all_best_model.pth')
        with open(f'{save_dir}/results.json', 'w') as f:
            f.write(json.dumps(best_logs))

    if i % 30 == 29:
        optimizer.param_groups[0]['lr'] /= 2
        print(
            f'Decrease decoder learning rate to {optimizer.param_groups[0]["lr"]}')

writer.close()