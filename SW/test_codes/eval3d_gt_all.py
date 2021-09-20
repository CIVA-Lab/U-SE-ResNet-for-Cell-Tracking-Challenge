import argparse
import os
import sys
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io as sio
import torch
from scipy.ndimage import label
from torchvision import transforms

from data import *
from postproc import *

filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(
        description='Generate a segmentation for a given CTC dataset.')
    parser.add_argument('--dataset',
                        help='The CTC Dataset (needs to be inside the `trainsets` directory.)')
    parser.add_argument('--sequence_id',
                        help='The sequence/video ID. (e.g. `01`, `02`)')
    parser.add_argument('--basedir', default='../../Data/test/',
                        help='Base directory where CTC datasets live.')
    parser.add_argument('--resize_to',
                        help='Resize datasets with non homogenous image size (e.g. Fluo-C2DL-MSC) to a similar size.')
    parser.add_argument('--area_threshold', type=int, default=200,
                        help='Threshold area under which detections are disregared (considered spurious).')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for inference (`cpu` for CPU, `cuda` for GPU).')

    args = parser.parse_args()

    basedir = os.path.join(args.basedir, args.dataset)
    sequence_id = args.sequence_id

    try:
        model = torch.load(f'../trained_models/gt_all_best_model.pth', map_location=args.device)
    except Exception as e:
        print(e)
        print(f'You do not have any trained model for the dataset {args.dataset},' +
            f'please make sure you run the following first' +
            f'\n\ttrain.py --dataset {args.dataset}\n')

        exit(1)

    subdir = os.path.join(basedir, sequence_id)
    paths = sorted([os.path.join(subdir, p) for p in os.listdir(subdir)])

    for path in paths:      
        image3d = sio.imread(path)
        image3d = normalize(image3d)

        _, H, W = image3d.shape
        newH = (H // 32 + 1) * 32 if H % 32 != 0 else H
        newW = (W // 32 + 1) * 32 if W % 32 != 0 else W

        uncrop = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((newH, newW)),
            transforms.ToTensor()
        ])

        crop = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((H, W)),
            transforms.ToTensor()
        ])

        # for each slice
        mask = []
        markers = []
        for i, image in enumerate(image3d):
            
            # Squeeze twice for batch B and channel C
            image = torch.Tensor(image).unsqueeze(0)
            image = uncrop(image).unsqueeze(0)
            image = image.to(args.device)

            net_out = model(image).squeeze()
            net_out = net_out.detach().cpu()
            m1, m2 = net_out

            m1 = crop(m1).squeeze().numpy()
            m2 = crop(m2).squeeze().numpy()

            mask.append(m1)
            markers.append(m2)
            
        mask = np.stack(mask, axis=0)
        markers = np.stack(mask, axis=0)
        
        labeled_mask = postprocess_mask_and_markers_3d(mask, markers, area_thresh=1)

        labeled_mask = labeled_mask.astype('uint16')

        outname = path.split('/')[-1]
        outdir = '/'.join(path.split('/')[:-1]) + '_RES-allGT'

        os.makedirs(outdir, exist_ok=True)
        outpath = f'{outdir}/mask{outname[1:]}'
        print(f'Saving to: {outpath} ...')
        sio.imsave(outpath, labeled_mask)   

if __name__ == '__main__':
    main()
