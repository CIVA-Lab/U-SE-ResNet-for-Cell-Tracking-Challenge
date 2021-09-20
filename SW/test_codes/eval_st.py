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
from postproc import postprocess_mask_and_markers

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
        model = torch.load(f'../trained_models/st_sw_{args.dataset}_best_model.pth', map_location=args.device)
    except Exception as e:
        print(e)
        print(f'You do not have any trained model for the dataset {args.dataset},' +
            f'please make sure you run the following first' +
            f'\n\ttrain.py --dataset {args.dataset}\n')

        exit(1)

    subdir = os.path.join(basedir, sequence_id)
    paths = sorted([os.path.join(subdir, p) for p in os.listdir(subdir)])
    evalset = CTC2DDatasetEval(pd.DataFrame({
        'image': paths
    }))

    for i in range(len(evalset)):
        p = evalset.impaths[i]

        H, W = sio.imread(evalset.impaths[i]).shape
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((H, W)),
            transforms.ToTensor()
        ])
        net_out = model(evalset[i].unsqueeze(0).to(args.device)).squeeze()
        net_out = net_out.detach().cpu()

        mask, markers = net_out
        mask = transform(mask).squeeze().numpy()
        markers = transform(markers).squeeze().numpy()

        labeled_mask = postprocess_mask_and_markers(mask, markers, area_thresh=args.area_threshold)

        labeled_mask = labeled_mask.astype('uint16')

        outname = p.split('/')[-1]
        outdir = '/'.join(p.split('/')[:-1]) + '_RES-ST'

        os.makedirs(outdir, exist_ok=True)
        outpath = f'{outdir}/mask{outname[1:]}'
        print(f'Saving to: {outpath} ...')
        sio.imsave(outpath, labeled_mask)    

if __name__ == '__main__':
    main()
