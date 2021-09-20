import os
import argparse

import numpy as np
import scipy.ndimage as ndi
import skimage.io as sio
import torch
import glob
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from skimage import feature, morphology
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import get_markers

parser = argparse.ArgumentParser(
        description='Generate a segmentation for a given CTC dataset.')
parser.add_argument('--dataset',
                        help='The CTC Dataset (needs to be inside the `Data` directory.)')
parser.add_argument('--sequence_id',
                        help='The sequence/video ID. (e.g. `01`, `02`)')
parser.add_argument('--basedir', default='../Data/train/',
                        help='Base directory where CTC datasets live.')
parser.add_argument('--erosion', type=int, default=20,
                    help='The size of erosion needed to create the markers from the cell masks')

args = parser.parse_args()

basedir = os.path.join(args.basedir, args.dataset)
sequence_id = args.sequence_id

subdir = os.path.join(basedir, sequence_id)

print(subdir)

folderData = sorted(glob.glob(subdir + "/SEG/*.tif"))

markerDir = subdir + "/MARKER"
os.makedirs(markerDir, exist_ok=True)

for imgPath in folderData:
    img3d = sio.imread(imgPath)

    # for each slice
    markers = []

    for i, img in enumerate(img3d):
        marker = get_markers(img.astype('uint16'), args.erosion)
        markers.append(marker)

    markers = np.stack(markers, axis=0)
    markers = markers.astype('uint8')
    outname = imgPath.split('/')[-1]
    outpath = f'{markerDir}/{outname}'
    sio.imsave(outpath, markers)