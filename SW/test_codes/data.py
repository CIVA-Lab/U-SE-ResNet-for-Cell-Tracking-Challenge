import os
import re

import numpy as np
import skimage.io as sio
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

def clip_limit(im, clim=0.01):
    if im.dtype == np.dtype('uint8'):
        hist, *_ = np.histogram(im.reshape(-1),
                                bins=np.linspace(0, 255, 255),
                                density=True)
    elif im.dtype == np.dtype('uint16'):
        hist, *_ = np.histogram(im.reshape(-1),
                                bins=np.linspace(0, 65535, 65536),
                                density=True)
    cumh = 0
    for i, h in enumerate(hist):
        cumh += h
        if cumh > 0.01:
            break
    cumh = 1
    for j, h in reversed(list(enumerate(hist))):
        cumh -= h
        if cumh < (1 - 0.01):
            break
    im = np.clip(im, i, j)
    return im

def normalize(arr):
    arr = clip_limit(arr)
    arr = arr.astype('float32')
    return (arr - arr.min()) / (arr.max() - arr.min())


class CTC2DDatasetEval(Dataset):
    def __init__(self, df):
        self.impaths = list(df.image)

    def __len__(self):
        return len(self.impaths)

    def __getitem__(self, idx):
        image = normalize(sio.imread(self.impaths[idx]))        

        H, W = image.squeeze().shape
        newH = (H // 32 + 1) * 32 if H % 32 != 0 else H
        newW = (W // 32 + 1) * 32 if W % 32 != 0 else W

        # Pad by center cropping with larger dims
        image = TF.to_pil_image(image, mode='F')
        image = TF.center_crop(image, (newH, newW))
        image = TF.to_tensor(image)

        return image.type(torch.FloatTensor)
        