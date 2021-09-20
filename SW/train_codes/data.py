import os
import re
import cv2

import numpy as np
import scipy.ndimage as ndi
import skimage.io as sio
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from skimage import feature, morphology
from skimage.measure import label
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from skimage.color import rgb2gray

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

def get_markers(lab, erosion):
    markers = np.zeros_like(lab)
    for i in range(1, lab.max() + 1):
        mask = lab == i
        eroded_mask = morphology.binary_erosion(
            mask, np.ones((erosion, erosion)))
        markers[eroded_mask] = 1
    return markers.astype('float32')

def resize_all_slices(im, resize_to):
    im = im.astype('float64')
    return resize(im.transpose(1, 2, 0), resize_to, anti_aliasing=False) \
                .transpose(2, 0, 1).astype('uint16')

def get_weight_map(im):
    borders = feature.canny(im, low_threshold=.1, use_quantiles=True)
    dist_im = ndi.distance_transform_edt(1 - borders)
    wdist = ((dist_im.max() - dist_im)/dist_im.max())
    return wdist

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.abs(torch.randn(tensor.size()) * self.std + self.mean)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_file_lists(basedir, use_gold_when_available=True):
    imdirs = sorted([os.path.join(basedir, p) for p in os.listdir(basedir)
                     if p.find('_') == -1
                     and os.path.isdir(os.path.join(basedir, p))])
    gtdirs = [f'{p}_ST' for p in imdirs]

    gtpaths = sorted([os.path.join(gtdir, 'SEG', p)
                      for gtdir in gtdirs
                      for p in os.listdir(os.path.join(gtdir, 'SEG'))])
    # replace st with existing gt
    if use_gold_when_available:
        for i, p in enumerate(gtpaths):
            gtp = p.replace('_ST', '_GT_ST')
            if os.path.isfile(gtp):
                gtpaths[i] = gtp

    impaths = sorted([os.path.join(imdir, p)
                      for imdir in imdirs
                      for p in os.listdir(os.path.join(imdir))])

    return impaths, gtpaths

def get_file_lists_st(basedir, use_gold_when_available=True):
    imdirs = sorted([os.path.join(basedir, p) for p in os.listdir(basedir)
                     if p.find('_') == -1
                     and os.path.isdir(os.path.join(basedir, p))])
    
    gtdirs = [f'{p}_ST' for p in imdirs]

    mrdirs = [f'{p}_ST' for p in imdirs]

    gtpaths = sorted([os.path.join(gtdir, 'SEG', p)
                      for gtdir in gtdirs
                      for p in os.listdir(os.path.join(gtdir, 'SEG'))])

    mrpaths = sorted([os.path.join(mrdir, 'MARKER', p)
                      for mrdir in mrdirs
                      for p in os.listdir(os.path.join(mrdir, 'MARKER'))])
                    
    # replace st with existing gt
    if use_gold_when_available:
        for i, p in enumerate(gtpaths):
            gtp = p.replace('_ST', '_GT_ST')
            if os.path.isfile(gtp):
                gtpaths[i] = gtp
                mrpaths[i] = gtp

    impaths = sorted([os.path.join(imdir, p)
                      for imdir in imdirs
                      for p in os.listdir(os.path.join(imdir))])

    return impaths, gtpaths, mrpaths

class CTC2DDatasetWithMarkersSW(Dataset):
    def __init__(self, df, augment=True, resolution=(512, 512),
                 imagenet_pretrained=True, erosion=20, resize_to=None):

        impaths = list(df.image)
        gtpaths = list(df.label)

        if resize_to is not None:
            gts = np.stack([resize(sio.imread(p), resize_to, anti_aliasing=False)
                            for p in tqdm(gtpaths)], axis=0)
            ims = np.stack([resize(normalize(sio.imread(p)), resize_to, anti_aliasing=True)
                            for p in tqdm(impaths)], axis=0)
            mks = np.stack([resize(get_markers(sio.imread(p), erosion=erosion), resize_to, anti_aliasing=False)
                            for p in tqdm(gtpaths)])
        else:
            gts = np.stack([sio.imread(p)
                            for p in tqdm(gtpaths)], axis=0)
            ims = np.stack([normalize(sio.imread(p))
                            for p in tqdm(impaths)], axis=0)
            mks = np.stack([get_markers(sio.imread(p), erosion=erosion)
                            for p in tqdm(gtpaths)])

        sws = np.stack([get_weight_map(gt) for gt in gts], axis=0)

        # binarize ground truth for segmentation
        gts = gts > 0

        self.augment = augment
        self.resolution = resolution
        self.ims = ims.astype('float32')
        self.gts = gts.astype('uint8')
        self.mks = mks.astype('uint8')
        self.sws = sws.astype('float32')

    def __len__(self):
        return self.ims.shape[0]

    def __getitem__(self, idx):
        image = self.ims[idx]
        label = self.gts[idx]
        marker = self.mks[idx]
        weight_map = self.sws[idx]

        # Data augmentation
        # -----------------
        if self.augment:

            # ToPILImage
            image = TF.to_pil_image(image, mode='F')
            label = TF.to_pil_image(label)
            marker = TF.to_pil_image(marker)
            weight_map = TF.to_pil_image(weight_map, mode='F')

            # Random horizontal flipping
            if np.random.rand() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
                marker = TF.hflip(marker)
                weight_map = TF.hflip(weight_map)

            # Random vertical flipping
            if np.random.rand() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)
                marker = TF.vflip(marker)
                weight_map = TF.vflip(weight_map)

            # Random Rotation
            angle = np.random.randint(91)
            image = TF.rotate(image, angle)
            label = TF.rotate(label, angle)
            marker = TF.rotate(marker, angle)
            weight_map = TF.rotate(weight_map, angle, fill=0.2)

            # Random crop to 512 x 512
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=self.resolution)
            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label, i, j, h, w)
            marker = TF.crop(marker, i, j, h, w)
            weight_map = TF.crop(weight_map, i, j, h, w)

            # Back to tensor
            image = TF.to_tensor(image)
            label = TF.to_tensor(label)
            marker = TF.to_tensor(marker)
            weight_map = TF.to_tensor(weight_map)

            # Add random noise
            if np.random.rand() > 0.5:
                noise = np.random.rand() * .2
                image = AddGaussianNoise(std=noise)(image)

        else:
            # ToPILImage
            H, W = image.squeeze().shape
            newH = (H // 32 + 1) * 32 if H % 32 != 0 else H
            newW = (W // 32 + 1) * 32 if W % 32 != 0 else W

            image = TF.to_pil_image(image, mode='F')
            label = TF.to_pil_image(label)
            marker = TF.to_pil_image(marker)
            weight_map = TF.to_pil_image(weight_map)

            # # Pad by center cropping with larger dims
            image = TF.center_crop(image, (newH, newW))
            label = TF.center_crop(label, (newH, newW))
            marker = TF.center_crop(marker, (newH, newW))
            weight_map = TF.center_crop(weight_map, (newH, newW))

            # Back to tensor
            image = TF.to_tensor(image)
            label = TF.to_tensor(label)
            marker = TF.to_tensor(marker)
            weight_map = TF.to_tensor(weight_map)

        label = torch.cat([label, marker], dim=0)
        return image.type(torch.FloatTensor), (label > 0).type(torch.FloatTensor), weight_map.type(torch.FloatTensor)

class CTC3DDatasetWithMarkersSW(Dataset):
    def __init__(self, df, augment=True, resolution=(512, 512),
                 imagenet_pretrained=True, resize_to=None, erosion=5):
        
        impaths = list(df.image)
        gtpaths = list(df.label)

        print('Loading images')
        if resize_to is None:
            ims = np.concatenate([normalize(sio.imread(p))
                            for p in tqdm(impaths)], axis=0)
        else:
            ims = np.concatenate([resize_all_slices(normalize(sio.imread(p)), resize_to)
                            for p in tqdm(impaths)], axis=0)
        print('Loading labels')

        if resize_to is None:
            gts = np.concatenate([sio.imread(p)
                        for p in tqdm(gtpaths)], axis=0)
        else:
            gts = np.concatenate([resize_all_slices(sio.imread(p), resize_to)
                        for p in tqdm(gtpaths)], axis=0)
        
        print('gts.max()', gts.max())
        mks = np.stack([get_markers(gt.astype('uint16'), erosion=erosion)
                            for gt in gts], axis=0)
        print('mks.max()', mks.max())

        sws = np.stack([get_weight_map(gt) for gt in gts], axis=0)

        # binarize ground truth for segmentation
        gts = gts > 0

        self.augment = augment
        self.resolution = resolution
        self.ims = ims.astype('float32')
        self.gts = gts.astype('uint8')
        self.mks = mks.astype('uint8')
        self.sws = sws.astype('float32')

    def __len__(self):
        return self.ims.shape[0]

    def __getitem__(self, idx):
        image = self.ims[idx]
        label = self.gts[idx]
        marker = self.mks[idx]
        weight_map = self.sws[idx]

        # Data augmentation
        # -----------------
        if self.augment:

            # ToPILImage
            image = TF.to_pil_image(image, mode='F')
            label = TF.to_pil_image(label)
            marker = TF.to_pil_image(marker)
            weight_map = TF.to_pil_image(weight_map, mode='F')

            # Random horizontal flipping
            if np.random.rand() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
                marker = TF.hflip(marker)
                weight_map = TF.hflip(weight_map)

            # Random vertical flipping
            if np.random.rand() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)
                marker = TF.vflip(marker)
                weight_map = TF.vflip(weight_map)

            # Random Rotation
            angle = np.random.randint(91)
            image = TF.rotate(image, angle)
            label = TF.rotate(label, angle)
            marker = TF.rotate(marker, angle)
            weight_map = TF.rotate(weight_map, angle)

            # Random crop to 512 x 512
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=self.resolution)
            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label, i, j, h, w)
            marker = TF.crop(marker, i, j, h, w)
            weight_map = TF.crop(weight_map, i, j, h, w)

            # Back to tensor
            image = TF.to_tensor(image)
            label = TF.to_tensor(label)
            marker = TF.to_tensor(marker)
            weight_map = TF.to_tensor(weight_map)

            # Add random noise
            if np.random.rand() > 0.5:
                noise = np.random.rand() * .2
                image = AddGaussianNoise(std=noise)(image)

        else:
            # ToPILImage
            H, W = image.squeeze().shape
            newH = (H // 32 + 1) * 32 if H % 32 != 0 else H
            newW = (W // 32 + 1) * 32 if W % 32 != 0 else W

            image = TF.to_pil_image(image, mode='F')
            label = TF.to_pil_image(label)
            marker = TF.to_pil_image(marker)
            weight_map = TF.to_pil_image(weight_map)

            # # Pad by center cropping with larger dims
            image = TF.center_crop(image, (newH, newW))
            label = TF.center_crop(label, (newH, newW))
            marker = TF.center_crop(marker, (newH, newW))
            weight_map = TF.center_crop(weight_map, (newH, newW))

            # Back to tensor
            image = TF.to_tensor(image)
            label = TF.to_tensor(label)
            marker = TF.to_tensor(marker)
            weight_map = TF.to_tensor(weight_map)

        label = torch.cat([label, marker], dim=0)
        return image.type(torch.FloatTensor), (label > 0).type(torch.FloatTensor), weight_map.type(torch.FloatTensor)

class CTC2DDatasetWithMarkersSWMR(Dataset):
    def __init__(self, df, augment=True, resolution=(512, 512),
                 imagenet_pretrained=True, erosion=20, resize_to=None):

        impaths = list(df.image)
        gtpaths = list(df.label)
        mrpaths = list(df.marker)

        if resize_to is not None:
            gts = np.stack([resize(sio.imread(p), resize_to, anti_aliasing=False)
                            for p in tqdm(gtpaths)], axis=0)
            ims = np.stack([resize(normalize(sio.imread(p)), resize_to, anti_aliasing=True)
                            for p in tqdm(impaths)], axis=0)
            mks = np.stack([resize(sio.imread(p), resize_to, anti_aliasing=False)
                            for p in tqdm(mrpaths)], axis=0)
        else:
            gts = np.stack([sio.imread(p)
                            for p in tqdm(gtpaths)], axis=0)
            ims = np.stack([normalize(sio.imread(p))
                            for p in tqdm(impaths)], axis=0)
            mks = np.stack([sio.imread(p)
                            for p in tqdm(mrpaths)], axis=0)

        sws = np.stack([get_weight_map(gt) for gt in gts], axis=0)

        # binarize ground truth for segmentation
        gts = gts > 0
         
        mks = mks > 0   
    
        self.augment = augment
        self.resolution = resolution
        self.ims = ims.astype('float32')
        self.gts = gts.astype('uint8')
        self.mks = mks.astype('uint8')
        self.sws = sws.astype('float32')

    def __len__(self):
        return self.ims.shape[0]

    def __getitem__(self, idx):
        image = self.ims[idx]
        label = self.gts[idx]
        marker = self.mks[idx]
        weight_map = self.sws[idx]

        # Data augmentation
        # -----------------
        if self.augment:

            # ToPILImage
            image = TF.to_pil_image(image, mode='F')
            label = TF.to_pil_image(label)
            marker = TF.to_pil_image(marker)
            weight_map = TF.to_pil_image(weight_map, mode='F')

            # Random horizontal flipping
            if np.random.rand() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
                marker = TF.hflip(marker)
                weight_map = TF.hflip(weight_map)

            # Random vertical flipping
            if np.random.rand() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)
                marker = TF.vflip(marker)
                weight_map = TF.vflip(weight_map)

            # Random Rotation
            angle = np.random.randint(91)
            image = TF.rotate(image, angle)
            label = TF.rotate(label, angle)
            marker = TF.rotate(marker, angle)
            weight_map = TF.rotate(weight_map, angle, fill=0.2)

            # Random crop to 512 x 512
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=self.resolution)
            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label, i, j, h, w)
            marker = TF.crop(marker, i, j, h, w)
            weight_map = TF.crop(weight_map, i, j, h, w)

            # Back to tensor
            image = TF.to_tensor(image)
            label = TF.to_tensor(label)
            marker = TF.to_tensor(marker)
            weight_map = TF.to_tensor(weight_map)

            # Add random noise
            if np.random.rand() > 0.5:
                noise = np.random.rand() * .2
                image = AddGaussianNoise(std=noise)(image)

        else:
            # ToPILImage
            H, W = image.squeeze().shape
            newH = (H // 32 + 1) * 32 if H % 32 != 0 else H
            newW = (W // 32 + 1) * 32 if W % 32 != 0 else W

            image = TF.to_pil_image(image, mode='F')
            label = TF.to_pil_image(label)
            marker = TF.to_pil_image(marker)
            weight_map = TF.to_pil_image(weight_map)

            # # Pad by center cropping with larger dims
            image = TF.center_crop(image, (newH, newW))
            label = TF.center_crop(label, (newH, newW))
            marker = TF.center_crop(marker, (newH, newW))
            weight_map = TF.center_crop(weight_map, (newH, newW))

            # Back to tensor
            image = TF.to_tensor(image)
            label = TF.to_tensor(label)
            marker = TF.to_tensor(marker)
            weight_map = TF.to_tensor(weight_map)

        label = torch.cat([label, marker], dim=0)
        return image.type(torch.FloatTensor), (label > 0).type(torch.FloatTensor), weight_map.type(torch.FloatTensor)

class CTC3DDatasetWithMarkersSWMR(Dataset):
    def __init__(self, df, augment=True, resolution=(512, 512),
                 imagenet_pretrained=True, resize_to=None, erosion=5):
        
        impaths = list(df.image)
        gtpaths = list(df.label)
        mrpaths = list(df.marker)

        if resize_to is not None:
            gts = np.concatenate([resize_all_slices(sio.imread(p), resize_to)
                        for p in tqdm(gtpaths)], axis=0)

            ims = np.concatenate([resize_all_slices(normalize(sio.imread(p)), resize_to)
                            for p in tqdm(impaths)], axis=0)

            mks = np.concatenate([resize_all_slices(sio.imread(p), resize_to)
                        for p in tqdm(mrpaths)], axis=0)
        else:
            gts = np.concatenate([sio.imread(p)
                        for p in tqdm(gtpaths)], axis=0)
            
            ims = np.concatenate([normalize(sio.imread(p))
                            for p in tqdm(impaths)], axis=0)

            mks = np.concatenate([sio.imread(p)
                        for p in tqdm(mrpaths)], axis=0)

    
        sws = np.stack([get_weight_map(gt) for gt in gts], axis=0)

        # binarize ground truth for segmentation
        gts = gts > 0

        mks = mks > 0   
    
        self.augment = augment
        self.resolution = resolution
        self.ims = ims.astype('float32')
        self.gts = gts.astype('uint8')
        self.mks = mks.astype('uint8')
        self.sws = sws.astype('float32')

    def __len__(self):
        return self.ims.shape[0]

    def __getitem__(self, idx):
        image = self.ims[idx]
        label = self.gts[idx]
        marker = self.mks[idx]
        weight_map = self.sws[idx]

        # Data augmentation
        # -----------------
        if self.augment:

            # ToPILImage
            image = TF.to_pil_image(image, mode='F')
            label = TF.to_pil_image(label)
            marker = TF.to_pil_image(marker)
            weight_map = TF.to_pil_image(weight_map, mode='F')

            # Random horizontal flipping
            if np.random.rand() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
                marker = TF.hflip(marker)
                weight_map = TF.hflip(weight_map)

            # Random vertical flipping
            if np.random.rand() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)
                marker = TF.vflip(marker)
                weight_map = TF.vflip(weight_map)

            # Random Rotation
            angle = np.random.randint(91)
            image = TF.rotate(image, angle)
            label = TF.rotate(label, angle)
            marker = TF.rotate(marker, angle)
            weight_map = TF.rotate(weight_map, angle)

            # Random crop to 512 x 512
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=self.resolution)
            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label, i, j, h, w)
            marker = TF.crop(marker, i, j, h, w)
            weight_map = TF.crop(weight_map, i, j, h, w)

            # Back to tensor
            image = TF.to_tensor(image)
            label = TF.to_tensor(label)
            marker = TF.to_tensor(marker)
            weight_map = TF.to_tensor(weight_map)

            # Add random noise
            if np.random.rand() > 0.5:
                noise = np.random.rand() * .2
                image = AddGaussianNoise(std=noise)(image)

        else:
            # ToPILImage
            H, W = image.squeeze().shape
            newH = (H // 32 + 1) * 32 if H % 32 != 0 else H
            newW = (W // 32 + 1) * 32 if W % 32 != 0 else W

            image = TF.to_pil_image(image, mode='F')
            label = TF.to_pil_image(label)
            marker = TF.to_pil_image(marker)
            weight_map = TF.to_pil_image(weight_map)

            # # Pad by center cropping with larger dims
            image = TF.center_crop(image, (newH, newW))
            label = TF.center_crop(label, (newH, newW))
            marker = TF.center_crop(marker, (newH, newW))
            weight_map = TF.center_crop(weight_map, (newH, newW))

            # Back to tensor
            image = TF.to_tensor(image)
            label = TF.to_tensor(label)
            marker = TF.to_tensor(marker)
            weight_map = TF.to_tensor(weight_map)

        label = torch.cat([label, marker], dim=0)
        return image.type(torch.FloatTensor), (label > 0).type(torch.FloatTensor), weight_map.type(torch.FloatTensor)
