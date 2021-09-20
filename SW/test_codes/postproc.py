import cc3d
import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
from skimage import measure, morphology, segmentation


def remove_small_components(tensor, area_thresh=200):
    out = (tensor > 0.5).astype('uint8') * 255

    nc, cc, stats, *_ = cv2.connectedComponentsWithStats(out)
    for i, stat  in enumerate(stats):
        area = stat[-1]
        if area < area_thresh:
            cc[cc == i] = 0

    cc = (cc > 0).astype('float32')
    
    out = (cc < .5).astype('uint8') * 255
    nc, cc, stats, *_ = cv2.connectedComponentsWithStats(out)
    for i, stat  in enumerate(stats):
        area = stat[-1]
        if area < area_thresh:
            cc[cc == i] = 0

    cc = (cc < 0.5).astype('float32')
    return cc

def remove_small_components_label(label, area_thresh=200):
    label = cc3d.connected_components(label > 0.5, connectivity=6)

    for i in range(1, label.max() + 1):
        if np.sum(label == i) < area_thresh:
            label[label == i] = 0
    return label > 0


def postprocess(pred, area_thresh=200):
    pred = remove_small_components(pred)
    _, distance = morphology.medial_axis(pred > 0, return_distance=True)
    maxima = morphology.dilation(morphology.local_maxima(distance), np.ones((5, 5)))
    markers = measure.label(maxima)
    labels = segmentation.watershed(-distance, markers, mask=pred)
    for i in range(1, labels.max() + 1):
        if np.sum(labels == i) < area_thresh:
            labels[labels == i] = 0
    return labels

def postprocess_mask_and_markers(mask, markers, area_thresh=50):
    
    # Label markers if not labeled
    markers = measure.label(markers > 0.5)
    # Remove small components
    mask = remove_small_components(mask, area_thresh=area_thresh)

    # Calculate distance transform for watershed
    _, distance = morphology.medial_axis(mask > 0, return_distance=True)

    # Use watershed
    labels = segmentation.watershed(-distance, markers, mask=mask)

    # remove small components
    for i in range(1, labels.max() + 1):
        if np.sum(labels == i) < area_thresh:
            labels[labels == i] = 0
            
    return labels

def postprocess_mask_and_markers_3d(mask, markers, area_thresh=50):
    # Label markers if not labeled
    markers = ndi.median_filter(markers, footprint=np.ones((5, 1, 1)))
    markers = cc3d.connected_components(markers > 0.5, connectivity=6)
    
    # Remove small components
    mask = remove_small_components_label(mask, area_thresh=area_thresh)

    # Calculate distance transform for watershed
    distance = ndi.morphology.distance_transform_edt(mask > 0, 
                            return_distances=True, return_indices=False)

    # Use watershed
    distance = ((1 - distance) * 255).astype('uint8')
    labels = ndi.watershed_ift(distance, markers.astype('int')) * mask

    # remove small components
    for i in range(1, labels.max() + 1):
        if np.sum(labels == i) < area_thresh:
            labels[labels == i] = 0
            
    return labels
