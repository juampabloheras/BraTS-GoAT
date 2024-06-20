# File to store string to function dictionaries for arg parsing
# Update whenever new loss or model functions are needed

import torch.nn as nn
import losses.new_losses as lf # in same dir
from losses import EdgeLoss3D # in same directory
import torch
import numpy as np
import model
import  matplotlib.pyplot as plt
from monai.metrics import HausdorffDistanceMetric
import os
from skimage.metrics import structural_similarity as ssim
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


from losses import losses as lf2 # in same directory
from scipy.spatial.distance import cdist


LOSS_STR_TO_FUNC = {
    'mse': nn.MSELoss(),
    'cross-entropy': nn.CrossEntropyLoss(),
    # 'mask-regulizer': lf2.Maskregulizer(),
    # 'edge-loss': EdgeLoss3D.GMELoss3D(),
    # 'dice': lf.DiceLoss(), # not sure if this is correct
    # 'focal': lf.FocalLoss() # not sure if this is correct
}

MODEL_STR_TO_FUNC = {
        'unet3d': model.UNet3D,
} 


def split_seg_labels(seg):
    seg_squeezed = seg.squeeze(1)  # Squeezes dimension 1 if it's of size 1 (necessary for batch size > 1)
    seg3 = torch.zeros((seg.shape[0], 3, seg_squeezed.shape[1], seg_squeezed.shape[2], seg_squeezed.shape[3]), device=seg.device)
    
    # Split the segmentation labels into 3 channels
    seg3[:, 0, :, :, :] = torch.where(seg_squeezed == 1, 1., 0.)
    seg3[:, 1, :, :, :] = torch.where(seg_squeezed == 2, 1., 0.)
    seg3[:, 2, :, :, :] = torch.where(seg_squeezed == 3, 1., 0.)
    return seg3


## LAYER FREEZING
FREEZE_STR_TO_LAYERS = {
    'encoder': ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'Conv6', 'Conv7'],
    'decoder': ['Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4', 'Up3', 'Up_conv3', 'Conv_1x13', 'Up2', 'Up_conv2', 'Conv_1x12', 'Up1', 'Up_conv1', 'Conv_1x11'],
    'middle' : ['Conv5', 'Conv6', 'Conv7', 'Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4'],
    'none' : [],
    'deep_decoder': ['Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4'],
    'decoder_test2': ['Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4', 'Up3', 'Up_conv3', 'Conv_1x13'],
    'decoder_test3': ['Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4', 'Up3', 'Up_conv3', 'Conv_1x13', 'Up2', 'Up_conv2', 'Conv_1x12']
}

def freeze_layers(model, frozen_layers):

    for name, param in model.named_parameters():
        needs_freezing = False
        for layer in frozen_layers:
            if layer in name:
                needs_freezing = True
                break
        if needs_freezing:
            print(f'Freezing parameter {name}')
            param.requires_grad = False


def check_frozen(model, frozen_layers):

    for name, param in model.named_parameters():
        needs_freezing = False
        for layer in frozen_layers:
            if layer in name:
                needs_freezing = True
                break
        if needs_freezing:
            if param.requires_grad:
                print(f'Warning! Param {name} should not require grad but does')
                break
            else:
                print(f'Parameter {name} is frozen')
