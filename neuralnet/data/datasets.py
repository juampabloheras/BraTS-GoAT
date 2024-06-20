import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt

import numpy as np
import cv2 as cv

from . import trans
from torchvision import transforms


class LoadDatasets(Dataset):

    def __init__(self, data_path, transforms, normalized=True, gt_provided=True, partial_file_names = True):
        self.transforms = transforms
        self.normalized = normalized
        self.gt_provided = gt_provided
        if partial_file_names == False:
            self.paths =  [os.path.join(data_path, filename) for filename in os.listdir(data_path) if filename.endswith('.pkl')]
        else:
            self.paths = self._get_matching_files(data_path, partial_file_names)

    # def _get_matching_files(self, data_path, partial_file_names):
    #     matching_files = []
    #     # print("length of partial names", len(partial_file_names))
    #     self.count = 0

    #     file_path = data_path
    #     temp = []
    #     if isinstance(data_path,list):
    #         file_path = data_path[0] ##### 
    #         for path in data_path:
    #             temp += os.listdir(path)
    #         data_path = temp

    #     print(f"Number of subjects loaded: {len(data_path)}")
        
    #     for filename in data_path:
    #         if any (partial_name in filename for partial_name in partial_file_names) and filename not in matching_files:
    #             matching_files.append(file_path + '/' + filename) 
    #             self.count += 1

    #     return matching_files
    

    def _get_matching_files(self, data_path, partial_file_names):
        matching_files = []
        # print("length of partial names", len(partial_file_names))
        self.count = 0

        file_path = data_path

        ############ THIS NEEDS TO BE IMPROVED ##############
        ### code uses this case
        temp = []
        if isinstance(data_path,list):
            file_path = data_path[0] ##### 
            for path in data_path:
                temp += os.listdir(path)
            data_path = temp

        if isinstance(data_path,str): # FOR make2dMRI.py
            data_path = os.listdir(data_path)

        if not isinstance(data_path, (list, tuple)):
            data_path = [data_path]

        ########################################################
        
        print(f"Number of subjects loaded: {len(data_path)}")

        # print('data_path', data_path)
        for filename in data_path:
            if any (partial_name in filename for partial_name in partial_file_names) and filename not in matching_files:
                matching_files.append(file_path + '/' + filename) 
                self.count += 1

        return matching_files

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        if self.gt_provided:
            x1, x2, x3, x4, y1 = pkload(path)
        else:
            x1, x2, x3, x4 = pkload(path)

        x1, x2, x3, x4 = x1[None, ...], x2[None, ...],x3[None, ...],x4[None, ...]
        if self.gt_provided:
            y1 = y1[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)

        if self.gt_provided:
            x1, x2, x3, x4, y1 = self.transforms([x1, x2, x3, x4, y1])
        else:
            x1, x2, x3, x4 = self.transforms([x1, x2, x3, x4])

        if self.normalized:

            norm_tf = transforms.Compose([trans.Normalize0_1()])
            x1, x2, x3, x4 = norm_tf([x1, x2, x3, x4])


        x1 = np.ascontiguousarray(x1)# [Bsize,channelsHeight,,Width,Depth]
        x2 = np.ascontiguousarray(x2)
        x3 = np.ascontiguousarray(x3)
        x4 = np.ascontiguousarray(x4)
        if self.gt_provided:
            y1 = np.ascontiguousarray(y1)


        x1, x2, x3, x4 = torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4)
        if self.gt_provided:
            y1 = torch.from_numpy(y1)

        # Get case_id from filename
        filename = path.split('/')[-1]
        case_info = str(filename.split('.')[0]) # e.g. from BraTS-GoAT-00000.pkl will produce 'BraTS-GoAT-00000'

        if self.gt_provided:
            data = x1, x2, x3, x4, y1
        else:
            data = x1, x2, x3, x4

        return case_info, data

    def __len__(self):
        return len(self.paths)


class LoadDatasetswDomain(Dataset):
    def __init__(self, data_path, transforms, normalized=True, gt_provided=True, partial_file_names = False):
        self.transforms = transforms
        self.normalized = normalized
        self.gt_provided = gt_provided
        if partial_file_names != False:
            self.paths = self._get_matching_files(data_path, partial_file_names)
        else:
            self.paths = data_path
       #  print(self.paths)

    def _get_matching_files(self, data_path, partial_file_names):
        matching_files = []
        # print("length of partial names", len(partial_file_names))
        self.count = 0

        file_path = data_path

        if isinstance(data_path,str):
            data_path = os.listdir(data_path)
        if not isinstance(data_path, (list, tuple)):
            data_path = [data_path]
        
        # print('data_path', data_path)
        for filename in data_path:
            if any (partial_name in filename for partial_name in partial_file_names) and filename not in matching_files:
                matching_files.append(file_path + '/' + filename) 
                self.count += 1
        print(self.count)
        return matching_files

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]

        # Classification
        if '-GLI-' in path:
            classification = 0
        elif '-SSA-' in path:
            classification = 1
        else:
            print('Image not classified... path = ', path)
    

        if self.gt_provided:
            try:
                x1, x2, x3, x4, y1 = pkload(path)
            except ValueError as e:
                loaded_values = pkload(path)
                print(f"Error loading file {path}: {e}. Loaded values: {loaded_values}")
           #  print("-----Done-----")
        else:
            x1, x2, x3, x4 = pkload(path)

        x1, x2, x3, x4 = x1[None, ...], x2[None, ...],x3[None, ...],x4[None, ...]
        if self.gt_provided:
            y1 = y1[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)

        if self.gt_provided:
            x1, x2, x3, x4, y1 = self.transforms([x1, x2, x3, x4, y1])
        else:
            x1, x2, x3, x4 = self.transforms([x1, x2, x3, x4])

        if self.normalized:

            norm_tf = transforms.Compose([trans.Normalize0_1()])
            x1, x2, x3, x4 = norm_tf([x1, x2, x3, x4])


        x1 = np.ascontiguousarray(x1)# [Bsize,channelsHeight,,Width,Depth]
        x2 = np.ascontiguousarray(x2)
        x3 = np.ascontiguousarray(x3)
        x4 = np.ascontiguousarray(x4)
        if self.gt_provided:
            y1 = np.ascontiguousarray(y1)


        x1, x2, x3, x4 = torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4)
        if self.gt_provided:
            y1 = torch.from_numpy(y1)

        # Get case_id from filename - ADDED by Ethan 17 July 2023
        filename = path.split('/')[-1]
        case_info = tuple(filename.split('.')[0].split('-')[2:4]) #(case_id, timepoint)

        if self.gt_provided:
            data = x1, x2, x3, x4, y1
        else:
            data = x1, x2, x3, x4

        return case_info, data, classification

    def __len__(self):
        return len(self.paths)
        