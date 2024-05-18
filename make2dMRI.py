from email.mime import image
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import numpy as np
from neuralnet.data import trans
from neuralnet.data.datasets import LoadDatasetswClusterID
from torchvision import transforms



# Function to make 2D pngs of selected slice of all 3D MRI scans in a directory in pkl format, and save them in an out directory.
def make2dMRI(in_dir, out_dir, contrasts_list=[0,1, 2], slice_no=63, gt_provided=True):
    # Data transforms
    data_transforms = trans.Compose([
        trans.CenterCropBySize([128, 192, 128]),
        trans.NumpyType((np.float32, np.float32, np.float32, np.float32, np.float32)),
    ])

    # Load dataset and make DataLoader
    dataset = LoadDatasetswClusterID(in_dir, data_transforms, {}, gt_provided=gt_provided, partial_file_names=False)
    dl = DataLoader(dataset, batch_size=3, num_workers=2)

    # Make sure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    compiled_X_list = []
    compiled_Y_list = []
    # Iterate over DataLoader
    for batch_idx, (filename_ids, imgs, _) in enumerate(dl):
        # Iterate over each image in the batch
        for i in range(len(filename_ids)):
            filename_id = filename_ids[i]

            # print(f'Type Imags: {type(imgs)}')
            # print(f'Shape Imgs: {np.shape(np.array(imgs))}')

            slice_list = []
            for contrast_no in contrasts_list: 
                slice = np.array(imgs)[contrast_no,i,0, slice_no, :, :]
                slice_list.append(slice)

            X = np.array(slice_list)
            Y = np.array(imgs)[4,i,0, slice_no, :, :]

            compiled_X_list.append(X)
            compiled_Y_list.append(Y)
    
    compiled_X_array = np.array(compiled_X_list)
    compiled_Y_array = np.array(compiled_Y_list)


    # Define image save path, check if image has already been made
    save_path_npz = os.path.join(out_dir, f'{filename_id}.npz')            

    np.savez(save_path_npz, x_train=compiled_X_array, y_train=compiled_Y_array)
    print(f'Saved {filename_id} in {save_path_npz}!')


if __name__ == '__main__':
    in_dir = '/gscratch/scrubbed/juampablo/BraTS-GoAT/DATA/training'
    out_dir = '/gscratch/scrubbed/juampablo/BraTS-GoAT/2DImagesNPZ2'
    make2dMRI(in_dir, out_dir)