import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import numpy as np
from neuralnet.data import trans
from neuralnet.data.datasets import LoadDatasetswClusterID
from torchvision import transforms



# Function to make 2D pngs of selected slice of all 3D MRI scans in a directory in pkl format, and save them in an out directory.
def make2dMRI(in_dir, out_dir, gt_provided = True, slice_no = 64, contrast_no = 0):
    # files_list = os.listdir(in_dir)
    # files_list = [os.path.join(in_dir, filename) for filename in files_list]

    data_transforms = transforms.Compose([trans.CenterCropBySize([128,192,128]), 
                                              trans.NumpyType((np.float32, np.float32,np.float32, np.float32,np.float32)),
                                              ])
    
    dataset = LoadDatasetswClusterID(in_dir, data_transforms, {} , gt_provided=gt_provided, partial_file_names= False) # Loads as (case_info, data, clusterID), where data contains (x1, x2, x3, x4, segmentation)
    dl = DataLoader(dataset, batch_size=1, num_workers=3)

    for filename_id, imgs, _ in dl:
        image = imgs[contrast_no]

        # Plotting
        plt.figure(figsize=(10, 10), dpi=300)
        plt.imshow(image.numpy()[0, 0, :, :,slice_no], cmap='gray')   
        plt.savefig(os.path.join(out_dir, f'{filename_id}.png'))
        plt.close()

        print(f"Plotted figure {filename_id}, saved in {os.path.join(out_dir, f'{filename_id}.png')}!")


if __name__ == '__main__':
    in_dir = '/gscratch/scrubbed/juampablo/BraTS-GoAT/DATA/training'
    out_dir = '/gscratch/scrubbed/juampablo/BraTS-GoAT/2DImages'
    make2dMRI(in_dir, out_dir)