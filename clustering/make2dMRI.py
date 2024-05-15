import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys

# To import from sibling directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from neuralnet.data.datasets import LoadDatasetswClusterID



# Function to make 2D pngs of selected slice of all 3D MRI scans in a directory in pkl format, and save them in an out directory.
def make2dMRI(in_dir, out_dir, gt_provided = True, slice_no = 64, contrast_no = 0):
    files_list = os.listdir(in_dir)
    dataset = LoadDatasetswClusterID(in_dir, gt_provided=gt_provided, partial_file_names= files_list) # Loads as (case_info, data, clusterID), where data contains (x1, x2, x3, x4, segmentation)
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