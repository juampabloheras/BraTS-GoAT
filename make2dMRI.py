import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import numpy as np
from neuralnet.data import trans
from neuralnet.data.datasets import LoadDatasetswClusterID
from torchvision import transforms



# Function to make 2D pngs of selected slice of all 3D MRI scans in a directory in pkl format, and save them in an out directory.
def make2dMRI(in_dir, out_dir, gt_provided=True, slice_no=64, contrast_no=0):
    # Data transforms
    data_transforms = trans.Compose([
        trans.CenterCropBySize([128, 192, 128]),
        trans.NumpyType((np.float32, np.float32, np.float32, np.float32, np.float32)),
    ])

    # Load dataset and make dataloader
    dataset = LoadDatasetswClusterID(in_dir, data_transforms, {}, gt_provided=gt_provided, partial_file_names=False)
    dl = DataLoader(dataset, batch_size=10, num_workers=3)

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Iterate over DataLoader
    for batch_idx, (filename_ids, imgs, _) in enumerate(dl):
        # Iterate over each image in the batch
        for i in range(len(filename_ids)):
            filename_id = filename_ids[i]
            image = imgs[i][contrast_no]

            # Plotting
            plt.figure(figsize=(10, 10), dpi=300)
            plt.imshow(image.numpy()[0, :, :, slice_no], cmap='gray')
            plt.savefig(os.path.join(out_dir, f'{filename_id}.png'))
            plt.close()

            print(f"Filename ID: {filename_id}")
            print(f"Filename ID type: {type(filename_id)}")
            print(f"Plotted figure {filename_id}, saved in {os.path.join(out_dir, f'{filename_id}.png')}!")


if __name__ == '__main__':
    in_dir = '/gscratch/scrubbed/juampablo/BraTS-GoAT/DATA/training'
    out_dir = '/gscratch/scrubbed/juampablo/BraTS-GoAT/2DImages'
    make2dMRI(in_dir, out_dir)