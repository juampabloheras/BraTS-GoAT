import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# Imports to test models
from torch.utils.data import DataLoader
from data.datasets import LoadDatasetswClusterID, LoadDatasets
from utils import *
import random
from torchvision import transforms
from data import trans

from metrics import Dice, HD95

import surface_distance



# Component blocks
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out1,ch_out2,k1,k2,s1,s2):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.InstanceNorm3d(ch_in),
            nn.Conv3d(ch_in, ch_out1, kernel_size=k1,stride=s1,padding=1,bias=True),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(ch_out1),
            nn.Conv3d(ch_out1, ch_out2, kernel_size=k2,stride=s2,padding=1,bias=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
      
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(ch_in, ch_out,kernel_size=2,stride=2,padding=1,bias=True,output_padding=1,dilation=2),
            nn.InstanceNorm3d(ch_in),
 	        nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x) 
        return x
    
    
    
class U_Net3D(nn.Module):
    def __init__(self,nf = 8, img_ch=4,output_ch=3):
        super(U_Net3D,self).__init__()
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2) # .to(device='cuda:1')
        self.Conv1 = conv_block(ch_in=img_ch,ch_out1=nf*2,ch_out2=nf*2,k1=3,k2=3,s1=1,s2=1) # .to(device='cuda:1')
        self.Conv2 = conv_block(ch_in=nf*2,ch_out1=nf*3,ch_out2=nf*3,k1=3,k2=3,s1=2,s2=1) # .to(device='cuda:1')
        self.Conv3 = conv_block(ch_in=nf*3,ch_out1=nf*4,ch_out2=nf*4,k1=3,k2=3,s1=2,s2=1) #  # .to(device='cuda:1')
        self.Conv4 = conv_block(ch_in=nf*4,ch_out1=nf*6,ch_out2=nf*6,k1=3,k2=3,s1=2,s2=1) # .to(device='cuda:1')
        self.Conv5 = conv_block(ch_in=nf*6,ch_out1=nf*8,ch_out2=nf*8,k1=3,k2=3,s1=2,s2=1) # .to(device='cuda:1')
        self.Conv6 = conv_block(ch_in=nf*8,ch_out1=nf*12,ch_out2=nf*12,k1=3,k2=3,s1=2,s2=1) # .to(device='cuda:1')
        self.Conv7 = conv_block(ch_in=nf*12,ch_out1=nf*16,ch_out2=nf*16,k1=3,k2=3,s1=2,s2=1) # .to(device='cuda:1')

        self.Up6 = up_conv(ch_in=nf*16,ch_out=nf*12) #.to(device='cuda:0')
        self.Up_conv6 = conv_block(ch_in=nf*24, ch_out1=nf*12, ch_out2=nf*12,k1=3,k2=3,s1=1,s2=1) # .to(device='cuda:0')
        
        self.Up5 = up_conv(ch_in=nf*12,ch_out=nf*8) # .to(device='cuda:0')
        self.Up_conv5 = conv_block(ch_in=nf*16, ch_out1=nf*8, ch_out2=nf*8,k1=3,k2=3,s1=1,s2=1) # .to(device='cuda:0')

        self.Up4 = up_conv(ch_in=nf*8,ch_out=nf*6) # .to(device='cuda:0')
        self.Up_conv4 = conv_block(ch_in=nf*12, ch_out1=nf*6, ch_out2=nf*6,k1=3,k2=3,s1=1,s2=1) # .to(device='cuda:0')
        
        self.Up3 = up_conv(ch_in=nf*6,ch_out=nf*4) # .to(device='cuda:0')
        self.Up_conv3 = conv_block(ch_in=nf*8, ch_out1=nf*4,ch_out2=nf*4,k1=3,k2=3,s1=1,s2=1) # .to(device='cuda:0')
        self.Conv_1x13 = nn.Conv3d(nf*4,output_ch,kernel_size=1,stride=1,padding=0) # .to(device='cuda:0')
        
        self.Up2 = up_conv(ch_in=output_ch,ch_out=nf*3) # .to(device='cuda:0')
        self.Up_conv2 = conv_block(ch_in=nf*6 , ch_out1=nf*3,ch_out2=nf*3,k1=3,k2=3,s1=1,s2=1) # .to(device='cuda:0')
        self.Conv_1x12 = nn.Conv3d(nf*3,output_ch,kernel_size=1,stride=1,padding=0) # .to(device='cuda:0')
        
        self.Up1 = up_conv(ch_in=output_ch,ch_out=nf*2) # .to(device='cuda:0')
        self.Up_conv1 = conv_block(ch_in=nf*4, ch_out1=nf*2,ch_out2=nf*2,k1=3,k2=3,s1=1,s2=1) # .to(device='cuda:0')
        self.Conv_1x11 = nn.Conv3d(nf*2,output_ch,kernel_size=1,stride=1,padding=0) # .to(device='cuda:0')
        
        self.Sig =nn.Sigmoid() # .to(device='cuda:0')

    def forward(self,x):
        # encoding path
        x = x # .to(device='cuda:1')
        x1 = self.Conv1(x)

        x2 = self.Conv2(x1)       
        
        x3 = self.Conv3(x2)
    
        x4 = self.Conv4(x3)       
    
        x5 = self.Conv5(x4)       
        
        x6 = self.Conv6(x5)

        x7 = self.Conv7(x6)
        

        # decoding + concat path
    
        d6 = self.Up6(x7) # d6 = self.Up6(x7.to(device='cuda:0'))

        latent = d6

        d6 = torch.cat((x6,d6),dim=1) #d6 = torch.cat((x6.to(device='cuda:0'),d6),dim=1)
        d6 = self.Up_conv6(d6)

        d5 = self.Up5(d6)
        d5 = torch.cat((x5,d5),dim=1) #d5 = torch.cat((x5.to(device='cuda:0'),d5),dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x4,d4),dim=1) # d4 = torch.cat((x4.to(device='cuda:0'),d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x3,d3),dim=1)  #d3 = torch.cat((x3.to(device='cuda:0'),d3),dim=1) 
        d3 = self.Up_conv3(d3)
        d3 = self.Conv_1x13(d3)
        d3 = self.Sig(d3)
        
        d2 = self.Up2(d3)
        d2 = torch.cat((x2,d2),dim=1)   # d2 = torch.cat((x2.to(device='cuda:0'),d2),dim=1)
        d2 = self.Up_conv2(d2)
        d2 = self.Conv_1x12(d2)
        d2 = self.Sig(d2)
        
        d1 = self.Up1(d2)
        d1 = torch.cat((x1,d1),dim=1)  #d1 = torch.cat((x1.to(device='cuda:0'),d1),dim=1)
        d1 = self.Up_conv1(d1)
        d1 = self.Conv_1x11(d1)
        d1 = self.Sig(d1)

        segmentation = d1

        return segmentation, latent  # .to(device='cuda:1')




class UNet3D(nn.Module):
    def __init__(self, nf = 8, img_ch = 4, output_ch = 3):
        super().__init__()
        self.UNet3D = U_Net3D(nf = nf, img_ch=img_ch, output_ch= output_ch)
    def forward(self, input):
        segmentation, latent = self.UNet3D(input)
        return segmentation
    

def testModel(model, data_dir, num_samples, train_on_overlap = True, num_workers = 3):
    
    '''
    Code to debug models.
    '''
    
    # Configuration settings
    data_dir = data_dir
    batch_size = 1

    num_workers = num_workers
    data_transforms = transforms.Compose([trans.CenterCropBySize([128,192,128]), 
                                              trans.NumpyType((np.float32, np.float32,np.float32, np.float32,np.float32)),
                                              ])
    train_file_names = random.sample(os.listdir(data_dir), num_samples)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_on_overlap = train_on_overlap  

    # Load dataset
    dataset = LoadDatasets(data_dir, data_transforms,
                                         normalized=True, gt_provided=True, 
                                         partial_file_names=train_file_names)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    for batch in dataloader:
        subject_id, imgs = batch

        seg = imgs[4]
        seg3 = split_seg_labels(seg).to(device)
        mask = torch.zeros_like(seg3) if train_on_overlap else seg3.float()
        if train_on_overlap:
            mask[:, 0] = seg3[:, 0] + seg3[:, 1] + seg3[:, 2]
            mask[:, 1] = seg3[:, 0] + seg3[:, 2]
            mask[:, 2] = seg3[:, 2]
            mask = mask.float()

        x_in = torch.cat(imgs[:4], dim=1).to(device)
        
        # Run model
        model = model.to(device)  # Ensure the model is on the correct device
        output = model(x_in)

        # Print outputs for verification
        print('Subject ID: ', subject_id)
        print("Input Shape: ", np.shape(x_in))
        print("Output shape: ", np.shape(output[0] if isinstance(output, list) and output else output))
        print("Segmentation shape: ", np.shape(mask))

        D1, D2, D3, D_avg = compute_metric(output, mask, Dice())
        HD1, HD2, HD3, HD_avg = compute_metric(output, mask, HD95())
        print(f"Dice Scores: D1={D1}, D2={D2}, D3={D3}, D_avg={D_avg}")
        print(f"Hausdorff Distances (95%): HD1={HD1}, HD2={HD2}, HD3={HD3}, HD_avg={HD_avg}")

    return






# Test metrics
def compute_metric(output, mask, metric):
    D1 = metric(output[:, 0, :, :, :], mask[:, 0, :, :, :])
    D2 = metric(output[:, 1, :, :, :], mask[:, 1, :, :, :])
    D3 = metric(output[:, 2, :, :, :], mask[:, 2, :, :, :])
    D_list = [D1, D2, D3]
    D_list = [D for D in D_list if D is not  float('NaN')]
    D_avg = sum(D_list) / len(D_list) if D_list else float('NaN')
    return D1, D2, D3, D_avg


if __name__ == '__main__':
    model = UNet3D(nf = 8)
    data_dir = '/gscratch/scrubbed/juampablo/BraTS-GoAT/DATA/training'
    num_samples = 4
    testModel(model, data_dir, num_samples)






