import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


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
    
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    

# Domain classifier
class domain_classifier(nn.Module):
    def __init__(self, CH_IN, num_domains = 3):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv3d(CH_IN, 64, kernel_size=3)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=1, padding =1)
        
        # Pooling layer
        self.pool = nn.MaxPool3d(kernel_size = 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_domains)

    def forward(self, x):
        # Convolution and Pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Fully-connected layers
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        y = F.softmax(self.fc2(x), dim=1)  # Softmax for multiclass classification
        
        return y


# Encoder-Decoder architecture
class EncoderDecoder3D(nn.Module):
    def __init__(self, nf = 32, img_ch=4, output_ch=3):
        super(EncoderDecoder3D, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out1=nf*2, ch_out2=nf*2, k1=3, k2=3, s1=1, s2=1)
        self.Conv2 = conv_block(ch_in=nf*2, ch_out1=nf*3, ch_out2=nf*3, k1=3, k2=3, s1=2, s2=1)
        self.Conv3 = conv_block(ch_in=nf*3, ch_out1=nf*4, ch_out2=nf*4, k1=3, k2=3, s1=2, s2=1)
        self.Conv4 = conv_block(ch_in=nf*4, ch_out1=nf*6, ch_out2=nf*6, k1=3, k2=3, s1=2, s2=1)
        self.Conv5 = conv_block(ch_in=nf*6, ch_out1=nf*8, ch_out2=nf*8, k1=3, k2=3, s1=2, s2=1)
        self.Conv6 = conv_block(ch_in=nf*8, ch_out1=nf*12, ch_out2=nf*12, k1=3, k2=3, s1=2, s2=1)
        self.Conv7 = conv_block(ch_in=nf*12, ch_out1=nf*16, ch_out2=nf*16, k1=3, k2=3, s1=2, s2=1)

        # Decoding path (upsampling)
        self.Up6 = up_conv(ch_in=nf*16, ch_out=nf*12)
        self.Up5 = up_conv(ch_in=nf*12, ch_out=nf*8)
        self.Up4 = up_conv(ch_in=nf*8, ch_out=nf*6)
        self.Up3 = up_conv(ch_in=nf*6, ch_out=nf*4)
        self.Up2 = up_conv(ch_in=nf*4, ch_out=nf*3)
        self.Up1 = up_conv(ch_in=nf*3, ch_out=nf*2)

        self.FinalConv = nn.Conv3d(nf*2, output_ch, kernel_size=1, stride=1, padding=0)
        self.Sig = nn.Sigmoid()

    def forward(self, x):
        # Encoding path
        x1 = self.Conv1(x)
        x2 = self.Conv2(x1)
        x3 = self.Conv3(x2)
        x4 = self.Conv4(x3)
        x5 = self.Conv5(x4)
        x6 = self.Conv6(x5)
        x7 = self.Conv7(x6)

        d6 = self.Up6(x7)
        d5 = self.Up5(d6)
        d4 = self.Up4(d5)
        d3 = self.Up3(d4)
        d2 = self.Up2(d3)
        d1 = self.Up1(d2)
        out = self.FinalConv(d1)
        out = self.Sig(out)

        return [out, d6]

    def __str__(self):
        num_params = sum(p.numel() for p in self.parameters())
        return f"EncoderDecoder3D (with {num_params:,} parameters)"
    
class U_Net3D(nn.Module):
    def __init__(self,nf, img_ch=4,output_ch=3):
        super(U_Net3D,self).__init__()
        nf= 8
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

        return segmentation, latent, x1, x2, x3, x4, x5, x6  # .to(device='cuda:1')



class DANNEncoderDecoder3D(nn.Module):
    def __init__(self, img_ch= 4, output_ch = 3, num_domains = 3):
        super().__init__()

        nf = 8 # Model size factor

        self.autoencoder = EncoderDecoder3D(nf, img_ch=img_ch, output_ch = output_ch)
        self.classifier = domain_classifier(CH_IN=nf*12, num_domains = num_domains)

    def forward(self, input, alpha = 1): # Default alpha value of 1, if None, will not implement gradient reversal
        
        segmentation, latent = self.autoencoder(input)

        if alpha is not None: 
            reversed_input = GradReverse.apply(latent, alpha) 
            classification = self.classifier(reversed_input)
                
        else:
            # If alpha is not provided, perform regular classification
            classification = self.classifier(latent)
            
        return [segmentation, classification, latent]
    


class DANNUNet3DExtendedLatent(nn.Module):
    def __init__(self, img_ch = 4, output_ch = 3, num_domains = 3):
        super().__init__()
        nf = 8

        self.UNet3D = U_Net3D(nf, img_ch=img_ch, output_ch= output_ch)
        self.classifier = domain_classifier(CH_IN=nf*47, num_domains = num_domains) # nf*47 is sum of channels in extended latent

    def forward(self, input, alpha = 1):
        segmentation, latent, x1, x2, x3, x4, x5, x6 = self.UNet3D(input)

        # Flatten each tensor
        latent_flat = latent.view(latent.size(0), -1)
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)
        x3_flat = x3.view(x3.size(0), -1)
        x4_flat = x4.view(x4.size(0), -1)
        x5_flat = x5.view(x5.size(0), -1)
        x6_flat = x6.view(x6.size(0), -1)

        # Concatenate the flattened tensors
        extended_latent = torch.cat([latent_flat, x1_flat, x2_flat, x3_flat, x4_flat, x5_flat, x6_flat], dim=1)


        if alpha is not None:
            reversed_latent = GradReverse.apply(extended_latent, alpha) # reverse gradient at the bottleneck
            classification = self.classifier(reversed_latent) # classify using the latent + x1, x2, ... , x6

        else:
            # If alpha is not provided, perform regular classification
            classification = self.classifier(extended_latent)
        
        return [segmentation, classification, extended_latent]



class DANNUNet3D(nn.Module):
    def __init__(self, img_ch = 4, output_ch = 3, num_domains = 3):
        super().__init__()
        nf = 8

        self.UNet3D = U_Net3D(nf, img_ch=img_ch, output_ch= output_ch)
        self.classifier = domain_classifier(CH_IN=nf*12, num_domains = num_domains)

    def forward(self, input, alpha = 1):
        segmentation, latent, x1, x2, x3, x4, x5, x6 = self.UNet3D(input)

        if alpha is not None: 
            reversed_input = GradReverse.apply(latent, alpha) 
            classification = self.classifier(reversed_input)
                
        else:
            # If alpha is not provided, perform regular classification
            classification = self.classifier(latent)
            
        return [segmentation, classification, latent]
