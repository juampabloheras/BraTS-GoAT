# General packages
import argparse
import pickle
import os
import random
import numpy as np


# Import our modules
from train_utils import *
from utils import *
from data import trans
from data.datasets import LoadDatasets
from metrics import Dice, HD95

# PyTorch Imports
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor

# Pytorch Lightning, Trainer Imports
import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import TQDMProgressBar


# Weights and Biases logger
from lightning.pytorch.loggers import WandbLogger

# To "properly utilize" tensor cores
torch.set_float32_matmul_precision('medium')

# Define Lightning Training Module 
class LitUNet(L.LightningModule):
    def __init__(self, model, init_lr, train_on_overlap, eval_on_overlap, loss_functions, weights, power, max_epochs):
        super().__init__()
        self.model  = model 
        self.init_lr = init_lr
        self.train_on_overlap = train_on_overlap
        self.eval_on_overlap = eval_on_overlap
        self.loss_functions = loss_functions
        self.weights = weights
        self.power = power
        self.max_epochs = max_epochs
        self.save_hyperparameters()
        self.Dice = Dice() 
        self.HD95 = HD95() 


    def compute_loss(self, output, mask, loss_functs, loss_weights):
        """Computes weighted loss between model output and ground truth, summed across each region."""
        loss = 0.
        for n, loss_function in enumerate(loss_functs):      
            temp = 0
            for i in range(3):
                temp += loss_function(output[:,i:i+1], mask[:,i:i+1])

            loss += temp * loss_weights[n]
        return loss

    def compute_metric(self, output, mask, metric):
        D1 = metric(output[:, 0, :, :, :], mask[:, 0, :, :, :])
        D2 = metric(output[:, 1, :, :, :], mask[:, 1, :, :, :])
        D3 = metric(output[:, 2, :, :, :], mask[:, 2, :, :, :])
        D_list = [D1, D2, D3]
        D_list = [D for D in D_list if D is not float('NaN')]
        D_avg = sum(D_list) / len(D_list) if D_list else float('NaN')
        return D1, D2, D3, D_avg


    def training_step(self, batch, batch_idx): 
        
        # Set capturable = True to avoid error in outdated PyTorch
        optimizer = self.optimizers()
        optimizer.param_groups[0]['capturable'] = True

        # Unpack batch data
        subject_id, imgs = batch

        x1 = imgs[0]
        x2 = imgs[1]
        x3 = imgs[2]
        x4 = imgs[3]
        seg = imgs[4]

        seg3 = split_seg_labels(seg).to(self.device) # self.device is defined by PyTorch lightning during execution

        # Set the target either as overlapping or disjoint regions
        if self.train_on_overlap:
            # Combine the segmentation labels into partially overlapping regions
            mask = torch.zeros_like(seg3)
            mask[:,0] = seg3[:, 0] + seg3[:, 1] + seg3[:, 2] #WHOLE TUMOR
            mask[:,1] = seg3[:, 0] + seg3[:, 2] #TUMOR CORE
            mask[:,2] = seg3[:, 2] #ENHANCING TUMOR
            mask = mask.float()
        else:
            mask = seg3.float()

        x_in = torch.cat((x1, x2, x3, x4), dim=1)


        output = self.model(x_in) 
        output = output.float()

        segmentation_loss = self.compute_loss(output, mask, self.loss_functions, self.weights)

        loss = segmentation_loss # Loss used for backpropagation

        # Compute dice and HD95 scores
        Dice1, Dice2, Dice3, mean_Dice = self.compute_metric(output, mask, self.Dice)
        # HD1, HD2, HD3, mean_HD = self.compute_metric(output, mask, self.HD95)

        # Log losses to WandB
        self.log("seg_loss", segmentation_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("backprop_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        self.log("Dice1", Dice1, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Dice2", Dice2, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Dice3", Dice3, on_step=False, on_epoch=True, sync_dist=True)
        self.log("mean_Dice", mean_Dice, on_step=False, on_epoch=True, sync_dist=True) 

        # self.log("HD1", HD1, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("HD2", HD2, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("HD3", HD3, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("mean_HD", mean_HD, on_step=False, on_epoch=True, sync_dist=True) 
 
        return loss
    
    def validation_step(self, batch, batch_idx):

        subject_id, imgs = batch

        # Unpack the data
        x1 = imgs[0]
        x2 = imgs[1]
        x3 = imgs[2]
        x4 = imgs[3]
        seg = imgs[4]

        # print('Seg shape: ', np.shape(seg))
        seg3 = split_seg_labels(seg).to(self.device)

        # Set the target either as overlapping or disjoint regions
        if self.train_on_overlap:
            # Combine the segmentation labels into partially overlapping regions
            mask = torch.zeros_like(seg3)
            mask[:,0] = seg3[:, 0] + seg3[:, 1] + seg3[:, 2] #WHOLE TUMOR
            mask[:,1] = seg3[:, 0] + seg3[:, 2] #TUMOR CORE
            mask[:,2] = seg3[:, 2] #ENHANCING TUMOR
            mask = mask.float()
        else:
            mask = seg3.float()

        x_in = torch.cat((x1, x2, x3, x4), dim=1)

        output = self.model(x_in)
        output = output.float()

        segmentation_loss = self.compute_loss(output, mask, self.loss_functions, self.weights)

        loss = segmentation_loss 

        # Compute dice and HD95 scores
        Dice1, Dice2, Dice3, mean_Dice = self.compute_metric(output, mask, self.Dice)
        HD1, HD2, HD3, mean_HD = self.compute_metric(output, mask, self.HD95)

        # Log losses to WandB
        self.log("seg_loss_val", segmentation_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("backprop_loss_val", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('epoch_loss_val', loss, on_step=False, on_epoch=True, sync_dist=True) # Logs mean loss per epoch

        self.log("Dice1_val", Dice1, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Dice2_val", Dice2, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Dice3_val", Dice3, on_step=False, on_epoch=True, sync_dist=True)
        self.log("mean_Dice_val", mean_Dice, on_step=False, on_epoch=True, sync_dist=True) 

        self.log("HD1_val", HD1, on_step=False, on_epoch=True, sync_dist=True)
        self.log("HD2_val", HD2, on_step=False, on_epoch=True, sync_dist=True)
        self.log("HD3_val", HD3, on_step=False, on_epoch=True, sync_dist=True)
        self.log("mean_HD_val", mean_HD, on_step=False, on_epoch=True, sync_dist=True) 

    def lr_lambda(self, current_epoch):
        """Custom learning rate scheduler."""
        return np.power(1 - (current_epoch / self.max_epochs), self.power)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.init_lr, weight_decay = 0, amsgrad= True)
        lr_scheduler_config = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_lambda),
            'interval': 'epoch',  
            'frequency': 1,
            'name': 'Adam_lr'
        }
        return [optimizer], [lr_scheduler_config]



# Define Lightning Data Module    
class UNetDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 1, test_data_dir: str = "path/to/dir", folds_dir: str = "path/to/dir", fold_no: int = 0):
        super().__init__()
        self.data_dir = data_dir # because data_dir is currently setup as list, and i only want the first item
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.folds_dir = folds_dir
        self.fold_no = fold_no
        self.transforms = transforms.Compose([trans.CenterCropBySize([128,192,128]), 
                                              trans.NumpyType((np.float32, np.float32,np.float32, np.float32,np.float32)),
                                              ])


    def setup(self, stage: str):
        train_file_names, val_file_names = self.load_file_names(self.data_dir, self.folds_dir, self.fold_no)        
        if stage == 'fit':
            self.brats_train = LoadDatasets(self.data_dir, self.transforms, normalized=True, gt_provided=True, partial_file_names = train_file_names)
            self.brats_val = LoadDatasets(self.data_dir, self.transforms, normalized=True, gt_provided=True, partial_file_names = val_file_names)

        if stage == 'test':
            self.brats_test = LoadDatasets(self.test_data_dir, self.transforms, normalized=True, gt_provided= False, partial_file_names = False)

    def train_dataloader(self):
        print(f'Length of train dataset: {len(self.brats_train)}')
        return DataLoader(self.brats_train, batch_size=self.batch_size, num_workers=3)

    def val_dataloader(self):        
        print(f'Length of val dataset: {len(self.brats_val)}')
        return DataLoader(self.brats_val, batch_size=self.batch_size, num_workers=3) 

    def test_dataloader(self):
        print(f'Length of test dataset: {len(self.brats_test)}')
        return DataLoader(self.brats_test, batch_size=self.batch_size, num_workers=3) 

    @staticmethod
    def load_file_names(data_dir, folds_dir, fold_no):
        val_dir = os.path.join(folds_dir , sorted( os.listdir(folds_dir) )[fold_no])
        val_file_names = load_fold_file(val_dir)

        # List comprehension, indented for readability
        train_file_names = [
                file_name
                for directory in data_dir
                    for file_name in os.listdir(directory)
                        if file_name not in val_file_names
            ]         
        return train_file_names, val_file_names
   





if __name__ == '__main__':

    (train_dir, test_dir, ckpt_dir, out_dir, loss_str, weights, 
                model_str, folds_dir, fold_no, run_identifier, max_epochs, lr,
                                        power, eval_on_overlap, train_on_overlap) = parse_args() #definition in train_utils


    data_dir = train_dir
    test_data_dir = test_dir
    ckpt_dir = ckpt_dir
    out_dir = out_dir
    loss_functions = [LOSS_STR_TO_FUNC[l] for l in loss_str] # from utils
    weights = weights
    model_str = model_str
    folds_dir = folds_dir
    fold_no = fold_no
    run_identifier = run_identifier
    max_epochs = max_epochs
    init_lr = lr
    power =  power
    eval_on_overlap = eval_on_overlap
    train_on_overlap = train_on_overlap

    batch_size = 1  ######
    nf = 8 # model size factor (usually 32 when training full model)

    # Instantiate model with correct size, and number of input and output channels
    model_architecture = MODEL_STR_TO_FUNC[model_str](nf = nf, img_ch = 4, output_ch = 3) # from utils

    # Make checkpoint directory
    new_ckpt_dir = os.path.join(out_dir, 'new_checkpoints')
    if not os.path.exists(new_ckpt_dir):
        os.makedirs(new_ckpt_dir)
        os.system('chmod a+rwx ' + new_ckpt_dir)

    # Instantiate DataModule
    print(f'Loading Fold {fold_no}')
    dm = UNetDataModule(data_dir = data_dir, batch_size = batch_size, test_data_dir = test_data_dir, folds_dir = folds_dir, fold_no = fold_no)
    
    # Set seeds
    seed_everything(42, workers = True) # sets seeds for numpy, torch and python.random for reproducibility of results.

    # Define Lightning callbacks (https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(every_n_epochs = 2, 
                                          dirpath=new_ckpt_dir, 
                                          filename='train-unet-{epoch:02d}-{seg_loss:.4f}-fold{fold_no:02d}', # Note filename is NOT an fstring
                                            save_last = True, 
                                            monitor = 'seg_loss', 
                                            mode = 'min', 
                                            save_top_k = 5)  

    # Define Model
    model = LitUNet(model_architecture, init_lr, train_on_overlap, eval_on_overlap, loss_functions, weights, power, max_epochs)

    # Define WandB logger
    wandb_logger = logger_setup(project_name = "BraTS Practice Runs Juampablo", experiment_name = f"UNet-fold{fold_no}-{run_identifier}", out_dir = out_dir) # in train_utils

    TRAINER_KWARGS = {
    'max_epochs': max_epochs,
    'default_root_dir': out_dir,
    'strategy': 'auto',
    'callbacks': [lr_monitor, checkpoint_callback, TQDMProgressBar(refresh_rate=75)],
    'logger': wandb_logger,
    }

    # Define trainer
    trainer = Trainer(**TRAINER_KWARGS) 

    if os.path.exists(os.path.join(new_ckpt_dir,'last.ckpt')):
        print(f'Loading checkpoint from: {os.path.join(new_ckpt_dir,"last.ckpt")}. Training model from here.')
        trainer.fit(model, datamodule=dm, ckpt_path= os.path.join(new_ckpt_dir,'last.ckpt') )
    else:
        print("Starting training from the beginning.")
        trainer.fit(model, datamodule=dm)

    trainer.validate(datamodule=dm)