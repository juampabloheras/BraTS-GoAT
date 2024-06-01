import argparse
import torch
from utils import *
from data import trans
from data.datasets import LoadDatasetswClusterID

from torchvision import transforms
from train_utils import *
from torch.utils.data import DataLoader

import pickle

import os
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import lightning as L
from lightning.pytorch.cli import LightningCLI

# Trainer Imports
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import TQDMProgressBar


# Weights and Biases logger
from lightning.pytorch.loggers import WandbLogger
import random

# To "properly utilize tensor cores"
torch.set_float32_matmul_precision('medium')



# Lightning Module 
class LitGoAT(L.LightningModule):
    def __init__(self, model, alpha, init_lr, train_on_overlap, eval_on_overlap, loss_functions, loss_weights, weights, power, max_epochs):
        super().__init__()
        self.model  = model 
        self.init_lr = init_lr
        self.train_on_overlap = train_on_overlap
        self.eval_on_overlap = eval_on_overlap
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights
        self.domain_criterion = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.weights = weights
        self.power = power
        self.max_epochs = max_epochs
        self.save_hyperparameters()
        # self.automatic_optimization = False 

        # Initialize dicts for logging
        self.seg_loss_train_dict = {}
        self.class_loss_train_dict = {}
        self.seg_loss_val_dict = {}
        self.class_loss_val_dict = {}

        self.dice_loss = LOSS_STR_TO_FUNC['dice'] # from utils


    # @staticmethod
    def compute_loss(self, output, mask, loss_functs, loss_weights):
        """Computes weighted loss between model output and ground truth, summed across each region."""
        loss = 0.
        for n, loss_function in enumerate(loss_functs):      
            temp = 0
            for i in range(3):
                temp += loss_function(output[:,i:i+1], mask[:,i:i+1])

            loss += temp * loss_weights[n]
        return loss
    
    def forward(self, x): # Forward pass used for inference, not training
        x = self.model(x, self.alpha)
        return x

    def training_step(self, batch, batch_idx): 
        
        # Set capturable = True to circumvent error
        optimizer = self.optimizers()
        optimizer.param_groups[0]['capturable'] = True

        subject_id, imgs, true_classification = batch

        # Unpack the data
        x1 = imgs[0]
        x2 = imgs[1]
        x3 = imgs[2]
        x4 = imgs[3]
        seg = imgs[4]

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


        output, pred_classification, latent = self.model(x_in, self.alpha) # equivalent to self.model(x_in, self.alpha) and self.forward(x_in)
        output = output.float()

        segmentation_loss = self.compute_loss(output, mask, self.loss_functions, self.weights)
        classifier_loss = self.domain_criterion(pred_classification, true_classification)

        loss = self.loss_weights[0]*segmentation_loss + self.loss_weights[1]*classifier_loss

        dice_loss = self.compute_loss(output, mask, [self.dice_loss], [1])

        # Log losses to TensorBoard (changing to WandB soon..)
        self.log("seg_loss", segmentation_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("classif_loss", classifier_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("backprop_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('dice_loss', dice_loss, on_step=False, on_epoch=True, sync_dist=True) # Logs mean loss per epoch
        
        if true_classification.nelement() == 1:
            cluster_id = true_classification.item()
            self.seg_loss_train_dict.setdefault(cluster_id, []).append(segmentation_loss.detach().cpu())
            self.class_loss_train_dict.setdefault(cluster_id, []).append(classifier_loss.detach().cpu())
        else:
            for classification in true_classification:
                cluster_id = classification.item()
                self.seg_loss_train_dict.setdefault(cluster_id, []).append(segmentation_loss.detach().cpu())
                self.class_loss_train_dict.setdefault(cluster_id, []).append(classifier_loss.detach().cpu())

        return loss
    
    def validation_step(self, batch, batch_idx):

        subject_id, imgs, true_classification = batch

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

        output, pred_classification, latent = self.model(x_in, self.alpha) # equivalent to self.model(x_in, self.alpha) and self.forward(x_in)
        output = output.float()

        # print('Output shape: ', np.shape(output))
        # print('Mask shape: ', np.shape(mask))

        segmentation_loss = self.compute_loss(output, mask, self.loss_functions, self.weights)
        classifier_loss = self.domain_criterion(pred_classification, true_classification)

        loss = self.loss_weights[0]*segmentation_loss + self.loss_weights[1]*classifier_loss

        # Log losses to TensorBoard (changing to WandB soon..)
        self.log("seg_loss_val", segmentation_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("classif_loss_val", classifier_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("backprop_loss_val", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('epoch_loss_val', loss, on_step=False, on_epoch=True, sync_dist=True) # Logs mean loss per epoch

        if true_classification.nelement() == 1:
            cluster_id = true_classification.item()
            self.seg_loss_val_dict.setdefault(cluster_id, []).append(segmentation_loss.detach().cpu())
            self.class_loss_val_dict.setdefault(cluster_id, []).append(classifier_loss.detach().cpu())
        else:
            for classification in true_classification:
                cluster_id = classification.item()
                self.seg_loss_val_dict.setdefault(cluster_id, []).append(segmentation_loss.detach().cpu())
                self.class_loss_val_dict.setdefault(cluster_id, []).append(classifier_loss.detach().cpu())


    def on_train_epoch_end(self):
        assert set(self.seg_loss_val_dict.keys()) == set(self.class_loss_val_dict.keys()), "Clusters in seg loss and class loss dictionaries are different."
        for clusterID in self.seg_loss_val_dict.keys():
            self.log(f'cluster{clusterID}_seg_loss_val', np.mean(self.seg_loss_val_dict[clusterID]), sync_dist = True)
            self.log(f'cluster{clusterID}_classif_loss_val', np.mean(self.class_loss_val_dict[clusterID]), sync_dist = True)

        assert set(self.seg_loss_train_dict.keys()) == set(self.class_loss_train_dict.keys()), "Clusters in seg loss and class loss dictionaries are different."
        for clusterID in self.seg_loss_train_dict.keys():
            self.log(f'cluster{clusterID}_seg_loss_train', np.mean(self.seg_loss_train_dict[clusterID]), sync_dist = True)
            self.log(f'cluster{clusterID}_classif_loss_train', np.mean(self.class_loss_train_dict[clusterID]), sync_dist = True)

        # Reset after every epoch
        self.seg_loss_train_dict = {}
        self.class_loss_train_dict = {}
        self.seg_loss_val_dict = {}
        self.class_loss_val_dict = {}
             
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.init_lr, weight_decay = 0, amsgrad= True)
        lr_scheduler_config = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_lambda),
            'interval': 'epoch',  
            'frequency': 1,
            'name': 'Adam_lr'
        }
        return [optimizer], [lr_scheduler_config]

    def lr_lambda(self, current_epoch):
        """Custom learning rate scheduler."""
        return np.power(1 - (current_epoch / self.max_epochs), self.power)


# Lightning Data Module    
class BraTSDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 1, test_data_dir: str = "path/to/dir", folds_dir: str = "path/to/dir", fold_no: int = 0, cluster_mapping: dict = {}):
        super().__init__()
        self.data_dir = data_dir # because data_dir is currently setup as list, and i only want the first item
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.folds_dir = folds_dir
        self.fold_no = fold_no
        self.cluster_mapping = cluster_mapping
        self.transforms = transforms.Compose([trans.CenterCropBySize([128,192,128]), 
                                              trans.NumpyType((np.float32, np.float32,np.float32, np.float32,np.float32)),
                                              ])


    def setup(self, stage: str):
        train_file_names, val_file_names = self.load_file_names(self.data_dir, self.folds_dir, self.fold_no)        
        if stage == 'fit':
            self.brats_train = LoadDatasetswClusterID(self.data_dir, self.transforms, self.cluster_mapping,  normalized=True, gt_provided=True, partial_file_names = train_file_names)
            self.brats_val = LoadDatasetswClusterID(self.data_dir, self.transforms, self.cluster_mapping,  normalized=True, gt_provided=True, partial_file_names = val_file_names)

        if stage == 'test':
            self.brats_test = LoadDatasetswClusterID(self.test_data_dir, self.transforms, self.cluster_mapping,  normalized=True, gt_provided= False, partial_file_names = False)

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

    (alpha, train_dir, test_dir, ckpt_dir, out_dir, loss_str, weights, loss_weights, 
            model_str, partial_file_names, folds_dir, fold_no, cluster_dict, run_identifier, max_epochs, 
                                lr, power, eval_on_overlap, train_on_overlap) = parse_args() #definition in train_utils


    alpha = alpha
    init_lr = lr
    train_on_overlap = train_on_overlap
    eval_on_overlap = eval_on_overlap
    loss_functions = [LOSS_STR_TO_FUNC[l] for l in loss_str]
    loss_weights = loss_weights
    power =  power
    max_epochs = max_epochs
    cluster_dict_path = cluster_dict

    data_dir = train_dir
    batch_size = 1  ######
    test_data_dir = test_dir
    folds_dir = folds_dir
    fold_no = fold_no

    # Load cluster mapping
    with open(cluster_dict_path, 'rb') as file:
        cluster_mapping = pickle.load(file)
    num_clusters = len(cluster_mapping)

    # Instantiate model with correct number of clusters
    model_architecture = MODEL_STR_TO_FUNC[model_str](img_ch = 4, output_ch = 3, num_domains = num_clusters)

    # Make checkpoint directory
    new_ckpt_dir = os.path.join(out_dir, 'new_checkpoints')
    if not os.path.exists(new_ckpt_dir):
        os.makedirs(new_ckpt_dir)
        os.system('chmod a+rwx ' + new_ckpt_dir)

    # Instantiate DataModule
    dm = BraTSDataModule(data_dir = data_dir, batch_size = batch_size, test_data_dir = test_data_dir, folds_dir = folds_dir, fold_no = fold_no, cluster_mapping=cluster_mapping)
    
    # Set seeds
    seed_everything(42, workers = True) # sets seeds for numpy, torch and python.random.

    # Define callbacks
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(every_n_epochs = 2, dirpath=new_ckpt_dir, filename='train-GoAT-{epoch:02d}-{seg_loss:.4f}-fold{fold_no:02d}', save_last = True, monitor = 'seg_loss', mode = 'min', save_top_k = 5)  # Note filename is NOT an fstring

    # Define Model
    model = LitGoAT(model_architecture, alpha, init_lr, train_on_overlap, eval_on_overlap, loss_functions, loss_weights, weights, power, max_epochs)

    # Define WandB logger
    # wandb_logger = WandbLogger(project="CSE547 Final Project", name = f"Debugging-GoAT-fold{fold_no}-{run_identifier}")

    wandb_logger = logger_setup(project_name = "CSE547 Final Project", experiment_name = f"Debugging-GoAT-fold{fold_no}-{run_identifier}") # in train_utils

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

