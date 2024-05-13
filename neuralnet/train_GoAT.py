import argparse
import torch
from utils import *
from data import trans
from data.datasets import LoadDatasetswClusterID

#  import transforms ####
from torchvision import transforms
from train_utils import load_fold_file
from torch.utils.data import DataLoader

import pickle

import os
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import lightning as L
from lightning.pytorch.cli import LightningCLI

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from train_utils import *


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



    @staticmethod
    def compute_loss(output, seg, loss_functs, loss_weights):
        """Computes weighted loss between model output and ground truth, summed across each region."""
        loss = 0.
        for n, loss_function in enumerate(loss_functs):      
            temp = 0
            for i in range(3):
                temp += loss_function(output[:,i:i+1], seg[:,i:i+1])

            loss += temp * loss_weights[n]
        return loss

    def training_step(self, batch, batch_idx): 

        subject_id, imgs, true_classification = batch

        # Unpack the data
        x1 = imgs[0]
        x2 = imgs[1]
        x3 = imgs[2]
        x4 = imgs[3]
        seg = imgs[4]

        seg3 = split_seg_labels(seg)

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

        output, pred_classification, latent = self(x_in, self.alpha) # equivalent to self.model(x_in)
        output = output.float()

        segmentation_loss = self.compute_loss(output, seg, self.loss_functions, self.weights)
        classifier_loss = self.domain_criterion(pred_classification, true_classification)

        loss = self.loss_weights[0]*segmentation_loss + self.loss_weights[1]*classifier_loss

        # Log losses to TensorBoard (changing to WandB soon..)
        self.log("seg_loss", segmentation_loss, on_step=False, on_epoch=True)
        self.log("classif_loss", classifier_loss, on_step=False, on_epoch=True)
        self.log("backprop_loss", loss, on_step=False, on_epoch=True)
        self.log('epoch_loss', loss, on_step=False, on_epoch=True) # Logs mean loss per epoch

        try:
            self.log(f'cluster{true_classification}_seg_loss', segmentation_loss)
            self.log(f'cluster{true_classification}_classif_loss', classifier_loss)
        except:
            print(f"Classification {true_classification} not understood.")

        return loss
    
    def validation_step(self, batch, batch_idx):
        # if self.calculate_eval_metrics == False:
        print("In Validation.....")
        pass
            
             
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.init_lr, weight_decay = 0, amsgrad= True)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_lambda),
            'interval': 'epoch',  
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]

    def lr_lambda(self, current_epoch):
        """Custom learning rate scheduler."""
        return np.power(1 - (current_epoch / self.max_epochs), self.power)


# Lightning Data Module    
class BraTSDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 1, test_data_dir: str = "path/to/dir", folds_dir: str = "path/to/dir", fold_no: int = 0, cluster_mapping: dict = {}):
        super().__init__()
        self.data_dir = data_dir
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
            self.brats_test = LoadDatasetswClusterID(self.data_dir, self.transforms, self.cluster_mapping,  normalized=True, gt_provided= True, partial_file_names = os.listdir(self.test_data_dir))

    def train_dataloader(self):
        return DataLoader(self.brats_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.brats_val, batch_size=self.batch_size) 

    def test_dataloader(self):
        return DataLoader(self.brats_test, batch_size=self.batch_size) 

    @staticmethod
    def load_file_names(data_dir, folds_dir, fold_no):
        val_dir = os.path.join(folds_dir , sorted( os.listdir(folds_dir) )[fold_no])
        val_file_names = load_fold_file(val_dir)

        print(f"Data dir, type: {data_dir}, {type(data_dir)}")

        # train_file_names = [name for name in os.listdir(data_dir) if name not in val_file_names]

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
            model_str, partial_file_names, folds_dir, fold_no, cluster_dict, max_epochs, lr, power, eval_on_overlap, train_on_overlap) = parse_args() #definition in train_utils


    model_architecture = MODEL_STR_TO_FUNC[model_str]
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
    batch_size = 1
    test_data_dir = test_dir
    folds_dir = folds_dir
    fold_no = fold_no

    # Load cluster mapping
    with open(cluster_dict_path, 'rb') as file:
        cluster_mapping = pickle.load(file)

    # Make checkpoint directory
    new_ckpt_dir = os.path.join(out_dir, 'new_checkpoints')
    if not os.path.exists(new_ckpt_dir):
        os.makedirs(new_ckpt_dir)
        os.system('chmod a+rwx ' + new_ckpt_dir)

    # Instantiate DataModule
    dm = BraTSDataModule(data_dir = data_dir, batch_size = batch_size, test_data_dir = test_data_dir, folds_dir = folds_dir, fold_no = fold_no, cluster_mapping=cluster_mapping)
    
    # Instantiate Trainer
    seed_everything(42, workers = True) # sets seeds for numpy, torch and python.random.
    checkpoint_callback = ModelCheckpoint(every_n_epochs = 2, dirpath=new_ckpt_dir, filename="train-GoAT-{epoch:02d}-{seg_loss:.2f}")
    trainer = Trainer(fast_dev_run=1, max_epochs=max_epochs, default_root_dir=out_dir, deterministic = True) # Will automatically train with system devices and the maximum number of GPUs available (see documentation here: https://lightning.ai/docs/pytorch/stable/common/trainer.html)

    model = LitGoAT(model_architecture, alpha, init_lr, train_on_overlap, eval_on_overlap, loss_functions, loss_weights, weights, power, max_epochs)

    if os.path.exists(ckpt_dir):
        model = LitGoAT.load_from_checkpoint(ckpt_dir)

    trainer.fit(model, datamodule=dm)
    trainer.validate(datamodule=dm)

