# File to store string to function dictionaries for arg parsing
# Update whenever new loss or model functions are needed

import torch.nn as nn
import losses.new_losses as lf # in same dir
from losses import lf2 # in same directory
from losses import EdgeLoss3D # in same directory
import torch
import numpy as np
import model
import  matplotlib.pyplot as plt
from monai.metrics import HausdorffDistanceMetric
import os
from skimage.metrics import structural_similarity as ssim
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder



# from models import model_GAN_segmentation


LOSS_STR_TO_FUNC = {
    'mse': nn.MSELoss(),
    'cross-entropy': nn.CrossEntropyLoss(),
    'mask-regulizer': lf2.Maskregulizer(),
    'edge-loss': EdgeLoss3D.GMELoss3D(),
    'dice': lf.DiceLoss(),
    'focal': lf.FocalLoss()
    # 'hd'
}

MODEL_STR_TO_FUNC = {
        'autoencoder-with-classifier': model.DANNEncoderDecoder3D(),
        'unet-DANN': model.DANNUNet3D(),
        'extended-latent-unet-DANN': model.DANNUNet3DExtendedLatent()

} # All models here have output: [segmentation, classification, latent]

def split_seg_labels(seg):
    # Split the segmentation labels into 3 channels
    seg3 = torch.from_numpy(np.zeros(seg.shape))
    seg3 = torch.cat((seg3,seg3,seg3), dim=1)
    seg3[:,0,:,:,:] = torch.where(seg == 1, 1.,0.)
    seg3[:,1,:,:,:] = torch.where(seg == 2, 1.,0.)
    seg3[:,2,:,:,:] = torch.where(seg == 3, 1.,0.)
    return seg3


## LAYER FREEZING

FREEZE_STR_TO_LAYERS = {
    'encoder': ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'Conv6', 'Conv7'],
    'decoder': ['Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4', 'Up3', 'Up_conv3', 'Conv_1x13', 'Up2', 'Up_conv2', 'Conv_1x12', 'Up1', 'Up_conv1', 'Conv_1x11'],
    'middle' : ['Conv5', 'Conv6', 'Conv7', 'Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4'],
    'none' : [],
    'deep_decoder': ['Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4'],
    'decoder_test2': ['Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4', 'Up3', 'Up_conv3', 'Conv_1x13'],
    'decoder_test3': ['Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4', 'Up3', 'Up_conv3', 'Conv_1x13', 'Up2', 'Up_conv2', 'Conv_1x12']
}

def freeze_layers(model, frozen_layers):

    for name, param in model.named_parameters():
        needs_freezing = False
        for layer in frozen_layers:
            if layer in name:
                needs_freezing = True
                break
        if needs_freezing:
            print(f'Freezing parameter {name}')
            param.requires_grad = False


def check_frozen(model, frozen_layers):

    for name, param in model.named_parameters():
        needs_freezing = False
        for layer in frozen_layers:
            if layer in name:
                needs_freezing = True
                break
        if needs_freezing:
            if param.requires_grad:
                print(f'Warning! Param {name} should not require grad but does')
                break
            else:
                print(f'Parameter {name} is frozen')

def val(val_loader, model, train_on_overlap, eval_on_overlap, criterions, weights, out_dir, epoch, model_str): # out_dir should be out dir for fold for epoch ***.

    if model_str == 'unet-with-classifier':
        model_eval = MODEL_STR_TO_FUNC['eval-unet-with-classifier']
    else:
        model_eval = MODEL_STR_TO_FUNC[model_str]
    model_eval.load_state_dict(model.state_dict())
    
    domain_criterion = nn.BCELoss()
    plots_dir = out_dir + f'plots/plots-cv-epoch{epoch}/'
    if not os.path.exists(plots_dir):   
        os.makedirs(plots_dir)
        os.system('chmod a+rwx ' + plots_dir)
    
    # Define lists to keep track of metrics
    loss_log = []
    ssim_log = []
    D_log = []
    D_log1, D_log2, D_log3 = [], [], []

    # Create object to calculate HD95 metric
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")

    latent_dictionary = {}
    with torch.no_grad():
        for case_info, data, classification in val_loader:
            case_id, timepoint = case_info
            case_id = case_id[0]
            timepoint = timepoint[0]
            
            # Put model in eval mode
            model_eval.eval()

            # Unpack the data
            data = [t.cuda() for t in data]
            x1 = data[0]
            x2 = data[1]
            x3 = data[2]
            x4 = data[3]
            seg = data[4]

            try:
                seg3 = split_seg_labels(seg)
            except:
                print("Error, skipping test example")
                continue

            if eval_on_overlap:
                # Split the segmentation labels into partialy overlapping regions
                mask = torch.zeros_like(seg3)
                mask[:,0] = seg3[:, 0] + seg3[:, 1] + seg3[:, 2] #WHOLE TUMOR
                mask[:,1] = seg3[:, 0] + seg3[:, 2] #TUMOR CORE
                mask[:,2] = seg3[:, 2] #ENHANCING TUMOR
                mask = mask.float()
            else:
                mask = seg3.float()


            # Run the model on input data to get predicted seg labels
            x_in = torch.cat((x1, x2, x3, x4), dim=1)

            if model_str == 'unet-with-classifier':
                output, classif, latent  = model_eval(x_in)
                output = output.float()
                classif = classif.float()
                classif = classif.squeeze(1)
                true_classification = classification.float()
    
                
                class_string = str(classification)
                if class_string not in latent_dictionary:
                    latent_dictionary[class_string] = []
    
                latent_dictionary[class_string].append(latent.view(-1).cpu().numpy())
                
                domain_loss = domain_criterion(classif.to(device='cuda:1'), true_classification.to(device='cuda:1'))
            # print("Domain classification loss: ", domain_loss)
            
            # Computing overall loss according to loss criterions and corresponding weights
            loss = 0.
            for n, loss_function in enumerate(criterions):      
                loss += loss_function(output.to(device='cuda:1'), mask.to(device='cuda:1')) * weights[n]
            loss_log.append(loss.detach().cpu())

            # Convert output probabilities to predictions
            if train_on_overlap:
                # Thresholds
                t1 = 0.45
                t2 = 0.4
                t3 = 0.45
                output_ = np.squeeze(output.cpu().detach().numpy())
                c1, c2, c3 = output_[0] > t1, output_[1] > t2, output_[2] > t3
                pred = (c1 > 0).astype(np.uint8) # NCR
                pred[(c2 == False) * (c1 == True)] = 2 # ED
                pred[(c3 == True) * (c1 == True)] = 3 # ET
                output_plot = np.zeros_like(output_)
                output_plot[0] = (pred == 1) #NCR
                output_plot[1] = (pred == 2) #ED
                output_plot[2] = (pred == 3) #ET
                output_plot = output_plot.astype(np.uint8)
            else:
                t = 0.5 #threshold
                output_ = np.squeeze(output.cpu().detach().numpy())
                c1, c2, c3 = output_[0], output_[1], output_[2]
                max_label = np.maximum(np.maximum(c1, c2), c3)
                pred = np.zeros_like(output_)
                pred[0] = np.where(c1 < max_label, 0, max_label)
                pred[1] = np.where(c2 < max_label, 0, max_label)
                pred[2] = np.where(c3 < max_label, 0, max_label)
                output_plot = np.zeros_like(output_)
                for c in range(0, 3):
                    output_plot[c] = np.where(pred[c] > t, 1., 0.)
                output_plot = output_plot.astype(np.uint8)

            if eval_on_overlap:
                # Combine prediction labels into partially overlapping regions
                mov = np.zeros_like(output_plot)
                mov[0] = output_plot[0] + output_plot[1] + output_plot[2] #WHOLE TUMOR
                mov[1] = output_plot[0] + output_plot[2] #TUMOR CORE
                mov[2] = output_plot[2] #ENHANCING TUMOR
            else:
                mov = output_plot
            
            tar = np.squeeze(mask.detach().cpu().numpy())

            # Calculating similarity score between prediction and ground truth
            ssim1 = ssim(mov[  0, :, :, :], tar[  0, :, :, :], data_range=mov[  0, :, :, :].max() - mov[  0, :, :, :].min())
            ssim2 = ssim(mov[  1, :, :, :], tar[  1, :, :, :], data_range=mov[  1, :, :, :].max() - mov[  1, :, :, :].min())
            ssim3 = ssim(mov[  2, :, :, :], tar[  2, :, :, :], data_range=mov[  2, :, :, :].max() - mov[  2, :, :, :].min())
            ssim_round = (ssim1+ssim2+ssim3)/3
            ssim_log.append(ssim_round)
            
            # Calculating Dice scores
            DiceScore = Dice()
            D1 = DiceScore(mov[  0, :, :, :], tar[  0, :, :, :])
            D2 = DiceScore(mov[  1, :, :, :], tar[  1, :, :, :])
            D3 = DiceScore(mov[  2, :, :, :], tar[  2, :, :, :])
            D_round = (D1+D2+D3)/3
            D_log.append(D_round)
            D_log1.append(D1)
            D_log2.append(D2)
            D_log3.append(D3)

            # Calculating Hausdorff distance
            mov_ = torch.from_numpy(np.expand_dims(mov, axis=0))
            hd_metric(y_pred=mov_, y=mask)

            # Plotting
            nslice = 64
            plt.figure(figsize=(20, 9), dpi=300)
            plt.subplot(3, 4, 1)
            plt.imshow(x1.cpu().detach().numpy()[0, 0, :, :,nslice], cmap='gray')
            plt.title('t1n')
            plt.subplot(3, 4, 2)
            plt.imshow(x2.cpu().detach().numpy()[0, 0, :, :, nslice], cmap='gray')
            plt.title('t2w')
            plt.subplot(3, 4, 3)
            plt.imshow(x3.cpu().detach().numpy()[0, 0, :, :, nslice], cmap='gray')
            plt.title('t2f')
            plt.subplot(3, 4, 4)
            plt.imshow(x4.cpu().detach().numpy()[0, 0, :, :, nslice], cmap='gray')
            plt.title('t1c')
            
            plt.subplot(3, 4, 5)
            plt.imshow(output_plot[ 0, :, :, nslice])
            plt.title('Prediction, label 1 (NCR)')
            plt.subplot(3, 4, 6)
            plt.imshow(output_plot[ 1, :, :, nslice])
            plt.title('Prediction, label 2 (ED)')
            plt.subplot(3, 4, 7)
            plt.imshow(output_plot[ 2, :, :, nslice])
            plt.title('Prediction, label 3 (ET)')
            
            #plt.subplot(3, 4, 8)
            # Convert pred into 3 channels for plotting
            #all_labels_pred = np.zeros_like(output_plot)
            #all_labels_pred[0] = (output_plot[0] == 1)
            #all_labels_pred[1] = (output_plot[1] == 1) * 2.
            #all_labels_pred[2] = (output_plot[2] == 1) * 3.
            #all_labels_pred=np.moveaxis(all_labels_pred, 0, 3)
            #all_labels_pred=np.expand_dims(all_labels_pred, axis=0)
            #plt.imshow(all_labels_pred[0, :, :, nslice, :])
            
            plt.subplot(3, 4, 8)
            combined_labels = np.argmax(output_plot[:, :, :, nslice], axis=0)  # Use argmax to select the label with the highest prediction per pixel
            plt.imshow(combined_labels)  # This will automatically use different colors for different labels
            
            plt.title('Prediction, all labels')
            
            plt.subplot(3, 4, 9)
            plt.imshow(seg3.cpu().detach().numpy()[0, 0, :, :, nslice])
            plt.title('Ground truth, label 1 (NCR)')
            plt.subplot(3, 4, 10)
            plt.imshow(seg3.cpu().detach().numpy()[0, 1, :, :, nslice])
            plt.title('Ground truth, label 2 (ED)')
            plt.subplot(3, 4, 11)
            plt.imshow(seg3.cpu().detach().numpy()[0, 2, :, :, nslice])
            plt.title('Ground truth, label 3 (ET)')
            plt.subplot(3, 4, 12)
            all_labels=np.moveaxis(seg3.cpu().detach().numpy(), 1, 4)
            plt.imshow(all_labels[0, :, :, nslice, :])
            plt.title('Ground truth, all labels')

            plt.savefig(os.path.join(plots_dir, f'{case_id}-{timepoint}.png'))
            plt.close()

        # Convert metrics to arrays and compute means
        mean_loss = np.mean(np.asarray(loss_log))
        mean_ssim = np.mean(np.asarray(ssim_log))
        mean_D = np.mean(np.asarray(D_log))
        mean_D1 = np.mean(np.asarray(D_log1))
        mean_D2 = np.mean(np.asarray(D_log2))
        mean_D3 = np.mean(np.asarray(D_log3))

        metric_batch = hd_metric.aggregate()
        hd1 = metric_batch[0].item()
        hd2 = metric_batch[1].item()
        hd3 = metric_batch[2].item()
        mean_hd = torch.mean(metric_batch).item()
        # Uncomment line below to see array of hd values per channel per batch/example
        # print(hd_metric._synced_tensors)

        output_data = ''
        output_data += f'Average loss={mean_loss:.4f}\n'
        output_data += f'Average similarity score={mean_ssim}\n'
        output_data += f'Average overall Dice score={mean_D}\n'
        output_data += f'Average Dice score 1 = {mean_D1}\n'
        output_data += f'Average Dice score 2 = {mean_D2}\n'
        output_data += f'Average Dice score 3 = {mean_D3}\n'
        output_data += f'Average overall HD95 = {mean_hd}\n'
        output_data += f'Average HD95 1 = {hd1}\n'
        output_data += f'Average HD95 2 = {hd2}\n'
        output_data += f'Average HD95 3 = {hd3}\n'

        print(output_data)
        f=open(out_dir + 'metrics.txt', 'w')
        f.write(output_data)
        f.close()

    if model_str == 'unet-with-classifier':
        # Plot tSNE
        all_data = []
        labels = []
        for class_label, latent_lists in latent_dictionary.items():
            for latent in latent_lists:
                all_data.append(latent)
                labels.append(class_label)
    
        # Convert to numpy arrays
        all_data_np = np.array(all_data)
        labels_np = np.array(labels)
    
        # Encode labels to integers if they are not already
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels_np)
    
        # Run t-SNE
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(all_data_np)
    
        # Plot
        plt.figure(figsize=(16,10))
        scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels_encoded, cmap='viridis')
        plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)))
        plt.clim(-0.5, len(label_encoder.classes_)-0.5)
        plt.xticks([])
        plt.yticks([])
        plt.title('t-SNE visualization of latent vectors')
        plt.xlabel('t-SNE axis 1')
        plt.ylabel('t-SNE axis 2')
    
        # Optionally add a legend
        plt.legend(handles=scatter.legend_elements()[0], labels=label_encoder.classes_)
    
        plt.savefig(os.path.join(plots_dir, f'tSNE-epoch{epoch}.png'))
        plt.close()
    
    return mean_loss, mean_ssim, mean_D, mean_D1, mean_D2, mean_D3, mean_hd, hd1, hd2, hd3


# Evaluation metric functions
class Dice():
    def __init__(self):
        pass
    def __call__(self, y_pred, y_true):
        tol=1e-12
        numerator = (y_pred * y_true).sum()
        denominator = y_pred.sum() + y_true.sum()
        return ((2 * numerator) + tol)/ (denominator + tol)