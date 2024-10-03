Under construction: Contains code for CSE547 Final Project.

## Navigating the Repo
Radiomics results extracted from BraTS 2024 GoAT Training Dataset: /GoAT

Radiomics results extracted from BraTS 2023 Glioma Training Dataset: /Glioma

Code for neural network: /neuralnet


## Project Motivation
Current deep learning based tumor segmentation efforts are trained on highly-specialized datasets typically 
containing little variation in factors such as lesion types, medical institutions, and demographics. This 
specialized training limits the generalizability and robustness of the models, thereby reducing their potential 
for translation to a clinical setting.  For this reason, we aim to create brain tumor segmentation algorithms 
capable of adapting and generalizing to different scenarios with little prior information or data on the target classes.

## Dataset Used
In this project we will be using data from the Brain Tumor Segmentation (BraTS) Generalizability Across
Tumors (GoAT) Challenge. The dataset contains 2200 samples of multi-institutional, preoperative, routine 
clinically-acquired multi-parametric 3D MRI (mpMRI) scans of different
brain tumors. The mpMRI scans are available as NIfTI files (.nii.gz) and describe a) native
(T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated 
Inversion Recovery (T2-FLAIR) volumes, and were acquired with different clinical
protocols and various scanners from multiple data contributing institutions. All the imaging
datasets have been annotated manually by one to four raters, following the same annotation
protocol and their annotations were approved by experienced neuro-radiologists. Annotations
comprise the GD-enhancing tumor (ET — label 3), the peritumoral edematous/invaded
tissue (ED — label 2), and the necrotic tumor core (NCR — label 1).
The ground truth data were created after pre-processing, i.e., co-registered to the same
anatomical template, interpolated to the same resolution (1 mm3), and skull-stripped.

## Description of Approach
The strategy chosen involves framing the tumor segmentation generalization problem as a domain adaptation problem, where
the aim is to simultaneously improve performance in all domains. In this context, we 
define a domain as a group in an artificial clustering of the training datasets, and we aim to improve performance
over all clusters. A natural clustering for this dataset would be based on tumor type; however, given that we 
do not have this information, the idea is to produce new clusterings using an unsupervised approach. Furthermore, 
we plan to analyze the sensitivity of the model performance to the clustering approach chosen, and interpret the 
results to try to understand which sets of features improve segmentation.



## Model Architecture
![547diag](https://github.com/user-attachments/assets/54b4dbd6-5ffb-4af2-8b2c-dc02df7bb485)

