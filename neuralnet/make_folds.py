import numpy as np
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime
import pandas as pd

def make_folds(in_dir, out_dir, num_folds):

    """
    Splits a dataset into multiple folds for cross-validation and saves each fold's file names in separate text files.

    Args:
        in_dir (str): The directory path containing the dataset files to be divided into folds.
            Example: '/path/to/dataset' where the dataset consists of individual file entries.
        
        out_dir (str): The output directory path where the fold directories will be created.
            Example: '/path/to/output' where each fold's file names will be saved in separate text files within a generated directory.
            
        num_folds (int): The number of folds to divide the dataset into.
            Example: 5 for a 5-fold cross-validation setup.

    Example Usage:
        To split a dataset located in '/data/my_dataset' into 5 folds and save the fold information in '/data/folds_output':
        >>> make_folds('/data/my_dataset', '/data/folds_output', 5)

        This will create a new directory in '/data/folds_output', such as 'cv-5-20230101-0', and within this directory, there will be 5 text files (fold_1.txt, fold_2.txt, ..., fold_5.txt), each containing the names of the files allocated to that fold.
    """

    # Load data
    file_names = os.listdir(in_dir)
    random.shuffle(file_names)
    folds = np.array_split(file_names, num_folds)
    fold_lengths = [len(fold) for fold in folds]

    # Name and create directory
    today_str = datetime.now().strftime("%Y%m%d")
    cv_dir_name = f'cv-{num_folds}-{today_str}-0' # Initially append -0 to the directory name
    cv_dir = os.path.join(out_dir, cv_dir_name)

    i = 1
    while os.path.exists(cv_dir):
        cv_dir = cv_dir[:-2] + f'-{i}' # Modify the suffix incrementally if the directory exists
        i += 1

    os.makedirs(cv_dir) # Create directory with established name

    # Make files with fold names and add to directory
    for i, fold in enumerate(folds):
        fold_filename = os.path.join(cv_dir, f'fold_{i+1}.txt') # Change to .txt 
        with open(fold_filename, 'w') as f:
            for filename in fold:
                f.write("%s\n" % filename)

    print(f"All folds saved to {cv_dir}")
    print(f"Fold sizes: {fold_lengths}")

if __name__ == '__main__':
    in_dir = '/gscratch/kurtlab/brats2023/data/brats-ssa/Processed-NEWTrainingData'
    out_dir = '/gscratch/kurtlab/brats2023/repos/juampablo/brats2023/brats2023/base/CVFolds'
    num_folds = 10
    make_folds(in_dir, out_dir, num_folds)

