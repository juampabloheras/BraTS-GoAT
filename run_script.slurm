#!/bin/bash


#SBATCH --job-name=make2d
#SBATCH --mail-user=jehr@uw.edu 
#SBATCH --mail-type=ALL
#SBATCH -A kurtlab
#SBATCH -p gpu-a40
#SBATCH --mem=100G  # Assuming you meant to use 230G based on your command, adjust if necessary
#SBATCH --gpus-per-node=a40:2
#SBATCH --ntasks=2
#SBATCH --time=03:00:00
#SBATCH --output=/gscratch/kurtlab/juampablo/BraTS-GoAT/output_val_eval.txt
#SBATCH --error=/gscratch/kurtlab/juampablo/BraTS-GoAT/error_val_eval.txt
#SBATCH --chdir=/mmfs1/gscratch/kurtlab/juampablo/BraTS-GoAT/



source ~/.bashrc
source activate undergraddl



python3 make2dMRI.py

