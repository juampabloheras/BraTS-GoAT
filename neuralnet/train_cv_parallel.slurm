#!/bin/bash
#SBATCH --job-name=train-cv-GoAT
#SBATCH --mail-user=jehr@uw.edu
#SBATCH --mail-type=ALL
#SBATCH -A kurtlab
#SBATCH -p gpu-a40
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=/gscratch/kurtlab/juampablo/BraTS-GoAT/TESTS/test0531_agglom_k3/output_train.txt
#SBATCH --error=/gscratch/kurtlab/juampablo/BraTS-GoAT/TESTS/test0531_agglom_k3/error_train.txt
#SBATCH --chdir=/gscratch/kurtlab/juampablo/BraTS-GoAT/neuralnet


# Number of cross-validation folds
NUM_FOLDS=5

### Parameters that don't change across folds
# Directories
TRAIN_DIR='/gscratch/scrubbed/juampablo/BraTS-GoAT/DATA/training'
TEST_DIR='/gscratch/scrubbed/juampablo/BraTS-GoAT/DATA/validation'
CKPT_DIR='None'
BASE_OUT_DIR='/gscratch/kurtlab/juampablo/BraTS-GoAT/TESTS/test0531_agglom_k3/out_dir'
FOLDS_DIR='/gscratch/kurtlab/juampablo/BraTS-GoAT/neuralnet/CVFolds/cv-5-tianyi'


# Training parameters
LOSS=('mse' 'cross-entropy')
WEIGHTS=(1 0.07177350190342198)
LOSS_WEIGHTS=(1 0.02)
TRAIN_ON_OVERLAP='--no-train_overlap'
EVAL_ON_OVERLAP='--no-eval_on_overlap'

# Model and parameters
MODEL='unet-DANN'
ALPHA=1

# Data Loader
PARTIAL_FILE_NAMES_DIR=('All')

# Used to index runs
RUN_IDENTIFIER="agglomk3-$(($RANDOM%1000 * $RANDOM + $RANDOM))"
CLUSTER_DICT='/mmfs1/gscratch/kurtlab/juampablo/BraTS-GoAT/neuralnet/cluster_dicts/agglom_k3_clusters.pkl'


# Cross-validation training
for FOLD_NO in $(seq 0 $(($NUM_FOLDS - 1))); do
    OUT_DIR="${BASE_OUT_DIR}/fold_${FOLD_NO}"
    TRAIN_FILES_DIR="${OUT_DIR}/train_files"
    
    # Create the output and train_files directories
    mkdir -p "$TRAIN_FILES_DIR"

    # Copy train files to the train_files directory for bookkeeping of tests
    cp train_GoAT.py utils.py model.py train_cv_parallel.slurm "$TRAIN_FILES_DIR" 


    # Track settings information in text file

    settings_info="Train Directory: ${TRAIN_DIR[@]}
    Test Directory: ${TEST_DIR[@]}
    Checkpoint Directory: $CKPT_DIR
    Alpha: $ALPHA
    Loss: ${LOSS[@]}
    Loss Weights: ${LOSS_WEIGHTS[@]}
    Weights: ${WEIGHTS[@]}
    Model: $MODEL
    Partial File Names Directory: ${PARTIAL_FILE_NAMES_DIR[@]}
    Folds Directory: $FOLDS_DIR
    Fold Number: $FOLD_NO
    Train on Overlap: $TRAIN_ON_OVERLAP
    Eval on Overlap: $EVAL_ON_OVERLAP"
    
    # Save the settings to a text file
    echo "$settings_info" > "${OUT_DIR}/train_settings.txt"   



    # Submit a separate Slurm job for each fold
    sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=train-fold-${FOLD_NO}
#SBATCH --mail-user=jehr@uw.edu
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-node=a40:2
#SBATCH --account=kurtlab
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=220G
#SBATCH --time=6:00:00
#SBATCH --chdir=/mmfs1/gscratch/kurtlab/juampablo/BraTS-GoAT/neuralnet
#SBATCH --export=all
#SBATCH --output=${OUT_DIR}/output_train_fold_${FOLD_NO}.txt
#SBATCH --error=${OUT_DIR}/error_train_fold_${FOLD_NO}.txt

source ~/.bashrc
source activate undergraddl

python3 train_GoAT.py \
    --train_dir ${TRAIN_DIR[@]} \
    --test_dir ${TEST_DIR[@]} \
    --ckpt_dir $CKPT_DIR \
    --out_dir $OUT_DIR \
    --loss ${LOSS[@]} \
    --loss_weights ${LOSS_WEIGHTS[@]} \
    --weights ${WEIGHTS[@]} \
    --model $MODEL \
    --alpha $ALPHA \
    --partial_file_names_dir ${PARTIAL_FILE_NAMES_DIR[@]} \
    --folds_dir $FOLDS_DIR \
    --fold_no $FOLD_NO \
    --cluster_dict $CLUSTER_DICT \
    --run_identifier $RUN_IDENTIFIER \
    $TRAIN_ON_OVERLAP $EVAL_ON_OVERLAP
EOT
done

# python3 train_GoAT.py --train_dir ${TRAIN_DIR[@]} --test_dir ${TEST_DIR[@]}  --ckpt_dir $CKPT_DIR --out_dir $OUT_DIR --loss ${LOSS[@]} --loss_weights ${LOSS_WEIGHTS[@]} --weights ${WEIGHTS[@]} --model $MODEL --alpha $ALPHA --partial_file_names_dir ${PARTIAL_FILE_NAMES_DIR[@]} --folds_dir $FOLDS_DIR --fold_no $FOLD_NO --cluster_dict $CLUSTER_DICT --run_identifier $(($RANDOM%1000 * $RANDOM + $RANDOM)) $TRAIN_ON_OVERLAP $EVAL_ON_OVERLAP > output_debug.log 2> error_debug.log
