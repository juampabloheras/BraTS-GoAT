#!/bin/bash
#SBATCH --job-name=practiceBraTS
#SBATCH --mail-user=jehr@uw.edu
#SBATCH --mail-type=ALL
#SBATCH -A kurtlab
#SBATCH -p gpu-a40
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=/gscratch/kurtlab/juampablo/BraTS-Practice/TESTS/practiceBraTS/output_train.txt
#SBATCH --error=/gscratch/kurtlab/juampablo/BraTS-Practice/TESTS/practiceBraTS/error_train.txt
#SBATCH --chdir=/gscratch/kurtlab/juampablo/BraTS-GoAT/neuralnet


# Number of cross-validation folds
NUM_FOLDS=5

### Parameters that don't change across folds
# Directories
TRAIN_DIR='/gscratch/scrubbed/juampablo/BraTS-GoAT/DATA/training'
TEST_DIR='/gscratch/scrubbed/juampablo/BraTS-GoAT/DATA/validation'
CKPT_DIR='None'
BASE_OUT_DIR='/gscratch/kurtlab/juampablo/BraTS-Practice/TESTS/practiceBraTS/out_dir'
FOLDS_DIR='/gscratch/kurtlab/juampablo/BraTS-GoAT/neuralnet/CVFolds/cv-5-tianyi'


# Training parameters
LOSS=('mse' 'cross-entropy')
WEIGHTS=(1 0.07177350190342198) # Weights for each loss function
TRAIN_ON_OVERLAP='--no-train_overlap'
EVAL_ON_OVERLAP='--no-eval_on_overlap'

# Model and parameters
MODEL='unet3d'

# Used to index runs
RUN_IDENTIFIER="testrun0619"


# Cross-validation training
for FOLD_NO in $(seq 0 $(($NUM_FOLDS - 1))); do
    OUT_DIR="${BASE_OUT_DIR}/fold_${FOLD_NO}"
    TRAIN_FILES_DIR="${OUT_DIR}/train_files"
    
    # Create the output and train_files directories
    mkdir -p "$TRAIN_FILES_DIR"

    # Copy train files to the train_files directory for bookkeeping of tests    
    cp train_GoAT.py utils.py model.py "$0" "$TRAIN_FILES_DIR"  # $0 represents the current script


    # Track settings information in text file
    settings_info="Train Directory: ${TRAIN_DIR[@]}
    Test Directory: ${TEST_DIR[@]}
    Checkpoint Directory: $CKPT_DIR
    Loss: ${LOSS[@]}
    Weights: ${WEIGHTS[@]}
    Model: $MODEL
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
#SBATCH --time=8:00:00
#SBATCH --chdir=/mmfs1/gscratch/kurtlab/juampablo/BraTS-GoAT/neuralnet
#SBATCH --export=all
#SBATCH --output=${OUT_DIR}/output_train_fold_${FOLD_NO}.txt
#SBATCH --error=${OUT_DIR}/error_train_fold_${FOLD_NO}.txt

source ~/.bashrc
source activate undergraddl

python3 lightningmodules.py \
    --train_dir ${TRAIN_DIR[@]} \
    --test_dir ${TEST_DIR[@]} \
    --ckpt_dir $CKPT_DIR \
    --out_dir $OUT_DIR \
    --loss ${LOSS[@]} \
    --weights ${WEIGHTS[@]} \
    --model $MODEL \
    --folds_dir $FOLDS_DIR \
    --fold_no $FOLD_NO \
    --run_identifier $RUN_IDENTIFIER \
    $TRAIN_ON_OVERLAP $EVAL_ON_OVERLAP
EOT
done

# Below line useful for debugging in salloc mode
# python3 lightningmodules.py --train_dir ${TRAIN_DIR[@]} --test_dir ${TEST_DIR[@]}  --ckpt_dir $CKPT_DIR --out_dir $OUT_DIR --loss ${LOSS[@]} --weights ${WEIGHTS[@]} --model $MODEL --folds_dir $FOLDS_DIR --fold_no $FOLD_NO --run_identifier $RUN_IDENTIFIER $TRAIN_ON_OVERLAP $EVAL_ON_OVERLAP > output_debug.log 2> error_debug.log
