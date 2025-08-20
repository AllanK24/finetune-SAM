#!/bin/bash

# Set which GPUs to use
export CUDA_VISIBLE_DEVICES="0"

# --- Variables ---
# Use 'vit_b' or 'vit_l', etc.
ARCH="vit_b"
# The name of your dataset, used for creating the checkpoint directory
DATASET_NAME="xrayhip"
# The root of your Kaggle working directory
BASE_DIR="/kaggle/working"
# The path to the finetune-SAM code
FINETUNE_SAM_DIR="${BASE_DIR}/finetune-SAM"

# --- Path Arguments for the Python Script ---
# Full path to the SAM model weights
SAM_CKPT="${BASE_DIR}/sam_vit_b_weights/sam_vit_b_01ec64.pth"
# The directory where your CSVs say the data is. Your CSVs have paths like
# "xrayhip/images/...", so the base input folder is "/kaggle/input/".
IMG_FOLDER="/kaggle/input"
MASK_FOLDER="/kaggle/input"

# Path to the test
TEST_IMG_LIST="${IMG_FOLDER}/${DATASET_NAME}/test.csv"

# Where to save the new model checkpoints
DIR_CHECKPOINT="${BASE_DIR}/2D-SAM_${ARCH}_${DATASET_NAME}"

# Run the Python script
python "${FINETUNE_SAM_DIR}/val_finetune_noprompt.py" \
    -finetune_type "lora" \
    -arch "$ARCH" \
    -dataset_name "$DATASET_NAME" \
    -sam_ckpt "$SAM_CKPT" \
    -img_folder "$IMG_FOLDER" \
    -mask_folder "$MASK_FOLDER" \
    -test_img_list "$TEST_IMG_LIST" \
    -dir_checkpoint "$DIR_CHECKPOINT"