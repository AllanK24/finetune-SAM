#!/bin/bash

# Set which GPUs to use
export CUDA_VISIBLE_DEVICES="0,1"

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

# Path to the train and val csvs
TRAIN_IMG_LIST="${IMG_FOLDER}/${DATASET_NAME}/train.csv"
VAL_IMG_LIST="${IMG_FOLDER}/${DATASET_NAME}/val.csv"

# Where to save the new model checkpoints
DIR_CHECKPOINT="${BASE_DIR}/2D-SAM_${ARCH}_${DATASET_NAME}"

# --- Run the Python Training Script ---
python "${FINETUNE_SAM_DIR}/DDP_splitgpu_train_finetune_noprompt.py" \
    -if_warmup True \
    -if_split_encoder_gpus True \
    -finetune_type "lora" \ 
    -arch "$ARCH" \
    -dataset_name "$DATASET_NAME" \
    -sam_ckpt "$SAM_CKPT" \
    -img_folder "$IMG_FOLDER" \
    -mask_folder "$MASK_FOLDER" \
    -train_img_list "$TRAIN_IMG_LIST" \
    -val_img_list "$VAL_IMG_LIST" \
    -epochs 200 \
    -warmup_period 200 \
    -lr 0.0001 \
    -b 4 \
    -if_encoder_lora_layer True \
    -if_decoder_lora_layer True \
    -encoder_lora_layer "[]" \
    -dir_checkpoint "$DIR_CHECKPOINT"