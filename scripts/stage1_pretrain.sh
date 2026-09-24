#!/bin/bash

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

MODEL_PATH="Qwen/Qwen3-8B"
TEXT_DIR="./datasets/pretrain_text"
OUTPUT_DIR="./output/stage1"
DEEPSPEED="./scripts/zero3.json"

NPROC_PER_NODE=${NPROC_PER_NODE:-8}

torchrun --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=127.0.0.1 \
    --master_port=16666 \
    train/stage1_pretrain.py \
    --model_name_or_path "${MODEL_PATH}" \
    --text_dir "${TEXT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-6 \
    --max_length 2048 \
    --bf16 \
    --deepspeed "${DEEPSPEED}"
