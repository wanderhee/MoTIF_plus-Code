#!/bin/bash

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

QWEN3VL_PATH="Qwen/Qwen3-VL-8B-Instruct"
LLM_PATH="./output/stage1/final_model"
QFORMER_CKPT="./output/stage2/qformer.pt"
INSTRUCT_JSON="./datasets/videomodified.json"
DATA_FOLDER="./datasets"
OUTPUT_DIR="./output/stage3"
DEEPSPEED="./scripts/zero3.json"

NPROC_PER_NODE=${NPROC_PER_NODE:-8}

torchrun --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=127.0.0.1 \
    --master_port=16667 \
    train/stage3_qlora.py \
    --qwen3vl_path "${QWEN3VL_PATH}" \
    --llm_path "${LLM_PATH}" \
    --qformer_ckpt "${QFORMER_CKPT}" \
    --instruct_json "${INSTRUCT_JSON}" \
    --data_folder "${DATA_FOLDER}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_query 32 \
    --qformer_hidden 768 \
    --qformer_layers 6 \
    --lora_r 64 \
    --lora_alpha 128 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 \
    --max_length 2048 \
    --bf16 \
    --deepspeed "${DEEPSPEED}"
