#!/bin/bash

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

QWEN3VL_PATH="Qwen/Qwen3-VL-8B-Instruct"
LLM_PATH="./output/stage1/final_model"
ALIGN_JSON="./datasets/videomodified.json"
DATA_FOLDER="./datasets"
OUTPUT_DIR="./output/stage2"

python train/stage2_align.py \
    --qwen3vl_path "${QWEN3VL_PATH}" \
    --llm_path "${LLM_PATH}" \
    --align_json "${ALIGN_JSON}" \
    --data_folder "${DATA_FOLDER}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_query 32 \
    --qformer_hidden 768 \
    --qformer_layers 6 \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --bf16
