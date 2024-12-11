#!/bin/bash

MODEL_PATH=$1
DATA_ROOT=$2
RESULTS_DIR=$3

# 檢查必要的目錄是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory does not exist: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root directory does not exist: $DATA_ROOT"
    exit 1
fi

# 檢查子目錄數量
FOLDER_COUNT=$(ls -d "$DATA_ROOT"/*/ 2>/dev/null | wc -l)
if [ "$FOLDER_COUNT" -ne 8 ]; then
    echo "Error: Expected 8 subdirectories in data root, found $FOLDER_COUNT"
    exit 1
fi

# 獲取當前SLURM任務的GPU ID
GPU_ID=$SLURM_LOCALID

# 處理指定的資料夾
FOLDERS=($(ls -d "$DATA_ROOT"/*/ 2>/dev/null))
FOLDER="${FOLDERS[$GPU_ID]}"
FOLDER_NAME=$(basename "$FOLDER")
OUTPUT_FILE="$RESULTS_DIR/${FOLDER_NAME}_results.json"

echo "Processing $FOLDER_NAME on GPU $GPU_ID"

# 執行評估
python llama_eval_singlethread.py \
    --model_path "$MODEL_PATH" \
    --folder_path "$FOLDER" \
    --gpu_id "$GPU_ID" \
    --output_path "$OUTPUT_FILE" \
    > "$RESULTS_DIR/${FOLDER_NAME}.log" 2>&1