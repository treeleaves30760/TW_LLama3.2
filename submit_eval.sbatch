#!/bin/bash
#SBATCH --job-name=llama_eval          # 作業名稱
#SBATCH --nodes=1                      # 請求節點數
#SBATCH --ntasks-per-node=8            # 每個節點上的任務數(對應原本的8個資料夾)
#SBATCH --cpus-per-task=1              # 每個任務使用的CPU核心數
#SBATCH --gres=gpu:8                   # 請求8張GPU
#SBATCH --time=24:00:00               # 最長執行時間
#SBATCH --output=slurm_%j.out          # 標準輸出文件
#SBATCH --error=slurm_%j.err           # 標準錯誤文件

# 獲取命令行參數
MODEL_PATH=$1
DATA_ROOT=$2

# 檢查參數
if [ -z "$MODEL_PATH" ] || [ -z "$DATA_ROOT" ]; then
    echo "Usage: sbatch submit_eval.sbatch <path_to_model> <path_to_data_directory>"
    exit 1
fi

# 創建結果目錄
RESULTS_DIR="results/evaluation_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# 記錄開始時間和參數
LOG_FILE="$RESULTS_DIR/evaluation.log"
echo "Starting evaluation at $(date)" > "$LOG_FILE"
echo "Model path: $MODEL_PATH" >> "$LOG_FILE"
echo "Data root: $DATA_ROOT" >> "$LOG_FILE"

# 啟動評估腳本
srun bash run_eval_sbatch.sh "$MODEL_PATH" "$DATA_ROOT" "$RESULTS_DIR"

# 在所有評估完成後合併結果
python combine_results.py \
    --results_dir "$RESULTS_DIR" \
    --model_name "$(basename "$MODEL_PATH")" \
    --output_dir "$RESULTS_DIR/final_results" \
    >> "$LOG_FILE" 2>&1

echo "Evaluation complete. Results are available in $RESULTS_DIR"