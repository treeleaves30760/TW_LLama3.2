#!/bin/bash

# Help function
print_usage() {
    echo "Usage: $0 --model_path <path_to_model> --data_root <path_to_data_directory>"
    echo
    echo "Options:"
    echo "  --model_path    Path to the model directory"
    echo "  --data_root     Root directory containing the 8 data folders"
    echo
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            print_usage
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$MODEL_PATH" ] || [ -z "$DATA_ROOT" ]; then
    echo "Error: Missing required parameters"
    print_usage
fi

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory does not exist: $MODEL_PATH"
    exit 1
fi

# Check if data root exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root directory does not exist: $DATA_ROOT"
    exit 1
fi

# Count number of subdirectories
FOLDER_COUNT=$(ls -d "$DATA_ROOT"/*/ 2>/dev/null | wc -l)
if [ "$FOLDER_COUNT" -ne 8 ]; then
    echo "Error: Expected 8 subdirectories in data root, found $FOLDER_COUNT"
    exit 1
fi

# Create results directory
RESULTS_DIR="results/evaluation_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Log file for the script
LOG_FILE="$RESULTS_DIR/evaluation.log"
echo "Starting evaluation at $(date)" > "$LOG_FILE"
echo "Model path: $MODEL_PATH" >> "$LOG_FILE"
echo "Data root: $DATA_ROOT" >> "$LOG_FILE"

# Array to store process IDs
declare -a PIDS=()

# Counter for GPU IDs
GPU_ID=0

# Start evaluation for each subfolder
for FOLDER in "$DATA_ROOT"/*/; do
    if [ -d "$FOLDER" ]; then
        FOLDER_NAME=$(basename "$FOLDER")
        OUTPUT_FILE="$RESULTS_DIR/${FOLDER_NAME}_results.json"
        
        echo "Starting evaluation for $FOLDER_NAME on GPU $GPU_ID" >> "$LOG_FILE"
        
        # Run the evaluator in background
        python llama_eval_singlethread.py \
            --model_path "$MODEL_PATH" \
            --folder_path "$FOLDER" \
            --gpu_id "$GPU_ID" \
            --output_path "$OUTPUT_FILE" \
            > "$RESULTS_DIR/${FOLDER_NAME}.log" 2>&1 &
        
        # Store the process ID
        PIDS+=($!)
        
        echo "Started process $! for $FOLDER_NAME on GPU $GPU_ID" >> "$LOG_FILE"
        
        # Increment GPU ID
        ((GPU_ID++))
    fi
done

echo "All evaluation processes started. Waiting for completion..."
echo "You can monitor individual progress in the log files in $RESULTS_DIR/"

# Wait for all processes to complete
for PID in "${PIDS[@]}"; do
    wait $PID
    EXIT_CODE=$?
    echo "Process $PID completed with exit code $EXIT_CODE" >> "$LOG_FILE"
done

echo "All evaluations completed. Running results combination..."

# Run the results combiner
python combine_results.py \
    --results_dir "$RESULTS_DIR" \
    --model_name "$(basename "$MODEL_PATH")" \
    --output_dir "$RESULTS_DIR/final_results" \
    >> "$LOG_FILE" 2>&1

echo "Evaluation complete. Results are available in $RESULTS_DIR"
echo "Check $LOG_FILE for the full execution log"