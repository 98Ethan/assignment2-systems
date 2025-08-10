#!/bin/bash

# Usage: ./benchmark_all_sizes.sh [-f] [-p] [-w warmup_steps] [-t timing_steps] [-h]

BENCHMARK_MODE=""
WARMUP_STEPS=10
TIMING_STEPS=100
USE_NSYS=""

while getopts "pfw:t:h" opt; do
    case $opt in
        f) BENCHMARK_MODE="--forward_only" ;;
        p) USE_NSYS="true" ;;
        w) WARMUP_STEPS="$OPTARG" ;;
        t) TIMING_STEPS="$OPTARG" ;;
        h) echo "Usage: $0 [-f] [-p] [-w warmup_steps] [-t timing_steps] [-h]"
           echo "  -f: inference mode (forward pass only)"
           echo "  -p: enable nsys profiling"
           echo "  -w: number of warmup steps (default: 10)"
           echo "  -t: number of timing steps (default: 100)"
           echo "  -h: show this help"
           exit 0 ;;
    esac
done

if [ "$BENCHMARK_MODE" = "--forward_only" ]; then
    MODE_NAME="inf"
    echo "Running INFERENCE benchmarks (forward pass only)"
else
    MODE_NAME="train"
    echo "Running TRAINING benchmarks (forward + backward pass)"
fi

if [ "$USE_NSYS" = "true" ]; then
    echo "NSYS profiling: ENABLED"
else
    echo "NSYS profiling: DISABLED"
fi

echo "Warmup: $WARMUP_STEPS, Timing: $TIMING_STEPS"

# Create logs/benchmark directory
RESULTS_DIR="logs/benchmark"
NSYS_DIR="logs/nsys"
mkdir -p $RESULTS_DIR
if [ "$USE_NSYS" = "true" ]; then
    mkdir -p $NSYS_DIR
    echo "Results will be saved to: $RESULTS_DIR/"
    echo "NSYS profiles will be saved to: $NSYS_DIR/"
else
    echo "Results will be saved to: $RESULTS_DIR/"
fi

# Model configurations from the table
declare -a CONFIGS=(
    "small 768 12 12 3072"
    "medium 1024 24 16 4096"
    "large 1280 36 20 5120"
    "xl 1600 48 25 6400"
    "2.7B 2560 32 32 10240"
)

# Context lengths to sweep
declare -a CONTEXT_LENGTHS=(128 256 512 1024)

echo "=================================================================="

# Run benchmarks for each configuration and context length
for config in "${CONFIGS[@]}"; do
    read -r name d_model num_layers num_heads d_ff <<< "$config"
    
    for context_length in "${CONTEXT_LENGTHS[@]}"; do
        # Generate filename with model parameters and context length
        LOG_FILE="$RESULTS_DIR/${MODE_NAME}_d${d_model}_l${num_layers}_h${num_heads}_ff${d_ff}_ctx${context_length}_w${WARMUP_STEPS}_t${TIMING_STEPS}.csv"
        
        echo ""
        echo "BENCHMARKING $name MODEL (context_length=$context_length)"
        echo "Config: d_model=$d_model, num_layers=$num_layers, num_heads=$num_heads, d_ff=$d_ff, context_length=$context_length"
        echo "Log file: $LOG_FILE"
        echo "=================================================================="
        
        if [ "$USE_NSYS" = "true" ]; then
            NSYS_FILE="$NSYS_DIR/${MODE_NAME}_d${d_model}_l${num_layers}_h${num_heads}_ff${d_ff}_ctx${context_length}_w${WARMUP_STEPS}_t${TIMING_STEPS}.nsys-rep"
            echo "NSYS profile: $NSYS_FILE"
            nsys profile --output="$NSYS_FILE" --force-overwrite=true --trace=cuda,nvtx uv run cs336_systems/benchmark.py --model_name $name --d_model $d_model --num_layers $num_layers --num_heads $num_heads --d_ff $d_ff --context_length $context_length --warmup_steps $WARMUP_STEPS --timing_steps $TIMING_STEPS --log_file $LOG_FILE $BENCHMARK_MODE
        else
            uv run cs336_systems/benchmark.py --model_name $name --d_model $d_model --num_layers $num_layers --num_heads $num_heads --d_ff $d_ff --context_length $context_length --warmup_steps $WARMUP_STEPS --timing_steps $TIMING_STEPS --log_file $LOG_FILE $BENCHMARK_MODE
        fi
    done
done

echo ""
echo "ALL BENCHMARKS COMPLETED!"
echo "=================================================================="