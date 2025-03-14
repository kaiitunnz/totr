#!/bin/bash

#SBATCH --job-name=totr-llm
#SBATCH --gres=gpu:h100-96:1    # Set as per availability
#SBATCH --time=12:00:00         # Set this carefully to avoid penalty

set -e

model=meta-llama/Llama-3.1-8B-Instruct
host=0.0.0.0
port=8010

vllm serve $model \
    --host=$host \
    --port=$port \
    --enable-prefix-caching
