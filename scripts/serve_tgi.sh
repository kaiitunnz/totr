#!/bin/bash

set -e

model=meta-llama/Meta-Llama-3-8B-Instruct
volume=$HF_HOME # share a volume with the Docker container to avoid downloading weights every run
port=8010

docker run --gpus all --shm-size 64g -p $port:80 -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference \
    --model-id $model
