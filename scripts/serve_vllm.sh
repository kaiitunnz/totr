#!/bin/bash

set -e

model=meta-llama/Llama-3.1-8B-Instruct
host=localhost
port=8010

vllm serve $model \
    --host=$host \
    --port=$port \
    --enable-prefix-caching
