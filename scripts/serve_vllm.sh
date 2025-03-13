#!/bin/bash

set -e

model=meta-llama/Meta-Llama-3-8B-Instruct
host=localhost
port=8010

vllm serve $model \
    --host=$host \
    --port=$port \
    --enable-prefix-caching
