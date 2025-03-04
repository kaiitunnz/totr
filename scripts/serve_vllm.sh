#!/bin/bash

set -e

source .env

vllm serve $LLM_MODEL \
    --host=$LLM_SERVER_HOST \
    --port=$LLM_SERVER_PORT
