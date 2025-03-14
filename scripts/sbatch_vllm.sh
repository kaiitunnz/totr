
#!/bin/bash

set -e

#SBATCH --job-name=totr-llm
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=12:00:00         # Set this carefully to avoid penalty

sbatch --gres=gpu:h100-47:1 scripts/server_vllm.sh
