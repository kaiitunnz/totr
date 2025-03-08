## Installation

1. Create and activate a Conda environment
   ```sh
   conda create totr python=3.11 -y
   conda activate totr
   ```
2. Install Poetry for dependency management.

   ```sh
   pip install poetry
   ```

3. Run the following command to install all the dependencies.
   ```sh
   poetry install --with lint,vllm
   ```
   Note that `lint` is optional but strongly recommended for code linting and formatting. On the other hand, you can remove `vllm` if you do not plan to use vLLM as the LLM server.

## Project Structure
- [`benchmark`](benchmark): Benchmarking tasks, utility code, and baselines' implementation.
- [`configs`](configs): Configuration files for different models, datasets, and systems.
- [`datasets`](datasets): Dataset-related files.
- [`prompts`](prompts): Collection of prompts used in the experiments.
- [`results`](results): Benchmark results.
- [`scripts`](scripts): Utility scripts for developing and running experiments.
- [`src`](src): ToTR's implementation.
- [`tests`](tests): Various test cases. Currently contains only basic tests without unit testing. It can be a reference for how different functions and classes are used.
- [`config.json`](config.json) Default configuration file. Will be removed in the future.

## Utility scripts

- [`scripts/format.sh`](scripts/format.sh): Script for code linting, formatting, type-checking, and spell-checking.

  Usage:

  ```sh
  bash scripts/format.sh --all
  ```

- [`scripts/serve_tgi.sh`](scripts/serve_tgi.sh): Script for starting the HuggingFace TGI server.

  Usage:

  ```sh
  bash scripts/serve_tgi.sh
  ```

- [`scripts/serve_vllm.sh`](scripts/serve_vllm.sh): Script for serving the vLLM server.

  Usage:

  ```sh
  bash scripts/serve_vllm.sh
  ```
