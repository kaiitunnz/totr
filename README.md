## Installation

1. Clone this repository.

   ```sh
   git clone https://github.com/kaiitunnz/totr.git
   cd totr
   ```

2. Create and activate a Conda environment

   ```sh
   conda create -n totr python=3.11 -y
   conda activate totr
   ```

3. Install Poetry for dependency management.

   ```sh
   pip install poetry
   ```

4. Run the following command to install all the dependencies.

   ```sh
   poetry install --with lint,vllm
   ```

   Note that `lint` is optional but strongly recommended for code linting and formatting. On the other hand, you can remove `vllm` if you do not plan to use vLLM as the LLM server.

5. Install SpaCy.

   ```sh
   python -m spacy download en_core_web_sm
   ```

6. Download the datasets used in the experiments.

   ```sh
   bash scripts/download_processed_data.sh
   ```

7. Set up a RAG database following the [Preparing Database for RAG](#preparing-database-for-rag) section.

## Preparing Database for RAG

1. Install Elasticsearch 7.10 (source: [IRCoT](https://github.com/StonyBrookNLP/ircot)). See the following options:
   <details>
   <summary>MacOS (Homebrew)</summary>

   ```sh
   # source: https://www.elastic.co/guide/en/elasticsearch/reference/current/brew.html
   brew tap elastic/tap
   brew install elastic/tap/elasticsearch-full # if it doesn't work: try 'brew untap elastic/tap' first: untap>tap>install.
   ```

   To run the server,

   ```sh
   brew services start elastic/tap/elasticsearch-full # to start the server
   brew services stop elastic/tap/elasticsearch-full # to stop the server
   ```

   </details>

   <details>
   <summary>MacOS (wget)</summary>

   ```sh
   # source: https://www.elastic.co/guide/en/elasticsearch/reference/current/targz.html
   wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-darwin-x86_64.tar.gz
   wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-darwin-x86_64.tar.gz.sha512
   shasum -a 512 -c elasticsearch-7.10.2-darwin-x86_64.tar.gz.sha512
   tar -xzf elasticsearch-7.10.2-darwin-x86_64.tar.gz
   rm elasticsearch-7.10.2-linux-x86_64.tar.gz elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
   ```

   To run the server,

   ```sh
   cd elasticsearch-7.10.2/
   ./bin/elasticsearch # start the server
   pkill -f elasticsearch # to stop the server
   ```

   </details>

   <details>
   <summary>Linux (wget)</summary>

   ```sh
   # source: https://www.elastic.co/guide/en/elasticsearch/reference/8.1/targz.html
   wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz
   wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
   shasum -a 512 -c elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
   tar -xzf elasticsearch-7.10.2-linux-x86_64.tar.gz
   rm elasticsearch-7.10.2-linux-x86_64.tar.gz elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
   ```

   To run the server,

   ```sh
   cd elasticsearch-7.10.2/
   ./bin/elasticsearch # start the server
   pkill -f elasticsearch # to stop the server
   ```

   </details>

2. Download the datasets using the following command. The downloaded files will be stored in the [raw_data](raw_data) directory.

   ```sh
   bash scripts/download_raw_data.sh
   ```

3. Build indices for the downloaded datasets in Elasticsearch. First, ensure that Elasticsearch is running in the background. Then, run the following command:

   ```sh
   python -m totr.retriever.build_es_index --all
   ```

   You can also choose to build indices for specific datasets. For example, using the following command:

   ```sh
   python -m totr.retriever.build_es_index --datasets hotpotqa multihoprag
   ```

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

- [`scripts/sbatch_vllm.sh`](scripts/sbatch_vllm.sh): Script for serving the vLLM server on Slurm cluster. See this [section](#running-llm-server-on-slurm-cluster) for example usage.

## Running LLM server on Slurm cluster

1. Log in to your Slurm login node.

2. Clone this repository and set up the environment with the following commands:

   ```sh
   # Clone this repository
   git clone https://github.com/kaiitunnz/totr.git
   cd totr

   # Create and activate a Conda environment
   conda create -n totr python=3.11 -y
   conda activate totr

   # Install the dependencies for running an LLM server
   pip install poetry
   poetry install --only vllm
   ```

3. Submit a batch job using the following command. You may need to set appropriate arguments for sbatch in [`scripts/sbatch_vllm.sh`](scripts/sbatch_vllm.sh).

   ```sh
   sbatch scripts/sbatch_vllm.sh
   ```

4. Check your allocated node with the following command:

   ```sh
   squeue -u $USER
   ```

5. Log out from the Slurm login node and start ssh tunneling with the following command:

   ```sh
   ssh -L 8010:<gpu-node>:8010 <user>@<login-node-address>
   ```

   or in the background with the following command (in this case, you need to kill the process by yourself):

   ```sh
   ssh -fN -L 8010:<gpu-node>:8010 <user>@<login-node-address>
   ```

6. Now you can run benchmarking scripts or connect to the LLM server from your local host at the following address: `http://localhost:8010`.

7. To stop the LLM server before it is timed out, run the following command with the appropriate `job-id` obtained from the `squeue` command.

   ```sh
   scancel <job-id>
   ```

## Running benchmarks

Run the following command:

```sh
python benchmark/bench.py --verbose
```

The benchmark results will be saved to the [`results`](results) directory.
