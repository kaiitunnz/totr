<!-- omit in toc -->

# Tree-of-Thought Retrieval: Enhancing Multi-Hop Question-Answering Beyond Chain-of-Thought

This repository provides the implementation of Tree-of-Thought Retrieval (ToTR) and Self-Consistency Retrieval (SCR), two novel frameworks designed to enhance retrieval-augmented generation for knowledge-intensive multi-hop question answering tasks. By exploring diverse reasoning paths, these methods aim to improve retrieval coverage and robustness, addressing limitations in existing approaches.

<!-- omit in toc -->

## Table of Contents

- [Instructions for Reproducing the Paper Results](#instructions-for-reproducing-the-paper-results)
- [Installation](#installation)
- [Preparing Database for RAG](#preparing-database-for-rag)
- [Project Structure](#project-structure)
- [Utility scripts](#utility-scripts)
- [Running LLM server on Slurm cluster](#running-llm-server-on-slurm-cluster)
- [Running benchmarks](#running-benchmarks)

## Instructions for Reproducing the Paper Results

This section describes how to reproduce the results presented in the paper in detail.

**Important:** Reproducing the results from scratch involves downloading the corpora used by the datasets, setting up a vector database (Elasticsearch), ingesting the corpora into the vector database, and running inference on relatively large LMs, which may take up to a few days in total. Moreover, you need to download and run the LLMs and, thus, need to make sure you have sufficient disk storage and GPU memory to store the models. If these constraints prevent you from running the code, we encourage you to download raw predictions and results and inspect them instead. You can use the following command to download the results and skip to step 7. You may also need to install necessary dependencies first (see [Installation](#installation)).

```sh
bash scripts/download_results.sh
```

To reproduce the paper results,

1. Install the dependencies following the [Installation](#installation) section. You also need to install the vllm dependency group, as our config files assume that you use [vLLM's OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).
2. Set up a vector database (Elasticsearch) following the [Preparing Database for RAG](#preparing-database-for-rag) section. We recommend you to build Elasticsearch indices only for the datasets to be used, namely, hotpotqa, multihoprag, and musique.
3. Run Elasticsearch in the background. (This should already be done when preparing the database.)
4. Run vLLM's OpenAI-compatible server in the server using the following command:
   ```sh
   bash scripts/serve_vllm.sh
   ```
5. Run the benchmark code following the [Running benchmarks](#running-benchmarks) section. Note that in order to use a different LM, you have to modify the [scripts/serve_vllm.sh](scripts/serve_vllm.sh) file to use the desired LM and change the model name in [benchmark/bench.py](benchmark/bench.py). The prediction and performance results will be saved to the [results](results) directory.
6. You may also run the code for the ablation study using the following command
   ```sh
   python benchmark/ablation.py --verbose
   ```
   Note that you also need to modify the model name in the [benchmark/ablation.py](benchmark/ablation.py) manually for different models to be evaluated.
7. Run the Python notebook, [notebooks/results.ipynb](notebooks/results.ipynb), to generate plots for the paper results. Note that since ToTR, SCR, and ReAct employ temperature sampling and asynchronous execution, it is very complicated to obtain deterministic results. Therefore, your reproduced results may be slightly different from the ones in the paper.

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

6. Set up a RAG database following the [Preparing Database for RAG](#preparing-database-for-rag) section.

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
   python -m totr.retriever.build_es_index --datasets hotpotqa multihoprag musique
   ```

## Project Structure

- [`benchmark`](benchmark): Benchmarking tasks, utility code, and baselines' implementation.
- [`configs`](configs): Configuration files for different models, datasets, and systems.
- [`datasets`](datasets): Dataset-related files.
- [`notebooks`](notebooks): Useful notebooks, for example, [results.ipynb](notebooks/results.ipynb), which generates plots for the paper results.
- [`prompts`](prompts): Collection of prompts used in the experiments.
- [`results`](results): Benchmark results.
- [`scripts`](scripts): Utility scripts for developing and running experiments.
- [`src`](src): ToTR's implementation.
- [`tests`](tests): Various test cases. Currently contains only basic tests without unit testing. It can be a reference for how different functions and classes are used.

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
python benchmark/bench.py --verbose --test
```

The benchmark results will be saved to the [`results`](results) directory. You may omit the `--test`flag if you want to perform validation instead.
