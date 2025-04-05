import asyncio
from argparse import ArgumentParser, Namespace
from pathlib import Path

from bench_tasks.hotpotqa import run_hotpotqa
from bench_tasks.multihoprag import run_multihoprag
from bench_tasks.musique import run_musique
from transformers.utils import logging

from base import ResultHandler
from totr.config import Config
from totr import SCR


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--dataset-dir", type=Path, default="datasets")
    parser.add_argument("--result-dir", type=Path, default="results")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


async def benchmark_num_paths(
    model_name: str,
    dataset: str,
    dataset_dir: Path,
    result_dir: Path,
    verbose: bool,
    overwrite: bool,
    seed: int,
) -> None:
    base_config_path = Path("configs").resolve()
    config = Config.from_json(
        base_config_path.joinpath(dataset, model_name.split("/")[-1] + ".json")
    )
    system_name = f"scr_{config.identifier}"

    batch_size = 4
    num_chains = [3, 5, 8, 11, 15]

    if dataset == "hotpotqa":
        benchmark_func = run_hotpotqa
    elif dataset == "multihoprag":
        benchmark_func = run_multihoprag
    elif dataset == "musique":
        benchmark_func = run_musique
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print("Evaluating the system on HotpotQA (varying numbers of paths)...")
    for num_chain in num_chains:
        print(f">> Evaluating {system_name} with {num_chain} paths...")
        try:
            result_handler = ResultHandler(
                system_name + f"_{num_chain}paths",
                dataset + "_ablation_num_paths",
                result_dir,
                test=True,
                save_results=True,
                overwrite=overwrite,
            )
        except FileExistsError:
            print(">> Results already exist. Skipped.")
            continue
        config.scr.num_chains = num_chain
        system = SCR(config, seed=seed)
        await benchmark_func(system, dataset_dir, result_handler, batch_size, verbose)
        print(f">> Results: {result_handler.metrics}")


async def main(
    dataset: str,
    dataset_dir: Path,
    result_dir: Path,
    verbose: bool,
    overwrite: bool,
    seed: int,
) -> None:
    logging.set_verbosity(40)

    dataset_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name = "Qwen/Qwen2.5-7B-Instruct"

    # 1. Varying numbers of paths
    await benchmark_num_paths(
        model_name, dataset, dataset_dir, result_dir, verbose, overwrite, seed
    )


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            args.dataset,
            args.dataset_dir,
            args.result_dir,
            args.verbose,
            args.overwrite,
            args.seed,
        )
    )
