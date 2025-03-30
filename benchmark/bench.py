import asyncio
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterable, Tuple

from base import QAMixin, ResultHandler
from bench_tasks.hotpotqa import run_hotpotqa
from ircot import IRCoT
from react.config import ReActFullConfig
from react.react import ReAct
from baseline.baseline import Base
from transformers.utils import logging

from totr import SCR, ToTR
from totr.config import Config


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default="datasets")
    parser.add_argument("--result-dir", type=Path, default="results")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--test", "-t", action="store_true")
    return parser.parse_args()


async def evaluate_hotpotqa(
    model_name: str,
    dataset_dir: Path,
    result_dir: Path,
    test: bool,
    verbose: bool,
    overwrite: bool,
) -> None:
    def systems(model_name: str) -> Iterable[Tuple[str, QAMixin]]:
        base_config_path = Path("configs").resolve()
        config = Config.from_json(
            base_config_path.joinpath("hotpotqa", model_name.split("/")[-1] + ".json")
        )
        # 1. IRCoT
        yield f"ircot_{config.identifier}", IRCoT(config)
        # 2. ToTR
        yield f"totr_{config.identifier}", ToTR(config, seed=0)
        # 3. SCR
        yield f"scr_{config.identifier}", SCR(config, seed=0)
        # 4. ReAct
        config = ReActFullConfig.from_json(
            base_config_path.joinpath("hotpotqa", model_name.split("/")[-1] + ".json")
        )
        yield f"react_{config.identifier}", ReAct(config)
        # 5. Baseline (No retriever)
        config = Config.from_json(
            base_config_path.joinpath("hotpotqa", model_name.split("/")[-1] + "_no_context.json")
        )
        yield f"baseline_no_retrieval_CoT_{config.identifier}", Base(config)
        # 6. Baseline (One retrieval)
        config = Config.from_json(
            base_config_path.joinpath("hotpotqa", model_name.split("/")[-1] + ".json")
        )
        yield f"baseline_single_retrieval_CoT_{config.identifier}", Base(config, with_retrieval=True)

    bench_name = "hotpotqa"
    batch_sizes = {IRCoT: 8, ToTR: 1, SCR: 4, ReAct: 1, Base: 1}

    print("Evaluating systems on HotpotQA...")
    for system_name, system in systems(model_name):
        print(f">> Evaluating {system_name}...")
        batch_size = batch_sizes[type(system)]
        try:
            result_handler = ResultHandler(
                system_name,
                bench_name,
                result_dir,
                test=test,
                save_results=True,
                overwrite=overwrite,
            )
        except FileExistsError:
            print(">> Results already exist. Skipped.")
            continue
        await run_hotpotqa(system, dataset_dir, result_handler, batch_size, verbose)
        print(f">> Results: {result_handler.metrics}")


async def main(
    dataset_dir: Path, result_dir: Path, test: bool, verbose: bool, overwrite: bool
) -> None:
    logging.set_verbosity(40)

    dataset_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    model_names = [
        # "google/flan-t5-large",
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        # "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
        # "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ]

    for model_name in model_names:
        # 1. HotpotQA
        await evaluate_hotpotqa(
            model_name, dataset_dir, result_dir, test, verbose, overwrite
        )


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(args.dataset_dir, args.result_dir, args.test, args.verbose, args.overwrite)
    )
