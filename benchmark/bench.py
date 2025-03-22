import asyncio
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterable, Tuple

from base import QAMixin, ResultHandler
from bench_tasks.hotpotqa import run_hotpotqa
from ircot import IRCoT
from transformers.utils import logging

from totr import SCR, ReAct, ToTR
from totr.config import Config


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default="datasets")
    parser.add_argument("--result-dir", type=Path, default="results")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


async def evaluate_hotpotqa(
    config: Config, dataset_dir: Path, result_dir: Path, verbose: bool, overwrite: bool
) -> None:
    def systems() -> Iterable[Tuple[str, QAMixin]]:
        # 1. IRCoT
        yield f"ircot_{config.identifier}", IRCoT(config)
        # 2. ToTR
        yield f"totr_{config.identifier}", ToTR(config, seed=0)
        # 3. SCR
        yield f"scr_{config.identifier}", SCR(config, seed=0)
        # 4. ReAct
        yield f"ReAct_{config.identifier}", ReAct(config)

    bench_name = "hotpotqa"
    batch_sizes = {IRCoT: 16, ToTR: 1, SCR: 4, ReAct: 1}

    print("Evaluating systems on HotpotQA...")
    for system_name, system in systems():
        print(f">> Evaluating {system_name}...")
        batch_size = batch_sizes[type(system)]
        try:
            result_handler = ResultHandler(
                system_name,
                bench_name,
                result_dir,
                save_results=True,
                overwrite=overwrite,
            )
        except FileExistsError:
            print(">> Results already exist. Skipped.")
            continue
        await run_hotpotqa(system, dataset_dir, result_handler, batch_size, verbose)
        print(f">> Results: {result_handler.metrics}")


async def main(
    dataset_dir: Path, result_dir: Path, verbose: bool, overwrite: bool
) -> None:
    logging.set_verbosity(40)

    dataset_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    model_names = [
        # "google/flan-t5-large",
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        # "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ]
    base_config_path = Path("configs").resolve()

    for model_name in model_names:
        # 1. HotpotQA
        config = Config.from_json(
            base_config_path.joinpath("hotpotqa", model_name.split("/")[-1] + ".json")
        )
        await evaluate_hotpotqa(config, dataset_dir, result_dir, verbose, overwrite)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.dataset_dir, args.result_dir, args.verbose, args.overwrite))
