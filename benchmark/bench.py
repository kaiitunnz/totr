import asyncio
from argparse import ArgumentParser, Namespace
from pathlib import Path

from bench_tasks.hotpotqa import run_hotpotqa
from ircot import IRCoT
from transformers.utils import logging

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
    # 1. IRCoT
    system = IRCoT(config)
    system_name = f"ircot_{config.identifier}"
    result = await run_hotpotqa(system, dataset_dir, verbose)
    result.save_to(result_dir, system_name, overwrite)


async def main(
    dataset_dir: Path, result_dir: Path, verbose: bool, overwrite: bool
) -> None:
    logging.set_verbosity(40)

    dataset_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    model_names = [
        "google/flan-t5-large",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B",
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
