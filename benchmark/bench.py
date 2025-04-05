import asyncio
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterable, Tuple, Union

from base import QAMixin, ResultHandler
from baseline.baseline import BaseRAG
from bench_tasks.hotpotqa import run_hotpotqa, run_hotpotqa_oracle
from bench_tasks.multihoprag import run_multihoprag, run_multihoprag_oracle
from bench_tasks.musique import run_musique, run_musique_oracle
from ircot import IRCoT
from react.config import ReActFullConfig
from react.react import ReAct
from transformers.utils import logging

from totr import SCR, ToTR
from totr.config import Config
from totr.ir import QAModel


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default="datasets")
    parser.add_argument("--result-dir", type=Path, default="results")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--test", "-t", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


async def evaluate_hotpotqa(
    model_name: str,
    dataset_dir: Path,
    result_dir: Path,
    test: bool,
    verbose: bool,
    overwrite: bool,
    seed: int,
) -> None:
    def systems(model_name: str) -> Iterable[Tuple[str, Union[QAMixin, QAModel]]]:
        base_config_path = Path("configs").resolve()
        config = Config.from_json(
            base_config_path.joinpath("hotpotqa", model_name.split("/")[-1] + ".json")
        )
        # 1. No RAG
        yield f"norag_{config.identifier}", BaseRAG(
            config, with_retrieval=False, seed=seed
        )
        # 2. One-shot RAG
        yield f"oneshotrag_{config.identifier}", BaseRAG(
            config, with_retrieval=True, seed=seed
        )
        # 3. IRCoT
        yield f"ircot_{config.identifier}", IRCoT(config)
        # 4. ReAct
        react_config = ReActFullConfig.from_json(
            base_config_path.joinpath("hotpotqa", model_name.split("/")[-1] + ".json")
        )
        yield f"react_{react_config.identifier}", ReAct(react_config)
        # 5. SCR
        yield f"scr_{config.identifier}", SCR(config, seed=seed)
        # 6. ToTR
        yield f"totr_{config.identifier}", ToTR(config, seed=seed)
        # 7. Oracle
        yield f"oracle_{config.identifier}", QAModel(config, with_retrieval=True)

    bench_name = "hotpotqa"
    batch_sizes = {BaseRAG: 16, IRCoT: 8, SCR: 4, ReAct: 4, ToTR: 1, QAModel: 16}

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
        if isinstance(system, QAModel):
            await run_hotpotqa_oracle(
                system, dataset_dir, result_handler, batch_size, verbose
            )
        else:
            await run_hotpotqa(system, dataset_dir, result_handler, batch_size, verbose)
        print(f">> Results: {result_handler.metrics}")


async def evaluate_multihoprag(
    model_name: str,
    dataset_dir: Path,
    result_dir: Path,
    test: bool,
    verbose: bool,
    overwrite: bool,
    seed: int,
) -> None:
    def systems(model_name: str) -> Iterable[Tuple[str, Union[QAMixin, QAModel]]]:
        base_config_path = Path("configs").resolve()
        config = Config.from_json(
            base_config_path.joinpath(
                "multihoprag", model_name.split("/")[-1] + ".json"
            )
        )
        # 1. No RAG
        yield f"norag_{config.identifier}", BaseRAG(
            config, with_retrieval=False, seed=seed
        )
        # 2. One-shot RAG
        yield f"oneshotrag_{config.identifier}", BaseRAG(
            config, with_retrieval=True, seed=seed
        )
        # 3. IRCoT
        yield f"ircot_{config.identifier}", IRCoT(config)
        # 4. ReAct
        react_config = ReActFullConfig.from_json(
            base_config_path.joinpath(
                "multihoprag", model_name.split("/")[-1] + ".json"
            )
        )
        yield f"react_{react_config.identifier}", ReAct(react_config)
        # 5. SCR
        yield f"scr_{config.identifier}", SCR(config, seed=seed)
        # 6. ToTR
        yield f"totr_{config.identifier}", ToTR(config, seed=seed)
        # 7. Oracle
        yield f"oracle_{config.identifier}", QAModel(config, with_retrieval=True)

    bench_name = "multihoprag"
    batch_sizes = {BaseRAG: 16, IRCoT: 8, SCR: 4, ReAct: 4, ToTR: 1, QAModel: 16}

    print("Evaluating systems on Multihop-RAG...")
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
        if isinstance(system, QAModel):
            await run_multihoprag_oracle(
                system, dataset_dir, result_handler, batch_size, verbose
            )
        else:
            await run_multihoprag(
                system, dataset_dir, result_handler, batch_size, verbose
            )
        print(f">> Results: {result_handler.metrics}")


async def evaluate_musique(
    model_name: str,
    dataset_dir: Path,
    result_dir: Path,
    test: bool,
    verbose: bool,
    overwrite: bool,
    seed: int,
) -> None:
    def systems(model_name: str) -> Iterable[Tuple[str, Union[QAMixin, QAModel]]]:
        base_config_path = Path("configs").resolve()
        config = Config.from_json(
            base_config_path.joinpath("musique", model_name.split("/")[-1] + ".json")
        )
        # 1. No RAG
        yield f"norag_{config.identifier}", BaseRAG(
            config, with_retrieval=False, seed=seed
        )
        # 2. One-shot RAG
        yield f"oneshotrag_{config.identifier}", BaseRAG(
            config, with_retrieval=True, seed=seed
        )
        # 3. IRCoT
        yield f"ircot_{config.identifier}", IRCoT(config)
        # 4. ReAct
        react_config = ReActFullConfig.from_json(
            base_config_path.joinpath("musique", model_name.split("/")[-1] + ".json")
        )
        yield f"react_{react_config.identifier}", ReAct(react_config)
        # 5. SCR
        yield f"scr_{config.identifier}", SCR(config, seed=seed)
        # 6. ToTR
        yield f"totr_{config.identifier}", ToTR(config, seed=seed)
        # 7. Oracle
        yield f"oracle_{config.identifier}", QAModel(config, with_retrieval=True)

    bench_name = "musique"
    batch_sizes = {BaseRAG: 16, IRCoT: 8, SCR: 4, ReAct: 4, ToTR: 1, QAModel: 16}

    print("Evaluating systems on MuSiQue...")
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
        if isinstance(system, QAModel):
            await run_musique_oracle(
                system, dataset_dir, result_handler, batch_size, verbose
            )
        else:
            await run_musique(system, dataset_dir, result_handler, batch_size, verbose)
        print(f">> Results: {result_handler.metrics}")


async def main(
    dataset_dir: Path,
    result_dir: Path,
    test: bool,
    verbose: bool,
    overwrite: bool,
    seed: int,
) -> None:
    logging.set_verbosity(40)

    dataset_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    model_names = [
        "meta-llama/Llama-3.1-8B-Instruct",
        # "meta-llama/Meta-Llama-3.1-8B-Instruct",
        # "Qwen/Qwen2.5-7B-Instruct",
        # "meta-llama/Llama-3.2-1B-Instruct",
        # "meta-llama/Llama-3.2-3B-Instruct",
    ]

    for model_name in model_names:
        # 1. HotpotQA
        await evaluate_hotpotqa(
            model_name, dataset_dir, result_dir, test, verbose, overwrite, seed
        )

        # 2. Multihop-RAG
        await evaluate_multihoprag(
            model_name, dataset_dir, result_dir, test, verbose, overwrite, seed
        )

        # 3. MuSiQue
        await evaluate_musique(
            model_name, dataset_dir, result_dir, test, verbose, overwrite, seed
        )


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            args.dataset_dir,
            args.result_dir,
            args.test,
            args.verbose,
            args.overwrite,
            args.seed,
        )
    )
