import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Tuple, Union

import tqdm
from base import QAMixin, ResultHandler

from totr.ir import QAModel
from totr.retriever import ElasticsearchRetriever, RetrieverRegistry

from .metrics import update_metrics


def load_multihoprag(path: Union[str, Path]) -> List[Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    return data


async def _run_multihoprag(
    answer_func: Callable[
        [Dict[str, Any]], Coroutine[Any, Any, Tuple[Dict[str, Any], str]]
    ],
    dataset_root_dir: Union[str, Path],
    result_handler: ResultHandler,
    batch_size: int = 1,
    verbose: bool = True,
) -> None:
    metrics: Dict[str, float] = {"em": 0, "f1": 0, "prec": 0, "recall": 0}
    dataset_dir = Path(dataset_root_dir, "multihoprag")
    if result_handler.test:
        data = load_multihoprag(dataset_dir / "multihoprag_test.json")
    else:
        data = load_multihoprag(dataset_dir / "multihoprag_dev.json")
    n = len(data)

    # Check existing predictions
    data_dict = {d["query_id"]: d for d in data}
    for pred in result_handler.predictions:
        del data_dict[pred["query_id"]]
        update_metrics(metrics, pred["prediction"], pred["ground_truth"])
    data = list(data_dict.values())

    pbar = tqdm.tqdm(total=len(data), disable=not verbose)
    for i in range(0, len(data), batch_size):
        tasks = [
            asyncio.create_task(answer_func(item)) for item in data[i : i + batch_size]
        ]
        for task in tasks:
            item, answer = await task
            ground_truth = item["answer"]
            update_metrics(metrics, answer, ground_truth)
            pbar.update(1)
            result_handler.add_prediction(
                {
                    "query_id": item["query_id"],
                    "prediction": answer,
                    "ground_truth": ground_truth,
                }
            )

    for k in metrics.keys():
        metrics[k] /= n

    result_handler.set_metrics(metrics)


async def run_multihoprag(
    qa_system: QAMixin,
    dataset_root_dir: Union[str, Path],
    result_handler: ResultHandler,
    batch_size: int = 1,
    verbose: bool = True,
) -> None:
    async def answer_func(item: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        answer = await qa_system.answer(item["query"])
        return item, answer

    await _run_multihoprag(
        answer_func, dataset_root_dir, result_handler, batch_size, verbose
    )


async def run_multihoprag_oracle(
    qa_model: QAModel,
    dataset_root_dir: Union[str, Path],
    result_handler: ResultHandler,
    batch_size: int = 1,
    verbose: bool = True,
) -> None:
    config = qa_model.config
    retriever = RetrieverRegistry.get(config.retriever.retriever_name, config)
    if not isinstance(retriever, ElasticsearchRetriever):
        raise ValueError("Only ElasticsearchRetriever is supported for now")

    async def answer_func(item: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        evidence_list = item["evidence_list"]
        titles = []
        paragraphs = []
        for evidence in evidence_list:
            title = evidence["title"]
            fact = evidence["fact"].lower()
            retrieved = await retriever.retrieve_paragraphs_with_title(
                title, "multihoprag", max_retrieval_count=100
            )
            for item in retrieved:
                if fact in item["paragraph_text"].lower():
                    titles.append(title)
                    paragraphs.append(item["paragraph_text"])
                    break
            else:
                raise ValueError("No relevant paragraphs found")

        answer = await qa_model.answer(item["query"], titles, paragraphs)
        return item, answer

    await _run_multihoprag(
        answer_func, dataset_root_dir, result_handler, batch_size, verbose
    )
