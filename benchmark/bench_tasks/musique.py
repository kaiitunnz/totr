import asyncio
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import jsonlines
import tqdm
from base import QAMixin, ResultHandler

from .metrics import update_metrics


def load_musique(path: Union[str, Path]) -> List[Dict[str, Any]]:
    with jsonlines.open(path) as reader:
        data = [item for item in reader]
    return data


async def run_musique(
    qa_system: QAMixin,
    dataset_root_dir: Union[str, Path],
    result_handler: ResultHandler,
    batch_size: int = 1,
    verbose: bool = True,
) -> None:
    async def answer_func(item: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        answer = await qa_system.answer(item["question_text"])
        return item, answer

    metrics: Dict[str, float] = {"em": 0, "f1": 0, "prec": 0, "recall": 0}
    dataset_dir = Path(dataset_root_dir, "musique")
    if result_handler.test:
        data = load_musique(dataset_dir / "musique_test.jsonl")
    else:
        data = load_musique(dataset_dir / "musique_dev.jsonl")
    n = len(data)

    # Check existing predictions
    data_dict = {d["question_id"]: d for d in data}
    for pred in result_handler.predictions:
        del data_dict[pred["question_id"]]
        update_metrics(metrics, pred["prediction"], pred["ground_truth"])
    data = list(data_dict.values())

    pbar = tqdm.tqdm(total=len(data), disable=not verbose)
    for i in range(0, len(data), batch_size):
        tasks = [
            asyncio.create_task(answer_func(item)) for item in data[i : i + batch_size]
        ]
        for task in tasks:
            item, answer = await task
            ground_truth = item["answers_objects"][0]["spans"][0]
            update_metrics(metrics, answer, ground_truth)
            pbar.update(1)
            result_handler.add_prediction(
                {
                    "question_id": item["question_id"],
                    "prediction": answer,
                    "ground_truth": ground_truth,
                }
            )

    for k in metrics.keys():
        metrics[k] /= n

    result_handler.set_metrics(metrics)
