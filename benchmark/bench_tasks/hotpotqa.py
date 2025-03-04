import asyncio
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Awaitable, Dict, List, Tuple, Union

import jsonlines
import tqdm
from base import BenchmarkResult, QAMixin


def load_hotpotqa(path: Union[str, Path]) -> List[Dict[str, Any]]:
    with jsonlines.open(path) as reader:
        data = [item for item in reader]
    return data


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def update_metrics(
    metrics: Dict[str, float], prediction: str, ground_truth: str
) -> None:
    em = exact_match_score(prediction, ground_truth)
    f1, prec, recall = f1_score(prediction, ground_truth)
    metrics["em"] += float(em)
    metrics["f1"] += f1
    metrics["prec"] += prec
    metrics["recall"] += recall


async def run_hotpotqa(
    qa_system: QAMixin, dataset_root_dir: Union[str, Path], verbose: bool = True
) -> BenchmarkResult:
    async def answer_func(item: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        answer = await qa_system.answer(item["question_text"])
        return item, answer

    batch_size = 16
    bench_name = "hotpotqa"
    preds = {}
    metrics: Dict[str, float] = {"em": 0, "f1": 0, "prec": 0, "recall": 0}
    data = load_hotpotqa(Path(dataset_root_dir, bench_name, "test_subsampled.jsonl"))

    pbar = tqdm.tqdm(total=len(data), disable=not verbose)
    task: Awaitable[Tuple[Dict[str, Any], str]]
    for i in range(0, len(data), batch_size):
        for task in asyncio.as_completed(
            [answer_func(item) for item in data[i : i + batch_size]]
        ):
            item, answer = await task
            preds[item["question_id"]] = answer
            ground_truth = item["answers_objects"][0]["spans"][0]
            update_metrics(metrics, answer, ground_truth)
            pbar.update(1)

    n = len(data)
    for k in metrics.keys():
        metrics[k] /= n

    result = BenchmarkResult(bench_name, preds, metrics)
    return result
