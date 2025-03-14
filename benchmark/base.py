import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import jsonlines


class QAMixin(Protocol):
    async def answer(self, question: str) -> str: ...


class ResultHandler:
    def __init__(
        self,
        system_name: str,
        result_dir: Optional[Path] = None,
        save_results: bool = True,
        overwrite: bool = False,
    ) -> None:
        if save_results and result_dir is None:
            raise ValueError("result_dir must not be None if save_results is True")

        self.system_name = system_name
        self.save_results = save_results
        self.overwrite = overwrite

        self._result_dir = result_dir
        self._bench_name: Optional[str] = None
        self._predictions: List[Any] = []
        self._metrics: Optional[Dict[str, Any]] = None

    @property
    def bench_name(self) -> str:
        if self._bench_name is None:
            raise ValueError("Bench name has not been set")
        return self._bench_name

    @property
    def predictions(self) -> List[Any]:
        return self._predictions

    @property
    def metrics(self) -> Dict[str, Any]:
        if self._metrics is None:
            raise ValueError("Metrics has not been set")
        return self._metrics

    @property
    def result_dir(self) -> Path:
        if self._result_dir is None:
            raise ValueError("Result directory has not been set")
        return self._result_dir

    @property
    def bench_result_dir(self) -> Path:
        return self.result_dir / self.bench_name

    def set_bench_name(self, bench_name: str) -> None:
        self._bench_name = bench_name

    def add_prediction(self, pred: Any) -> None:
        self._predictions.append(pred)
        if not self.save_results:
            return

        bench_dir = self.bench_result_dir
        bench_dir.mkdir(exist_ok=True)

        pred_file = bench_dir / f"{self.system_name}_preds.jsonl"

        with jsonlines.open(pred_file, "a") as f:
            f.write(pred)

    def set_metrics(self, metrics: Dict[str, Any]) -> None:
        self._metrics = metrics
        if not self.save_results:
            return

        bench_dir = self.bench_result_dir
        bench_dir.mkdir(exist_ok=True)

        metrics_file = bench_dir / f"{self.system_name}_metrics.json"

        if not self.overwrite:
            if metrics_file.exists():
                user_inp = input("File exists. Do you wish to overwrite? ([y], n)")
                if user_inp and user_inp.lower() != "y":
                    return

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
