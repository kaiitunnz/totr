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
        bench_name: str,
        result_dir: Optional[Path] = None,
        test: bool = False,
        save_results: bool = True,
        overwrite: bool = False,
    ) -> None:
        if save_results and result_dir is None:
            raise ValueError("result_dir must not be None if save_results is True")

        self.system_name = system_name
        self.test = test
        self.bench_name = bench_name if test else bench_name + "_dev"
        self.save_results = save_results
        self.overwrite = overwrite

        self._result_dir = result_dir
        self._predictions: List[Any] = []
        self._metrics: Optional[Dict[str, Any]] = None

        self._check_result_dir()

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

    @property
    def metric_file_path(self) -> Path:
        return self.bench_result_dir / f"{self.system_name}_metrics.json"

    @property
    def pred_file_path(self) -> Path:
        return self.bench_result_dir / f"{self.system_name}_preds.jsonl"

    def _check_result_dir(self) -> None:
        if len(self.predictions) > 0:
            raise ValueError(
                "Predictions have already been updated. "
                "Please ensure to call this function only once during initialization."
            )

        self.bench_result_dir.mkdir(exist_ok=True)

        if self.overwrite:
            if self.save_results and self.pred_file_path.exists():
                self.pred_file_path.unlink()
            return

        if self.save_results and self.metric_file_path.exists():
            user_inp = input("Result file exists. Do you wish to overwrite? ([y], n): ")
            if user_inp and user_inp.lower() != "y":
                raise FileExistsError()

        if self.pred_file_path.exists():
            with jsonlines.open(self.pred_file_path, "r") as reader:
                self._predictions.extend(reader)
            print(
                f"Prediction file exists. {len(self._predictions)} predictions found. "
                "Continue with the remaining samples."
            )

    def add_prediction(self, pred: Any) -> None:
        self._predictions.append(pred)
        if not self.save_results:
            return

        with jsonlines.open(self.pred_file_path, "a") as writer:
            writer.write(pred)

    def set_metrics(self, metrics: Dict[str, Any]) -> None:
        self._metrics = metrics
        if not self.save_results:
            return

        with open(self.metric_file_path, "w") as f:
            json.dump(metrics, f, indent=2)
