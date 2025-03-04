import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Protocol


class QAMixin(Protocol):
    async def answer(self, question: str) -> str: ...


@dataclass
class BenchmarkResult:
    bench_name: str
    predictions: Any
    metrics: Dict[str, Any]

    def save_to(self, path: Path, system_name: str, overwrite: bool) -> None:
        bench_dir = path / self.bench_name
        bench_dir.mkdir(exist_ok=True)

        pred_file = bench_dir / f"{system_name}_preds.json"
        metrics_file = bench_dir / f"{system_name}_metrics.json"

        if not overwrite:
            if pred_file.exists() or metrics_file.exists():
                user_inp = input("File exists. Do you wish to overwrite? ([y], n)")
                if user_inp and user_inp.lower() != "y":
                    return

        with open(pred_file, "w") as f:
            json.dump(self.predictions, f)

        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f)
