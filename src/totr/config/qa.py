from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class QAConfig:
    answer_mode: Literal["direct", "cot"] = "direct"
    cot_question_prefix: str = (
        "Answer the following question by reasoning step-by-step.\n"
    )
    direct_question_prefix: str = "Answer the following question.\n"
    answer_regex: Optional[str] = None
    remove_full_stop: bool = True
