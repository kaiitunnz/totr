from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class QAConfig:
    answer_mode: Literal["direct", "cot"] = "direct"
    cot_question_prefix: str = (
        "Answer the following question by reasoning step-by-step.\n"
    )
    direct_question_prefix: str = "Answer the following question.\n"
    react_question_prefix: str = (
        "Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be two types: \n(1) Search[query], which searches the database with the query.\n(2) Finish[answer], which returns the answer and finishes the task.\nHere are some examples.\n"
    )
    cot_answer_regex: Optional[str] = None
    direct_answer_regex: Optional[str] = None
    remove_full_stop: bool = True
    use_retriever_answer: bool = False

    @property
    def answer_regex(self) -> Optional[str]:
        if self.answer_mode == "cot":
            return self.cot_answer_regex
        return self.direct_answer_regex
