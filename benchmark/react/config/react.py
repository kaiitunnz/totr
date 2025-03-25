from dataclasses import dataclass, field
from typing import Dict, Optional

from totr.config.generation import GenerationConfig


@dataclass
class ReActConfig:
    react_prompt_filename: str
    max_step: int = 5
    react_question_prefix: str = (
        "Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be two types: \n(1) Search[query], which searches the database with the query.\n(2) Finish[answer], which returns the answer and finishes the task.\nHere are some examples.\n"
    )
    retriever_gen_config_dict: Optional[Dict] = None
    retriever_gen_config: GenerationConfig = field(init=False)

    def __post_init__(self) -> None:
        if self.retriever_gen_config_dict is None:
            self.retriever_gen_config = GenerationConfig()
        else:
            self.retriever_gen_config = GenerationConfig(
                **self.retriever_gen_config_dict
            )
