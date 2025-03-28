import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from totr.config import Config
from totr.config.generation import GenerationConfig
from totr.config.llm import LLMConfig
from totr.config.prompt import PromptConfig
from totr.config.qa import QAConfig
from totr.config.retriever import RetrieverConfig
from totr.config.scr import SCRConfig
from totr.config.totr import ToTRConfig


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


@dataclass
class ReActFullConfig(Config):
    react: ReActConfig

    @classmethod
    def from_json(
        cls,
        path_or_json: Union[
            Union[str, Path], Dict[str, Dict[str, Any]]
        ] = "config.json",
    ) -> "ReActFullConfig":
        if isinstance(path_or_json, dict):
            obj = path_or_json
        else:
            with open(path_or_json) as f:
                obj = json.load(f)

        llm_config = LLMConfig(**obj["llm"])

        generation_kwargs: Dict[str, Any] = (
            dict(model=llm_config.model) | obj["generation"]
        )
        generation_config = GenerationConfig(**generation_kwargs)

        retriever_config = RetrieverConfig(**obj["retriever"])
        prompt_config = PromptConfig(**obj["prompt"])
        qa_config = QAConfig(**obj["qa"])

        totr_retriever_generation_dict: Dict[str, Any] = (
            dict(model=llm_config.model) | obj["totr"]["retriever_gen_config_dict"]
        )
        obj["totr"]["retriever_gen_config_dict"] = totr_retriever_generation_dict
        totr_config = ToTRConfig(**obj["totr"])

        scr_retriever_generation_dict: Dict[str, Any] = (
            dict(model=llm_config.model) | obj["scr"]["retriever_gen_config_dict"]
        )
        obj["scr"]["retriever_gen_config_dict"] = scr_retriever_generation_dict
        scr_config = SCRConfig(**obj["scr"])

        react_retriever_generation_dict: Dict[str, Any] = (
            dict(model=llm_config.model) | obj["react"]["retriever_gen_config_dict"]
        )
        obj["react"]["retriever_gen_config_dict"] = react_retriever_generation_dict
        react_config = ReActConfig(**obj["react"])

        return cls(
            llm=llm_config,
            generation=generation_config,
            retriever=retriever_config,
            prompt=prompt_config,
            qa=qa_config,
            totr=totr_config,
            scr=scr_config,
            react=react_config,
        )
