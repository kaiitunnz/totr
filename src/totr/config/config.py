import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

from .generation import GenerationConfig
from .llm import LLMConfig
from .prompt import PromptConfig
from .qa import QAConfig
from .retriever import RetrieverConfig


@dataclass
class Config:
    llm: LLMConfig
    generation: GenerationConfig
    retriever: RetrieverConfig
    prompt: PromptConfig
    qa: QAConfig

    @classmethod
    def from_json(
        cls,
        path_or_json: Union[
            Union[str, Path], Dict[str, Dict[str, Any]]
        ] = "config.json",
    ) -> "Config":
        if isinstance(path_or_json, dict):
            obj = path_or_json
        else:
            with open(path_or_json) as f:
                obj = json.load(f)

        llm_config = LLMConfig(**obj["llm"])
        generation_config = GenerationConfig(**obj["generation"])
        retriever_config = RetrieverConfig(**obj["retriever"])
        prompt_config = PromptConfig(**obj["prompt"])
        qa_config = QAConfig(**obj["qa"])

        return cls(
            llm=llm_config,
            generation=generation_config,
            retriever=retriever_config,
            prompt=prompt_config,
            qa=qa_config,
        )
