import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

from totr.config.generation import GenerationConfig
from totr.config.llm import LLMConfig
from totr.config.prompt import PromptConfig
from .react import ReActConfig
from totr.config.retriever import RetrieverConfig


@dataclass
class ReActFullConfig:
    llm: LLMConfig
    generation: GenerationConfig
    retriever: RetrieverConfig
    prompt_config: PromptConfig
    react: ReActConfig

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

        generation_kwargs: Dict[str, Any] = (
            dict(model=llm_config.model) | obj["generation"]
        )
        generation_config = GenerationConfig(**generation_kwargs)

        retriever_config = RetrieverConfig(**obj["retriever"])
        prompt_config = PromptConfig(**obj["prompt"])

        react_retriever_generation_dict: Dict[str, Any] = (
            dict(model=llm_config.model) | obj["react"]["retriever_gen_config_dict"]
        )
        obj["react"]["retriever_gen_config_dict"] = react_retriever_generation_dict
        react_config = ReActConfig(**obj["react"])

        return cls(
            llm=llm_config,
            generation=generation_config,
            retriever=retriever_config,
            prompt_config=prompt_config,
            react=react_config,
        )

    def with_model(self, model: str) -> "Config":
        self.llm.model = model
        self.generation.model = model
        return self

    @property
    def identifier(self) -> str:
        config: List[str] = [self.llm.model.split("/")[-1]]
        return "_".join(config)
