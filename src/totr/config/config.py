import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

from .generation import GenerationConfig
from .llm import LLMConfig
from .prompt import PromptConfig
from .qa import QAConfig
from .retriever import RetrieverConfig
from .scr import SCRConfig
from .totr import ToTRConfig


@dataclass
class Config:
    llm: LLMConfig
    generation: GenerationConfig
    retriever: RetrieverConfig
    prompt: PromptConfig
    qa: QAConfig
    totr: ToTRConfig
    scr: SCRConfig

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

        return cls(
            llm=llm_config,
            generation=generation_config,
            retriever=retriever_config,
            prompt=prompt_config,
            qa=qa_config,
            totr=totr_config,
            scr=scr_config,
        )

    def with_model(self, model: str) -> "Config":
        self.llm.model = model
        self.generation.model = model
        return self

    @property
    def identifier(self) -> str:
        config: List[str] = [self.llm.model.split("/")[-1], self.qa.answer_mode]
        if self.qa.use_retriever_answer:
            config.append("retr-ans")
        return "_".join(config)
