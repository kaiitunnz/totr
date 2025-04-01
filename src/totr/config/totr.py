from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

from .generation import GenerationConfig


@dataclass
class ToTRConfig:
    search_method: Literal["beam", "dfs"] = "beam"
    branch_method: Literal["thought", "retrieved"] = "thought"
    num_samples: int = 2
    beam_size: int = 5
    max_depth: int = 10
    retriever_gen_config_dict: Optional[Dict] = None
    retriever_gen_config: GenerationConfig = field(init=False)
    similarity_threshold: float = 1.0

    def __post_init__(self) -> None:
        if self.retriever_gen_config_dict is None:
            self.retriever_gen_config = GenerationConfig()
        else:
            self.retriever_gen_config = GenerationConfig(
                **self.retriever_gen_config_dict
            )
