from dataclasses import dataclass, field
from typing import Dict, Optional

from .generation import GenerationConfig


@dataclass
class SCRConfig:
    num_chains: int = 5
    retriever_gen_config_dict: Optional[Dict] = None
    retriever_gen_config: GenerationConfig = field(init=False)
    similarity_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.retriever_gen_config_dict is None:
            self.retriever_gen_config = GenerationConfig()
        else:
            self.retriever_gen_config = GenerationConfig(
                **self.retriever_gen_config_dict
            )
