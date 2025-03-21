from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class GenerationConfig:
    model: Optional[str] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream_options: Optional[Any] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    @property
    def do_sample(self) -> bool:
        return self.temperature is not None and self.temperature > 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def openai_kwargs(self) -> Dict[str, Any]:
        kwargs = self.as_dict()
        kwargs.pop("top_k")
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # kwargs["timeout"] = None
        return kwargs
