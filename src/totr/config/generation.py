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
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def openai_kwargs(self) -> Dict[str, Any]:
        kwargs = self.as_dict()
        do_sample = kwargs.pop("do_sample")
        kwargs.pop("top_k")
        # kwargs["timeout"] = None

        if not do_sample:
            kwargs["temperature"] = 0

        return kwargs
