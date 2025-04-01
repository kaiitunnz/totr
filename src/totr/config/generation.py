from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from totr.utils.transformers import get_tokenizer


@dataclass
class GenerationConfig:
    model: Optional[str] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = field(init=False)
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = "\n"
    stream_options: Optional[Any] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    disallowed_tokens: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.disallowed_tokens is None:
            self.logit_bias = None
            return
        tokenizer = get_tokenizer(self.model)
        self.logit_bias = {}
        for disallowed_token in self.disallowed_tokens:
            disallowed_token = disallowed_token.strip()
            for token in [disallowed_token, " " + disallowed_token]:
                encoded = tokenizer.encode(token, add_special_tokens=False)
                if len(encoded) != 1:
                    raise ValueError(
                        f"Disallowed token '{token}' must be a single token"
                    )
                self.logit_bias[str(encoded[0])] = -100

    @property
    def do_sample(self) -> bool:
        return self.temperature is not None and self.temperature > 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def openai_kwargs(self) -> Dict[str, Any]:
        kwargs = self.as_dict()
        kwargs.pop("top_k")
        kwargs.pop("disallowed_tokens")
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # kwargs["timeout"] = None
        return kwargs
