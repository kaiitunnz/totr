from dataclasses import dataclass
from typing import Dict, Optional

CONTEXT_WINDOW_SIZE_MAP: Dict[str, int] = {
    "EleutherAI/gpt-j-6b": 2048,
    "google/flan-t5-base": 512,
    "google/flan-t5-large": 512,
    "google/flan-t5-xl": 512,
    "google/flan-t5-xxl": 512,
    "meta-llama/Meta-Llama-3-8B": 8192,
    "meta-llama/Meta-Llama-3-8B-Instruct": 8192,
    "meta-llama/Llama-3.1-8B": 131072,
    "meta-llama/Llama-3.1-8B-Instruct": 131072,
    "meta-llama/Llama-3.2-3B": 131072,
    "meta-llama/Llama-3.2-3B-Instruct": 131072,
    "Qwen/Qwen2.5-3B": 32768,
    "Qwen/Qwen2.5-3B-Instruct": 32768,
    "Qwen/Qwen2.5-7B": 131072,
    "Qwen/Qwen2.5-7B-Instruct": 131072,
}


@dataclass
class LLMConfig:
    engine: str
    host: Optional[str]
    port: Optional[int]
    api_key: str
    model: str
    overriding_context_window: Optional[int] = None

    @property
    def base_url(self) -> str:
        if self.host is None or self.port is None:
            raise ValueError(
                f"None host or port found: host={self.host}, port={self.port}"
            )
        return f"http://{self.host}:{self.port}/v1"

    @property
    def context_window_size(self) -> int:
        if self.overriding_context_window is not None:
            return self.overriding_context_window
        return CONTEXT_WINDOW_SIZE_MAP[self.model]
