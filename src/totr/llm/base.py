from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..config import Config
from ..config.generation import GenerationConfig


class Message:
    __slots__ = ["role", "content"]

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def as_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    def __repr__(self) -> str:
        return str(self.as_dict())


class BaseLLM(ABC):
    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def complete(
        self, prompt: str, config: Optional[GenerationConfig] = None
    ) -> List[str]:
        pass

    @abstractmethod
    async def complete_async(
        self, prompt: str, config: Optional[GenerationConfig] = None
    ) -> List[str]:
        pass

    @abstractmethod
    def chat(
        self, messages: List[Message], config: Optional[GenerationConfig] = None
    ) -> List[Message]:
        pass

    @abstractmethod
    async def chat_async(
        self, messages: List[Message], config: Optional[GenerationConfig] = None
    ) -> List[Message]:
        pass
