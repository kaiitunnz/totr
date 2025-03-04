"""
Adapted from https://github.com/metauto-ai/GPTSwarm/blob/main/swarm/llm/llm_registry.py
"""

from typing import Hashable, Iterable, Mapping, Type, TypeVar

from class_registry import ClassRegistry

from ..config import Config
from .base import BaseLLM

T = TypeVar("T")


class LLMRegistry:
    registry: ClassRegistry[BaseLLM] = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)

    @classmethod
    def keys(cls) -> Iterable[Hashable]:
        return cls.registry.keys()

    @classmethod
    def items(cls) -> Mapping[Hashable, Type[BaseLLM]]:
        return dict(cls.registry.items())

    @classmethod
    def get(cls, key: str, config: Config) -> BaseLLM:
        return cls.registry.get(key, config=config)
