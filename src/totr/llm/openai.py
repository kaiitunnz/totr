from typing import List, Optional

import backoff
from openai import AsyncOpenAI, OpenAI, RateLimitError
from openai.types.chat import ChatCompletion
from openai.types.completion import Completion

from ..config.generation import GenerationConfig
from ..config.llm import LLMConfig
from .base import BaseLLM, Message
from .registry import LLMRegistry


@LLMRegistry.register("openai")
class OpenAILLM(BaseLLM):
    @backoff.on_exception(
        backoff.expo, exception=RateLimitError, max_tries=10, max_time=60
    )
    def complete(
        self, prompt: str, config: Optional[GenerationConfig] = None
    ) -> List[str]:
        if config is None:
            config = self.config.generation

        client = OpenAI(
            api_key=self.llm_config.api_key, base_url=self.llm_config.base_url
        )

        response: Completion = client.completions.create(
            prompt=prompt, **config.openai_kwargs()
        )
        res_messages = [choice.text for choice in response.choices]
        return res_messages

    @backoff.on_exception(
        backoff.expo, exception=RateLimitError, max_tries=10, max_time=60
    )
    async def complete_async(
        self, prompt: str, config: Optional[GenerationConfig] = None
    ) -> List[str]:
        if config is None:
            config = self.config.generation

        client = AsyncOpenAI(
            api_key=self.llm_config.api_key, base_url=self.llm_config.base_url
        )

        response: Completion = await client.completions.create(
            prompt=prompt, **config.openai_kwargs()
        )
        res_messages = [choice.text for choice in response.choices]
        return res_messages

    @backoff.on_exception(
        backoff.expo, exception=RateLimitError, max_tries=10, max_time=60
    )
    def chat(
        self, messages: List[Message], config: Optional[GenerationConfig] = None
    ) -> List[Message]:
        if config is None:
            config = self.config.generation

        client = OpenAI(
            api_key=self.llm_config.api_key, base_url=self.llm_config.base_url
        )

        formated_messages = [message.as_dict() for message in messages]
        response: ChatCompletion = client.chat.completions.create(
            messages=formated_messages,  # type: ignore
            **config.openai_kwargs(),
        )

        res_messages = []
        for choice in response.choices:
            message = choice.message
            if message.content is None:
                raise Exception("Invalid server response")
            res_messages.append(Message(role=message.role, content=message.content))

        return res_messages

    @backoff.on_exception(
        backoff.expo, exception=RateLimitError, max_tries=10, max_time=60
    )
    async def chat_async(
        self, messages: List[Message], config: Optional[GenerationConfig] = None
    ) -> List[Message]:
        if config is None:
            config = self.config.generation

        client = AsyncOpenAI(
            api_key=self.llm_config.api_key, base_url=self.llm_config.base_url
        )

        formated_messages = [message.as_dict() for message in messages]
        response: ChatCompletion = await client.chat.completions.create(
            messages=formated_messages,  # type: ignore
            **config.openai_kwargs(),
        )

        res_messages = []
        for choice in response.choices:
            message = choice.message
            if message.content is None:
                raise Exception("Invalid server response")
            res_messages.append(Message(role=message.role, content=message.content))

        return res_messages

    @property
    def llm_config(self) -> LLMConfig:
        return self.config.llm
