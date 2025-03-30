from typing import List, Optional, Tuple

import backoff
from openai import APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError
from openai.types.chat import ChatCompletion
from openai.types.completion import Completion

from ..config.generation import GenerationConfig
from ..config.llm import LLMConfig
from .base import BaseLLM, Message
from .registry import LLMRegistry


@LLMRegistry.register("openai")
class OpenAILLM(BaseLLM):
    @backoff.on_exception(
        backoff.expo, exception=(RateLimitError, APITimeoutError), max_tries=10
    )
    def _complete(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        logprobs: Optional[int] = None,
    ) -> Completion:
        if config is None:
            config = self.config.generation

        client = OpenAI(
            api_key=self.llm_config.api_key, base_url=self.llm_config.base_url
        )

        response: Completion = client.completions.create(
            prompt=prompt, logprobs=logprobs, **config.openai_kwargs()
        )
        return response

    def complete(
        self, prompt: str, config: Optional[GenerationConfig] = None
    ) -> List[str]:
        response = self._complete(prompt, config)
        res_messages = [choice.text for choice in response.choices]
        return res_messages

    def complete_with_logprobs(
        self, prompt: str, config: Optional[GenerationConfig] = None
    ) -> List[Tuple[List[str], List[float]]]:
        response = self._complete(prompt, config, logprobs=1)
        res_messages: List[Tuple[List[str], List[float]]] = []
        for choice in response.choices:
            logprobs = choice.logprobs
            assert logprobs is not None
            assert logprobs.token_logprobs is not None
            assert logprobs.tokens is not None
            res_messages.append((logprobs.tokens, logprobs.token_logprobs))
        return res_messages

    @backoff.on_exception(
        backoff.expo, exception=(RateLimitError, APITimeoutError), max_tries=10
    )
    async def _complete_async(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        logprobs: Optional[int] = None,
    ) -> Completion:
        if config is None:
            config = self.config.generation

        client = AsyncOpenAI(
            api_key=self.llm_config.api_key, base_url=self.llm_config.base_url
        )

        response: Completion = await client.completions.create(
            prompt=prompt, logprobs=logprobs, **config.openai_kwargs()
        )
        return response

    async def complete_async(
        self, prompt: str, config: Optional[GenerationConfig] = None
    ) -> List[str]:
        response = await self._complete_async(prompt, config)
        res_messages = [choice.text for choice in response.choices]
        return res_messages

    async def complete_async_with_logprobs(
        self, prompt: str, config: Optional[GenerationConfig] = None
    ) -> List[Tuple[List[str], List[float]]]:
        response = await self._complete_async(prompt, config, logprobs=1)
        res_messages = []
        for choice in response.choices:
            logprobs = choice.logprobs
            assert logprobs is not None
            assert logprobs.token_logprobs is not None
            assert logprobs.tokens is not None
            res_messages.append((logprobs.tokens, logprobs.token_logprobs))
        return res_messages

    @backoff.on_exception(
        backoff.expo, exception=(RateLimitError, APITimeoutError), max_tries=10
    )
    def _chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
        logprobs: Optional[bool] = None,
    ) -> ChatCompletion:
        if config is None:
            config = self.config.generation

        client = OpenAI(
            api_key=self.llm_config.api_key, base_url=self.llm_config.base_url
        )

        formated_messages = [message.as_dict() for message in messages]
        response: ChatCompletion = client.chat.completions.create(
            messages=formated_messages,  # type: ignore
            logprobs=logprobs,
            **config.openai_kwargs(),
        )
        return response

    def chat(
        self, messages: List[Message], config: Optional[GenerationConfig] = None
    ) -> List[Message]:
        response = self._chat(messages, config)
        res_messages = []
        for choice in response.choices:
            message = choice.message
            if message.content is None:
                raise Exception("Invalid server response")
            res_messages.append(Message(role=message.role, content=message.content))
        return res_messages

    def chat_with_logprobs(
        self, messages: List[Message], config: Optional[GenerationConfig] = None
    ) -> List[Tuple[List[str], List[float]]]:
        response = self._chat(messages, config, logprobs=True)
        res_messages: List[Tuple[List[str], List[float]]] = []
        for choice in response.choices:
            message = choice.message
            if message.content is None:
                raise Exception("Invalid server response")
            logprobs = choice.logprobs
            assert logprobs is not None
            assert logprobs.content is not None
            tokens, token_logprobs = [], []
            for token in logprobs.content:
                tokens.append(token.token)
                token_logprobs.append(token.logprob)
            res_messages.append((tokens, token_logprobs))
        return res_messages

    @backoff.on_exception(
        backoff.expo, exception=(RateLimitError, APITimeoutError), max_tries=10
    )
    async def _chat_async(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
        logprobs: Optional[bool] = None,
    ) -> ChatCompletion:
        if config is None:
            config = self.config.generation

        client = AsyncOpenAI(
            api_key=self.llm_config.api_key, base_url=self.llm_config.base_url
        )

        formated_messages = [message.as_dict() for message in messages]
        response: ChatCompletion = await client.chat.completions.create(
            messages=formated_messages,  # type: ignore
            logprobs=logprobs,
            **config.openai_kwargs(),
        )
        return response

    async def chat_async(
        self, messages: List[Message], config: Optional[GenerationConfig] = None
    ) -> List[Message]:
        response = await self._chat_async(messages, config)
        res_messages = []
        for choice in response.choices:
            message = choice.message
            if message.content is None:
                raise Exception("Invalid server response")
            res_messages.append(Message(role=message.role, content=message.content))
        return res_messages

    async def chat_async_with_logprobs(
        self, messages: List[Message], config: Optional[GenerationConfig] = None
    ) -> List[Tuple[List[str], List[float]]]:
        response = await self._chat_async(messages, config, logprobs=True)
        res_messages: List[Tuple[List[str], List[float]]] = []
        for choice in response.choices:
            message = choice.message
            if message.content is None:
                raise Exception("Invalid server response")
            logprobs = choice.logprobs
            assert logprobs is not None
            assert logprobs.content is not None
            tokens, token_logprobs = [], []
            for token in logprobs.content:
                tokens.append(token.token)
                token_logprobs.append(token.logprob)
            res_messages.append((tokens, token_logprobs))
        return res_messages

    @property
    def llm_config(self) -> LLMConfig:
        return self.config.llm
