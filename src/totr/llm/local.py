from typing import Any, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import GenerationConfig as HFGenerationConfig
from transformers import GenerationMixin, PreTrainedTokenizerBase

from ..config import Config
from ..config.generation import GenerationConfig
from .base import BaseLLM, Message
from .registry import LLMRegistry


@LLMRegistry.register("local")
class LocalLLM(BaseLLM):
    config: Config
    model: GenerationMixin
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    chat_role: str = "assistant"

    def __init__(self, config: Config) -> None:
        cls = self.__class__
        if hasattr(cls, "model"):
            return

        cls.config = config
        model_name = config.llm.model
        if "flan-t5" in model_name:
            cls.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, device_map="auto"
            )
        else:
            cls.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto"
            )
        cls.tokenizer = AutoTokenizer.from_pretrained(model_name)
        cls.device = cls.model.device  # type: ignore

    def complete(
        self, prompt: str, config: Optional[GenerationConfig] = None
    ) -> List[str]:
        inputs: Any = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs, generation_config=self._get_hf_generation_config(config)
        )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded

    def complete_with_logprobs(
        self, prompt: str, config: Optional[GenerationConfig] = None
    ) -> List[Tuple[List[str], List[float]]]:
        raise NotImplementedError("TODO")

    async def complete_async(
        self, prompt: str, config: Optional[GenerationConfig] = None
    ) -> List[str]:
        return self.complete(prompt, config)

    async def complete_async_with_logprobs(
        self, prompt: str, config: Optional[GenerationConfig] = None
    ) -> List[Tuple[List[str], List[float]]]:
        raise NotImplementedError("TODO")

    def chat(
        self, messages: List[Message], config: Optional[GenerationConfig] = None
    ) -> List[Message]:
        message_dict_list = [message.as_dict() for message in messages]
        inputs: Any = self.tokenizer.apply_chat_template(
            message_dict_list,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        inputs = inputs.to(self.device)
        outputs = self.model.generate(
            **inputs, generation_config=self._get_hf_generation_config(config)
        )
        input_len = inputs["input_ids"].size(1)
        outputs = outputs[:, input_len:]
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [Message(role=self.chat_role, content=content) for content in decoded]

    def chat_with_logprobs(
        self, messages: List[Message], config: Optional[GenerationConfig] = None
    ) -> List[Tuple[List[str], List[float]]]:
        raise NotImplementedError("TODO")

    async def chat_async(
        self, messages: List[Message], config: Optional[GenerationConfig] = None
    ) -> List[Message]:
        return self.chat(messages, config)

    async def chat_async_with_logprobs(
        self, messages: List[Message], config: Optional[GenerationConfig] = None
    ) -> List[Tuple[List[str], List[float]]]:
        raise NotImplementedError("TODO")

    def _get_hf_generation_config(
        self, config: Optional[GenerationConfig]
    ) -> HFGenerationConfig:
        if config is None:
            config = self.config.generation
        return HFGenerationConfig(
            max_length=config.max_tokens,
            stop_strings=config.stop,
            do_sample=config.do_sample,
            use_cache=True,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            pad_token_id=self.tokenizer.eos_token_id,
        )
