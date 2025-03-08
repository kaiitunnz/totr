import re
from typing import List, Optional

from totr.config import Config
from totr.llm.registry import LLMRegistry
from totr.utils.prompt import (
    create_prompt,
    fit_prompt_in_context_window,
    read_prompt_file,
    retrieved_to_context,
)


class IRCoTQAModel:
    def __init__(self, config: Config) -> None:
        self.model_name = config.llm.model
        self.llm = LLMRegistry.get(config.llm.engine, config)
        self.context_window_size = config.llm.context_window_size
        self.max_tokens = config.generation.max_tokens or 200
        self.max_para_word_count = config.retriever.max_para_word_count
        answer_regex = config.qa.answer_regex
        self.answer_regex = None if answer_regex is None else re.compile(answer_regex)
        self.remove_full_stop = config.qa.remove_full_stop

        if config.qa.answer_mode == "direct":
            prompt_file = config.prompt.direct_prompt_file
            self.question_prefix = config.qa.direct_question_prefix
        else:
            prompt_file = config.prompt.cot_prompt_file
            self.question_prefix = config.qa.cot_question_prefix

        self.example_prompt = read_prompt_file(
            fpath=prompt_file,
            filter_by_key_values={"qid": config.prompt.prompt_example_ids},
            order_by_key=True,
            shuffle=False,
            tokenizer_name=self.model_name,
            context_window_size=None,
            estimated_generation_length=0,
        )

    async def _generate(self, context: str, question: str) -> str:
        prompt = create_prompt(
            self.example_prompt,
            context,
            question,
            question_prefix=self.question_prefix,
        )
        prompt = prompt.rstrip()
        prompt = fit_prompt_in_context_window(
            prompt=prompt,
            tokenizer_name=self.model_name,
            context_window=self.context_window_size,
            estimated_generation_length=self.max_tokens,
            shuffle=False,
            last_is_test_example=True,
        )
        outputs = await self.llm.complete_async(prompt)
        return outputs[0]

    def _extract_answer(self, sentence: str) -> Optional[str]:
        if self.answer_regex is None:
            return sentence
        matched = self.answer_regex.match(sentence)
        if not matched:
            return None
        answer = matched.group(1).strip()
        assert answer is not None
        if self.remove_full_stop and answer.endswith("."):
            answer = answer[:-1]
        return answer

    async def answer(
        self, question: str, retrieved_titles: List[str], retrieved_paras: List[str]
    ) -> str:
        context = retrieved_to_context(
            retrieved_titles, retrieved_paras, self.max_para_word_count
        )
        output = await self._generate(context, question)
        answer = self._extract_answer(output)
        if answer is None:
            return output
        return answer
