import re
from dataclasses import replace
from typing import List, Optional, Tuple
from pathlib import Path

import spacy

from totr.config import Config
from totr.config.generation import GenerationConfig
from totr.llm import LLMRegistry
from totr.retriever import RetrieverRegistry
from totr.utils.prompt import read_prompt_file, retrieved_to_context
from totr.utils.retriever import is_para_closely_matching, remove_wh_words
from totr.utils.transformers import seed_everything
from totr.llm.base import Message

def create_prompt(
    example_prompt: str,
    observation: str,
    question: str,
    question_prefix: Optional[str] = None,
) -> str:
    test_example_str = (
        (f"{observation}\n\n" if observation else "")
        + f"Q: {question}"
    )
    user_prompt = "\n\n\n".join([example_prompt, test_example_str]).strip()
    if question_prefix is not None:
        user_prompt = question_prefix + user_prompt
    assistant_prompt = "A:"

    messages = [
        Message(role="system", content=(
                "You are a question answerer. You answer the given question "
                'and always end your response with "So the answer is:" followed by '
                "your final answer without additional explanation."
            )),
        Message(role="user", content=user_prompt),
        Message(role="assistant", content=assistant_prompt),
    ]
    return messages


class Helper:
    def __init__(
        self,
        config: Config,
        retriever_gen_config: Optional[GenerationConfig] = None,
        seed: Optional[int] = None,
    ):
        # LLM
        self.model_name = config.llm.model
        self.llm = LLMRegistry.get(config.llm.engine, config)
        self.context_window_size = config.llm.context_window_size
        self.max_tokens = config.generation.max_tokens or 200

        # Retriever
        self.retriever = RetrieverRegistry.get(config.retriever.retriever_name, config)
        self.retrieval_count = config.retriever.hit_count_per_step
        self.corpus_name = config.retriever.corpus_name
        self.allowed_paragraph_types: Optional[List[str]] = None
        self.document_type = config.retriever.document_type
        self.max_para_count = config.retriever.max_para_count
        self.skip_long_paras = config.retriever.skip_long_paras
        self.max_para_word_count = config.retriever.max_para_word_count
        self.max_gen_sent_count = config.retriever.max_gen_sent_count
        self.spacy = spacy.load("en_core_web_sm")
        # self.tokenizer = get_tokenizer(self.model_name)

        # Retrieval
        self.question_prefix = config.qa.cot_question_prefix
        self.retriever_gen_config = retriever_gen_config or config.generation

        # Answer extraction
        answer_regex = config.retriever.answer_regex
        self.answer_regex = None if answer_regex is None else re.compile(answer_regex)

        # Full prompt
        # self.example_prompt = ""
        self.example_prompt = read_prompt_file(
            fpath=config.prompt.cot_prompt_file,
            shuffle=False,
            tokenizer_name=self.model_name,
            context_window_size=None,
            estimated_generation_length=0,
        )

        if seed is not None:
            seed_everything(seed)

    async def generate(
        self,
        question: str,
        retrieved_titles: List[str],
        retrieved_paras: List[str],
    ) -> Tuple[str, str, bool]:
        observation = retrieved_to_context(
            retrieved_titles, retrieved_paras, self.max_para_word_count
        )

        prompt = create_prompt(
            self.example_prompt,
            observation,
            question,
            question_prefix=self.question_prefix,
        )
        # prompt = prompt.rstrip()
        # print(f"Prompt before context window: {prompt}")
        # prompt = fit_prompt_in_context_window(
        #     prompt=prompt,
        #     tokenizer_name=self.model_name,
        #     context_window=self.context_window_size,
        #     estimated_generation_length=self.max_tokens,
        #     remove_method="last",
        #     shuffle=False,
        #     last_is_test_example=True,
        # )
        # print(f"Prompt after context window: {prompt}")
        gen_config: Optional[GenerationConfig]
        gen_config = self.retriever_gen_config
        # print(f"Prompt: {prompt}")
        outputs = await self.llm.chat_async(prompt, gen_config)
        # print(f"Generated answer: {outputs[0].content}")
        return self._extract_answer(outputs[0].content)

    async def retrieve_one_step(
        self, query: str, retrieved_titles: List[str], retrieved_paras: List[str]
    ) -> None:
        query = remove_wh_words(query)
        allowed_paragraph_types = (
            [None]
            if self.allowed_paragraph_types is None
            else self.allowed_paragraph_types
        )
        for paragraph_type in allowed_paragraph_types:
            if not query.strip():
                continue
            paragraph_types = None if paragraph_type is None else [paragraph_type]
            # TODO: check valid_titles
            result = await self.retriever.retrieve(
                query,
                max_hits_count=self.retrieval_count,
                corpus_name=self.corpus_name,
                allowed_paragraph_types=paragraph_types,
                document_type=self.document_type,
            )
            for item in result:
                if len(retrieved_paras) >= self.max_para_count:
                    break

                if item["corpus_name"] != self.corpus_name:
                    raise Exception(
                        f"Incorrect retrieved corpus name: {item['corpus_name']} "
                        f"(expected {self.corpus_name})"
                    )

                title = item["title"]
                paragraph_text = item["paragraph_text"]
                assert isinstance(title, str)
                assert isinstance(paragraph_text, str)

                if len(paragraph_text.split(" ")) > 600 and self.skip_long_paras:
                    continue

                if is_para_closely_matching(
                    retrieved_titles, retrieved_paras, title, paragraph_text
                ):
                    continue

                retrieved_titles.append(title)
                retrieved_paras.append(paragraph_text)
    
    def _extract_answer(self, sentence: str) -> Optional[str]:
        if self.answer_regex is None:
            return sentence
        matched = self.answer_regex.match(sentence)
        if not matched:
            return ""
        answer = matched.group(1).strip()
        if answer is None:
            return ""
        if answer.endswith("."):
            answer = answer[:-1]
        return answer


class Base:
    def __init__(self, config: Config, with_retrieval: bool=False) -> None:
        self.helper = Helper(config, config.totr.retriever_gen_config)
        self.with_retrieval = with_retrieval

    async def answer(self, question: str) -> str:
        query = question
        final_answer = None
        retrieved_titles: List[str]
        retrieved_paras: List[str]
        # 1. Retrieve relevant paragraphs
        retrieved_titles, retrieved_paras = [], []
        if self.with_retrieval:
            await self.helper.retrieve_one_step(
                query, retrieved_titles, retrieved_paras
            )

        # 2. Generate a thought
        final_answer = await self.helper.generate(
            question, retrieved_titles, retrieved_paras
        )

        return final_answer
