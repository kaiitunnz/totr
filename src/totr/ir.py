import re
from dataclasses import replace
from typing import List, Optional, Tuple

import spacy
import spacy.lang

from .config import Config
from .config.generation import GenerationConfig
from .llm import LLMRegistry
from .retriever import RetrieverRegistry
from .utils.prompt import create_and_fit_prompt, read_prompt_file, retrieved_to_context
from .utils.retriever import (
    is_para_closely_matching,
    remove_reasoning_sentences,
    remove_wh_words,
)
from .utils.transformers import seed_everything

sent_tokenizer = spacy.load("en_core_web_sm")


class IRHelper:
    def __init__(
        self,
        config: Config,
        retriever_gen_config: Optional[GenerationConfig] = None,
        with_retrieval: bool = True,
        seed: Optional[int] = None,
    ):
        # LLM
        self.model_name = config.llm.model
        self.llm = LLMRegistry.get(config.llm.engine, config)
        self.context_window_size = config.llm.context_window_size
        self.is_chat = config.llm.is_chat
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
        self.document_prefix = config.retriever.document_prefix

        # Retrieval
        self.question_prefix = config.qa.cot_question_prefix
        self.retriever_gen_config = retriever_gen_config
        self.with_retrieval = with_retrieval

        # Answer extraction
        answer_regex = config.retriever.answer_regex
        self.answer_regex = None if answer_regex is None else re.compile(answer_regex)
        self.answer_split_regex = re.compile(config.retriever.answer_split_regex)
        self.remove_full_stop = config.qa.remove_full_stop

        # Full prompt
        prompt_file = (
            config.prompt.cot_prompt_file
            if self.with_retrieval
            else config.prompt.no_context_cot_prompt_file
        )
        self.examples = read_prompt_file(
            fpath=prompt_file,
            shuffle=False,
            tokenizer_name=self.model_name,
            context_window_size=None,
            estimated_generation_length=0,
        )

        if seed is not None:
            seed_everything(seed)

    def should_stop(
        self, retrieved_titles: List[str], generated_sentences: List[str]
    ) -> bool:
        return (
            len(retrieved_titles) >= self.max_para_count
            or len(generated_sentences) >= self.max_gen_sent_count
        )

    async def generate(
        self,
        question: str,
        generated_sentences: List[str],
        retrieved_titles: List[str],
        retrieved_paras: List[str],
        is_main_branch: bool = True,
    ) -> str:
        context = retrieved_to_context(
            retrieved_titles,
            retrieved_paras,
            self.max_para_word_count,
            self.document_prefix,
        )
        generated_answer = " ".join([sent.strip() for sent in generated_sentences])
        prompt = create_and_fit_prompt(
            tokenizer_name=self.model_name,
            is_chat=self.is_chat,
            examples=self.examples,
            context=context,
            question=question,
            partial_answer=generated_answer,
            question_prefix=self.question_prefix,
            context_window=self.context_window_size,
            estimated_generation_length=self.max_tokens,
            shuffle=False,
        )
        gen_config: Optional[GenerationConfig]
        if is_main_branch and self.retriever_gen_config is not None:
            # Use greedy decoding on main branch
            gen_config = replace(self.retriever_gen_config, temperature=0)
        else:
            gen_config = self.retriever_gen_config
        outputs = await self.llm.complete_async(prompt, gen_config)
        return outputs[0]

    async def generate_with_logprobs(
        self,
        question: str,
        generated_sentences: List[str],
        retrieved_titles: List[str],
        retrieved_paras: List[str],
        is_main_branch: bool = True,
    ) -> Tuple[List[str], List[float]]:
        context = retrieved_to_context(
            retrieved_titles,
            retrieved_paras,
            self.max_para_word_count,
            self.document_prefix,
        )
        generated_answer = " ".join([sent.strip() for sent in generated_sentences])
        prompt = create_and_fit_prompt(
            tokenizer_name=self.model_name,
            is_chat=self.is_chat,
            examples=self.examples,
            context=context,
            question=question,
            partial_answer=generated_answer,
            question_prefix=self.question_prefix,
            context_window=self.context_window_size,
            estimated_generation_length=self.max_tokens,
            shuffle=False,
        )
        gen_config: Optional[GenerationConfig]
        if is_main_branch and self.retriever_gen_config is not None:
            # Use greedy decoding on main branch
            gen_config = replace(self.retriever_gen_config, temperature=0)
        else:
            gen_config = self.retriever_gen_config
        outputs = await self.llm.complete_async_with_logprobs(prompt, gen_config)
        return outputs[0]

    async def evaluate_answer_confidence(
        self,
        question: str,
        generated_sentences: List[str],
        new_generation: str,
        retrieved_titles: List[str],
        retrieved_paras: List[str],
    ) -> float:
        generated_answer = " ".join(
            [sent.strip() for sent in [*generated_sentences, new_generation]]
        )
        generated_answer = self.answer_split_regex.split(
            generated_answer,
            maxsplit=1,
        )[0]
        generated_answer = generated_answer + " So the answer is:"
        _, logprobs = await self.generate_with_logprobs(
            question,
            [generated_answer],
            retrieved_titles,
            retrieved_paras,
            is_main_branch=True,
        )
        return sum(logprobs) / len(logprobs)  # Normalize log probabilities

    async def get_answer(
        self,
        question: str,
        generated_sentences: List[str],
        retrieved_titles: List[str],
        retrieved_paras: List[str],
    ) -> str:
        generated_answer = " ".join([sent.strip() for sent in generated_sentences])
        generated_answer = generated_answer + " So the answer is:"
        output = await self.generate(
            question,
            [generated_answer],
            retrieved_titles,
            retrieved_paras,
            is_main_branch=True,
        )
        answer = _post_process_answer(
            output, self.remove_full_stop, self.answer_split_regex
        )
        return answer

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

    def get_first_sentence(self, text: str) -> Optional[str]:
        new_sentences = [sent.text for sent in sent_tokenizer(text).sents]
        if len(new_sentences) == 0:
            return None
        return new_sentences[0]

    def extract_answer(self, sentence: str) -> Optional[str]:
        if self.answer_regex is None:
            return None
        matched = self.answer_regex.match(sentence)
        if not matched:
            return None
        answer = matched.group(1)
        answer = _post_process_answer(
            answer, self.remove_full_stop, self.answer_split_regex
        )
        return answer

    def get_next_query(self, question: str, generated_sentences: List[str]) -> str:
        factual_sentences = remove_reasoning_sentences(generated_sentences)
        last_sentence = factual_sentences[-1] if factual_sentences else ""
        query = last_sentence if last_sentence else question
        return query

    def select_retrieved(
        self,
        retrieved_titles_list: List[List[str]],
        retrieved_paras_list: List[List[str]],
    ) -> Tuple[List[str], List[str]]:
        merged_titles: List[str] = []
        merged_paras: List[str] = []
        for retrieved_titles, retrieved_paras in zip(
            retrieved_titles_list, retrieved_paras_list
        ):
            for title, para in zip(retrieved_titles, retrieved_paras):
                if len(merged_paras) >= self.max_para_count:
                    break
                if is_para_closely_matching(merged_titles, merged_paras, title, para):
                    continue
                merged_titles.append(title)
                merged_paras.append(para)
            else:
                continue
            break
        return merged_titles, merged_paras


class QAModel:
    def __init__(self, config: Config, with_retrieval: bool = True) -> None:
        self.config = config
        # LLM
        self.model_name = config.llm.model
        self.llm = LLMRegistry.get(config.llm.engine, config)
        self.context_window_size = config.llm.context_window_size
        self.is_chat = config.llm.is_chat
        self.max_tokens = config.generation.max_tokens or 200
        self.max_para_word_count = config.retriever.max_para_word_count

        # Document prefix
        self.document_prefix = config.retriever.document_prefix

        # Question prefix
        self.with_retrieval = with_retrieval
        if config.qa.answer_mode == "direct":
            prompt_file = (
                config.prompt.direct_prompt_file
                if self.with_retrieval
                else config.prompt.no_context_direct_prompt_file
            )
            self.question_prefix = config.qa.direct_question_prefix
        else:
            prompt_file = (
                config.prompt.cot_prompt_file
                if self.with_retrieval
                else config.prompt.no_context_cot_prompt_file
            )
            self.question_prefix = config.qa.cot_question_prefix

        # Answer extraction
        answer_regex = config.qa.answer_regex
        self.answer_regex = None if answer_regex is None else re.compile(answer_regex)
        self.remove_full_stop = config.qa.remove_full_stop
        self.answer_split_regex = re.compile(config.retriever.answer_split_regex)

        # Full prompt
        self.examples = read_prompt_file(
            fpath=prompt_file,
            shuffle=False,
            tokenizer_name=self.model_name,
            context_window_size=None,
            estimated_generation_length=0,
        )

    async def _generate(
        self,
        question: str,
        retrieved_titles: List[str],
        retrieved_paras: List[str],
        partial_answer: Optional[str] = None,
    ) -> str:
        context = (
            retrieved_to_context(
                retrieved_titles,
                retrieved_paras,
                self.max_para_word_count,
                self.document_prefix,
            )
            if self.with_retrieval
            else None
        )
        prompt = create_and_fit_prompt(
            tokenizer_name=self.model_name,
            is_chat=self.is_chat,
            examples=self.examples,
            context=context,
            question=question,
            partial_answer=partial_answer,
            question_prefix=self.question_prefix,
            context_window=self.context_window_size,
            estimated_generation_length=self.max_tokens,
            shuffle=False,
        )
        outputs = await self.llm.complete_async(prompt)
        return outputs[0]

    def _extract_answer(self, sentence: str) -> Optional[str]:
        if self.answer_regex is None:
            return sentence
        matched = self.answer_regex.match(sentence)
        if not matched:
            return None
        answer = matched.group(1)
        return _post_process_answer(
            answer, self.remove_full_stop, self.answer_split_regex
        )

    async def answer(
        self, question: str, retrieved_titles: List[str], retrieved_paras: List[str]
    ) -> str:
        output = await self._generate(question, retrieved_titles, retrieved_paras)
        answer = self._extract_answer(output)
        if answer is None:
            output = await self._generate(
                question,
                retrieved_titles,
                retrieved_paras,
                output.rstrip() + " So the answer is:",
            )
            answer = _post_process_answer(
                output, self.remove_full_stop, self.answer_split_regex
            )
        return answer


def _post_process_answer(
    answer: str, remove_full_stop: bool, answer_split_regex: re.Pattern
) -> str:
    answer = answer.strip()
    # Extract the last part after the answer split regex
    answer = answer_split_regex.split(answer)[-1].strip()
    # Extract the first sentence
    sents = list(sent_tokenizer(answer).sents)
    if len(sents) > 0:
        answer = sents[0].text
    # Remove the trailing full stop
    if remove_full_stop and answer.endswith("."):
        answer = answer[:-1]
    return answer
