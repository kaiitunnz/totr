import re
from typing import List, Optional, Tuple

import spacy

from totr.config import Config
from totr.llm import LLMRegistry
from totr.retriever import RetrieverRegistry
from totr.utils.prompt import (
    create_prompt,
    fit_prompt_in_context_window,
    read_prompt_file,
    retrieved_to_context,
)
from totr.utils.retriever import (
    is_para_closely_matching,
    remove_reasoning_sentences,
    remove_wh_words,
)


class IRCoTRetriever:
    def __init__(self, config: Config) -> None:
        self.model_name = config.llm.model
        self.llm = LLMRegistry.get(config.llm.engine, config)
        self.context_window_size = config.llm.context_window_size
        self.max_tokens = config.generation.max_tokens or 200
        self.retriever = RetrieverRegistry.get(config.retriever.retriever_name, config)
        self.retrieval_count = config.retriever.hit_count_per_step
        self.corpus_name = config.retriever.corpus_name
        self.allowed_paragraph_types: Optional[List[str]] = None
        self.document_type = config.retriever.document_type
        self.skip_long_paras = config.retriever.skip_long_paras
        self.max_para_count = config.retriever.max_para_count
        self.max_gen_sent_count = config.retriever.max_gen_sent_count
        self.max_para_word_count = config.retriever.max_para_word_count
        self.question_prefix = config.qa.cot_question_prefix
        answer_regex = config.retriever.answer_regex
        self.answer_regex = None if answer_regex is None else re.compile(answer_regex)
        self.remove_full_stop = config.qa.remove_full_stop
        self.spacy = spacy.load("en_core_web_sm")

        # Read full prompt
        self.example_prompt = read_prompt_file(
            fpath=config.prompt.cot_prompt_file,
            filter_by_key_values={"qid": config.prompt.prompt_example_ids},
            order_by_key=True,
            shuffle=False,
            tokenizer_name=self.model_name,
            context_window_size=None,
            estimated_generation_length=0,
        )

    async def _generate(
        self, context: str, question: str, generated_answer: str
    ) -> str:
        prompt = create_prompt(
            self.example_prompt,
            context,
            question,
            generated_answer,
            self.question_prefix,
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

    async def _retrieve_one_step(
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

    def _get_first_sentence(self, text: str) -> Optional[str]:
        new_sentences = [sent.text for sent in self.spacy(text).sents]
        if len(new_sentences) == 0:
            return None
        return new_sentences[0]

    def _extract_answer(self, sentence: str) -> Optional[str]:
        if self.answer_regex is None:
            return None
        matched = self.answer_regex.match(sentence)
        if not matched:
            return None
        answer = matched.group(1).strip()
        assert answer is not None
        if self.remove_full_stop and answer.endswith("."):
            answer = answer[:-1]
        return answer

    async def retrieve(
        self, question: str
    ) -> Tuple[List[str], List[str], Optional[str]]:
        retrieved_titles, retrieved_paras = [], []

        query = question
        generated_sentences: List[str] = []
        final_answer: Optional[str] = None
        while (
            len(retrieved_titles) < self.max_para_count
            and len(generated_sentences) < self.max_gen_sent_count
        ):
            # 1. Retrieve relevant paragraphs
            await self._retrieve_one_step(query, retrieved_titles, retrieved_paras)

            # 2. Generate a thought
            context = retrieved_to_context(
                retrieved_titles, retrieved_paras, self.max_para_word_count
            )
            # Generate a new sentence
            generated_answer = " ".join(generated_sentences)
            new_generation = await self._generate(context, question, generated_answer)
            new_sentence = self._get_first_sentence(new_generation)
            if new_sentence is None:
                break
            generated_sentences.append(new_sentence)

            # Extract answer if any
            if self.answer_regex is not None:
                final_answer = self._extract_answer(new_sentence)
                if final_answer is not None:
                    break

            # Update query
            factual_sentences = remove_reasoning_sentences(generated_sentences)
            last_sentence = factual_sentences[-1] if factual_sentences else ""
            query = last_sentence if last_sentence else question

        return retrieved_titles, retrieved_paras, final_answer
