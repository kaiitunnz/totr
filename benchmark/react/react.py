import re
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

import spacy

from totr.config.generation import GenerationConfig
from totr.llm import LLMRegistry
from totr.llm.base import Message
from totr.retriever import RetrieverRegistry
from totr.utils.prompt import read_prompt_file, retrieved_to_context
from totr.utils.retriever import is_para_closely_matching, remove_wh_words
from totr.utils.transformers import seed_everything

from .config import ReActFullConfig


class REACTRHelper:
    def __init__(
        self,
        config: ReActFullConfig,
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
        self.document_prefix = config.retriever.document_prefix
        self.spacy = spacy.load("en_core_web_sm")

        # Retrieval
        self.question_prefix = config.react.react_question_prefix
        self.retriever_gen_config = retriever_gen_config or config.generation
        self.max_step = config.react.max_step

        # Answer extraction
        answer_regex = config.retriever.answer_regex
        self.answer_regex = None if answer_regex is None else re.compile(answer_regex)

        # Full prompt
        self.example_prompt = "\n\n\n".join(
            read_prompt_file(
                fpath=Path(
                    config.prompt.prompt_directory,
                    config.prompt.prompt_dataset,
                    config.react.react_prompt_filename,
                ).resolve(),
                shuffle=False,
                tokenizer_name=self.model_name,
                context_window_size=None,
                estimated_generation_length=0,
            )
        )

        if seed is not None:
            seed_everything(seed)

    async def generate(
        self,
        chat_history: List[Message],
        retrieved_titles: List[str],
        retrieved_paras: List[str],
        step: int,
    ) -> Tuple[str, bool]:
        observation = retrieved_to_context(
            retrieved_titles,
            retrieved_paras,
            self.max_para_word_count,
            self.document_prefix,
        )

        chat_history.append(
            Message(role="user", content=f"Observation {step}:\n{observation}")
        )

        gen_config: Optional[GenerationConfig]
        gen_config = self.retriever_gen_config
        outputs = await self.llm.chat_async(chat_history, gen_config)

        try:
            thought, action = outputs[0].content.strip().split(f"\nAction {step}: ")
        except Exception:
            print("No actions returned", outputs)
            # n_badcalls += 1
            # n_calls += 1
            thought = outputs[0].content.strip().split("\n")[0]
            action = await self.generate_action(chat_history, thought, step)

        query, done, isvalid = self.get_action(action)

        #! Should I implement retry?
        if not isvalid:
            # raise Exception(f"Invalid action: {action}")
            print(f"Invalid action: {action}")
            return "", True

        # print(f"{thought}")
        # print(f"Action {step}: {action}")
        chat_history.append(
            Message(role="assistant", content=f"{thought}\nAction {step}: {action}")
        )
        return query, done

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

    def get_action(self, action: str) -> Tuple[str, bool, bool]:
        action = action.strip()
        query = ""
        done = False
        isvalid = True

        search = re.compile(r"search\[.*?\]", re.IGNORECASE)
        finish = re.compile(r"finish\[.*?\]", re.IGNORECASE)

        matched_finish = finish.search(action)
        if matched_finish:
            query = matched_finish.group(0)[7:-1]
            done = True
        else:
            matched_search = search.search(action)
            if matched_search:
                query = matched_search.group(0)[7:-1]
            else:
                isvalid = False
        return query, done, isvalid

    async def generate_action(
        self, chat_history: List[Message], thought: str, step: int
    ) -> str:
        gen_config = replace(self.retriever_gen_config, stop=["\n"])
        outputs = await self.llm.chat_async(
            chat_history
            + [Message(role="assistant", content=f"{thought}\nAction {step}:")],
            gen_config,
        )
        action = outputs[0].content.strip()
        return action

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


class ReAct:
    def __init__(self, config: ReActFullConfig) -> None:
        self.helper = REACTRHelper(config, config.react.retriever_gen_config)

    async def answer(self, question: str) -> str:
        query = question
        step = 1
        done = False
        final_answer = None
        chat_history = []
        sys_chat = self.helper.example_prompt.strip()
        if self.helper.question_prefix is not None:
            sys_chat = self.helper.question_prefix + sys_chat
        chat_history.append(Message(role="system", content=sys_chat))
        chat_history.append(Message(role="user", content=f"Question: {question}"))
        retrieved_titles: List[str]
        retrieved_paras: List[str]
        while not done and step <= self.helper.max_step:
            # 1. Retrieve relevant paragraphs
            retrieved_titles, retrieved_paras = [], []
            await self.helper.retrieve_one_step(
                query, retrieved_titles, retrieved_paras
            )

            # 2. Generate a thought
            query, done = await self.helper.generate(
                chat_history, retrieved_titles, retrieved_paras, step=step
            )
            # print(query)
            step += 1
        final_answer = query
        # print(chat_history)

        return final_answer
