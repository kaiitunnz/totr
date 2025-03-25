import re
from dataclasses import replace
from typing import List, Optional, Tuple
from pathlib import Path

import spacy

from .config import ReActFullConfig
from totr.config.generation import GenerationConfig
from totr.llm import LLMRegistry
from totr.retriever import RetrieverRegistry
from totr.utils.prompt import read_prompt_file, retrieved_to_context
from totr.utils.retriever import is_para_closely_matching, remove_wh_words
from totr.utils.transformers import seed_everything


def create_prompt(
    example_prompt: str,
    observation: str,
    question: str,
    step: int,
    partial_answer: Optional[str] = None,
    question_prefix: Optional[str] = None,
) -> str:
    answer = f"\n{partial_answer}" if partial_answer else ""
    test_example_str = (
        f"Question: {question}"
        + answer
        + f"\nObservation {step}:\n{observation}"
        + "\n"
        + f"Thought {step}:"
    )
    prompt = "\n\n\n".join([example_prompt, test_example_str]).strip()
    if question_prefix is not None:
        prompt = question_prefix + prompt
    return prompt


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
        self.spacy = spacy.load("en_core_web_sm")

        # Retrieval
        self.question_prefix = config.react.react_question_prefix
        self.retriever_gen_config = retriever_gen_config or config.generation
        self.max_step = config.react.max_step

        # Answer extraction
        answer_regex = config.retriever.answer_regex
        self.answer_regex = None if answer_regex is None else re.compile(answer_regex)

        # Full prompt
        # self.example_prompt = ""
        self.example_prompt = read_prompt_file(
            fpath=Path(
                config.prompt_config.prompt_directory, config.prompt_config.prompt_dataset, config.react.react_prompt_filename
            ).resolve(),  
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
        partial_answer: str,
        retrieved_titles: List[str],
        retrieved_paras: List[str],
        step: int,
        is_main_branch: bool = True,
    ) -> Tuple[str, str, bool]:
        observation = retrieved_to_context(
            retrieved_titles, retrieved_paras, self.max_para_word_count
        )

        prompt = create_prompt(
            self.example_prompt,
            observation,
            question,
            step,
            partial_answer=partial_answer,
            question_prefix=self.question_prefix,
        )
        prompt = prompt.rstrip()
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
        # if is_main_branch and self.retriever_gen_config is not None:
        #     # Use greedy decoding on main branch
        #     gen_config = replace(
        #         self.retriever_gen_config, temperature=None, do_sample=False
        #     )
        # else:
        #     gen_config = self.retriever_gen_config
        gen_config = self.retriever_gen_config
        outputs = await self.llm.complete_async(prompt, gen_config)

        try:
            thought, action = outputs[0].strip().split(f"\nAction {step}: ")
        except Exception:
            print("No actions returned", outputs)
            # n_badcalls += 1
            # n_calls += 1
            thought = outputs[0].strip().split("\n")[0]
            action = await self.generate_action(prompt, thought, step)

        query, done, isvalid = self.get_action(action)

        #! Should I implement retry?
        if not isvalid:
            # raise Exception(f"Invalid action: {action}")
            print(f"Invalid action: {action}")
            return partial_answer, "", True

        partial_answer = (
            partial_answer.strip()
            + f"\nObservation {step}:\n{observation}\nThought {step}: {thought}\nAction {step}: {action}\n"
        )
        # print(f"Partial answer: {partial_answer}")
        return partial_answer, query, done

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

    async def generate_action(self, prompt: str, thought: str, step: int) -> str:
        gen_config = replace(self.retriever_gen_config, stop=["\n"])
        output = await self.llm.complete_async(
            prompt + f"\nThought {step}: {thought}\nAction {step}:", gen_config
        )
        action = output[0].strip()
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
        partial_answer = ""
        step = 1
        done = False
        final_answer = None
        retrieved_titles: List[str]
        retrieved_paras: List[str]
        while not done and step <= self.helper.max_step:
            # 1. Retrieve relevant paragraphs
            retrieved_titles, retrieved_paras = [], []
            await self.helper.retrieve_one_step(
                query, retrieved_titles, retrieved_paras
            )

            # 2. Generate a thought
            partial_answer, query, done = await self.helper.generate(
                question, partial_answer, retrieved_titles, retrieved_paras, step=step
            )
            # print(query)
            step += 1
        print(partial_answer)
        final_answer = query

        return final_answer
