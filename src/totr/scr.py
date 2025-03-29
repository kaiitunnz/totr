import asyncio
from typing import List, Optional, Tuple

from .config import Config
from .ir import IRHelper, QAModel
from .utils.retriever import rerank_answers


class SCRRetriever:
    def __init__(self, config: Config, seed: Optional[int] = None) -> None:
        self.helper = IRHelper(config, config.scr.retriever_gen_config, seed)

        self.num_chains = config.scr.num_chains
        self.similarity_threshold = config.scr.similarity_threshold

    async def _retrieve_one_chain(
        self, question: str, is_main_chain: bool
    ) -> Tuple[List[str], List[str], List[str], Optional[str]]:
        retrieved_titles: List[str] = []
        retrieved_paras: List[str] = []

        query = question
        generated_sentences: List[str] = []
        final_answer: Optional[str] = None
        while not self.helper.should_stop(retrieved_titles, generated_sentences):
            # 1. Retrieve relevant paragraphs
            await self.helper.retrieve_one_step(
                query, retrieved_titles, retrieved_paras
            )

            # 2. Generate a thought
            new_generation = await self.helper.generate(
                question,
                generated_sentences,
                retrieved_titles,
                retrieved_paras,
                is_main_chain,
            )
            new_sentence = self.helper.get_first_sentence(new_generation)
            if new_sentence is None:
                break
            generated_sentences.append(new_sentence)

            # Extract answer if any
            final_answer = self.helper.extract_answer(new_sentence)
            if final_answer is not None:
                break

            # Update query
            query = self.helper.get_next_query(question, generated_sentences)

        return retrieved_titles, retrieved_paras, generated_sentences, final_answer

    async def retrieve(
        self, question: str
    ) -> Tuple[List[str], List[str], Optional[str]]:
        assert self.num_chains > 0
        retrieval_tasks = [
            asyncio.create_task(self._retrieve_one_chain(question, i == 0))
            for i in range(self.num_chains)
        ]
        results = await asyncio.gather(*retrieval_tasks)

        # 1. Extract chains with answers
        titles_chains: List[List[str]] = []
        paras_chains: List[List[str]] = []
        retrieved_counts: List[int] = []
        answers_chains: List[str] = []
        for titles, paras, generated_sentences, answer in results:
            titles_chains.append(titles)
            paras_chains.append(paras)
            retrieved_counts.append(len(titles))
            if answer is None:
                answer = await self.helper.get_answer(
                    question, generated_sentences, titles, paras
                )
            answers_chains.append(answer)

        # 2. Sort chains by answers
        chain_ranks = rerank_answers(
            answers_chains, retrieved_counts, self.similarity_threshold
        )

        retrieved_titles_list: List[List[str]] = []
        retrieved_paras_list: List[List[str]] = []
        for i in chain_ranks:
            retrieved_titles_list.append(titles_chains[i])
            retrieved_paras_list.append(paras_chains[i])
        retrieved_titles, retrieved_paras = self.helper.select_retrieved(
            retrieved_titles_list, retrieved_paras_list
        )
        final_answer = answers_chains[chain_ranks[0]]

        return retrieved_titles, retrieved_paras, final_answer


class SCR:
    def __init__(self, config: Config, seed: Optional[int] = None) -> None:
        self.retriever = SCRRetriever(config, seed)
        self.qa = QAModel(config)
        self.use_retriever_answer = config.qa.use_retriever_answer

    async def retrieve(self, question: str) -> Tuple[List[str], List[str]]:
        titles, paras, _ = await self.retriever.retrieve(question)
        return titles, paras

    async def answer(
        self, question: str, use_retriever_answer: Optional[bool] = None
    ) -> str:
        titles, paras, answer = await self.retriever.retrieve(question)
        if use_retriever_answer is None:
            use_retriever_answer = self.use_retriever_answer
        if use_retriever_answer and answer is not None:
            return answer
        answer = await self.qa.answer(question, titles, paras)
        return answer
