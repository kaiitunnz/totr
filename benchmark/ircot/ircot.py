from typing import List, Optional, Tuple

from totr.config import Config
from totr.ir import IRHelper, QAModel


class IRCoTRetriever:
    def __init__(self, config: Config) -> None:
        self.helper = IRHelper(config)

    async def retrieve(
        self, question: str
    ) -> Tuple[List[str], List[str], Optional[str]]:
        retrieved_titles, retrieved_paras = [], []

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
                question, generated_sentences, retrieved_titles, retrieved_paras
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

        return retrieved_titles, retrieved_paras, final_answer


class IRCoT:
    def __init__(self, config: Config) -> None:
        self.retriever = IRCoTRetriever(config)
        self.qa = QAModel(config)

    async def retrieve(self, question: str) -> Tuple[List[str], List[str]]:
        titles, paras, _ = await self.retriever.retrieve(question)
        return titles, paras

    async def answer(self, question: str, use_retriever_answer: bool = False) -> str:
        titles, paras, answer = await self.retriever.retrieve(question)
        if use_retriever_answer and answer is not None:
            return answer
        answer = await self.qa.answer(question, titles, paras)
        return answer
