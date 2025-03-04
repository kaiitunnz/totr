from typing import List, Tuple

from ircot.qa import IRCoTQAModel
from ircot.retriever import IRCoTRetriever

from totr.config import Config


class IRCoT:
    def __init__(self, config: Config) -> None:
        self.retriever = IRCoTRetriever(config)
        self.qa = IRCoTQAModel(config)

    async def retrieve(self, question: str) -> Tuple[List[str], List[str]]:
        titles, paras, _ = await self.retriever.retrieve(question)
        return titles, paras

    async def answer(self, question: str, use_retriever_answer: bool = False) -> str:
        titles, paras, answer = await self.retriever.retrieve(question)
        if use_retriever_answer and answer is not None:
            return answer
        answer = await self.qa.answer(question, titles, paras)
        return answer
