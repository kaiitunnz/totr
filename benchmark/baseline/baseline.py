from typing import Optional

from totr.config import Config
from totr.ir import IRHelper, QAModel


class BaseRAG:
    def __init__(
        self, config: Config, with_retrieval: bool = False, seed: Optional[int] = None
    ) -> None:
        self.helper = IRHelper(
            config, config.totr.retriever_gen_config, with_retrieval, seed
        )
        self.qa = QAModel(config, with_retrieval)
        self.with_retrieval = with_retrieval

    async def answer(self, question: str) -> str:
        retrieved_titles, retrieved_paras = [], []
        if self.with_retrieval:
            # 1. Retrieve relevant paragraphs
            await self.helper.retrieve_one_step(
                question, retrieved_titles, retrieved_paras
            )

        answer = await self.qa.answer(question, retrieved_titles, retrieved_paras)
        return answer
