from typing import List

from totr.config import Config
from totr.reactr import REACTRHelper


class ReAct:
    def __init__(self, config: Config) -> None:
        self.helper = REACTRHelper(config, config.react.retriever_gen_config)

    async def answer(self, question: str) -> str:
        query = question
        partial_answer = ""
        step = 1
        done = False
        final_answer = None
        while not done and step <= self.helper.max_step:
            # 1. Retrieve relevant paragraphs
            retrieved_titles: List[str] = []
            retrieved_paras: List[str] = []
            await self.helper.retrieve_one_step(
                query, retrieved_titles, retrieved_paras
            )

            # 2. Generate a thought
            partial_answer, query, done = await self.helper.generate(
                question, partial_answer, retrieved_titles, retrieved_paras, step=step
            )
            # print(query)
            step += 1
        # print(partial_answer)
        final_answer = query

        return final_answer
