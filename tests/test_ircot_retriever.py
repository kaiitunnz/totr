import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "benchmark"))

import asyncio

from ircot.retriever import IRCoTRetriever
from totr.config import Config
from totr.utils.prompt import retrieved_to_context


async def main():
    question = (
        "What is the nationality of the foreign born victim of Singapore's "
        "caning punishment before Oliver Fricker experienced the same?"
    )
    config = Config.from_json(Path("configs/hotpotqa/flan-t5-large.json"))
    retriever = IRCoTRetriever(config)
    titles, paragraphs, answer = await retriever.retrieve(question)
    print("-" * 30 + "Context" + "-" * 30)
    print(retrieved_to_context(titles, paragraphs, retriever.max_para_word_count))
    print("-" * 30 + "Answer" + "-" * 30)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
