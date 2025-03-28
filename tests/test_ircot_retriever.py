import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import asyncio

from benchmark.ircot.ircot import IRCoTRetriever
from totr.config import Config
from totr.utils.prompt import retrieved_to_context


async def main():
    question = (
        "What is the nationality of the foreign born victim of Singapore's "
        "caning punishment before Oliver Fricker experienced the same?"
    )
    config = Config.from_json(Path("configs/hotpotqa/Llama-3.1-8B-Instruct_chat.json"))
    retriever = IRCoTRetriever(config)
    titles, paragraphs, answer = await retriever.retrieve(question)
    print("-" * 30 + "Context" + "-" * 30)
    print(
        retrieved_to_context(titles, paragraphs, retriever.helper.max_para_word_count)
    )
    print("-" * 30 + "Answer" + "-" * 30)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
