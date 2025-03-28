import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import asyncio

from benchmark.react.config import ReActFullConfig
from benchmark.react.react import ReAct


async def main() -> None:
    question = "What is the gross leasable area of the shopping mall served by the Merrick Boulevard buses?"
    config = ReActFullConfig.from_json("configs/hotpotqa/Llama-3.1-8B-Instruct.json")
    react = ReAct(config)
    answer = await react.answer(question)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
