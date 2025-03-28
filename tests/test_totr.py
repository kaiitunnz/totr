import asyncio
from pathlib import Path

from totr.config import Config
from totr.totr import ToTR


async def main() -> None:
    question = (
        "What is the nationality of the foreign born victim of Singapore's "
        "caning punishment before Oliver Fricker experienced the same?"
    )
    config = Config.from_json(Path("configs/hotpotqa/Llama-3.1-8B-Instruct_chat.json"))
    totr = ToTR(config, seed=0)
    answer = await totr.answer(question)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
