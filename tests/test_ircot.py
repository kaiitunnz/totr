import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import asyncio

from benchmark.ircot import IRCoT
from totr.config import Config


async def main() -> None:
    question = (
        "What is the nationality of the foreign born victim of Singapore's "
        "caning punishment before Oliver Fricker experienced the same?"
    )
    config = Config.from_json()
    ircot = IRCoT(config)
    answer = await ircot.answer(question)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
