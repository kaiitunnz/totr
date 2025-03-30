from totr.config import Config
from totr.utils.prompt import read_prompt_file

config = Config.from_json("configs/hotpotqa/Llama-3.1-8B-Instruct.json")
examples = read_prompt_file(
    config.prompt.direct_prompt_file,
    shuffle=False,
    tokenizer_name=config.llm.model,
    context_window_size=None,
    estimated_generation_length=500,
    removal_method="last",
)
for example in examples:
    print(repr(example))
    print()
