from totr.config import Config
from totr.utils.prompt import read_prompt_file

config = Config.from_json()
prompt = read_prompt_file(
    config.prompt.cot_prompt_file,
    filter_by_key_values={"qid": config.prompt.prompt_example_ids},
    order_by_key=True,
    shuffle=False,
    tokenizer_name=config.llm.model,
    context_window_size=None,
    estimated_generation_length=500,
    removal_method="last",
)
print(prompt)
