import os

from transformers import AutoTokenizer


def get_tokenizer(model_name: str):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return AutoTokenizer.from_pretrained(model_name)
