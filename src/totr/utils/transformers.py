import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer


def get_tokenizer(model_name: str):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return AutoTokenizer.from_pretrained(model_name)


def seed_everything(seed: int):
    """
    Adapted from https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
