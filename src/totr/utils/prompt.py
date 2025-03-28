"""
Adapted from https://github.com/StonyBrookNLP/ircot/blob/3c1820f698eea5eeddb4fba3c56b64c961e063e4/commaqa/inference/prompt_reader.py
"""

import json
import random
from pathlib import Path
from typing import Dict, Hashable, List, Literal, Optional, Tuple

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .transformers import get_tokenizer


def read_prompt_file(
    fpath: Path,
    metadata_prefix: str = "# METADATA: ",
    filter_by_key_values: Optional[Dict[str, List[str]]] = None,
    order_by_key: bool = True,
    shuffle: bool = False,
    tokenizer_name: Optional[str] = None,
    context_window_size: Optional[int] = None,
    test_to_train_length_scale: int = 1,
    estimated_generation_length: int = 500,
    removal_method: Literal["last", "longest"] = "last",
) -> List[str]:
    def get_next_key(
        filter_key_values: Optional[Tuple[str, List[str]]],
        current_key: Hashable,
        metadata_line: str,
    ) -> Hashable:
        if filter_key_values is None:
            if current_key is None:
                return 0
            assert isinstance(current_key, int)
            return current_key + 1
        metadata_str = metadata_line.replace(metadata_prefix, "", 1)
        metadata = json.loads(metadata_str)
        next_key = metadata[filter_key_values[0]]
        return next_key

    if filter_by_key_values is not None:
        if len(filter_by_key_values) > 1:
            raise ValueError("Only one key can be used for filtering")
        filter_key_values = next(iter(filter_by_key_values.items()))
    else:
        filter_key_values = None

    with fpath.open() as f:
        all_prompt_lines = [line + "\n" for line in f.read().strip().split("\n")]

    # Extract examples
    example_dict: Dict[Hashable, List[str]] = {}
    current_key: Hashable = None
    for line in all_prompt_lines:
        stripped = line.strip()
        if stripped.startswith(metadata_prefix):
            current_key = get_next_key(filter_key_values, current_key, stripped)
            example_dict[current_key] = []
        else:
            example_dict[current_key].append(line)

    # Filter examples
    if filter_key_values is not None:
        valid_values = filter_key_values[1]
        if order_by_key:
            examples = [example_dict[v] for v in valid_values]
        else:
            examples = [v for k, v in example_dict.items() if k in valid_values]
    else:
        examples = list(example_dict.values())

    prompt_examples_texts = ["".join(example).strip() for example in examples]

    # Fit the examples in the context window.
    if tokenizer_name is not None and context_window_size is not None:
        prompt_examples_texts = fit_examples_in_context_window(
            prompt_examples_texts,
            tokenizer_name,
            context_window_size,
            estimated_generation_length=estimated_generation_length,
            test_to_train_length_scale=test_to_train_length_scale,
            removal_method=removal_method,
        )

    # Shuffle the examples if needed.
    if (filter_key_values is None or not order_by_key) and shuffle:
        random.shuffle(prompt_examples_texts)

    return prompt_examples_texts


def fit_examples_in_context_window(
    examples: List[str],
    tokenizer_name: str,
    context_window_size: int,
    estimated_generation_length: int,
    test_to_train_length_scale: int = 1,
    removal_method: Literal["last", "longest"] = "last",
) -> List[str]:
    if len(examples) == 1:
        # Nothing to compress. Return it as it is.
        return examples

    # Try to compress it dynamically (if needed).
    examples = examples.copy()
    tokenizer = get_tokenizer(tokenizer_name)
    example_lengths = [len(tokenizer.tokenize(example)) for example in examples]

    while example_lengths:
        max_example_length = max(example_lengths)
        estimated_test_example_length = max_example_length * test_to_train_length_scale
        estimated_total_length = (
            sum(example_lengths)
            + estimated_test_example_length
            + estimated_generation_length
        )

        if estimated_total_length <= context_window_size:
            break

        if removal_method == "last":
            examples.pop()
            example_lengths.pop()
        elif removal_method == "longest":
            max_length_index = example_lengths.index(max_example_length)
            examples.pop(max_length_index)
            example_lengths.pop(max_length_index)
        else:
            raise ValueError(f"Unknown removal method: {removal_method}")

    return examples


def fit_prompt_in_context_window(
    prompt: str,
    tokenizer_name: str,
    context_window: int,
    estimated_generation_length: int,
    shuffle: bool = False,
    remove_method: Literal["first", "last", "random", "largest"] = "first",
    example_delimiter: str = "\n\n\n",
    last_is_test_example: bool = True,
    buffer_token_count: int = 20,
) -> str:
    examples = [example.strip() for example in prompt.strip().split(example_delimiter)]
    examples = [example for example in examples if example]

    tokenizer = get_tokenizer(tokenizer_name)
    example_lengths = [len(tokenizer.tokenize(example)) for example in examples]

    test_example = None
    test_example_length = 0
    if last_is_test_example:
        test_example = examples.pop(-1)
        test_example_length = example_lengths.pop(-1)

    updated_length = (
        sum(example_lengths)
        + test_example_length
        + estimated_generation_length
        + buffer_token_count
    )
    while example_lengths and updated_length > context_window:
        if remove_method == "first":
            remove_index = 0
        elif remove_method == "last":
            remove_index = -1
        elif remove_method == "random":
            remove_index = random.randint(0, len(examples) - 1)
        elif remove_method == "largest":
            remove_index = example_lengths.index(max(example_lengths))
        else:
            raise Exception(f"Unexpected remove_method: {remove_method}.")

        examples.pop(remove_index)
        popped_length = example_lengths.pop(remove_index)
        updated_length -= popped_length

    if shuffle:
        random.shuffle(examples)

    if test_example is None:
        updated_prompt = example_delimiter.join(examples)
    else:
        updated_prompt = example_delimiter.join(examples + [test_example])

    if updated_length > context_window:
        updated_lines = updated_prompt.split("\n")
        # Truncate from the beginning of the prompt.
        while updated_lines:
            updated_lines.pop(0)
            if len(tokenizer.tokenize("\n".join(updated_lines))) <= context_window:
                break
        updated_prompt = "\n".join(updated_lines)

    return updated_prompt


def para_to_text(title: str, para: str, max_num_words: int) -> str:
    # Note: the split and join must happen before the attaching title+para.
    # also don't split() because that disrupts the new lines.
    para = " ".join(para.split(" ")[:max_num_words]).strip()
    para = (
        para
        if para.startswith("Wikipedia Title: ")
        else "Wikipedia Title: " + title + "\n" + para
    )
    return para


def retrieved_to_context(
    titles: List[str], paras: List[str], max_para_word_count: int
) -> str:
    return "\n\n".join(
        [
            para_to_text(title, para, max_para_word_count)
            for title, para in zip(titles, paras)
        ]
    )


def create_prompt(
    examples: List[str],
    context: str,
    question: str,
    partial_answer: Optional[str] = None,
    question_prefix: Optional[str] = None,
    example_delimiter: str = "\n\n\n",
) -> str:
    if question_prefix is not None:
        question = question_prefix + question
    example_prompt = example_delimiter.join(examples).strip()
    answer = f"A: {partial_answer}" if partial_answer is not None else "A:"
    test_example_str = context + "\n\n" + f"Q: {question}" + "\n" + answer
    prompt = example_delimiter.join([example_prompt, test_example_str]).strip()
    return prompt


def apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    examples: List[str],
    context: str,
    question: str,
    partial_answer: Optional[str],
    question_prefix: Optional[str],
    example_delimiter: str,
) -> str:
    if question_prefix is not None:
        question = question_prefix + question
    question_prompt = context + "\n\n" + f"Q: {question}"
    user_prompt = example_delimiter.join(examples + [question_prompt])
    assistant_prompt = f"A: {partial_answer}" if partial_answer is not None else "A:"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a question answerer. You answer the given question "
                'and always end your response with "So the answer is:" followed by '
                "your final answer without additional explanation."
            ),
        },
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_prompt},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, continue_final_message=True
    )
    assert isinstance(formatted_prompt, str)
    return formatted_prompt


def create_and_fit_prompt(
    tokenizer_name: str,
    is_chat: bool,
    examples: List[str],
    context: str,
    question: str,
    partial_answer: Optional[str],
    question_prefix: Optional[str],
    context_window: int,
    estimated_generation_length: int,
    shuffle: bool = False,
    remove_method: Literal["first", "last", "random", "largest"] = "first",
    example_delimiter: str = "\n\n\n",
    buffer_token_count: int = 20,
) -> str:
    if not is_chat:
        prompt = create_prompt(
            examples,
            context,
            question,
            partial_answer,
            question_prefix,
            example_delimiter,
        )
        prompt = fit_prompt_in_context_window(
            prompt,
            tokenizer_name,
            context_window,
            estimated_generation_length,
            shuffle,
            remove_method,
            example_delimiter,
            last_is_test_example=True,
            buffer_token_count=buffer_token_count,
        )
        return prompt

    examples = examples.copy()
    tokenizer = get_tokenizer(tokenizer_name)
    formatted_prompt = apply_chat_template(
        tokenizer,
        examples,
        context,
        question,
        partial_answer,
        question_prefix,
        example_delimiter,
    )

    total_length = (
        len(tokenizer.tokenize(formatted_prompt))
        + estimated_generation_length
        + buffer_token_count
    )
    example_lengths = [len(tokenizer.tokenize(example)) for example in examples]
    while example_lengths and total_length > context_window:
        if remove_method == "first":
            remove_index = 0
        elif remove_method == "last":
            remove_index = -1
        elif remove_method == "random":
            remove_index = random.randint(0, len(examples) - 1)
        elif remove_method == "largest":
            remove_index = example_lengths.index(max(example_lengths))
        else:
            raise Exception(f"Unexpected remove_method: {remove_method}.")

        examples.pop(remove_index)
        popped_length = example_lengths.pop(remove_index)
        total_length -= popped_length

    if shuffle:
        random.shuffle(examples)

    if total_length > context_window:
        raise ValueError("Cannot truncate the prompt to fit in the context window.")

    formatted_prompt = apply_chat_template(
        tokenizer,
        examples,
        context,
        question,
        partial_answer,
        question_prefix,
        example_delimiter,
    )
    return formatted_prompt
