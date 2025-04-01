import re
from typing import List

import numpy as np
from rapidfuzz import fuzz
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer


def remove_wh_words(text: str) -> str:
    wh_words = {"who", "what", "when", "where", "why", "which", "how", "does", "is"}
    words = [word for word in text.split(" ") if word.strip().lower() not in wh_words]
    text = " ".join(words)
    return text


def is_para_closely_matching(
    existing_titles: List[str],
    existing_paras: List[str],
    new_title: str,
    new_para: str,
    match_threshold: float = 90,
) -> bool:

    if new_title in existing_titles and new_para in existing_paras:
        return True

    assert match_threshold > 1.0, "The threshold is 0-100 scaled."

    assert len(existing_titles) == len(existing_paras)
    for existing_title, existing_para in zip(existing_titles, existing_paras):
        condition_1 = fuzz.ratio(existing_title, new_title) >= match_threshold
        condition_2 = fuzz.ratio(existing_para, new_para) >= match_threshold
        if condition_1 and condition_2:
            return True
    return False


def is_reasoning_sentence(sentence: str) -> bool:
    starters = ["thus ", "thus,", "so ", "so,", "that is,", "therefore", "hence"]
    for starter in starters:
        if sentence.lower().startswith(starter):
            return True

    regex = re.compile(
        r"(.*)(\d[\d,]*\.?\d+|\d+) ([+-]) (\d[\d,]*\.?\d+|\d+) = (\d[\d,]*\.?\d+|\d+)(.*)"
    )
    match = bool(re.match(regex, sentence))
    if match:
        return True

    return False


def remove_reasoning_sentences(sentences: List[str]) -> List[str]:
    return [sentence for sentence in sentences if not is_reasoning_sentence(sentence)]


def rerank_answers(
    answers: List[str], retrieved_counts: List[int], threshold_factor: float
) -> List[int]:
    num_answers = len(answers)
    if num_answers == 0:
        return []
    if num_answers == 1:
        return [0]

    vectorizer = CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b")
    try:
        X = vectorizer.fit_transform(answers)
        similarities_ = (X @ X.T) / X.shape[-1]  # type: ignore
    except ValueError:
        similarities_ = np.zeros((num_answers, num_answers))

    similarities: np.ndarray
    if isinstance(similarities_, spmatrix):
        similarities = np.array(similarities_.todense())
    else:
        similarities = similarities_
    np.fill_diagonal(similarities, 0)
    similarities = similarities.sum(axis=0) / (similarities.shape[0] - 1)

    # Sort by similarity in desc and the number of retrieved documents in asc
    max_similarity = similarities.max()
    similarity_threshold = max_similarity * threshold_factor
    filtered = [
        (sim, -retrieved_count, i)
        for i, (sim, retrieved_count) in enumerate(zip(similarities, retrieved_counts))
        if sim > similarity_threshold or sim == max_similarity
    ]
    filtered.sort(reverse=True)

    ranks = [i for _, _, i in filtered]
    return ranks
