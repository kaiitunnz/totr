from dataclasses import dataclass
from typing import Optional


def string_to_bool(string: str) -> bool:
    return string.lower() == "true"


@dataclass
class RetrieverConfig:
    elasticsearch_host: str
    elasticsearch_port: int
    retriever_name: str
    hit_count_per_step: int
    corpus_name: str
    document_type: str
    skip_long_paras: bool
    max_para_count: int
    max_gen_sent_count: int
    max_para_word_count: int
    disable_exit: bool
    answer_regex: Optional[str] = ".* answer is:? (.*)\\.?"

    @property
    def elasticsearch_url(self) -> str:
        return f"http://{self.elasticsearch_host}:{self.elasticsearch_port}"
