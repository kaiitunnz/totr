"""
Build ES (Elasticsearch) BM25 Index.
Adapted from https://github.com/StonyBrookNLP/ircot/blob/main/retriever_server/build_index.py
"""

import argparse
import bz2
import hashlib
import io
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import base58
import dill
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm


REGISTERED_DATASETS = ("hotpotqa", "iirc", "2wikimultihopqa", "musique")


def hash_object(o: Any) -> str:
    """Returns a character hash code of arbitrary Python objects."""
    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        dill.dump(o, buffer)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()


def make_hotpotqa_documents(
    raw_data_dir: Path, elasticsearch_index: str, metadata: Optional[Dict] = None
) -> Iterable[Dict[str, Any]]:
    raw_glob_filepath = os.path.join(
        "hotpotqa", "wikpedia-paragraphs", "*", "wiki_*.bz2"
    )
    metadata = metadata or {"idx": 1}
    assert "idx" in metadata
    for filepath in tqdm(raw_data_dir.glob(raw_glob_filepath)):
        for datum in bz2.BZ2File(filepath).readlines():
            instance = json.loads(datum.strip())

            id_ = hash_object(instance)[:32]
            title = instance["title"]
            sentences_text = [e.strip() for e in instance["text"]]
            paragraph_text = " ".join(sentences_text)
            url = instance["url"]
            is_abstract = True
            paragraph_index = 0

            es_paragraph = {
                "id": id_,
                "title": title,
                "paragraph_index": paragraph_index,
                "paragraph_text": paragraph_text,
                "url": url,
                "is_abstract": is_abstract,
            }
            document = {
                "_op_type": "create",
                "_index": elasticsearch_index,
                "_id": metadata["idx"],
                "_source": es_paragraph,
            }
            yield (document)
            metadata["idx"] += 1


def make_iirc_documents(
    raw_data_dir: Path, elasticsearch_index: str, metadata: Optional[Dict] = None
) -> Iterable[Dict[str, Any]]:
    raw_filepath = raw_data_dir.joinpath("iirc", "context_articles.json")

    metadata = metadata or {"idx": 1}
    assert "idx" in metadata

    random.seed(13370)  # Don't change.

    with open(raw_filepath, "r") as file:
        full_data = json.load(file)

        for title, page_html in tqdm(full_data.items()):
            page_soup = BeautifulSoup(page_html, "html.parser")
            paragraph_texts = [
                text
                for text in page_soup.text.split("\n")
                if text.strip() and len(text.strip().split()) > 10
            ]

            # IIRC has a positional bias. 70% of the times, the first
            # is the supporting one, and almost all are in 1st 20.
            # So we scramble them to make it more challenging retrieval
            # problem.
            paragraph_indices_and_texts = [
                (paragraph_index, paragraph_text)
                for paragraph_index, paragraph_text in enumerate(paragraph_texts)
            ]
            random.shuffle(paragraph_indices_and_texts)
            for paragraph_index, paragraph_text in paragraph_indices_and_texts:
                url = ""
                id_ = hash_object(title + paragraph_text)
                is_abstract = paragraph_index == 0
                es_paragraph = {
                    "id": id_,
                    "title": title,
                    "paragraph_index": paragraph_index,
                    "paragraph_text": paragraph_text,
                    "url": url,
                    "is_abstract": is_abstract,
                }
                document = {
                    "_op_type": "create",
                    "_index": elasticsearch_index,
                    "_id": metadata["idx"],
                    "_source": es_paragraph,
                }
                yield (document)
                metadata["idx"] += 1


def make_2wikimultihopqa_documents(
    raw_data_dir: Path, elasticsearch_index: str, metadata: Optional[Dict] = None
) -> Iterable[Dict[str, Any]]:
    raw_filepaths = [
        raw_data_dir.joinpath("2wikimultihopqa", "train.json"),
        raw_data_dir.joinpath("2wikimultihopqa", "dev.json"),
        raw_data_dir.joinpath("2wikimultihopqa", "test.json"),
    ]
    metadata = metadata or {"idx": 1}
    assert "idx" in metadata

    used_full_ids = set()
    for raw_filepath in raw_filepaths:

        with open(raw_filepath, "r") as file:
            full_data = json.load(file)
            for instance in tqdm(full_data):

                for paragraph in instance["context"]:

                    title = paragraph[0]
                    paragraph_text = " ".join(paragraph[1])
                    paragraph_index = 0
                    url = ""
                    is_abstract = paragraph_index == 0

                    full_id = hash_object(" ".join([title, paragraph_text]))
                    if full_id in used_full_ids:
                        continue

                    used_full_ids.add(full_id)
                    id_ = full_id[:32]

                    es_paragraph = {
                        "id": id_,
                        "title": title,
                        "paragraph_index": paragraph_index,
                        "paragraph_text": paragraph_text,
                        "url": url,
                        "is_abstract": is_abstract,
                    }
                    document = {
                        "_op_type": "create",
                        "_index": elasticsearch_index,
                        "_id": metadata["idx"],
                        "_source": es_paragraph,
                    }
                    yield (document)
                    metadata["idx"] += 1


def make_musique_documents(
    raw_data_dir: Path, elasticsearch_index: str, metadata: Optional[Dict] = None
) -> Iterable[Dict[str, Any]]:
    raw_filepaths = [
        raw_data_dir.joinpath("musique", "musique_ans_v1.0_dev.jsonl"),
        raw_data_dir.joinpath("musique", "musique_ans_v1.0_test.jsonl"),
        raw_data_dir.joinpath("musique", "musique_ans_v1.0_train.jsonl"),
        raw_data_dir.joinpath("musique", "musique_full_v1.0_dev.jsonl"),
        raw_data_dir.joinpath("musique", "musique_full_v1.0_test.jsonl"),
        raw_data_dir.joinpath("musique", "musique_full_v1.0_train.jsonl"),
    ]
    metadata = metadata or {"idx": 1}
    assert "idx" in metadata

    used_full_ids = set()
    for raw_filepath in raw_filepaths:

        with open(raw_filepath, "r") as file:
            for line in tqdm(file.readlines()):
                if not line.strip():
                    continue
                instance = json.loads(line)

                for paragraph in instance["paragraphs"]:

                    title = paragraph["title"]
                    paragraph_text = paragraph["paragraph_text"]
                    paragraph_index = 0
                    url = ""
                    is_abstract = paragraph_index == 0

                    full_id = hash_object(" ".join([title, paragraph_text]))
                    if full_id in used_full_ids:
                        continue

                    used_full_ids.add(full_id)
                    id_ = full_id[:32]

                    es_paragraph = {
                        "id": id_,
                        "title": title,
                        "paragraph_index": paragraph_index,
                        "paragraph_text": paragraph_text,
                        "url": url,
                        "is_abstract": is_abstract,
                    }
                    document = {
                        "_op_type": "create",
                        "_index": elasticsearch_index,
                        "_id": metadata["idx"],
                        "_source": es_paragraph,
                    }
                    yield (document)
                    metadata["idx"] += 1


def build_index(
    es: Elasticsearch, raw_data_dir: Path, dataset: str, force: bool
) -> None:
    print(f"Building index for '{dataset}'...")

    paragraphs_index_settings = {
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "english",
                },
                "paragraph_index": {"type": "integer"},
                "paragraph_text": {
                    "type": "text",
                    "analyzer": "english",
                },
                "url": {
                    "type": "text",
                    "analyzer": "english",
                },
                "is_abstract": {"type": "boolean"},
            }
        }
    }

    index_exists = es.indices.exists(dataset)
    print(">>", "Index already exists" if index_exists else "Index doesn't exist.")

    # delete index if exists
    if index_exists:
        if not force:
            feedback = input(
                ">> "
                f"Index {dataset} already exists. "
                f"Are you sure you want to delete it?"
            )
            if not (feedback.startswith("y") or feedback == ""):
                exit("Termited by user.")
        es.indices.delete(index=dataset)

    # create index
    print(">>", "Creating Index ...")
    es.indices.create(index=dataset, body=paragraphs_index_settings)

    if dataset == "hotpotqa":
        make_documents = make_hotpotqa_documents
    elif dataset == "iirc":
        make_documents = make_iirc_documents
    elif dataset == "2wikimultihopqa":
        make_documents = make_2wikimultihopqa_documents
    elif dataset == "musique":
        make_documents = make_musique_documents
    else:
        raise Exception(f"Unknown dataset_name {dataset}")

    # Bulk-insert documents into index
    print(">>", "Inserting Paragraphs ...")
    result = bulk(
        es,
        make_documents(raw_data_dir, dataset),
        raise_on_error=True,  # set to true o/w it'll fail silently and only show less docs.
        raise_on_exception=True,  # set to true o/w it'll fail silently and only show less docs.
        max_retries=2,  # it's exp backoff starting 2, more than 2 retries will be too much.
        request_timeout=500,
    )
    es.indices.refresh(dataset)  # actually updates the count.
    document_count = result[0]
    print(">>", f"Index {dataset} is ready. Added {document_count} documents.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index paragraphs in Elasticsearch")
    parser.add_argument(
        "--datasets",
        help="Names of the datasets to build index for",
        action="extend",
        nargs="*",
        type=str,
        choices=REGISTERED_DATASETS,
    )
    parser.add_argument(
        "--raw-data-dir",
        help="Path to raw data directory",
        type=str,
        default="raw_data",
    )
    parser.add_argument(
        "--force",
        help="Force delete before creating new index.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--all",
        "-a",
        help="Build index for all registered datasets.",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    raw_data_dir = Path(args.raw_data_dir).resolve()
    datasets = REGISTERED_DATASETS if args.all else args.datasets

    if len(datasets) == 0:
        print("No dataset to be processed. Exit.")
        exit()

    # Connect elastic-search
    elastic_host = "localhost"
    elastic_port = 9200
    es = Elasticsearch(
        [{"host": elastic_host, "port": elastic_port}],
        max_retries=2,  # it's exp backoff starting 2, more than 2 retries will be too much.
        timeout=500,
        retry_on_timeout=True,
    )

    print(f"Building index for {len(datasets)} datasets:", ", ".join(datasets))
    for dataset in datasets:
        build_index(es, raw_data_dir, dataset, args.force)
