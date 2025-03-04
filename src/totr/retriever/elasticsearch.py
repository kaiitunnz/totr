from collections import OrderedDict
from typing import Any, Dict, List, Optional

from elasticsearch import AsyncElasticsearch

from ..config import Config
from .base import BaseRetriever
from .registry import RetrieverRegistry


@RetrieverRegistry.register("elasticsearch")
class ElasticsearchRetriever(BaseRetriever):
    """
    Some useful resources for constructing ES queries:
    # https://stackoverflow.com/questions/28768277/elasticsearch-difference-between-must-and-should-bool-query
    # https://stackoverflow.com/questions/49826587/elasticsearch-query-to-match-two-different-fields-with-exact-values

    # bool/must acts as AND
    # bool/should acts as OR
    # bool/filter acts as binary filter w/o score (unlike must and should).
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config=config)
        self.client = AsyncElasticsearch(config.retriever.elasticsearch_url, timeout=30)

    async def retrieve(
        self,
        query_text: str,
        max_hits_count: int = 3,
        max_buffer_count: int = 100,
        document_type: str = "paragraph_text",
        allowed_titles: Optional[List[str]] = None,
        allowed_paragraph_types: Optional[List[str]] = None,
        paragraph_index: Optional[int] = None,
        corpus_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if document_type not in ("title", "paragraph_text", "title_paragraph_text"):
            raise ValueError("Invalid document_type.")

        if paragraph_index is not None and document_type != "paragraph_text":
            raise ValueError(
                "paragraph_index not valid for the document_type of paragraph_text."
            )

        if document_type in ("paragraph_text", "title_paragraph_text"):
            is_abstract = (
                True if corpus_name == "hotpotqa" else None
            )  # Note "None" and not False
            query_title_field_too = document_type == "title_paragraph_text"
            paragraphs_results = await self.retrieve_paragraphs(
                query_text=query_text,
                is_abstract=is_abstract,
                max_hits_count=max_hits_count,
                allowed_titles=allowed_titles,
                allowed_paragraph_types=allowed_paragraph_types,
                paragraph_index=paragraph_index,
                corpus_name=corpus_name,
                query_title_field_too=query_title_field_too,
                max_buffer_count=max_buffer_count,
            )

        elif document_type == "title":
            paragraphs_results = await self.retrieve_titles(
                query_text=query_text,
                max_hits_count=max_hits_count,
                corpus_name=corpus_name,
            )

        return paragraphs_results

    async def retrieve_paragraphs(
        self,
        query_text: Optional[str] = None,
        corpus_name: Optional[str] = None,
        is_abstract: Optional[bool] = None,
        allowed_titles: Optional[List[str]] = None,
        allowed_paragraph_types: Optional[List[str]] = None,
        query_title_field_too: Optional[bool] = False,
        paragraph_index: Optional[int] = None,
        max_buffer_count: int = 100,
        max_hits_count: int = 10,
    ) -> List[Dict]:

        query: Dict[str, Any] = {
            "size": max_buffer_count,
            # what records are needed in result
            "_source": [
                "id",
                "title",
                "paragraph_text",
                "url",
                "is_abstract",
                "paragraph_index",
            ],
            "query": {
                "bool": {
                    "should": [],
                    "must": [],
                }
            },
        }

        if query_text is not None:
            # must is too strict for this:
            query["query"]["bool"]["should"].append(
                {"match": {"paragraph_text": query_text}}
            )

        if query_title_field_too:
            query["query"]["bool"]["should"].append({"match": {"title": query_text}})

        if is_abstract is not None:
            query["query"]["bool"]["filter"] = [{"match": {"is_abstract": is_abstract}}]

        if allowed_titles is not None:
            if len(allowed_titles) == 1:
                query["query"]["bool"]["must"] += [
                    {"match": {"title": _title}} for _title in allowed_titles
                ]
            else:
                query["query"]["bool"]["should"] += [
                    {"bool": {"must": {"match": {"title": _title}}}}
                    for _title in allowed_titles
                ]

        if allowed_paragraph_types is not None:
            if len(allowed_paragraph_types) == 1:
                query["query"]["bool"]["must"] += [
                    {"match": {"paragraph_type": _paragraph_type}}
                    for _paragraph_type in allowed_paragraph_types
                ]
            else:
                query["query"]["bool"]["should"] += [
                    {"bool": {"must": {"match": {"title": _paragraph_type}}}}
                    for _paragraph_type in allowed_paragraph_types
                ]

        if paragraph_index is not None:
            query["query"]["bool"]["should"].append(
                {"match": {"paragraph_index": paragraph_index}}
            )

        assert query["query"]["bool"]["should"] or query["query"]["bool"]["must"]

        if not query["query"]["bool"]["must"]:
            query["query"]["bool"].pop("must")

        if not query["query"]["bool"]["should"]:
            query["query"]["bool"].pop("should")

        result: Any = await self.client.search(index=corpus_name, body=query)

        retrieval = []
        if result.get("hits") is not None and result["hits"].get("hits") is not None:
            retrieval = result["hits"]["hits"]
            text2retrieval = OrderedDict()
            for item in retrieval:
                text = item["_source"]["paragraph_text"].strip().lower()
                text2retrieval[text] = item
            retrieval = list(text2retrieval.values())

        retrieval = sorted(retrieval, key=lambda e: e["_score"], reverse=True)
        retrieval = retrieval[:max_hits_count]
        for retrieval_ in retrieval:
            retrieval_["_source"]["score"] = retrieval_["_score"]
        retrieval = [e["_source"] for e in retrieval]

        if allowed_titles is not None:
            lower_allowed_titles = [e.lower().strip() for e in allowed_titles]
            retrieval = [
                item
                for item in retrieval
                if item["title"].lower().strip() in lower_allowed_titles
            ]

        for retrieval_ in retrieval:
            retrieval_["corpus_name"] = corpus_name

        return retrieval

    async def retrieve_titles(
        self,
        query_text: str,
        corpus_name: Optional[str] = None,
        max_buffer_count: int = 100,
        max_hits_count: int = 10,
    ) -> List[Dict]:

        query = {
            "size": max_buffer_count,
            # what records are needed in the result.
            "_source": [
                "id",
                "title",
                "paragraph_text",
                "url",
                "is_abstract",
                "paragraph_index",
            ],
            "query": {
                "bool": {
                    "must": [
                        {"match": {"title": query_text}},
                    ],
                    "filter": [
                        {
                            "match": {"is_abstract": True}
                        },  # so that same title doesn't show up many times.
                    ],
                }
            },
        }

        result: Any = await self.client.search(index=corpus_name, body=query)

        retrieval = []
        if result.get("hits") is not None and result["hits"].get("hits") is not None:
            retrieval = result["hits"]["hits"]
            text2retrieval = OrderedDict()
            for item in retrieval:
                text = item["_source"]["title"].strip().lower()
                text2retrieval[text] = item
            retrieval = list(text2retrieval.values())[:max_hits_count]

        retrieval = [e["_source"] for e in retrieval]

        for retrieval_ in retrieval:
            retrieval_["corpus_name"] = corpus_name

        return retrieval
