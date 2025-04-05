import asyncio
from collections import OrderedDict
from threading import Thread
from typing import Any, Dict, List, Optional, Union

from elasticsearch import AsyncElasticsearch

from ..config import Config
from ..utils.pool import AsyncPool
from ..utils.queue import TSQueue
from .base import BaseRetriever
from .registry import RetrieverRegistry


class _RetrieverHelper:
    class RetrieverData:
        __slots__ = "task_id", "client", "corpus_name", "query"

        def __init__(
            self,
            task_id: int,
            client: AsyncElasticsearch,
            corpus_name: Optional[str],
            query: Dict[str, Any],
        ) -> None:
            self.task_id = task_id
            self.client = client
            self.corpus_name = corpus_name
            self.query = query

    def __init__(self) -> None:
        self._channel: TSQueue[
            Union[AsyncElasticsearch, _RetrieverHelper.RetrieverData]
        ] = TSQueue()
        self._result_pool: AsyncPool[int, Any] = AsyncPool()
        self._counter: int = 0

        self._worker_thread: Thread = Thread(
            target=lambda: asyncio.run(self._worker_loop()), daemon=True
        )
        self._worker_thread.start()

    async def _worker_loop(self) -> None:
        while True:
            job = await self._channel.get()
            if isinstance(job, AsyncElasticsearch):
                await job.close()
                continue

            result: Any = await job.client.search(index=job.corpus_name, body=job.query)
            await self._result_pool.put(job.task_id, result)

    async def search(
        self, client: AsyncElasticsearch, index: Optional[str], body: Dict[str, Any]
    ) -> Any:
        self._counter += 1
        task_id = self._counter
        await self._channel.put(self.RetrieverData(task_id, client, index, body))
        return await self._result_pool.get(task_id)

    def close(self, client: AsyncElasticsearch):
        self._channel.put_nowait(client)

    def __del__(self):
        self._channel.close()
        self._worker_thread.join()


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

    _helper = _RetrieverHelper()

    def __init__(self, config: Config) -> None:
        super().__init__(config=config)
        self.client = AsyncElasticsearch(
            config.retriever.elasticsearch_url, timeout=None
        )

    def __del__(self) -> None:
        self._helper.close(self.client)

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

        result: Any = await self._helper.search(
            self.client, index=corpus_name, body=query
        )

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

        result: Any = await self._helper.search(
            self.client, index=corpus_name, body=query
        )

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

    async def retrieve_paragraphs_with_title(
        self,
        title: str,
        corpus_name: Optional[str] = None,
        max_retrieval_count: int = 100,
    ) -> List[Dict]:
        query = {
            "size": max_retrieval_count,
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
                        {"match": {"title": title}},
                    ]
                }
            },
        }

        result: Any = await self._helper.search(
            self.client, index=corpus_name, body=query
        )

        title = title.strip().lower()
        retrieval = []
        if result.get("hits") is not None and result["hits"].get("hits") is not None:
            retrieved = result["hits"]["hits"]
            for item in retrieved:
                item_title: str = item["_source"]["title"]
                if title == item_title.strip().lower():
                    retrieval.append(item["_source"])

        for item in retrieval:
            item["corpus_name"] = corpus_name

        return retrieval
