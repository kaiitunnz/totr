from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..config import Config


class BaseRetriever(ABC):
    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
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
        pass
