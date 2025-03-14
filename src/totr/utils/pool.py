import asyncio
from typing import Dict, Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")

_DELAY: float = 0.01


class AsyncPool(Generic[K, V]):
    """
    An asynchronous data pool based on dictionary.
    """

    def __init__(self, delay: float = _DELAY) -> None:
        self._pool: Dict[K, V] = {}
        self._delay: float = delay

    async def get(self, key: K, delay: Optional[float] = None) -> V:
        delay = self._delay if delay is None else delay
        while True:
            if key in self._pool:
                return self._pool[key]
            await asyncio.sleep(delay)

    def get_nowait(self, key: K, default: Optional[V] = None) -> Optional[V]:
        return self._pool.get(key, default)

    async def pop(self, key: K, delay: Optional[float] = None) -> V:
        delay = self._delay if delay is None else delay
        while True:
            if key in self._pool:
                return self._pool.pop(key)
            await asyncio.sleep(delay)

    def pop_nowait(self, key: K, default: Optional[V] = None) -> Optional[V]:
        return self._pool.pop(key, default)

    async def put(
        self, key: K, obj: V, delay: Optional[float] = None, exist_ok: bool = False
    ) -> None:
        delay = self._delay if delay is None else delay
        while True:
            if key not in self._pool:
                self._pool[key] = obj
                return
            if not exist_ok:
                raise ValueError("Key already exists")
            await asyncio.sleep(delay)

    def put_nowait(self, key: K, obj: V, exist_ok: bool = False) -> None:
        if (not exist_ok) and (key in self._pool):
            raise ValueError("Key already exists")
        self._pool[key] = obj

    def empty(self) -> bool:
        return len(self._pool) == 0

    def dump(self) -> Dict[K, V]:
        return self._pool.copy()
