import asyncio
import queue
from queue import Empty, Full
from typing import Generic, Optional, TypeVar, Union

T = TypeVar("T")


class CloseSignal:
    pass


_CLOSE_SIGNAL = CloseSignal()
_DELAY: float = 0


class TSQueue(Generic[T]):
    """
    Asynchronous wrapper for thread-safe queue.
    """

    def __init__(self, delay: float = _DELAY) -> None:
        self._queue: queue.Queue[Union[T, CloseSignal]] = queue.Queue()
        self._delay: float = delay
        self._is_closed: bool = False

    def _unwrap(self, message: Union[T, CloseSignal]) -> T:
        if isinstance(message, CloseSignal):
            self._is_closed = True
            raise asyncio.CancelledError()
        return message

    async def get(self, delay: Optional[float] = None) -> T:
        delay = self._delay if delay is None else delay
        while True:
            try:
                return self.get_nowait()
            except Empty:
                await asyncio.sleep(delay)

    def get_nowait(self) -> T:
        if self._is_closed:
            raise asyncio.CancelledError()
        return self._unwrap(self._queue.get_nowait())

    async def put(self, obj: T, delay: Optional[float] = None) -> None:
        delay = self._delay if delay is None else delay
        while True:
            try:
                self.put_nowait(obj)
                return
            except Full:
                await asyncio.sleep(delay)

    def put_nowait(self, obj: T) -> None:
        if self._is_closed:
            raise asyncio.CancelledError()
        self._queue.put_nowait(obj)

    def close(self) -> None:
        if not self._is_closed:
            self._is_closed = True
            self._queue.put_nowait(_CLOSE_SIGNAL)
