"""Lazy ownership boundary for injectable async HTTP clients."""

from __future__ import annotations

import inspect
from typing import Any


class LazyAsyncHttpClient:
    """Create an httpx client on first use and close only clients we own."""

    def __init__(self, client: Any | None = None, **client_kwargs: Any):
        self._client = client
        self._client_kwargs = client_kwargs
        self._owns_client = client is None

    def get(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(**self._client_kwargs)
        return self._client

    async def aclose(self) -> None:
        if not self._owns_client or self._client is None:
            return
        client, self._client = self._client, None
        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result
