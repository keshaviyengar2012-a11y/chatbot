"""Web search client used to pull current, citable information for the tutor.

The API key is read from the environment (WEB_SEARCH_API_KEY) and is never stored in the
repository. The default provider is Brave Search; set WEB_SEARCH_URL to use another endpoint
that accepts the same query parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_URL = "https://api.search.brave.com/res/v1/web/search"


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class WebSearchError(RuntimeError):
    pass


class WebSearch:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 8.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("WEB_SEARCH_API_KEY", "")
        self.base_url = base_url or os.environ.get("WEB_SEARCH_URL", DEFAULT_URL)
        self._client = client or httpx.Client(timeout=timeout)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, count: int = 5) -> list[SearchResult]:
        if not self.enabled:
            return []
        try:
            resp = self._client.get(
                self.base_url,
                params={"q": query, "count": count},
                headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise WebSearchError(f"web search failed: {exc}") from exc
        return parse_results(resp.json(), count)


def parse_results(payload: dict[str, Any], count: int) -> list[SearchResult]:
    items = (payload.get("web") or {}).get("results") or []
    results = []
    for item in items[:count]:
        url, title = item.get("url"), item.get("title")
        if not url or not title:
            continue
        snippet = item.get("description") or ""
        results.append(SearchResult(title=title.strip(), url=url, snippet=snippet.strip()))
    return results
