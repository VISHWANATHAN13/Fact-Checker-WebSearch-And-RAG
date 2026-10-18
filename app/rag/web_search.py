"""External web-search fallback for FactCheck RAG."""

from ddgs import DDGS


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the public web and return normalized evidence records.

    Each result contains a title, URL, snippet, and source type.
    Search results are evidence candidates, not automatically verified facts.
    """
    if not query or not query.strip():
        return []

    try:
        with DDGS() as search_client:
            results = search_client.text(
                query.strip(),
                max_results=max_results,
                safesearch="moderate",
            )

        evidence = []

        for result in results or []:
            url = result.get("href") or result.get("url")
            title = result.get("title", "Untitled source")
            snippet = result.get("body", "")

            if not url or not snippet:
                continue

            evidence.append(
                {
                    "source_type": "web",
                    "source": title,
                    "url": url,
                    "content": snippet,
                }
            )

        return evidence

    except Exception as exc:
        # Do not pretend the search succeeded if networking fails.
        raise RuntimeError(
            f"External web search failed: {exc}"
        ) from exc
