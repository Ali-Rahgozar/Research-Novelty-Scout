import json
import os
import requests
from pathlib import Path


OPENALEX_URL = "https://api.openalex.org/works"

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = json.load(f)

RESULTS_PER_QUERY = config["results_per_query"]

# The key raises the rate limit (30 req/s vs 10) and is required for paging
# past the first page. config.json wins; the env var is a convenience fallback.
OPENALEX_API_KEY = config.get("openalex_api_key") or os.getenv("OPENALEX_API_KEY")


def reconstruct_abstract(inverted_index):
    """Reconstruct an abstract from OpenAlex's inverted index format."""
    if not inverted_index:
        return None

    words = []

    for word, positions in inverted_index.items():
        for position in positions:
            words.append((position, word))

    words.sort()

    return " ".join(word for _, word in words)


def fetch_papers(query: str, per_page: int = RESULTS_PER_QUERY) -> list[dict]:
    """Query OpenAlex and return structured paper records.

    Raises requests.RequestException on network / HTTP errors.
    """
    params = {
        "search": query,
        "per-page": per_page,
        "select": (
            "id,doi,display_name,publication_year,"
            "authorships,abstract_inverted_index,"
            "cited_by_count"
        ),
    }

    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY

    response = requests.get(OPENALEX_URL, params=params, timeout=30)
    response.raise_for_status()

    papers = []

    for paper in response.json().get("results", []):
        authors = [
            a.get("author", {}).get("display_name")
            for a in paper.get("authorships", [])
            if a.get("author", {}).get("display_name")
        ]

        papers.append({
            "id": paper.get("id"),
            "doi": paper.get("doi"),
            "title": paper.get("display_name") or "Unknown title",
            "year": paper.get("publication_year") or "Unknown year",
            "authors": authors[:5],
            "citations": paper.get("cited_by_count", 0),
            "abstract": reconstruct_abstract(
                paper.get("abstract_inverted_index")
            ),
        })

    return papers


def format_papers(papers: list[dict]) -> str:
    """Render paper records as the text the LLM sees."""
    blocks = []

    for i, paper in enumerate(papers, start=1):
        lines = [
            f"{i}. {paper['title']}",
            # The system prompt asks the model to dedupe by OpenAlex ID, so the
            # ID has to actually be visible in the tool output.
            f"   OpenAlex ID: {paper['id']}",
            f"   Year: {paper['year']}",
            f"   Authors: {', '.join(paper['authors'])}",
            f"   Citations: {paper['citations']}",
        ]

        if paper["doi"]:
            lines.append(f"   DOI: {paper['doi']}")

        if paper["abstract"]:
            lines.append(f"   Abstract: {paper['abstract']}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def search_papers_with_data(query: str) -> tuple[str, list[dict]]:
    """Return (text for the LLM, structured records for logging / eval)."""
    try:
        papers = fetch_papers(query)
    except requests.RequestException as exc:
        return f"OpenAlex search failed: {exc}", []

    if not papers:
        return f"No papers found for query: {query}", []

    return format_papers(papers), papers


def search_papers(query: str) -> str:
    """Search OpenAlex for papers relevant to a research query."""
    return search_papers_with_data(query)[0]