import requests


OPENALEX_URL = "https://api.openalex.org/works"


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


def search_papers(query: str) -> str:
    """Search OpenAlex for papers relevant to a research query."""

    params = {
        "search": query,
        "per-page": 5,
        "select": (
            "id,doi,display_name,publication_year,"
            "authorships,abstract_inverted_index,"
            "cited_by_count"
        ),
    }

    try:
        response = requests.get(
            OPENALEX_URL,
            params=params,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()

    except requests.RequestException as exc:
        return f"OpenAlex search failed: {exc}"

    results = data.get("results", [])

    if not results:
        return f"No papers found for query: {query}"

    papers = []

    for i, paper in enumerate(results, start=1):
        title = paper.get("display_name", "Unknown title")
        year = paper.get("publication_year", "Unknown year")
        doi = paper.get("doi")
        citations = paper.get("cited_by_count", 0)

        authors = [
            author.get("author", {}).get("display_name")
            for author in paper.get("authorships", [])
            if author.get("author", {}).get("display_name")
        ]

        abstract = reconstruct_abstract(
            paper.get("abstract_inverted_index")
        )

        paper_text = [
            f"{i}. {title}",
            f"   Year: {year}",
            f"   Authors: {', '.join(authors[:5])}",
            f"   Citations: {citations}",
        ]

        if doi:
            paper_text.append(f"   DOI: {doi}")

        if abstract:
            paper_text.append(f"   Abstract: {abstract}")

        papers.append("\n".join(paper_text))

    return "\n\n".join(papers)