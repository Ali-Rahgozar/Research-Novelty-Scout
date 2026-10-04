import json





def search_papers(query: str) -> str:
    """Temporary fake search tool used to test the agent loop."""
    papers = [
        {
            "title": "A Study on Tokenization Efficiency in Multilingual Language Models",
            "authors": ["Alice Smith", "Bob Chen"],
            "year": 2025,
            "abstract": "We investigate how subword tokenization affects token efficiency across languages in multilingual language models.",
            "relevance": "high"
        },
        {
            "title": "Language-Specific Tokenizers for Low-Resource NLP",
            "authors": ["Carol Jones"],
            "year": 2024,
            "abstract": "We compare multilingual and language-specific tokenizers for several low-resource languages.",
            "relevance": "medium"
        }
    ]

    return json.dumps(
        {
            "query": query,
            "results": papers
        },
        ensure_ascii=False
    )


TOOLS = {
    "search_papers": search_papers,
}


