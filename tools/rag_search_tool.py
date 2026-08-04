"""
tools/rag_search_tool.py

Pure retrieval logic against the persisted Chroma collection built by
ingestion/build_index.py. No Agent definition lives here — just the
tool function(s) that any agent can be handed.
"""

import chromadb
from chromadb.utils import embedding_functions
from crewai.tools import tool


CHROMA_PERSIST_DIR = "ingestion/chroma_db"
COLLECTION_NAME = "mental_health_kb"
EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_URL = "http://localhost:11434/api/embeddings"

TOP_K = 3  # how many chunks to retrieve per query 


def _get_collection():
    """Connect to the existing persisted collection (read-only usage)."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    embedding_fn = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_URL,
        model_name=EMBEDDING_MODEL,
    )

    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def _format_results(results) -> str:
    """Turn raw Chroma query results into a clean, citable context block."""
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        return "No relevant passages found in the knowledge base."

    blocks = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        condition = meta.get("condition_tag", "")
        url = meta.get("url", "")

        header = (
            f"[Passage {i}] (source: {source} | condition: {condition} | "
            f"section: {section} | relevance: {1 - dist:.2f})"
        )
        body = doc if not url else f"{doc}\n(url: {url})"
        blocks.append(f"{header}\n{body}")

    return "\n\n".join(blocks)


def search(query: str, condition_tag: str | None = None, country: str | None = None,
           n_results: int = TOP_K) -> str:
    """
    Core retrieval function — usable standalone (for testing) or wrapped as a tool.

    Args:
        query: the user's message / search query
        condition_tag: optional filter, e.g. "depression", "anxiety" (from Planner)
        country: optional filter for locale-specific resources (e.g. helplines)
        n_results: how many chunks to return
    """
    collection = _get_collection()

    where_filter = {}
    if condition_tag:
        where_filter["condition_tag"] = condition_tag
    if country:
        where_filter["country"] = country

    query_kwargs = {"query_texts": [query], "n_results": n_results}
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)
    return _format_results(results)


@tool("Knowledge Base Retriever")
def rag_search_tool(query: str, condition_tag: str = "", country: str = "") -> str:
    """
    Searches the mental health knowledge base (WHO, NIMH, NAMI, Mind UK,
    findahelpline) for passages relevant to the user's message. Use
    condition_tag (e.g. 'depression', 'anxiety') to narrow results when
    the topic is known. Use country to prioritize locale-specific
    resources like helplines. Returns formatted passages with source
    attribution for grounding a response.
    """
    return search(
        query=query,
        condition_tag=condition_tag or None,
        country=country or None,
    )

if __name__ == "__main__":
    test_queries = [
        ("I feel hopeless and can't sleep, what is depression?", None, None),
        ("I'm constantly worried about everything", "anxiety", None),
    ]

    for query, condition, country in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Query: {query!r} | condition_tag={condition} | country={country}")
        print("=" * 70)
        print(search(query, condition_tag=condition, country=country))
