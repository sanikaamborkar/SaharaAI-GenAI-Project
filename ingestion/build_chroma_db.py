import math
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# CONFIG — adjust these if your paths/names differ
# ---------------------------------------------------------------------------
CSV_PATH = "master_rag_documents.csv"
CHROMA_PERSIST_DIR = "./chroma_db"          # where the DB is saved on disk
COLLECTION_NAME = "mental_health_kb"

# Switched from Ollama's nomic-embed-text to sentence-transformers:
# runs in-process (no background server needed), which is required for
# deployment on Hugging Face Spaces / Streamlit Cloud where you can't run
# a persistent Ollama daemon. all-MiniLM-L6-v2 is small (~80MB), fast, and
# well-established for semantic search.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Rough chars-per-token ~4 for English text, so 400 tokens ≈ 1600 chars
CHUNK_SIZE = 1600        # characters, ~400 tokens
CHUNK_OVERLAP = 400      # characters, ~100 tokens

BATCH_SIZE = 100         # how many chunks to add to Chroma per call

# Metadata columns to carry over from the CSV onto every chunk
METADATA_COLUMNS = [
    "source", "condition_tag", "section", "resource_type",
    "url", "collection", "country", "phone", "doc_id",
]


def clean_metadata_value(value):
    """Chroma metadata can't store NaN/None-as-float; coerce to '' or native types."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def load_and_chunk(csv_path: str):
    """Read the CSV and split each row's content into character-based chunks."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_ids = []
    all_documents = []
    all_metadatas = []

    for _, row in df.iterrows():
        content = str(row["content"])
        sub_chunks = splitter.split_text(content)

        for i, chunk_text in enumerate(sub_chunks):
            chunk_id = f"{row['doc_id']}_{i}"
            metadata = {col: clean_metadata_value(row.get(col)) for col in METADATA_COLUMNS}
            metadata["sub_chunk_num"] = i
            metadata["total_sub_chunks"] = len(sub_chunks)

            all_ids.append(chunk_id)
            all_documents.append(chunk_text)
            all_metadatas.append(metadata)

    print(f"Produced {len(all_documents)} chunks from {len(df)} source rows "
          f"(chunk_size={CHUNK_SIZE} chars, overlap={CHUNK_OVERLAP} chars)")
    return all_ids, all_documents, all_metadatas


def build_collection(ids, documents, metadatas):
    """Embed and persist everything into a Chroma collection."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # In-process embedding model — no server, no network call, works
    # identically whether run locally or inside a deployed container.
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )

    # Fresh build each run: drop any existing collection with the same name
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(documents)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  added chunks {start}-{end - 1} / {total}")

    print(f"\nChroma collection '{COLLECTION_NAME}' persisted to '{CHROMA_PERSIST_DIR}' "
          f"with {collection.count()} chunks.")
    return collection


def sanity_check_query(collection):
    """Quick smoke test so you can confirm retrieval works before wiring up the Retriever agent."""
    test_query = "I feel hopeless and can't sleep, what is depression?"
    results = collection.query(query_texts=[test_query], n_results=3)

    print(f"\nSanity check query: {test_query!r}")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )):
        print(f"\n  Result {i + 1} (distance={dist:.4f})")
        print(f"    source: {meta.get('source')} | condition: {meta.get('condition_tag')} "
              f"| section: {meta.get('section')}")
        print(f"    text: {doc[:200]}...")


if __name__ == "__main__":
    ids, documents, metadatas = load_and_chunk(CSV_PATH)
    collection = build_collection(ids, documents, metadatas)
    sanity_check_query(collection)
