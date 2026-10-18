
from hashlib import sha256

from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.rag.vectorstore import get_vectorstore


def _document_key(document: Document) -> str:
    """Create a stable key to identify the same retrieved chunk."""

    source = document.metadata.get(
        "source_path",
        document.metadata.get("source", "unknown"),
    )
    page = document.metadata.get("page", "")
    content = document.page_content

    unique_content = f"{source}|{page}|{content}"

    return sha256(unique_content.encode("utf-8")).hexdigest()


def _load_indexed_documents(
    vectorstore: Chroma,
) -> list[Document]:
    """Load indexed text and metadata from the Chroma collection."""

    records = vectorstore.get(
        include=["documents", "metadatas"]
    )

    contents = records.get("documents") or []
    metadatas = records.get("metadatas") or []

    documents = []

    for content, metadata in zip(contents, metadatas):
        if not content:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata=metadata or {},
            )
        )

    return documents


def _reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    rrf_k: int = 60,
) -> list[Document]:
    """Merge ranked lists using Reciprocal Rank Fusion."""

    scores: dict[str, float] = {}
    documents_by_key: dict[str, Document] = {}

    for ranked_documents in ranked_lists:
        seen_in_list = set()

        for rank, document in enumerate(ranked_documents, start=1):
            key = _document_key(document)

            # Avoid counting the same chunk twice in one ranking.
            if key in seen_in_list:
                continue

            seen_in_list.add(key)

            scores[key] = scores.get(key, 0.0) + (
                1.0 / (rrf_k + rank)
            )

            documents_by_key[key] = document

    sorted_keys = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    fused_documents = []

    for key in sorted_keys:
        document = documents_by_key[key]

        # Preserve the original document and its metadata.
        metadata = dict(document.metadata)
        metadata["rrf_score"] = round(scores[key], 6)

        fused_documents.append(
            Document(
                page_content=document.page_content,
                metadata=metadata,
            )
        )

    return fused_documents


def hybrid_search(
    query: str,
    k: int = 5,
    fetch_k: int = 10,
    rrf_k: int = 60,
    vectorstore: Chroma | None = None,
) -> list[Document]:
    """
    Retrieve evidence using semantic search and BM25,
    then merge the rankings with Reciprocal Rank Fusion.

    Args:
        query: User's search query.
        k: Number of final results to return.
        fetch_k: Number of candidates retrieved by each method.
        rrf_k: RRF smoothing constant.
        vectorstore: Optional existing Chroma vector store.

    Returns:
        Documents ranked by their combined RRF scores.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if k <= 0 or fetch_k <= 0:
        raise ValueError("k and fetch_k must be greater than zero.")

    if rrf_k < 0:
        raise ValueError("rrf_k cannot be negative.")

    store = vectorstore or get_vectorstore()

    # 1. Load the indexed corpus for keyword retrieval.
    corpus_documents = _load_indexed_documents(store)

    if not corpus_documents:
        print("ChromaDB contains no indexed documents.")
        return []

    # 2. Semantic retrieval using Chroma embeddings.
    semantic_results = store.similarity_search(
        query=query,
        k=fetch_k,
    )

    # 3. Keyword retrieval using BM25.
    bm25_retriever = BM25Retriever.from_documents(
        corpus_documents,
        k=fetch_k,
    )

    bm25_results = bm25_retriever.invoke(query)

    # 4. Combine both ranked lists using RRF.
    fused_results = _reciprocal_rank_fusion(
        [semantic_results, bm25_results],
        rrf_k=rrf_k,
    )

    return fused_results[:k]
