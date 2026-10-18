import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.llm_config.llm import get_embeddings



load_dotenv()


def get_vectorstore() -> Chroma:
    # vectorstore.py is at app/rag/vectorstore.py
    # parents[2] resolves to the project root.
    project_root = Path(__file__).resolve().parents[2]

    persist_directory = project_root / "app" / "db"
    collection_name = "rag_evidence_openai1536"

    persist_directory.mkdir(parents=True, exist_ok=True)

    print("Chroma directory:", persist_directory)
    print("Chroma collection:", collection_name)
    print("Embedding model:", os.getenv(
        "EMBEDDING_MODEL", "text-embedding-3-small"
    ))

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_directory),
    )



def generate_document_id(document: Document) -> str:
    """Generate a deterministic ID for a document chunk."""

    source = document.metadata.get("source_path", "unknown")
    page = document.metadata.get("page", "")
    content = document.page_content

    unique_content = f"{source}|{page}|{content}"

    return hashlib.sha256(
        unique_content.encode("utf-8")
    ).hexdigest()


def index_documents(
    documents: list[Document],
    vectorstore: Chroma | None = None,
) -> int:
    """Store document chunks in Chroma and return the indexed count."""

    if not documents:
        print("No document chunks to index.")
        return 0

    store = vectorstore or get_vectorstore()

    ids = [
        generate_document_id(document)
        for document in documents
    ]

    store.add_documents(
        documents=documents,
        ids=ids,
    )

    print(f"Indexed {len(documents)} document chunks.")

    return len(documents)


def search_documents(
    query: str,
    k: int = 5,
    vectorstore: Chroma | None = None,
) -> list[Document]:
    """Retrieve the top-k semantically relevant chunks."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    store = vectorstore or get_vectorstore()

    return store.similarity_search(
        query=query,
        k=k,
    )
