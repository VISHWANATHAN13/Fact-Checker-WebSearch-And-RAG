from app.rag.ingestion import load_documents
from app.rag.chunking import chunk_documents
from app.rag.vectorstore import index_documents

def main():
    # 1. Load documents
    documents = load_documents("app/data")

    if not documents:
        print("Add a supported document to app/data first.")
        raise SystemExit(0)

    print(f"Loaded {len(documents)} document sections.")

    # 2. Split documents into chunks
    chunks = chunk_documents(
        documents,
        chunk_size=400,
        chunk_overlap=80
    )

    print(f"Created {len(chunks)} chunks.")

    # 3. Preview the first three chunks
    for chunk in chunks[:3]:
        print("\n--- CHUNK ---")
        print("Source:", chunk.metadata.get("source"))
        print("Page:", chunk.metadata.get("page", "N/A"))
        print("Chunk ID:", chunk.metadata.get("chunk_id"))
        print("Content:", chunk.page_content[:250])

    # 4. Embed and store chunks in ChromaDB
    indexed_count = index_documents(chunks)

    print(f"\nIngestion complete. Indexed {indexed_count} chunks.")

if __name__ == "__main__":
    main()
