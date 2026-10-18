from app.rag.vectorstore import search_documents, get_vectorstore
import os


def main():
    query = "Who won the 2003 Cricket World Cup final?"

    print(f"Query: {query}")
    print("=" * 60)

    # Verify the active vector store configuration
    vectorstore = get_vectorstore()
    print("Chroma collection:", vectorstore._collection.name)
    print("Embedding model:", os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    print("=" * 60)

    # Retrieve relevant documents
    results = search_documents(query=query, k=5, vectorstore=vectorstore)

    if not results:
        print("No relevant documents found.")
        return

    for rank, doc in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print("Source:", doc.metadata.get("source", "Unknown"))
        print("Chunk ID:", doc.metadata.get("chunk_id", "N/A"))
        print("Content:", doc.page_content)


if __name__ == "__main__":
    main()