
from app.rag.hybrid_retrieval import hybrid_search


def main():
    query = "Who scored 140 not out in the 2003 World Cup final?"

    print("=" * 70)
    print("HYBRID RETRIEVAL TEST")
    print("=" * 70)
    print(f"Query: {query}")

    results = hybrid_search(
        query=query,
        k=5,
        fetch_k=10,
    )

    if not results:
        print("\nNo documents retrieved.")
        return

    print(f"\nRetrieved {len(results)} results.")

    for rank, document in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print("Source:", document.metadata.get("source", "Unknown"))
        print("Chunk ID:", document.metadata.get("chunk_id", "N/A"))
        print("RRF Score:", document.metadata.get("rrf_score", "N/A"))
        print("Evidence:")
        print(document.page_content)

    print("\nHybrid retrieval test completed.")


if __name__ == "__main__":
    main()
