from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
from pathlib import Path

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    faiss_index_path = Path("faiss_store/faiss.index")
    metadata_path = Path("faiss_store/metadata.pkl")

    if faiss_index_path.exists() and metadata_path.exists():
        store.load()
    else:
        store.build_from_documents(docs)
    
    rag_search = RAGSearch()
    query = "What is LEAN?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)