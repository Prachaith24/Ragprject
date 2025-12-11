from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from vectorstore import vector_store
from chunking import chunks
from typing import List

# We inherit from BaseRetriever to get standard LangChain functionality
class HybridRetriever(BaseRetriever):
    # Type hints for the fields (required by BaseRetriever/Pydantic)
    vector_retriever: any
    bm25_retriever: any
    weights: list = [0.5, 0.5]

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """
        This method is called automatically when you use .invoke()
        """
        # 1. Use .invoke() instead of .get_relevant_documents() for the sub-retrievers
        #    This fixes the 'VectorStoreRetriever' error you saw earlier.
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # 2. Combine and deduplicate based on content
        seen = set()
        combined_docs = []
        
        # Add vector results first (weighted higher)
        for doc in vector_docs[:3]:  # Top 3 from vector
            doc_key = doc.page_content[:100]
            if doc_key not in seen:
                seen.add(doc_key)
                combined_docs.append(doc)
        
        # Add BM25 results
        for doc in bm25_docs[:3]:  # Top 3 from BM25
            doc_key = doc.page_content[:100]
            if doc_key not in seen:
                seen.add(doc_key)
                combined_docs.append(doc)
        
        return combined_docs[:5]  # Return top 5 combined

# Vector retriever (ChromaDB)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# BM25 retriever (keyword-based)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# Create hybrid retriever
ensemble_retriever = HybridRetriever(
    vector_retriever=vector_retriever,
    bm25_retriever=bm25_retriever,
    weights=[0.5, 0.5]
)