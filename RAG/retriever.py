from typing import List, Dict, Any

from .vector_store import VectorStore
from .embeddings import EmbeddingManager
from .config import DEFAULT_TOP_K, DEFAULT_SCORE_THRESHOLD


class RAGRetriever:
    """Retriever Pipeline From Vectorstore."""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """Initialize the RAGRetriever with vector store and embedding manager."""
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K, 
                score_threshold: float = DEFAULT_SCORE_THRESHOLD) -> List[Dict[str, Any]]:
        """Retrieve the most relevant documents for a given query."""
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top_k: {top_k}, Score Threshold: {score_threshold}")

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        if query_embedding is None:
            print("Error generating query embedding.")
            return []

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, doc, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "document": doc,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents above the score threshold {score_threshold}.")
            else:
                print("No documents found in the vector store.")

            return retrieved_docs
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
