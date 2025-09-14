from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any

from .retriever import RAGRetriever
from .config import GEMINI_API_KEY, DEFAULT_LLM_MODEL


class SimpleRAG:
    """Simple RAG pipeline with Gemini LLM."""
    
    def __init__(self, retriever: RAGRetriever, 
                 model_name: str = DEFAULT_LLM_MODEL,
                 api_key: str = GEMINI_API_KEY):
        """Initialize the RAG pipeline."""
        self.retriever = retriever
        
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            api_key=api_key, 
            temperature=0.1, 
            max_tokens=1024
        )
    
    def generate_answer(self, query: str, top_k: int = 3) -> str:
        """Generate answer using RAG pipeline."""
        # Retrieve context
        results = self.retriever.retrieve(query=query, top_k=top_k)
        
        # Build context
        context = "\n\n".join([doc['document'] for doc in results]) if results else ''
        
        if not context:
            return "No relevant context retrieved to answer user query."
        
        # Generate answer using LLM
        prompt = f"""Use the following context to answer the question concisely.
        Context:
        {context}

        Question:
        {query}

        Answer:
        """
        
        response = self.llm.invoke([prompt])
        return response.content
