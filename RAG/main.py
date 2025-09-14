"""
Main workflow module for Traditional RAG Implementation
This module demonstrates the complete RAG pipeline workflow.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

from .data_ingestion import DocumentLoader, split_documents
from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .retriever import RAGRetriever
from .rag_pipeline import SimpleRAG
from .config import (
    DATA_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    DEFAULT_EMBEDDING_MODEL, DEFAULT_COLLECTION_NAME
)


class RAGWorkflow:
    """
    Main workflow class that orchestrates the entire RAG pipeline.
    """
    
    def __init__(self, 
                 data_dir: str = None,
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                 collection_name: str = DEFAULT_COLLECTION_NAME,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        """
        Initialize the RAG workflow.
        
        Args:
            data_dir: Directory containing documents to process
            embedding_model: Name of the embedding model to use
            collection_name: Name of the vector store collection
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        self.data_dir = data_dir or str(DATA_DIR)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.doc_loader = None
        self.embedding_manager = None
        self.vector_store = None
        self.retriever = None
        self.rag_pipeline = None
        
        # Data storage
        self.documents = []
        self.chunks = []
        
        print(f"RAG Workflow initialized with:")
        print(f"  Data Directory: {self.data_dir}")
        print(f"  Embedding Model: {self.embedding_model}")
        print(f"  Collection Name: {self.collection_name}")
        print(f"  Chunk Size: {self.chunk_size}")
        print(f"  Chunk Overlap: {self.chunk_overlap}")
    
    def setup_components(self):
        """Initialize all RAG components."""
        print("\n" + "="*50)
        print("SETTING UP RAG COMPONENTS")
        print("="*50)
        
        # Initialize document loader
        print("\n1. Initializing Document Loader...")
        self.doc_loader = DocumentLoader(data_dir=self.data_dir)
        
        # Initialize embedding manager
        print("\n2. Initializing Embedding Manager...")
        self.embedding_manager = EmbeddingManager(model_name=self.embedding_model)
        
        # Initialize vector store
        print("\n3. Initializing Vector Store...")
        self.vector_store = VectorStore(collection_name=self.collection_name)
        
        # Initialize retriever
        print("\n4. Initializing RAG Retriever...")
        self.retriever = RAGRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager
        )
        
        # Initialize RAG pipeline
        print("\n5. Initializing RAG Pipeline...")
        self.rag_pipeline = SimpleRAG(retriever=self.retriever)
        
        print("\n✅ All components initialized successfully!")
    
    def load_and_process_documents(self):
        """Load documents from various formats and split into chunks."""
        print("\n" + "="*50)
        print("DOCUMENT LOADING AND PROCESSING")
        print("="*50)
        
        # Load documents
        print("\n1. Loading documents from all supported formats...")
        self.documents = self.doc_loader.load_all_documents()
        print(f"✅ Loaded {len(self.documents)} documents")
        
        if not self.documents:
            print("⚠️  No documents found! Please check your data directory.")
            return False
        
        # Split documents into chunks
        print(f"\n2. Splitting documents into chunks (size: {self.chunk_size}, overlap: {self.chunk_overlap})...")
        self.chunks = split_documents(
            self.documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        print(f"✅ Created {len(self.chunks)} chunks")
        
        return True
    
    def create_embeddings_and_store(self):
        """Generate embeddings and store in vector database."""
        print("\n" + "="*50)
        print("EMBEDDING GENERATION AND STORAGE")
        print("="*50)
        
        if not self.chunks:
            print("❌ No chunks available. Please load and process documents first.")
            return False
        
        # Extract text from chunks
        print("\n1. Extracting text content from chunks...")
        texts = [chunk.page_content for chunk in self.chunks]
        print(f"✅ Extracted text from {len(texts)} chunks")
        
        # Generate embeddings
        print("\n2. Generating embeddings...")
        embeddings = self.embedding_manager.generate_embeddings(texts)
        
        if embeddings is None:
            print("❌ Failed to generate embeddings")
            return False
        
        # Store in vector database
        print("\n3. Storing embeddings in vector database...")
        self.vector_store.add_documents(self.chunks, embeddings)
        print("✅ Successfully stored all embeddings in vector database")
        
        return True
    
    def query_rag_system(self, query: str, top_k: int = 3, show_retrieved_docs: bool = True) -> str:
        """
        Query the RAG system and return the answer.
        
        Args:
            query: Question to ask the RAG system
            top_k: Number of documents to retrieve
            show_retrieved_docs: Whether to display retrieved documents
            
        Returns:
            Generated answer from the RAG system
        """
        print("\n" + "="*50)
        print("QUERYING RAG SYSTEM")
        print("="*50)
        
        print(f"\n📝 Query: {query}")
        
        if not self.rag_pipeline:
            print("❌ RAG pipeline not initialized. Please run setup_components() first.")
            return "Error: RAG system not properly initialized."
        
        # Show retrieved documents if requested
        if show_retrieved_docs:
            print(f"\n🔍 Retrieving top {top_k} relevant documents...")
            retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
            
            if retrieved_docs:
                print(f"\n📚 Retrieved {len(retrieved_docs)} documents:")
                for i, doc in enumerate(retrieved_docs, 1):
                    print(f"\n  Document {i}:")
                    print(f"    Similarity Score: {doc['similarity_score']:.4f}")
                    print(f"    Content Preview: {doc['document'][:200]}...")
                    if doc['metadata']:
                        print(f"    Source: {doc['metadata'].get('source', 'Unknown')}")
            else:
                print("⚠️  No relevant documents found for the query.")
        
        # Generate answer
        print(f"\n🤖 Generating answer using RAG pipeline...")
        answer = self.rag_pipeline.generate_answer(query, top_k=top_k)
        
        print(f"\n💬 Answer: {answer}")
        
        return answer
    
    def interactive_query_session(self):
        """Start an interactive query session where users can ask questions."""
        print("\n" + "="*60)
        print("🤖 INTERACTIVE RAG QUERY SESSION")
        print("="*60)
        print("Enter your questions below. Type 'quit', 'exit', or 'q' to stop.")
        print("Type 'info' to see system information.")
        print("="*60)
        
        if not self.rag_pipeline:
            print("❌ RAG system not ready. Please run the complete workflow first.")
            return
        
        while True:
            try:
                # Get user input
                user_query = input("\n🔍 Your Question: ").strip()
                
                # Check for exit commands
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using the RAG system! Goodbye!")
                    break
                
                # Check for info command
                if user_query.lower() == 'info':
                    self.get_system_info()
                    continue
                
                # Skip empty queries
                if not user_query:
                    print("⚠️ Please enter a question.")
                    continue
                
                # Process the query
                print(f"\n🔄 Processing your question...")
                answer = self.query_rag_system(user_query, top_k=3, show_retrieved_docs=False)
                
                # Show quick result
                print(f"\n💡 Quick Answer: {answer}")
                
                # Ask if user wants to see retrieved documents
                show_docs = input("\n📚 Would you like to see the source documents? (y/n): ").strip().lower()
                if show_docs in ['y', 'yes']:
                    retrieved_docs = self.retriever.retrieve(user_query, top_k=3)
                    if retrieved_docs:
                        print(f"\n📚 Source Documents:")
                        for i, doc in enumerate(retrieved_docs, 1):
                            print(f"\n  📄 Document {i}:")
                            print(f"     Similarity: {doc['similarity_score']:.4f}")
                            print(f"     Preview: {doc['document'][:300]}...")
                            if doc['metadata']:
                                print(f"     Source: {doc['metadata'].get('source', 'Unknown')}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error processing query: {str(e)}")
                print("Please try again with a different question.")

    def run_complete_workflow(self, sample_queries: Optional[List[str]] = None, interactive: bool = True):
        """
        Run the complete RAG workflow from start to finish.
        
        Args:
            sample_queries: List of sample queries to test the system
            interactive: Whether to start interactive session after setup
        """
        print("\n" + "="*60)
        print("🚀 STARTING COMPLETE RAG WORKFLOW")
        print("="*60)
        
        try:
            # Step 1: Setup components
            self.setup_components()
            
            # Step 2: Load and process documents
            success = self.load_and_process_documents()
            if not success:
                return
            
            # Step 3: Create embeddings and store
            success = self.create_embeddings_and_store()
            if not success:
                return
            
            # Step 4: Test with sample queries if provided
            if sample_queries:
                print("\n" + "="*50)
                print("TESTING WITH SAMPLE QUERIES")
                print("="*50)
                
                for i, query in enumerate(sample_queries, 1):
                    print(f"\n{'='*20} Query {i} {'='*20}")
                    self.query_rag_system(query, top_k=3, show_retrieved_docs=False)
            
            print("\n" + "="*60)
            print("✅ RAG WORKFLOW COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            # Start interactive session if requested
            if interactive:
                self.interactive_query_session()
            else:
                print("\nYour RAG system is now ready to use!")
                print("You can query it using: workflow.query_rag_system('Your question here')")
                print("Or start interactive session with: workflow.interactive_query_session()")
            
        except Exception as e:
            print(f"\n❌ Error in workflow: {str(e)}")
            raise
    
    def get_system_info(self):
        """Display information about the current RAG system."""
        print("\n" + "="*50)
        print("RAG SYSTEM INFORMATION")
        print("="*50)
        
        print(f"📁 Data Directory: {self.data_dir}")
        print(f"🤖 Embedding Model: {self.embedding_model}")
        print(f"🗃️  Collection Name: {self.collection_name}")
        print(f"📄 Documents Loaded: {len(self.documents)}")
        print(f"🔧 Chunks Created: {len(self.chunks)}")
        print(f"⚙️  Chunk Size: {self.chunk_size}")
        print(f"🔄 Chunk Overlap: {self.chunk_overlap}")
        
        if self.vector_store and self.vector_store.collection:
            doc_count = self.vector_store.collection.count()
            print(f"🗄️  Documents in Vector Store: {doc_count}")
        
        print(f"🎯 System Status: {'✅ Ready' if self.rag_pipeline else '❌ Not Ready'}")


def main():
    """
    Main function to demonstrate the RAG workflow.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Traditional RAG System')
    parser.add_argument('--no-interactive', action='store_true', 
                       help='Skip interactive session after setup')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing documents to process')
    parser.add_argument('--query', type=str, default=None,
                       help='Single query to process and exit')
    
    args = parser.parse_args()
    
    # Initialize workflow
    workflow = RAGWorkflow(data_dir=args.data_dir)
    
    if args.query:
        # Single query mode
        print("🚀 Setting up RAG system for single query...")
        workflow.setup_components()
        workflow.load_and_process_documents()
        workflow.create_embeddings_and_store()
        
        print(f"\n🔍 Processing query: {args.query}")
        answer = workflow.query_rag_system(args.query, top_k=3)
        print(f"\n✅ Query processing complete!")
    else:
        # Full workflow with optional interactive session
        interactive = not args.no_interactive
        workflow.run_complete_workflow(interactive=interactive)
    
    return workflow


if __name__ == "__main__":
    workflow = main()
