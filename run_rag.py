"""
Simple script to run the RAG system
"""

from RAG import RAGWorkflow

def main():
    """Run the RAG system with interactive queries."""
    print("🚀 Starting Traditional RAG System...")
    
    # Initialize and run the complete workflow
    workflow = RAGWorkflow()
    workflow.run_complete_workflow(interactive=True)

if __name__ == "__main__":
    main()
