"""
Simple script to run the Agentic RAG system
Similar to run_rag.py but for agentic capabilities
"""

from Agentic_RAG.agentic_main import AgenticRAGWorkflow

def main():
    """Run the Agentic RAG system with interactive queries."""
    print("🤖 Starting Agentic RAG System for Hotel Food Sales...")
    
    # Initialize and run the complete workflow
    workflow = AgenticRAGWorkflow(
        data_dir="data",
        embedding_model="all-MiniLM-L6-v2",
        collection_name="hotel_sales_agentic"
    )
    
    success = workflow.run_complete_workflow(interactive=True)
    
    if not success:
        print("❌ Failed to initialize Agentic RAG system")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
