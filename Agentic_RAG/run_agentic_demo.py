"""
Quick Demo Script for Agentic RAG Pipeline
==========================================

This script provides a quick way to run a demonstration of the Agentic RAG system
without the full pipeline execution. Useful for testing and quick demonstrations.

Usage:
    python run_agentic_demo.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Agentic_RAG.agentic_main import AgenticRAGPipeline, SAMPLE_QUERIES

async def quick_demo():
    """Run a quick demonstration with pre-generated data"""
    print("🚀 Quick Agentic RAG Demo")
    print("=" * 50)
    
    # Check if data exists
    data_dir = Path("data")
    required_files = ["hotel_food_sales.csv", "menu_analysis.txt"]
    
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    
    if missing_files:
        print("⚠️ Required data files not found. Running full pipeline first...")
        pipeline = AgenticRAGPipeline()
        await pipeline.run_complete_pipeline()
    else:
        print("✅ Found existing data files. Loading system...")
        pipeline = AgenticRAGPipeline()
        await pipeline._load_documents()
        await pipeline._initialize_agentic_system()
        await pipeline._process_documents()
    
    # Test with a few sample queries
    demo_queries = SAMPLE_QUERIES[:5]
    
    print(f"\n🧪 Testing with {len(demo_queries)} sample queries...")
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 40)
        
        try:
            response = await pipeline.agentic_rag.process_query(query)
            
            print(f"✅ Success!")
            print(f"🎯 Confidence: {response['confidence']}")
            print(f"📊 Quality: {response['quality_assessment']['score']}/10")
            print(f"📝 Answer: {response['answer'][:200]}...")
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:100]}...")
    
    print(f"\n✅ Demo completed! Run 'python agentic_main.py' for full experience.")

if __name__ == "__main__":
    asyncio.run(quick_demo())
