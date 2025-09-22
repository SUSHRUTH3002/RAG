"""
Flask Web Application for Agentic RAG System
Similar to app.py but with agentic capabilities for hotel food sales data
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import uuid
import time
import threading
import asyncio
from datetime import datetime
import traceback
import os
import sys

# Add the agentic module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Agentic_RAG.agentic_main import AgenticRAGWorkflow

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'agentic-rag-secret-key-change-in-production'
CORS(app)

# Global variables for Agentic RAG system
agentic_workflow = None
is_system_ready = False
system_status = "Initializing..."

def initialize_agentic_system():
    """Initialize the Agentic RAG system in a separate thread"""
    global agentic_workflow, is_system_ready, system_status
    
    try:
        system_status = "Loading Agentic RAG components..."
        print("🤖 Initializing Agentic RAG system...")
        
        # Initialize workflow
        agentic_workflow = AgenticRAGWorkflow(
            data_dir="data",
            embedding_model="all-MiniLM-L6-v2",
            collection_name="hotel_sales_agentic_web"
        )
        
        system_status = "Setting up components..."
        success = agentic_workflow.setup_components()
        
        if not success:
            system_status = "Error: Failed to setup components"
            return
        
        system_status = "Loading and analyzing sales data..."
        success = agentic_workflow.load_and_process_documents()
        
        if not success:
            system_status = "Error: Failed to load documents"
            return
        
        system_status = "Creating embeddings..."
        success = agentic_workflow.create_embeddings_and_store()
        
        if not success:
            system_status = "Error: Failed to create embeddings"
            return
        
        system_status = "Setting up agentic agents..."
        success = agentic_workflow.setup_agentic_rag()
        
        if success:
            agentic_workflow.setup_retriever()
            is_system_ready = True
            system_status = "Ready"
            print("✅ Agentic RAG system initialized successfully!")
            
            # Show data overview
            if agentic_workflow.sales_data is not None:
                print(f"📊 Loaded {len(agentic_workflow.sales_data)} sales records")
                print(f"📈 Total Revenue: ${agentic_workflow.sales_data['TotalAmount'].sum():,.2f}")
                print(f"👥 Unique Customers: {agentic_workflow.sales_data['CustomerID'].nunique()}")
        else:
            system_status = "Error: Failed to setup agentic system"
            
    except Exception as e:
        system_status = f"Error: {str(e)}"
        print(f"❌ Error initializing Agentic RAG system: {e}")
        traceback.print_exc()

# Start Agentic RAG system initialization in background
threading.Thread(target=initialize_agentic_system, daemon=True).start()

@app.route('/')
def index():
    """Serve the main agentic chat interface"""
    return render_template('agentic_index.html')

@app.route('/api/status')
def get_status():
    """Get the current system status"""
    sales_data_info = {}
    if agentic_workflow and agentic_workflow.sales_data is not None:
        sales_data_info = {
            'total_orders': len(agentic_workflow.sales_data),
            'total_revenue': float(agentic_workflow.sales_data['TotalAmount'].sum()),
            'unique_customers': int(agentic_workflow.sales_data['CustomerID'].nunique()),
            'unique_menu_items': int(agentic_workflow.sales_data['MenuItem'].nunique()),
            'date_range': {
                'start': str(agentic_workflow.sales_data['OrderTime'].min().date()),
                'end': str(agentic_workflow.sales_data['OrderTime'].max().date())
            }
        }
    
    return jsonify({
        'ready': is_system_ready,
        'status': system_status,
        'document_count': len(agentic_workflow.documents) if agentic_workflow else 0,
        'chunk_count': len(agentic_workflow.chunks) if agentic_workflow else 0,
        'sales_data': sales_data_info,
        'agents_available': list(agentic_workflow.agentic_rag.agents.keys()) if agentic_workflow and agentic_workflow.agentic_rag else []
    })

@app.route('/api/query', methods=['POST'])
def query_agentic_rag():
    """Handle Agentic RAG queries"""
    try:
        if not is_system_ready:
            return jsonify({
                'error': 'Agentic system not ready',
                'status': system_status
            }), 503
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Empty query'}), 400
        
        use_agentic = data.get('use_agentic', True)
        top_k = data.get('top_k', 3)
        
        # Generate query ID for tracking
        query_id = str(uuid.uuid4())
        
        print(f"🔍 Processing {'agentic' if use_agentic else 'traditional'} query [{query_id}]: {query}")
        
        start_time = time.time()
        
        if use_agentic:
            # Use agentic RAG
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            agentic_response = loop.run_until_complete(
                agentic_workflow.process_agentic_query(query)
            )
            loop.close()
            
            processing_time = time.time() - start_time
            
            response = {
                'query_id': query_id,
                'query': query,
                'answer': agentic_response.get('answer', 'No answer generated'),
                'confidence': agentic_response.get('confidence', 'unknown'),
                'quality_score': agentic_response.get('quality_assessment', {}).get('score', 0),
                'insights': {
                    'key_points': agentic_response.get('insights', {}).get('key_points', []),
                    'patterns': agentic_response.get('insights', {}).get('patterns', []),
                    'correlations': agentic_response.get('insights', {}).get('correlations', [])
                },
                'agents_used': agentic_response.get('processing_metadata', {}).get('agents_used', []),
                'sources': [
                    {
                        'content': doc.get('content', '')[:300] + "..." if len(doc.get('content', '')) > 300 else doc.get('content', ''),
                        'source': doc.get('source', 'Unknown'),
                        'relevance_score': doc.get('relevance_score', 0)
                    }
                    for doc in agentic_response.get('sources', [])[:3]
                ],
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'type': 'agentic'
            }
        else:
            # Use traditional retrieval for comparison
            retrieved_docs = agentic_workflow.retriever.retrieve(query, top_k=top_k)
            
            processing_time = time.time() - start_time
            
            response = {
                'query_id': query_id,
                'query': query,
                'answer': "Traditional retrieval completed. Use agentic mode for full analysis.",
                'retrieved_docs': [
                    {
                        'content': doc['document'][:300] + "..." if len(doc['document']) > 300 else doc['document'],
                        'similarity_score': round(doc['similarity_score'], 4),
                        'source': doc['metadata'].get('source', 'Unknown'),
                        'rank': doc['rank']
                    }
                    for doc in retrieved_docs
                ],
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'type': 'traditional'
            }
        
        print(f"✅ Query [{query_id}] processed successfully in {processing_time:.2f}s")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/sales-overview')
def get_sales_overview():
    """Get sales data overview"""
    if not agentic_workflow or agentic_workflow.sales_data is None:
        return jsonify({'error': 'Sales data not available'}), 503
    
    try:
        sales_data = agentic_workflow.sales_data
        
        overview = {
            'total_orders': len(sales_data),
            'total_revenue': float(sales_data['TotalAmount'].sum()),
            'average_order_value': float(sales_data['TotalAmount'].mean()),
            'unique_customers': int(sales_data['CustomerID'].nunique()),
            'unique_menu_items': int(sales_data['MenuItem'].nunique()),
            'date_range': {
                'start': str(sales_data['OrderTime'].min().date()),
                'end': str(sales_data['OrderTime'].max().date())
            },
            'top_items': sales_data.groupby('MenuItem')['TotalAmount'].sum().sort_values(ascending=False).head(5).to_dict(),
            'category_performance': sales_data.groupby('Category')['TotalAmount'].sum().to_dict(),
            'payment_methods': sales_data.groupby('PaymentMethod')['TotalAmount'].sum().to_dict()
        }
        
        return jsonify(overview)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system-info')
def get_system_info():
    """Get detailed system information"""
    if not agentic_workflow:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        vector_count = agentic_workflow.vector_store.collection.count() if agentic_workflow.vector_store else 0
        
        info = {
            'status': system_status,
            'ready': is_system_ready,
            'data_directory': str(agentic_workflow.data_dir),
            'embedding_model': agentic_workflow.embedding_model,
            'collection_name': agentic_workflow.collection_name,
            'chunk_size': agentic_workflow.chunk_size,
            'chunk_overlap': agentic_workflow.chunk_overlap,
            'documents_loaded': len(agentic_workflow.documents),
            'chunks_created': len(agentic_workflow.chunks),
            'vectors_stored': vector_count,
            'agents_available': list(agentic_workflow.agentic_rag.agents.keys()) if agentic_workflow.agentic_rag else [],
            'sales_data_loaded': agentic_workflow.sales_data is not None,
            'embedding_dimension': agentic_workflow.embedding_manager.model.get_sentence_embedding_dimension() if agentic_workflow.embedding_manager and agentic_workflow.embedding_manager.model else 0
        }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🌐 Starting Agentic RAG Web Application...")
    print("📊 Processing hotel food sales data with multi-agent analysis")
    print("📁 Make sure hotel_food_sales.csv is in the 'data/' directory")
    print("🚀 Application will be available at: http://localhost:5001")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
