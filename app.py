"""
Flask Web Application for Traditional RAG System
Provides a web interface for querying the RAG system
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import uuid
import time
import threading
from datetime import datetime
import traceback
import os
import sys

# Add the RAG module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from RAG import RAGWorkflow

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
CORS(app)

# Global variables for RAG system
rag_workflow = None
is_system_ready = False
system_status = "Initializing..."

def initialize_rag_system():
    """Initialize the RAG system in a separate thread"""
    global rag_workflow, is_system_ready, system_status
    
    try:
        system_status = "Loading RAG components..."
        print("🚀 Initializing RAG system...")
        
        # Initialize workflow
        rag_workflow = RAGWorkflow(
            data_dir="data",
            embedding_model="all-MiniLM-L6-v2",
            collection_name="web_app_docs"
        )
        
        system_status = "Setting up components..."
        rag_workflow.setup_components()
        
        system_status = "Loading documents..."
        success = rag_workflow.load_and_process_documents()
        
        if success:
            system_status = "Creating embeddings..."
            success = rag_workflow.create_embeddings_and_store()
            
            if success:
                is_system_ready = True
                system_status = "Ready"
                print("✅ RAG system initialized successfully!")
            else:
                system_status = "Error: Failed to create embeddings"
                print("❌ Failed to create embeddings")
        else:
            system_status = "Error: No documents found"
            print("❌ No documents found")
            
    except Exception as e:
        system_status = f"Error: {str(e)}"
        print(f"❌ Error initializing RAG system: {e}")
        traceback.print_exc()

# Start RAG system initialization in background
threading.Thread(target=initialize_rag_system, daemon=True).start()

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get the current system status"""
    return jsonify({
        'ready': is_system_ready,
        'status': system_status,
        'document_count': len(rag_workflow.documents) if rag_workflow else 0,
        'chunk_count': len(rag_workflow.chunks) if rag_workflow else 0
    })

@app.route('/api/query', methods=['POST'])
def query_rag():
    """Handle RAG queries"""
    try:
        if not is_system_ready:
            return jsonify({
                'error': 'System not ready',
                'status': system_status
            }), 503
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Empty query'}), 400
        
        top_k = data.get('top_k', 3)
        
        # Generate query ID for tracking
        query_id = str(uuid.uuid4())
        
        print(f"🔍 Processing query [{query_id}]: {query}")
        
        # Get retrieved documents
        retrieved_docs = rag_workflow.retriever.retrieve(query, top_k=top_k)
        
        # Generate answer
        answer = rag_workflow.rag_pipeline.generate_answer(query, top_k=top_k)
        
        # Prepare response
        response = {
            'query_id': query_id,
            'query': query,
            'answer': answer,
            'retrieved_docs': [
                {
                    'content': doc['document'][:300] + "..." if len(doc['document']) > 300 else doc['document'],
                    'similarity_score': round(doc['similarity_score'], 4),
                    'source': doc['metadata'].get('source', 'Unknown'),
                    'rank': doc['rank']
                }
                for doc in retrieved_docs
            ],
            'timestamp': datetime.now().isoformat(),
            'processing_time': time.time()
        }
        
        print(f"✅ Query [{query_id}] processed successfully")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/system-info')
def get_system_info():
    """Get detailed system information"""
    if not rag_workflow:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        vector_count = rag_workflow.vector_store.collection.count() if rag_workflow.vector_store else 0
        
        info = {
            'status': system_status,
            'ready': is_system_ready,
            'data_directory': rag_workflow.data_dir,
            'embedding_model': rag_workflow.embedding_model,
            'collection_name': rag_workflow.collection_name,
            'chunk_size': rag_workflow.chunk_size,
            'chunk_overlap': rag_workflow.chunk_overlap,
            'documents_loaded': len(rag_workflow.documents),
            'chunks_created': len(rag_workflow.chunks),
            'vectors_stored': vector_count,
            'embedding_dimension': rag_workflow.embedding_manager.model.get_sentence_embedding_dimension() if rag_workflow.embedding_manager and rag_workflow.embedding_manager.model else 0
        }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🌐 Starting Flask RAG Web Application...")
    print("📁 Make sure your documents are in the 'data/' directory")
    print("🚀 Application will be available at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
