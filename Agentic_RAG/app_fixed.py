"""
Fixed Flask Web Application for Agentic RAG System
Addresses the Windows socket issue and provides a stable web interface
"""

from flask import Flask, render_template, request, jsonify
import asyncio
import threading
import time
import os
import sys
from datetime import datetime
import traceback

# Add the project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask app with specific configuration for Windows
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'

# Global variables
rag_workflow = None
is_system_ready = False
system_status = "Initializing..."
initialization_thread = None

def initialize_rag_system():
    """Initialize the RAG system in a separate thread"""
    global rag_workflow, is_system_ready, system_status
    
    try:
        system_status = "Loading components..."
        print("🚀 Initializing Agentic RAG system...")
        
        from Agentic_RAG.agentic_main import AgenticRAGWorkflow
        
        # Initialize workflow
        rag_workflow = AgenticRAGWorkflow(
            data_dir="data",
            embedding_model="all-MiniLM-L6-v2",
            collection_name="web_app_docs"
        )
        
        system_status = "Setting up components..."
        success = rag_workflow.setup_components()
        
        if success:
            system_status = "Loading documents..."
            success = rag_workflow.load_and_process_documents()
            
            if success:
                system_status = "Creating embeddings..."
                success = rag_workflow.create_embeddings_and_store()
                
                if success:
                    system_status = "Setting up agentic RAG..."
                    success = rag_workflow.setup_agentic_rag()
                    
                    if success:
                        is_system_ready = True
                        system_status = "Ready"
                        print("✅ Agentic RAG system initialized successfully!")
                    else:
                        system_status = "Error: Failed to setup agentic RAG"
                else:
                    system_status = "Error: Failed to create embeddings"
            else:
                system_status = "Error: Failed to load documents"
        else:
            system_status = "Error: Failed to setup components"
            
    except Exception as e:
        system_status = f"Error: {str(e)}"
        print(f"❌ Error initializing RAG system: {e}")
        traceback.print_exc()

@app.route('/')
def index():
    """Serve a simple HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agentic RAG System</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .ready { background-color: #d4edda; color: #155724; }
            .loading { background-color: #fff3cd; color: #856404; }
            .error { background-color: #f8d7da; color: #721c24; }
            .chat-container { border: 1px solid #ddd; height: 400px; overflow-y: auto; padding: 10px; margin: 10px 0; }
            .query-input { width: 70%; padding: 10px; }
            .submit-btn { padding: 10px 20px; background-color: #007bff; color: white; border: none; cursor: pointer; }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .user-message { background-color: #e9ecef; text-align: right; }
            .bot-message { background-color: #f8f9fa; }
        </style>
    </head>
    <body>
        <h1>🤖 Agentic RAG System</h1>
        
        <div id="status" class="status loading">Loading...</div>
        
        <div id="chat-container" class="chat-container">
            <div class="message bot-message">
                👋 Welcome! I'm your Agentic RAG assistant. Once the system is ready, you can ask me about your data.
            </div>
        </div>
        
        <div>
            <input type="text" id="query-input" class="query-input" placeholder="Ask me about your data..." disabled>
            <button onclick="sendQuery()" class="submit-btn" id="send-btn" disabled>Send</button>
        </div>
        
        <script>
            let systemReady = false;
            
            function checkStatus() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        const statusDiv = document.getElementById('status');
                        const queryInput = document.getElementById('query-input');
                        const sendBtn = document.getElementById('send-btn');
                        
                        if (data.ready) {
                            statusDiv.className = 'status ready';
                            statusDiv.innerHTML = `✅ System Ready - ${data.document_count} documents loaded`;
                            queryInput.disabled = false;
                            sendBtn.disabled = false;
                            systemReady = true;
                        } else {
                            statusDiv.className = 'status loading';
                            statusDiv.innerHTML = `⏳ ${data.status}`;
                        }
                    })
                    .catch(error => {
                        const statusDiv = document.getElementById('status');
                        statusDiv.className = 'status error';
                        statusDiv.innerHTML = '❌ Error checking status';
                    });
            }
            
            function sendQuery() {
                if (!systemReady) return;
                
                const queryInput = document.getElementById('query-input');
                const query = queryInput.value.trim();
                
                if (!query) return;
                
                // Add user message
                addMessage(query, 'user-message');
                queryInput.value = '';
                
                // Add loading message
                const loadingId = addMessage('🤔 Thinking...', 'bot-message');
                
                // Send query
                fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                })
                .then(response => response.json())
                .then(data => {
                    // Remove loading message
                    document.getElementById(loadingId).remove();
                    
                    if (data.error) {
                        addMessage(`❌ Error: ${data.error}`, 'bot-message');
                    } else {
                        let response = `📝 ${data.answer}`;
                        if (data.confidence) {
                            response += `\\n\\n🎯 Confidence: ${data.confidence}`;
                        }
                        addMessage(response, 'bot-message');
                    }
                })
                .catch(error => {
                    document.getElementById(loadingId).remove();
                    addMessage('❌ Error processing query', 'bot-message');
                });
            }
            
            function addMessage(text, className) {
                const chatContainer = document.getElementById('chat-container');
                const messageDiv = document.createElement('div');
                const messageId = 'msg-' + Date.now();
                messageDiv.id = messageId;
                messageDiv.className = 'message ' + className;
                messageDiv.innerHTML = text.replace(/\\n/g, '<br>');
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
                return messageId;
            }
            
            // Handle Enter key
            document.getElementById('query-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendQuery();
                }
            });
            
            // Check status every 2 seconds until ready
            const statusInterval = setInterval(() => {
                checkStatus();
                if (systemReady) {
                    clearInterval(statusInterval);
                }
            }, 2000);
            
            // Initial status check
            checkStatus();
        </script>
    </body>
    </html>
    """
    return html_content

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'ready': is_system_ready,
        'status': system_status,
        'document_count': len(rag_workflow.documents) if rag_workflow else 0,
        'chunk_count': len(rag_workflow.chunks) if rag_workflow else 0
    })

@app.route('/api/query', methods=['POST'])
def query_rag():
    """Handle RAG queries"""
    if not is_system_ready:
        return jsonify({'error': 'System not ready'}), 503
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Empty query'}), 400
        
        # Process query asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response = loop.run_until_complete(rag_workflow.process_agentic_query(query))
            
            # Clean up the response format
            answer = response.get('answer', 'No answer generated')
            
            # Enhanced text processing for better formatting
            import re
            
            # Convert numbered sections to proper HTML structure
            answer = re.sub(r'\*\*(\d+\.\s*[^*]+):\*\*', r'<h4>\1</h4>', answer)
            
            # Convert bold markdown to HTML
            answer = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', answer)
            
            # Convert italic markdown to HTML
            answer = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', answer)
            
            # Handle bullet points better
            answer = re.sub(r'\n\s*\*\s*\*\*([^*]+)\*\*:\s*([^\n]+)', r'<br><br><strong>\1:</strong> \2', answer)
            answer = re.sub(r'\n\s*\*\s*([^\n]+)', r'<br>• \1', answer)
            
            # Handle paragraph breaks
            answer = re.sub(r'\n\s*\n', '<br><br>', answer)
            
            # Clean up remaining single line breaks
            answer = re.sub(r'\n', '<br>', answer)
            
            # Handle headers (convert ### to h4, ## to h3, etc.)
            answer = re.sub(r'#{4,}\s*([^\n<]+)', r'<h5>\1</h5>', answer)
            answer = re.sub(r'#{3}\s*([^\n<]+)', r'<h4>\1</h4>', answer)
            answer = re.sub(r'#{2}\s*([^\n<]+)', r'<h3>\1</h3>', answer)
            answer = re.sub(r'#{1}\s*([^\n<]+)', r'<h3>\1</h3>', answer)
            
            # Clean up extra spaces and normalize
            answer = re.sub(r'\s+', ' ', answer).strip()
            
            # Fix common formatting issues
            answer = answer.replace('<br><br><br>', '<br><br>');
            answer = answer.replace('<br> <br>', '<br><br>');
            
            return jsonify({
                'query': query,
                'answer': answer,
                'confidence': response.get('confidence', 'unknown'),
                'quality_score': response.get('quality_score', 'N/A'),
                'processing_time': response.get('processing_time', 0),
                'agents_used': response.get('agents_used', []),
                'insights': response.get('insights', {}),
                'timestamp': datetime.now().isoformat()
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sales-overview')
def get_sales_overview():
    """Get sales overview data"""
    if not is_system_ready:
        return jsonify({'error': 'System not ready'}), 503
    
    try:
        # Mock data - replace with actual data extraction
        return jsonify({
            'total_orders': 1234,
            'total_revenue': 45678.90,
            'average_order_value': 37.05,
            'unique_customers': 567,
            'unique_menu_items': 89
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def start_initialization():
    """Start the initialization in a separate thread"""
    global initialization_thread
    if initialization_thread is None or not initialization_thread.is_alive():
        initialization_thread = threading.Thread(target=initialize_rag_system, daemon=True)
        initialization_thread.start()

if __name__ == '__main__':
    print("🌐 Starting Fixed Agentic RAG Web Application...")
    print("📁 Make sure your documents are in the 'data/' directory")
    print("🚀 Application will be available at: http://localhost:5001")
    
    # Start initialization
    start_initialization()
    
    # Run Flask with specific settings for Windows
    try:
        app.run(
            debug=False,  # Disable debug mode to avoid socket issues
            host='127.0.0.1',  # Use localhost instead of 0.0.0.0
            port=5001,  # Use a different port
            threaded=True,  # Enable threading
            use_reloader=False  # Disable auto-reloader
        )
    except Exception as e:
        print(f"❌ Web server error: {e}")
        print("💡 Try running the test script instead: python run_agentic_test.py")
