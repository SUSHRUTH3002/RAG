"""
Simple script to run the RAG Web Application
"""

if __name__ == "__main__":
    print("🌐 Starting RAG Web Application...")
    print("📁 Make sure your documents are in the 'data/' directory")
    print("🚀 Application will be available at: http://localhost:5000")
    print("⏳ Please wait while the system loads your documents...")
    print("\n" + "="*60)
    
    from app import app
    app.run(debug=True, host='0.0.0.0', port=5000)
