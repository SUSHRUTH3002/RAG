"""
Simple script to run the Agentic RAG Web Application
Similar to run_web_app.py but for agentic capabilities
"""

if __name__ == "__main__":
    print("🌐 Starting Agentic RAG Web Application...")
    print("🤖 Multi-agent analysis for hotel food sales data")
    print("📊 Advanced business intelligence with AI agents")
    print("📁 Make sure hotel_food_sales.csv is in the 'data/' directory")
    print("🚀 Application will be available at: http://localhost:5001")
    print("⏳ Please wait while the agentic system loads...")
    print("\n" + "="*60)
    
    from Agentic_RAG.agentic_app import app
    app.run(debug=True, host='0.0.0.0', port=5001)
