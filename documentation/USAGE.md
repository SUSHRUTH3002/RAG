# Detailed Usage Instructions

## Getting Started

### Step 1: Installation
```bash
# Clone or download the project
cd "e:\Projects\Traditional RAG"

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Your Documents
```bash
# Create data directory if it doesn't exist
mkdir data

# Copy your documents to the data directory
# Supported formats: .pdf, .txt, .csv, .xlsx
cp your_documents.pdf data/
cp your_text_files.txt data/
```

### Step 3: Run the RAG System
```bash
# Start interactive mode
python -m RAG.main
```

## Command Line Options

### Basic Commands

```bash
# Interactive mode (default)
python -m RAG.main

# Single query mode
python -m RAG.main --query "Your question here"

# Non-interactive setup only
python -m RAG.main --no-interactive

# Custom data directory
python -m RAG.main --data-dir "/path/to/documents"

# Simple run (uses defaults)
python run_rag.py
```

### Command Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--query` | Process single query and exit | `--query "What is AI?"` |
| `--no-interactive` | Skip interactive session | `--no-interactive` |
| `--data-dir` | Custom document directory | `--data-dir "/path/to/docs"` |
| `--help` | Show help message | `--help` |

## Interactive Session Guide

### Starting a Session
```bash
python -m RAG.main
```

You'll see output like:
```
🚀 STARTING COMPLETE RAG WORKFLOW
==================================================
SETTING UP RAG COMPONENTS
==================================================
...
🤖 INTERACTIVE RAG QUERY SESSION
==================================================
Enter your questions below. Type 'quit', 'exit', or 'q' to stop.
Type 'info' to see system information.
==================================================

🔍 Your Question: 
```

### Available Commands in Interactive Mode

| Command | Action | Example |
|---------|--------|---------|
| `Any question` | Ask about your documents | `What is machine learning?` |
| `info` | Show system information | `info` |
| `quit` | Exit the session | `quit` |
| `exit` | Exit the session | `exit` |
| `q` | Quick exit | `q` |
| `Ctrl+C` | Force exit | (keyboard shortcut) |

### Example Interactive Session
```
🔍 Your Question: What are the main topics in the documents?

🔄 Processing your question...
==================================================
QUERYING RAG SYSTEM
==================================================

📝 Query: What are the main topics in the documents?

🔍 Retrieving top 3 relevant documents...
Generating embedding for 1 texts
Embedding generated with shape: (1, 384)
Retrieved 3 documents above the score threshold 0.0.

🤖 Generating answer using RAG pipeline...

💬 Answer: Based on the documents, the main topics include...

💡 Quick Answer: Based on the documents, the main topics include...

📚 Would you like to see the source documents? (y/n): y

📚 Source Documents:

  📄 Document 1:
     Similarity: 0.8542
     Preview: This document discusses machine learning algorithms...
     Source: research_paper.pdf

🔍 Your Question: Tell me more about deep learning

🔍 Your Question: info

==================================================
RAG SYSTEM INFORMATION
==================================================
📁 Data Directory: e:\Projects\Traditional RAG\data
🤖 Embedding Model: all-MiniLM-L6-v2
🗃️  Collection Name: pdf_documents
📄 Documents Loaded: 15
🔧 Chunks Created: 127
⚙️  Chunk Size: 1000
🔄 Chunk Overlap: 200
🗄️  Documents in Vector Store: 127
🎯 System Status: ✅ Ready

🔍 Your Question: quit

👋 Thank you for using the RAG system! Goodbye!
```

## Jupyter Notebook Usage

### Starting the Notebook
```bash
jupyter notebook notebook/experiment.ipynb
```

### Notebook Sections

1. **Cell 1-2**: Import modules and setup
2. **Cell 3**: Option 1 - Complete automated workflow
3. **Cell 4-6**: Custom queries and system info
4. **Cell 7**: Option 2 - Step-by-step manual workflow
5. **Cell 8-11**: Manual execution of each step
6. **Cell 12**: Option 3 - Using individual components
7. **Cell 13-16**: Component-based usage examples

### Running Different Approaches

#### Approach 1: Automated Workflow
```python
# Run this cell for complete automation
workflow = RAGWorkflow()
workflow.run_complete_workflow()
```

#### Approach 2: Step-by-Step
```python
# Setup components
workflow = RAGWorkflow()
workflow.setup_components()

# Load documents
workflow.load_and_process_documents()

# Create embeddings
workflow.create_embeddings_and_store()

# Query system
answer = workflow.query_rag_system("Your question")
```

#### Approach 3: Individual Components
```python
# Use components directly
from RAG import DocumentLoader, EmbeddingManager, VectorStore

loader = DocumentLoader()
documents = loader.load_pdf_files()
# ... continue with other components
```

## Programmatic Usage

### Basic Example
```python
from RAG import RAGWorkflow

# Initialize and run
workflow = RAGWorkflow(data_dir="my_documents/")
workflow.run_complete_workflow(interactive=False)

# Ask questions
answer = workflow.query_rag_system("What is the main topic?")
print(answer)
```

### Advanced Configuration
```python
from RAG import RAGWorkflow

# Custom configuration
workflow = RAGWorkflow(
    data_dir="documents/",
    embedding_model="all-MiniLM-L6-v2",
    collection_name="my_collection",
    chunk_size=500,
    chunk_overlap=100
)

# Run with custom queries
sample_queries = [
    "What are the key findings?",
    "Explain the methodology",
    "What are the conclusions?"
]

workflow.run_complete_workflow(
    sample_queries=sample_queries,
    interactive=False
)
```

### Component-Level Usage
```python
from RAG import (
    DocumentLoader, split_documents, EmbeddingManager,
    VectorStore, RAGRetriever, SimpleRAG
)

# Load and process documents
loader = DocumentLoader("data/")
documents = loader.load_all_documents()
chunks = split_documents(documents)

# Create embeddings
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.generate_embeddings([c.page_content for c in chunks])

# Store in vector database
vector_store = VectorStore("my_collection")
vector_store.add_documents(chunks, embeddings)

# Setup retrieval
retriever = RAGRetriever(vector_store, embedding_manager)
rag_pipeline = SimpleRAG(retriever)

# Query
answer = rag_pipeline.generate_answer("Your question", top_k=5)
```

## Configuration Options

### Modifying Configuration
Edit `RAG/config.py`:

```python
# Document processing
DEFAULT_CHUNK_SIZE = 1000        # Size of text chunks
DEFAULT_CHUNK_OVERLAP = 200      # Overlap between chunks

# Embedding model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model

# LLM configuration
DEFAULT_LLM_MODEL = "gemini-1.5-flash"         # Gemini model
GEMINI_API_KEY = "your-api-key-here"           # Your API key

# Retrieval settings
DEFAULT_TOP_K = 5               # Number of documents to retrieve
DEFAULT_SCORE_THRESHOLD = 0.0   # Minimum similarity threshold
```

### Environment Variables
You can also use environment variables:

```bash
export GEMINI_API_KEY="your-api-key-here"
export RAG_DATA_DIR="/path/to/documents"
export RAG_CHUNK_SIZE="1000"
```

## Best Practices

### Document Preparation
1. **Organize documents** in the `data/` directory
2. **Use clear filenames** for better source tracking
3. **Remove unnecessary documents** to improve relevance
4. **Ensure documents are readable** (not corrupted or password-protected)

### Query Formulation
1. **Be specific** in your questions
2. **Use relevant keywords** from your documents
3. **Ask one question at a time** for better results
4. **Provide context** when needed

### Performance Optimization
1. **Adjust chunk size** based on document type
2. **Use appropriate embedding models** for your domain
3. **Monitor system resources** during processing
4. **Process large datasets** in batches

### Troubleshooting
1. **Check logs** for error messages
2. **Verify document formats** are supported
3. **Ensure API keys** are correctly configured
4. **Monitor disk space** for vector storage

## Examples and Use Cases

### Research Papers
```bash
# Place academic papers in data/
python -m RAG.main --query "What are the main research findings?"
```

### Technical Documentation
```bash
# Load technical manuals
python -m RAG.main --query "How do I configure the system?"
```

### Business Reports
```bash
# Analyze business documents
python -m RAG.main --query "What are the key performance indicators?"
```

### Mixed Document Types
```bash
# Process various document formats
python -m RAG.main --query "Give me a summary of all the information"
```

## Getting Help

### Documentation
- `README.md` - Overview and quick start
- `USAGE.md` - This detailed guide
- `notebook/experiment.ipynb` - Interactive examples

### Debug Information
```bash
# Run with verbose output
python -m RAG.main --query "test" 2>&1 | tee debug.log
```

### Common Commands
```bash
# Check if everything is working
python -c "from RAG import RAGWorkflow; print('✅ RAG system is ready!')"

# Test with sample data
python -m RAG.main --query "test query" --data-dir "data/"

# Get system info only
python -c "from RAG import RAGWorkflow; w=RAGWorkflow(); w.setup_components(); w.get_system_info()"
```
