# Traditional RAG (Retrieval-Augmented Generation) System

A complete implementation of a Traditional RAG system with modular architecture, supporting multiple document formats and providing both programmatic and interactive interfaces.

## 🚀 Quick Start

### Prerequisites
```bash
# Install required dependencies
pip install -r requirements.txt
```

### Basic Usage
```bash
# Navigate to project directory
cd "e:\Projects\Traditional RAG"

# Run interactive RAG system
python -m RAG.main
```

## 📁 Project Structure

```
e:\Projects\Traditional RAG\
├── RAG/                    # Main RAG package
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration settings
│   ├── data_ingestion.py  # Document loading and processing
│   ├── embeddings.py      # Embedding generation
│   ├── vector_store.py    # Vector database management
│   ├── retriever.py       # Document retrieval
│   ├── rag_pipeline.py    # RAG pipeline implementation
│   └── main.py           # Main workflow orchestration
├── notebook/
│   └── experiment.ipynb   # Jupyter notebook for experimentation
├── data/                  # Place your documents here
│   ├── VectorStore/       # Vector database storage
│   └── sample_data/       # Sample documents
├── requirements.txt       # Python dependencies
├── run_rag.py            # Simple run script
├── README.md             # This file
└── USAGE.md              # Detailed usage instructions
```

## 🎯 Usage Modes

### 1. Interactive Mode (Recommended)
Start an interactive session where you can ask unlimited questions:

```bash
python -m RAG.main
```

**Features:**
- ✅ Automatic document loading and processing
- 🤖 Interactive Q&A session
- 📚 Option to view source documents
- ℹ️ System information with `info` command
- 🚪 Exit with `quit`, `exit`, or `q`

### 2. Single Query Mode
Process a single question and exit:

```bash
python -m RAG.main --query "What is machine learning?"
```

### 3. Non-Interactive Mode
Set up the system without starting interactive session:

```bash
python -m RAG.main --no-interactive
```

### 4. Custom Data Directory
Specify a custom directory for your documents:

```bash
python -m RAG.main --data-dir "/path/to/your/documents"
```

### 5. Simple Run Script
Use the simplified run script:

```bash
python run_rag.py
```

## 📄 Supported Document Formats

Place your documents in the `data/` directory. Supported formats:

- **PDF files** (`.pdf`) - Research papers, reports, manuals
- **Text files** (`.txt`) - Plain text documents
- **CSV files** (`.csv`) - Structured data
- **Excel files** (`.xlsx`) - Spreadsheets and data

## 💻 Interactive Session Commands

Once in the interactive session:

| Command | Description |
|---------|-------------|
| `Your question here` | Ask any question about your documents |
| `info` | Display system information and statistics |
| `quit` / `exit` / `q` | Exit the interactive session |
| `Ctrl+C` | Interrupt and exit the session |

## 🔧 Configuration

Modify `RAG/config.py` to customize:

```python
# Model configurations
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gemini-1.5-flash"

# Text processing
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Retrieval parameters
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.0
```

## 📊 Jupyter Notebook Usage

For development and experimentation:

```bash
jupyter notebook notebook/experiment.ipynb
```

The notebook provides three approaches:
1. **Automated Workflow** - Complete automation with `RAGWorkflow`
2. **Step-by-Step** - Manual execution for learning
3. **Component-Based** - Direct access to individual components

## 🔄 Workflow Steps

The RAG system follows these steps:

1. **📁 Document Loading** - Load documents from various formats
2. **✂️ Text Splitting** - Split documents into manageable chunks
3. **🧠 Embedding Generation** - Create vector embeddings using SentenceTransformers
4. **🗄️ Vector Storage** - Store embeddings in ChromaDB
5. **🔍 Query Processing** - Retrieve relevant documents for queries
6. **🤖 Answer Generation** - Generate answers using Gemini LLM

## 🎛️ Advanced Usage

### Programmatic Usage

```python
from RAG import RAGWorkflow

# Initialize workflow
workflow = RAGWorkflow(
    data_dir="path/to/documents",
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=1000
)

# Run complete workflow
workflow.run_complete_workflow(interactive=False)

# Query the system
answer = workflow.query_rag_system("Your question here")
print(answer)
```

### Using Individual Components

```python
from RAG import DocumentLoader, EmbeddingManager, VectorStore, RAGRetriever, SimpleRAG

# Load documents
loader = DocumentLoader("data/")
documents = loader.load_pdf_files()

# Generate embeddings
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.generate_embeddings([doc.page_content for doc in documents])

# Store in vector database
vector_store = VectorStore()
vector_store.add_documents(documents, embeddings)

# Set up retrieval and generation
retriever = RAGRetriever(vector_store, embedding_manager)
rag_pipeline = SimpleRAG(retriever)

# Ask questions
answer = rag_pipeline.generate_answer("Your question")
```

## 🔍 Example Queries

Try these example questions:

- "What are the main topics covered in the documents?"
- "Explain the methodology used in the research"
- "What are the key findings or conclusions?"
- "Summarize the technical approach described"
- "What technologies or tools are mentioned?"

## 🐛 Troubleshooting

### Common Issues

1. **No documents found**
   - Ensure documents are in the `data/` directory
   - Check supported file formats

2. **API key errors**
   - Update `GEMINI_API_KEY` in `RAG/config.py`
   - Ensure you have a valid Google Gemini API key

3. **Memory issues**
   - Reduce `DEFAULT_CHUNK_SIZE` in config
   - Process documents in smaller batches

4. **Slow performance**
   - Use a smaller embedding model
   - Reduce the number of retrieved documents (`top_k`)

### Getting Help

- Check the Jupyter notebook for detailed examples
- Review the configuration in `RAG/config.py`
- Examine log messages for specific error details

## 🚀 Next Steps

- Add more document types
- Implement query rewriting
- Add conversation memory
- Integrate citation tracking
- Implement answer quality evaluation

## 📝 License

This project is for educational and research purposes.
