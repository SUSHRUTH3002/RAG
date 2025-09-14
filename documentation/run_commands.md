# RAG System Run Commands

## Prerequisites
Make sure you have installed all required dependencies:
```bash
pip install -r requirements.txt
```

## Running the RAG System

### 1. Interactive Mode (Recommended)
Run the complete RAG system with interactive query session:

```bash
# From project root directory
python -m RAG.main

# Or using the simple run script
python run_rag.py
```

This will:
- Set up all RAG components
- Load and process documents from the `data/` directory
- Create embeddings and store in vector database
- Start an interactive session where you can ask questions

### 2. Single Query Mode
Process a single query and exit:

```bash
python -m RAG.main --query "Tell me about machine learning"
```

### 3. Non-Interactive Mode
Run setup without interactive session:

```bash
python -m RAG.main --no-interactive
```

### 4. Custom Data Directory
Specify a custom data directory:

```bash
python -m RAG.main --data-dir "/path/to/your/documents"
```

## Interactive Session Commands

Once in the interactive session, you can:

- **Ask questions**: Just type your question and press Enter
- **View system info**: Type `info` to see system information
- **Exit**: Type `quit`, `exit`, or `q` to stop the session
- **Interrupt**: Press `Ctrl+C` to interrupt the session

## Example Usage

```bash
# Start interactive RAG system
python -m RAG.main

# The system will process your documents and then show:
🔍 Your Question: What is machine learning?

# Type your question and get instant answers!
```

## Supported Document Formats

Place your documents in the `data/` directory. Supported formats:
- PDF files (`.pdf`)
- Text files (`.txt`)
- CSV files (`.csv`)
- Excel files (`.xlsx`)

## Jupyter Notebook

For development and experimentation, use the Jupyter notebook:

```bash
jupyter notebook notebook/experiment.ipynb
```

The notebook provides three usage options:
1. Complete automated workflow
2. Step-by-step manual workflow
3. Individual component usage
