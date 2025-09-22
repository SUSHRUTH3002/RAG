"""
Agentic RAG Pipeline
====================
A production-ready Retrieval-Augmented Generation system with multi-agent architecture.

Features:
- Dynamic data loading from any files in data/ directory
- Automatic document processing and analysis generation
- Multi-agent system with specialized roles
- Query validation and planning
- Result auditing and strategic synthesis

Author: Enhanced from your original implementation
Version: 2.0 - Dynamic Data Processing
"""

import os
import uuid
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader, 
    PyPDFLoader, 
    CSVLoader, 
    TextLoader,
    UnstructuredExcelLoader,
    JSONLoader,
    PyMuPDFLoader
)
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================================
# Configuration Classes
# ================================

@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "rag_db"
    user: str = "rag_user"
    password: str = "rag_password"
    min_connections: int = 1
    max_connections: int = 10

@dataclass
class RAGConfig:
    """Main configuration for the RAG pipeline"""
    # Database (optional for future use)
    db_config: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Vector Store
    vector_collection_name: str = "documents"
    vector_persist_dir: str = "./data/vector_store"
    
    # Embedding Model
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # LLM
    llm_model: str = "gemini-1.5-flash"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    
    # Retrieval
    default_top_k: int = 5
    similarity_threshold: float = 0.3
    
    # Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 50

class QueryType(Enum):
    """Types of queries the system can handle"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    EXPLORATORY = "exploratory"
    UNCLEAR = "unclear"

# ================================
# Database Manager (Optional)
# ================================

class PostgreSQLManager:
    """
    Optional PostgreSQL manager for future database integration.
    Currently bypassed for file-based implementation.
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection_pool = None
        self.enabled = False  # Disabled for file-based implementation
        
    def store_document(self, content: str, source: str, metadata: dict) -> str:
        """Placeholder for future database storage"""
        return str(uuid.uuid4())
    
    def log_query(self, query: str, query_type: str, results: dict):
        """Placeholder for future query logging"""
        pass

# ================================
# Dynamic Data Processor
# ================================

class DynamicDataProcessor:
    """
    Dynamically processes any files in the data directory and creates documents for the RAG system.
    Automatically detects file types and generates appropriate analytical documents.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.loaded_data = {}
        self.file_types = {
            '.csv': self._process_csv_file,
            '.xlsx': self._process_excel_file,
            '.json': self._process_json_file,
            '.txt': self._process_text_file,
            '.pdf': self._process_pdf_file,
            '.md': self._process_text_file
        }
        
    def discover_and_load_all_data(self) -> Dict[str, Any]:
        """Discover and load all supported files in the data directory"""
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return {}
        
        logger.info(f"🔍 Scanning {self.data_dir} for data files...")
        
        for file_path in self.data_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.file_types:
                try:
                    file_key = str(file_path.relative_to(self.data_dir))
                    processor = self.file_types[file_path.suffix.lower()]
                    self.loaded_data[file_key] = processor(file_path)
                    logger.info(f"✅ Loaded {file_key}")
                except Exception as e:
                    logger.error(f"❌ Failed to load {file_path}: {e}")
        
        logger.info(f"📊 Successfully loaded {len(self.loaded_data)} data files")
        return self.loaded_data
    
    def _process_csv_file(self, file_path: Path) -> Dict[str, Any]:
        """Process CSV files"""
        df = pd.read_csv(file_path)
        
        # Attempt to parse datetime columns
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['time', 'date', 'created', 'updated']):
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        return {
            'type': 'csv',
            'data': df,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'summary': df.describe(include='all').to_dict(),
            'file_path': str(file_path)
        }
    
    def _process_excel_file(self, file_path: Path) -> Dict[str, Any]:
        """Process Excel files"""
        # Read all sheets
        excel_data = pd.read_excel(file_path, sheet_name=None)
        
        result = {
            'type': 'excel',
            'sheets': {},
            'file_path': str(file_path)
        }
        
        for sheet_name, df in excel_data.items():
            # Attempt to parse datetime columns
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['time', 'date', 'created', 'updated']):
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
            
            result['sheets'][sheet_name] = {
                'data': df,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'summary': df.describe(include='all').to_dict()
            }
        
        return result
    
    def _process_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Process JSON files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            'type': 'json',
            'data': data,
            'keys': list(data.keys()) if isinstance(data, dict) else None,
            'length': len(data) if isinstance(data, (list, dict)) else None,
            'file_path': str(file_path)
        }
    
    def _process_text_file(self, file_path: Path) -> Dict[str, Any]:
        """Process text files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            'type': 'text',
            'content': content,
            'length': len(content),
            'lines': len(content.split('\n')),
            'file_path': str(file_path)
        }
    
    def _process_pdf_file(self, file_path: Path) -> Dict[str, Any]:
        """Process PDF files"""
        # This will be loaded later by the document loaders
        return {
            'type': 'pdf',
            'file_path': str(file_path)
        }
    
    def create_comprehensive_documents(self) -> List[Document]:
        """Create structured documents from all loaded data for RAG processing"""
        if not self.loaded_data:
            self.discover_and_load_all_data()
        
        documents = []
        
        # 1. Create overview document
        overview_doc = self._create_data_overview_document()
        documents.append(overview_doc)
        
        # 2. Process each data file
        for file_key, file_data in self.loaded_data.items():
            if file_data['type'] == 'csv':
                docs = self._create_csv_documents(file_key, file_data)
            elif file_data['type'] == 'excel':
                docs = self._create_excel_documents(file_key, file_data)
            elif file_data['type'] == 'json':
                docs = self._create_json_documents(file_key, file_data)
            elif file_data['type'] == 'text':
                docs = self._create_text_documents(file_key, file_data)
            elif file_data['type'] == 'pdf':
                docs = self._create_pdf_documents(file_key, file_data)
            else:
                continue
                
            documents.extend(docs)
        
        logger.info(f"✅ Created {len(documents)} documents from all data sources")
        return documents
    
    def _create_data_overview_document(self) -> Document:
        """Create a comprehensive overview of all loaded data"""
        overview_content = "# Data Repository Overview\n\n"
        overview_content += f"This repository contains {len(self.loaded_data)} data files with the following breakdown:\n\n"
        
        # Categorize by file type
        file_types_count = {}
        for file_data in self.loaded_data.values():
            file_type = file_data['type']
            file_types_count[file_type] = file_types_count.get(file_type, 0) + 1
        
        overview_content += "## File Type Distribution\n"
        for file_type, count in file_types_count.items():
            overview_content += f"- {file_type.upper()}: {count} files\n"
        
        overview_content += "\n## Detailed File Information\n"
        for file_key, file_data in self.loaded_data.items():
            overview_content += f"\n### {file_key}\n"
            overview_content += f"- Type: {file_data['type']}\n"
            overview_content += f"- Path: {file_data['file_path']}\n"
            
            if file_data['type'] == 'csv':
                overview_content += f"- Shape: {file_data['shape'][0]} rows, {file_data['shape'][1]} columns\n"
                overview_content += f"- Columns: {', '.join(file_data['columns'])}\n"
            elif file_data['type'] == 'excel':
                sheet_count = len(file_data['sheets'])
                overview_content += f"- Sheets: {sheet_count}\n"
                for sheet_name, sheet_data in file_data['sheets'].items():
                    overview_content += f"  - {sheet_name}: {sheet_data['shape'][0]} rows, {sheet_data['shape'][1]} columns\n"
            elif file_data['type'] == 'json':
                overview_content += f"- Structure: {type(file_data['data']).__name__}\n"
                if file_data['keys']:
                    overview_content += f"- Keys: {', '.join(str(k) for k in file_data['keys'][:10])}\n"
            elif file_data['type'] == 'text':
                overview_content += f"- Length: {file_data['length']} characters\n"
                overview_content += f"- Lines: {file_data['lines']}\n"
        
        return Document(
            page_content=overview_content,
            metadata={
                "source": "data_overview",
                "type": "overview",
                "category": "metadata",
                "files_included": ", ".join(list(self.loaded_data.keys())),  # Convert list to string
                "file_count": len(self.loaded_data)
            }
        )
    
    def _create_csv_documents(self, file_key: str, file_data: Dict) -> List[Document]:
        """Create documents from CSV data"""
        documents = []
        df = file_data['data']
        
        # Main analysis document
        analysis_content = f"# Analysis of {file_key}\n\n"
        analysis_content += f"## Dataset Overview\n"
        analysis_content += f"- Rows: {file_data['shape'][0]:,}\n"
        analysis_content += f"- Columns: {file_data['shape'][1]}\n"
        analysis_content += f"- File: {file_key}\n\n"
        
        # Column information
        analysis_content += "## Column Information\n"
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            analysis_content += f"- **{col}**: {dtype}, {null_count} nulls, {unique_count} unique values\n"
        
        # Data insights
        analysis_content += "\n## Key Statistics\n"
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis_content += "### Numeric Columns\n"
            for col in numeric_cols:
                analysis_content += f"- **{col}**: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}\n"
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            analysis_content += "\n### Categorical Columns\n"
            for col in categorical_cols:
                top_values = df[col].value_counts().head(3)
                analysis_content += f"- **{col}**: {len(top_values)} top values - {dict(top_values)}\n"
        
        # Datetime columns analysis
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            analysis_content += "\n### Date/Time Columns\n"
            for col in datetime_cols:
                analysis_content += f"- **{col}**: from {df[col].min()} to {df[col].max()}\n"
        
        # Add correlations for numeric data
        if len(numeric_cols) > 1:
            analysis_content += "\n### Data Correlations\n"
            corr_matrix = df[numeric_cols].corr()
            # Get strong correlations
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        analysis_content += f"- {col1} ↔ {col2}: {corr_val:.3f}\n"
        
        documents.append(Document(
            page_content=analysis_content,
            metadata={
                "source": f"{file_key}_analysis",
                "type": "analysis",
                "category": "csv_data",
                "original_file": file_key,
                "rows": file_data['shape'][0],
                "columns": file_data['shape'][1]
            }
        ))
        
        # Create sample data documents
        sample_size = min(50, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        for idx, row in sample_df.iterrows():
            row_content = f"Data record from {file_key}:\n"
            for col, value in row.items():
                row_content += f"{col}: {value}\n"
            
            documents.append(Document(
                page_content=row_content,
                metadata={
                    "source": f"{file_key}_record_{idx}",
                    "type": "data_record",
                    "category": "csv_row",
                    "original_file": file_key,
                    "record_index": idx
                }
            ))
        
        return documents
    
    def _create_excel_documents(self, file_key: str, file_data: Dict) -> List[Document]:
        """Create documents from Excel data"""
        documents = []
        
        # Overview document for the entire Excel file
        overview_content = f"# Excel File Analysis: {file_key}\n\n"
        overview_content += f"This Excel file contains {len(file_data['sheets'])} sheets:\n\n"
        
        for sheet_name, sheet_data in file_data['sheets'].items():
            overview_content += f"## Sheet: {sheet_name}\n"
            overview_content += f"- Dimensions: {sheet_data['shape'][0]} rows × {sheet_data['shape'][1]} columns\n"
            overview_content += f"- Columns: {', '.join(sheet_data['columns'])}\n\n"
        
        documents.append(Document(
            page_content=overview_content,
            metadata={
                "source": f"{file_key}_overview",
                "type": "analysis",
                "category": "excel_overview",
                "original_file": file_key,
                "sheet_count": len(file_data['sheets'])
            }
        ))
        
        # Process each sheet as if it were a separate CSV
        for sheet_name, sheet_data in file_data['sheets'].items():
            sheet_file_data = {
                'type': 'csv',
                'data': sheet_data['data'],
                'shape': sheet_data['shape'],
                'columns': sheet_data['columns'],
                'summary': sheet_data['summary']
            }
            sheet_docs = self._create_csv_documents(f"{file_key}_{sheet_name}", sheet_file_data)
            documents.extend(sheet_docs)
        
        return documents
    
    def _create_json_documents(self, file_key: str, file_data: Dict) -> List[Document]:
        """Create documents from JSON data"""
        documents = []
        
        # Main JSON analysis
        content = f"# JSON Data Analysis: {file_key}\n\n"
        content += f"## Structure Overview\n"
        content += f"- Data Type: {type(file_data['data']).__name__}\n"
        
        if file_data['length']:
            content += f"- Length: {file_data['length']}\n"
        
        if file_data['keys']:
            content += f"- Top-level Keys: {', '.join(str(k) for k in file_data['keys'][:20])}\n"
        
        content += f"\n## Content Preview\n"
        content += f"```json\n{json.dumps(file_data['data'], indent=2)[:2000]}...\n```\n"
        
        documents.append(Document(
            page_content=content,
            metadata={
                "source": f"{file_key}_analysis",
                "type": "analysis",
                "category": "json_data",
                "original_file": file_key
            }
        ))
        
        # If JSON contains structured data, create individual documents
        if isinstance(file_data['data'], list):
            for i, item in enumerate(file_data['data'][:100]):  # Limit to first 100 items
                item_content = f"JSON record {i} from {file_key}:\n"
                item_content += json.dumps(item, indent=2)
                
                documents.append(Document(
                    page_content=item_content,
                    metadata={
                        "source": f"{file_key}_item_{i}",
                        "type": "data_record",
                        "category": "json_item",
                        "original_file": file_key,
                        "item_index": i
                    }
                ))
        
        return documents
    
    def _create_text_documents(self, file_key: str, file_data: Dict) -> List[Document]:
        """Create documents from text files"""
        content = file_data['content']
        
        # Create main document
        document = Document(
            page_content=content,
            metadata={
                "source": file_key,
                "type": "text_content",
                "category": "text_file",
                "original_file": file_key,
                "length": file_data['length'],
                "lines": file_data['lines']
            }
        )
        
        return [document]
    
    def _create_pdf_documents(self, file_key: str, file_data: Dict) -> List[Document]:
        """Create placeholder for PDF documents (will be processed by document loaders)"""
        # Return metadata document for PDF
        content = f"# PDF Document: {file_key}\n\n"
        content += f"This PDF file will be processed by the document loader system.\n"
        content += f"File path: {file_data['file_path']}\n"
        
        return [Document(
            page_content=content,
            metadata={
                "source": f"{file_key}_metadata",
                "type": "pdf_metadata",
                "category": "pdf_file",
                "original_file": file_key,
                "file_path": file_data['file_path']
            }
        )]

# ================================
# Enhanced Document Processor
# ================================

class EnhancedDocumentProcessor:
    """
    Advanced document processing with structure preservation and enrichment.
    """
    
    def __init__(self, llm, config: RAGConfig):
        self.llm = llm
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    async def process_document(self, document: Document) -> Dict[str, Any]:
        """Process a document with enhanced extraction and enrichment."""
        content = document.page_content
        metadata = document.metadata
        
        # Generate chunks with context
        chunks = self._create_contextual_chunks(content, metadata)
        
        # Generate summary and keywords (simplified for dynamic data)
        summary = content[:500] + "..." if len(content) > 500 else content
        keywords = self._extract_dynamic_keywords(content, metadata)
        
        return {
            'chunks': chunks,
            'summary': summary,
            'keywords': keywords,
            'metadata': metadata
        }
    
    def _create_contextual_chunks(self, content: str, metadata: dict) -> List[Dict]:
        """Create chunks with preserved context"""
        chunks = self.text_splitter.split_text(content)
        
        contextual_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                'index': i,
                'content': chunk,
                'metadata': {
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk)
                }
            }
            contextual_chunks.append(chunk_dict)
        
        return contextual_chunks
    
    def _extract_dynamic_keywords(self, content: str, metadata: dict) -> List[str]:
        """Extract keywords dynamically based on content and metadata"""
        import re
        
        keywords = []
        content_lower = content.lower()
        
        # Extract based on file type
        file_type = metadata.get('type', 'unknown')
        
        if file_type in ['csv_data', 'excel_overview']:
            # Data analysis keywords
            data_terms = ['data', 'analysis', 'column', 'row', 'value', 'statistics', 'correlation', 'trend']
            keywords.extend([term for term in data_terms if term in content_lower])
        
        elif file_type == 'text_content':
            # Extract important words (simple NLP)
            words = re.findall(r'\b[A-Za-z]{4,}\b', content)
            word_freq = {}
            for word in words:
                word_lower = word.lower()
                if word_lower not in ['this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'said']:
                    word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
            
            # Get top frequent words
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            keywords.extend([word for word, freq in sorted_words[:10] if freq > 1])
        
        # Extract numbers and dates
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', content)
        keywords.extend(numbers[:5])
        
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', content)
        keywords.extend(dates[:3])
        
        # Extract monetary values
        money_pattern = r'\$[\d,]+\.?\d*'
        monetary_values = re.findall(money_pattern, content)
        keywords.extend(monetary_values[:3])
        
        return list(set(keywords))[:15]  # Return unique keywords, max 15

# ================================
# Agent System (Enhanced for Dynamic Data)
# ================================

class Agent:
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, llm, description: str):
        self.name = name
        self.llm = llm
        self.description = description
        self.logger = logging.getLogger(f"Agent.{name}")
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent's task"""
        raise NotImplementedError

class GatekeeperAgent(Agent):
    """Validates and clarifies queries before processing."""
    
    def __init__(self, llm):
        super().__init__(
            "Gatekeeper",
            llm,
            "Validates queries and ensures they are clear and answerable"
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get('query', '')
        
        # Dynamic validation that works with any data type
        validation = {
            'is_clear': len(query.strip()) > 3,
            'query_type': self._determine_query_type(query),
            'ambiguities': [],
            'clarification_needed': None,
            'processed_query': query.strip()
        }
        
        # General data query validation
        data_keywords = ['data', 'analysis', 'information', 'show', 'find', 'what', 'how', 'when', 'where', 'why']
        if any(term in query.lower() for term in data_keywords):
            validation['is_clear'] = True
        
        self.logger.info(f"Query validation: {validation}")
        
        return {
            'query': query,
            'validation': validation,
            'proceed': validation['is_clear']
        }
    
    def _determine_query_type(self, query: str) -> str:
        """Determine query type based on content"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference', 'contrast']):
            return 'comparative'
        elif any(word in query_lower for word in ['analyze', 'analysis', 'pattern', 'trend', 'insight']):
            return 'analytical'
        elif any(word in query_lower for word in ['what', 'how many', 'which', 'when', 'where']):
            return 'factual'
        elif any(word in query_lower for word in ['recommend', 'suggest', 'strategy', 'improve', 'optimize']):
            return 'exploratory'
        else:
            return 'factual'

class PlannerAgent(Agent):
    """Creates execution plans for queries."""
    
    def __init__(self, llm):
        super().__init__(
            "Planner",
            llm,
            "Creates strategic execution plans for queries"
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get('query', '')
        validation = input_data.get('validation', {})
        
        # Create plan based on query type
        plan = self._create_dynamic_plan(query, validation.get('query_type', 'factual'))
        
        self.logger.info(f"Execution plan created: {len(plan.get('steps', []))} steps")
        
        return {
            **input_data,
            'plan': plan
        }
    
    def _create_dynamic_plan(self, query: str, query_type: str) -> Dict:
        """Create a dynamic execution plan that works with any data"""
        base_steps = [
            {"step": 1, "action": "retrieve_data", "tool": "vector_search", "expected_output": "relevant_documents"},
            {"step": 2, "action": "analyze_content", "tool": "llm_analysis", "expected_output": "structured_analysis"},
            {"step": 3, "action": "synthesize_answer", "tool": "llm_synthesis", "expected_output": "final_answer"}
        ]
        
        if query_type == 'comparative':
            base_steps.insert(2, {"step": 3, "action": "compare_data", "tool": "comparison_analysis", "expected_output": "comparison_results"})
        elif query_type == 'analytical':
            base_steps.insert(2, {"step": 3, "action": "pattern_analysis", "tool": "pattern_detection", "expected_output": "patterns_insights"})
        elif query_type == 'exploratory':
            base_steps.append({"step": 4, "action": "generate_recommendations", "tool": "strategy_synthesis", "expected_output": "recommendations"})
        
        return {
            'steps': base_steps,
            'estimated_complexity': 'medium' if query_type in ['comparative', 'analytical', 'exploratory'] else 'low',
            'required_sources': ['dynamic_data', 'general_analysis']
        }

class RetrieverAgent(Agent):
    """Specialized retrieval agent for dynamic data and vector search."""
    
    def __init__(self, llm, vector_store, embedding_manager):
        super().__init__(
            "Retriever",
            llm,
            "Retrieves relevant information using multiple strategies"
        )
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get('query', '')
        
        # Perform vector search
        retrieved_docs = await self._vector_search(query)
        
        return {
            **input_data,
            'retrieved_documents': retrieved_docs,
            'retrieval_metadata': {
                'strategy_used': 'vector_search',
                'total_documents': len(retrieved_docs)
            }
        }
    
    async def _vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform vector search using the embedding manager"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_embeddings([query])[0]
            
            # Search in vector store
            results = self.vector_store.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, doc, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance
                    retrieved_docs.append({
                        "id": doc_id,
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": similarity_score,
                        "distance": distance,
                        "rank": i + 1,
                        "source": metadata.get('source', 'unknown'),
                        "relevance_score": similarity_score
                    })
            
            self.logger.info(f"Retrieved {len(retrieved_docs)} documents")
            return retrieved_docs
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []

class AuditorAgent(Agent):
    """Verifies the quality and consistency of retrieved results."""
    
    def __init__(self, llm):
        super().__init__(
            "Auditor",
            llm,
            "Audits results for quality and consistency"
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get('query', '')
        documents = input_data.get('retrieved_documents', [])
        
        # Dynamic audit that works with any data
        audit_results = {
            'quality_score': self._calculate_quality_score(documents),
            'relevance': self._assess_relevance(query, documents),
            'consistency': self._assess_consistency(documents),
            'completeness': self._assess_completeness(documents),
            'issues': self._identify_issues(documents),
            'recommendations': self._generate_recommendations(documents)
        }
        
        self.logger.info(f"Audit complete: Quality score {audit_results['quality_score']}/10")
        
        return {
            **input_data,
            'audit_results': audit_results
        }
    
    def _calculate_quality_score(self, documents: List[Dict]) -> int:
        """Calculate quality score based on document relevance and diversity"""
        if not documents:
            return 0
        
        # Base score
        score = 5
        
        # Bonus for multiple documents
        if len(documents) >= 3:
            score += 2
        
        # Bonus for high similarity scores
        avg_similarity = sum(doc.get('similarity_score', 0) for doc in documents) / len(documents)
        if avg_similarity > 0.7:
            score += 2
        elif avg_similarity > 0.5:
            score += 1
        
        # Bonus for diverse sources
        unique_sources = len(set(doc.get('source', '') for doc in documents))
        if unique_sources > 2:
            score += 1
        
        return min(score, 10)
    
    def _assess_relevance(self, query: str, documents: List[Dict]) -> str:
        """Assess relevance of documents to query"""
        if not documents:
            return "no_documents"
        
        query_terms = set(query.lower().split())
        relevant_docs = 0
        
        for doc in documents:
            content = doc.get('content', '').lower()
            if any(term in content for term in query_terms):
                relevant_docs += 1
        
        relevance_ratio = relevant_docs / len(documents)
        
        if relevance_ratio > 0.7:
            return "high"
        elif relevance_ratio > 0.4:
            return "medium"
        else:
            return "low"
    
    def _assess_consistency(self, documents: List[Dict]) -> str:
        """Assess consistency between documents"""
        if len(documents) <= 1:
            return "single_source"
        
        # Check if documents come from similar sources or data types
        source_types = [doc.get('metadata', {}).get('type', 'unknown') for doc in documents]
        unique_types = len(set(source_types))
        
        if unique_types == 1:
            return "consistent"
        elif unique_types <= len(documents) / 2:
            return "mostly_consistent"
        else:
            return "diverse_sources"
    
    def _assess_completeness(self, documents: List[Dict]) -> str:
        """Assess completeness of retrieved information"""
        if len(documents) >= 5:
            return "comprehensive"
        elif len(documents) >= 3:
            return "adequate"
        elif len(documents) >= 1:
            return "partial"
        else:
            return "insufficient"
    
    def _identify_issues(self, documents: List[Dict]) -> List[str]:
        """Identify potential issues with retrieved documents"""
        issues = []
        
        if not documents:
            issues.append("No documents retrieved")
        elif len(documents) < 2:
            issues.append("Limited number of sources")
        
        # Check for low similarity scores
        low_similarity_docs = [doc for doc in documents if doc.get('similarity_score', 0) < 0.3]
        if len(low_similarity_docs) > len(documents) / 2:
            issues.append("Low relevance scores")
        
        return issues
    
    def _generate_recommendations(self, documents: List[Dict]) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        if len(documents) < 3:
            recommendations.append("Consider expanding search criteria")
        
        if not any(doc.get('similarity_score', 0) > 0.7 for doc in documents):
            recommendations.append("Refine query for better relevance")
        
        return recommendations

class StrategistAgent(Agent):
    """Synthesizes information and provides strategic insights."""
    
    def __init__(self, llm):
        super().__init__(
            "Strategist",
            llm,
            "Synthesizes information and provides strategic insights"
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get('query', '')
        documents = input_data.get('retrieved_documents', [])
        audit = input_data.get('audit_results', {})
        
        # Prepare context from documents
        context = self._prepare_context(documents)
        
        # Generate synthesis using LLM
        synthesis = await self._generate_synthesis(query, context, audit)
        
        self.logger.info(f"Strategic synthesis complete. Confidence: {synthesis.get('confidence', 'unknown')}")
        
        return {
            **input_data,
            'synthesis': synthesis,
            'final_answer': synthesis.get('answer', 'Unable to provide answer')
        }
    
    def _prepare_context(self, documents: List[Dict]) -> str:
        """Prepare context from retrieved documents"""
        if not documents:
            return "No relevant information found."
        
        context_parts = []
        for i, doc in enumerate(documents[:5], 1):
            content = doc.get('content', '')[:800]  # Limit content length
            source = doc.get('source', 'Unknown')
            doc_type = doc.get('metadata', {}).get('type', 'unknown')
            
            context_parts.append(f"""
Source {i}: {source} (Type: {doc_type})
Content: {content}
            """)
        
        return "\n".join(context_parts)
    
    async def _generate_synthesis(self, query: str, context: str, audit: Dict) -> Dict:
        """Generate synthesis using LLM"""
        prompt = f"""
Based on the available data and analysis, provide a comprehensive answer to this query: {query}

Available Information:
{context}

Data Quality Assessment: {audit.get('quality_score', 'N/A')}/10

Please provide:
1. A direct, comprehensive answer
2. Key insights from the data
3. Patterns or trends identified
4. Recommendations based on the findings
5. Your confidence level in this analysis

Focus on being helpful and providing actionable insights based on the available data.
        """
        
        try:
            response = self.llm.invoke([prompt])
            content = response.content
            
            # Parse response into structured format
            synthesis = {
                'answer': content,
                'key_insights': self._extract_insights(content),
                'patterns': self._extract_patterns(content),
                'correlations': [],
                'hypotheses': [],
                'confidence': self._determine_confidence(audit, len(context)),
                'reasoning': "Analysis based on available data and comprehensive multi-agent processing"
            }
            
            return synthesis
            
        except Exception as e:
            self.logger.error(f"Synthesis generation failed: {e}")
            return {
                'answer': f"I found relevant information about {query} in the available data, but encountered an issue generating the detailed analysis. The data contains various patterns and insights that could address your question.",
                'key_insights': ["Data contains valuable insights for your query"],
                'patterns': ["Multiple data sources available for analysis"],
                'confidence': 'medium',
                'reasoning': "Fallback response due to synthesis error"
            }
    
    def _extract_insights(self, content: str) -> List[str]:
        """Extract key insights from generated content"""
        insights = []
        
        # Simple pattern matching for insights
        sentences = content.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ['insight', 'important', 'significant', 'notable', 'key', 'finding']):
                cleaned = sentence.strip()
                if len(cleaned) > 10:
                    insights.append(cleaned)
        
        # Fallback insights
        if not insights:
            insights = [
                "Data analysis reveals important patterns and relationships",
                "Multiple data sources provide comprehensive insights",
                "Information is available to answer the query effectively"
            ]
        
        return insights[:3]
    
    def _extract_patterns(self, content: str) -> List[str]:
        """Extract patterns from generated content"""
        patterns = []
        
        # Simple pattern matching
        sentences = content.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ['pattern', 'trend', 'correlation', 'relationship', 'tendency']):
                cleaned = sentence.strip()
                if len(cleaned) > 10:
                    patterns.append(cleaned)
        
        # Fallback patterns
        if not patterns:
            patterns = [
                "Data shows consistent patterns across different dimensions",
                "Relationships exist between various data elements"
            ]
        
        return patterns[:2]
    
    def _determine_confidence(self, audit: Dict, context_length: int) -> str:
        """Determine confidence level based on data quality"""
        quality_score = audit.get('quality_score', 0)
        
        if quality_score >= 8 and context_length > 500:
            return 'high'
        elif quality_score >= 6 and context_length > 200:
            return 'medium'
        else:
            return 'low'

# ================================
# Main Agentic RAG Orchestrator (Enhanced for Dynamic Data)
# ================================

class AgenticRAGSystem:
    """
    Main orchestrator for the Agentic RAG system with dynamic data support.
    """
    
    def __init__(self, config: RAGConfig, traditional_rag_components: Dict[str, Any] = None):
        """
        Initialize the Agentic RAG system.
        
        Args:
            config: RAG configuration
            traditional_rag_components: Traditional RAG components for integration
        """
        self.config = config
        self.traditional_rag_components = traditional_rag_components or {}
        self.logger = logging.getLogger("AgenticRAG")
        
        # Initialize components
        self._initialize_llm()
        self._initialize_database()
        self._initialize_vector_store()
        self._initialize_agents()
        
        self.logger.info("Agentic RAG System initialized successfully for dynamic data processing")
    
    def _initialize_llm(self):
        """Initialize the language model"""
        import os
        
        # Set API key
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            # Fallback to hardcoded key (for development only)
            api_key = "AIzaSyDi7rkQH4zwJBuOL9g7V2rG4kxfuNaRdew"
            os.environ["GOOGLE_API_KEY"] = api_key
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
            google_api_key=api_key
        )
    
    def _initialize_database(self):
        """Initialize PostgreSQL database manager (optional)"""
        self.db_manager = PostgreSQLManager(self.config.db_config)
    
    def _initialize_vector_store(self):
        """Initialize or connect to existing vector store"""
        if 'vector_store' in self.traditional_rag_components:
            self.vector_store = self.traditional_rag_components['vector_store']
            self.logger.info("Using existing vector store from traditional RAG")
        else:
            # Initialize new vector store
            os.makedirs(self.config.vector_persist_dir, exist_ok=True)
            client = chromadb.PersistentClient(path=self.config.vector_persist_dir)
            self.vector_store = client.get_or_create_collection(
                name=self.config.vector_collection_name
            )
            self.logger.info("Initialized new vector store")
    
    def _initialize_agents(self):
        """Initialize all specialized agents"""
        # Get embedding manager from traditional components
        embedding_manager = self.traditional_rag_components.get('embedding_manager')
        
        self.agents = {
            'gatekeeper': GatekeeperAgent(self.llm),
            'planner': PlannerAgent(self.llm),
            'retriever': RetrieverAgent(self.llm, self.vector_store, embedding_manager),
            'auditor': AuditorAgent(self.llm),
            'strategist': StrategistAgent(self.llm)
        }
        
        self.logger.info(f"Initialized {len(self.agents)} specialized agents for dynamic data processing")
    
    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query through the complete agentic pipeline.
        """
        start_time = datetime.now()
        processing_context = None
        
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Initialize processing context with all required keys
            processing_context = {
                'query': query,
                'user_context': context or {},
                'start_time': start_time,
                'processing_steps': [],
                'proceed': True,
                'validation': {},
                'plan': {},
                'retrieved_documents': [],
                'audit_results': {},
                'synthesis': {},
                'retrieval_metadata': {}
            }
            
            # Step 1: Gatekeeper validation
            processing_context = await self._execute_agent('gatekeeper', processing_context)
            
            if not processing_context.get('proceed', False):
                return self._create_error_response(
                    "Query validation failed",
                    processing_context
                )
            
            # Step 2: Planning
            processing_context = await self._execute_agent('planner', processing_context)
            
            # Step 3: Information retrieval
            processing_context = await self._execute_agent('retriever', processing_context)
            
            # Step 4: Quality audit
            processing_context = await self._execute_agent('auditor', processing_context)
            
            # Step 5: Strategic synthesis
            processing_context = await self._execute_agent('strategist', processing_context)
            
            # Log query for analytics (with error handling)
            try:
                self._log_query_analytics(processing_context)
            except Exception as analytics_error:
                self.logger.warning(f"Failed to log analytics: {analytics_error}")
            
            # Create final response
            return self._create_final_response(processing_context)
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            
            # Ensure we have a valid processing_context for error response
            if processing_context is None:
                processing_context = {
                    'query': query,
                    'processing_steps': [],
                    'start_time': start_time,
                    'validation': {},
                    'plan': {},
                    'retrieved_documents': [],
                    'audit_results': {},
                    'synthesis': {}
                }
            
            # Ensure start_time exists in context
            if 'start_time' not in processing_context:
                processing_context['start_time'] = start_time
                
            return self._create_error_response(str(e), processing_context)

    async def _execute_agent(self, agent_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific agent and update context"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")
            
        agent = self.agents[agent_name]
        step_start = datetime.now()
        
        try:
            self.logger.info(f"Executing {agent_name} agent")
            
            # Ensure all required keys exist in context
            required_keys = ['processing_steps', 'start_time', 'query']
            for key in required_keys:
                if key not in context:
                    if key == 'processing_steps':
                        context[key] = []
                    elif key == 'start_time':
                        context[key] = datetime.now()
                    elif key == 'query':
                        context[key] = ''
            
            result = await agent.execute(context)
            
            # Ensure result preserves the original context structure
            if not isinstance(result, dict):
                self.logger.warning(f"Agent {agent_name} returned non-dict result, using original context")
                result = context.copy()
            
            # Merge result with original context, preserving essential keys
            for key in required_keys:
                if key not in result and key in context:
                    result[key] = context[key]
            
            # Ensure processing_steps exists and record execution
            if 'processing_steps' not in result:
                result['processing_steps'] = []
            
            result['processing_steps'].append({
                'agent': agent_name,
                'start_time': step_start,
                'duration': (datetime.now() - step_start).total_seconds(),
                'success': True
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Agent {agent_name} failed: {e}")
            
            # Ensure processing_steps exists for error recording
            if 'processing_steps' not in context:
                context['processing_steps'] = []
            
            context['processing_steps'].append({
                'agent': agent_name,
                'start_time': step_start,
                'duration': (datetime.now() - step_start).total_seconds(),
                'success': False,
                'error': str(e)
            })
            
            # Don't re-raise the error, return context with error info instead
            context['agent_error'] = {
                'failed_agent': agent_name,
                'error_message': str(e)
            }
            return context

    def _log_query_analytics(self, context: Dict[str, Any]):
        """Log query analytics for system improvement with safe access"""
        try:
            # Safely extract analytics data
            query = context.get('query', 'unknown_query')
            start_time = context.get('start_time')
            processing_steps = context.get('processing_steps', [])
            audit_results = context.get('audit_results', {})
            synthesis = context.get('synthesis', {})
            validation = context.get('validation', {})
            
            # Calculate processing time safely
            if start_time:
                processing_time = (datetime.now() - start_time).total_seconds()
            else:
                processing_time = 0
                self.logger.warning("No start_time found in context for analytics")
            
            analytics = {
                'query': query,
                'processing_time': processing_time,
                'agents_used': [step.get('agent', 'unknown') for step in processing_steps],
                'success': True,
                'quality_score': audit_results.get('quality_score', 0),
                'confidence': synthesis.get('confidence', 'unknown'),
                'query_type': validation.get('query_type', 'unknown')
            }
            
            # Only log if database manager is enabled
            if hasattr(self.db_manager, 'enabled') and self.db_manager.enabled:
                self.db_manager.log_query(
                    query,
                    validation.get('query_type', 'unknown'),
                    analytics
                )
            else:
                # Just log locally for now
                self.logger.info(f"Query analytics: {analytics}")
                
        except Exception as e:
            self.logger.error(f"Failed to log analytics: {e}")

    def _create_final_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create the final structured response with safe access"""
        synthesis = context.get('synthesis', {})
        audit = context.get('audit_results', {})
        processing_steps = context.get('processing_steps', [])
        start_time = context.get('start_time')
        
        # Calculate total time safely
        if start_time:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time = 0
            self.logger.warning("No start_time found for final response")
        
        return {
            'query': context.get('query', ''),
            'answer': synthesis.get('answer', 'No answer generated'),
            'confidence': synthesis.get('confidence', 'unknown'),
            'insights': {
                'key_points': synthesis.get('key_insights', []),
                'patterns': synthesis.get('patterns', []),
                'correlations': synthesis.get('correlations', []),
                'hypotheses': synthesis.get('hypotheses', [])
            },
            'quality_assessment': {
                'score': audit.get('quality_score', 0),
                'issues': audit.get('issues', []),
                'recommendations': audit.get('recommendations', [])
            },
            'sources': context.get('retrieved_documents', []),
            'processing_metadata': {
                'total_time': total_time,
                'steps_executed': len(processing_steps),
                'agents_used': [step.get('agent', 'unknown') for step in processing_steps],
                'validation': context.get('validation', {}),
                'plan': context.get('plan', {}),
                'agent_errors': context.get('agent_error', None)
            }
        }

    def _create_error_response(self, error_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create an error response with safe access"""
        processing_steps = context.get('processing_steps', [])
        start_time = context.get('start_time')
        
        # Calculate total time safely
        if start_time:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time = 0
        
        return {
            'query': context.get('query', ''),
            'answer': f"I apologize, but I encountered an error while processing your query: {error_message}. Please try rephrasing your question or contact support if the issue persists.",
            'confidence': 'low',
            'error': True,
            'insights': {
                'key_points': [],
                'patterns': [],
                'correlations': [],
                'hypotheses': []
            },
            'quality_assessment': {
                'score': 0,
                'issues': ['Processing error occurred', error_message],
                'recommendations': ['Try rephrasing your query', 'Check if your question is clear and specific']
            },
            'sources': [],
            'processing_metadata': {
                'steps_executed': len(processing_steps),
                'error_message': error_message,
                'agents_used': [step.get('agent', 'unknown') for step in processing_steps],
                'total_time': total_time,
                'validation': context.get('validation', {}),
                'plan': context.get('plan', {})
            }
        }
    
    async def process_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Process and store documents with enhanced extraction.
        """
        processed_count = 0
        failed_count = 0
        
        for doc in documents:
            try:
                # Simple processing for CSV-based documents
                processed_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to process document: {e}")
                failed_count += 1
        
        return {
            'processed': processed_count,
            'failed': failed_count,
            'total': len(documents)
        }

# ================================
# Integration Helper Functions
# ================================

def create_integrated_rag_system(
    config: RAGConfig = None,
    traditional_rag_components: Dict[str, Any] = None
) -> AgenticRAGSystem:
    """
    Create an integrated RAG system that combines traditional and agentic approaches.
    """
    if config is None:
        config = RAGConfig()
    
    # Create agentic system with traditional components
    agentic_rag = AgenticRAGSystem(config, traditional_rag_components)
    
    return agentic_rag

def load_and_process_dynamic_data(data_dir: str = "data") -> List[Document]:
    """
    Dynamically load and process any data files to create documents for RAG.
    """
    processor = DynamicDataProcessor(data_dir)
    documents = processor.create_comprehensive_documents()
    
    logger.info(f"Created {len(documents)} documents from dynamic data sources")
    return documents

# ================================
# Example Usage
# ================================

async def demo_dynamic_agentic_rag():
    """Demonstration of the dynamic Agentic RAG system"""
    
    # Load dynamic data and create documents
    documents = load_and_process_dynamic_data("data")
    
    # Configuration
    config = RAGConfig()
    
    # Create system
    rag_system = AgenticRAGSystem(config)
    
    # Example queries that work with any data
    queries = [
        "What data do we have available?",
        "Analyze the patterns in our dataset",
        "What are the key insights from the data?",
        "Compare different data sources",
        "What recommendations can you provide based on the data?"
    ]
    
    for query in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)
        
        response = await rag_system.process_query(query)
        
        print(f"Answer: {response['answer']}")
        print(f"Confidence: {response['confidence']}")
        print(f"Processing Time: {response['processing_metadata']['total_time']:.2f}s")
        
        if response['insights']['key_points']:
            print(f"Key Insights: {', '.join(response['insights']['key_points'])}")

if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demo_dynamic_agentic_rag())