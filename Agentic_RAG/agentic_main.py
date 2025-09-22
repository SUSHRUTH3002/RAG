"""
Agentic RAG Workflow - Similar to RAG.py but with multi-agent architecture
Processes hotel food sales data and provides agentic RAG capabilities
"""

import asyncio
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Agentic_RAG.agentic_rag import (
    AgenticRAGSystem, 
    RAGConfig, 
    create_integrated_rag_system
)
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgenticRAGWorkflow:
    """
    Agentic RAG Workflow for Hotel Food Sales Data
    Similar structure to your existing RAGWorkflow but with agentic capabilities
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 collection_name: str = "hotel_sales_docs",
                 chunk_size: int = 800,
                 chunk_overlap: int = 100):
        """Initialize the Agentic RAG Workflow"""
        
        self.data_dir = Path(data_dir)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Core components
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.embedding_manager = None
        self.vector_store = None
        self.agentic_rag = None
        self.sales_data = None
        
        # Status tracking
        self.is_ready = False
        self.status = "Initialized"
        
        logger.info("Agentic RAG Workflow initialized")
    
    def setup_components(self):
        """Set up all RAG components - similar to your existing setup"""
        try:
            self.status = "Setting up components..."
            logger.info("🔧 Setting up Agentic RAG components...")
            
            # Set up embedding manager
            self._setup_embedding_manager()
            
            # Set up vector store
            self._setup_vector_store()
            
            # Set up text splitter
            self._setup_text_splitter()
            
            self.status = "Components ready"
            logger.info("✅ Components setup completed")
            return True
            
        except Exception as e:
            self.status = f"Setup error: {str(e)}"
            logger.error(f"❌ Error setting up components: {e}")
            return False
    
    def _setup_embedding_manager(self):
        """Setup embedding manager"""
        class EmbeddingManager:
            def __init__(self, model_name: str):
                self.model_name = model_name
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            
            def generate_embeddings(self, texts: List[str]) -> np.ndarray:
                logger.info(f"Generating embeddings for {len(texts)} texts")
                return self.model.encode(texts, show_progress_bar=True)
        
        self.embedding_manager = EmbeddingManager(self.embedding_model)
    
    def _setup_vector_store(self):
        """Setup vector store"""
        class VectorStore:
            def __init__(self, collection_name: str, persist_directory: str):
                self.collection_name = collection_name
                self.persist_directory = persist_directory
                
                os.makedirs(persist_directory, exist_ok=True)
                self.client = chromadb.PersistentClient(path=persist_directory)
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": "Hotel food sales documents"}
                )
                logger.info(f"Vector store initialized: {collection_name}")
            
            def add_documents(self, documents: List[Any], embeddings: np.ndarray):
                ids = []
                metadatas = []
                documents_text = []
                embeddings_list = []
                
                for i, (doc, emb) in enumerate(zip(documents, embeddings)):
                    doc_id = str(uuid.uuid4())
                    ids.append(doc_id)
                    
                    # Clean metadata to ensure ChromaDB compatibility
                    metadata = dict(doc.metadata)
                    metadata['doc_index'] = i
                    metadata['context_length'] = len(doc.page_content)
                    
                    # Convert any list values to strings
                    for key, value in metadata.items():
                        if isinstance(value, list):
                            metadata[key] = ", ".join(str(item) for item in value)
                        elif isinstance(value, dict):
                            metadata[key] = str(value)
                        elif value is None:
                            metadata[key] = ""
                        # Keep only str, int, float, bool types
                        elif not isinstance(value, (str, int, float, bool)):
                            metadata[key] = str(value)
                    
                    metadatas.append(metadata)
                    documents_text.append(doc.page_content)
                    embeddings_list.append(emb.tolist())
                
                self.collection.add(
                    ids=ids,
                    documents=documents_text,
                    embeddings=embeddings_list,
                    metadatas=metadatas
                )
                logger.info(f"Added {len(documents)} documents to vector store")
        
        persist_dir = str(self.data_dir / "agentic_vector_store")
        self.vector_store = VectorStore(self.collection_name, persist_dir)
    
    def _setup_text_splitter(self):
        """Setup text splitter"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_and_analyze_sales_data(self):
        """Load and analyze the hotel food sales CSV data"""
        try:
            self.status = "Loading sales data..."
            logger.info("📊 Loading hotel food sales data...")
            
            csv_file = self.data_dir / "hotel_food_sales.csv"
            if not csv_file.exists():
                logger.error(f"❌ Sales data file not found: {csv_file}")
                return False
            
            # Load CSV data
            self.sales_data = pd.read_csv(csv_file)
            logger.info(f"✅ Loaded {len(self.sales_data)} sales records")
            
            # Parse dates and add analysis columns
            self.sales_data['OrderTime'] = pd.to_datetime(self.sales_data['OrderTime'])
            self.sales_data['Date'] = self.sales_data['OrderTime'].dt.date
            self.sales_data['Hour'] = self.sales_data['OrderTime'].dt.hour
            self.sales_data['DayOfWeek'] = self.sales_data['OrderTime'].dt.day_name()
            self.sales_data['IsWeekend'] = self.sales_data['OrderTime'].dt.weekday >= 5
            self.sales_data['Month'] = self.sales_data['OrderTime'].dt.month_name()
            
            return True
            
        except Exception as e:
            self.status = f"Data loading error: {str(e)}"
            logger.error(f"❌ Error loading sales data: {e}")
            return False
    
    def generate_analytical_reports(self):
        """Generate analytical reports from sales data"""
        try:
            self.status = "Generating reports..."
            logger.info("📋 Generating analytical reports...")
            
            if self.sales_data is None:
                logger.error("❌ No sales data available")
                return False
            
            reports = []
            
            # Business Intelligence Report
            bi_report = self._create_business_report()
            reports.append(Document(
                page_content=bi_report,
                metadata={"source": "business_intelligence_report.txt", "type": "analysis"}
            ))
            
            # Menu Performance Report
            menu_report = self._create_menu_report()
            reports.append(Document(
                page_content=menu_report,
                metadata={"source": "menu_performance_report.txt", "type": "analysis"}
            ))
            
            # Customer Analysis Report
            customer_report = self._create_customer_report()
            reports.append(Document(
                page_content=customer_report,
                metadata={"source": "customer_analysis_report.txt", "type": "analysis"}
            ))
            
            self.documents.extend(reports)
            logger.info(f"✅ Generated {len(reports)} analytical reports")
            return True
            
        except Exception as e:
            self.status = f"Report generation error: {str(e)}"
            logger.error(f"❌ Error generating reports: {e}")
            return False
    
    def _create_business_report(self):
        """Create business intelligence report"""
        daily_revenue = self.sales_data.groupby('Date')['TotalAmount'].sum()
        category_performance = self.sales_data.groupby('Category').agg({
            'TotalAmount': ['sum', 'count', 'mean']
        }).round(2)
        
        report = f"""
# Hotel Restaurant Business Intelligence Report

## Executive Summary
Analysis of {len(self.sales_data):,} orders from {self.sales_data['OrderTime'].min().date()} to {self.sales_data['OrderTime'].max().date()}.

## Financial Performance
- Total Revenue: ${self.sales_data['TotalAmount'].sum():,.2f}
- Average Order Value: ${self.sales_data['TotalAmount'].mean():.2f}
- Total Orders: {len(self.sales_data):,}
- Unique Customers: {self.sales_data['CustomerID'].nunique()}

## Top Revenue Items
{self.sales_data.groupby('MenuItem')['TotalAmount'].sum().sort_values(ascending=False).head(10).to_string()}

## Category Performance
{category_performance.to_string()}

## Payment Method Distribution
{self.sales_data.groupby('PaymentMethod').agg({'TotalAmount': 'sum', 'OrderID': 'count'}).to_string()}

## Peak Hours Analysis
{self.sales_data.groupby('Hour').agg({'TotalAmount': 'sum', 'OrderID': 'count'}).to_string()}

## Strategic Recommendations
1. Focus on promoting top revenue generating items
2. Optimize staffing during peak hours
3. Develop targeted marketing for underperforming categories
4. Implement customer loyalty programs for frequent diners
        """
        return report
    
    def _create_menu_report(self):
        """Create menu performance report"""
        menu_performance = self.sales_data.groupby('MenuItem').agg({
            'TotalAmount': ['sum', 'count', 'mean'],
            'Quantity': 'sum'
        }).round(2)
        
        report = f"""
# Menu Performance Analysis Report

## Menu Overview
Analysis of {self.sales_data['MenuItem'].nunique()} unique menu items.

## Top Performers by Revenue
{menu_performance.sort_values(('TotalAmount', 'sum'), ascending=False).head(15).to_string()}

## Most Popular Items by Order Count
{menu_performance.sort_values(('TotalAmount', 'count'), ascending=False).head(15).to_string()}

## Category Analysis
{self.sales_data.groupby('Category').agg({'TotalAmount': ['sum', 'mean'], 'Quantity': 'sum'}).to_string()}

## Price Point Analysis
- Premium Items (>$400): {len(self.sales_data[self.sales_data['Price'] > 400]['MenuItem'].unique())} items
- Mid-range Items ($200-$400): {len(self.sales_data[(self.sales_data['Price'] >= 200) & (self.sales_data['Price'] <= 400)]['MenuItem'].unique())} items
- Value Items (<$200): {len(self.sales_data[self.sales_data['Price'] < 200]['MenuItem'].unique())} items

## Menu Engineering Recommendations
1. Promote high-revenue, high-frequency items
2. Review pricing strategy for underperforming items
3. Consider seasonal menu variations
4. Expand successful categories
        """
        return report
    
    def _create_customer_report(self):
        """Create customer analysis report"""
        customer_stats = self.sales_data.groupby('CustomerID').agg({
            'TotalAmount': ['sum', 'count', 'mean'],
            'OrderTime': ['min', 'max']
        }).round(2)
        
        report = f"""
# Customer Analysis Report

## Customer Overview
Analysis of {self.sales_data['CustomerID'].nunique()} unique customers.

## Customer Value Distribution
- Average Customer Value: ${customer_stats[('TotalAmount', 'sum')].mean():.2f}
- Top 10% Customer Threshold: ${customer_stats[('TotalAmount', 'sum')].quantile(0.9):.2f}
- Most Valuable Customer: ${customer_stats[('TotalAmount', 'sum')].max():.2f}

## Customer Frequency
- Average Orders per Customer: {customer_stats[('TotalAmount', 'count')].mean():.1f}
- Most Frequent Customer: {customer_stats[('TotalAmount', 'count')].max()} orders

## Payment Preferences
{self.sales_data.groupby('PaymentMethod').agg({'CustomerID': 'nunique', 'TotalAmount': 'sum'}).to_string()}

## Customer Behavior by Day
{self.sales_data.groupby('DayOfWeek').agg({'CustomerID': 'nunique', 'TotalAmount': 'mean'}).to_string()}

## Retention Strategy Recommendations
1. Implement VIP program for high-value customers
2. Create loyalty rewards for frequent visitors
3. Develop targeted promotions by customer segment
4. Enhance customer experience during peak times
        """
        return report
    
    def load_and_process_documents(self):
        """Load and process documents dynamically from all files in data directory"""
        try:
            self.status = "Loading documents..."
            logger.info("📁 Dynamically loading and processing all documents...")
            
            # Use dynamic data processor instead of hardcoded hotel sales logic
            from Agentic_RAG.agentic_rag import DynamicDataProcessor
            
            data_processor = DynamicDataProcessor(str(self.data_dir))
            
            # Load all available data dynamically
            dynamic_documents = data_processor.create_comprehensive_documents()
            self.documents.extend(dynamic_documents)
            logger.info(f"✅ Loaded {len(dynamic_documents)} documents from dynamic data processor")
            
            # Also load traditional document types (PDF, TXT, etc.)
            self._load_traditional_documents()
            
            logger.info(f"✅ Total documents loaded: {len(self.documents)}")
            return len(self.documents) > 0
            
        except Exception as e:
            self.status = f"Document loading error: {str(e)}"
            logger.error(f"❌ Error loading documents: {e}")
            return False
    
    def _load_traditional_documents(self):
        """Load traditional document types (PDF, TXT, etc.)"""
        try:
            # Load PDF files
            for pdf_file in self.data_dir.glob("**/*.pdf"):
                try:
                    loader = PyMuPDFLoader(str(pdf_file))
                    docs = loader.load()
                    self.documents.extend(docs)
                    logger.info(f"✅ Loaded PDF: {pdf_file.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load PDF {pdf_file.name}: {e}")
            
            # Load text files
            for txt_file in self.data_dir.glob("**/*.txt"):
                try:
                    loader = TextLoader(str(txt_file), encoding="utf-8")
                    docs = loader.load()
                    self.documents.extend(docs)
                    logger.info(f"✅ Loaded text file: {txt_file.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load text file {txt_file.name}: {e}")
            
            # Load markdown files
            for md_file in self.data_dir.glob("**/*.md"):
                try:
                    loader = TextLoader(str(md_file), encoding="utf-8")
                    docs = loader.load()
                    self.documents.extend(docs)
                    logger.info(f"✅ Loaded markdown file: {md_file.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load markdown file {md_file.name}: {e}")
                
        except Exception as e:
            logger.error(f"❌ Error loading traditional documents: {e}")
    
    def create_embeddings_and_store(self):
        """Create embeddings and store in vector database"""
        try:
            self.status = "Creating embeddings..."
            logger.info("🔧 Creating document chunks and embeddings...")
            
            if not self.documents:
                logger.error("❌ No documents to process")
                return False
            
            # Create chunks
            all_chunks = []
            for doc in self.documents:
                chunks = self.text_splitter.split_documents([doc])
                all_chunks.extend(chunks)
            
            self.chunks = all_chunks
            logger.info(f"✅ Created {len(self.chunks)} document chunks")
            
            # Generate embeddings
            texts = [chunk.page_content for chunk in self.chunks]
            self.embeddings = self.embedding_manager.generate_embeddings(texts)
            
            # Store in vector database
            self.vector_store.add_documents(self.chunks, self.embeddings)
            
            self.status = "Embeddings created"
            logger.info("✅ Embeddings created and stored successfully")
            return True
            
        except Exception as e:
            self.status = f"Embedding error: {str(e)}"
            logger.error(f"❌ Error creating embeddings: {e}")
            return False
    
    def setup_agentic_rag(self):
        """Setup the agentic RAG system"""
        try:
            self.status = "Setting up agentic system..."
            logger.info("🤖 Setting up Agentic RAG system...")
            
            # Set API key for Google AI
            import os
            os.environ["GOOGLE_API_KEY"] = "AIzaSyDi7rkQH4zwJBuOL9g7V2rG4kxfuNaRdew"
            
            # Also set the legacy key name for compatibility
            os.environ["GEMINI_API_KEY"] = "AIzaSyDi7rkQH4zwJBuOL9g7V2rG4kxfuNaRdew"
            
            # Create configuration with explicit API key
            config = RAGConfig(
                vector_persist_dir=str(self.data_dir / "agentic_vector_store"),
                vector_collection_name=self.collection_name,
                llm_model="gemini-1.5-flash",
                llm_temperature=0.1,
                default_top_k=5,
                similarity_threshold=0.3,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # Create integrated system
            traditional_components = {
                'vector_store': self.vector_store.collection,
                'embedding_manager': self.embedding_manager
            }
            
            self.agentic_rag = create_integrated_rag_system(
                config=config,
                traditional_rag_components=traditional_components
            )
            
            self.status = "Agentic system ready"
            logger.info("✅ Agentic RAG system setup completed")
            return True
            
        except Exception as e:
            self.status = f"Agentic setup error: {str(e)}"
            logger.error(f"❌ Error setting up agentic RAG: {e}")
            return False
    
    def setup_retriever(self):
        """Setup retriever for traditional RAG compatibility"""
        class RAGRetriever:
            def __init__(self, vector_store, embedding_manager):
                self.vector_store = vector_store
                self.embedding_manager = embedding_manager
            
            def retrieve(self, query: str, top_k: int = 5):
                query_embedding = self.embedding_manager.generate_embeddings([query])[0]
                
                results = self.vector_store.collection.query(
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
                            "document": doc,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
                        })
                
                return retrieved_docs
        
        self.retriever = RAGRetriever(self.vector_store, self.embedding_manager)
    
    def run_complete_workflow(self, interactive: bool = False):
        """Run the complete workflow - similar to your existing workflow"""
        try:
            logger.info("🚀 Starting Agentic RAG Workflow for Hotel Food Sales")
            
            # Step 1: Setup components
            if not self.setup_components():
                return False
            
            # Step 2: Load and process documents
            if not self.load_and_process_documents():
                return False
            
            # Step 3: Create embeddings
            if not self.create_embeddings_and_store():
                return False
            
            # Step 4: Setup agentic RAG
            if not self.setup_agentic_rag():
                return False
            
            # Step 5: Setup retriever for compatibility
            self.setup_retriever()
            
            self.is_ready = True
            self.status = "Ready"
            logger.info("✅ Agentic RAG Workflow completed successfully!")
            
            if interactive:
                self.run_interactive_mode()
            
            return True
            
        except Exception as e:
            self.status = f"Workflow error: {str(e)}"
            logger.error(f"❌ Workflow failed: {e}")
            return False
    
    def run_interactive_mode(self):
        """Run interactive query mode"""
        logger.info("🔍 Entering interactive mode...")
        
        print("\n" + "=" * 80)
        print("🏨 HOTEL FOOD SALES - AGENTIC RAG ANALYSIS")
        print("=" * 80)
        print("Ask questions about the hotel food sales data and business operations.")
        print("Type 'quit' to exit, 'help' for examples, or 'stats' for data overview.")
        print("=" * 80)
        
        while True:
            try:
                print("\n" + "-" * 40)
                user_query = input("🔍 Enter your query: ").strip()
                
                if user_query.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif user_query.lower() == 'help':
                    self._show_example_queries()
                    continue
                elif user_query.lower() == 'stats':
                    self._show_data_stats()
                    continue
                elif not user_query:
                    continue
                
                # Process query with agentic RAG
                response = asyncio.run(self.process_agentic_query(user_query))
                self._display_response(response)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    async def process_agentic_query(self, query: str):
        """Process query using agentic RAG system"""
        try:
            logger.info(f"Processing agentic query: {query}")
            response = await self.agentic_rag.process_query(query)
            return response
        except Exception as e:
            logger.error(f"Error processing agentic query: {e}")
            return {
                'answer': f"Error processing query: {str(e)}",
                'confidence': 'low',
                'error': True
            }
    
    def _show_example_queries(self):
        """Show example queries"""
        examples = [
            "What are the top 5 best-selling menu items by revenue?",
            "How do weekend sales compare to weekday sales?",
            "Which payment method is most popular?",
            "What are the peak hours for restaurant operations?",
            "Which customers are the most valuable?",
            "What menu items should we promote or discontinue?",
            "How can we optimize staffing based on sales patterns?",
            "What are the seasonal trends in our data?"
        ]
        
        print("\n📋 Example Business Intelligence Queries:")
        for i, example in enumerate(examples, 1):
            print(f"  {i:2d}. {example}")
    
    def _show_data_stats(self):
        """Show data statistics"""
        if self.sales_data is not None:
            print(f"\n📊 Hotel Food Sales Data Overview:")
            print(f"  • Total Orders: {len(self.sales_data):,}")
            print(f"  • Total Revenue: ${self.sales_data['TotalAmount'].sum():,.2f}")
            print(f"  • Unique Customers: {self.sales_data['CustomerID'].nunique()}")
            print(f"  • Menu Items: {self.sales_data['MenuItem'].nunique()}")
            print(f"  • Date Range: {self.sales_data['OrderTime'].min().date()} to {self.sales_data['OrderTime'].max().date()}")
            print(f"  • Documents Processed: {len(self.documents)}")
            print(f"  • Document Chunks: {len(self.chunks)}")
    
    def _display_response(self, response):
        """Display agentic RAG response"""
        print(f"\n✅ Response:")
        print(f"🎯 Confidence: {response.get('confidence', 'unknown')}")
        
        if 'quality_assessment' in response:
            print(f"📊 Quality Score: {response['quality_assessment']['score']}/10")
        
        print(f"\n📝 Answer:")
        print(response.get('answer', 'No answer available'))
        
        if response.get('insights', {}).get('key_points'):
            print(f"\n💡 Key Insights:")
            for insight in response['insights']['key_points'][:3]:
                print(f"  • {insight}")
        
        if response.get('insights', {}).get('patterns'):
            print(f"\n🔍 Patterns Identified:")
            for pattern in response['insights']['patterns'][:2]:
                print(f"  • {pattern}")
        
        if 'processing_metadata' in response:
            agents_used = response['processing_metadata'].get('agents_used', [])
            if agents_used:
                print(f"\n🤖 Agents Used: {', '.join(agents_used)}")

# Main execution
if __name__ == "__main__":
    workflow = AgenticRAGWorkflow()
    success = workflow.run_complete_workflow(interactive=True)
    if not success:
        print("❌ Workflow failed!")
        sys.exit(1)
        logger.info(f"  Doc {i+1}: {source} ({content_length} chars)")
    
    async def _initialize_agentic_system(self):
        """Initialize the agentic RAG system"""
        logger.info("🤖 Initializing Agentic RAG System...")
        
        # Create configuration
        config = RAGConfig(
            vector_persist_dir=str(self.data_dir / "vector_store"),
            vector_collection_name="hotel_documents",
            llm_model="gemini-1.5-flash",
            llm_temperature=0.1,
            default_top_k=5,
            similarity_threshold=0.3,
            chunk_size=800,
            chunk_overlap=100
        )
        
        # Set API key
        os.environ["GEMINI_API_KEY"] = "AIzaSyDi7rkQH4zwJBuOL9g7V2rG4kxfuNaRdew"
        
        # Initialize system
        self.agentic_rag = AgenticRAGSystem(config)
        
        logger.info("✅ Agentic RAG System initialized")
        logger.info(f"  - Agents: {list(self.agentic_rag.agents.keys())}")
        logger.info(f"  - Vector Store: {config.vector_persist_dir}")
        logger.info(f"  - Collection: {config.vector_collection_name}")
    
    async def _process_documents(self):
        """Process documents through the agentic system"""
        logger.info("⚙️ Processing documents through agentic pipeline...")
        
        if not self.documents:
            logger.warning("No documents to process")
            return
        
        try:
            # Process documents
            results = await self.agentic_rag.process_documents(self.documents)
            
            logger.info(f"✅ Document processing completed: {results['processed']}/{results['total']} documents processed")
            if results['failed'] > 0:
                logger.warning(f"⚠️ {results['failed']} documents failed to process")
                
        except Exception as e:
            logger.error(f"❌ Error processing documents: {e}")
            # Continue with existing vector store if available
    
    async def _test_query_processing(self):
        """Test the system with various query types related to hotel food sales"""
        logger.info("🧪 Testing query processing with hotel food sales queries...")
        
        self.test_queries = [
            # Business Intelligence queries
            "What are the top 5 best-selling menu items by total revenue?",
            "How does weekend performance compare to weekday sales?",
            "Which payment method is most popular among customers?",
            "What is the average order value and how has it changed over time?",
            
            # Operational queries  
            "What are the peak hours for restaurant operations?",
            "Which menu categories generate the highest revenue?",
            "How can we optimize staffing based on order patterns?",
            "What are the most efficient payment processing methods?",
            
            # Customer Analysis queries
            "What are the customer dining patterns and preferences?",
            "Which customers are the most valuable to the business?",
            "How can we improve customer retention and loyalty?",
            "What are the customer segmentation insights?",
            
            # Strategic queries
            "What menu items should we promote or discontinue?",
            "What are the opportunities for revenue growth?",
            "How can we optimize our menu pricing strategy?",
            "What operational improvements would have the biggest impact?"
        ]
        
        results = []
        
        print("\n" + "=" * 80)
        print("🧪 HOTEL FOOD SALES QUERY TESTING RESULTS")
        print("=" * 80)
        
        for i, query in enumerate(self.test_queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            print("-" * 60)
            
            start_time = time.time()
            
            try:
                response = await self.agentic_rag.process_query(query)
                processing_time = time.time() - start_time
                
                # Store results
                results.append({
                    'query': query,
                    'success': True,
                    'response': response,
                    'processing_time': processing_time
                })
                
                # Display key information
                print(f"✅ Status: Success")
                print(f"⏱️ Processing Time: {processing_time:.2f}s")
                print(f"🎯 Confidence: {response['confidence']}")
                print(f"📊 Quality Score: {response['quality_assessment']['score']}/10")
                print(f"🤖 Agents Used: {', '.join(response['processing_metadata']['agents_used'])}")
                
                # Show answer preview
                answer = response['answer']
                preview = answer[:300] + "..." if len(answer) > 300 else answer
                print(f"📝 Answer Preview: {preview}")
                
                # Show insights if available
                if response['insights']['key_points']:
                    print(f"💡 Key Insight: {response['insights']['key_points'][0]}")
                
            except Exception as e:
                processing_time = time.time() - start_time
                
                results.append({
                    'query': query,
                    'success': False,
                    'error': str(e),
                    'processing_time': processing_time
                })
                
                print(f"❌ Status: Failed")
                print(f"⏱️ Processing Time: {processing_time:.2f}s")
                print(f"🚨 Error: {str(e)[:100]}...")
        
        self.test_results = results
        logger.info(f"✅ Query testing completed: {len([r for r in results if r['success']])}/{len(results)} successful")
    
    # ...existing code for other methods...
    
    async def _analyze_performance(self):
        """Analyze system performance metrics"""
        logger.info("📊 Analyzing system performance...")
        
        if not hasattr(self, 'test_results'):
            logger.warning("No test results available for analysis")
            return
        
        successful_results = [r for r in self.test_results if r['success']]
        failed_results = [r for r in self.test_results if not r['success']]
        
        # Calculate metrics
        total_queries = len(self.test_results)
        success_rate = len(successful_results) / total_queries * 100
        avg_processing_time = sum(r['processing_time'] for r in successful_results) / len(successful_results) if successful_results else 0
        
        # Confidence distribution
        confidence_dist = {}
        quality_scores = []
        agents_usage = {}
        
        for result in successful_results:
            response = result['response']
            
            # Confidence distribution
            confidence = response['confidence']
            confidence_dist[confidence] = confidence_dist.get(confidence, 0) + 1
            
            # Quality scores
            quality_scores.append(response['quality_assessment']['score'])
            
            # Agent usage
            for agent in response['processing_metadata']['agents_used']:
                agents_usage[agent] = agents_usage.get(agent, 0) + 1
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Performance report
        print("\n" + "=" * 80)
        print("📊 SYSTEM PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        print(f"\n📈 Overall Metrics:")
        print(f"  • Total Queries Processed: {total_queries}")
        print(f"  • Success Rate: {success_rate:.1f}%")
        print(f"  • Average Processing Time: {avg_processing_time:.2f}s")
        print(f"  • Average Quality Score: {avg_quality:.1f}/10")
        
        print(f"\n🎯 Confidence Distribution:")
        for confidence, count in confidence_dist.items():
            percentage = count / len(successful_results) * 100
            print(f"  • {confidence.capitalize()}: {count} queries ({percentage:.1f}%)")
        
        print(f"\n🤖 Agent Usage:")
        for agent, count in agents_usage.items():
            percentage = count / len(successful_results) * 100
            print(f"  • {agent.capitalize()}: {count} times ({percentage:.1f}%)")
        
        if failed_results:
            print(f"\n❌ Failed Queries:")
            for result in failed_results:
                print(f"  • {result['query'][:50]}... - {result['error'][:50]}...")
        
        logger.info("✅ Performance analysis completed")
    
    async def _generate_final_report(self):
        """Generate comprehensive final report"""
        logger.info("📋 Generating final execution report...")
        
        report = {
            'execution_summary': {
                'timestamp': datetime.now().isoformat(),
                'data_records_analyzed': len(self.sales_data) if self.sales_data is not None else 0,
                'total_documents_processed': len(self.documents),
                'total_queries_tested': len(getattr(self, 'test_results', [])),
                'pipeline_status': 'completed'
            },
            'data_analysis': {
                'total_revenue': float(self.sales_data['TotalAmount'].sum()) if self.sales_data is not None else 0,
                'total_orders': len(self.sales_data) if self.sales_data is not None else 0,
                'unique_customers': int(self.sales_data['CustomerID'].nunique()) if self.sales_data is not None else 0,
                'unique_menu_items': int(self.sales_data['MenuItem'].nunique()) if self.sales_data is not None else 0,
                'date_range': {
                    'start': str(self.sales_data['OrderTime'].min().date()) if self.sales_data is not None else None,
                    'end': str(self.sales_data['OrderTime'].max().date()) if self.sales_data is not None else None
                }
            },
            'system_configuration': {
                'llm_model': self.config.llm_model,
                'vector_store': self.config.vector_collection_name,
                'chunk_size': self.config.chunk_size,
                'similarity_threshold': self.config.similarity_threshold
            },
            'test_results': getattr(self, 'test_results', [])
        }
        
        # Save report
        with open(self.data_dir / "execution_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print("\n" + "=" * 80)
        print("📋 FINAL EXECUTION REPORT")
        print("=" * 80)
        
        print(f"\n✅ Pipeline Execution Summary:")
        print(f"  • Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  • Sales Records Analyzed: {report['data_analysis']['total_orders']:,}")
        print(f"  • Total Revenue Analyzed: ${report['data_analysis']['total_revenue']:,.2f}")
        print(f"  • Documents Processed: {len(self.documents)}")
        print(f"  • Queries Tested: {len(getattr(self, 'test_results', []))}")
        print(f"  • Status: ✅ COMPLETED")
        
        print(f"\n📁 Generated Files:")
        print(f"  • business_intelligence_report.txt - BI analysis and insights")
        print(f"  • customer_analysis_report.txt - Customer segmentation and behavior")
        print(f"  • operational_insights_report.txt - Operations and efficiency analysis")
        print(f"  • menu_performance_report.txt - Menu item and category performance")
        print(f"  • execution_report.json - Complete execution log")
        
        print(f"\n🎯 Next Steps:")
        print(f"  1. Review generated reports for business insights")
        print(f"  2. Test additional queries using the interactive mode")
        print(f"  3. Implement recommendations from analysis reports")
        print(f"  4. Scale system for production deployment")
        
        logger.info("✅ Final report generated successfully")

# Test queries for different scenarios
SAMPLE_QUERIES = [
    # Business Intelligence Queries
    "What are the top 5 best-selling menu items by revenue?",
    "How do weekend sales compare to weekday sales?",
    "Which payment methods are most popular?",
    "What is the average order value by customer segment?",
    
    # Operational Queries
    "What are the peak hours for restaurant operations?",
    "How can we optimize staffing based on sales patterns?",
    "Which menu categories need operational improvements?",
    "What are the most efficient service patterns?",
    
    # Strategic Queries
    "What menu items should we promote or discontinue?",
    "How can we increase customer retention?",
    "What pricing strategies would maximize revenue?",
    "What are the growth opportunities in our data?",
    
    # Customer Experience Queries
    "Which customers are most valuable to our business?",
    "How do customer preferences vary by time and day?",
    "What are the patterns in customer ordering behavior?",
    "How can we personalize the dining experience?"
]

async def run_interactive_mode(workflow: AgenticRAGWorkflow):
    """Run interactive query mode for hotel food sales data"""
    print("\n" + "=" * 80)
    print("🔍 INTERACTIVE HOTEL FOOD SALES ANALYSIS MODE")
    print("=" * 80)
    print("Ask questions about the hotel food sales data and business operations.")
    print("Type 'quit' to exit, 'help' for sample queries, or 'stats' for data overview.")
    
    while True:
        print("\n" + "-" * 40)
        user_query = input("🔍 Enter your query: ").strip()
        
        if user_query.lower() == 'quit':
            print("👋 Goodbye!")
            break
        elif user_query.lower() == 'help':
            print("\n📋 Sample Business Intelligence Queries:")
            for i, query in enumerate(SAMPLE_QUERIES[:12], 1):
                print(f"  {i:2d}. {query}")
            continue
        elif user_query.lower() == 'stats':
            if workflow.sales_data is not None:
                print(f"\n📊 Data Overview:")
                print(f"  • Total Orders: {len(workflow.sales_data):,}")
                print(f"  • Total Revenue: ${workflow.sales_data['TotalAmount'].sum():,.2f}")
                print(f"  • Unique Customers: {workflow.sales_data['CustomerID'].nunique()}")
                print(f"  • Menu Items: {workflow.sales_data['MenuItem'].nunique()}")
                print(f"  • Date Range: {workflow.sales_data['OrderTime'].min().date()} to {workflow.sales_data['OrderTime'].max().date()}")
            continue
        elif not user_query:
            continue
        
        try:
            print(f"\n⏳ Processing your query...")
            start_time = time.time()
            
            response = await workflow.agentic_rag.process_query(user_query)
            processing_time = time.time() - start_time
            
            print(f"\n✅ Response (processed in {processing_time:.2f}s):")
            print(f"🎯 Confidence: {response['confidence']}")
            print(f"📊 Quality Score: {response['quality_assessment']['score']}/10")
            print(f"\n📝 Answer:")
            print(response['answer'])
            
            if response['insights']['key_points']:
                print(f"\n💡 Key Insights:")
                for insight in response['insights']['key_points'][:3]:
                    print(f"  • {insight}")
                    
            if response['insights']['patterns']:
                print(f"\n🔍 Patterns Identified:")
                for pattern in response['insights']['patterns'][:2]:
                    print(f"  • {pattern}")
                    
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")

async def main():
    """Main execution function"""
    print("🏨 Hotel Food Sales - Agentic RAG Analysis Pipeline")
    print("=" * 60)
    
    try:
        # Initialize workflow
        workflow = AgenticRAGWorkflow()
        
        # Run complete workflow
        success = workflow.run_complete_workflow(interactive=False)
        
        if success:
            # Ask user for interactive mode
            print("\n" + "=" * 80)
            user_choice = input("Would you like to enter interactive query mode for business analysis? (y/n): ").strip().lower()
            
            if user_choice == 'y':
                await run_interactive_mode(workflow)
            else:
                print("Pipeline execution completed. Check the data directory for comprehensive business reports.")
        else:
            print("❌ Workflow failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        print(f"❌ Fatal error: {e}")
        return 1
    
    return 0

# Main execution
if __name__ == "__main__":
    workflow = AgenticRAGWorkflow()
    success = workflow.run_complete_workflow(interactive=True)
    if not success:
        print("❌ Workflow failed!")
        sys.exit(1)
