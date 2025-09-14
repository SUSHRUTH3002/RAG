import numpy as np
import chromadb
import uuid
import os
from typing import List, Any

from .config import DEFAULT_COLLECTION_NAME, VECTOR_STORE_DIR


class VectorStore:
    """Manages a vector store for embeddings using ChromaDB."""
    
    def __init__(self, collection_name: str = DEFAULT_COLLECTION_NAME, 
                 persist_directory: str = None):
        """Initialize the VectorStore with ChromaDB client and collection."""
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(VECTOR_STORE_DIR)
        self.client = None
        self.collection = None
        self.__initialize_store()

    def __initialize_store(self):
        """Initialize the ChromaDB client and collection."""
        try:
            print("Initializing ChromaDB client and collection...")

            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Collection for storing document embeddings"}
            )
            print(f"ChromaDB client and collection '{self.collection_name}' initialized.")
            print(f"Existing documents: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing ChromaDB client or collection: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Add documents and their embeddings to the vector store."""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings.")
        
        print(f"Adding {len(documents)} documents to the vector store...")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['context_length'] = len(doc.page_content) 
            metadatas.append(metadata)

            documents_text.append(doc.page_content if hasattr(doc, 'page_content') else str(doc))
            embeddings_list.append(emb.tolist())
        
        try:
            self.collection.add(
                ids=ids,
                documents=documents_text,
                embeddings=embeddings_list,
                metadatas=metadatas
            )
            print(f"Successfully added {len(documents)} documents to the collection.")
            print(f"Total documents in collection: {self.collection.count()}")
        
        except Exception as e:
            print(f"Error adding documents to the collection: {e}")
            raise

    def add_documents_batch(self, documents: List[Any], embedding_manager, batch_size: int = 100):
        """Add documents in batches."""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_embeddings = embedding_manager.generate_embeddings(
                [doc.page_content for doc in batch]
            )
            self.add_documents(batch, batch_embeddings)
