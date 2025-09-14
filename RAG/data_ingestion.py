from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader, DirectoryLoader, PyMuPDFLoader, 
    CSVLoader, UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Any
import os

from .config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DATA_DIR


class DocumentLoader:
    """Handles loading documents from various file formats."""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or str(DATA_DIR)
    
    def load_text_files(self, glob_pattern: str = "**/*.txt") -> List[Document]:
        """Load text files from directory."""
        loader = DirectoryLoader(
            self.data_dir,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf8"},
            show_progress=False
        )
        return loader.load()
    
    def load_pdf_files(self, glob_pattern: str = "**/*.pdf") -> List[Document]:
        """Load PDF files from directory."""
        loader = DirectoryLoader(
            self.data_dir,
            glob=glob_pattern,
            loader_cls=PyMuPDFLoader,
            show_progress=False
        )
        return loader.load()
    
    def load_csv_files(self, glob_pattern: str = "**/*.csv") -> List[Document]:
        """Load CSV files from directory."""
        loader = DirectoryLoader(
            self.data_dir,
            glob=glob_pattern,
            loader_cls=CSVLoader,
            show_progress=False
        )
        return loader.load()
    
    def load_excel_files(self, glob_pattern: str = "**/*.xlsx") -> List[Document]:
        """Load Excel files from directory."""
        loader = DirectoryLoader(
            self.data_dir,
            glob=glob_pattern,
            loader_cls=UnstructuredExcelLoader,
            show_progress=False
        )
        return loader.load()
    
    def load_all_documents(self) -> List[Document]:
        """Load all supported document types."""
        all_docs = []
        all_docs.extend(self.load_text_files())
        all_docs.extend(self.load_pdf_files())
        all_docs.extend(self.load_csv_files())
        all_docs.extend(self.load_excel_files())
        return all_docs


def split_documents(documents: List[Document], 
                   chunk_size: int = DEFAULT_CHUNK_SIZE, 
                   chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Document]:
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    if split_docs:
        print("\nExample chunk:")
        print("Content: ", split_docs[0].page_content[:500])
        print("Metadata: ", split_docs[0].metadata)

    return split_docs
