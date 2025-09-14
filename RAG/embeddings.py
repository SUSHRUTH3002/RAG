import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

from .config import DEFAULT_EMBEDDING_MODEL


class EmbeddingManager:
    """Handles embedding generation using SentenceTransformer."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """Initialize the EmbeddingManager with a specific model."""
        self.model_name = model_name
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the SentenceTransformer model."""
        try:
            print(f"Loading Embedding Model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully. Embedding dimension is", 
                  self.model.get_sentence_embedding_dimension())
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        try:
            if not self.model:
                raise ValueError("Model is not loaded.")
            
            print(f"Generating embedding for {len(texts)} texts")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            print("Embedding generated with shape:", embeddings.shape)
            return embeddings
        except Exception as e:
            print(f"Error generating embedding for texts: {e}")
            return None
