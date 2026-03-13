"""Vector-based knowledge base service using FAISS and OpenAI embeddings."""

import json
import os
from typing import List, Optional, Dict, Any

import faiss
import numpy as np
from openai import OpenAI
from loguru import logger

from src.core.config import settings
from src.services.base import BaseService


class VectorKBService(BaseService):
    """Vector-based knowledge base using FAISS for similarity search with OpenAI embeddings."""

    # OpenAI embedding model dimensions
    EMBEDDING_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(self):
        """Initialize vector KB service with OpenAI embeddings."""
        self.embedding_model = settings.OPENAI_EMBEDDING_MODEL
        self.client = None
        self.faiss_index = None
        self.documents: Dict[int, Dict[str, Any]] = {}
        self.doc_counter = 0
        self.vector_store_path = settings.VECTOR_STORE_PATH
        self.index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
        self.metadata_path = os.path.join(self.vector_store_path, "documents.json")

        # Create directory if it doesn't exist
        os.makedirs(self.vector_store_path, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize the vector KB service with OpenAI client."""
        await super().initialize()
        try:
            logger.info(f"Initializing OpenAI embeddings with model: {self.embedding_model}")
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

            # Get embedding dimension from model
            embedding_dim = self.EMBEDDING_DIMENSIONS.get(self.embedding_model, 1536)

            # Load existing index if available
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                await self._load_index()
                logger.info(f"Loaded existing FAISS index (dimension: {embedding_dim})")
            else:
                # Create new empty index
                self.faiss_index = faiss.IndexFlatL2(embedding_dim)
                logger.info(f"Created new FAISS index (dimension: {embedding_dim})")

        except Exception as e:
            logger.error(f"Failed to initialize vector KB: {e}")
            raise

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using OpenAI API.

        Args:
            text: Text to embed

        Returns:
            Embedding as numpy array
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            embedding = response.data[0].embedding
            return np.array([embedding], dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise

    async def _load_index(self) -> None:
        """Load FAISS index and metadata from disk."""
        try:
            self.faiss_index = faiss.read_index(self.index_path)

            with open(self.metadata_path, "r") as f:
                data = json.load(f)
                self.documents = {int(k): v for k, v in data.items()}
                self.doc_counter = len(self.documents)

            logger.info(f"Loaded {self.doc_counter} documents from disk")

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

    async def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            faiss.write_index(self.faiss_index, self.index_path)

            with open(self.metadata_path, "w") as f:
                json.dump(self.documents, f, indent=2)

            logger.debug("Saved FAISS index to disk")

        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise

    async def add_document(
        self,
        title: str,
        content: str,
        category: Optional[str] = None,
        source_url: Optional[str] = None,
    ) -> int:
        """Add document to knowledge base with OpenAI embeddings.

        Args:
            title: Document title
            content: Document content
            category: Optional category
            source_url: Optional source URL

        Returns:
            Document ID
        """
        try:
            # Generate embedding using OpenAI
            embedding = await self._get_embedding(content)

            # Add to FAISS index
            self.faiss_index.add(embedding)

            # Store metadata
            self.doc_counter += 1
            doc_id = self.doc_counter

            self.documents[doc_id] = {
                "id": doc_id,
                "title": title,
                "content": content,
                "category": category,
                "source_url": source_url,
            }

            # Save to disk
            await self._save_index()

            logger.info(f"Added document {doc_id}: {title}")
            return doc_id

        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise

    async def search(
        self, query: str, category: Optional[str] = None, limit: int = 5, threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search knowledge base by semantic similarity using OpenAI embeddings.

        Args:
            query: Search query
            category: Optional category filter
            limit: Max results to return
            threshold: Similarity threshold (0-1)

        Returns:
            List of similar documents with scores
        """
        try:
            if self.faiss_index is None or self.doc_counter == 0:
                logger.debug("Knowledge base is empty")
                return []

            # Generate query embedding using OpenAI
            query_embedding = await self._get_embedding(query)

            # Search in FAISS index
            distances, indices = self.faiss_index.search(query_embedding, k=min(limit * 2, self.doc_counter))

            results = []
            for distance, idx in zip(distances[0], indices):
                if idx == -1:  # Invalid index
                    continue

                doc_id = idx + 1  # Convert from 0-indexed to 1-indexed
                if doc_id not in self.documents:
                    continue

                doc = self.documents[doc_id]

                # Convert L2 distance to similarity (0-1)
                similarity = 1 / (1 + distance)

                if similarity < threshold:
                    continue

                if category and doc.get("category") != category:
                    continue

                results.append({
                    "id": doc_id,
                    "title": doc["title"],
                    "content": doc["content"],
                    "category": doc.get("category"),
                    "source_url": doc.get("source_url"),
                    "similarity_score": float(similarity),
                })

            results = results[:limit]
            logger.debug(f"Found {len(results)} documents for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document dict or None
        """
        try:
            if doc_id not in self.documents:
                logger.warning(f"Document {doc_id} not found")
                return None

            return self.documents[doc_id]

        except Exception as e:
            logger.error(f"Failed to get document: {e}")
            return None

    async def get_by_category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get documents by category.

        Args:
            category: Category name
            limit: Max documents

        Returns:
            List of documents
        """
        try:
            results = [
                doc for doc in self.documents.values()
                if doc.get("category") == category
            ][:limit]

            logger.debug(f"Retrieved {len(results)} documents from category: {category}")
            return results

        except Exception as e:
            logger.error(f"Failed to get documents by category: {e}")
            return []

    async def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents into context string for LLM.

        Args:
            documents: List of document dicts

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documentation found."

        context_parts = ["Relevant documentation:"]
        for i, doc in enumerate(documents, 1):
            similarity = doc.get("similarity_score", 0)
            context_parts.append(f"\n{i}. **{doc['title']}** (Relevance: {similarity:.1%})")
            context_parts.append(f"   {doc['content'][:300]}...")

        return "\n".join(context_parts)

    async def delete_document(self, doc_id: int) -> bool:
        """Delete document from knowledge base.

        Note: FAISS doesn't support deletion, so we mark as deleted.

        Args:
            doc_id: Document ID

        Returns:
            True if deleted
        """
        try:
            if doc_id not in self.documents:
                logger.warning(f"Document {doc_id} not found")
                return False

            del self.documents[doc_id]
            await self._save_index()

            logger.info(f"Deleted document {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False

    async def clear_all(self) -> bool:
        """Clear all documents and reset index.

        Returns:
            True if successful
        """
        try:
            embedding_dim = self.EMBEDDING_DIMENSIONS.get(self.embedding_model, 1536)
            self.faiss_index = faiss.IndexFlatL2(embedding_dim)
            self.documents = {}
            self.doc_counter = 0
            await self._save_index()

            logger.info("Cleared all documents from knowledge base")
            return True

        except Exception as e:
            logger.error(f"Failed to clear knowledge base: {e}")
            return False

    async def health_check(self) -> dict:
        """Check KB service health."""
        try:
            return {
                "status": "healthy",
                "service": "vector_kb",
                "documents_count": len(self.documents),
                "embedding_model": self.embedding_model,
                "index_dimension": self.faiss_index.d if self.faiss_index else 0,
            }
        except Exception as e:
            logger.error(f"KB service health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": "vector_kb",
                "error": str(e),
            }
