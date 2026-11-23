from __future__ import annotations

import logging
import threading
import uuid
from typing import List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import ConfigDict
from pinecone import Pinecone, ServerlessSpec
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Embedding model name and dimension are now driven from settings
# (EMBEDDING_MODEL_NAME and EMBEDDING_DIM in app/core/config.py).
# Changing the model there is enough — no code change required here.


# ---------------------------------------------------------------------------
# Lightweight retriever compatible with LangChain's BaseRetriever interface
# ---------------------------------------------------------------------------
class PineconeRetriever(BaseRetriever):
    """Thin LangChain-compatible retriever wrapping the Pinecone SDK."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    vector_store_service: object  # VectorStoreService instance
    k: int = 3
    filter: Optional[dict] = None  # Pinecone metadata filter, e.g. {"position": "Striker"}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        svc = self.vector_store_service  # type: VectorStoreService
        embedding = svc._embed(query)
        index = svc._get_index()
        results = index.query(
            vector=embedding,
            top_k=self.k,
            include_metadata=True,
            filter=self.filter,
        )
        docs = []
        for match in results.get("matches", []):
            metadata = dict(match.get("metadata") or {})
            content = metadata.get("content", "")
            if match.get("score") is not None:
                metadata["score"] = match.get("score")
            docs.append(Document(page_content=content, metadata=metadata))
        return docs


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------
class VectorStoreService:
    def __init__(self):
        self._pc: Optional[Pinecone] = None
        self._embedder = None
        self._index = None
        self._index_ready = False
        self._lock = threading.Lock()  # these methods run in worker threads

    # ------------------------------------------------------------------
    # Lazy helpers
    # ------------------------------------------------------------------
    def _get_pinecone(self) -> Pinecone:
        if not settings.PINECONE_API_KEY:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail="PINECONE_API_KEY is not configured. Set it in .env.",
            )
        if self._pc is None:
            self._pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        return self._pc

    def _get_embedder(self):
        """Lazy-load the fastembed (ONNX) model. Runs locally, no torch needed.

        The model is downloaded to FASTEMBED_CACHE_PATH on first use; the Docker
        image downloads it at build time.
        """
        with self._lock:
            if self._embedder is None:
                from fastembed import TextEmbedding
                logger.info("Loading embedding model: %s ...", settings.EMBEDDING_MODEL_NAME)
                self._embedder = TextEmbedding(settings.EMBEDDING_MODEL_NAME)
                logger.info("Embedding model loaded.")
        return self._embedder

    def _get_index(self):
        if self._index is None:
            pc = self._get_pinecone()
            self._index = pc.Index(settings.PINECONE_INDEX_NAME)
        return self._index

    def _embed(self, text: str) -> List[float]:
        """Embed one text (CPU-bound: call it from a worker thread in async code)."""
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed, L2-normalized to match the cosine index."""
        model = self._get_embedder()
        vectors = []
        for e in model.embed(texts, batch_size=32):
            norm = float(np.linalg.norm(e)) or 1.0
            vectors.append((e / norm).tolist())
        return vectors

    def warm_up(self) -> None:
        """Load the embedding model ahead of the first request."""
        self._embed("warm-up")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initialize_index(self):
        """Ensure the Pinecone index exists. Checks Pinecone only once per process."""
        if self._index_ready:
            return
        pc = self._get_pinecone()
        existing = [idx.name for idx in pc.list_indexes()]
        if settings.PINECONE_INDEX_NAME not in existing:
            logger.info(
                "Creating Pinecone index: %s (dim=%d, metric=%s)",
                settings.PINECONE_INDEX_NAME,
                settings.EMBEDDING_DIM,
                settings.PINECONE_METRIC,
            )
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=settings.EMBEDDING_DIM,
                metric=settings.PINECONE_METRIC,
                spec=ServerlessSpec(
                    cloud=settings.PINECONE_CLOUD,
                    region=settings.PINECONE_REGION,
                ),
            )
        else:
            logger.info("Pinecone index '%s' already exists.", settings.PINECONE_INDEX_NAME)
        self._index_ready = True

    def add_documents(self, documents: List[Document]) -> None:
        """Embed and upsert documents into Pinecone."""
        index = self._get_index()

        # Batch embed all documents at once for efficiency
        texts = [doc.page_content for doc in documents]
        embeddings = self._embed_batch(texts)

        vectors = []
        for doc, embedding in zip(documents, embeddings):
            meta = {**doc.metadata, "content": doc.page_content}
            # player_id gives a stable ID, so re-ingesting a player updates it in place.
            # Without one, use a random ID so documents never overwrite each other.
            vec_id = doc.metadata.get("player_id") or uuid.uuid4().hex
            vectors.append({"id": str(vec_id), "values": embedding, "metadata": meta})

        if vectors:
            for start in range(0, len(vectors), settings.PINECONE_BATCH_SIZE):
                batch = vectors[start : start + settings.PINECONE_BATCH_SIZE]
                index.upsert(vectors=batch)
            logger.info("Upserted %d vectors to Pinecone.", len(vectors))

    def as_retriever(self, search_kwargs: dict = None) -> PineconeRetriever:
        search_kwargs = search_kwargs or {}
        return PineconeRetriever(
            vector_store_service=self,
            k=search_kwargs.get("k", settings.RAG_RETRIEVAL_K),
            filter=search_kwargs.get("filter"),
        )


vector_store_service = VectorStoreService()
