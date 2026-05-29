from typing import List
from app.services.vector_store import vector_store_service
from app.models.schemas import PlayerProfile
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        pass

    async def ingest_profiles(self, profiles: List[PlayerProfile]) -> int:
        """Embed player profiles and upsert them into Pinecone."""
        # Ensure the Pinecone index exists
        vector_store_service.initialize_index()

        documents = []
        for p in profiles:
            content = (
                f"Player: {p.name}, Position: {p.position}, Team: {p.team}. "
                f"Skills: {', '.join(p.skills)}. "
                f"Summary: {p.recent_performance_summary or 'N/A'}"
            )
            metadata = p.model_dump()
            documents.append(Document(page_content=content, metadata=metadata))

        vector_store_service.add_documents(documents)
        logger.info("RAG ingested %d player profiles.", len(documents))
        return len(documents)

    async def retrieve_context(self, query: str, k: int = 3) -> list[Document]:
        """Retrieve relevant player contexts based on query."""
        retriever = vector_store_service.as_retriever()
        docs = retriever.invoke(query)
        return docs


rag_service = RAGService()

