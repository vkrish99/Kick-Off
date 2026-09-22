"""
Tests for RAGService.
All vector store and embedding calls are mocked.
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from app.services.rag import RAGService
from app.models.schemas import PlayerProfile
from app.core.config import get_settings

rag_module_settings = get_settings()


@pytest.fixture
def rag():
    return RAGService()


@pytest.fixture
def sample_profiles():
    return [
        PlayerProfile(
            player_id="haaland_9",
            name="Erling Haaland",
            position="Striker",
            team="Manchester City",
            skills=["finishing", "pace", "aerial ability"],
            recent_performance_summary="Scored 2 goals vs Arsenal.",
        ),
        PlayerProfile(
            player_id="debruyne_17",
            name="Kevin De Bruyne",
            position="Midfielder",
            team="Manchester City",
            skills=["passing", "vision", "set pieces"],
            recent_performance_summary="3 assists in last 2 matches.",
        ),
    ]


class TestRAGServiceIngest:
    @patch("app.services.rag.vector_store_service")
    async def test_ingest_profiles_returns_correct_count(self, mock_vs, rag, sample_profiles):
        mock_vs.initialize_index.return_value = None
        mock_vs.add_documents.return_value = None

        count = await rag.ingest_profiles(sample_profiles)

        assert count == 2

    @patch("app.services.rag.vector_store_service")
    async def test_ingest_profiles_calls_initialize_and_add(self, mock_vs, rag, sample_profiles):
        mock_vs.initialize_index.return_value = None
        mock_vs.add_documents.return_value = None

        await rag.ingest_profiles(sample_profiles)

        mock_vs.initialize_index.assert_called_once()
        mock_vs.add_documents.assert_called_once()

    @patch("app.services.rag.vector_store_service")
    async def test_ingest_profiles_document_contains_player_name(self, mock_vs, rag, sample_profiles):
        mock_vs.initialize_index.return_value = None
        mock_vs.add_documents.return_value = None

        await rag.ingest_profiles(sample_profiles)

        docs = mock_vs.add_documents.call_args[0][0]
        contents = [d.page_content for d in docs]
        assert any("Erling Haaland" in c for c in contents)
        assert any("Kevin De Bruyne" in c for c in contents)

    @patch("app.services.rag.vector_store_service")
    async def test_ingest_profiles_raises_on_pinecone_failure(self, mock_vs, rag, sample_profiles):
        mock_vs.initialize_index.side_effect = Exception("Pinecone connection refused")

        with pytest.raises(Exception, match="Pinecone connection refused"):
            await rag.ingest_profiles(sample_profiles)

    @patch("app.services.rag.vector_store_service")
    async def test_ingest_empty_profiles(self, mock_vs, rag):
        mock_vs.initialize_index.return_value = None
        mock_vs.add_documents.return_value = None

        count = await rag.ingest_profiles([])

        assert count == 0
        mock_vs.add_documents.assert_called_once_with([])


class TestRAGServiceRetrieve:
    @patch("app.services.rag.vector_store_service")
    async def test_retrieve_context_returns_documents(self, mock_vs, rag):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="Erling Haaland profile data", metadata={"player_id": "haaland_9"}),
        ]
        mock_vs.as_retriever.return_value = mock_retriever

        docs = await rag.retrieve_context("Who is the best striker?")

        assert len(docs) == 1
        assert docs[0].page_content == "Erling Haaland profile data"

    @patch("app.services.rag.vector_store_service")
    async def test_retrieve_context_calls_invoke_with_query(self, mock_vs, rag):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_vs.as_retriever.return_value = mock_retriever

        await rag.retrieve_context("Who should start in midfield?")

        mock_retriever.invoke.assert_called_once_with("Who should start in midfield?")

    @patch("app.services.rag.vector_store_service")
    async def test_retrieve_context_empty_result(self, mock_vs, rag):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_vs.as_retriever.return_value = mock_retriever

        docs = await rag.retrieve_context("unknown query xyz")

        assert docs == []

    @patch("app.services.rag.vector_store_service")
    async def test_retrieve_context_respects_k_parameter(self, mock_vs, rag):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_vs.as_retriever.return_value = mock_retriever

        await rag.retrieve_context("query", k=5)

        mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 5})

    @patch("app.services.rag.vector_store_service")
    async def test_retrieve_context_forwards_metadata_filter(self, mock_vs, rag):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_vs.as_retriever.return_value = mock_retriever

        await rag.retrieve_context("query", k=5, filters={"position": "Striker"})

        mock_vs.as_retriever.assert_called_once_with(
            search_kwargs={"k": 5, "filter": {"position": "Striker"}}
        )

    @patch("app.services.rag.vector_store_service")
    async def test_retrieve_context_defaults_k_from_settings(self, mock_vs, rag):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_vs.as_retriever.return_value = mock_retriever

        await rag.retrieve_context("query")

        mock_vs.as_retriever.assert_called_once_with(
            search_kwargs={"k": rag_module_settings.RAG_RETRIEVAL_K}
        )
