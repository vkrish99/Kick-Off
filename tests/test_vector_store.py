"""
Tests for VectorStoreService / PineconeRetriever.
All Pinecone and embedding calls are mocked.
"""
from unittest.mock import MagicMock
from app.services.vector_store import VectorStoreService
from app.core.config import get_settings

settings = get_settings()


def make_service_with_mocks(query_return=None):
    svc = VectorStoreService()
    svc._embed = MagicMock(return_value=[0.1, 0.2, 0.3])
    mock_index = MagicMock()
    mock_index.query.return_value = query_return or {"matches": []}
    svc._get_index = MagicMock(return_value=mock_index)
    return svc, mock_index


class TestAsRetriever:
    def test_defaults_k_to_settings_value(self):
        svc, _ = make_service_with_mocks()
        retriever = svc.as_retriever()
        assert retriever.k == settings.RAG_RETRIEVAL_K
        assert retriever.filter is None

    def test_explicit_k_overrides_default(self):
        svc, _ = make_service_with_mocks()
        retriever = svc.as_retriever(search_kwargs={"k": 7})
        assert retriever.k == 7

    def test_filter_is_forwarded(self):
        svc, _ = make_service_with_mocks()
        retriever = svc.as_retriever(search_kwargs={"filter": {"position": "Striker"}})
        assert retriever.filter == {"position": "Striker"}


class TestPineconeRetrieverQuery:
    def test_passes_k_and_filter_to_pinecone_query(self):
        svc, mock_index = make_service_with_mocks()
        retriever = svc.as_retriever(search_kwargs={"k": 5, "filter": {"team": "Manchester City"}})

        retriever.invoke("Who plays for City?")

        mock_index.query.assert_called_once_with(
            vector=[0.1, 0.2, 0.3],
            top_k=5,
            include_metadata=True,
            filter={"team": "Manchester City"},
        )

    def test_maps_matches_to_documents(self):
        svc, _ = make_service_with_mocks(
            query_return={
                "matches": [
                    {
                        "id": "haaland_9",
                        "metadata": {"content": "Erling Haaland profile", "team": "Manchester City"},
                    }
                ]
            }
        )
        retriever = svc.as_retriever()

        docs = retriever.invoke("Tell me about Haaland")

        assert len(docs) == 1
        assert docs[0].page_content == "Erling Haaland profile"
        assert docs[0].metadata["team"] == "Manchester City"
