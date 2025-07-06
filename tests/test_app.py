"""Tests FastAPI endpoints."""

from http import HTTPStatus
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document as LangChainDocument

from financeqa.app.main import app
from financeqa.app.schema import ReferencedDoc, ReferencedResponse
from financeqa.app.security import get_current_api_key

app.dependency_overrides = {get_current_api_key: lambda: True}


def test_health() -> None:
    """Tests /ready endpoint."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == HTTPStatus.OK


def test_query_endpoint_with_context_documents():
    """Tests the /query endpoint with mocked context documents and completions."""

    with patch(
        "financeqa.app.routers.chat.chat.get_context_documents", new_callable=AsyncMock
    ) as mock_get_context_documents, patch(
        "financeqa.app.routers.chat.chat.HFChatCompletion.get_completions",
        new_callable=AsyncMock,
    ) as mock_get_completions:
        mock_get_context_documents.return_value = [
            LangChainDocument(
                page_content="Mocked document 1",
                metadata={"db_document_name": "doc1", "page_number": 1},
            ),
            LangChainDocument(
                page_content="Mocked document 2",
                metadata={"db_document_name": "doc1", "page_number": 2},
            ),
        ]

        mock_get_completions.return_value = ReferencedResponse(
            response="Mocked response",
            references=[ReferencedDoc(year=2023, quarter="Q2", company="MSFT", page=2)],
        )

        with TestClient(app) as client:
            payload = [{"message": "Test message"}]
            response = client.post("/query", json=payload)

            # assert expected mocked response
            assert response.status_code == 200
            assert response.json() == {
                "response": "Mocked response",
                "references": [{"year": 2023, "quarter": "Q2", "company": "MSFT", "page": 2}],
            }

            # assert mocks were called
            mock_get_context_documents.assert_called_once_with("Test message")
            mock_get_completions.assert_called_once()
