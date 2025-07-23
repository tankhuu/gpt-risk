import pytest
from unittest.mock import patch, MagicMock
import json

# Import the tool to be tested
from src.app.tools.databricks_rag import query_databricks_vector_search


class MockDocument:
    """A mock class for LangChain's Document."""

    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


@patch("gpt_risk.tools.databricks_rag.DatabricksVectorSearch")
@patch("gpt_risk.tools.databricks_rag.DatabricksEmbeddings")
def test_query_databricks_vector_search_success(mock_embeddings, mock_dvs_client):
    """
    Unit test for the Databricks RAG tool on a successful run.
    Mocks the Databricks client and retriever.
    """
    # Setup mock retriever and its return value
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []

    # The DatabricksVectorSearch instance will have the as_retriever method
    mock_dvs_instance = MagicMock()
    mock_dvs_instance.as_retriever.return_value = mock_retriever
    mock_dvs_client.return_value = mock_dvs_instance

    # Call the tool
    result_str = query_databricks_vector_search.invoke(
        {"query": "customer history", "index_name": "test_index"}
    )
    result = json.loads(result_str)

    # Assertions
    mock_dvs_client.assert_called_once()
    mock_retriever.invoke.assert_called_with("customer history")
    assert isinstance(result, list)
    assert len(result) == 2
    assert result["content"] == "Doc 1 content"
    assert result[1]["source"] == "doc2.pdf"


@patch(
    "gpt_risk.tools.databricks_rag.DatabricksVectorSearch",
    side_effect=Exception("Connection failed"),
)
@patch("gpt_risk.tools.databricks_rag.DatabricksEmbeddings")
def test_query_databricks_vector_search_failure(mock_embeddings, mock_dvs_client):
    """
    Unit test for the Databricks RAG tool when an exception occurs.
    """
    result_str = query_databricks_vector_search.invoke(
        {"query": "customer history", "index_name": "test_index"}
    )
    result = json.loads(result_str)

    assert "error" in result
    assert "Connection failed" in result["error"]
