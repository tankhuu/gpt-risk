import pytest
from unittest.mock import patch, AsyncMock
import json

# Sample data for testing
FRAUD_TRANSACTION_PAYLOAD = {
    "transaction_id": "t12345",
    "customer_id": "c67890",
    "transaction_amount": 2500.75,
    "merchant_id": "m54321",
    "timestamp": "2025-07-22T08:30:00Z",
}

CREDIT_APPLICATION_PAYLOAD = {
    "application_id": "a98765",
    "customer_id": "c11223",
    "loan_amount": 50000,
    "annual_income": 85000,
    "employment_length_years": 5,
    "dti_ratio": 0.3,
}


@pytest.mark.anyio
async def test_chat_endpoint_fraud_check(test_client):
    """
    Integration test for the /chat endpoint with a fraud check request.
    Mocks the LangGraph gpt_risk to isolate the API layer.
    """
    # Mock the response stream from the langgraph gpt_risk
    mock_stream_response = [
        {"request_type": "fraud_check"},
        {"ml_tool_output": '{"fraud_probability": 0.95}'},
        {"rag_context": "Context about previous transactions."},
        {"final_summary": "High risk of fraud detected."},
    ]

    # Use patch with AsyncMock for the async generator
    with patch(
        "gpt_risk.main.langgraph_app.astream", new_callable=AsyncMock
    ) as mock_astream:
        # Configure the mock to be an async iterator
        async def async_gen():
            for item in mock_stream_response:
                yield item

        mock_astream.return_value = async_gen()

        # Make the API call
        response = await test_client.post(
            "/chat",
            json={"query": FRAUD_TRANSACTION_PAYLOAD, "thread_id": "test_fraud_123"},
        )

        # Assertions
        assert response.status_code == 200
        assert response.json() == {"response": "High risk of fraud detected."}
        mock_astream.assert_called_once()


@pytest.mark.anyio
async def test_chat_endpoint_credit_risk(test_client):
    """
    Integration test for the /chat endpoint with a credit risk request.
    """
    mock_stream_response = [
        {"request_type": "credit_risk"},
        {"ml_tool_output": '{"default_probability": 0.12}'},
        {"rag_context": "Context about applicant's credit history."},
        {"final_summary": "Low risk applicant."},
    ]

    with patch(
        "gpt_risk.main.langgraph_app.astream", new_callable=AsyncMock
    ) as mock_astream:

        async def async_gen():
            for item in mock_stream_response:
                yield item

        mock_astream.return_value = async_gen()

        response = await test_client.post(
            "/chat",
            json={"query": CREDIT_APPLICATION_PAYLOAD, "thread_id": "test_credit_456"},
        )

        assert response.status_code == 200
        assert response.json() == {"response": "Low risk applicant."}
        mock_astream.assert_called_once()
