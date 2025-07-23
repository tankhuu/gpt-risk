import pytest
from unittest.mock import patch, MagicMock
from src.app.state import AgentState
from src.app.graph import (
    triage_node,
    fraud_agent_node,
    credit_agent_node,
    synthesis_node,
    route_request,
)


# --- Test Triage Node and Routing ---


@patch("gpt_risk.graph.triage_llm")
def test_triage_node(mock_triage_llm):
    """Unit test for the triage_node."""
    # Mock the LLM's response
    mock_triage_llm.invoke.return_value = MagicMock(content="fraud_check")

    initial_state = AgentState(input_data={"transaction_id": "123"}, messages="")

    result_state = triage_node(initial_state)

    assert result_state["request_type"] == "fraud_check"
    mock_triage_llm.invoke.assert_called_once()


def test_route_request():
    """Unit test for the conditional routing logic."""
    fraud_state = AgentState(request_type="fraud_check")
    credit_state = AgentState(request_type="credit_risk")
    general_state = AgentState(request_type="general_query")

    assert route_request(fraud_state) == "fraud_agent"
    assert route_request(credit_state) == "credit_agent"
    assert route_request(general_state) == "synthesis"


# --- Test Agent Nodes ---


@patch("gpt_risk.graph.query_databricks_vector_search")
@patch("gpt_risk.graph.run_fraud_detection_model")
@patch("gpt_risk.graph.fraud_agent_llm")
def test_fraud_agent_node(mock_llm, mock_fraud_tool, mock_rag_tool):
    """Unit test for the fraud_agent_node."""
    # Mock the LLM to return tool calls
    mock_llm.invoke.return_value = MagicMock(
        tool_calls=[
            {"name": "run_fraud_detection_model", "args": {"transaction_data": {}}},
            {"name": "query_databricks_vector_search", "args": {"query": "history"}},
        ]
    )

    # Mock the tool outputs
    mock_fraud_tool.invoke.return_value = '{"fraud_probability": 0.98}'
    mock_rag_tool.invoke.return_value = '[{"content": "some context"}]'

    initial_state = AgentState(input_data={"amount": 5000}, messages="")

    result_state = fraud_agent_node(initial_state)

    mock_llm.invoke.assert_called_once()
    mock_fraud_tool.invoke.assert_called_once()
    mock_rag_tool.invoke.assert_called_once()

    assert result_state["ml_tool_output"] == '{"fraud_probability": 0.98}'
    assert result_state["rag_context"] == '[{"content": "some context"}]'


@patch("gpt_risk.graph.query_databricks_vector_search")
@patch("gpt_risk.graph.run_credit_risk_model")
@patch("gpt_risk.graph.credit_agent_llm")
def test_credit_agent_node(mock_llm, mock_credit_tool, mock_rag_tool):
    """Unit test for the credit_agent_node."""
    mock_llm.invoke.return_value = MagicMock(
        tool_calls=[
            {"name": "run_credit_risk_model", "args": {"loan_application_data": {}}},
            {"name": "query_databricks_vector_search", "args": {"query": "history"}},
        ]
    )

    mock_credit_tool.invoke.return_value = '{"default_probability": 0.05}'
    mock_rag_tool.invoke.return_value = '[{"content": "credit history"}]'

    initial_state = AgentState(input_data={"loan_amount": 10000}, messages="")

    result_state = credit_agent_node(initial_state)

    mock_llm.invoke.assert_called_once()
    mock_credit_tool.invoke.assert_called_once()
    mock_rag_tool.invoke.assert_called_once()

    assert result_state["ml_tool_output"] == '{"default_probability": 0.05}'
    assert result_state["rag_context"] == '[{"content": "credit history"}]'


@patch("gpt_risk.graph.synthesis_llm")
def test_synthesis_node(mock_synthesis_llm):
    """Unit test for the synthesis_node."""
    mock_synthesis_llm.invoke.return_value = MagicMock(
        content="Final detailed summary."
    )

    # A state after an agent node has run
    input_state = AgentState(
        input_data={"amount": 5000},
        request_type="fraud_check",
        ml_tool_output='{"fraud_probability": 0.98}',
        rag_context='[{"content": "some context"}]',
        messages="",
    )

    result_state = synthesis_node(input_state)

    mock_synthesis_llm.invoke.assert_called_once()
    assert result_state["final_summary"] == "Final detailed summary."
