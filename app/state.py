from typing import TypedDict, List, Annotated, Optional
import operator


class AgentState(TypedDict):
    """
    Represents the state of our financial agent. This state is passed
    between nodes in the LangGraph.
    """

    # Input data from the user
    input_data: dict

    # The type of request determined by the triage agent
    request_type: str  # 'fraud_check', 'credit_risk', or 'general_query'

    # Conversational history and agent scratchpad
    messages: Annotated[list, operator.add]

    # Data retrieved from tools
    rag_context: Optional[str]
    ml_tool_output: Optional[str]  # Storing as JSON string

    # Final outputs for the user
    final_summary: Optional[str]
    mitigation_steps: Optional[List[str]]

    # System state for error handling
    error_log: Optional[str]
