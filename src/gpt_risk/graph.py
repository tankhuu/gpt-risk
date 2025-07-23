from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, ToolMessage
from src.app.state import AgentState
from src.app.llms import get_gemini_llm, get_qwen_llm, get_llama_llm
from src.app.tools.ml_models import run_fraud_detection_model, run_credit_risk_model
from src.app.tools.databricks_rag import query_databricks_vector_search
from src.app.config import settings
import json

# --- Initialize Models and Tools ---
triage_llm = get_qwen_llm()
synthesis_llm = get_gemini_llm()
fraud_agent_llm = get_gemini_llm().bind_tools(
    [run_fraud_detection_model, query_databricks_vector_search]
)
credit_agent_llm = get_llama_llm().bind_tools(
    [run_credit_risk_model, query_databricks_vector_search]
)


# --- Agent Nodes ---


def triage_node(state: AgentState) -> dict:
    """
    Determines the type of request (fraud, credit, or general) based on the input.
    """
    prompt = f"""
    You are a request triage expert for a financial institution.
    Analyze the user's input and classify it into one of three categories: 'fraud_check', 'credit_risk', or 'general_query'.
    - 'fraud_check' is for analyzing a single financial transaction for fraud. Keywords: transaction, merchant, amount.
    - 'credit_risk' is for assessing a loan application for default risk. Keywords: loan, income, dti, application.
    - 'general_query' is for all other questions.

    User Input:
    {json.dumps(state["input_data"])}

    Classification:
    """
    response = triage_llm.invoke(prompt)
    request_type = response.content.strip().lower()

    # Basic validation
    if request_type not in ["fraud_check", "credit_risk", "general_query"]:
        request_type = "general_query"

    return {"request_type": request_type, "messages": ""}


def fraud_agent_node(state: AgentState) -> dict:
    """
    Handles fraud detection tasks by calling ML and RAG tools.
    """
    input_str = json.dumps(state["input_data"])
    prompt = f"Analyze the following transaction for fraud. First, use the `run_fraud_detection_model` tool. Then, use the `query_databricks_vector_search` tool with the index '{settings.FRAUD_RAG_INDEX_NAME}' to find related historical patterns for the customer or merchant. Transaction: {input_str}"

    response = fraud_agent_llm.invoke(prompt)

    ml_output = ""
    rag_output = ""

    for tool_call in response.tool_calls:
        if tool_call["name"] == "run_fraud_detection_model":
            ml_output = run_fraud_detection_model.invoke(tool_call["args"])
        elif tool_call["name"] == "query_databricks_vector_search":
            # Ensure the index name is passed correctly
            tool_call["args"]["index_name"] = settings.FRAUD_RAG_INDEX_NAME
            rag_output = query_databricks_vector_search.invoke(tool_call["args"])

    return {"ml_tool_output": ml_output, "rag_context": rag_output}


def credit_agent_node(state: AgentState) -> dict:
    """
    Handles credit risk assessment tasks.
    """
    input_str = json.dumps(state["input_data"])
    prompt = f"Assess the credit risk for the following loan application. First, use the `run_credit_risk_model` tool. Then, use the `query_databricks_vector_search` tool with the index '{settings.CREDIT_RAG_INDEX_NAME}' to get the applicant's financial history. Application: {input_str}"

    response = credit_agent_llm.invoke(prompt)

    ml_output = ""
    rag_output = ""

    for tool_call in response.tool_calls:
        if tool_call["name"] == "run_credit_risk_model":
            ml_output = run_credit_risk_model.invoke(tool_call["args"])
        elif tool_call["name"] == "query_databricks_vector_search":
            tool_call["args"]["index_name"] = settings.CREDIT_RAG_INDEX_NAME
            rag_output = query_databricks_vector_search.invoke(tool_call["args"])

    return {"ml_tool_output": ml_output, "rag_context": rag_output}


def synthesis_node(state: AgentState) -> dict:
    """
    Synthesizes all gathered information into a final report for the user.
    """
    prompt = f"""
    You are a senior financial risk analyst. Your task is to create a concise, clear, and actionable summary based on the provided data.

    Original User Request:
    {json.dumps(state["input_data"])}

    Request Type: {state["request_type"]}

    Machine Learning Model Output:
    {state["ml_tool_output"]}

    Retrieved Context from Knowledge Base:
    {state["rag_context"]}

    Based on all the information above, provide a final summary. The summary should be in Markdown format and include:
    1.  **Overall Assessment:** A clear, one-sentence conclusion (e.g., "High risk of fraud detected.").
    2.  **Key Evidence:** 2-3 bullet points summarizing the evidence from the ML model and retrieved context that supports your assessment.
    3.  **Recommended Actions:** A list of 2-3 concrete, actionable steps for the banker to take next.
    """
    response = synthesis_llm.invoke(prompt)
    return {"final_summary": response.content}


# --- Conditional Routing ---


def route_request(state: AgentState) -> str:
    """
    Routes the workflow based on the request type determined by the triage node.
    """
    if state["request_type"] == "fraud_check":
        return "fraud_agent"
    elif state["request_type"] == "credit_risk":
        return "credit_agent"
    else:
        # For general queries, we can route directly to a synthesis node
        # that uses a general-purpose RAG tool (not implemented here for brevity)
        # or just end. For now, we'll route to synthesis for a simple response.
        return "synthesis"


# --- Build the Graph ---

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("triage", triage_node)
workflow.add_node("fraud_agent", fraud_agent_node)
workflow.add_node("credit_agent", credit_agent_node)
workflow.add_node("synthesis", synthesis_node)

# Define the workflow edges
workflow.set_entry_point("triage")

workflow.add_conditional_edges(
    "triage",
    route_request,
    {
        "fraud_agent": "fraud_agent",
        "credit_agent": "credit_agent",
        "synthesis": "synthesis",  # Route general queries directly to synthesis
    },
)

workflow.add_edge("fraud_agent", "synthesis")
workflow.add_edge("credit_agent", "synthesis")
workflow.add_edge("synthesis", END)

# Compile the graph
app = workflow.compile()
