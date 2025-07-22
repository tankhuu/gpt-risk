import gradio as gr
from fastapi import FastAPI
import uvicorn
import httpx
import json
from pydantic import BaseModel
import os

# Mount the Gradio app
from app.graph import app as langgraph_app

# --- FastAPI App ---

app = FastAPI(
    title="Financial AI Assistant API",
    description="API for interacting with the financial risk assessment agent.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    query: dict
    thread_id: str


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint to interact with the financial agent.
    Streams the final response back.
    """
    config = {"configurable": {"thread_id": request.thread_id}}
    final_response = ""
    async for event in langgraph_app.astream(
        {"input_data": request.query, "messages": ""},
        config=config,
        stream_mode="values",
    ):
        if "final_summary" in event and event["final_summary"]:
            final_response = event["final_summary"]

    return {"response": final_response}


# --- Gradio UI ---


def create_gradio_ui():
    """Creates and returns the Gradio ChatInterface."""

    async def chat_fn(message, history):
        thread_id = (
            "user_session_123"  # In a real app, this would be unique per user/session
        )

        # Attempt to parse the message as JSON, otherwise treat as a text query
        try:
            query_data = json.loads(message)
        except json.JSONDecodeError:
            query_data = {"text_query": message}

        # Use httpx to call the FastAPI backend
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "http://127.0.0.1:8000/chat",
                    json={"query": query_data, "thread_id": thread_id},
                    timeout=120.0,
                )
                response.raise_for_status()
                result = response.json()

                # Stream the response to the UI
                full_response = result.get("response", "Sorry, I encountered an error.")
                for i in range(len(full_response)):
                    yield full_response[: i + 1]

            except httpx.RequestError as e:
                yield f"Error connecting to the backend: {e}"
            except Exception as e:
                yield f"An unexpected error occurred: {e}"

    # Example JSON inputs for easy testing
    fraud_example = json.dumps(
        {
            "transaction_id": "t12345",
            "customer_id": "c67890",
            "transaction_amount": 2500.75,
            "merchant_id": "m54321",
            "timestamp": "2025-07-22T08:30:00Z",
        },
        indent=2,
    )

    credit_example = json.dumps(
        {
            "application_id": "a98765",
            "customer_id": "c11223",
            "loan_amount": 50000,
            "annual_income": 85000,
            "employment_length_years": 5,
            "dti_ratio": 0.3,
        },
        indent=2,
    )

    interface = gr.ChatInterface(
        fn=chat_fn,
        title="Financial AI Assistant",
        description="Your AI partner for real-time fraud detection and credit risk analysis. Paste a JSON object for a transaction or loan application, or ask a general question.",
        examples=["Analyze this transaction:", fraud_example],
        # ["Assess this loan application:", credit_example],
        cache_examples=False,
    )
    return interface


# Mount the Gradio app to the FastAPI app
gradio_ui = create_gradio_ui()
app = gr.mount_gradio_app(app, gradio_ui, path="/")


def run_app():
    """Function to run the Uvicorn server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run_app()
