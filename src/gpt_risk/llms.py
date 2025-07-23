from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from src.app.config import settings


def get_gemini_llm():
    """Initializes and returns the Gemini Pro LLM."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest", google_api_key=settings.GOOGLE_API_KEY
    )


def get_qwen_llm():
    """
    Initializes and returns the Qwen LLM.
    NOTE: This uses the OpenAI client as a proxy for an OpenAI-compatible API.
    Update the base_url if you are self-hosting or using a different provider.
    """
    return ChatOpenAI(
        model="qwen-32b-chat",  # Or the specific model name you are using
        api_key=settings.OPENAI_API_KEY,
        # base_url="http://localhost:8000/v1" # Example for a local server
    )


def get_llama_llm():
    """
    Initializes and returns the Llama 3 LLM.
    NOTE: This uses the OpenAI client as a proxy.
    Update the base_url for your specific Llama 3 hosting endpoint.
    """
    return ChatOpenAI(
        model="llama-3-70b-instruct",
        api_key=settings.OPENAI_API_KEY,
        # base_url="http://your-llama-api-endpoint/v1"
    )
