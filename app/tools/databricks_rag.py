from langchain_core.tools import tool
from databricks_langchain import DatabricksVectorSearch, DatabricksEmbeddings
from app.config import settings
import json


@tool
def query_databricks_vector_search(query: str, index_name: str) -> str:
    """
    Queries a Databricks Vector Search index to retrieve relevant context.
    Input should be a natural language query and the name of the index to search.
    Returns a JSON string of the retrieved documents.
    """
    try:
        embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

        dvs = DatabricksVectorSearch(
            endpoint=settings.DATABRICKS_HOST,
            index_name=index_name,
            embedding=embeddings,
            databricks_token=settings.DATABRICKS_TOKEN,
        )

        retriever = dvs.as_retriever(search_kwargs={"k": 3})
        results = retriever.invoke(query)

        # Format results for the LLM
        formatted_results = [
            {"source": doc.metadata.get("source", "N/A"), "content": doc.page_content}
            for doc in results
        ]
        return json.dumps(formatted_results)

    except Exception as e:
        # In a real app, you'd want more specific error handling
        return json.dumps({"error": f"Databricks RAG query failed: {str(e)}"})
