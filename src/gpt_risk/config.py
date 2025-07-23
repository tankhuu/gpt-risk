from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM API Keys
    GOOGLE_API_KEY: str
    OPENAI_API_KEY: str

    # Databricks Configuration
    DATABRICKS_HOST: str
    DATABRICKS_TOKEN: str

    # Databricks Vector Search Index Names
    FRAUD_RAG_INDEX_NAME: str
    CREDIT_RAG_INDEX_NAME: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
