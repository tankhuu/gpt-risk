import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from typing import Generator

# Import the FastAPI gpt_risk instance
from src.app.main import app


@pytest_asyncio.fixture(scope="function")
async def test_client() -> Generator:
    """
    Fixture for creating an async test client for the FastAPI gpt_risk.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
