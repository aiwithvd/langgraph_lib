# tests/test_client.py

import pytest
from langgraph_lib.client import AgentClient

@pytest.fixture
def client():
    return AgentClient(base_url="http://localhost:8000", auth_secret="test_secret")

def test_invoke(client):
    response = client.invoke("Hello, world!")
    assert response.type == "ai"
    assert isinstance(response.content, str)
