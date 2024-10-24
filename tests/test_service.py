# tests/test_service.py

from fastapi.testclient import TestClient
from langgraph_lib.service import app

client = TestClient(app)

def test_invoke_endpoint():
    response = client.post("/invoke", json={"message": "Hello, world!"})
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "ai"
    assert isinstance(data["content"], str)
