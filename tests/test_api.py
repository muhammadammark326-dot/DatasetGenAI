"""Integration tests for FastAPI server and endpoints."""

from fastapi.testclient import TestClient
from app.api.server import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_plan_api():
    res = client.post(
        "/api/plan",
        json={"request": "Generate 10 physics questions", "provider": "mock", "overrides": {"number_of_examples": 10}},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["number_of_examples"] == 10
    assert data["domain"] == "physics"


def test_list_datasets():
    res = client.get("/api/datasets")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_list_traces():
    res = client.get("/api/traces")
    assert res.status_code == 200
    assert isinstance(res.json(), list)
