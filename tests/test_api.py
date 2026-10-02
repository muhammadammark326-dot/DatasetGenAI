"""Integration tests for FastAPI server and endpoints."""

from fastapi.testclient import TestClient
from datasetgen.app.server import app

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


def test_generate_api():
    # 1. Plan
    plan_res = client.post(
        "/api/plan",
        json={"request": "Generate 5 physics questions", "provider": "mock", "overrides": {"number_of_examples": 5}},
    )
    assert plan_res.status_code == 200
    blueprint = plan_res.json()

    # 2. Generate
    gen_res = client.post(
        "/api/generate",
        json={
            "user_request": "Generate 5 physics questions",
            "blueprint": blueprint,
            "provider": "mock",
            "batch_size": 5,
        },
    )
    assert gen_res.status_code == 200
    data = gen_res.json()
    assert data["accepted_count"] == 5
    assert "report" in data
    assert "acceptance_rate" in data["report"]
    assert "sample_examples" in data

