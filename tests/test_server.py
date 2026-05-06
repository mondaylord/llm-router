"""Smoke test for the FastAPI app."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_route_endpoint_returns_decision():
    # Construct a fresh app with no env config so it uses defaults.
    from llm_router.server.app import app

    with TestClient(app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "rules_count" in body

        r = client.post("/route", json={"prompt": "hi"})
        assert r.status_code == 200
        body = r.json()
        assert body["tier"] == "weak"
        assert body["layer"] == "rule"

        r = client.post("/route", json={
            "prompt": "Tell me about quantum computing.",
            "session_id": "s1",
        })
        assert r.status_code == 200
        assert r.json()["tier"] in {"weak", "mid", "strong"}

        r = client.post("/route", json={"prompt": ""})
        assert r.status_code == 400


def test_route_rejects_unknown_fields():
    from llm_router.server.app import app

    with TestClient(app) as client:
        r = client.post("/route", json={"prompt": "hi", "garbage": True})
        assert r.status_code == 422
