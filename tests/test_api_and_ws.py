"""Integration tests for FastAPI REST endpoints, WebSocket HITL handoff, and security controls."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

# Ensure importing backend.main during test collection never creates ./app.db in the repo root
_IMPORT_TEMP_DIR = tempfile.mkdtemp(prefix="ai_support_hitl_import_")
os.environ.setdefault("DATABASE_PATH", str(Path(_IMPORT_TEMP_DIR) / "boot.db"))
atexit.register(shutil.rmtree, _IMPORT_TEMP_DIR, ignore_errors=True)

import backend.main as main_mod
from backend.database import Database, SessionState
from backend.rag_engine import RAGEngine
from backend.ws_manager import ConnectionManager, IPRateLimiter


@pytest.fixture()
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    test_db = Database(db_path=str(tmp_path / "api_test.db"))
    test_rag = RAGEngine(db=test_db, provider="openai", api_key="sk-test-key")

    async def fake_embed_texts(texts: list[str]) -> list[np.ndarray]:
        return [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) for _ in texts]

    async def fake_openai_chat(system_prompt: str, history: list[dict[str, str]], user_prompt: str) -> str:
        return "Grounded response from test knowledge base."

    monkeypatch.setattr(test_rag, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(test_rag, "_call_openai_chat", fake_openai_chat)

    limiter = IPRateLimiter(max_requests=5, window_seconds=60)
    manager = ConnectionManager(db=test_db, rag=test_rag, rate_limiter=limiter)

    monkeypatch.setattr(main_mod, "db", test_db)
    monkeypatch.setattr(main_mod, "rag_engine", test_rag)
    monkeypatch.setattr(main_mod, "rate_limiter", limiter)
    monkeypatch.setattr(main_mod, "ws_manager", manager)
    monkeypatch.setattr(main_mod, "ALLOW_ALL_ORIGINS", False)
    monkeypatch.setattr(
        main_mod,
        "ALLOWED_ORIGIN_SET",
        {"http://localhost:8000", "http://127.0.0.1:8000"},
    )
    monkeypatch.setattr(main_mod, "AGENT_API_KEY", "")

    with TestClient(main_mod.app) as client:
        yield client


def test_health_and_frontend_routes(app_client: TestClient) -> None:
    health = app_client.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["ai"]["provider"] == "openai"

    assert app_client.get("/dashboard").status_code == 200
    assert app_client.get("/widget.js").status_code == 200
    assert app_client.get("/demo").status_code == 200


def test_kb_ingest_search_and_delete_endpoints(app_client: TestClient) -> None:
    res_ingest = app_client.post(
        "/api/kb/ingest/text",
        json={
            "title": "Pricing FAQ",
            "content": "The Starter plan costs $29/month and Pro costs $99/month.",
            "source_type": "markdown",
        },
    )
    assert res_ingest.status_code == 200
    doc = res_ingest.json()["document"]
    assert doc["title"] == "Pricing FAQ"

    res_search = app_client.post(
        "/api/kb/search",
        json={"query": "How much is the Pro plan?", "top_k": 2},
    )
    assert res_search.status_code == 200
    results = res_search.json()["results"]
    assert len(results) == 1
    assert results[0]["document_title"] == "Pricing FAQ"

    res_del = app_client.delete(f"/api/kb/documents/{doc['id']}")
    assert res_del.status_code == 200
    assert res_del.json()["deleted"] is True


def test_agent_api_key_protects_rest_and_ws_first_frame_auth(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_mod, "AGENT_API_KEY", "super-secret-agent-key")

    # Unauthenticated REST call must return 401
    assert app_client.get("/api/kb/documents").status_code == 401
    assert (
        app_client.get(
            "/api/kb/documents",
            headers={"X-Agent-Key": "super-secret-agent-key"},
        ).status_code
        == 200
    )

    # First-frame WebSocket authentication without query-string key leakage
    with app_client.websocket_connect("/ws/agent") as ws_agent:
        ws_agent.send_json({"type": "auth", "api_key": "super-secret-agent-key"})
        init_payload = ws_agent.receive_json()
        assert init_payload["type"] == "dashboard_init"

    # Invalid first-frame key must be closed with policy violation (1008)
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with app_client.websocket_connect("/ws/agent") as ws_bad:
            ws_bad.send_json({"type": "auth", "api_key": "wrong-key"})
            ws_bad.receive_json()
    assert exc_info.value.code == 1008


def test_websocket_origin_validation_blocks_cswsh_and_host_spoofing(
    app_client: TestClient,
) -> None:
    # 1. Disallowed origin on /ws/chat must fail even if Host header is spoofed to match
    with pytest.raises(WebSocketDisconnect) as chat_exc:
        with app_client.websocket_connect(
            "/ws/chat/evil-session",
            headers={
                "Origin": "https://evil-attacker.com",
                "Host": "evil-attacker.com",
            },
        ) as ws:
            ws.receive_json()
    assert chat_exc.value.code == 1008

    # 2. Disallowed origin on /ws/agent must also fail (CSWSH protection)
    with pytest.raises(WebSocketDisconnect) as agent_exc:
        with app_client.websocket_connect(
            "/ws/agent",
            headers={"Origin": "https://evil-attacker.com"},
        ) as ws_agent:
            ws_agent.receive_json()
    assert agent_exc.value.code == 1008


def test_x_forwarded_for_ignored_by_default_and_rate_limiting_enforced(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "false")
    with app_client.websocket_connect(
        "/ws/chat/xff-session",
        headers={"X-Forwarded-For": "203.0.113.99"},
    ) as ws:
        init_msg = ws.receive_json()
        # Must NOT record the spoofed X-Forwarded-For address when TRUST_PROXY_HEADERS=false
        assert init_msg["session"]["client_ip"] != "203.0.113.99"


def test_websocket_hitl_escalation_takeover_and_release(app_client: TestClient) -> None:
    with app_client.websocket_connect("/ws/agent") as ws_agent:
        assert ws_agent.receive_json()["type"] == "dashboard_init"

        with app_client.websocket_connect("/ws/chat/hitl-flow-1?visitor_name=Dana") as ws_visitor:
            v_init = ws_visitor.receive_json()
            assert v_init["type"] == "session_init"
            assert v_init["session"]["state"] == SessionState.BOT_ACTIVE.value

            # Agent receives session_updated notification
            assert ws_agent.receive_json()["type"] == "session_updated"

            # Visitor clicks "Talk to a human"
            ws_visitor.send_json({"type": "request_human"})
            v_sys_msg = ws_visitor.receive_json()
            v_state_change = ws_visitor.receive_json()
            assert v_sys_msg["type"] == "message"
            assert v_state_change["type"] == "state_change"
            assert v_state_change["session"]["state"] == SessionState.WAITING_FOR_HUMAN.value

            # Drain agent events (message, state_change, escalation)
            agent_events = [ws_agent.receive_json()["type"] for _ in range(3)]
            assert "escalation" in agent_events

            # Agent clicks "Take Over"
            ws_agent.send_json(
                {
                    "type": "take_over",
                    "session_id": "hitl-flow-1",
                    "agent_name": "Agent Alex",
                }
            )
            # Visitor receives typing=False, system join message, and state_change -> HUMAN_ACTIVE
            v_events = [ws_visitor.receive_json() for _ in range(3)]
            state_evt = next(e for e in v_events if e["type"] == "state_change")
            assert state_evt["session"]["state"] == SessionState.HUMAN_ACTIVE.value
            assert state_evt["session"]["assigned_agent"] == "Agent Alex"

            # Drain corresponding agent broadcasts
            for _ in range(3):
                ws_agent.receive_json()

            # Agent releases session back to bot
            ws_agent.send_json(
                {
                    "type": "release_to_bot",
                    "session_id": "hitl-flow-1",
                    "agent_name": "Agent Alex",
                }
            )
            v_rel_events = [ws_visitor.receive_json() for _ in range(2)]
            rel_state_evt = next(e for e in v_rel_events if e["type"] == "state_change")
            assert rel_state_evt["session"]["state"] == SessionState.BOT_ACTIVE.value
