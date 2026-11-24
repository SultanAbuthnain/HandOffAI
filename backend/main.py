"""
FastAPI Application Entrypoint for AI Customer Support Widget with RAG & HITL.

Serves:
- Embeddable Shadow DOM widget (`/widget.js`)
- Real-time Human-in-the-Loop Agent Dashboard (`/dashboard`)
- Built-in widget sandbox page (`/demo`) for instant local testing
- WebSocket endpoints for visitors (`/ws/chat/{session_id}`) and agents (`/ws/agent`)
- REST APIs for Knowledge Base ingestion, retrieval testing, and session control
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

try:
    from backend.database import Database, SessionState
    from backend.rag_engine import LLMProviderError, RAGConfigurationError, RAGEngine
    from backend.ws_manager import ConnectionManager, IPRateLimiter
except ImportError:  # Allows running directly from inside `backend/`
    from database import Database, SessionState
    from rag_engine import LLMProviderError, RAGConfigurationError, RAGEngine
    from ws_manager import ConnectionManager, IPRateLimiter

# ---------------------------------------------------------------------------
# Paths & Environment Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_support_hitl")

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB upload cap for .txt / .md files


def _normalize_origin(origin: str) -> str:
    cleaned = (origin or "").strip().rstrip("/")
    if not cleaned or cleaned == "*":
        return cleaned
    parsed = urlparse(cleaned)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"
    return cleaned.lower()


def _parse_allowed_origins() -> tuple[bool, set[str]]:
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
    entries = [_normalize_origin(item) for item in raw.split(",") if item.strip()]
    if "*" in entries:
        return True, {"*"}
    return False, set(entries)


ALLOW_ALL_ORIGINS, ALLOWED_ORIGIN_SET = _parse_allowed_origins()
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "").strip()


def _is_truthy_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def is_origin_allowed(origin: Optional[str], request_host: Optional[str] = None) -> bool:
    """
    Validate an incoming Origin header against `ALLOWED_ORIGINS`.
    Does not trust client-supplied `Host` headers to prevent Host+Origin spoofing.
    """
    _ = request_host
    if ALLOW_ALL_ORIGINS:
        return True
    if not origin:
        # Non-browser clients or same-origin navigation without Origin header
        return True

    norm_origin = _normalize_origin(origin)
    return norm_origin in ALLOWED_ORIGIN_SET


def _extract_client_ip_from_ws(ws: WebSocket) -> str:
    if _is_truthy_env("TRUST_PROXY_HEADERS", "false"):
        forwarded = ws.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    if ws.client and ws.client.host:
        return ws.client.host
    return "unknown"


def _verify_agent_secret(provided: Optional[str]) -> bool:
    if not AGENT_API_KEY:
        return True
    if not provided:
        return False
    return hmac.compare_digest(provided.strip(), AGENT_API_KEY)


async def require_agent_auth(
    x_agent_key: Optional[str] = Header(default=None, alias="X-Agent-Key"),
) -> None:
    """FastAPI dependency protecting agent & KB endpoints when AGENT_API_KEY is set."""
    if not _verify_agent_secret(x_agent_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Agent-Key header.",
        )


# ---------------------------------------------------------------------------
# Core Service Singletons & Lifespan
# ---------------------------------------------------------------------------

db_path_env = os.getenv("DATABASE_PATH", "./app.db")
resolved_db_path = (
    str((PROJECT_ROOT / db_path_env).resolve())
    if not Path(db_path_env).is_absolute()
    else db_path_env
)

db = Database(db_path=resolved_db_path)
rag_engine = RAGEngine(db=db)
rate_limiter = IPRateLimiter()
ws_manager = ConnectionManager(db=db, rag=rag_engine, rate_limiter=rate_limiter)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(
        "Starting AI Support HITL server | provider=%s | model=%s | db=%s | origins=%s",
        rag_engine.provider,
        rag_engine.chat_model,
        db.db_path,
        "*" if ALLOW_ALL_ORIGINS else sorted(ALLOWED_ORIGIN_SET),
    )
    if not rag_engine.is_configured:
        logger.warning(
            "LLM_API_KEY is not configured yet. Set LLM_API_KEY in .env to enable AI replies and KB embeddings."
        )
    yield
    await ws_manager.shutdown()
    logger.info("AI Support HITL server shut down cleanly.")


app = FastAPI(
    title="AI Customer Support Widget (RAG + HITL)",
    description="Self-hosted, zero-dependency AI support widget with local RAG and real-time Human-in-the-Loop handoff.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_ALL_ORIGINS else sorted(ALLOWED_ORIGIN_SET),
    allow_credentials=not ALLOW_ALL_ORIGINS,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class IngestTextRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Document title")
    content: str = Field(..., min_length=1, max_length=500_000, description="Plain text or Markdown content")
    source_type: str = Field(default="text", pattern="^(text|markdown)$")


class SearchKBRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=20)
    min_similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class AgentActionRequest(BaseModel):
    agent_name: str = Field(default="Support Agent", min_length=1, max_length=60)


class AgentMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)
    agent_name: str = Field(default="Support Agent", min_length=1, max_length=60)


# ---------------------------------------------------------------------------
# Frontend & Static Asset Routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", include_in_schema=False)
async def serve_dashboard() -> FileResponse:
    dashboard_file = STATIC_DIR / "dashboard.html"
    if not dashboard_file.exists():
        raise HTTPException(status_code=404, detail="static/dashboard.html not found.")
    return FileResponse(
        dashboard_file,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/widget.js", include_in_schema=False)
async def serve_widget_js() -> FileResponse:
    widget_file = STATIC_DIR / "widget.js"
    if not widget_file.exists():
        raise HTTPException(status_code=404, detail="static/widget.js not found.")
    return FileResponse(
        widget_file,
        media_type="application/javascript; charset=utf-8",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
async def serve_demo_page(request: Request) -> HTMLResponse:
    """Minimal host-website page to test the Shadow DOM `<script>` embed immediately."""
    base_url = str(request.base_url).rstrip("/")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Widget Embed Demo — AI Support HITL</title>
  <style>
    /* Intentionally aggressive host styles to prove Shadow DOM isolation */
    * {{ box-sizing: border-box; font-family: Georgia, serif; }}
    body {{
      margin: 0; min-height: 100vh; padding: 3rem 1.5rem;
      background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
      color: #0f172a;
    }}
    button {{ background: #dc2626 !important; color: #fef08a !important; border-radius: 0 !important; }}
    .card {{
      max-width: 720px; margin: 0 auto; background: #fff; padding: 2rem;
      border-radius: 16px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }}
    code, pre {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      background: #0f172a; color: #e2e8f0; border-radius: 8px;
    }}
    pre {{ padding: 1rem; overflow-x: auto; font-size: 0.875rem; }}
    a {{ color: #2563eb; text-decoration: none; font-weight: 600; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Host Website Demo (Shadow DOM Isolation Test)</h1>
    <p>
      This page intentionally applies aggressive global CSS (including red <code>button !important</code> rules)
      to verify that the embedded Shadow DOM chat widget in the bottom-right corner remains completely unaffected.
    </p>
    <p>Open the <a href="/dashboard" target="_blank">Agent Dashboard (/dashboard)</a> in a second tab to watch live chats, escalations, and manual takeovers in real time.</p>
    <h3>Single-Tag Embed Snippet</h3>
    <pre>&lt;script src="{base_url}/widget.js" data-title="Support Assistant" data-color="#2563eb"&gt;&lt;/script&gt;</pre>
  </div>
  <script src="{base_url}/widget.js" data-title="Support Assistant" data-color="#2563eb"></script>
</body>
</html>"""
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# Health & Config Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    kb_stats = await asyncio.to_thread(db.get_kb_stats)
    return {
        "status": "ok",
        "ai": rag_engine.health_info(),
        "kb": kb_stats,
        "auth_required": bool(AGENT_API_KEY),
        "active_visitors": ws_manager.active_visitor_count,
        "active_agents": ws_manager.active_agent_count,
    }


# ---------------------------------------------------------------------------
# WebSocket Endpoints
# ---------------------------------------------------------------------------


@app.websocket("/ws/chat/{session_id}")
async def websocket_visitor_chat(
    ws: WebSocket,
    session_id: str,
    visitor_name: str = Query(default="Visitor", max_length=80),
) -> None:
    origin = ws.headers.get("origin")
    host = ws.headers.get("host")
    if not is_origin_allowed(origin, host):
        logger.warning("Rejected WebSocket connection from disallowed origin: %s", origin)
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Origin not allowed")
        return

    clean_session_id = session_id.strip()[:120] or secrets.token_hex(16)
    client_ip = _extract_client_ip_from_ws(ws)

    # Rate-limit creation of brand-new sessions to prevent SQLite session-table flooding
    existing_session = await asyncio.to_thread(db.get_session, clean_session_id)
    if existing_session is None:
        allowed, _retry_after = rate_limiter.allow(client_ip)
        if not allowed:
            logger.warning("Rate-limited new session creation from IP %s", client_ip)
            await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Rate limit exceeded")
            return

    await ws_manager.connect_visitor(
        ws=ws,
        session_id=clean_session_id,
        visitor_name=visitor_name,
        origin=origin,
        client_ip=client_ip,
    )

    try:
        while True:
            data = await ws.receive_json()
            if isinstance(data, dict):
                await ws_manager.handle_visitor_event(ws, clean_session_id, data, client_ip)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Visitor WebSocket error for session %s", clean_session_id)
    finally:
        await ws_manager.disconnect_visitor(ws, clean_session_id)


@app.websocket("/ws/agent")
async def websocket_agent_dashboard(
    ws: WebSocket,
    api_key: Optional[str] = Query(default=None),
) -> None:
    origin = ws.headers.get("origin")
    if not is_origin_allowed(origin):
        logger.warning("Rejected agent WebSocket connection from disallowed origin: %s", origin)
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Origin not allowed")
        return

    header_key = ws.headers.get("x-agent-key")
    handshake_key = api_key or header_key
    already_accepted = False

    if AGENT_API_KEY:
        if handshake_key is not None:
            if not _verify_agent_secret(handshake_key):
                await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
                return
        else:
            # Support first-frame in-band authentication so the browser dashboard
            # does not have to leak AGENT_API_KEY in URL query parameters.
            await ws.accept()
            already_accepted = True
            try:
                first_msg = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
            except Exception:
                await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
                return
            if not isinstance(first_msg, dict) or first_msg.get("type") != "auth" or not _verify_agent_secret(
                str(first_msg.get("api_key", ""))
            ):
                await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
                return

    await ws_manager.connect_agent(ws, already_accepted=already_accepted)
    try:
        while True:
            data = await ws.receive_json()
            if isinstance(data, dict):
                if data.get("type") == "auth":
                    continue
                await ws_manager.handle_agent_event(ws, data)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Agent dashboard WebSocket error")
    finally:
        ws_manager.disconnect_agent(ws)


# ---------------------------------------------------------------------------
# Knowledge Base (RAG) REST Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/kb/documents", dependencies=[Depends(require_agent_auth)])
async def list_kb_documents() -> dict[str, Any]:
    documents = await asyncio.to_thread(db.list_documents)
    stats = await asyncio.to_thread(db.get_kb_stats)
    return {"documents": documents, "stats": stats}


@app.post("/api/kb/ingest/text", dependencies=[Depends(require_agent_auth)])
async def ingest_kb_text(body: IngestTextRequest) -> dict[str, Any]:
    try:
        doc = await rag_engine.ingest_text(
            title=body.title,
            content=body.content,
            source_type=body.source_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RAGConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    await ws_manager.notify_kb_updated()
    return {"document": doc}


@app.post("/api/kb/ingest/file", dependencies=[Depends(require_agent_auth)])
async def ingest_kb_file(file: UploadFile = File(...)) -> dict[str, Any]:
    content_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    try:
        doc = await rag_engine.ingest_file(
            filename=file.filename or "document.txt",
            content_bytes=content_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RAGConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    await ws_manager.notify_kb_updated()
    return {"document": doc}


@app.delete("/api/kb/documents/{document_id}", dependencies=[Depends(require_agent_auth)])
async def delete_kb_document(document_id: str) -> dict[str, Any]:
    deleted = await asyncio.to_thread(db.delete_document, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    await ws_manager.notify_kb_updated()
    return {"deleted": True, "document_id": document_id}


@app.post("/api/kb/search", dependencies=[Depends(require_agent_auth)])
async def search_kb(body: SearchKBRequest) -> dict[str, Any]:
    try:
        chunks = await rag_engine.retrieve(
            query=body.query,
            top_k=body.top_k,
            min_similarity=body.min_similarity,
        )
    except RAGConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "query": body.query,
        "results": [
            {
                **chunk.to_dict(),
                "content": chunk.content,
            }
            for chunk in chunks
        ],
    }


# ---------------------------------------------------------------------------
# Session & HITL Control REST Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/sessions", dependencies=[Depends(require_agent_auth)])
async def list_sessions(
    state: Optional[SessionState] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    sessions = await asyncio.to_thread(db.list_sessions, state, limit, offset)
    enriched = [
        {**s, "visitor_online": ws_manager.is_visitor_online(s["id"])}
        for s in sessions
    ]
    return {"sessions": enriched}


@app.get("/api/sessions/{session_id}", dependencies=[Depends(require_agent_auth)])
async def get_session_detail(session_id: str) -> dict[str, Any]:
    session = await asyncio.to_thread(db.get_session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    messages = await asyncio.to_thread(db.get_messages, session_id, 500)
    return {
        "session": {**session, "visitor_online": ws_manager.is_visitor_online(session_id)},
        "messages": messages,
    }


@app.post("/api/sessions/{session_id}/takeover", dependencies=[Depends(require_agent_auth)])
async def takeover_session_endpoint(
    session_id: str,
    body: Optional[AgentActionRequest] = None,
) -> dict[str, Any]:
    agent_name = body.agent_name if body else "Support Agent"
    session = await ws_manager.take_over_session(session_id, agent_name=agent_name)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session": session}


@app.post("/api/sessions/{session_id}/release", dependencies=[Depends(require_agent_auth)])
async def release_session_endpoint(
    session_id: str,
    body: Optional[AgentActionRequest] = None,
) -> dict[str, Any]:
    agent_name = body.agent_name if body else "Support Agent"
    session = await ws_manager.release_session_to_bot(session_id, agent_name=agent_name)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session": session}


@app.post("/api/sessions/{session_id}/messages", dependencies=[Depends(require_agent_auth)])
async def post_agent_message_endpoint(
    session_id: str,
    body: AgentMessageRequest,
) -> dict[str, Any]:
    message = await ws_manager.send_agent_message(
        session_id=session_id,
        content=body.content,
        agent_name=body.agent_name,
    )
    if not message:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": message}


@app.delete("/api/sessions/{session_id}", dependencies=[Depends(require_agent_auth)])
async def delete_session_endpoint(session_id: str) -> dict[str, Any]:
    deleted = await asyncio.to_thread(db.delete_session, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    await ws_manager.notify_session_deleted(session_id)
    return {"deleted": True, "session_id": session_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
