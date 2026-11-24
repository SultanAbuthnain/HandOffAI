"""
Real-time WebSocket Connection Manager, HITL State Machine Coordinator,
and IP-based Sliding-Window Rate Limiter.

Key guarantees:
- Strict HITL state machine (`BOT_ACTIVE`, `WAITING_FOR_HUMAN`, `HUMAN_ACTIVE`)
  backed by SQLite (`database.py`).
- Immediate cancellation of in-flight AI generation tasks upon human takeover.
- Dual race-condition guard: per-session asyncio lock + atomic SQLite state
  verification (`state == 'BOT_ACTIVE'`) right before persisting and dispatching
  any AI message over WebSocket.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Optional

import httpx
from fastapi import WebSocket

try:
    from backend.database import Database, SessionState
    from backend.rag_engine import (
        DEFAULT_ESCALATION_MESSAGE,
        LLMProviderError,
        RAGConfigurationError,
        RAGEngine,
    )
except ImportError:  # Allows running directly from inside `backend/`
    from database import Database, SessionState
    from rag_engine import (
        DEFAULT_ESCALATION_MESSAGE,
        LLMProviderError,
        RAGConfigurationError,
        RAGEngine,
    )

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4000


class IPRateLimiter:
    """In-memory sliding-window rate limiter keyed by client IP address."""

    def __init__(
        self,
        max_requests: Optional[int] = None,
        window_seconds: Optional[int] = None,
    ) -> None:
        self.max_requests = max(
            1,
            int(max_requests if max_requests is not None else os.getenv("RATE_LIMIT_REQUESTS", "20")),
        )
        self.window_seconds = max(
            1,
            int(
                window_seconds
                if window_seconds is not None
                else os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")
            ),
        )
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, client_ip: str) -> tuple[bool, int]:
        """
        Record a request for `client_ip` if within limits.
        Returns `(allowed, retry_after_seconds)`.
        """
        key = (client_ip or "unknown").strip() or "unknown"
        now = time.monotonic()
        cutoff = now - self.window_seconds

        bucket = self._hits[key]
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= self.max_requests:
            retry_after = max(1, int(self.window_seconds - (now - bucket[0])) + 1)
            return False, retry_after

        bucket.append(now)

        # Periodically prune stale IP keys to prevent unbounded memory growth
        if len(self._hits) > 5000:
            stale_keys = [ip for ip, dq in self._hits.items() if not dq or dq[-1] <= cutoff]
            for ip in stale_keys:
                del self._hits[ip]

        return True, 0


class ConnectionManager:
    """Manages visitor & agent WebSockets, HITL transitions, and async AI tasks."""

    def __init__(
        self,
        db: Database,
        rag: RAGEngine,
        rate_limiter: Optional[IPRateLimiter] = None,
    ) -> None:
        self.db = db
        self.rag = rag
        self.rate_limiter = rate_limiter or IPRateLimiter()

        # Active WebSocket connections
        self._visitor_sockets: dict[str, set[WebSocket]] = defaultdict(set)
        self._agent_sockets: set[WebSocket] = set()

        # In-flight AI generation tasks & per-session critical-section locks
        self._ai_tasks: dict[str, asyncio.Task[None]] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    def is_visitor_online(self, session_id: str) -> bool:
        return bool(self._visitor_sockets.get(session_id))

    @property
    def active_visitor_count(self) -> int:
        return sum(1 for sockets in self._visitor_sockets.values() if sockets)

    @property
    def active_agent_count(self) -> int:
        return len(self._agent_sockets)

    def _enrich_session(self, session: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(session)
        enriched["visitor_online"] = self.is_visitor_online(session["id"])
        return enriched

    # ------------------------------------------------------------------
    # Low-Level Safe WebSocket Sending & Broadcasting
    # ------------------------------------------------------------------

    async def _safe_send(self, ws: WebSocket, payload: dict[str, Any]) -> bool:
        try:
            await ws.send_json(payload)
            return True
        except Exception:
            return False

    async def send_to_visitor(self, session_id: str, payload: dict[str, Any]) -> None:
        sockets = list(self._visitor_sockets.get(session_id, ()))
        if not sockets:
            return
        dead: list[WebSocket] = []
        for ws in sockets:
            ok = await self._safe_send(ws, payload)
            if not ok:
                dead.append(ws)
        for ws in dead:
            self._visitor_sockets[session_id].discard(ws)
        if not self._visitor_sockets.get(session_id):
            self._visitor_sockets.pop(session_id, None)

    async def broadcast_to_agents(self, payload: dict[str, Any]) -> None:
        if not self._agent_sockets:
            return
        dead: list[WebSocket] = []
        for ws in list(self._agent_sockets):
            ok = await self._safe_send(ws, payload)
            if not ok:
                dead.append(ws)
        for ws in dead:
            self._agent_sockets.discard(ws)

    async def broadcast_to_session_and_agents(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> None:
        await asyncio.gather(
            self.send_to_visitor(session_id, payload),
            self.broadcast_to_agents(payload),
        )

    # ------------------------------------------------------------------
    # Connection Lifecycle: Visitors & Agents
    # ------------------------------------------------------------------

    async def connect_visitor(
        self,
        ws: WebSocket,
        session_id: str,
        visitor_name: str = "Visitor",
        origin: Optional[str] = None,
        client_ip: Optional[str] = None,
        visitor_token: Optional[str] = None,
    ) -> dict[str, Any]:
        await ws.accept()

        session = await asyncio.to_thread(
            self.db.get_or_create_session,
            session_id,
            visitor_name,
            None,
            origin,
            client_ip,
            visitor_token,
        )
        messages = await asyncio.to_thread(self.db.get_messages, session["id"], 200)

        # Register socket AFTER async DB reads and send `session_init` immediately
        # so concurrent background broadcasts never arrive before `session_init`.
        self._visitor_sockets[session_id].add(ws)
        enriched = self._enrich_session(session)

        await self._safe_send(
            ws,
            {
                "type": "session_init",
                "session": enriched,
                "messages": messages,
            },
        )
        await self.broadcast_to_agents(
            {
                "type": "session_updated",
                "session": enriched,
            }
        )
        return enriched

    async def disconnect_visitor(self, ws: WebSocket, session_id: str) -> None:
        sockets = self._visitor_sockets.get(session_id)
        if sockets is not None:
            sockets.discard(ws)
            if not sockets:
                self._visitor_sockets.pop(session_id, None)

        if not self.is_visitor_online(session_id):
            if session_id not in self._ai_tasks:
                lock = self._session_locks.get(session_id)
                if lock is not None and not lock.locked():
                    self._session_locks.pop(session_id, None)
            await self.broadcast_to_agents(
                {
                    "type": "presence",
                    "session_id": session_id,
                    "visitor_online": False,
                }
            )

    async def connect_agent(self, ws: WebSocket, already_accepted: bool = False) -> None:
        if not already_accepted:
            await ws.accept()
        self._agent_sockets.add(ws)

        sessions = await asyncio.to_thread(self.db.list_sessions, None, 200, 0)
        kb_stats = await asyncio.to_thread(self.db.get_kb_stats)
        enriched_sessions = [self._enrich_session(s) for s in sessions]

        await self._safe_send(
            ws,
            {
                "type": "dashboard_init",
                "sessions": enriched_sessions,
                "kb_stats": kb_stats,
                "ai_info": self.rag.health_info(),
            },
        )

    def disconnect_agent(self, ws: WebSocket) -> None:
        self._agent_sockets.discard(ws)

    # ------------------------------------------------------------------
    # In-Flight AI Task Management
    # ------------------------------------------------------------------

    def _cancel_ai_task(self, session_id: str) -> bool:
        """Immediately cancel any pending AI generation task for `session_id`."""
        task = self._ai_tasks.pop(session_id, None)
        if task is not None and not task.done():
            task.cancel()
            return True
        return False

    async def shutdown(self) -> None:
        """Cancel all running AI tasks cleanly on server shutdown."""
        tasks = list(self._ai_tasks.values())
        self._ai_tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Visitor Inbound Events
    # ------------------------------------------------------------------

    async def handle_visitor_event(
        self,
        ws: WebSocket,
        session_id: str,
        payload: dict[str, Any],
        client_ip: str,
    ) -> None:
        event_type = str(payload.get("type", "")).strip().lower()

        if event_type == "ping":
            await self._safe_send(ws, {"type": "pong"})
            return

        if event_type == "request_human":
            allowed, retry_after = self.rate_limiter.allow(client_ip)
            if not allowed:
                await self._safe_send(
                    ws,
                    {
                        "type": "error",
                        "code": "rate_limited",
                        "message": f"Too many requests. Please wait {retry_after}s and try again.",
                        "retry_after": retry_after,
                    },
                )
                return
            await self.escalate_session_by_visitor(
                session_id=session_id,
                reason="Customer clicked 'Talk to a human'",
            )
            return

        if event_type == "set_email":
            allowed, retry_after = self.rate_limiter.allow(client_ip)
            if not allowed:
                await self._safe_send(
                    ws,
                    {
                        "type": "error",
                        "code": "rate_limited",
                        "message": f"Too many requests. Please wait {retry_after}s and try again.",
                        "retry_after": retry_after,
                    },
                )
                return
            email = str(payload.get("email", "")).strip()[:120]
            if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
                await self._safe_send(
                    ws,
                    {
                        "type": "error",
                        "code": "invalid_email",
                        "message": "Please enter a valid email address.",
                    },
                )
                return
            updated = await asyncio.to_thread(self.db.update_visitor_email, session_id, email)
            if updated:
                enriched = self._enrich_session(updated)
                await self.broadcast_to_session_and_agents(
                    session_id,
                    {"type": "state_change", "session": enriched},
                )
            return

        if event_type == "message":
            raw_content = str(payload.get("content", "")).strip()
            if not raw_content:
                return
            if len(raw_content) > MAX_MESSAGE_LENGTH:
                await self._safe_send(
                    ws,
                    {
                        "type": "error",
                        "code": "message_too_long",
                        "message": f"Message exceeds the {MAX_MESSAGE_LENGTH}-character limit.",
                    },
                )
                return

            allowed, retry_after = self.rate_limiter.allow(client_ip)
            if not allowed:
                await self._safe_send(
                    ws,
                    {
                        "type": "error",
                        "code": "rate_limited",
                        "message": f"You are sending messages too quickly. Please wait {retry_after}s and try again.",
                        "retry_after": retry_after,
                    },
                )
                return

            await self._process_visitor_message(session_id, raw_content)
            return

        await self._safe_send(
            ws,
            {
                "type": "error",
                "code": "unknown_event",
                "message": f"Unsupported event type '{event_type}'.",
            },
        )

    async def _process_visitor_message(self, session_id: str, content: str) -> None:
        # Persist visitor message first
        message = await asyncio.to_thread(self.db.add_message, session_id, "user", content)
        session = await asyncio.to_thread(self.db.get_session, session_id)
        if not session:
            return

        enriched = self._enrich_session(session)

        await self.broadcast_to_session_and_agents(
            session_id,
            {
                "type": "message",
                "session_id": session_id,
                "message": message,
                "session": enriched,
            },
        )

        # Strict HITL check: only generate an AI reply when session is BOT_ACTIVE
        current_state = SessionState(session["state"])
        if current_state != SessionState.BOT_ACTIVE:
            return

        # Cancel any superseded AI generation task for this session and spawn a fresh one
        self._cancel_ai_task(session_id)
        task = asyncio.create_task(
            self._run_ai_reply(session_id, content),
            name=f"ai-reply-{session_id}",
        )
        self._ai_tasks[session_id] = task

    async def _send_escalation_webhook(
        self,
        session: dict[str, Any],
        reason: str,
    ) -> None:
        """Fire an optional Slack/Discord/JSON webhook when a session escalates."""
        webhook_url = os.getenv("ESCALATION_WEBHOOK_URL", "").strip()
        if not webhook_url:
            return
        visitor_label = session.get("visitor_name") or "Visitor"
        email_part = f" ({session['visitor_email']})" if session.get("visitor_email") else ""
        text = (
            f"[HandoffAI Escalation] {visitor_label}{email_part} "
            f"[#{session['id'][:8]}]: {reason}"
        )
        payload = {
            "event": "escalation",
            "text": text,
            "content": text,
            "session_id": session["id"],
            "visitor_name": visitor_label,
            "visitor_email": session.get("visitor_email"),
            "reason": reason,
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(webhook_url, json=payload)
        except Exception as exc:
            logger.warning("Escalation webhook delivery failed: %s", exc)

    async def _run_ai_reply(self, session_id: str, user_text: str) -> None:
        current_task = asyncio.current_task()
        stream_id = f"stream-{uuid.uuid4().hex[:12]}"
        streamed_any = False

        await self.broadcast_to_session_and_agents(
            session_id,
            {
                "type": "typing",
                "session_id": session_id,
                "sender": "bot",
                "is_typing": True,
            },
        )

        try:
            # Verify state before incurring LLM latency/cost
            state_before = await asyncio.to_thread(self.db.get_session_state, session_id)
            if state_before != SessionState.BOT_ACTIVE:
                return

            history = await asyncio.to_thread(
                self.db.get_recent_conversation_history,
                session_id,
                8,
            )

            rag_reply_text: str
            should_escalate: bool
            escalation_reason: Optional[str]
            metadata: dict[str, Any]

            async def _on_delta(delta: str) -> None:
                nonlocal streamed_any
                if self._ai_tasks.get(session_id) is not current_task:
                    return
                state_mid = await asyncio.to_thread(self.db.get_session_state, session_id)
                if state_mid != SessionState.BOT_ACTIVE:
                    return
                streamed_any = True
                await self.send_to_visitor(
                    session_id,
                    {
                        "type": "message_stream",
                        "session_id": session_id,
                        "stream_id": stream_id,
                        "delta": delta,
                    },
                )

            try:
                rag_response = await self.rag.generate_reply_stream(
                    user_message=user_text,
                    conversation_history=history,
                    on_delta=_on_delta,
                )
                rag_reply_text = rag_response.reply
                should_escalate = rag_response.should_escalate
                escalation_reason = rag_response.escalation_reason
                metadata = {
                    "sources": rag_response.sources,
                    "confidence": rag_response.confidence,
                    "provider": rag_response.provider,
                    "model": rag_response.model,
                    "escalated": rag_response.should_escalate,
                }
            except asyncio.CancelledError:
                raise
            except (RAGConfigurationError, LLMProviderError) as exc:
                logger.warning("RAG/LLM unavailable for session %s: %s", session_id, exc)
                rag_reply_text = (
                    "Our AI assistant is temporarily unavailable. "
                    "I've alerted a human support agent who will assist you shortly."
                )
                should_escalate = True
                escalation_reason = f"AI provider error: {exc}"
                metadata = {"error": str(exc), "escalated": True}
            except Exception as exc:
                logger.exception("Unexpected error during AI generation for session %s", session_id)
                rag_reply_text = DEFAULT_ESCALATION_MESSAGE
                should_escalate = True
                escalation_reason = "Unexpected AI pipeline error"
                metadata = {"error": str(exc), "escalated": True}

            # CRITICAL RACE-CONDITION GUARD:
            # Lock the session, verify state == 'BOT_ACTIVE' both in SQLite write transaction
            # AND immediately before dispatching over WebSocket.
            lock = self._get_session_lock(session_id)
            async with lock:
                state_now = await asyncio.to_thread(self.db.get_session_state, session_id)
                if state_now != SessionState.BOT_ACTIVE:
                    logger.info(
                        "Discarding AI reply for session %s because state changed to %s.",
                        session_id,
                        state_now,
                    )
                    if streamed_any:
                        await self.send_to_visitor(
                            session_id,
                            {
                                "type": "message_stream_abort",
                                "session_id": session_id,
                                "stream_id": stream_id,
                            },
                        )
                    return

                bot_msg = await asyncio.to_thread(
                    self.db.add_bot_message_if_active,
                    session_id,
                    rag_reply_text,
                    metadata,
                    should_escalate,
                    escalation_reason,
                )
                if bot_msg is None:
                    logger.info(
                        "Atomic SQLite guard discarded AI reply for session %s (human took over).",
                        session_id,
                    )
                    if streamed_any:
                        await self.send_to_visitor(
                            session_id,
                            {
                                "type": "message_stream_abort",
                                "session_id": session_id,
                                "stream_id": stream_id,
                            },
                        )
                    return

                # Final verification before WebSocket dispatch
                post_state = await asyncio.to_thread(self.db.get_session_state, session_id)
                expected_post_state = (
                    SessionState.WAITING_FOR_HUMAN
                    if should_escalate
                    else SessionState.BOT_ACTIVE
                )
                if post_state != expected_post_state:
                    logger.info(
                        "Skipping WebSocket dispatch for session %s (post_state=%s).",
                        session_id,
                        post_state,
                    )
                    return

                updated_session = await self._get_enriched_session_summary(session_id)
                await self.broadcast_to_session_and_agents(
                    session_id,
                    {
                        "type": "message",
                        "session_id": session_id,
                        "stream_id": stream_id if streamed_any else None,
                        "message": bot_msg,
                        "session": updated_session,
                    },
                )

                if should_escalate and updated_session:
                    await self.broadcast_to_session_and_agents(
                        session_id,
                        {
                            "type": "state_change",
                            "session": updated_session,
                        },
                    )
                    reason_text = escalation_reason or "Escalated to human support"
                    await self.broadcast_to_agents(
                        {
                            "type": "escalation",
                            "session": updated_session,
                            "reason": reason_text,
                        }
                    )
                    asyncio.create_task(self._send_escalation_webhook(updated_session, reason_text))

        except asyncio.CancelledError:
            logger.info("Cancelled in-flight AI generation for session %s.", session_id)
            if streamed_any:
                await self.send_to_visitor(
                    session_id,
                    {
                        "type": "message_stream_abort",
                        "session_id": session_id,
                        "stream_id": stream_id,
                    },
                )
        finally:
            is_active_task = self._ai_tasks.get(session_id) is current_task
            if is_active_task:
                self._ai_tasks.pop(session_id, None)
                await self.broadcast_to_session_and_agents(
                    session_id,
                    {
                        "type": "typing",
                        "session_id": session_id,
                        "sender": "bot",
                        "is_typing": False,
                    },
                )

    async def escalate_session_by_visitor(
        self,
        session_id: str,
        reason: str = "Customer requested human assistance",
    ) -> Optional[dict[str, Any]]:
        """Explicit visitor escalation (e.g. clicking 'Talk to a human')."""
        lock = self._get_session_lock(session_id)
        async with lock:
            self._cancel_ai_task(session_id)
            updated = await asyncio.to_thread(
                self.db.update_session_state,
                session_id,
                SessionState.WAITING_FOR_HUMAN,
                None,
                reason,
                [SessionState.BOT_ACTIVE],
            )
            if not updated:
                return await self._get_enriched_session_summary(session_id)

            sys_msg = await asyncio.to_thread(
                self.db.add_message,
                session_id,
                "system",
                "You requested a human agent. Someone from our support team will join shortly.",
                {"event": "visitor_escalated"},
            )
            enriched = await self._get_enriched_session_summary(session_id)

        await self.broadcast_to_session_and_agents(
            session_id,
            {
                "type": "message",
                "session_id": session_id,
                "message": sys_msg,
                "session": enriched,
            },
        )
        if enriched:
            await self.broadcast_to_session_and_agents(
                session_id,
                {"type": "state_change", "session": enriched},
            )
            await self.broadcast_to_agents(
                {
                    "type": "escalation",
                    "session": enriched,
                    "reason": reason,
                }
            )
            asyncio.create_task(self._send_escalation_webhook(enriched, reason))
        return enriched

    # ------------------------------------------------------------------
    # Agent Inbound Events & Shared HITL Actions (Used by WS + REST)
    # ------------------------------------------------------------------

    async def handle_agent_event(self, ws: WebSocket, payload: dict[str, Any]) -> None:
        event_type = str(payload.get("type", "")).strip().lower()

        if event_type == "ping":
            await self._safe_send(ws, {"type": "pong"})
            return

        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            await self._safe_send(
                ws,
                {"type": "error", "message": "Missing required field 'session_id'."},
            )
            return

        agent_name = (str(payload.get("agent_name", "Support Agent")).strip() or "Support Agent")[:60]

        if event_type == "take_over":
            result = await self.take_over_session(session_id, agent_name=agent_name)
            if result is None:
                await self._safe_send(ws, {"type": "error", "message": "Session not found."})
            return

        if event_type == "release_to_bot":
            result = await self.release_session_to_bot(session_id, agent_name=agent_name)
            if result is None:
                await self._safe_send(ws, {"type": "error", "message": "Session not found."})
            return

        if event_type == "agent_message":
            content = str(payload.get("content", "")).strip()
            if not content:
                return
            if len(content) > MAX_MESSAGE_LENGTH:
                await self._safe_send(
                    ws,
                    {"type": "error", "message": f"Message exceeds {MAX_MESSAGE_LENGTH} characters."},
                )
                return
            msg = await self.send_agent_message(session_id, content, agent_name=agent_name)
            if msg is None:
                await self._safe_send(ws, {"type": "error", "message": "Session not found."})
            return

        if event_type == "typing":
            await self.send_to_visitor(
                session_id,
                {
                    "type": "typing",
                    "session_id": session_id,
                    "sender": "agent",
                    "agent_name": agent_name,
                    "is_typing": bool(payload.get("is_typing", False)),
                },
            )
            return

        await self._safe_send(
            ws,
            {"type": "error", "message": f"Unsupported agent event '{event_type}'."},
        )

    async def take_over_session(
        self,
        session_id: str,
        agent_name: str = "Support Agent",
    ) -> Optional[dict[str, Any]]:
        """
        Manual agent takeover:
        Immediately cancels any pending AI task, locks session to `HUMAN_ACTIVE`,
        and broadcasts live state + system notification to visitor and dashboards.
        """
        clean_agent = (agent_name or "Support Agent").strip()[:60] or "Support Agent"
        lock = self._get_session_lock(session_id)

        async with lock:
            # Abort any in-flight AI generation immediately
            self._cancel_ai_task(session_id)

            existing = await asyncio.to_thread(self.db.get_session, session_id)
            if not existing:
                return None

            already_human = existing["state"] == SessionState.HUMAN_ACTIVE.value
            updated = await asyncio.to_thread(
                self.db.update_session_state,
                session_id,
                SessionState.HUMAN_ACTIVE,
                clean_agent,
                None,
                None,
            )
            if not updated:
                return None

            sys_msg = None
            if not already_human:
                sys_msg = await asyncio.to_thread(
                    self.db.add_message,
                    session_id,
                    "system",
                    f"{clean_agent} has joined the conversation.",
                    {"event": "take_over", "agent_name": clean_agent},
                )

            enriched = await self._get_enriched_session_summary(session_id)

        await self.broadcast_to_session_and_agents(
            session_id,
            {"type": "typing", "session_id": session_id, "sender": "bot", "is_typing": False},
        )
        if sys_msg is not None:
            await self.broadcast_to_session_and_agents(
                session_id,
                {
                    "type": "message",
                    "session_id": session_id,
                    "message": sys_msg,
                    "session": enriched,
                },
            )
        if enriched:
            await self.broadcast_to_session_and_agents(
                session_id,
                {"type": "state_change", "session": enriched},
            )
        return enriched

    async def release_session_to_bot(
        self,
        session_id: str,
        agent_name: str = "Support Agent",
    ) -> Optional[dict[str, Any]]:
        """Return control of a session from a human agent back to the AI bot."""
        clean_agent = (agent_name or "Support Agent").strip()[:60] or "Support Agent"
        lock = self._get_session_lock(session_id)

        async with lock:
            existing = await asyncio.to_thread(self.db.get_session, session_id)
            if not existing:
                return None

            already_bot = existing["state"] == SessionState.BOT_ACTIVE.value
            updated = await asyncio.to_thread(
                self.db.update_session_state,
                session_id,
                SessionState.BOT_ACTIVE,
                None,
                None,
                None,
            )
            if not updated:
                return None

            sys_msg = None
            if not already_bot:
                sys_msg = await asyncio.to_thread(
                    self.db.add_message,
                    session_id,
                    "system",
                    "Conversation has been handed back to the AI assistant.",
                    {"event": "release_to_bot", "agent_name": clean_agent},
                )

            enriched = await self._get_enriched_session_summary(session_id)

        if sys_msg is not None:
            await self.broadcast_to_session_and_agents(
                session_id,
                {
                    "type": "message",
                    "session_id": session_id,
                    "message": sys_msg,
                    "session": enriched,
                },
            )
        if enriched:
            await self.broadcast_to_session_and_agents(
                session_id,
                {"type": "state_change", "session": enriched},
            )
        return enriched

    async def send_agent_message(
        self,
        session_id: str,
        content: str,
        agent_name: str = "Support Agent",
    ) -> Optional[dict[str, Any]]:
        """Send a human agent reply. Automatically locks to HUMAN_ACTIVE if needed."""
        clean_agent = (agent_name or "Support Agent").strip()[:60] or "Support Agent"
        clean_content = (content or "").strip()
        if not clean_content:
            return None

        lock = self._get_session_lock(session_id)
        state_changed = False

        async with lock:
            existing = await asyncio.to_thread(self.db.get_session, session_id)
            if not existing:
                return None

            # Auto-lock to HUMAN_ACTIVE if agent replies before clicking Take Over
            if existing["state"] != SessionState.HUMAN_ACTIVE.value:
                self._cancel_ai_task(session_id)
                await asyncio.to_thread(
                    self.db.update_session_state,
                    session_id,
                    SessionState.HUMAN_ACTIVE,
                    clean_agent,
                    None,
                    None,
                )
                state_changed = True

            msg = await asyncio.to_thread(
                self.db.add_message,
                session_id,
                "agent",
                clean_content,
                {"agent_name": clean_agent},
            )
            enriched = await self._get_enriched_session_summary(session_id)

        if state_changed and enriched:
            await self.broadcast_to_session_and_agents(
                session_id,
                {"type": "state_change", "session": enriched},
            )

        await self.broadcast_to_session_and_agents(
            session_id,
            {
                "type": "message",
                "session_id": session_id,
                "message": msg,
                "session": enriched,
            },
        )
        return msg

    async def notify_kb_updated(self) -> None:
        """Inform connected dashboards whenever the Knowledge Base changes."""
        kb_stats = await asyncio.to_thread(self.db.get_kb_stats)
        await self.broadcast_to_agents({"type": "kb_updated", "kb_stats": kb_stats})

    async def notify_session_deleted(self, session_id: str) -> None:
        self._cancel_ai_task(session_id)
        self._session_locks.pop(session_id, None)
        await self.broadcast_to_agents({"type": "session_deleted", "session_id": session_id})

    async def _get_enriched_session_summary(self, session_id: str) -> Optional[dict[str, Any]]:
        raw = await asyncio.to_thread(self.db.get_session, session_id)
        return self._enrich_session(raw) if raw else None
