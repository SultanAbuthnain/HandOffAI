"""
SQLite persistence layer for sessions, messages, HITL state machine,
and local vector storage (float32 L2-normalized numpy embeddings).

Designed for zero external database dependencies: all application state and
knowledge-base embeddings live in a single local `app.db` file with WAL mode
enabled and atomic transactions for race-condition prevention.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np


class SessionState(str, Enum):
    """Strict HITL state machine values persisted in SQLite."""

    BOT_ACTIVE = "BOT_ACTIVE"
    WAITING_FOR_HUMAN = "WAITING_FOR_HUMAN"
    HUMAN_ACTIVE = "HUMAN_ACTIVE"


VALID_SENDERS = frozenset({"user", "bot", "agent", "system"})


def _utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _normalize_vector(vec: np.ndarray | list[float]) -> np.ndarray:
    """Convert embedding to a contiguous L2-normalized float32 1D array."""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("Embedding vector cannot be empty.")
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        arr = arr / norm
    return np.ascontiguousarray(arr, dtype=np.float32)


class Database:
    """Thread-safe SQLite database manager for HITL sessions and RAG vectors."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        resolved_path = db_path or os.getenv("DATABASE_PATH", "./app.db")
        self.db_path = Path(resolved_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self._vector_cache_lock = threading.RLock()
        self._vector_cache: dict[
            tuple[Optional[str], Optional[int]],
            tuple[list[dict[str, Any]], np.ndarray],
        ] = {}
        self.init_db()

    def _invalidate_vector_cache(self) -> None:
        with self._vector_cache_lock:
            self._vector_cache.clear()

    def _create_connection(self, autocommit: bool = False) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=15.0,
            isolation_level=None if autocommit else "",
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    @contextmanager
    def _read_conn(self) -> Iterator[sqlite3.Connection]:
        conn = self._create_connection(autocommit=True)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _write_tx(self) -> Iterator[sqlite3.Connection]:
        """Acquire an exclusive SQLite write transaction (BEGIN IMMEDIATE)."""
        with self._write_lock:
            conn = self._create_connection(autocommit=True)
            try:
                conn.execute("BEGIN IMMEDIATE;")
                yield conn
                conn.execute("COMMIT;")
            except Exception:
                try:
                    conn.execute("ROLLBACK;")
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()

    def init_db(self) -> None:
        """Create schema and enable WAL mode for concurrent read/write safety."""
        with self._write_lock:
            conn = self._create_connection(autocommit=True)
            try:
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA synchronous = NORMAL;")
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        state TEXT NOT NULL DEFAULT 'BOT_ACTIVE'
                            CHECK (state IN ('BOT_ACTIVE', 'WAITING_FOR_HUMAN', 'HUMAN_ACTIVE')),
                        visitor_name TEXT NOT NULL DEFAULT 'Visitor',
                        visitor_email TEXT,
                        origin TEXT,
                        client_ip TEXT,
                        assigned_agent TEXT,
                        escalation_reason TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        last_message_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        sender TEXT NOT NULL
                            CHECK (sender IN ('user', 'bot', 'agent', 'system')),
                        content TEXT NOT NULL,
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        source_type TEXT NOT NULL DEFAULT 'text',
                        content_hash TEXT NOT NULL,
                        chunk_count INTEGER NOT NULL DEFAULT 0,
                        char_count INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        embedding_dim INTEGER NOT NULL,
                        embedding_model TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_sessions_state_last_msg
                        ON sessions(state, last_message_at DESC);

                    CREATE INDEX IF NOT EXISTS idx_messages_session_id
                        ON messages(session_id, id ASC);

                    CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                        ON chunks(document_id);

                    CREATE INDEX IF NOT EXISTS idx_chunks_model_dim
                        ON chunks(embedding_model, embedding_dim);

                    CREATE INDEX IF NOT EXISTS idx_documents_content_hash
                        ON documents(content_hash);
                    """
                )
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Row Serialization Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        return data

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> dict[str, Any]:
        meta_raw = row["metadata"]
        metadata: dict[str, Any] = {}
        if meta_raw:
            try:
                metadata = json.loads(meta_raw)
            except json.JSONDecodeError:
                metadata = {}
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "sender": row["sender"],
            "content": row["content"],
            "metadata": metadata,
            "created_at": row["created_at"],
        }

    # ------------------------------------------------------------------
    # Session & HITL State Machine Operations
    # ------------------------------------------------------------------

    def create_session(
        self,
        session_id: Optional[str] = None,
        visitor_name: str = "Visitor",
        visitor_email: Optional[str] = None,
        origin: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> dict[str, Any]:
        sid = (session_id or "").strip() or str(uuid.uuid4())
        now = _utc_now_iso()
        clean_name = (visitor_name or "Visitor").strip()[:80] or "Visitor"

        with self._write_tx() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    id, state, visitor_name, visitor_email, origin, client_ip,
                    assigned_agent, escalation_reason, created_at, updated_at, last_message_at
                )
                VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)
                """,
                (
                    sid,
                    SessionState.BOT_ACTIVE.value,
                    clean_name,
                    visitor_email,
                    origin,
                    client_ip,
                    now,
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
            return self._row_to_session(row)

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        with self._read_conn() as conn:
            row = conn.execute(
                """
                SELECT
                    s.*,
                    (
                        SELECT COUNT(*)
                        FROM messages m
                        WHERE m.session_id = s.id
                    ) AS message_count,
                    (
                        SELECT m.content
                        FROM messages m
                        WHERE m.session_id = s.id
                        ORDER BY m.id DESC
                        LIMIT 1
                    ) AS last_message_preview,
                    (
                        SELECT m.sender
                        FROM messages m
                        WHERE m.session_id = s.id
                        ORDER BY m.id DESC
                        LIMIT 1
                    ) AS last_message_sender
                FROM sessions s
                WHERE s.id = ?
                """,
                (session_id,),
            ).fetchone()
            return self._row_to_session(row) if row else None

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        visitor_name: str = "Visitor",
        visitor_email: Optional[str] = None,
        origin: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> dict[str, Any]:
        sid = (session_id or "").strip() or str(uuid.uuid4())
        now = _utc_now_iso()
        clean_name = (visitor_name or "Visitor").strip()[:80] or "Visitor"

        with self._write_tx() as conn:
            existing = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (sid,),
            ).fetchone()
            if existing:
                # Refresh origin/client_ip if previously unset
                if (origin and not existing["origin"]) or (client_ip and not existing["client_ip"]):
                    conn.execute(
                        """
                        UPDATE sessions
                        SET origin = COALESCE(origin, ?),
                            client_ip = COALESCE(client_ip, ?),
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (origin, client_ip, now, sid),
                    )
                    existing = conn.execute(
                        "SELECT * FROM sessions WHERE id = ?",
                        (sid,),
                    ).fetchone()
                return self._row_to_session(existing)

            conn.execute(
                """
                INSERT INTO sessions (
                    id, state, visitor_name, visitor_email, origin, client_ip,
                    assigned_agent, escalation_reason, created_at, updated_at, last_message_at
                )
                VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)
                """,
                (
                    sid,
                    SessionState.BOT_ACTIVE.value,
                    clean_name,
                    visitor_email,
                    origin,
                    client_ip,
                    now,
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
            return self._row_to_session(row)

    def get_session_state(self, session_id: str) -> Optional[SessionState]:
        """Fast state lookup used right before dispatching AI responses."""
        with self._read_conn() as conn:
            row = conn.execute(
                "SELECT state FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            return SessionState(row["state"]) if row else None

    def update_session_state(
        self,
        session_id: str,
        new_state: SessionState | str,
        assigned_agent: Optional[str] = None,
        escalation_reason: Optional[str] = None,
        expected_states: Optional[Iterable[SessionState | str]] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Atomically transition a session's HITL state.

        If `expected_states` is supplied, the update only succeeds if the
        current persisted state is in `expected_states`. Returns the updated
        session dict, or None if the session does not exist or failed the guard.
        """
        target = SessionState(new_state)
        now = _utc_now_iso()

        with self._write_tx() as conn:
            current = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if not current:
                return None

            if expected_states is not None:
                allowed = {SessionState(s).value for s in expected_states}
                if current["state"] not in allowed:
                    return None

            if target == SessionState.BOT_ACTIVE:
                next_agent = None
                next_reason = None
            elif target == SessionState.HUMAN_ACTIVE:
                next_agent = assigned_agent or current["assigned_agent"] or "Human Agent"
                next_reason = escalation_reason if escalation_reason is not None else current["escalation_reason"]
            else:  # WAITING_FOR_HUMAN
                next_agent = None
                next_reason = escalation_reason or current["escalation_reason"] or "Escalated to human support"

            conn.execute(
                """
                UPDATE sessions
                SET state = ?,
                    assigned_agent = ?,
                    escalation_reason = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (target.value, next_agent, next_reason, now, session_id),
            )
            updated = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            return self._row_to_session(updated)

    def list_sessions(
        self,
        state: Optional[SessionState | str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List sessions ordered by priority:
        WAITING_FOR_HUMAN first, then HUMAN_ACTIVE, then BOT_ACTIVE,
        sorted by most recent activity within each state.
        """
        params: list[Any] = []
        where_clause = ""
        if state is not None:
            where_clause = "WHERE s.state = ?"
            params.append(SessionState(state).value)

        params.extend([max(1, min(limit, 500)), max(0, offset)])

        query = f"""
            SELECT
                s.*,
                (
                    SELECT COUNT(*)
                    FROM messages m
                    WHERE m.session_id = s.id
                ) AS message_count,
                (
                    SELECT m.content
                    FROM messages m
                    WHERE m.session_id = s.id
                    ORDER BY m.id DESC
                    LIMIT 1
                ) AS last_message_preview,
                (
                    SELECT m.sender
                    FROM messages m
                    WHERE m.session_id = s.id
                    ORDER BY m.id DESC
                    LIMIT 1
                ) AS last_message_sender
            FROM sessions s
            {where_clause}
            ORDER BY
                CASE s.state
                    WHEN 'WAITING_FOR_HUMAN' THEN 0
                    WHEN 'HUMAN_ACTIVE' THEN 1
                    ELSE 2
                END ASC,
                s.last_message_at DESC
            LIMIT ? OFFSET ?
        """
        with self._read_conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_session(r) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        with self._write_tx() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Message Operations
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        sender: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if sender not in VALID_SENDERS:
            raise ValueError(f"Invalid sender '{sender}'. Must be one of {sorted(VALID_SENDERS)}.")

        now = _utc_now_iso()
        meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None

        with self._write_tx() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (session_id, sender, content, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, sender, content, meta_json, now),
            )
            conn.execute(
                """
                UPDATE sessions
                SET updated_at = ?, last_message_at = ?
                WHERE id = ?
                """,
                (now, now, session_id),
            )
            row = conn.execute(
                "SELECT * FROM messages WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
            return self._row_to_message(row)

    def add_bot_message_if_active(
        self,
        session_id: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        escalate_to_human: bool = False,
        escalation_reason: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Atomic race-condition guard for AI replies.

        Verifies `state == 'BOT_ACTIVE'` inside an exclusive SQLite write
        transaction right before persisting the bot message (and optionally
        transitioning to `WAITING_FOR_HUMAN`). If a human agent took over
        while the LLM was generating, this returns None without mutating DB.
        """
        now = _utc_now_iso()
        meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None

        with self._write_tx() as conn:
            session_row = conn.execute(
                "SELECT state FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if not session_row or session_row["state"] != SessionState.BOT_ACTIVE.value:
                return None

            cursor = conn.execute(
                """
                INSERT INTO messages (session_id, sender, content, metadata, created_at)
                VALUES (?, 'bot', ?, ?, ?)
                """,
                (session_id, content, meta_json, now),
            )

            if escalate_to_human:
                conn.execute(
                    """
                    UPDATE sessions
                    SET state = ?,
                        escalation_reason = ?,
                        updated_at = ?,
                        last_message_at = ?
                    WHERE id = ?
                    """,
                    (
                        SessionState.WAITING_FOR_HUMAN.value,
                        escalation_reason or "AI escalated conversation to human support",
                        now,
                        now,
                        session_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE sessions
                    SET updated_at = ?, last_message_at = ?
                    WHERE id = ?
                    """,
                    (now, now, session_id),
                )

            row = conn.execute(
                "SELECT * FROM messages WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
            return self._row_to_message(row)

    def get_messages(
        self,
        session_id: str,
        limit: int = 200,
        before_id: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(limit, 1000))
        with self._read_conn() as conn:
            if before_id is not None:
                rows = conn.execute(
                    """
                    SELECT * FROM (
                        SELECT * FROM messages
                        WHERE session_id = ? AND id < ?
                        ORDER BY id DESC
                        LIMIT ?
                    ) sub
                    ORDER BY id ASC
                    """,
                    (session_id, before_id, safe_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM (
                        SELECT * FROM messages
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    ) sub
                    ORDER BY id ASC
                    """,
                    (session_id, safe_limit),
                ).fetchall()
            return [self._row_to_message(r) for r in rows]

    def get_recent_conversation_history(
        self,
        session_id: str,
        limit: int = 8,
    ) -> list[dict[str, str]]:
        """
        Return recent user/bot/agent turns formatted as chat history
        (`{"role": "user" | "assistant", "content": "..."}`) for LLM context.
        """
        safe_limit = max(1, min(limit, 50))
        with self._read_conn() as conn:
            rows = conn.execute(
                """
                SELECT sender, content FROM (
                    SELECT id, sender, content
                    FROM messages
                    WHERE session_id = ? AND sender IN ('user', 'bot', 'agent')
                    ORDER BY id DESC
                    LIMIT ?
                ) sub
                ORDER BY id ASC
                """,
                (session_id, safe_limit),
            ).fetchall()

        history: list[dict[str, str]] = []
        for row in rows:
            role = "user" if row["sender"] == "user" else "assistant"
            history.append({"role": role, "content": row["content"]})
        return history

    # ------------------------------------------------------------------
    # Knowledge Base & Local Vector Store Operations
    # ------------------------------------------------------------------

    def create_document_with_chunks(
        self,
        title: str,
        source_type: str,
        raw_content: str,
        chunk_texts: list[str],
        embeddings: list[np.ndarray],
        embedding_model: str,
    ) -> dict[str, Any]:
        """Persist a document and its L2-normalized chunk embeddings atomically."""
        if len(chunk_texts) != len(embeddings):
            raise ValueError("chunk_texts and embeddings must have the same length.")
        if not chunk_texts:
            raise ValueError("Cannot store a document with zero chunks.")

        doc_id = str(uuid.uuid4())
        now = _utc_now_iso()
        content_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
        clean_title = title.strip()[:200] or "Untitled Document"

        normalized_blobs: list[tuple[bytes, int]] = []
        for vec in embeddings:
            norm_vec = _normalize_vector(vec)
            normalized_blobs.append((norm_vec.tobytes(), int(norm_vec.shape[0])))

        with self._write_tx() as conn:
            # Replace any existing document with identical content_hash and embedding_model
            # so re-ingesting the same file never duplicates chunks in Top-K search.
            existing_docs = conn.execute(
                """
                SELECT DISTINCT d.id
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.content_hash = ?
                  AND (c.embedding_model = ? OR c.embedding_model IS NULL)
                """,
                (content_hash, embedding_model),
            ).fetchall()
            for old_doc in existing_docs:
                conn.execute("DELETE FROM documents WHERE id = ?", (old_doc["id"],))

            conn.execute(
                """
                INSERT INTO documents (
                    id, title, source_type, content_hash, chunk_count, char_count, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    clean_title,
                    source_type,
                    content_hash,
                    len(chunk_texts),
                    len(raw_content),
                    now,
                ),
            )

            conn.executemany(
                """
                INSERT INTO chunks (
                    document_id, chunk_index, content, embedding,
                    embedding_dim, embedding_model, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        doc_id,
                        idx,
                        text,
                        sqlite3.Binary(blob),
                        dim,
                        embedding_model,
                        now,
                    )
                    for idx, (text, (blob, dim)) in enumerate(zip(chunk_texts, normalized_blobs))
                ],
            )

            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            self._invalidate_vector_cache()
            return dict(row)

    def list_documents(self) -> list[dict[str, Any]]:
        with self._read_conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    d.*,
                    (
                        SELECT c.embedding_model
                        FROM chunks c
                        WHERE c.document_id = d.id
                        LIMIT 1
                    ) AS embedding_model
                FROM documents d
                ORDER BY d.created_at DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def get_document(self, document_id: str) -> Optional[dict[str, Any]]:
        with self._read_conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            return dict(row) if row else None

    def delete_document(self, document_id: str) -> bool:
        with self._write_tx() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            deleted = cursor.rowcount > 0
            if deleted:
                self._invalidate_vector_cache()
            return deleted

    def get_all_chunks_for_search(
        self,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        """
        Load all matching chunks and return `(metadata_list, matrix)` where
        `matrix` is a float32 numpy array of shape `(N, D)` ready for cosine
        similarity via dot product. Uses an in-memory cache invalidated on KB changes.
        """
        cache_key = (embedding_model, embedding_dim)
        with self._vector_cache_lock:
            cached = self._vector_cache.get(cache_key)
            if cached is not None:
                return list(cached[0]), cached[1]

        conditions: list[str] = []
        params: list[Any] = []
        if embedding_model is not None:
            conditions.append("c.embedding_model = ?")
            params.append(embedding_model)
        if embedding_dim is not None:
            conditions.append("c.embedding_dim = ?")
            params.append(embedding_dim)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT
                c.id,
                c.document_id,
                c.chunk_index,
                c.content,
                c.embedding,
                c.embedding_dim,
                c.embedding_model,
                d.title AS document_title
            FROM chunks c
            INNER JOIN documents d ON d.id = c.document_id
            {where_clause}
            ORDER BY c.id ASC
        """

        with self._read_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        if not rows:
            empty = np.empty((0, 0), dtype=np.float32)
            with self._vector_cache_lock:
                self._vector_cache[cache_key] = ([], empty)
            return [], empty

        metadata_list: list[dict[str, Any]] = []
        vectors: list[np.ndarray] = []
        expected_dim = embedding_dim or rows[0]["embedding_dim"]

        for row in rows:
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            if vec.shape[0] != expected_dim:
                continue
            vectors.append(vec)
            metadata_list.append(
                {
                    "chunk_id": row["id"],
                    "document_id": row["document_id"],
                    "document_title": row["document_title"],
                    "chunk_index": row["chunk_index"],
                    "content": row["content"],
                }
            )

        if not vectors:
            empty = np.empty((0, 0), dtype=np.float32)
            with self._vector_cache_lock:
                self._vector_cache[cache_key] = ([], empty)
            return [], empty

        matrix = np.vstack(vectors).astype(np.float32, copy=False)
        with self._vector_cache_lock:
            self._vector_cache[cache_key] = (metadata_list, matrix)
        return list(metadata_list), matrix

    def get_kb_stats(self) -> dict[str, Any]:
        with self._read_conn() as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            models = [
                row["embedding_model"]
                for row in conn.execute(
                    "SELECT DISTINCT embedding_model FROM chunks ORDER BY embedding_model"
                ).fetchall()
            ]
        return {
            "document_count": int(doc_count),
            "chunk_count": int(chunk_count),
            "embedding_models": models,
        }
