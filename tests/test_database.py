"""Unit tests for SQLite persistence, atomic HITL state transitions, and local vector store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.database import Database, SessionState, _normalize_vector


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    return Database(db_path=str(tmp_path / "test_hitl.db"))


def test_session_lifecycle_and_enriched_get_session(db: Database) -> None:
    session = db.create_session(
        session_id="sess-100",
        visitor_name="Alice",
        origin="http://localhost:8000",
        client_ip="127.0.0.1",
    )
    assert session["id"] == "sess-100"
    assert session["state"] == SessionState.BOT_ACTIVE.value

    # Before any messages, enriched fields should reflect 0 messages
    fetched_empty = db.get_session("sess-100")
    assert fetched_empty is not None
    assert fetched_empty["message_count"] == 0
    assert fetched_empty["last_message_preview"] is None
    assert fetched_empty["last_message_sender"] is None

    # Add user and bot messages and verify O(1) enriched get_session fields
    db.add_message("sess-100", "user", "What is your refund policy?")
    db.add_message("sess-100", "bot", "We offer a 30-day money-back guarantee.", {"confidence": 0.92})

    fetched = db.get_session("sess-100")
    assert fetched is not None
    assert fetched["message_count"] == 2
    assert fetched["last_message_preview"] == "We offer a 30-day money-back guarantee."
    assert fetched["last_message_sender"] == "bot"


def test_hitl_state_machine_and_expected_states_guard(db: Database) -> None:
    db.create_session(session_id="sess-state", visitor_name="Bob")

    # Guard should reject transition if current state is not in expected_states
    rejected = db.update_session_state(
        "sess-state",
        SessionState.BOT_ACTIVE,
        expected_states=[SessionState.HUMAN_ACTIVE],
    )
    assert rejected is None

    # Escalate BOT_ACTIVE -> WAITING_FOR_HUMAN
    escalated = db.update_session_state(
        "sess-state",
        SessionState.WAITING_FOR_HUMAN,
        escalation_reason="Low confidence",
        expected_states=[SessionState.BOT_ACTIVE],
    )
    assert escalated is not None
    assert escalated["state"] == SessionState.WAITING_FOR_HUMAN.value
    assert escalated["escalation_reason"] == "Low confidence"

    # Take over WAITING_FOR_HUMAN -> HUMAN_ACTIVE
    taken_over = db.update_session_state(
        "sess-state",
        SessionState.HUMAN_ACTIVE,
        assigned_agent="Sarah",
    )
    assert taken_over is not None
    assert taken_over["state"] == SessionState.HUMAN_ACTIVE.value
    assert taken_over["assigned_agent"] == "Sarah"

    # Release HUMAN_ACTIVE -> BOT_ACTIVE clears assigned_agent and escalation_reason
    released = db.update_session_state(
        "sess-state",
        SessionState.BOT_ACTIVE,
    )
    assert released is not None
    assert released["state"] == SessionState.BOT_ACTIVE.value
    assert released["assigned_agent"] is None
    assert released["escalation_reason"] is None


def test_atomic_bot_message_guard_prevents_race_condition(db: Database) -> None:
    db.create_session(session_id="sess-race", visitor_name="Charlie")

    # Bot can reply while BOT_ACTIVE
    msg1 = db.add_bot_message_if_active("sess-race", "Initial grounded answer")
    assert msg1 is not None
    assert msg1["sender"] == "bot"

    # Simulate human takeover while another AI reply is in flight
    db.update_session_state("sess-race", SessionState.HUMAN_ACTIVE, assigned_agent="Agent Dan")

    # Atomic guard must discard the bot message and return None
    discarded = db.add_bot_message_if_active("sess-race", "Late AI reply that must be blocked")
    assert discarded is None

    messages = db.get_messages("sess-race")
    assert len(messages) == 1
    assert messages[0]["content"] == "Initial grounded answer"


def test_vector_normalization_deduplication_and_cache_invalidation(db: Database) -> None:
    vec = _normalize_vector([3.0, 4.0, 0.0, 0.0])
    assert np.isclose(np.linalg.norm(vec), 1.0)
    assert np.allclose(vec, np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32))

    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    doc1 = db.create_document_with_chunks(
        title="Refund Policy",
        source_type="markdown",
        raw_content="Full refund within 30 days.\nCancel anytime.",
        chunk_texts=["Full refund within 30 days.", "Cancel anytime."],
        embeddings=[v1, v2],
        embedding_model="test-embed-v1",
    )
    assert doc1["chunk_count"] == 2

    meta, matrix = db.get_all_chunks_for_search("test-embed-v1", 4)
    assert len(meta) == 2
    assert matrix.shape == (2, 4)

    # Re-ingesting identical raw_content with the same embedding_model replaces duplicate
    doc2 = db.create_document_with_chunks(
        title="Refund Policy v2",
        source_type="markdown",
        raw_content="Full refund within 30 days.\nCancel anytime.",
        chunk_texts=["Full refund within 30 days.", "Cancel anytime."],
        embeddings=[v1, v2],
        embedding_model="test-embed-v1",
    )
    assert doc2["id"] != doc1["id"]
    stats = db.get_kb_stats()
    assert stats["document_count"] == 1
    assert stats["chunk_count"] == 2

    # Deleting the document invalidates cache and cascades chunk deletion
    assert db.delete_document(doc2["id"]) is True
    meta_after, matrix_after = db.get_all_chunks_for_search("test-embed-v1", 4)
    assert meta_after == []
    assert matrix_after.size == 0
