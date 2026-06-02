# conversation_store.py
# Persists every conversation turn to a JSON file.
# Each session is stored as a list of turns under its session_id.
# File is updated after every single message — no data loss on crash.

import json
from datetime import datetime, timezone
from pathlib import Path

CONVERSATIONS_FILE = Path("conversations.json")


def _load() -> dict:
    if CONVERSATIONS_FILE.exists():
        try:
            return json.loads(CONVERSATIONS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save(data: dict) -> None:
    CONVERSATIONS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_turn(
    session_id: str,
    user_query: str,
    bot_answer: str,
    intent: str,
    language: str,
    top_score: float,
    escalated: bool,
) -> None:
    """
    Appends one turn to the JSON store and writes to disk immediately.

    JSON structure:
    {
      "default": [
        {
          "turn": 1,
          "timestamp": "2026-06-03T10:45:00Z",
          "user": "What time is check-in?",
          "bot": "Check-in time is 2:00 PM.",
          "intent": "booking_inquiry",
          "language": "english",
          "top_score": 0.72,
          "escalated": false
        },
        ...
      ]
    }
    """
    data = _load()

    if session_id not in data:
        data[session_id] = []

    turn_number = len(data[session_id]) + 1

    data[session_id].append({
        "turn":      turn_number,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "user":      user_query,
        "bot":       bot_answer,
        "intent":    intent,
        "language":  language,
        "top_score": round(float(top_score), 4),
        "escalated": escalated,
    })

    _save(data)


def clear_session(session_id: str) -> None:
    """Removes all turns for a session from the JSON store."""
    data = _load()
    if session_id in data:
        del data[session_id]
        _save(data)


def get_session(session_id: str) -> list:
    """Returns all turns for a session, or empty list if none."""
    return _load().get(session_id, [])