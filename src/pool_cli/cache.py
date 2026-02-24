from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_cache (
  file_key TEXT PRIMARY KEY NOT NULL,
  vector_json TEXT NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS assignment_cache (
  file_key TEXT PRIMARY KEY NOT NULL,
  payload_json TEXT NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS action_suggestion_cache (
  pool_name TEXT NOT NULL,
  signature TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  updated_at REAL NOT NULL,
  PRIMARY KEY (pool_name, signature)
);

CREATE TABLE IF NOT EXISTS user_cluster_cache (
  signature TEXT PRIMARY KEY NOT NULL,
  payload_json TEXT NOT NULL,
  updated_at REAL NOT NULL
);
"""


@dataclass(slots=True, frozen=True)
class CachedAssignment:
    pool_name: str
    confidence: float
    why: str

    def to_dict(self) -> dict[str, str | float]:
        return {
            "pool_name": self.pool_name,
            "confidence": float(self.confidence),
            "why": self.why,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CachedAssignment":
        return cls(
            pool_name=str(payload.get("pool_name", "Other")),
            confidence=float(payload.get("confidence", 0.0)),
            why=str(payload.get("why", "")),
        )


@dataclass(slots=True, frozen=True)
class CachedActionSuggestion:
    action_title: str
    why: str
    notes: str
    confidence: float

    def to_dict(self) -> dict[str, str | float]:
        return {
            "action_title": self.action_title,
            "why": self.why,
            "notes": self.notes,
            "confidence": float(self.confidence),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CachedActionSuggestion":
        return cls(
            action_title=str(payload.get("action_title", "")).strip(),
            why=str(payload.get("why", "")).strip(),
            notes=str(payload.get("notes", "")).strip(),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(slots=True, frozen=True)
class CachedUserPoolMetadata:
    name: str
    description: str
    action_title: str
    why: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "action_title": self.action_title,
            "why": self.why,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CachedUserPoolMetadata":
        return cls(
            name=str(payload.get("name", "")).strip(),
            description=str(payload.get("description", "")).strip(),
            action_title=str(payload.get("action_title", "")).strip(),
            why=str(payload.get("why", "")).strip(),
        )


class CacheDB:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CacheDB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


    def get_embedding(self, file_key: str) -> list[float] | None:
        row = self._conn.execute(
            "SELECT vector_json FROM embedding_cache WHERE file_key = ?",
            (file_key,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["vector_json"]))
        return [float(value) for value in payload]

    def put_embedding(self, file_key: str, vector: list[float]) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO embedding_cache(file_key, vector_json, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(file_key) DO UPDATE SET
                  vector_json=excluded.vector_json,
                  updated_at=excluded.updated_at
                """,
                (file_key, json.dumps(vector), time.time()),
            )


    def get_assignment(self, file_key: str) -> CachedAssignment | None:
        row = self._conn.execute(
            "SELECT payload_json FROM assignment_cache WHERE file_key = ?",
            (file_key,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload_json"]))
        return CachedAssignment.from_dict(payload)

    def put_assignment(self, file_key: str, assignment: CachedAssignment) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO assignment_cache(file_key, payload_json, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(file_key) DO UPDATE SET
                  payload_json=excluded.payload_json,
                  updated_at=excluded.updated_at
                """,
                (file_key, json.dumps(assignment.to_dict()), time.time()),
            )


    def get_action_suggestion(self, pool_name: str, signature: str) -> CachedActionSuggestion | None:
        row = self._conn.execute(
            "SELECT payload_json FROM action_suggestion_cache WHERE pool_name = ? AND signature = ?",
            (pool_name, signature),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload_json"]))
        return CachedActionSuggestion.from_dict(payload)

    def put_action_suggestion(self, pool_name: str, signature: str, suggestion: CachedActionSuggestion) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO action_suggestion_cache(pool_name, signature, payload_json, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(pool_name, signature) DO UPDATE SET
                  payload_json=excluded.payload_json,
                  updated_at=excluded.updated_at
                """,
                (pool_name, signature, json.dumps(suggestion.to_dict()), time.time()),
            )


    def get_user_cluster_metadata(self, signature: str) -> CachedUserPoolMetadata | None:
        row = self._conn.execute(
            "SELECT payload_json FROM user_cluster_cache WHERE signature = ?",
            (signature,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload_json"]))
        metadata = CachedUserPoolMetadata.from_dict(payload)
        if metadata.name and metadata.description and metadata.action_title:
            return metadata
        return None

    def put_user_cluster_metadata(self, signature: str, metadata: CachedUserPoolMetadata) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO user_cluster_cache(signature, payload_json, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(signature) DO UPDATE SET
                  payload_json=excluded.payload_json,
                  updated_at=excluded.updated_at
                """,
                (signature, json.dumps(metadata.to_dict()), time.time()),
            )
