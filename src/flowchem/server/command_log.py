"""Local SQLite audit trail of Flowchem HTTP commands, grouped by run_id."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

_MAX_BODY_CHARS = 4000


def _truncate(value: str | None) -> str | None:
    if value is None or len(value) <= _MAX_BODY_CHARS:
        return value
    return value[:_MAX_BODY_CHARS] + f"...<truncated {len(value) - _MAX_BODY_CHARS} chars>"


class CommandLogger:
    """SQLite-backed run_id/command audit trail for one Flowchem server instance."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                label TEXT,
                manifest TEXT,
                started_at REAL NOT NULL,
                ended_at REAL,
                status TEXT NOT NULL DEFAULT 'running'
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS command_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                timestamp REAL NOT NULL,
                method TEXT NOT NULL,
                path TEXT NOT NULL,
                query_params TEXT,
                request_body TEXT,
                status_code INTEGER,
                response_body TEXT,
                duration_ms REAL,
                error TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_command_log_run_id ON command_log(run_id)"
        )
        self._conn.commit()

    def start_run(self, label: str, manifest: dict[str, Any] | None = None) -> str:
        run_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO runs (run_id, label, manifest, started_at, status) "
            "VALUES (?, ?, ?, ?, 'running')",
            (run_id, label, json.dumps(manifest) if manifest else None, time.time()),
        )
        self._conn.commit()
        return run_id

    def end_run(self, run_id: str, status: str = "completed") -> bool:
        cursor = self._conn.execute(
            "UPDATE runs SET ended_at = ?, status = ? WHERE run_id = ?",
            (time.time(), status, run_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT run_id, label, manifest, started_at, ended_at, status "
            "FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row[0],
            "label": row[1],
            "manifest": json.loads(row[2]) if row[2] else None,
            "started_at": row[3],
            "ended_at": row[4],
            "status": row[5],
        }

    def get_run_commands(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT timestamp, method, path, query_params, request_body,
                   status_code, response_body, duration_ms, error
            FROM command_log WHERE run_id = ? ORDER BY id
            """,
            (run_id,),
        ).fetchall()
        return [
            {
                "timestamp": r[0],
                "method": r[1],
                "path": r[2],
                "query_params": r[3],
                "request_body": r[4],
                "status_code": r[5],
                "response_body": r[6],
                "duration_ms": r[7],
                "error": r[8],
            }
            for r in rows
        ]

    def log_command(
        self,
        run_id: str | None,
        method: str,
        path: str,
        query_params: str | None,
        request_body: str | None,
        status_code: int | None,
        response_body: str | None,
        duration_ms: float,
        error: str | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO command_log
                (run_id, timestamp, method, path, query_params, request_body,
                 status_code, response_body, duration_ms, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                time.time(),
                method,
                path,
                query_params,
                _truncate(request_body),
                status_code,
                _truncate(response_body),
                duration_ms,
                error,
            ),
        )
        self._conn.commit()
