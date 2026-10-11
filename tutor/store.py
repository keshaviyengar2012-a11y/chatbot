"""SQLite storage for student profiles and teacher-provided course material."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .profile import StudentProfile

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS materials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    text TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def save_student(self, profile: StudentProfile) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO students (id, data) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET data = excluded.data",
                (profile.id, json.dumps(profile.to_dict())),
            )

    def get_student(self, student_id: str) -> StudentProfile | None:
        row = self._conn.execute("SELECT data FROM students WHERE id = ?", (student_id,)).fetchone()
        return StudentProfile.from_dict(json.loads(row[0])) if row else None

    def add_material(self, student_id: str, title: str, text: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO materials (student_id, title, text) VALUES (?, ?, ?)",
                (student_id, title, text),
            )
        return cur.lastrowid

    def list_materials(self, student_id: str) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT title, text FROM materials WHERE student_id = ? ORDER BY id", (student_id,)
        ).fetchall()
        return [(title, text) for title, text in rows]

    def close(self) -> None:
        self._conn.close()
