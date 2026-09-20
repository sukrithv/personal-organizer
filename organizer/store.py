"""
store.py — durable state in SQLite (Phase 2).

TransactionLog   every applied move: queryable history + cross-session undo.
CorrectionLog    labeled examples (features -> the folder you actually wanted).
                 This is the training data Phase 3 consumes.

Replaces the Phase 1 JSON journal. Old .json journals under the state dir are no
longer read by undo; the DB is the source of truth from here on.
"""

from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path

STATE_DIR = Path.home() / ".file-organizer"
DEFAULT_DB = STATE_DIR / "organizer.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS actions (
    id         INTEGER PRIMARY KEY,
    ts         TEXT NOT NULL,
    session    TEXT NOT NULL,
    src        TEXT NOT NULL,
    dst        TEXT NOT NULL,
    source     TEXT,
    confidence REAL,
    reason     TEXT,
    undone     INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS corrections (
    id        INTEGER PRIMARY KEY,
    ts        TEXT NOT NULL,
    features  TEXT NOT NULL,
    label     TEXT NOT NULL,
    was       TEXT
);
"""


def _connect(db_path):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


class TransactionLog:
    def __init__(self, db_path=DEFAULT_DB):
        self.conn = _connect(db_path)
        self.session = time.strftime("%Y%m%d-%H%M%S")

    def record(self, move) -> None:
        self.conn.execute(
            "INSERT INTO actions(ts, session, src, dst, source, confidence, reason) "
            "VALUES (?,?,?,?,?,?,?)",
            (time.strftime("%Y-%m-%dT%H:%M:%S"), self.session,
             str(move.src), str(move.dst),
             getattr(move, "source", None), getattr(move, "confidence", None),
             move.reason))
        self.conn.commit()

    def last_session_id(self) -> str | None:
        row = self.conn.execute(
            "SELECT session FROM actions WHERE undone=0 "
            "ORDER BY id DESC LIMIT 1").fetchone()
        return row["session"] if row else None

    def undo_last(self, log=print) -> int:
        session = self.last_session_id()
        if not session:
            log("No runs to undo.")
            return 0
        rows = self.conn.execute(
            "SELECT * FROM actions WHERE session=? AND undone=0 ORDER BY id DESC",
            (session,)).fetchall()
        restored = 0
        for r in rows:
            src, dst = Path(r["src"]), Path(r["dst"])
            if dst.exists() and not src.exists():
                try:
                    src.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(dst), str(src))
                    restored += 1
                except OSError as e:
                    log(f"failed: {dst.name}: {e}")
            self.conn.execute("UPDATE actions SET undone=1 WHERE id=?", (r["id"],))
        self.conn.commit()
        self._prune_dirs([Path(r["dst"]).parent for r in rows])
        log(f"Restored {restored}/{len(rows)} file(s) from run {session}.")
        return restored

    @staticmethod
    def _prune_dirs(dirs) -> None:
        for d in sorted(set(dirs), key=lambda p: len(p.parts), reverse=True):
            p = d
            while p.exists() and p.is_dir() and not any(p.iterdir()):
                try:
                    p.rmdir(); p = p.parent
                except OSError:
                    break

    def history(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM actions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    def source_stats(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT source, COUNT(*) n, AVG(confidence) avg_conf "
            "FROM actions GROUP BY source ORDER BY n DESC").fetchall()


class CorrectionLog:
    def __init__(self, db_path=DEFAULT_DB):
        self.conn = _connect(db_path)

    def record(self, features: str, label: str, was: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO corrections(ts, features, label, was) VALUES (?,?,?,?)",
            (time.strftime("%Y-%m-%dT%H:%M:%S"), features, label, was))
        self.conn.commit()

    def export(self):
        for r in self.conn.execute("SELECT features, label FROM corrections"):
            yield r["features"], r["label"]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) c FROM corrections").fetchone()["c"]
