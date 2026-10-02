"""Service-owned durable snapshots, replayable events, and idempotent commands."""
import json
import os
from pathlib import Path
import sqlite3
import threading
import time


class Storage:
    def __init__(self, root: Path):
        self.root = root
        (root / "sessions").mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(root / "app.sqlite3", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
          CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, snapshot TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS events(session_id TEXT, seq INTEGER, kind TEXT, body TEXT,
              PRIMARY KEY(session_id, seq));
          CREATE TABLE IF NOT EXISTS commands(session_id TEXT, key TEXT, operation TEXT, response TEXT,
              PRIMARY KEY(session_id, key));
          CREATE TABLE IF NOT EXISTS frames(session_id TEXT, attempt TEXT, epoch TEXT, sequence INTEGER,
              digest TEXT, accepted INTEGER, PRIMARY KEY(session_id, attempt, epoch, sequence));
        """
        )
        self.db.commit()
        self.db.execute("PRAGMA optimize")

    def persist(self, state, kind, body=None):
        with self.lock, self.db:
            sid = state["id"]
            seq = self.db.execute(
                "SELECT COALESCE(MAX(seq),0)+1 FROM events WHERE session_id=?", (sid,)
            ).fetchone()[0]
            event = {"seq": seq, "type": kind, "data": body or {}, "time": time.time()}
            state["event_seq"] = seq
            self.db.execute(
                "INSERT OR REPLACE INTO sessions VALUES(?,?)", (sid, json.dumps(state))
            )
            self.db.execute(
                "INSERT INTO events VALUES(?,?,?,?)",
                (sid, seq, kind, json.dumps(event)),
            )
            return event

    def snapshot(self, sid):
        with self.lock:
            row = self.db.execute("SELECT snapshot FROM sessions WHERE id=?", (sid,)).fetchone()
            return json.loads(row[0]) if row else None

    def snapshots(self):
        with self.lock:
            return [
                json.loads(r[0])
                for r in self.db.execute(
                    "SELECT snapshot FROM sessions ORDER BY rowid DESC"
                )
            ]

    def events(self, sid, after):
        with self.lock:
            return [
                json.loads(r[0])
                for r in self.db.execute(
                    "SELECT body FROM events WHERE session_id=? AND seq>? ORDER BY seq LIMIT 200",
                    (sid, after),
                )
            ]

    def command(self, sid, key, operation, result=None):
        with self.lock, self.db:
            row = self.db.execute(
                "SELECT operation,response FROM commands WHERE session_id=? AND key=?",
                (sid, key),
            ).fetchone()
            if row:
                if row[0] != operation:
                    raise ValueError(
                        "Idempotency key already used for a different command"
                    )
                return json.loads(row[1])
            if result is not None:
                self.db.execute(
                    "INSERT INTO commands VALUES(?,?,?,?)",
                    (sid, key, operation, json.dumps(result)),
                )
            return None

    def frame(self, sid, attempt, epoch, sequence, digest=None, accepted=None):
        with self.lock, self.db:
            row = self.db.execute(
                "SELECT digest,accepted FROM frames WHERE session_id=? AND attempt=? AND epoch=? AND sequence=?",
                (sid, attempt, epoch, sequence),
            ).fetchone()
            if row:
                if digest is not None and row[0] != digest:
                    raise ValueError("A duplicate sequence has different audio")
                return row[1]
            if digest is not None and accepted is not None:
                self.db.execute(
                    "INSERT INTO frames VALUES(?,?,?,?,?,?)",
                    (sid, attempt, epoch, sequence, digest, accepted),
                )
            return None


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)
