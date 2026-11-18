"""SQLite persistence layer: schema, connection helpers and startup recovery."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone

from . import config

# Serialises the small set of multi-statement write transactions
# (revision acceptance). SQLite itself also serialises writers, but the lock
# makes the read-modify-write sequences below explicit and easy to reason about.
write_lock = threading.RLock()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(
        config.DB_PATH,
        timeout=30.0,
        isolation_level=None,  # autocommit; transactions are managed explicitly
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS cohorts (
    cohort_id        TEXT PRIMARY KEY,
    salt             TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    current_revision INTEGER
);

CREATE TABLE IF NOT EXISTS revisions (
    cohort_id        TEXT NOT NULL REFERENCES cohorts(cohort_id),
    revision         INTEGER NOT NULL,
    content_hash     TEXT NOT NULL,
    row_count        INTEGER NOT NULL,
    raw_path         TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (cohort_id, revision)
);

CREATE TABLE IF NOT EXISTS subjects (
    cohort_id        TEXT NOT NULL,
    subject_id       TEXT NOT NULL,
    pseudonym        TEXT NOT NULL,
    date_shift_days  INTEGER NOT NULL,
    PRIMARY KEY (cohort_id, subject_id),
    UNIQUE (cohort_id, pseudonym)
);

CREATE TABLE IF NOT EXISTS packages (
    cohort_id        TEXT NOT NULL,
    revision         INTEGER NOT NULL,
    status           TEXT NOT NULL,
    attempts         INTEGER NOT NULL DEFAULT 0,
    last_error       TEXT,
    next_retry_at    TEXT,
    package_path     TEXT,
    manifest_path    TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    PRIMARY KEY (cohort_id, revision)
);

CREATE INDEX IF NOT EXISTS idx_packages_claim
    ON packages(status, next_retry_at);
"""


def init_db() -> None:
    with write_lock:
        conn = get_conn()
        try:
            conn.executescript(SCHEMA)
        finally:
            conn.close()


def reset_interrupted_packages() -> int:
    """Startup recovery rule.

    A package left in ``generating`` means the process died mid-build. Only
    packages that still belong to their cohort's current revision may go back
    to the queue: they are reset to ``pending``, retried immediately (backoff
    cleared) and never served half-written. A ``failed`` package is treated the
    same way, keeping its error note in the database.

    Any non-terminal package whose revision is no longer current (it is
    superseded, but a crash or an older build may have left it pending /
    generating / failed) is withdrawn instead of being re-queued. This is what
    guarantees an old package stays withdrawn across a restart.
    """
    with write_lock:
        conn = get_conn()
        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            conn.execute("BEGIN IMMEDIATE")
            stale = conn.execute(
                """
                UPDATE packages
                   SET status = 'superseded',
                       next_retry_at = NULL,
                       updated_at = ?
                 WHERE status IN ('pending', 'generating', 'failed')
                   AND revision IS NOT (
                       SELECT c.current_revision FROM cohorts c
                        WHERE c.cohort_id = packages.cohort_id
                   )
                """,
                (now,),
            )
            current = conn.execute(
                """
                UPDATE packages
                   SET status = 'pending',
                       next_retry_at = NULL,
                       attempts = 0,
                       updated_at = ?
                 WHERE status IN ('generating', 'failed')
                   AND revision IS (
                       SELECT c.current_revision FROM cohorts c
                        WHERE c.cohort_id = packages.cohort_id
                   )
                """,
                (now,),
            )
            conn.execute("COMMIT")
            return stale.rowcount + current.rowcount
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
