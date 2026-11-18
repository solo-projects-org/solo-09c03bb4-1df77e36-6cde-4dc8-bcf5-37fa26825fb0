"""Transactional services: revision acceptance and package generation."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from . import config, deid
from .csvutil import Row, canonical_hash, parse_csv
from .database import get_conn, write_lock


class ConflictError(Exception):
    """The same revision number was already accepted with different content."""


class CohortNotFoundError(Exception):
    """The cohort does not exist."""


class RetryRejectedError(Exception):
    """A retry request cannot apply to the cohort's current package.

    Carries the current revision (when known) so callers can tell the data
    team which revision is actually in force instead of silently reviving an
    older, already withdrawn package.
    """

    def __init__(
        self,
        message: str,
        *,
        current_revision: str | None = None,
        package_status: str | None = None,
        http_status: int = 409,
    ) -> None:
        super().__init__(message)
        self.current_revision = current_revision
        self.package_status = package_status
        self.http_status = http_status


class StaleRevisionError(Exception):
    """The package being finalised no longer belongs to the current revision."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _cohort_dir(cohort_id: str) -> Path:
    return config.UPLOAD_DIR / cohort_id


def accept_revision(cohort_id: str, revision_label: str, rows: list[Row], raw: bytes) -> dict:
    """Validate and accept a new revision.

    Rules implemented here:

    * same (cohort, revision) + same content hash -> idempotent no-op, the
      existing package keeps its current state;
    * same (cohort, revision) + different hash -> :class:`ConflictError`
      (HTTP 409), nothing is changed;
    * a new revision supersedes every non-terminal package of the cohort and
      creates a fresh ``pending`` package, so accepting a revision immediately
      withdraws the current eligibility of the old package;
    * new subjects receive stable pseudonyms and per-subject date shifts,
      existing subjects keep theirs across revisions.
    """
    content_hash = canonical_hash(rows)
    now = _utcnow()
    tmp_path: Path | None = None

    with write_lock:
        conn = get_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT content_hash FROM revisions WHERE cohort_id = ? AND revision = ?",
                (cohort_id, revision_label),
            ).fetchone()
            if existing is not None:
                if existing["content_hash"] == content_hash:
                    conn.execute("COMMIT")
                    return {
                        "status": "duplicate",
                        "revision": revision_label,
                        "content_hash": content_hash,
                    }
                raise ConflictError(
                    f"修订 {revision_label} 已存在且内容不同；同修订号的内容冲突已拒绝"
                )

            cohort = conn.execute(
                "SELECT cohort_id, salt FROM cohorts WHERE cohort_id = ?",
                (cohort_id,),
            ).fetchone()
            if cohort is None:
                salt = deid.new_salt()
                conn.execute(
                    "INSERT INTO cohorts (cohort_id, salt, created_at) VALUES (?, ?, ?)",
                    (cohort_id, salt, now),
                )
            else:
                salt = cohort["salt"]

            # Assign stable identities to any subjects not seen before.
            max_idx_row = conn.execute(
                "SELECT COALESCE(MAX(CAST(SUBSTR(pseudonym, 2) AS INTEGER)), 0) AS m "
                "FROM subjects WHERE cohort_id = ?",
                (cohort_id,),
            ).fetchone()
            next_index = max_idx_row["m"] + 1
            for row in rows:
                known = conn.execute(
                    "SELECT 1 FROM subjects WHERE cohort_id = ? AND subject_id = ?",
                    (cohort_id, row.subject_id),
                ).fetchone()
                if known is None:
                    conn.execute(
                        "INSERT INTO subjects (cohort_id, subject_id, pseudonym, date_shift_days) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            cohort_id,
                            row.subject_id,
                            deid.pseudonym(next_index),
                            deid.subject_date_shift(salt, row.subject_id),
                        ),
                    )
                    next_index += 1

            raw_dir = _cohort_dir(cohort_id)
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / f"rev{revision_label}.csv"
            tmp_path = raw_path.with_suffix(".csv.tmp")
            tmp_path.write_bytes(raw)
            tmp_path.replace(raw_path)

            conn.execute(
                "INSERT INTO revisions (cohort_id, revision, content_hash, row_count, raw_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (cohort_id, revision_label, content_hash, len(rows), str(raw_path), now),
            )

            # Withdraw current eligibility of every older package, including
            # one already published: accepting a new revision immediately
            # retires the previously downloadable package.
            conn.execute(
                "UPDATE packages SET status = 'superseded', updated_at = ? "
                "WHERE cohort_id = ? AND status IN ('pending', 'generating', 'failed', 'published')",
                (now, cohort_id),
            )
            conn.execute(
                "INSERT INTO packages (cohort_id, revision, status, created_at, updated_at) "
                "VALUES (?, ?, 'pending', ?, ?)",
                (cohort_id, revision_label, now, now),
            )
            conn.execute(
                "UPDATE cohorts SET current_revision = ? WHERE cohort_id = ?",
                (revision_label, cohort_id),
            )
            conn.execute("COMMIT")
            return {
                "status": "accepted",
                "revision": revision_label,
                "content_hash": content_hash,
                "rows": len(rows),
            }
        except Exception:
            conn.execute("ROLLBACK")
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise
        finally:
            conn.close()


def rearm_current_package(cohort_id: str, expected_revision: str | None = None) -> dict:
    """Re-arm the current revision's package if — and only if — it truly failed.

    The read of ``cohorts.current_revision`` and the state change happen inside
    one ``BEGIN IMMEDIATE`` transaction guarded by the process write lock, so a
    new revision accepted just before or just after this call cannot be
    overtaken by a retry aimed at an older package:

    * the UPDATE itself is conditional on the row still being the cohort's
      current revision and still ``failed``;
    * if the caller tells us which revision it saw (``expected_revision``), a
      mismatch is rejected with the current revision in the message and
      nothing is modified — a stale request never revives an old package and
      never reports success;
    * a current package that is not ``failed`` is reported as such, again
      without touching it.
    """
    with write_lock:
        conn = get_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cohort = conn.execute(
                "SELECT current_revision FROM cohorts WHERE cohort_id = ?",
                (cohort_id,),
            ).fetchone()
            if cohort is None:
                conn.execute("COMMIT")
                raise CohortNotFoundError(cohort_id)

            current = cohort["current_revision"]
            current_label = None if current is None else str(current)
            if current_label is None:
                conn.execute("COMMIT")
                raise RetryRejectedError(
                    "队列尚未受理任何修订，没有可重试的包",
                    http_status=409,
                )

            if expected_revision is not None and str(expected_revision) != current_label:
                conn.execute("COMMIT")
                raise RetryRejectedError(
                    f"请求重试的修订 {expected_revision} 已不是现行修订，"
                    f"现行修订为 {current_label}；旧包保持已撤下，未重新排队",
                    current_revision=current_label,
                    http_status=409,
                )

            pkg = conn.execute(
                "SELECT status FROM packages WHERE cohort_id = ? AND revision = ?",
                (cohort_id, current),
            ).fetchone()
            if pkg is None:
                conn.execute("COMMIT")
                raise RetryRejectedError(
                    f"现行修订 {current_label} 没有待生成的包",
                    current_revision=current_label,
                    http_status=404,
                )

            status = pkg["status"]
            if status == "published":
                conn.execute("COMMIT")
                return {"status": "published", "revision": current_label}
            if status != "failed":
                conn.execute("COMMIT")
                raise RetryRejectedError(
                    f"现行修订 {current_label} 的包状态为 {status}，无需重试",
                    current_revision=current_label,
                    package_status=status,
                    http_status=409,
                )

            cur = conn.execute(
                """
                UPDATE packages
                   SET status = 'pending', next_retry_at = NULL, updated_at = ?
                 WHERE cohort_id = ? AND revision = ? AND status = 'failed'
                   AND revision = (
                       SELECT current_revision FROM cohorts WHERE cohort_id = ?
                   )
                """,
                (_utcnow(), cohort_id, current, cohort_id),
            )
            if cur.rowcount != 1:
                # Defensive: the guarded UPDATE means a revision accepted
                # concurrently wins; report the state instead of reviving.
                latest = conn.execute(
                    "SELECT current_revision FROM cohorts WHERE cohort_id = ?",
                    (cohort_id,),
                ).fetchone()
                conn.execute("COMMIT")
                raise RetryRejectedError(
                    "重试请求已过期：现行修订在受理过程中发生变化，旧包未重新排队",
                    current_revision=str(latest["current_revision"])
                    if latest and latest["current_revision"] is not None
                    else current_label,
                    http_status=409,
                )
            conn.execute("COMMIT")
            return {"status": "pending", "revision": current_label}
        except (CohortNotFoundError, RetryRejectedError):
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()


def claim_pending_package() -> tuple[str, str] | None:
    """Atomically claim one due package for building (``pending`` or a
    ``failed`` package whose backoff has elapsed).

    Only a package that is still its cohort's *current* revision can be
    claimed. A non-terminal row left behind for an older revision (for example
    by a crash between the supersede commit and the worker's next tick) is
    withdrawn to ``superseded`` in the same transaction and never built.
    """
    now = _utcnow()
    with write_lock:
        conn = get_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Withdraw any non-terminal package whose revision is no longer
            # current before looking for work. This also covers rows that were
            # ``failed`` and whose retry time had elapsed.
            conn.execute(
                """
                UPDATE packages
                   SET status = 'superseded', next_retry_at = NULL, updated_at = ?
                 WHERE status IN ('pending', 'generating', 'failed')
                   AND revision IS NOT (
                       SELECT c.current_revision FROM cohorts c
                        WHERE c.cohort_id = packages.cohort_id
                   )
                """,
                (now,),
            )
            row = conn.execute(
                """
                SELECT p.cohort_id, p.revision FROM packages p
                 WHERE (p.status = 'pending'
                    OR (p.status = 'failed' AND (p.next_retry_at IS NULL OR p.next_retry_at <= ?)))
                   AND p.revision IS (
                       SELECT c.current_revision FROM cohorts c
                        WHERE c.cohort_id = p.cohort_id
                   )
                 ORDER BY p.updated_at
                 LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            cur = conn.execute(
                """
                UPDATE packages
                   SET status = 'generating', attempts = attempts + 1, updated_at = ?
                 WHERE cohort_id = ? AND revision = ?
                   AND status IN ('pending', 'failed')
                   AND revision = (
                       SELECT current_revision FROM cohorts WHERE cohort_id = ?
                   )
                """,
                (now, row["cohort_id"], row["revision"], row["cohort_id"]),
            )
            if cur.rowcount != 1:
                conn.execute("COMMIT")
                return None
            conn.execute("COMMIT")
            return row["cohort_id"], row["revision"]
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()


def finalize_package(cohort_id: str, revision_label: str) -> None:
    """Build the de-identified package and publish it.

    The build runs outside any database transaction; the state is checked
    twice:

    1. before doing any work, the claimed revision must still be the cohort's
       current revision, otherwise nothing is built;
    2. the final UPDATE is conditional on the package still being
       ``generating`` and the revision still being current for the cohort.

    A late build for a revision that was superseded while the worker was busy
    is discarded instead of overwriting the newer revision's state, and any
    file it might have written is removed so it can never be downloaded.
    """
    conn = get_conn()
    try:
        current = conn.execute(
            "SELECT current_revision FROM cohorts WHERE cohort_id = ?",
            (cohort_id,),
        ).fetchone()
        if current is None or str(current["current_revision"]) != str(revision_label):
            raise StaleRevisionError(
                f"修订 {revision_label} 已不是队列当前修订，跳过生成"
            )
        claimed = conn.execute(
            "SELECT status FROM packages WHERE cohort_id = ? AND revision = ?",
            (cohort_id, revision_label),
        ).fetchone()
        if claimed is None or claimed["status"] != "generating":
            raise StaleRevisionError(
                f"修订 {revision_label} 的包未被领取（状态: "
                f"{claimed['status'] if claimed else 'missing'}），跳过生成"
            )

        revision = conn.execute(
            "SELECT * FROM revisions WHERE cohort_id = ? AND revision = ?",
            (cohort_id, revision_label),
        ).fetchone()
        if revision is None:
            raise RuntimeError("revision row disappeared")

        identity_rows = conn.execute(
            "SELECT subject_id, pseudonym, date_shift_days FROM subjects WHERE cohort_id = ?",
            (cohort_id,),
        ).fetchall()
        identity = {
            r["subject_id"]: (r["pseudonym"], r["date_shift_days"]) for r in identity_rows
        }

        raw = Path(revision["raw_path"]).read_bytes()
        rows: list[Row] = parse_csv(raw)
        payload, _manifest = deid.build_package(rows, identity)

        pkg_dir = config.PACKAGE_DIR / cohort_id
        pkg_dir.mkdir(parents=True, exist_ok=True)
        package_path = pkg_dir / f"rev{revision_label}_deid.zip"
        deid.write_package_atomic(package_path, payload)

        now = _utcnow()
        with write_lock:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                """
                UPDATE packages
                   SET status = 'published',
                       package_path = ?,
                       last_error = NULL,
                       next_retry_at = NULL,
                       updated_at = ?
                 WHERE cohort_id = ? AND revision = ?
                   AND status = 'generating'
                   AND revision = (SELECT current_revision FROM cohorts WHERE cohort_id = ?)
                """,
                (str(package_path), now, cohort_id, revision_label, cohort_id),
            )
            if cur.rowcount == 0:
                conn.execute("COMMIT")
                # Late build: a newer revision superseded this one. Discard the
                # artifact so it can never be mistaken for the current package.
                package_path.unlink(missing_ok=True)
                state = conn.execute(
                    "SELECT status FROM packages WHERE cohort_id = ? AND revision = ?",
                    (cohort_id, revision_label),
                ).fetchone()
                raise StaleRevisionError(
                    f"修订 {revision_label} 的迟到生成结果已丢弃 (当前状态: "
                    f"{state['status'] if state else 'missing'})"
                )
            conn.execute("COMMIT")
    finally:
        conn.close()


def mark_failed(cohort_id: str, revision_label: str, error: str) -> None:
    """Record a build failure with capped exponential backoff.

    A failure stays visible as to-be-generated: the worker retries it, a
    restart retries it immediately, and it can never be downloaded. The UPDATE
    only touches a package that is still ``generating`` *and* still belongs to
    the cohort's current revision, so a failure reported late for an already
    superseded revision cannot revive the withdrawn package.
    """
    delay = min(60, 2 ** min(_attempts(cohort_id, revision_label), 6))
    next_retry = datetime.now(timezone.utc).timestamp() + delay
    next_retry_text = datetime.fromtimestamp(next_retry, timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    with write_lock:
        conn = get_conn()
        try:
            conn.execute(
                """
                UPDATE packages
                   SET status = 'failed',
                       last_error = ?,
                       next_retry_at = ?,
                       updated_at = ?
                 WHERE cohort_id = ? AND revision = ? AND status = 'generating'
                   AND revision = (
                       SELECT current_revision FROM cohorts WHERE cohort_id = ?
                   )
                """,
                (error[:1000], next_retry_text, _utcnow(), cohort_id, revision_label, cohort_id),
            )
        finally:
            conn.close()


def _attempts(cohort_id: str, revision_label: str) -> int:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT attempts FROM packages WHERE cohort_id = ? AND revision = ?",
            (cohort_id, revision_label),
        ).fetchone()
        return row["attempts"] if row else 0
    finally:
        conn.close()
