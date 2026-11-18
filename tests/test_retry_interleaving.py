"""Race tests: retrying a failed package vs. accepting a newer revision.

The failure reported by the data team's browser may be stale by the time the
retry request arrives (or may arrive while the worker has just claimed the old
package). In every ordering the withdrawn old package must stay withdrawn, may
never be built or published, and a retry aimed at it must not report success.
"""
from __future__ import annotations

import io

import pytest


def _csv_bytes(rows: list[tuple]) -> bytes:
    buf = io.StringIO()
    buf.write("subject_id,visit_date,value\n")
    for r in rows:
        buf.write(",".join(str(x) for x in r) + "\n")
    return buf.getvalue().encode("utf-8")


def _upload(client, cohort: str, revision: str, payload: bytes):
    return client.post(
        f"/api/cohorts/{cohort}/revisions/{revision}",
        files={"file": ("rev.csv", payload, "text/csv")},
    )


ROWS_1 = [
    ("P001", "2026-01-10", 12.5),
    ("P002", "2026-01-15", 7.2),
]
ROWS_2 = [
    ("P001", "2026-02-10", 13.5),
    ("P002", "2026-02-15", 8.2),
    ("P003", "2026-02-20", 9.0),
]


def _pkg_states(client, cohort: str) -> dict[str, str]:
    status = client.get(f"/api/cohorts/{cohort}").json()
    return {p["revision"]: p["status"] for p in status["packages"]}


def _make_rev_fail(app_env, monkeypatch, cohort: str = "cohort-r") -> None:
    """Upload rev1 and let exactly one build attempt fail."""
    import app.worker as worker_mod

    services = app_env["modules"]["services"]
    real_finalize = services.finalize_package
    calls = {"n": 0}

    def fail_once(cohort_id, revision):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated build failure")
        return real_finalize(cohort_id, revision)

    monkeypatch.setattr(worker_mod, "finalize_package", fail_once)
    assert _upload(app_env["client"], cohort, "1", _csv_bytes(ROWS_1)).status_code == 202
    assert app_env["tick"]() is True
    assert _pkg_states(app_env["client"], cohort)["1"] == "failed"


def test_stale_retry_after_acceptance_is_rejected_and_old_stays_withdrawn(
    app_env, monkeypatch
):
    client = app_env["client"]
    _make_rev_fail(app_env, monkeypatch)

    # The browser rendered rev1 failed; meanwhile rev2 is accepted ("受理" wins).
    assert _upload(client, "cohort-r", "2", _csv_bytes(ROWS_2)).status_code == 202
    assert _pkg_states(client, "cohort-r") == {"1": "superseded", "2": "pending"}

    resp = client.post("/api/cohorts/cohort-r/retry", json={"revision": "1"})
    assert resp.status_code == 409
    body = resp.json()
    assert body["current_revision"] == "2"
    assert "2" in body["detail"]
    # No success is reported and the old package is untouched.
    assert "status" not in body or body.get("status") != "pending"
    assert _pkg_states(client, "cohort-r") == {"1": "superseded", "2": "pending"}

    # The new revision still builds and downloads normally.
    assert app_env["tick"]() is True
    assert _pkg_states(client, "cohort-r") == {"1": "superseded", "2": "published"}
    dl = client.get("/api/cohorts/cohort-r/package")
    assert dl.status_code == 200
    assert f"cohort-r_rev2_deid.zip" in dl.headers.get("content-disposition", "")
    assert client.get("/api/cohorts/cohort-r").json()["current_revision"] == "2"


def test_retry_without_hint_cannot_revive_superseded_package(app_env, monkeypatch):
    client = app_env["client"]
    _make_rev_fail(app_env, monkeypatch)
    assert _upload(client, "cohort-r", "2", _csv_bytes(ROWS_2)).status_code == 202

    # No revision hint: the retry sees rev2 (pending), never rev1 (superseded).
    resp = client.post("/api/cohorts/cohort-r/retry")
    assert resp.status_code == 409
    assert resp.json()["current_revision"] == "2"
    assert resp.json()["package_status"] == "pending"
    assert _pkg_states(client, "cohort-r") == {"1": "superseded", "2": "pending"}

    assert app_env["tick"]() is True
    assert _pkg_states(client, "cohort-r") == {"1": "superseded", "2": "published"}
    assert client.get("/api/cohorts/cohort-r/package").status_code == 200


def test_retry_completes_first_then_acceptance_withdraws_it_again(app_env, monkeypatch):
    client = app_env["client"]
    services = app_env["modules"]["services"]
    _make_rev_fail(app_env, monkeypatch)

    # Retry wins the race: rev1 is re-armed, then rev2 is accepted.
    resp = client.post("/api/cohorts/cohort-r/retry", json={"revision": "1"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "pending", "revision": "1", "current_revision": "1"}
    assert _upload(client, "cohort-r", "2", _csv_bytes(ROWS_2)).status_code == 202

    # Acceptance withdraws the just re-armed old package; the worker can only
    # claim the new revision.
    assert _pkg_states(client, "cohort-r") == {"1": "superseded", "2": "pending"}
    assert services.claim_pending_package() == ("cohort-r", 2)
    services.finalize_package("cohort-r", "2")
    assert _pkg_states(client, "cohort-r") == {"1": "superseded", "2": "published"}
    assert client.get("/api/cohorts/cohort-r/package").status_code == 200


def test_inflight_old_build_is_discarded_and_restart_does_not_requeue(
    app_env, monkeypatch
):
    client = app_env["client"]
    services = app_env["modules"]["services"]
    _make_rev_fail(app_env, monkeypatch)

    # Retry, then the worker claims rev1 (generating).
    assert client.post("/api/cohorts/cohort-r/retry").status_code == 200
    assert services.claim_pending_package() == ("cohort-r", 1)
    assert _pkg_states(client, "cohort-r")["1"] == "generating"

    # rev2 arrives while rev1 is building.
    assert _upload(client, "cohort-r", "2", _csv_bytes(ROWS_2)).status_code == 202

    # The in-flight old result cannot be published.
    with pytest.raises(services.StaleRevisionError):
        services.finalize_package("cohort-r", "1")
    pkg_dir = app_env["modules"]["config"].PACKAGE_DIR / "cohort-r"
    assert not (pkg_dir / "rev1_deid.zip").exists()

    # Restart recovery must not put the withdrawn revision back in the queue.
    app_env["modules"]["database"].reset_interrupted_packages()
    assert _pkg_states(client, "cohort-r") == {"1": "superseded", "2": "pending"}
    assert services.claim_pending_package() == ("cohort-r", 2)

    services.finalize_package("cohort-r", "2")
    assert _pkg_states(client, "cohort-r") == {"1": "superseded", "2": "published"}
    dl = client.get("/api/cohorts/cohort-r/package")
    assert dl.status_code == 200, dl.text
    assert "rev2" in dl.headers.get("content-disposition", "")


def test_restart_repairs_stale_nonterminal_rows_for_old_revisions(app_env):
    """A DB left with a non-terminal row for an old revision (crash/legacy)
    is repaired to superseded, never re-queued."""
    client = app_env["client"]
    services = app_env["modules"]["services"]
    db = app_env["modules"]["database"]

    assert _upload(client, "cohort-s", "1", _csv_bytes(ROWS_1)).status_code == 202
    assert _upload(client, "cohort-s", "2", _csv_bytes(ROWS_2)).status_code == 202
    assert _pkg_states(client, "cohort-s") == {"1": "superseded", "2": "pending"}

    # Simulate a crash that left the withdrawn revision marked generating,
    # and rev2 still pending.
    conn = db.get_conn()
    conn.execute(
        "UPDATE packages SET status='generating', attempts=3 WHERE cohort_id='cohort-s' AND revision=1"
    )
    conn.close()

    db.reset_interrupted_packages()
    assert _pkg_states(client, "cohort-s") == {"1": "superseded", "2": "pending"}
    assert services.claim_pending_package() == ("cohort-s", 2)

    services.finalize_package("cohort-s", "2")
    assert _pkg_states(client, "cohort-s")["2"] == "published"
    assert client.get("/api/cohorts/cohort-s/package").status_code == 200


def test_late_failure_for_superseded_revision_is_not_recorded(app_env, monkeypatch):
    client = app_env["client"]
    services = app_env["modules"]["services"]
    _make_rev_fail(app_env, monkeypatch)

    assert client.post("/api/cohorts/cohort-r/retry").status_code == 200
    assert services.claim_pending_package() == ("cohort-r", 1)
    assert _upload(client, "cohort-r", "2", _csv_bytes(ROWS_2)).status_code == 202

    # A build exception surfaces only after rev2 landed: it must not flip the
    # withdrawn package back to failed (which would make it retryable again).
    services.mark_failed("cohort-r", "1", "late boom")
    assert _pkg_states(client, "cohort-r") == {"1": "superseded", "2": "pending"}

    assert app_env["tick"]() is True
    assert _pkg_states(client, "cohort-r")["2"] == "published"


def test_retry_current_failed_package_with_matching_hint_still_works(
    app_env, monkeypatch
):
    client = app_env["client"]
    _make_rev_fail(app_env, monkeypatch)

    resp = client.post("/api/cohorts/cohort-r/retry", json={"revision": "1"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "pending"
    assert _pkg_states(client, "cohort-r")["1"] == "pending"

    assert app_env["tick"]() is True
    assert _pkg_states(client, "cohort-r")["1"] == "published"
    assert client.get("/api/cohorts/cohort-r/package").status_code == 200


def test_concurrent_retry_and_acceptance_keep_old_package_withdrawn(app_env, monkeypatch):
    """Fire retry and acceptance from two threads; whichever wins, rev1 ends
    superseded and rev2 is the only buildable/downloadable revision."""
    import threading

    client = app_env["client"]
    services = app_env["modules"]["services"]
    _make_rev_fail(app_env, monkeypatch)

    barrier = threading.Barrier(2)
    results: dict[str, object] = {}

    def do_retry():
        barrier.wait()
        results["retry"] = client.post(
            "/api/cohorts/cohort-r/retry", json={"revision": "1"}
        )

    def do_accept():
        barrier.wait()
        results["accept"] = _upload(client, "cohort-r", "2", _csv_bytes(ROWS_2))

    threads = [threading.Thread(target=do_retry), threading.Thread(target=do_accept)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()

    assert results["accept"].status_code == 202
    # The retry either lost (409, revision 2 current) or won slightly earlier
    # (200, rev1 re-armed); acceptance always supersedes rev1 after that.
    assert results["retry"].status_code in (200, 409)
    states = _pkg_states(client, "cohort-r")
    assert states["1"] == "superseded"
    assert states["2"] in ("pending", "generating", "published")

    # Only the new revision is ever claimable; the old one never comes back.
    assert services.claim_pending_package() == ("cohort-r", 2)
    services.finalize_package("cohort-r", "2")
    assert _pkg_states(client, "cohort-r") == {"1": "superseded", "2": "published"}
    assert client.get("/api/cohorts/cohort-r/package").status_code == 200

