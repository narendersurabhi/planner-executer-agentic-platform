from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("mcp")

from services.tailor.app import mcp as mcp_module


def _clear_iterative_cache() -> None:
    with mcp_module._ITERATIVE_DEDUPE_LOCK:  # type: ignore[attr-defined]
        mcp_module._ITERATIVE_DEDUPE_CACHE.clear()  # type: ignore[attr-defined]


def test_improve_iterative_dedupe_cache_hit(monkeypatch) -> None:
    _clear_iterative_cache()
    monkeypatch.setenv("TAILOR_ITERATIVE_INFLIGHT_WAIT_S", "0.1")

    key = "k1"
    expected = {"alignment_score": 92.0, "iterations": 4}

    first_calls = {"count": 0}

    def first_compute() -> dict:
        first_calls["count"] += 1
        return expected

    first = mcp_module._run_improve_iterative_with_dedupe(  # type: ignore[attr-defined]
        key=key,
        job_id="job-1",
        compute=first_compute,
    )
    assert first == expected
    assert first_calls["count"] == 1

    second_calls = {"count": 0}

    def second_compute() -> dict:
        second_calls["count"] += 1
        return {"alignment_score": 0.0}

    second = mcp_module._run_improve_iterative_with_dedupe(  # type: ignore[attr-defined]
        key=key,
        job_id="job-1",
        compute=second_compute,
    )
    assert second == expected
    assert second_calls["count"] == 0


def test_improve_iterative_dedupe_waits_for_inflight(monkeypatch) -> None:
    _clear_iterative_cache()
    monkeypatch.setenv("TAILOR_ITERATIVE_INFLIGHT_WAIT_S", "1")

    key = "k2"
    ready = threading.Event()
    result_from_owner = {"alignment_score": 90.0, "iterations": 2}
    calls = {"count": 0}

    def owner_compute() -> dict:
        calls["count"] += 1
        ready.set()
        time.sleep(0.2)
        return result_from_owner

    owner_result: dict = {}
    waiter_result: dict = {}

    def owner() -> None:
        owner_result.update(
            mcp_module._run_improve_iterative_with_dedupe(  # type: ignore[attr-defined]
                key=key,
                job_id="job-2",
                compute=owner_compute,
            )
        )

    def waiter() -> None:
        ready.wait(timeout=1.0)
        waiter_result.update(
            mcp_module._run_improve_iterative_with_dedupe(  # type: ignore[attr-defined]
                key=key,
                job_id="job-2",
                compute=lambda: {"alignment_score": -1.0},
            )
        )

    owner_thread = threading.Thread(target=owner)
    waiter_thread = threading.Thread(target=waiter)
    owner_thread.start()
    waiter_thread.start()
    owner_thread.join(timeout=2.0)
    waiter_thread.join(timeout=2.0)

    assert owner_result == result_from_owner
    assert waiter_result == result_from_owner
    assert calls["count"] == 1
