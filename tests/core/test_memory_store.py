import os
from datetime import timedelta

os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"

from libs.core import models  # noqa: E402
from services.api.app import database, memory_store  # noqa: E402

database.Base.metadata.create_all(bind=database.engine)


def _init_sqlite_store():
    return database.SessionLocal, memory_store


def test_memory_write_and_read_roundtrip():
    SessionLocal, memory_store = _init_sqlite_store()
    with SessionLocal() as db:
        entry = models.MemoryWrite(
            name="job_context",
            job_id="job-1",
            payload={"foo": "bar"},
        )
        created = memory_store.write_memory(db, entry)

        assert created.name == "job_context"
        assert created.payload["foo"] == "bar"
        assert created.job_id == "job-1"

        query = models.MemoryQuery(name="job_context", job_id="job-1", limit=10)
        results = memory_store.read_memory(db, query)
        assert len(results) == 1
        assert results[0].payload["foo"] == "bar"


def test_memory_scope_requires_job_id():
    SessionLocal, memory_store = _init_sqlite_store()
    with SessionLocal() as db:
        entry = models.MemoryWrite(
            name="job_context",
            payload={"foo": "bar"},
        )
        try:
            memory_store.write_memory(db, entry)
        except ValueError as exc:
            assert "job_id_required" in str(exc)
        else:
            raise AssertionError("Expected job_id requirement error")


def test_memory_write_conflict_on_mismatched_updated_at():
    SessionLocal, memory_store = _init_sqlite_store()
    with SessionLocal() as db:
        created = memory_store.write_memory(
            db,
            models.MemoryWrite(
                name="job_context",
                job_id="job-2",
                payload={"foo": "bar"},
            ),
        )

        ok_update = models.MemoryWrite(
            name="job_context",
            job_id="job-2",
            payload={"foo": "baz"},
            if_match_updated_at=created.updated_at,
        )
        updated = memory_store.write_memory(db, ok_update)
        assert updated.payload["foo"] == "baz"

        bad_update = models.MemoryWrite(
            name="job_context",
            job_id="job-2",
            payload={"foo": "qux"},
            if_match_updated_at=created.updated_at - timedelta(seconds=1),
        )
        try:
            memory_store.write_memory(db, bad_update)
        except ValueError as exc:
            assert "memory_conflict" in str(exc)
        else:
            raise AssertionError("Expected memory conflict error")
