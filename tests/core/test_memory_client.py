import json

from libs.core import memory_client


class FakeResponse:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_memory_client_read_success(monkeypatch):
    payload = json.dumps([{"payload": {"foo": "bar"}}])

    def fake_urlopen(request, timeout=5.0):
        return FakeResponse(payload)

    monkeypatch.setattr(memory_client, "urlopen", fake_urlopen)

    client = memory_client.MemoryClient("http://api")
    results = client.read(name="job_context", job_id="job-1")

    assert len(results) == 1
    assert results[0]["payload"]["foo"] == "bar"


def test_memory_client_write_success(monkeypatch):
    payload = json.dumps({"id": "mem-1", "payload": {"ok": True}})

    def fake_urlopen(request, timeout=5.0):
        assert request.method == "POST"
        return FakeResponse(payload)

    monkeypatch.setattr(memory_client, "urlopen", fake_urlopen)

    client = memory_client.MemoryClient("http://api")
    result = client.write({"name": "task_outputs", "job_id": "job-1", "payload": {"ok": True}})

    assert result is not None
    assert result["payload"]["ok"] is True
