from __future__ import annotations

from libs.core import tracing


def test_start_span_noop_mode(monkeypatch):
    monkeypatch.setattr(tracing, "_OTEL_AVAILABLE", False)
    with tracing.start_span("test.span", attributes={"a": 1, "b": True, "c": None}) as span:
        tracing.set_span_attributes(
            span,
            {
                "text": "value",
                "number": 2,
                "bool": False,
                "list": ["x", 1, {"ignored": True}],
            },
        )


def test_configure_tracing_noop_when_unavailable(monkeypatch):
    monkeypatch.setattr(tracing, "_OTEL_AVAILABLE", False)
    monkeypatch.setattr(tracing, "_TRACING_CONFIGURED", False)
    assert tracing.configure_tracing("worker") is False


def test_normalize_otlp_endpoint_adds_traces_path():
    assert (
        tracing._normalize_otlp_traces_endpoint("http://jaeger:4318")
        == "http://jaeger:4318/v1/traces"
    )
    assert (
        tracing._normalize_otlp_traces_endpoint("http://jaeger:4318/")
        == "http://jaeger:4318/v1/traces"
    )
    assert (
        tracing._normalize_otlp_traces_endpoint("http://jaeger:4318/v1/traces")
        == "http://jaeger:4318/v1/traces"
    )
