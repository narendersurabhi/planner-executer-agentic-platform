from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Mapping
from urllib.parse import urlparse, urlunparse

LOGGER = logging.getLogger(__name__)

_OTEL_AVAILABLE = True
try:  # pragma: no cover - optional dependency
    from opentelemetry import trace as _trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode
except Exception:  # noqa: BLE001
    _OTEL_AVAILABLE = False
    _trace = None  # type: ignore[assignment]
    OTLPSpanExporter = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    BatchSpanProcessor = None  # type: ignore[assignment]
    Status = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]

_TRACING_CONFIGURED = False


class _NoopSpan:
    def set_attribute(self, _key: str, _value: Any) -> None:
        return

    def record_exception(self, _exception: BaseException) -> None:
        return

    def set_status(self, _status: Any) -> None:
        return


class _NoopSpanContext:
    def __enter__(self) -> _NoopSpan:
        return _NoopSpan()

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        return False


class _NoopTracer:
    def start_as_current_span(self, _name: str) -> _NoopSpanContext:
        return _NoopSpanContext()


def configure_tracing(service_name: str, endpoint: str | None = None) -> bool:
    global _TRACING_CONFIGURED
    if _TRACING_CONFIGURED:
        return True
    if not _OTEL_AVAILABLE:
        LOGGER.info("tracing_unavailable", extra={"service": service_name})
        return False
    try:
        resolved_endpoint = _normalize_otlp_traces_endpoint(endpoint)
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = (
            OTLPSpanExporter(endpoint=resolved_endpoint)
            if resolved_endpoint
            else OTLPSpanExporter()
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        _trace.set_tracer_provider(provider)
        _TRACING_CONFIGURED = True
        return True
    except Exception:  # noqa: BLE001
        LOGGER.exception("tracing_configure_failed", extra={"service": service_name})
        return False


def get_tracer(name: str = "awe"):
    if not _OTEL_AVAILABLE:
        return _NoopTracer()
    return _trace.get_tracer(name)


@contextmanager
def start_span(
    name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    tracer_name: str = "awe",
) -> Iterator[Any]:
    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(name) as span:
        set_span_attributes(span, attributes)
        try:
            yield span
        except Exception as exc:  # noqa: BLE001
            if hasattr(span, "record_exception"):
                span.record_exception(exc)
            if _OTEL_AVAILABLE and Status is not None and StatusCode is not None:
                span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def set_span_attributes(span: Any, attributes: Mapping[str, Any] | None) -> None:
    if not attributes:
        return
    for key, value in attributes.items():
        if not isinstance(key, str) or not key:
            continue
        normalized = _normalize_attribute_value(value)
        if normalized is None:
            continue
        if hasattr(span, "set_attribute"):
            span.set_attribute(key, normalized)


def _normalize_attribute_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        normalized_list: list[Any] = []
        for item in value:
            normalized_item = _normalize_attribute_value(item)
            if isinstance(normalized_item, (bool, int, float, str)):
                normalized_list.append(normalized_item)
        return normalized_list or None
    return str(value)


def _normalize_otlp_traces_endpoint(endpoint: str | None) -> str | None:
    if endpoint is None:
        return None
    raw = endpoint.strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        return raw
    if parsed.path in {"", "/"}:
        path = "/v1/traces"
    elif parsed.path.endswith("/v1/traces"):
        path = parsed.path
    else:
        path = parsed.path.rstrip("/") + "/v1/traces"
    return urlunparse(parsed._replace(path=path))
