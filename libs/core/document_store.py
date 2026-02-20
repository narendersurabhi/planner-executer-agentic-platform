from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class DocumentStoreError(RuntimeError):
    pass


def _artifacts_root() -> Path:
    return Path(os.getenv("ARTIFACTS_DIR", "/shared/artifacts")).resolve()


def _storage_backend() -> str:
    return os.getenv("DOCUMENT_STORE_BACKEND", "filesystem").strip().lower()


def _s3_bucket() -> str:
    return os.getenv("DOCUMENT_STORE_S3_BUCKET", "").strip()


def _s3_prefix() -> str:
    return os.getenv("DOCUMENT_STORE_S3_PREFIX", "").strip().strip("/")


def _s3_key_for_path(relative_path: str) -> str:
    prefix = _s3_prefix()
    return f"{prefix}/{relative_path}" if prefix else relative_path


def artifact_relative_path(path: str) -> str:
    candidate = (path or "").strip()
    if not candidate:
        raise DocumentStoreError("path is required")
    candidate = candidate.replace("\\", "/")
    if candidate.startswith("/"):
        # Absolute path under artifacts root is acceptable; convert to relative.
        root = _artifacts_root()
        target = Path(candidate).resolve()
        if not str(target).startswith(str(root)):
            raise DocumentStoreError("path is outside artifacts root")
        candidate = str(target.relative_to(root)).replace("\\", "/")
    if candidate.startswith("artifacts/"):
        candidate = candidate[len("artifacts/") :]
    if candidate.startswith("./"):
        candidate = candidate[2:]
    normalized = Path(candidate).as_posix()
    if normalized.startswith("../") or normalized == "..":
        raise DocumentStoreError("path traversal is not allowed")
    return normalized


def artifact_local_path(path: str) -> Path:
    relative = artifact_relative_path(path)
    root = _artifacts_root()
    target = (root / relative).resolve()
    if not str(target).startswith(str(root)):
        raise DocumentStoreError("invalid artifact path")
    return target


def is_s3_enabled() -> bool:
    return _storage_backend() == "s3"


def _s3_client():
    import boto3

    kwargs: dict[str, Any] = {}
    endpoint = os.getenv("DOCUMENT_STORE_S3_ENDPOINT", "").strip()
    region = os.getenv("DOCUMENT_STORE_S3_REGION", "").strip()
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    if region:
        kwargs["region_name"] = region
    return boto3.client("s3", **kwargs)


def upload_artifact(path: str) -> str | None:
    if not is_s3_enabled():
        return None
    bucket = _s3_bucket()
    if not bucket:
        raise DocumentStoreError("DOCUMENT_STORE_S3_BUCKET is required for s3 backend")
    relative = artifact_relative_path(path)
    source = artifact_local_path(relative)
    if not source.exists() or not source.is_file():
        raise DocumentStoreError(f"artifact file not found:{relative}")
    key = _s3_key_for_path(relative)
    client = _s3_client()
    client.upload_file(str(source), bucket, key)
    return key


def download_artifact_bytes(path: str) -> bytes:
    if not is_s3_enabled():
        raise DocumentStoreError("s3 backend is not enabled")
    bucket = _s3_bucket()
    if not bucket:
        raise DocumentStoreError("DOCUMENT_STORE_S3_BUCKET is required for s3 backend")
    relative = artifact_relative_path(path)
    key = _s3_key_for_path(relative)
    client = _s3_client()
    response = client.get_object(Bucket=bucket, Key=key)
    body = response.get("Body")
    if body is None:
        raise DocumentStoreError("artifact body missing")
    return body.read()

