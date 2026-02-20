from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class MemoryClientError(Exception):
    pass


class MemoryClient:
    def __init__(self, base_url: str, timeout_s: float = 5.0) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.timeout_s = timeout_s

    def read(
        self,
        *,
        name: str,
        scope: Optional[str] = None,
        key: Optional[str] = None,
        job_id: Optional[str] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        limit: int = 50,
        include_expired: bool = False,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "name": name,
            "limit": limit,
            "include_expired": str(include_expired).lower(),
        }
        if scope:
            params["scope"] = scope
        if key:
            params["key"] = key
        if job_id:
            params["job_id"] = job_id
        if user_id:
            params["user_id"] = user_id
        if project_id:
            params["project_id"] = project_id
        url = f"{self.base_url}/memory/read?{urlencode(params)}"
        try:
            payload = _request_json(url, timeout_s=self.timeout_s)
        except MemoryClientError:
            return []
        if isinstance(payload, list):
            return [entry for entry in payload if isinstance(entry, dict)]
        return []

    def write(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/memory/write"
        try:
            payload = _request_json(
                url,
                method="POST",
                body=entry,
                timeout_s=self.timeout_s,
            )
        except MemoryClientError:
            return None
        return payload if isinstance(payload, dict) else None


def _request_json(
    url: str,
    *,
    method: str = "GET",
    body: Optional[Dict[str, Any]] = None,
    timeout_s: float = 5.0,
) -> Any:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise MemoryClientError(str(exc)) from exc
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MemoryClientError("Invalid JSON response") from exc
