#!/usr/bin/env python3
"""Verify the Agent Registry profile flow against a live API environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


JsonObject = dict[str, Any]


class VerificationError(RuntimeError):
    pass


def _json_dumps(payload: Any) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _request(
    base_url: str,
    method: str,
    path: str,
    *,
    body: JsonObject | None = None,
    bearer_token: str | None = None,
    timeout_s: float = 30.0,
) -> Any:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = _json_dumps(body)
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    request = Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise VerificationError(f"{method} {path} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise VerificationError(f"{method} {path} could not reach API: {exc}") from exc
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VerificationError(f"{method} {path} returned non-JSON response: {raw[:200]}") from exc


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _query(path: str, params: dict[str, str]) -> str:
    encoded = urlencode({key: value for key, value in params.items() if value})
    return f"{path}?{encoded}" if encoded else path


def _load_capabilities(base_url: str, bearer_token: str | None) -> set[str]:
    response = _request(
        base_url,
        "GET",
        "/capabilities?with_schemas=false",
        bearer_token=bearer_token,
    )
    items = response.get("items", []) if isinstance(response, dict) else []
    return {str(item.get("id", "")).strip() for item in items if isinstance(item, dict)}


def _agent_run_spec(
    *,
    primary_capability_id: str,
    goal: str = "",
    workspace_path: str = "",
    constraints: str = "",
    max_steps: int | None = None,
    extra_capability_id: str | None = None,
) -> JsonObject:
    primary_request = {
        "request_id": primary_capability_id,
        "capability_id": primary_capability_id,
        "execution_request_id": primary_capability_id,
    }
    input_bindings: JsonObject = {
        "goal": goal,
        "workspace_path": workspace_path,
    }
    if constraints:
        input_bindings["constraints"] = constraints
    if max_steps is not None:
        input_bindings["max_steps"] = max_steps
    steps: list[JsonObject] = [
        {
            "step_id": "primary_agent",
            "name": "PrimaryAgent",
            "description": "Run the primary agent.",
            "instruction": "Use the provided Agent Registry verification inputs.",
            "capability_request": primary_request,
            "input_bindings": input_bindings,
            "retry_policy": {
                "max_attempts": 1,
                "retry_class": "standard",
                "retryable_errors": [],
                "backoff_seconds": 0,
                "backoff_multiplier": 1,
                "jitter_seconds": 0,
            },
            "acceptance_policy": {
                "acceptance_criteria": [],
                "critic_required": False,
            },
            "depends_on": [],
        }
    ]
    capability_requests = [primary_request]
    dag_edges: list[list[str]] = []
    if extra_capability_id:
        extra_request = {
            "request_id": extra_capability_id,
            "capability_id": extra_capability_id,
            "execution_request_id": extra_capability_id,
        }
        steps.append(
            {
                "step_id": "extra_step",
                "name": "ExtraStep",
                "description": "Run an allowlisted extra capability.",
                "instruction": "Run the extra capability for Agent Registry verification.",
                "capability_request": extra_request,
                "input_bindings": {},
                "retry_policy": {
                    "max_attempts": 1,
                    "retry_class": "standard",
                    "retryable_errors": [],
                    "backoff_seconds": 0,
                    "backoff_multiplier": 1,
                    "jitter_seconds": 0,
                },
                "acceptance_policy": {
                    "acceptance_criteria": [],
                    "critic_required": False,
                },
                "depends_on": ["primary_agent"],
            }
        )
        capability_requests.append(extra_request)
        dag_edges.append(["primary_agent", "extra_step"])
    return {
        "version": "1",
        "kind": "api",
        "planner_version": "workbench_v1",
        "tasks_summary": goal or "Agent Registry e2e verification",
        "steps": steps,
        "dag_edges": dag_edges,
        "capability_requests": capability_requests,
        "metadata": {
            "surface": "studio_workbench",
            "workbench_mode": "agent",
            "verification": "agent_registry_phase_4",
        },
    }


def _verify_profile_launch(
    base_url: str,
    bearer_token: str | None,
    agent_definition_id: str,
    *,
    agent_definition_version_id: str | None = None,
    agent_definition_version_number: int | None = None,
    primary_capability_id: str,
    extra_capability_id: str | None,
    expected: JsonObject,
) -> JsonObject:
    body: JsonObject = {
        "agent_definition_id": agent_definition_id,
        "context_json": {"verification": "agent_registry_phase_4"},
        "run_spec": _agent_run_spec(
            primary_capability_id=primary_capability_id,
            extra_capability_id=extra_capability_id,
        ),
    }
    if agent_definition_version_id:
        body["agent_definition_version_id"] = agent_definition_version_id
    launch_response = _request(
        base_url,
        "POST",
        "/workbench/agent-runs",
        bearer_token=bearer_token,
        body=body,
        timeout_s=60.0,
    )
    _assert(isinstance(launch_response, dict), "Profile launch response was not an object.")
    run = launch_response.get("run")
    run_spec = launch_response.get("run_spec")
    _assert(isinstance(run, dict), "Profile launch response did not include a run.")
    _assert(isinstance(run_spec, dict), "Profile launch response did not include a run_spec.")
    _assert(
        run.get("metadata", {}).get("agent_definition_id") == agent_definition_id,
        "Profile launch did not persist agent_definition_id in run metadata.",
    )
    snapshot = run.get("metadata", {}).get("agent_definition_snapshot")
    _assert(
        isinstance(snapshot, dict),
        "Profile launch did not persist an agent definition snapshot.",
    )
    _assert(
        snapshot.get("agent_definition_id") == agent_definition_id,
        "Agent definition snapshot has the wrong id.",
    )
    if agent_definition_version_id:
        _assert(
            run.get("metadata", {}).get("agent_definition_version_id")
            == agent_definition_version_id,
            "Version launch did not persist agent_definition_version_id in metadata.",
        )
        _assert(
            snapshot.get("agent_definition_version_id") == agent_definition_version_id,
            "Version launch snapshot has the wrong version id.",
        )
        _assert(
            snapshot.get("agent_definition_version_number") == agent_definition_version_number,
            "Version launch snapshot has the wrong version number.",
        )
    steps = run_spec.get("steps")
    _assert(isinstance(steps, list) and len(steps) >= 1, "Profile launch returned no steps.")
    first_inputs = steps[0].get("input_bindings") if isinstance(steps[0], dict) else None
    _assert(isinstance(first_inputs, dict), "Profile launch first step has no input bindings.")
    for key, value in expected.items():
        _assert(
            first_inputs.get(key) == value,
            "Profile launch did not hydrate "
            f"{key!r}: expected {value!r}, got {first_inputs.get(key)!r}.",
        )
    persisted_run = _request(
        base_url,
        "GET",
        f"/runs/{run['id']}",
        bearer_token=bearer_token,
    )
    persisted_snapshot = persisted_run.get("metadata", {}).get("agent_definition_snapshot")
    _assert(
        isinstance(persisted_snapshot, dict)
        and persisted_snapshot.get("agent_definition_id") == agent_definition_id,
        "Persisted run metadata did not retain the agent definition snapshot.",
    )
    return {"run_id": run.get("id"), "job_id": run.get("job_id"), "status": run.get("status")}


def _verify_unsaved_launch(
    base_url: str,
    bearer_token: str | None,
    *,
    primary_capability_id: str,
) -> JsonObject:
    goal = "Agent Registry unsaved draft verification."
    launch_response = _request(
        base_url,
        "POST",
        "/workbench/agent-runs",
        bearer_token=bearer_token,
        body={
            "title": "Agent Registry unsaved draft verification",
            "goal": goal,
            "user_id": "agent-registry-e2e",
            "context_json": {"verification": "agent_registry_phase_4_unsaved"},
            "run_spec": _agent_run_spec(
                primary_capability_id=primary_capability_id,
                goal=goal,
                workspace_path="agent-registry-e2e",
                constraints="Do not modify files; verification only.",
                max_steps=1,
            ),
        },
        timeout_s=60.0,
    )
    _assert(isinstance(launch_response, dict), "Unsaved launch response was not an object.")
    run = launch_response.get("run")
    _assert(isinstance(run, dict), "Unsaved launch response did not include a run.")
    metadata = run.get("metadata", {})
    _assert(
        "agent_definition_id" not in metadata,
        "Unsaved launch unexpectedly persisted agent_definition_id.",
    )
    return {"run_id": run.get("id"), "job_id": run.get("job_id"), "status": run.get("status")}


def verify(args: argparse.Namespace) -> JsonObject:
    base_url = str(args.base_url).rstrip("/")
    bearer_token = str(args.bearer_token or "").strip() or None
    primary_capability_id = str(args.primary_capability_id).strip()
    requested_extra_capability_id = str(args.extra_capability_id or "").strip()
    run_id = f"agent-registry-e2e-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    user_id = str(args.user_id or "agent-registry-e2e").strip()
    default_goal = "Agent Registry profile verification."
    default_workspace_path = "agent-registry-e2e"
    default_constraints = [
        "Do not modify files; verification only.",
        "Return only readiness information.",
    ]
    default_max_steps = 1

    capabilities = _load_capabilities(base_url, bearer_token)
    _assert(
        primary_capability_id in capabilities,
        f"Primary capability {primary_capability_id!r} is not available in the target catalog.",
    )
    extra_capability_id = (
        requested_extra_capability_id if requested_extra_capability_id in capabilities else None
    )

    created = _request(
        base_url,
        "POST",
        "/agents/definitions",
        bearer_token=bearer_token,
        body={
            "name": f"Agent Registry E2E {run_id}",
            "description": "Temporary profile created by the Agent Registry phase 4 verifier.",
            "agent_capability_id": primary_capability_id,
            "instructions": "Use this profile only for Agent Registry e2e verification.",
            "default_goal": default_goal,
            "default_workspace_path": default_workspace_path,
            "default_constraints": default_constraints,
            "default_max_steps": default_max_steps,
            "model_config": {"verification": "agent_registry_phase_4"},
            "allowed_capability_ids": [extra_capability_id] if extra_capability_id else [],
            "memory_policy": {"mode": "disabled"},
            "guardrail_policy": {"verification": True},
            "workspace_policy": {"write_roots": [default_workspace_path]},
            "metadata": {"verification_run_id": run_id},
            "user_id": user_id,
        },
    )
    _assert(isinstance(created, dict), "Create definition response was not an object.")
    agent_definition_id = str(created.get("id") or "")
    _assert(agent_definition_id, "Create definition response did not include an id.")

    report: JsonObject = {
        "base_url": base_url,
        "verification_run_id": run_id,
        "agent_definition_id": agent_definition_id,
        "primary_capability_id": primary_capability_id,
        "extra_capability_id": extra_capability_id,
        "checks": [],
    }

    try:
        listed = _request(
            base_url,
            "GET",
            _query("/agents/definitions", {"user_id": user_id}),
            bearer_token=bearer_token,
        )
        _assert(isinstance(listed, list), "List definitions response was not a list.")
        _assert(
            any(item.get("id") == agent_definition_id for item in listed if isinstance(item, dict)),
            "Created definition did not appear in the enabled profile list.",
        )
        report["checks"].append("profile_listed")

        fetched = _request(
            base_url,
            "GET",
            f"/agents/definitions/{agent_definition_id}",
            bearer_token=bearer_token,
        )
        _assert(
            isinstance(fetched, dict)
            and fetched.get("agent_capability_id") == primary_capability_id,
            "Fetched definition did not match the created primary capability.",
        )
        report["checks"].append("profile_fetched")

        published = _request(
            base_url,
            "POST",
            f"/agents/definitions/{agent_definition_id}/versions",
            bearer_token=bearer_token,
            body={
                "version_note": "Agent Registry e2e published snapshot.",
                "published_by": "agent-registry-e2e",
                "metadata": {"verification_run_id": run_id},
            },
        )
        _assert(isinstance(published, dict), "Publish version response was not an object.")
        agent_definition_version_id = str(published.get("id") or "")
        _assert(agent_definition_version_id, "Publish version response did not include an id.")
        _assert(
            published.get("version_number") == 1,
            "First published Agent Definition version was not version 1.",
        )
        report["agent_definition_version_id"] = agent_definition_version_id
        report["checks"].append("profile_version_published")

        updated = _request(
            base_url,
            "PUT",
            f"/agents/definitions/{agent_definition_id}",
            bearer_token=bearer_token,
            body={
                "description": "Updated by Agent Registry phase 5 verifier.",
                "default_goal": "Changed after published version.",
                "default_workspace_path": "changed-after-publish",
                "default_constraints": ["changed after publish"],
                "default_max_steps": 2,
            },
        )
        _assert(
            isinstance(updated, dict)
            and updated.get("description") == "Updated by Agent Registry phase 5 verifier.",
            "Updated definition did not return the expected description.",
        )
        report["checks"].append("profile_updated")

        listed_versions = _request(
            base_url,
            "GET",
            f"/agents/definitions/{agent_definition_id}/versions",
            bearer_token=bearer_token,
        )
        _assert(isinstance(listed_versions, list), "List versions response was not a list.")
        _assert(
            [item.get("version_number") for item in listed_versions if isinstance(item, dict)]
            == [1],
            "Published versions did not return the expected immutable version list.",
        )
        report["checks"].append("profile_version_listed")

        profile_launch = _verify_profile_launch(
            base_url,
            bearer_token,
            agent_definition_id,
            primary_capability_id=primary_capability_id,
            extra_capability_id=extra_capability_id,
            expected={
                "goal": "Changed after published version.",
                "workspace_path": "changed-after-publish",
                "constraints": "changed after publish",
                "max_steps": 2,
            },
        )
        report["profile_launch"] = profile_launch
        report["checks"].append("profile_launch_snapshot_persisted")

        version_launch = _verify_profile_launch(
            base_url,
            bearer_token,
            agent_definition_id,
            agent_definition_version_id=agent_definition_version_id,
            agent_definition_version_number=1,
            primary_capability_id=primary_capability_id,
            extra_capability_id=extra_capability_id,
            expected={
                "goal": default_goal,
                "workspace_path": default_workspace_path,
                "constraints": "\n".join(default_constraints),
                "max_steps": default_max_steps,
            },
        )
        report["version_launch"] = version_launch
        report["checks"].append("profile_version_launch_snapshot_persisted")

        if not args.skip_unsaved_launch:
            unsaved_launch = _verify_unsaved_launch(
                base_url,
                bearer_token,
                primary_capability_id=primary_capability_id,
            )
            report["unsaved_launch"] = unsaved_launch
            report["checks"].append("unsaved_launch_without_profile")

        _request(
            base_url,
            "DELETE",
            f"/agents/definitions/{agent_definition_id}",
            bearer_token=bearer_token,
        )
        listed_after_delete = _request(
            base_url,
            "GET",
            _query("/agents/definitions", {"user_id": user_id}),
            bearer_token=bearer_token,
        )
        _assert(isinstance(listed_after_delete, list), "Post-delete list response was not a list.")
        _assert(
            not any(
                item.get("id") == agent_definition_id
                for item in listed_after_delete
                if isinstance(item, dict)
            ),
            "Deleted definition still appears in the default enabled profile list.",
        )
        report["checks"].append("profile_deleted_hidden_by_default")
        report["ok"] = True
        return report
    except Exception:
        if not args.keep_profile:
            try:
                _request(
                    base_url,
                    "DELETE",
                    f"/agents/definitions/{agent_definition_id}",
                    bearer_token=bearer_token,
                )
            except Exception:
                pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("AGENT_REGISTRY_E2E_BASE_URL", "http://127.0.0.1:18000"),
        help="Base URL for the API under test.",
    )
    parser.add_argument(
        "--bearer-token",
        default=os.getenv("AGENT_REGISTRY_E2E_BEARER_TOKEN", ""),
        help="Optional bearer token for the API under test.",
    )
    parser.add_argument(
        "--user-id",
        default=os.getenv("AGENT_REGISTRY_E2E_USER_ID", "agent-registry-e2e"),
        help="User id used to scope the temporary profile.",
    )
    parser.add_argument(
        "--primary-capability-id",
        default=os.getenv("AGENT_REGISTRY_E2E_PRIMARY_CAPABILITY", "codegen.autonomous"),
        help="Primary agent capability id to verify.",
    )
    parser.add_argument(
        "--extra-capability-id",
        default=os.getenv("AGENT_REGISTRY_E2E_EXTRA_CAPABILITY", "filesystem.workspace.list"),
        help="Optional extra capability id to attach when available.",
    )
    parser.add_argument(
        "--skip-unsaved-launch",
        action="store_true",
        default=os.getenv("AGENT_REGISTRY_E2E_SKIP_UNSAVED_LAUNCH", "").lower()
        in {"1", "true", "yes"},
        help="Skip the direct unsaved Agent Sandbox launch check.",
    )
    parser.add_argument(
        "--keep-profile",
        action="store_true",
        help="Keep the temporary profile if verification fails.",
    )
    parser.add_argument(
        "--output",
        default=os.getenv(
            "AGENT_REGISTRY_E2E_OUTPUT",
            "artifacts/evals/agent_registry_staging_e2e_report.json",
        ),
        help="Path to write the JSON verification report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = verify(args)
    except VerificationError as exc:
        print(f"agent_registry_e2e_failed:{exc}", file=sys.stderr)
        return 1
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
