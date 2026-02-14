from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

import yaml

from .models import PolicyDecision, PolicyDecisionType, Task


@dataclass
class PolicyConfig:
    allowlist: List[str]
    max_tool_calls: int
    max_runtime_s: int
    output_size_cap: int
    artifact_prefix: str
    sensitive_keys: List[str]


class PolicyEngine:
    def __init__(self, mode: str, config_path: str) -> None:
        self.mode = mode
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> PolicyConfig:
        data = yaml.safe_load(open(self.config_path, "r", encoding="utf-8"))
        policy = data.get("policy", {})
        return PolicyConfig(
            allowlist=policy.get("allowlist", []),
            max_tool_calls=policy.get("max_tool_calls", 5),
            max_runtime_s=policy.get("max_runtime_s", 120),
            output_size_cap=policy.get("output_size_cap", 50000),
            artifact_prefix=policy.get("artifact_prefix", "/shared/artifacts"),
            sensitive_keys=policy.get("sensitive_keys", ["ssn", "dob", "member_id"]),
        )

    def evaluate_task(self, task: Task, tool_http_fetch_enabled: bool) -> PolicyDecision:
        reasons: List[str] = []
        decision = PolicyDecisionType.allow
        if self.mode == "prod":
            for tool in task.tool_requests:
                if tool not in self.config.allowlist:
                    reasons.append(f"Tool not in allowlist: {tool}")
            if reasons:
                decision = PolicyDecisionType.deny
        if self.mode == "dev" and "http_fetch" in task.tool_requests and not tool_http_fetch_enabled:
            decision = PolicyDecisionType.deny
            reasons.append("http_fetch disabled by TOOL_HTTP_FETCH_ENABLED")
        if len(task.tool_requests) > self.config.max_tool_calls:
            decision = PolicyDecisionType.deny
            reasons.append("Exceeded max tool calls per task")
        return PolicyDecision(
            scope="task",
            decision=decision,
            reasons=reasons,
            rewrites=None,
            decided_at=datetime.utcnow(),
        )
