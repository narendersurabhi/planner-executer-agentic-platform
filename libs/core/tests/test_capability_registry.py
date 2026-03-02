from libs.core import capability_registry


def test_capability_allowlist_allows_by_default(monkeypatch):
    monkeypatch.delenv("CAPABILITY_GOVERNANCE_ENABLED", raising=False)
    monkeypatch.delenv("CAPABILITY_GOVERNANCE_MODE", raising=False)
    monkeypatch.delenv("ENABLED_CAPABILITIES", raising=False)
    monkeypatch.delenv("DISABLED_CAPABILITIES", raising=False)
    monkeypatch.delenv("WORKER_ENABLED_CAPABILITIES", raising=False)
    monkeypatch.delenv("WORKER_DISABLED_CAPABILITIES", raising=False)

    decision = capability_registry.evaluate_capability_allowlist("document.spec.generate", "worker")
    assert decision.allowed is True
    assert decision.reason == "allowed"
    assert decision.mode == "enforce"
    assert decision.violated is False


def test_capability_allowlist_denies_global_disabled(monkeypatch):
    monkeypatch.setenv("CAPABILITY_GOVERNANCE_ENABLED", "true")
    monkeypatch.setenv("CAPABILITY_GOVERNANCE_MODE", "enforce")
    monkeypatch.setenv("DISABLED_CAPABILITIES", "document.spec.generate")

    decision = capability_registry.evaluate_capability_allowlist("document.spec.generate", "worker")
    assert decision.allowed is False
    assert decision.reason == "global_disabled"
    assert decision.mode == "enforce"
    assert decision.violated is True


def test_capability_allowlist_respects_service_enabled(monkeypatch):
    monkeypatch.setenv("CAPABILITY_GOVERNANCE_ENABLED", "true")
    monkeypatch.setenv("CAPABILITY_GOVERNANCE_MODE", "enforce")
    monkeypatch.delenv("ENABLED_CAPABILITIES", raising=False)
    monkeypatch.delenv("DISABLED_CAPABILITIES", raising=False)
    monkeypatch.setenv("WORKER_ENABLED_CAPABILITIES", "document.spec.generate")

    denied = capability_registry.evaluate_capability_allowlist("document.spec.validate", "worker")
    assert denied.allowed is False
    assert denied.reason == "not_in_service_enabled"

    allowed = capability_registry.evaluate_capability_allowlist("document.spec.generate", "worker")
    assert allowed.allowed is True
    assert allowed.reason == "allowed"


def test_capability_allowlist_dry_run_does_not_block(monkeypatch):
    monkeypatch.setenv("CAPABILITY_GOVERNANCE_ENABLED", "true")
    monkeypatch.setenv("CAPABILITY_GOVERNANCE_MODE", "dry_run")
    monkeypatch.setenv("ENABLED_CAPABILITIES", "document.spec.generate")

    decision = capability_registry.evaluate_capability_allowlist("document.spec.validate", "worker")
    assert decision.allowed is True
    assert decision.mode == "dry_run"
    assert decision.violated is True
    assert decision.reason == "dry_run:not_in_global_enabled"
