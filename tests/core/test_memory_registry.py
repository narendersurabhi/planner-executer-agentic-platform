from libs.core.memory_registry import DEFAULT_MEMORY_SPECS, MemoryRegistry, default_memory_registry
from libs.core.models import MemoryScope, MemorySpec


def test_register_and_get_trims_name() -> None:
    registry = MemoryRegistry()
    spec = MemorySpec(
        name="  session_cache  ",
        description="Ephemeral session cache",
        scope=MemoryScope.session,
    )
    registry.register(spec)

    assert registry.has("session_cache")
    stored = registry.get("session_cache")
    assert stored.name == "session_cache"
    assert stored.scope == MemoryScope.session


def test_register_duplicate_name_raises() -> None:
    registry = MemoryRegistry()
    spec = MemorySpec(
        name="user_profile",
        description="User profile",
        scope=MemoryScope.user,
    )
    registry.register(spec)

    try:
        registry.register(spec)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("Expected ValueError for duplicate registration")


def test_register_invalid_ttl_raises() -> None:
    registry = MemoryRegistry()
    spec = MemorySpec(
        name="bad_ttl",
        description="Invalid ttl",
        scope=MemoryScope.session,
        ttl_seconds=0,
    )

    try:
        registry.register(spec)
    except ValueError as exc:
        assert "ttl_seconds" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid ttl_seconds")


def test_default_registry_contains_defaults() -> None:
    registry = default_memory_registry()
    default_names = {spec.name for spec in DEFAULT_MEMORY_SPECS}
    registry_names = {spec.name for spec in registry.list()}

    assert default_names.issubset(registry_names)
