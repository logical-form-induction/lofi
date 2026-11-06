import pytest
from tests.support.installed import entries, install, installed

from lofi.registry import Registry, UnknownTagError


def test_entries_name_each_member_under_the_family_group() -> None:
    points = entries("sink", {"first": 1, "second": 2})

    assert tuple(point.name for point in points) == ("first", "second")
    assert {point.group for point in points} == {"lofi.sink"}
    assert tuple(point.load() for point in points) == (1, 2)


def test_a_registry_over_given_members_resolves_exactly_those_members() -> None:
    registry = installed("sink", {"memory": "held"})

    assert registry.get("memory") == "held"
    assert registry.available() == ("memory",)

    with pytest.raises(UnknownTagError):
        registry.get("disk")


def test_installing_replaces_one_family_and_keeps_every_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kept = Registry[object]("rule").available()
    install(monkeypatch, "sink", entries("sink", {"memory": "held"}))

    assert Registry[object]("sink").available() == ("memory",)
    assert Registry[object]("sink").get("memory") == "held"
    assert Registry[object]("rule").available() == kept
