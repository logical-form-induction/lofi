import pytest

from lofi.registry._errors import (
    RegistryError,
    TagCollisionError,
    UnknownTagError,
)


def test_a_registry_error_records_its_family() -> None:
    refusal = RegistryError("registration failed", family="parser")

    assert refusal.family == "parser"
    assert str(refusal) == "registration failed"


def test_an_unknown_tag_lists_the_available_tags() -> None:
    refusal = UnknownTagError(
        family="parser", tag="missing", available=("cky", "diora")
    )

    assert refusal.tag == "missing"
    assert refusal.available == ("cky", "diora")
    assert "cky, diora" in str(refusal)


def test_an_unknown_tag_reports_an_empty_registry() -> None:
    refusal = UnknownTagError(family="parser", tag="missing", available=())

    assert "holds nothing" in str(refusal)


def test_a_collision_names_each_distribution() -> None:
    refusal = TagCollisionError(
        family="parser", tag="cky", distributions=("one", "two")
    )

    assert refusal.tag == "cky"
    assert refusal.distributions == ("one", "two")
    assert "one and two" in str(refusal)


def test_a_collision_requires_two_distributions() -> None:
    with pytest.raises(ValueError, match="needs two claimants"):
        TagCollisionError(family="parser", tag="cky", distributions=("one",))
