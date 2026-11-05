import pytest

from lofi.formalism._values import Budget, TheoryArtifact


def test_values_are_immutable() -> None:
    values = (
        Budget(
            depth=5,
            mass=1.0,
            beam=8,
            unary_rounds=1,
            threads=1,
            beta_fuel=1_000,
            max_category_arity=None,
            max_category_size=None,
        ),
        TheoryArtifact.from_specification("v1", b"{}"),
    )

    for value in values:
        with pytest.raises(AttributeError, match="immutable"):
            value.__delattr__("_digest")
