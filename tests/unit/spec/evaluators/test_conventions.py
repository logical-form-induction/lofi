import didactic.api as dx
import pytest

from lofi.spec.evaluators.conventions import ConventionSettings, ConventionSpec

DECLARED: dict[str, str | int] = {
    "punctuation": "strip",
    "unary_chains": "collapse",
    "trivial_spans": "drop",
    "minimum_length": 3,
    "aggregation": "sentence",
    "coverage": "zero",
    "self_arcs": "forbid",
    "gold_derivation": "canonical",
}


def test_every_declared_convention_is_kept() -> None:
    assert ConventionSpec.model_validate(DECLARED).model_dump() == DECLARED


@pytest.mark.parametrize("factor", sorted(DECLARED))
def test_an_omitted_convention_is_refused(factor: str) -> None:
    omitted = {name: value for name, value in DECLARED.items() if name != factor}

    with pytest.raises(dx.ValidationError):
        ConventionSpec.model_validate(omitted)


@pytest.mark.parametrize(
    ("factor", "value"),
    [
        ("punctuation", "drop"),
        ("aggregation", "token"),
        ("gold_derivation", "silver"),
        ("minimum_length", -1),
    ],
)
def test_a_value_outside_its_convention_is_refused(
    factor: str, value: str | int
) -> None:
    with pytest.raises(dx.ValidationError):
        ConventionSpec.model_validate(DECLARED | {factor: value})


def test_an_undeclared_convention_is_refused() -> None:
    with pytest.raises(dx.ValidationError):
        ConventionSpec.model_validate(DECLARED | {"labels": "keep"})


def test_the_settings_restate_every_declared_convention_by_name() -> None:
    settings = ConventionSpec.model_validate(DECLARED).settings()

    assert settings == ConventionSettings(
        punctuation="strip",
        unary_chains="collapse",
        trivial_spans="drop",
        minimum_length=3,
        aggregation="sentence",
        coverage="zero",
        self_arcs="forbid",
        gold_derivation="canonical",
    )
    assert settings._asdict() == DECLARED
