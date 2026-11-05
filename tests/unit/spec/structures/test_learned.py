"""Tests for the learned finite-structure source declaration."""

import didactic.api as dx
import pytest

from lofi.spec.structures import LearnedSettings, LearnedStructureSourceSpec


def declaration() -> LearnedStructureSourceSpec:
    return LearnedStructureSourceSpec(
        structures=2,
        carriers={"e": 2, "v": 1},
        full_function_space_cap=64,
        henkin_pool=8,
        denotations="free",
        seed=17,
        initialization_scale=0.5,
    )


def test_a_learned_source_resolves_immutable_interior_settings() -> None:
    settings = declaration().settings()

    assert isinstance(settings, LearnedSettings)
    assert settings.structures == 2
    assert settings.carriers == {"e": 2, "v": 1}
    assert settings.denotations == "free"
    assert settings.seed == 17
    assert settings.initialization_scale == 0.5
    assert not isinstance(settings.carriers, dict)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("structures", 0),
        ("full_function_space_cap", 0),
        ("henkin_pool", 0),
        ("seed", -1),
        ("carriers", {}),
        ("carriers", {"e": 0}),
        ("initialization_scale", 0.0),
    ],
)
def test_a_learned_source_refuses_invalid_frame_settings(
    field: str,
    value: int | dict[str, int],
) -> None:
    with pytest.raises(dx.ValidationError):
        declaration().with_(**{field: value})
