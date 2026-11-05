from pathlib import Path
from typing import TYPE_CHECKING, Final

import didactic.api as dx
import pytest

from lofi.spec.components.counterfactuals import (
    COUNTERFACTUAL_RENDERINGS,
    CounterfactualFilterSettings,
    CounterfactualSettings,
    CounterfactualsSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue

DECLARED: Final[dict[str, JsonValue]] = {
    "sentences": "candidates.txt",
    "unary_depth": 1,
    "readings": 8,
    "render": "declared",
    "pieces": "all",
    "specials": "exclude",
    "contiguous": True,
    "seed": 4,
}


def test_a_declaration_lowers_to_its_settings() -> None:
    assert CounterfactualsSpec.model_validate(DECLARED).settings() == (
        CounterfactualSettings(
            sentences=Path("candidates.txt"),
            unary_depth=1,
            readings=8,
            render="declared",
            pieces="all",
            specials="exclude",
            contiguous=True,
            seed=4,
            filter=None,
        )
    )


@pytest.mark.parametrize("field", sorted(DECLARED))
def test_every_field_is_required(field: str) -> None:
    partial = {key: value for key, value in DECLARED.items() if key != field}

    with pytest.raises(dx.ValidationError, match=field):
        CounterfactualsSpec.model_validate(partial)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sentences", ""),
        ("unary_depth", -1),
        ("readings", 1),
        ("render", "infix"),
        ("pieces", "middle"),
        ("specials", "drop"),
        ("seed", -1),
    ],
)
def test_an_out_of_range_value_is_refused(field: str, value: JsonValue) -> None:
    with pytest.raises(dx.ValidationError, match=field):
        CounterfactualsSpec.model_validate({**DECLARED, field: value})


def test_an_unknown_field_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="normalizer"):
        CounterfactualsSpec.model_validate({**DECLARED, "normalizer": "none"})


def test_every_rendering_is_declarable() -> None:
    for render in COUNTERFACTUAL_RENDERINGS:
        settings = CounterfactualsSpec.model_validate(
            {**DECLARED, "render": render}
        ).settings()

        assert settings.render == render


FILTER: Final[dict[str, JsonValue]] = {
    "scores": "scores.jsonl",
    "source": {"name": "public-ratings", "public": True, "fit_signature": None},
    "primary": 0.5,
    "sweep": [0.1, 0.3, 0.5, 0.7, 0.9],
}


def test_a_declared_score_filter_lowers_to_its_settings() -> None:
    settings = CounterfactualsSpec.model_validate(
        {**DECLARED, "filter": FILTER}
    ).settings()

    assert settings.filter == CounterfactualFilterSettings(
        scores=Path("scores.jsonl"),
        source="public-ratings",
        public=True,
        fit_signature=None,
        primary=0.5,
        sweep=(0.1, 0.3, 0.5, 0.7, 0.9),
    )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"sweep": [0.1, 0.3, 0.5, 0.7]}, "five"),
        ({"sweep": [0.1, 0.3, 0.5, 0.7, 0.7]}, "distinct"),
        ({"sweep": [0.1, 0.3, 0.4, 0.7, 0.9]}, "primary"),
        ({"primary": float("nan")}, "finite"),
        (
            {"source": {"name": "fitted", "public": True, "fit_signature": "ccg"}},
            "fit signature",
        ),
        (
            {"source": {"name": "private", "public": False, "fit_signature": None}},
            "public",
        ),
        ({"scores": ""}, "scores"),
    ],
)
def test_a_score_filter_that_cannot_support_its_primary_arm_is_refused(
    change: dict[str, JsonValue], match: str
) -> None:
    with pytest.raises(dx.ValidationError, match=match):
        CounterfactualsSpec.model_validate({**DECLARED, "filter": {**FILTER, **change}})
