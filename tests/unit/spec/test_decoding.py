from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.decoding import (
    BeamDecodingSpec,
    DecodingSpec,
    ExhaustiveDecodingSpec,
    FullSamplingSpec,
    GreedyDecodingSpec,
    NucleusSamplingSpec,
    SampleDecodingSpec,
    TopKSamplingSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_every_decoding_strategy_lowers_to_one_frozen_current_value() -> None:
    declarations = (
        GreedyDecodingSpec(),
        BeamDecodingSpec(width=4),
        ExhaustiveDecodingSpec(residual_mass=0.01, cap=128),
        SampleDecodingSpec(
            restriction=FullSamplingSpec(), draws=3, rescore_tolerance=1e-5
        ),
        SampleDecodingSpec(
            restriction=TopKSamplingSpec(k=5, temperature=0.7),
            draws=3,
            rescore_tolerance=1e-5,
        ),
        SampleDecodingSpec(
            restriction=NucleusSamplingSpec(p=0.9, temperature=0.8),
            draws=3,
            rescore_tolerance=1e-5,
        ),
    )

    settings = tuple(declaration.settings() for declaration in declarations)

    assert settings[1].width == 4
    assert settings[2].exact_residual == pytest.approx(1e-6)
    assert ExhaustiveDecodingSpec(cap=2, exact_residual=0.01).settings() == (
        ExhaustiveDecodingSpec(cap=2, exact_residual=0.01)
        .settings()
        ._replace(exact_residual=0.01)
    )
    assert settings[3].sampling is not None
    assert settings[3].sampling.kind == "full"
    assert settings[4].sampling is not None
    assert settings[4].sampling.k == 5
    assert settings[5].sampling is not None
    assert settings[5].sampling.p == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("document", "variant"),
    [
        ({"strategy": "greedy"}, GreedyDecodingSpec),
        ({"strategy": "beam", "width": 8}, BeamDecodingSpec),
        (
            {"strategy": "exhaustive", "residual_mass": 0.001, "cap": 32},
            ExhaustiveDecodingSpec,
        ),
        (
            {
                "strategy": "sample",
                "restriction": {"kind": "top-k", "k": 5},
                "draws": 4,
                "rescore_tolerance": 1e-5,
            },
            SampleDecodingSpec,
        ),
    ],
)
def test_each_strategy_is_reached_through_the_closed_declaration(
    document: dict[str, JsonValue], variant: type[DecodingSpec]
) -> None:
    assert type(DecodingSpec.model_validate(document)) is variant


@pytest.mark.parametrize(
    ("document", "field"),
    [
        ({"strategy": "beam", "width": 0}, "width"),
        (
            {"strategy": "exhaustive", "residual_mass": 1.0, "cap": 3},
            "residual_mass",
        ),
        ({"strategy": "exhaustive", "residual_mass": 0.1, "cap": 0}, "cap"),
        (
            {"strategy": "exhaustive", "cap": 3, "exact_residual": 1.0},
            "exact_residual",
        ),
        (
            {"strategy": "sample", "restriction": {"kind": "top-k", "k": 0}},
            "k",
        ),
        (
            {
                "strategy": "sample",
                "restriction": {"kind": "nucleus", "p": 0.0},
            },
            "p",
        ),
    ],
)
def test_invalid_strategy_settings_are_refused_at_the_document_boundary(
    document: dict[str, JsonValue], field: str
) -> None:
    with pytest.raises(dx.ValidationError, match=field):
        DecodingSpec.model_validate(document)
