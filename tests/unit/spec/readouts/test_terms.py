from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.model import ReadoutSpec
from lofi.spec.decoding import BeamDecodingSpec, ExhaustiveDecodingSpec
from lofi.spec.readouts import (
    DecodedMeaningSettings,
    DecodedMeaningSpec,
    ProofPushforwardSettings,
    ProofTermMeaningSettings,
    ProofTermMeaningSpec,
    ReadoutMass,
    ReadoutSource,
    TargetSource,
    TermsReadoutSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_a_term_readout_declares_marginal_rows_over_one_space() -> None:
    declared = ReadoutSpec.model_validate(
        {
            "kind": "terms",
            "space": "forms",
            "reads": "span",
            "targets": "grammar-goal",
            "output": "root-mass",
            "mass": "coupled",
        }
    )

    assert isinstance(declared, TermsReadoutSpec)
    assert declared == TermsReadoutSpec(
        space="forms",
        reads=ReadoutSource.SPAN,
        targets=TargetSource.GRAMMAR_GOAL,
        output="root-mass",
        mass=ReadoutMass.COUPLED,
    )
    assert declared.settings().decoding is None
    assert declared.settings().meanings == DecodedMeaningSettings("decoded")
    assert declared.required_chart_outputs() == {"spans", "alpha", "mu"}


def test_a_term_readout_declares_complete_term_decoding() -> None:
    declared = TermsReadoutSpec(
        space="forms", reads=ReadoutSource.INSIDE, decoding=BeamDecodingSpec(width=7)
    )

    assert declared.settings().decoding == BeamDecodingSpec(width=7).settings()
    assert declared.required_chart_outputs() == {"cells"}


def test_exhaustive_decoding_requests_its_optional_residual_override() -> None:
    inherited = TermsReadoutSpec(
        space="forms",
        reads=ReadoutSource.INSIDE,
        decoding=ExhaustiveDecodingSpec(cap=7),
    )
    overridden = inherited.with_(
        decoding=ExhaustiveDecodingSpec(residual_mass=0.05, cap=7)
    )
    inherited_request = inherited.approximation_request()
    overridden_request = overridden.approximation_request()

    assert inherited_request is not None
    assert overridden_request is not None
    assert inherited_request.residual_mass is None
    assert overridden_request.residual_mass == pytest.approx(0.05)


def test_a_term_readout_declares_proof_term_induction() -> None:
    declared = TermsReadoutSpec(
        space="forms",
        reads=ReadoutSource.INSIDE,
        decoding=BeamDecodingSpec(width=7),
        meanings=ProofTermMeaningSpec(weighting="multiplicity", residual_mass=0.999),
    )

    assert declared.settings().meanings == ProofTermMeaningSettings(
        "proof-term", "multiplicity", 0.999
    )
    assert declared.required_chart_outputs() == {"cells", "alpha", "actions"}


def test_proof_term_induction_requires_a_lexical_decoding_strategy() -> None:
    with pytest.raises(dx.ValidationError, match="proof-term meanings"):
        TermsReadoutSpec(
            space="forms",
            reads=ReadoutSource.INSIDE,
            meanings=ProofTermMeaningSpec(residual_mass=0.999),
        )


def test_proof_term_induction_declares_its_derived_cell_residual_bound() -> None:
    with pytest.raises(dx.ValidationError, match="residual_mass"):
        ProofTermMeaningSpec.model_validate({"kind": "proof-term"})


@pytest.mark.parametrize("bound", [-0.1, 1.0, float("nan")])
def test_a_derived_cell_residual_bound_is_a_probability_below_one(
    bound: float,
) -> None:
    with pytest.raises(dx.ValidationError, match="residual mass bound"):
        ProofTermMeaningSpec(residual_mass=bound)


@pytest.mark.parametrize(
    ("payload", "named"),
    [
        (
            {
                "kind": "terms",
                "space": "",
                "reads": "inside",
            },
            "space",
        ),
        (
            {"kind": "terms", "space": "forms", "reads": "inside", "output": ""},
            "output",
        ),
        (
            {
                "kind": "terms",
                "space": "forms",
                "reads": "inside",
                "meanings": {
                    "kind": "proof-term",
                    "weighting": "",
                    "residual_mass": 0.999,
                },
            },
            "meaning weighting",
        ),
        (
            {
                "kind": "terms",
                "space": "forms",
                "reads": "inside",
                "decoding": {"strategy": "greedy"},
                "targets": "grammar-goal",
            },
            "target mass",
        ),
    ],
)
def test_a_malformed_term_readout_is_refused(
    payload: dict[str, JsonValue], named: str
) -> None:
    with pytest.raises(dx.ValidationError, match=named):
        ReadoutSpec.model_validate(payload)


def test_a_term_readout_requires_an_explicit_chart_representation() -> None:
    with pytest.raises(dx.ValidationError, match="reads"):
        ReadoutSpec.model_validate({"kind": "terms", "space": "forms"})


def test_a_term_readout_publishes_each_target_mass_under_its_declared_name() -> None:
    untargeted = TermsReadoutSpec(space="categories", reads=ReadoutSource.INSIDE)
    targeted = untargeted.with_(targets=TargetSource.GRAMMAR_GOAL)
    renamed = targeted.with_(output="sentences")

    assert untargeted.published_masses() == {"grammar-goal"}
    assert targeted.published_masses() == {"grammar-goal"}
    assert renamed.published_masses() == {"sentences"}


def test_each_meaning_route_declares_the_chart_outputs_it_reads() -> None:
    decoded = DecodedMeaningSettings("decoded")
    proof = ProofTermMeaningSpec(residual_mass=0.5).settings()

    assert decoded.chart_outputs() == frozenset()
    assert proof.chart_outputs() == {"alpha", "actions"}


def test_only_proof_term_induction_declares_a_pushforward() -> None:
    assert DecodedMeaningSpec().settings().pushforward() is None
    assert ProofTermMeaningSpec(
        weighting="distinct", residual_mass=0.25
    ).settings().pushforward() == ProofPushforwardSettings("distinct", 0.25)
