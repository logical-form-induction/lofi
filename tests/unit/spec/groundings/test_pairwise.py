from typing import Literal

import pytest

from lofi.control_variate import MovingControlVariateSpec
from lofi.spec.decoding import ExhaustiveDecodingSpec, SampleDecodingSpec
from lofi.spec.groundings import PairwiseGroundingSpec
from lofi.spec.provers import ProofBudgetSpec
from lofi.spec.readouts import ReadoutSource, TermsReadoutSpec
from lofi.spec.scorers import ProverScorerSpec


def grounding(*, sampled: bool) -> PairwiseGroundingSpec:
    return PairwiseGroundingSpec(
        premise="premise-forms",
        hypothesis="hypothesis-forms",
        control_variate=MovingControlVariateSpec(decay=0.9) if sampled else None,
        scorer=ProverScorerSpec(
            budget=ProofBudgetSpec(calls_per_step=4),
        ),
    )


def readouts(*, sampled: bool) -> dict[str, TermsReadoutSpec]:
    decoding = (
        SampleDecodingSpec(draws=2, rescore_tolerance=1e-5)
        if sampled
        else ExhaustiveDecodingSpec(residual_mass=0.1, cap=8)
    )
    return {
        name: TermsReadoutSpec(
            space="semantics", reads=ReadoutSource.INSIDE, decoding=decoding
        )
        for name in ("premise-forms", "hypothesis-forms")
    }


@pytest.mark.parametrize("estimate", ["exact", "sampled"])
def test_pairwise_grounding_requires_a_control_variate_exactly_for_sampled_forms(
    estimate: Literal["exact", "sampled"],
) -> None:
    sampled = estimate == "sampled"
    declared = grounding(sampled=sampled)

    declared.validate_readouts(readouts(sampled=sampled))
    assert declared.form_readouts() == ("premise-forms", "hypothesis-forms")
    assert tuple(declared.nested_components()) == ("scorer",)


def test_pairwise_grounding_refuses_a_missing_sampled_control_variate() -> None:
    with pytest.raises(ValueError, match="requires a control variate"):
        grounding(sampled=True).with_(control_variate=None).validate_readouts(
            readouts(sampled=True)
        )


def test_pairwise_grounding_refuses_mixed_form_estimates() -> None:
    mixed = readouts(sampled=False)
    mixed["hypothesis-forms"] = TermsReadoutSpec(
        space="semantics",
        reads=ReadoutSource.INSIDE,
        decoding=SampleDecodingSpec(draws=2, rescore_tolerance=1e-5),
    )

    with pytest.raises(ValueError, match="mixes readout estimates"):
        grounding(sampled=False).validate_readouts(mixed)
