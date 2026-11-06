import didactic.api as dx
import pytest

from lofi.lf import SequentLimits
from lofi.lf.provers import ProofBudget
from lofi.spec.groundings.guided_proof import GuidedProofScorerSpec
from lofi.spec.provers import ProofBudgetSpec


def declaration() -> GuidedProofScorerSpec:
    return GuidedProofScorerSpec(
        readout="forms",
        order="first",
        steps=64,
        depth=8,
        top_k=4,
        instantiations=8,
        hidden=12,
        verification=ProofBudgetSpec(
            calls_per_step=8,
            wall_per_step=2.0,
            cpu_seconds=None,
        ),
    )


def test_guided_proof_resolves_every_search_and_verification_bound_once() -> None:
    resolved = declaration().settings()

    assert resolved.readout == "forms"
    assert resolved.order == "first"
    assert resolved.limits == SequentLimits(
        steps=64,
        depth=8,
        top_k=4,
        instantiations=8,
    )
    assert resolved.hidden == 12
    assert resolved.verification == ProofBudget(
        calls_per_step=8,
        wall_per_step=2.0,
        cpu_seconds=None,
    )
    assert declaration().checkpointed()


@pytest.mark.parametrize(
    "field",
    ["steps", "depth", "top_k", "instantiations", "hidden"],
)
def test_guided_proof_resources_are_positive(field: str) -> None:
    document = declaration().model_dump()
    document[field] = 0

    with pytest.raises(dx.ValidationError, match=field):
        GuidedProofScorerSpec.model_validate(document)


def test_guided_proof_requires_a_named_readout_and_supported_order() -> None:
    with pytest.raises(dx.ValidationError, match="readout"):
        declaration().with_(readout="")

    with pytest.raises(dx.ValidationError, match="order"):
        declaration().with_(order="zeroth")
