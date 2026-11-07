import didactic.api as dx
import pytest

from lofi.lf.provers import ProofBudget
from lofi.spec.groundings.neural_proof import (
    NeuralProofScorerSpec,
    SimilarityKernelSettings,
    SimilarityKernelSpec,
)
from lofi.spec.provers import ProofBudgetSpec


def declaration() -> NeuralProofScorerSpec:
    return NeuralProofScorerSpec(
        order="first",
        instantiations=8,
        templates=(
            "identity",
            "conjunction-elimination",
            "conjunction-introduction",
            "modus-ponens",
        ),
        depth=3,
        width=12,
        similarity=SimilarityKernelSpec(kind="rbf", temperature=0.75),
        verification=ProofBudgetSpec(
            calls_per_step=8,
            wall_per_step=2.0,
            cpu_seconds=None,
        ),
    )


def test_neural_proof_resolves_search_and_verification_resources_once() -> None:
    resolved = declaration().settings()

    assert resolved.templates == (
        "identity",
        "conjunction-elimination",
        "conjunction-introduction",
        "modus-ponens",
    )
    assert resolved.order == "first"
    assert resolved.instantiations == 8
    assert resolved.depth == 3
    assert resolved.width == 12
    assert resolved.similarity == SimilarityKernelSettings("rbf", 0.75)
    assert resolved.verification == ProofBudget(
        calls_per_step=8,
        wall_per_step=2.0,
        cpu_seconds=None,
    )
    assert declaration().checkpointed()


def test_neural_proof_refuses_repeated_or_empty_rule_templates() -> None:
    with pytest.raises(dx.ValidationError, match="rule template twice"):
        declaration().with_(templates=("identity", "identity"))

    with pytest.raises(dx.ValidationError, match="at least one rule template"):
        declaration().with_(templates=())


@pytest.mark.parametrize(
    ("field", "value"),
    [("depth", 0), ("width", 0), ("instantiations", 0)],
)
def test_neural_proof_resources_are_positive(field: str, value: int) -> None:
    document = declaration().model_dump()
    document[field] = value

    with pytest.raises(dx.ValidationError, match=field):
        NeuralProofScorerSpec.model_validate(document)


def test_similarity_temperature_is_positive() -> None:
    with pytest.raises(dx.ValidationError, match="temperature"):
        declaration().with_(
            similarity=SimilarityKernelSpec(kind="cosine", temperature=0.0)
        )


def test_neural_proof_refuses_an_unknown_declared_order() -> None:
    with pytest.raises(dx.ValidationError, match="order"):
        declaration().with_(order="zeroth")


def test_the_similarity_kernel_names_an_installed_plugin() -> None:
    declared = SimilarityKernelSpec(kind="laplacian", temperature=0.5)

    assert declared.settings() == SimilarityKernelSettings("laplacian", 0.5)
