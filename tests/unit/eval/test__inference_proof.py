from typing import TYPE_CHECKING, NamedTuple, cast

import pytest
import torch
from tests.language import language

from lofi.eval._inference_proof import prove_inference_beam
from lofi.lf import InferenceCell, ProverBound, TruthTableOracle
from lofi.lf.provers import ProofBudget, ProofService
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.spec.evaluators.inference import InferenceProofSettings
from lofi.spec.language import (
    ConstantSpec,
    LanguageIso,
    PrefixNotationSpec,
    registered_language,
)
from lofi.terms import Normalization

if TYPE_CHECKING:
    from lofi.data.synthetic import InterpretationFunction
    from lofi.spec.language import LanguageAssembly


class Interpretation(NamedTuple):
    language: LanguageAssembly
    service: ProofService
    normalization: Normalization


def interpreted() -> Interpretation:
    assembly = language(("rain", "wet"))
    service = ProofService(
        lambda: TruthTableOracle(
            assembly.store,
            assembly.profile,
            bound=ProverBound("truth-table", assembly.profile.fragment),
            atoms=4,
        ),
        profile=assembly.profile,
        workers=1,
        batch_size=8,
        timeout=5.0,
    )
    return Interpretation(
        assembly,
        service,
        Normalization(
            fuel=40,
            order="leftmost-outermost",
            eta="contract",
            associative_commutative=(),
        ),
    )


def interpreted_with_two_negations() -> Interpretation:
    source = registered_language("neo-davidsonian")
    assembly = LanguageIso().forward(
        source.with_(
            constants=(
                *source.constants,
                ConstantSpec(
                    name="complement",
                    type="t -> t",
                    classes=("sentential", "negation"),
                    notation=PrefixNotationSpec(
                        ascii="complement",
                        unicode="∁",
                        precedence=6,
                    ),
                ),
                ConstantSpec(name="rain", type="t"),
                ConstantSpec(name="wet", type="t"),
            ),
            logic=source.logic.with_(negation="not"),
        )
    )
    service = ProofService(
        lambda: TruthTableOracle(
            assembly.store,
            assembly.profile,
            bound=ProverBound("truth-table", assembly.profile.fragment),
            atoms=4,
        ),
        profile=assembly.profile,
        workers=1,
        batch_size=8,
        timeout=5.0,
    )
    return Interpretation(
        assembly,
        service,
        Normalization(
            fuel=40,
            order="leftmost-outermost",
            eta="contract",
            associative_commutative=(),
        ),
    )


def beam(assembly: LanguageAssembly, texts: tuple[str, ...]) -> WeightedTerms:
    return WeightedTerms(
        tuple(assembly.read(text) for text in texts),
        torch.tensor([-(index + 1.0) for index in range(len(texts))]),
        torch.tensor(float("-inf")),
        Estimate.TRUNCATED,
    )


@pytest.mark.parametrize(
    ("premise", "hypothesis", "cell"),
    [
        ("rain ∧ wet", "rain", InferenceCell.ENTAILS),
        ("rain", "¬rain", InferenceCell.CONTRADICTS),
        ("rain", "wet", InferenceCell.NEUTRAL),
        ("rain ∧ ¬rain", "wet", InferenceCell.DEGENERATE),
    ],
)
def test_proof_beam_preserves_all_four_joint_cells(
    premise: str,
    hypothesis: str,
    cell: InferenceCell,
) -> None:
    interpretation = interpreted()
    assembly = interpretation.language
    try:
        result = prove_inference_beam(
            cast("InterpretationFunction", interpretation),
            InferenceProofSettings(ProofBudget(), 1),
            (assembly.read(premise), assembly.read(hypothesis)),
            beam(assembly, (premise,)),
            beam(assembly, (hypothesis,)),
        )
        assert result.reference is cell
        assert result.candidates == (cell,)
        assert result.calls == 2
        assert result.memo_hits == 0
        assert result.questions == 2
        assert result.cpu_seconds >= 0.0
    finally:
        interpretation.service.close()


def test_spent_proof_budget_retains_abstentions_and_their_causes() -> None:
    interpretation = interpreted()
    assembly = interpretation.language
    try:
        result = prove_inference_beam(
            cast("InterpretationFunction", interpretation),
            InferenceProofSettings(ProofBudget(calls_per_step=0), 1),
            (assembly.read("rain"), assembly.read("wet")),
            beam(assembly, ("rain",)),
            beam(assembly, ("wet",)),
        )
        assert result.reference is None
        assert result.candidates == (None,)
        assert result.calls == 0
        assert dict(result.abstentions)["budget-spent"] == 2
    finally:
        interpretation.service.close()


def test_proof_beam_uses_designated_negation_among_multiple_roles() -> None:
    interpretation = interpreted_with_two_negations()
    assembly = interpretation.language
    try:
        result = prove_inference_beam(
            cast("InterpretationFunction", interpretation),
            InferenceProofSettings(ProofBudget(), 1),
            (assembly.read("rain"), assembly.read("wet")),
            beam(assembly, ("rain",)),
            beam(assembly, ("wet",)),
        )

        assert result.reference is InferenceCell.NEUTRAL
        assert result.candidates == (InferenceCell.NEUTRAL,)
    finally:
        interpretation.service.close()


def test_joint_beam_uses_published_weights_and_declared_depth() -> None:
    interpretation = interpreted()
    assembly = interpretation.language
    try:
        premise = beam(assembly, ("rain",))
        hypothesis = WeightedTerms(
            (assembly.read("rain"), assembly.read("wet")),
            torch.tensor([-4.0, -1.0]),
            torch.tensor(float("-inf")),
            Estimate.TRUNCATED,
        )
        result = prove_inference_beam(
            cast("InterpretationFunction", interpretation),
            InferenceProofSettings(ProofBudget(), 2),
            (assembly.read("rain"), assembly.read("rain")),
            premise,
            hypothesis,
        )
        assert result.candidates == (InferenceCell.NEUTRAL, InferenceCell.ENTAILS)
    finally:
        interpretation.service.close()
