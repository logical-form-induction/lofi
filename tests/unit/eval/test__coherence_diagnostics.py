from pathlib import Path

import pytest
from tests.conformance.evaluator import (
    HeldOutEvaluatorSuite,
    HeldOutSubject,
    packaged_held_out_pass,
)

from lofi.eval._coherence_diagnostics import (
    CoherenceDiagnosticGold,
    CoherenceDiagnosticPrediction,
    CoherenceDiagnosticsEvaluator,
)
from lofi.spec import PACKAGED_CONFIG_ROOT, RunSpecCodec
from lofi.spec.evaluators.coherence_diagnostics import CoherenceDiagnosticsSpec

CONFIG = (
    Path(__file__).parents[3] / "lofi/configs/run/synthetic-coherence-diagnostics.yaml"
)
CODEC = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT / "run", PACKAGED_CONFIG_ROOT))


def evaluator() -> CoherenceDiagnosticsEvaluator:
    """Return diagnostics bound to the packaged parser's held-out sites."""
    run = CODEC(CONFIG)
    declaration = run.evaluators["coherence-diagnostics"]
    assert isinstance(declaration, CoherenceDiagnosticsSpec)
    return CoherenceDiagnosticsEvaluator(
        packaged_held_out_pass().context,
        declaration.settings(),
    )


class TestCoherenceDiagnosticsObeyHeldOutEvaluatorLaws(HeldOutEvaluatorSuite):
    @pytest.fixture
    def held_out_subject(
        self,
    ) -> HeldOutSubject[CoherenceDiagnosticPrediction, CoherenceDiagnosticGold]:
        return HeldOutSubject(evaluator(), packaged_held_out_pass())


def test_every_diagnostic_family_is_computed_from_the_held_out_pass() -> None:
    held_out = packaged_held_out_pass()
    diagnostics = evaluator()
    predictions = {}

    for outputs in held_out.published:
        predictions.update(diagnostics.predict(outputs))

    for item, derivation in held_out.gold:
        predicted = predictions[item.item_id]
        gold = diagnostics.annotate(item, derivation)
        for metric in diagnostics.metrics:
            metric.update(predicted, gold, key=item.item_id)

    results = {metric.name: metric.compute() for metric in diagnostics.metrics}

    assert set(results) == {
        "coherence-loss-per-token",
        "coherence-loss-by-action",
        "coherence-loss-by-span-length",
        "controller-entropy-per-span",
        "decoder-entropy-per-span",
        "type-distribution-collapse-rate",
        "coherence-accuracy-correlation",
        "combinator-use-jsd",
        "root-mass-agreement",
    }
    assert all(result.n >= 0 for result in results.values())
    assert results["coherence-loss-by-action"].components
    assert results["coherence-loss-by-span-length"].components
    assert results["combinator-use-jsd"].components
    assert "mean-absolute-difference" in results["root-mass-agreement"].components


def test_an_unknown_objective_or_root_mass_is_refused_at_construction() -> None:
    run = CODEC(CONFIG)
    declaration = run.evaluators["coherence-diagnostics"]
    assert isinstance(declaration, CoherenceDiagnosticsSpec)
    settings = declaration.settings()
    held_out = packaged_held_out_pass()

    with pytest.raises(ValueError, match="unknown-objective"):
        CoherenceDiagnosticsEvaluator(
            held_out.context,
            settings._replace(objective="unknown-objective"),
        )

    with pytest.raises(ValueError, match="unknown-mass"):
        CoherenceDiagnosticsEvaluator(
            held_out.context,
            settings._replace(root_mass="unknown-mass"),
        )
