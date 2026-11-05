"""Tests for the pair-scorer declaration public surface."""

from lofi.spec.scorers import (
    ModelCheckingScorerSettings,
    ModelCheckingScorerSpec,
    ProverScorerSettings,
    ProverScorerSpec,
)
from lofi.spec.scorers.model_checking import (
    ModelCheckingScorerSettings as DefinedModelCheckingScorerSettings,
)
from lofi.spec.scorers.model_checking import (
    ModelCheckingScorerSpec as DefinedModelCheckingScorerSpec,
)
from lofi.spec.scorers.prover import ProverScorerSettings as DefinedProverScorerSettings
from lofi.spec.scorers.prover import ProverScorerSpec as DefinedProverScorerSpec


def test_the_scorer_surface_publishes_the_current_declarations() -> None:
    assert ModelCheckingScorerSettings is DefinedModelCheckingScorerSettings
    assert ModelCheckingScorerSpec is DefinedModelCheckingScorerSpec
    assert ProverScorerSettings is DefinedProverScorerSettings
    assert ProverScorerSpec is DefinedProverScorerSpec
