import didactic.api as dx
import pytest

from lofi.lf.models import SampledSettings
from lofi.spec.fuzzy_logic import FuzzyLogicSettings, FuzzyLogicSpec
from lofi.spec.scorers import ModelCheckingScorerSpec
from lofi.spec.structures import SampledStructureSourceSpec


def declaration() -> ModelCheckingScorerSpec:
    return ModelCheckingScorerSpec(
        source=SampledStructureSourceSpec(
            structures=10,
            carriers={"e": 2},
            density=0.5,
            full_function_space_cap=32,
            henkin_pool=4,
            seed=9,
            resample="once",
            atomic_sampling="uniform",
            denotation_streams="independent",
            function_sampling="table",
            truth_sampling="bernoulli",
        ),
        logic=FuzzyLogicSpec(
            tnorm="product",
            implication="residuum",
            quantifier="extreme",
            equality="crisp",
            margin=0.0,
        ),
        truth_threshold=0.75,
        abstain_band=0.1,
        premise_floor=0.5,
        unsatisfied_premise="undecided",
    )


def test_model_checking_resolves_every_declared_interior_policy() -> None:
    resolved = declaration().settings()

    assert resolved.source_kind == "sampled"
    assert isinstance(resolved.source, SampledSettings)
    assert resolved.logic == FuzzyLogicSettings(
        "product",
        "residuum",
        "extreme",
        "crisp",
        0.0,
    )
    assert resolved.truth_threshold == 0.75
    assert resolved.abstain_band == 0.1
    assert resolved.premise_floor == 0.5
    assert resolved.unsatisfied_premise == "undecided"


def test_a_model_checking_scorer_contributes_learned_checkpoint_state() -> None:
    assert declaration().checkpointed()


def test_an_abstention_band_cannot_cross_a_unit_interval_boundary() -> None:
    with pytest.raises(dx.ValidationError, match="abstention band"):
        declaration().with_(truth_threshold=0.1, abstain_band=0.2)
