import didactic.api as dx
import pytest

from lofi.lf.models import CountermodelSettings
from lofi.lf.provers import ProofBudget
from lofi.spec.countermodels import CountermodelSpec
from lofi.spec.provers import ProofBudgetSpec
from lofi.spec.scorers import ProverScorerSpec


def declaration() -> ProverScorerSpec:
    return ProverScorerSpec(
        budget=ProofBudgetSpec(
            calls_per_step=8,
            wall_per_step=2.0,
            cpu_seconds=None,
        ),
        countermodels=CountermodelSpec(
            method="exhaustive",
            carriers={"e": 2, "v": 1},
            structures=32,
            seed=7,
            full_function_space_cap=64,
        ),
    )


def test_prover_scorer_resolves_countermodel_search_once_at_the_boundary() -> None:
    resolved = declaration().settings()

    assert resolved.budget == ProofBudget(
        calls_per_step=8,
        wall_per_step=2.0,
        cpu_seconds=None,
    )
    assert isinstance(resolved.countermodels, CountermodelSettings)
    assert resolved.countermodels.method == "exhaustive"
    assert resolved.countermodels.carriers == {"e": 2, "v": 1}
    assert resolved.countermodels.structures == 32
    assert resolved.countermodels.seed == 7
    assert resolved.countermodels.full_function_space_cap == 64


def test_a_prover_scorer_contributes_no_checkpoint_state() -> None:
    assert not declaration().checkpointed()


def test_countermodel_resources_are_positive() -> None:
    countermodels = declaration().countermodels

    assert countermodels is not None

    with pytest.raises(dx.ValidationError, match="at least one structure"):
        declaration().with_(countermodels=countermodels.with_(structures=0))


def test_a_prover_scorer_refuses_the_deleted_beta_fuel_setting() -> None:
    document = declaration().model_dump()
    document["beta_fuel"] = 128

    with pytest.raises(dx.ValidationError, match="beta_fuel"):
        ProverScorerSpec.model_validate(document)
