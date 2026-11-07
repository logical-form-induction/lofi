from typing import TYPE_CHECKING

import pytest
import torch

import lofi.train

if TYPE_CHECKING:
    from pathlib import Path


def test_the_published_surface_names_the_rewrites_a_run_asked_for() -> None:
    memo = lofi.train.Optimization("memo", lofi.train.Rewrite.VALUE)
    regime = lofi.train.OptInSet([memo])

    assert memo in regime
    assert len(regime) == 1
    assert list(regime) == [memo]
    assert regime != lofi.train.OptInSet()


def test_the_published_surface_refuses_a_switch_for_a_schedule_rewrite() -> None:
    batching = lofi.train.Optimization("cell-batching", lofi.train.Rewrite.SCHEDULE)

    assert not batching.rewrite.requires_opt_in

    with pytest.raises(ValueError, match="cell-batching"):
        lofi.train.OptInSet([batching])


def test_the_published_surface_exposes_optimization_policy_values() -> None:
    case = lofi.train.OptimizationResult(
        loss_finite=True,
        stepped=False,
        scale=None,
    )

    assert case.loss_finite
    assert not case.stepped
    assert case.scale is None
    assert lofi.train.OptimizationPolicy is not None


def test_the_published_surface_exposes_the_autocast_policy() -> None:
    scaling = lofi.train.GradScaling(
        init_scale=4.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=10
    )
    options = lofi.train.StepOptions(accumulation_steps=2, scaling=scaling)

    assert options.scaling == scaling
    assert lofi.train.OptimizationState({}).scaler is None
    assert lofi.train.AutocastPolicy is not None


def test_the_published_surface_exposes_configured_training() -> None:
    for name in (
        "ConfiguredModel",
        "ParserFittingPlan",
        "ParserFittingRecord",
        "PreparedTraining",
        "AccuracyStopping",
        "AccuracyStoppingDecision",
        "EarlyStopping",
        "EarlyStoppingState",
        "StoppingDecision",
        "EpochReport",
        "FittingReport",
        "FittingRecord",
        "BestCheckpointId",
        "ResumeCheckpointId",
        "fit_training",
        "prepare_training",
        "validate_epoch",
    ):
        assert name in lofi.train.__all__


def test_the_published_surface_exposes_the_loss_assembly() -> None:
    assert lofi.train.metric_key("coherence", "entropy") == "loss/coherence/entropy"
    assert issubclass(lofi.train.InactiveLossError, Exception)
    assert issubclass(lofi.train.MisnamedObjectiveError, Exception)

    with pytest.raises(ValueError, match="no objectives"):
        lofi.train.LossAssembly(())

    for name in (
        "LossPlugin",
        "LossResult",
        "WeightedObjective",
        "WeightedTerm",
        "resolve_loss_section",
    ):
        assert name in lofi.train.__all__


def test_the_published_surface_exposes_the_checkpoint_store(tmp_path: Path) -> None:
    store = lofi.train.DirectoryStore(tmp_path / "checkpoints")

    assert store.list() == ()
    assert issubclass(lofi.train.CorruptCheckpointError, lofi.train.CheckpointError)

    for name in ("DirectoryStore", "StateTree", "read_manifest", "write_manifest"):
        assert name in lofi.train.__all__

    assert "RestorableRandomStreams" in lofi.train.__all__


def test_the_published_surface_exposes_completed_step_resume() -> None:
    position = lofi.train.ResumePosition(7, 23)

    assert position.step == 7
    assert position.data_position == 23
    assert callable(lofi.train.resume_from_checkpoint)

    for name in ("ResumePosition", "resume_from_checkpoint"):
        assert name in lofi.train.__all__


def test_the_published_surface_exposes_the_pretraining_steps() -> None:
    request = lofi.train.Draw(1, 2, step=0, horizon=3)

    assert request.count == 2
    assert lofi.train.Trained("controller") == lofi.train.Trained("controller")

    for name in (
        "ControllerStep",
        "DistanceBatcher",
        "DistanceEncoderStep",
        "PairDraws",
        "RuleBatch",
        "TrainedController",
        "TypeBatch",
        "distance_batch_seed",
        "scheduled_depth",
    ):
        assert name in lofi.train.__all__


def test_the_published_surface_exposes_the_parser_and_warm_up_steps() -> None:
    request = lofi.train.Utterances((("a", "b"),), (((), ()),), 2)
    featurized = lofi.train.Featurized(("a",), 1)

    assert request.items == (("a", "b"),)
    assert request == lofi.train.Utterances([["a", "b"]], (((), ()),), 2)
    assert featurized.batch == ("a",)
    assert featurized.step == 1

    for name in (
        "ChartForward",
        "Featurized",
        "ParserStep",
        "TokenWarmupStep",
        "Utterances",
        "WarmupBatch",
        "transfer_rows",
    ):
        assert name in lofi.train.__all__


def test_the_published_surface_resolves_declared_parameter_groups() -> None:
    model = torch.nn.Linear(2, 2)
    groups = lofi.train.ParameterGroups([lofi.train.ParameterGroup("weights", "")])
    active = lofi.train.ActiveGroups(["weights"])

    resolution = lofi.train.resolve_groups(groups, active, model)

    assert resolution.trainable == {"weight", "bias"}
    assert resolution.frozen == frozenset()
    assert issubclass(lofi.train.GroupError, ValueError)

    for name in ("GroupResolution", "ParameterDeclaration", "declare"):
        assert name in lofi.train.__all__


def test_the_published_surface_exposes_alternating_steps_and_tasks() -> None:
    for name in ("AlternatingStep", "DatasetTask", "Fitted"):
        assert name in lofi.train.__all__


def test_the_published_surface_declares_the_stages_of_a_protocol() -> None:
    stage = lofi.train.Stage("chart", lofi.train.ActiveGroups(["chart.layers"]), 2)

    assert stage.epochs == 2
    assert stage.active == lofi.train.ActiveGroups(["chart.layers"])
    recorded = lofi.train.StageRecord(
        name="chart", active=("chart.layers",), position=0
    )

    assert recorded.position == 0

    for name in (
        "StageCompletion",
        "StageFitting",
        "StagedProtocol",
        "staged_manifest",
    ):
        assert name in lofi.train.__all__


def test_the_published_surface_exposes_the_objective_error_family() -> None:
    family = (
        lofi.train.EveryItemAbstainedError,
        lofi.train.ImpossibleNegativeBracketError,
        lofi.train.ImproperMassError,
        lofi.train.InvalidCutpointError,
        lofi.train.MissingOutputError,
        lofi.train.SparseDivergenceError,
        lofi.train.StochasticFactoringError,
        lofi.train.UncomputedCellError,
        lofi.train.UndeclaredOutputError,
    )

    assert all(issubclass(error, lofi.train.ObjectiveError) for error in family)
    assert {error.__name__ for error in (lofi.train.ObjectiveError, *family)} <= set(
        lofi.train.__all__
    )
