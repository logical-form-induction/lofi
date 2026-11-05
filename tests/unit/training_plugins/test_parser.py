from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import pytest

from lofi.data import ForcedChoice, Unlabeled
from lofi.spec import (
    LossSection,
    RunSpec,
    RunSpecCodec,
    WeightedLossSpec,
    config_search_path,
)
from lofi.spec.constraint_policy import PolicyDomain
from lofi.spec.constraint_sources import (
    ClampSpec,
    ConstraintSourceSpec,
    GoldDerivationConstraintSourceSpec,
    GoldLexiconConstraintSourceSpec,
    GoldLogicalFormConstraintSourceSpec,
)
from lofi.spec.featurizers.derivation import DerivationFeaturizerSpec
from lofi.spec.featurizers.pair import PairFeaturizerSpec
from lofi.spec.losses.pinning import PinningLossSpec
from lofi.spec.losses.vector_anchor import VectorAnchorLossSpec
from lofi.spec.trainers.parser import ParserTrainerSpec
from lofi.training_plugins import (
    FeaturizerMismatchError,
    TrainingLength,
    TrainingUnit,
)
from lofi.training_plugins.parser import PLUGIN

if TYPE_CHECKING:
    from lofi.spec import FeaturizerSpec
    from lofi.spec.components.loss import LossSpec

CONFIG = Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-parser.yaml"
PINNING = CONFIG.with_name("synthetic-pinning-constituent.yaml")
GROUNDED = CONFIG.parents[3] / "experiments/l4/l4-run.yaml"
PACKAGED_CODEC = RunSpecCodec(search_path=(CONFIG.parent, CONFIG.parent.parent))


def _constraint_run(
    objectives: tuple[LossSpec, ...],
    *,
    require_constituent: bool,
    source_kind: Literal["derivation", "lexicon", "logical-form"] = "derivation",
) -> RunSpec:
    """Return one run whose source and constraint objectives are declared exactly."""
    run = RunSpecCodec(search_path=config_search_path(CONFIG.parents[2]))(PINNING)
    section = run.data.datasets["train"]
    clamp = ClampSpec(require_constituent=require_constituent)
    source: ConstraintSourceSpec
    match source_kind:
        case "derivation":
            source = cast(
                "GoldDerivationConstraintSourceSpec", section.constraint_source
            ).with_(clamp=clamp)
        case "lexicon":
            source = GoldLexiconConstraintSourceSpec(
                policy=PolicyDomain.SYNTACTIC, clamp=clamp
            )
        case "logical-form":
            source = GoldLogicalFormConstraintSourceSpec(
                policy=PolicyDomain.SEMANTIC, clamp=clamp
            )
    loss = cast("LossSection", run.loss)
    terms = tuple(
        term for term in loss.terms if not isinstance(term.objective, PinningLossSpec)
    )

    return run.with_(
        data=run.data.with_(
            datasets={"train": section.with_(constraint_source=source)}
        ),
        loss=loss.with_(
            terms=(
                *terms,
                *(
                    WeightedLossSpec(objective=objective, name=f"constraint-{index}")
                    for index, objective in enumerate(objectives)
                ),
            )
        ),
    )


def _pinning_run(
    *,
    require_constituent: bool,
    readouts: tuple[str | None, ...],
    source_kind: Literal["derivation", "lexicon", "logical-form"] = "derivation",
) -> RunSpec:
    """Return one pinning run with an exact source and objective route."""
    return _constraint_run(
        tuple(PinningLossSpec(readout=readout) for readout in readouts),
        require_constituent=require_constituent,
        source_kind=source_kind,
    )


STREAMS = (
    "training:train:validation-split",
    "training:train:training-batches",
    "training:train:validation-batches",
)


def test_the_parser_plugin_checks_and_names_its_sampler_streams() -> None:
    run_spec = PACKAGED_CODEC(CONFIG)

    assert PLUGIN.spec is ParserTrainerSpec
    assert PLUGIN.plan(run_spec.trainer, run_spec) == (
        "training:train:validation-split",
        "training:train:training-batches",
        "training:train:validation-batches",
    )


def test_the_parser_plugin_prices_a_run_at_its_declared_epochs() -> None:
    trainer = PACKAGED_CODEC(CONFIG).trainer

    assert isinstance(trainer, ParserTrainerSpec)
    assert PLUGIN.price(trainer.with_(epochs=7)) == TrainingLength(
        7, TrainingUnit.EPOCHS
    )


def test_negative_bracket_pinning_consumes_constraints_without_a_readout() -> None:
    bracket_run = _pinning_run(
        require_constituent=False,
        readouts=(None,),
        source_kind="lexicon",
    )

    assert PLUGIN.plan(bracket_run.trainer, bracket_run) == STREAMS


def test_gold_derivation_negative_brackets_are_refused_before_training() -> None:
    bracket_run = _pinning_run(require_constituent=False, readouts=(None,))

    with pytest.raises(
        ValueError,
        match=(
            r"constraint source 'gold-derivation' cannot serve the "
            r"'negative-bracket' route: .*every gold constituent"
        ),
    ):
        PLUGIN.plan(bracket_run.trainer, bracket_run)


def test_semantic_negative_brackets_need_no_logical_form_readout() -> None:
    run = _constraint_run(
        (PinningLossSpec(readout=None),),
        require_constituent=False,
        source_kind="logical-form",
    )

    assert PLUGIN.plan(run.trainer, run) == STREAMS


def test_semantic_target_constraints_still_need_a_logical_form_readout() -> None:
    run = _constraint_run(
        (PinningLossSpec(readout="types"),),
        require_constituent=True,
        source_kind="logical-form",
    )

    with pytest.raises(ValueError, match="semantic policy over sort"):
        PLUGIN.plan(run.trainer, run)


@pytest.mark.parametrize(
    "objectives",
    [
        (VectorAnchorLossSpec(readout="types"),),
        (PinningLossSpec(readout="types"), VectorAnchorLossSpec(readout="types")),
        (PinningLossSpec(readout="types"), PinningLossSpec(readout="types")),
    ],
    ids=["vector-anchor", "pinning-and-vector-anchor", "two-pinning"],
)
def test_every_consumer_of_the_target_route_is_planned_together(
    objectives: tuple[LossSpec, ...],
) -> None:
    run = _constraint_run(objectives, require_constituent=True)

    assert PLUGIN.plan(run.trainer, run) == STREAMS


def test_vector_anchors_cannot_consume_negative_brackets() -> None:
    run = _constraint_run(
        (PinningLossSpec(readout=None), VectorAnchorLossSpec(readout="types")),
        require_constituent=False,
    )

    with pytest.raises(
        ValueError, match=r"'negative-bracket'.*\['negative-bracket', 'target'\]"
    ):
        PLUGIN.plan(run.trainer, run)


@pytest.mark.parametrize(
    "case",
    [
        (True, (), r"at least one 'target' constraint consumer"),
        (True, (None,), r"'target' route.*\['negative-bracket'\]"),
        (False, ("types",), r"'negative-bracket' route.*\['target'\]"),
        (True, ("types", None), r"'target' route.*\['negative-bracket', 'target'\]"),
    ],
    ids=["missing", "bracket-for-target", "target-for-bracket", "mixed"],
)
def test_the_parser_refuses_a_missing_mismatched_or_mixed_constraint_route(
    case: tuple[bool, tuple[str | None, ...], str],
) -> None:
    require_constituent, readouts, message = case
    run = _pinning_run(
        require_constituent=require_constituent,
        readouts=readouts,
    )

    with pytest.raises(ValueError, match=message):
        PLUGIN.plan(run.trainer, run)


def test_nonconstraint_losses_do_not_require_a_constraint_source() -> None:
    run = PACKAGED_CODEC(CONFIG)

    assert run.data.datasets["train"].constraint_source is None
    assert all(
        not isinstance(term.objective, PinningLossSpec)
        for term in cast("LossSection", run.loss).terms
    )
    assert PLUGIN.plan(run.trainer, run)


def _featurized(run: RunSpec, featurizer: FeaturizerSpec, /) -> RunSpec:
    """Return the run with its training dataset read through another featurizer."""
    section = run.data.datasets["train"]

    return run.with_(
        data=run.data.with_(datasets={"train": section.with_(featurizer=featurizer)})
    )


def test_the_parser_presents_unlabeled_sentences_to_its_featurizer() -> None:
    run = _featurized(PACKAGED_CODEC(CONFIG), PairFeaturizerSpec())

    with pytest.raises(FeaturizerMismatchError) as refused:
        PLUGIN.plan(run.trainer, run)

    assert refused.value.role == "train"
    assert refused.value.presented is Unlabeled
    assert refused.value.accepted == frozenset({ForcedChoice})
    assert str(refused.value) == (
        "data.datasets.train.featurizer: the 'parser' trainer presents "
        "Unlabeled examples, but the 'pair' featurizer accepts ForcedChoice"
    )


def test_a_grounded_parser_presents_relational_pairs_to_its_featurizer() -> None:
    run = RunSpecCodec(search_path=config_search_path(CONFIG.parents[3]))(GROUNDED)

    assert PLUGIN.plan(run.trainer, run) == STREAMS

    refused = _featurized(run, DerivationFeaturizerSpec())

    with pytest.raises(
        FeaturizerMismatchError,
        match="presents Relational examples, but the 'derivation' featurizer",
    ):
        PLUGIN.plan(refused.trainer, refused)
