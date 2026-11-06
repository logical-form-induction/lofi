"""The packaged CCG supervision regimes compose and train."""

import math
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.declarations import train_spec

from lofi.data.synthetic.spec.corpus import SyntheticCorpusDecl
from lofi.run import packaged_options
from lofi.spec import RunSpec, RunSpecCodec, config_search_path
from lofi.spec.trainers import ParserTrainerSpec

if TYPE_CHECKING:
    from lofi.spec.grammar import RootGoalSpec

ROOT: Final = Path(__file__).resolve().parents[2]
RUNS: Final = ROOT / "lofi/configs/run"
REGIMES: Final = (
    "synthetic-positive-only",
    "synthetic-certified-negatives",
    "synthetic-labeled-strings",
)


def _regime(name: str, *overrides: str) -> RunSpec:
    """Compose one packaged regime through the public run codec."""
    return RunSpecCodec(search_path=config_search_path(ROOT))(
        RUNS / f"{name}.yaml",
        overrides=overrides,
    )


def _corpus(run: RunSpec) -> SyntheticCorpusDecl:
    """Return the regime's synthetic training corpus."""
    declared = run.data.datasets["train"].dataset

    if not isinstance(declared, SyntheticCorpusDecl):
        msg = f"a CCG regime holds {type(declared).__name__}, not a synthetic corpus"
        raise TypeError(msg)

    return declared


def test_the_packaged_run_group_lists_all_three_ccg_regimes() -> None:
    assert frozenset(REGIMES) <= frozenset(packaged_options("run"))


@pytest.mark.parametrize(
    ("name", "corruption", "noise"),
    [
        ("synthetic-positive-only", None, None),
        ("synthetic-certified-negatives", 0.25, None),
        ("synthetic-labeled-strings", 0.25, (0.1, 0.1)),
    ],
)
def test_each_ccg_regime_composes_with_its_target_labels_and_metric(
    name: str,
    corruption: float | None,
    noise: tuple[float, float] | None,
) -> None:
    run = _regime(name)
    corpus = _corpus(run)
    trainer = run.trainer
    loss = run.loss
    goal = cast("RootGoalSpec", corpus.grammar.goal)

    assert goal.categories == ("S",)
    assert isinstance(trainer, ParserTrainerSpec)
    assert trainer.early_stopping.metric == "loss/grammaticality"
    assert loss is not None
    assert {term.key: term.weight for term in loss.terms} == {
        "grammaticality": 1.0,
        "coherence": 0.1,
    }
    assert run.data.datasets["train"].featurizer is not None
    assert run.data.datasets["train"].featurizer.model_dump()["kind"] == "unlabeled"

    if corruption is None:
        assert corpus.corruption is None
    else:
        assert corpus.corruption is not None
        assert corpus.corruption.proportion.value == corruption

    if noise is None:
        assert corpus.noise is None
    else:
        assert corpus.noise is not None
        assert (
            corpus.noise.grammatical.rate,
            corpus.noise.ungrammatical.rate,
        ) == noise


def test_the_target_set_and_negative_proportion_are_composable_settings() -> None:
    changed = RunSpecCodec(search_path=config_search_path(ROOT))(
        RUNS / "synthetic-certified-negatives.yaml",
        overrides=("data/datasets/train/dataset/corruption/proportion=half",),
        extra={
            "data.datasets.train.dataset.grammar.goal.categories": ["NP"],
        },
    )
    corpus = _corpus(changed)
    goal = cast("RootGoalSpec", corpus.grammar.goal)

    assert goal.categories == ("NP",)
    assert corpus.corruption is not None
    assert corpus.corruption.proportion.value == 0.5


@pytest.mark.parametrize("name", REGIMES)
def test_each_ccg_regime_takes_one_finite_gradient_step(name: str) -> None:
    run = _regime(name, "trainer.accumulation_steps=1")
    metrics = train_spec(run)

    assert metrics["optimizer/loss_finite"] == 1.0
    assert metrics["optimizer/stepped"] == 1.0
    assert math.isfinite(metrics["loss/grammaticality"])
    assert cast("ParserTrainerSpec", run.trainer).early_stopping.metric in metrics
