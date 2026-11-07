"""Tests for declared prediction rescoring and checkpoint deltas."""

import json
from hashlib import sha256
from typing import TYPE_CHECKING, cast

import pytest
from tests.support.installed import installed

import lofi.eval._rescoring as rescoring_module
from lofi.eval._ports import Budget
from lofi.eval._rescoring import (
    RECALL_HOMOGENEITY_METRIC,
    CheckpointScore,
    DeclaredBudget,
    PredictionError,
    RecallHomogeneityMetric,
    RescoringDeltaTable,
    declared_conventions,
    read_declared_predictions,
    read_predictions,
    rescore_declared,
    rescore_frozen_checkpoint,
    rescore_predictions,
    rescoring_metric,
)
from lofi.eval._residual import Approximation
from lofi.eval._values import ConventionVector
from lofi.registry import Registry
from lofi.spec.evaluators.alignment import AlignmentSettings
from lofi.spec.reducers import ConventionGridSettings

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.formalism import Grammar
    from lofi.spec import RunSpec


def _released(**changed: JsonValue) -> str:
    items: list[JsonValue] = [
        {
            "item_id": f"{split}-1",
            "split": split,
            "length": 4,
            "punctuation": [],
            "gold": [
                {"start": 0, "end": 4, "label": "S", "depth": 0},
                {"start": 2, "end": 4, "label": "VP", "depth": 1},
            ],
            "spans": [
                {"start": 0, "end": 4, "label": "c0"},
                {"start": 2, "end": 4, "label": "c1"},
            ],
        }
        for split in ("dev", "test")
    ]
    body: dict[str, JsonValue] = {
        "system": "released-parser",
        "budget": {"depth": 4, "beam": 8, "mass": 0.02},
        "items": items,
    }
    return json.dumps(body | changed)


def _grid() -> ConventionGridSettings:
    return ConventionGridSettings(
        punctuation=("keep",),
        unary_chains=("keep",),
        trivial_spans=("keep",),
        minimum_length=(0,),
        aggregation=("corpus",),
        coverage=("ignore",),
        self_arcs=("allow",),
        gold_derivation=("canonical",),
    )


def _alignment() -> AlignmentSettings:
    return AlignmentSettings(
        kind="many-to-one",
        tie_break="least-gold-label",
        fit_on="dev",
        score_on="test",
        floor_replicates=19,
        floor_level=0.9,
        stability_replicates=19,
        stability_level=0.9,
    )


class _Config:
    def model_dump(self) -> dict[str, JsonValue]:
        return {"run": "rescoring-test"}


class _Grammar:
    fingerprint = "grammar:rescoring-test"


def test_a_released_set_is_read_with_the_digest_and_budget_of_its_text() -> None:
    text = _released()
    predictions = read_predictions(text)

    assert predictions.system == "released-parser"
    assert predictions.digest == sha256(text.encode("utf-8")).hexdigest()
    assert predictions.budget == DeclaredBudget(4, 8, mass=0.02)
    assert isinstance(predictions.budget, Budget)
    assert Approximation(predictions.budget, residual=None).budget is predictions.budget
    assert len(predictions.on("dev")) == 1
    assert len(predictions.on("test")) == 1
    assert predictions == read_predictions(text)
    assert hash(predictions) == hash(read_predictions(text))
    assert "ReleasedPredictions(" in repr(predictions)


def test_a_released_set_refuses_malformed_or_repeated_items() -> None:
    with pytest.raises(PredictionError, match="not JSON"):
        read_predictions("{")

    with pytest.raises(PredictionError, match="does not validate"):
        read_predictions("[]")

    with pytest.raises(PredictionError, match="system is empty"):
        read_predictions(_released(system=""))

    original = cast("dict[str, JsonValue]", json.loads(_released()))
    items = cast("list[JsonValue]", original["items"])

    with pytest.raises(ValueError, match="released twice"):
        read_predictions(_released(items=[items[0], items[0]]))


def test_declared_rescoring_has_no_static_convention_crossing() -> None:
    assert not hasattr(rescoring_module, "convention_crossing")
    vectors = declared_conventions(_grid(), (), registry=Registry("convention"))
    cells = rescore_predictions(
        read_predictions(_released()),
        alignments=(_alignment(),),
        metric=RECALL_HOMOGENEITY_METRIC,
        conventions=vectors,
    )
    rows = rescore_declared(
        read_predictions(_released()),
        run_spec=cast("RunSpec", _Config()),
        grammar=cast("Grammar", _Grammar()),
        freeze_id="freeze/1",
        alignments=(_alignment(),),
        metric=RECALL_HOMOGENEITY_METRIC,
        conventions=vectors,
    )

    assert len(cells) == 1
    assert cells[0].family == "many-to-one"
    assert cells[0].metric == "recall-homogeneity"
    assert cells[0].result.value == 1.0
    assert cells[0].to_spec()["metric"] == "recall-homogeneity"
    assert len(rows) == 1
    assert rows[0].row.metric == "recall-homogeneity"


def test_prediction_metric_and_user_convention_choices_use_registries() -> None:
    contributed = ConventionVector(
        punctuation="strip",
        unary_chains="collapse",
        trivial_spans="drop",
        minimum_length=3,
        aggregation="sentence",
        coverage="zero",
        self_arcs="forbid",
        gold_derivation="oracle",
    ).extended({"laboratory": "registered"})

    readers = installed("prediction_format", {"lofi-json": read_predictions})
    assert read_declared_predictions(
        _released(), "lofi-json", registry=readers
    ) == read_predictions(_released())
    metrics = installed(
        "rescoring_metric", {"recall-homogeneity": RecallHomogeneityMetric()}
    )
    assert (
        rescoring_metric("recall-homogeneity", registry=metrics).name
        == "recall-homogeneity"
    )
    providers = installed("convention", {"laboratory": lambda: (contributed,)})
    assert declared_conventions(_grid(), ("laboratory",), registry=providers) == (
        declared_conventions(_grid(), (), registry=providers)[0],
        contributed,
    )


class _CheckpointScorer:
    def __init__(self, scores: tuple[CheckpointScore, ...]) -> None:
        self.scores = scores
        self.seen: Path | None = None

    def __call__(self, checkpoint: Path, /) -> tuple[CheckpointScore, ...]:
        self.seen = checkpoint
        return self.scores


def test_a_frozen_checkpoint_is_rescored_at_a_new_head_as_a_delta_table(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "freeze-7.pt"
    scorer = _CheckpointScorer(
        (
            CheckpointScore("bracketing-f1", 0.71),
            CheckpointScore("lexical-accuracy", 0.81),
        )
    )
    table = rescore_frozen_checkpoint(
        checkpoint,
        (
            CheckpointScore("lexical-accuracy", 0.80),
            CheckpointScore("bracketing-f1", 0.70),
        ),
        scorer=scorer,
        tolerances={"lexical-accuracy": 0.005, "bracketing-f1": 0.02},
        frozen_revision="0" * 40,
        head_revision="1" * 40,
    )

    assert isinstance(table, RescoringDeltaTable)
    assert scorer.seen == checkpoint
    assert table.checkpoint == str(checkpoint)
    assert tuple(row.metric for row in table.rows) == (
        "bracketing-f1",
        "lexical-accuracy",
    )
    assert table.rows[0].delta == pytest.approx(0.01)
    assert table.rows[0].within_tolerance is True
    assert table.rows[1].within_tolerance is False
    assert table.changed == 1
    assert json.loads(json.dumps(table.to_spec())) == table.to_spec()


def test_checkpoint_rescoring_requires_the_committed_metric_set_and_tolerances(
    tmp_path: Path,
) -> None:
    scorer = _CheckpointScorer((CheckpointScore("new", 0.5),))

    with pytest.raises(ValueError, match="metric sets differ"):
        rescore_frozen_checkpoint(
            tmp_path / "freeze.pt",
            (CheckpointScore("old", 0.5),),
            scorer=scorer,
            tolerances={"old": 0.01},
            frozen_revision="0" * 40,
            head_revision="1" * 40,
        )

    with pytest.raises(ValueError, match="tolerance metrics differ"):
        rescore_frozen_checkpoint(
            tmp_path / "freeze.pt",
            (CheckpointScore("new", 0.5),),
            scorer=scorer,
            tolerances={"other": 0.01},
            frozen_revision="0" * 40,
            head_revision="1" * 40,
        )
