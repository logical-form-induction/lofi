"""Tests for trained-point predicted judgment sources."""

import json
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import pytest
from bead.items.item import Item

from lofi.judgment import design_of
from lofi.judgment._predicted import (
    PREDICTED_JUDGMENT_SOURCE_PLUGIN,
    PredictedJudgmentSource,
)
from lofi.lf import InferenceLabel
from lofi.spec.judgment_sources import PredictedJudgmentSourceSettings

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.judgment import InferenceText

ITEMS = Path(__file__).parents[2] / "data" / "judgment" / "items"


class _Point:
    def __init__(self) -> None:
        self.read: list[InferenceText] = []

    @property
    def response_levels(self) -> int:
        return 3

    @property
    def relation_labels(self) -> tuple[InferenceLabel, ...]:
        return (
            InferenceLabel.ENTAILS,
            InferenceLabel.CONTRADICTS,
            InferenceLabel.NEUTRAL,
        )

    def predict_responses(
        self, pairs: Sequence[InferenceText], /
    ) -> tuple[tuple[float, ...], ...]:
        self.read.extend(pairs)
        return tuple(
            ((0.7, 0.2, 0.1), (0.1, 0.8, 0.1))[index % 2] for index in range(len(pairs))
        )

    def predict_relations(
        self, pairs: Sequence[InferenceText], /
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            ((0.6, 0.3, 0.1), (0.2, 0.7, 0.1))[index % 2] for index in range(len(pairs))
        )


def _settings(
    *,
    inputs: Mapping[str, str] | None = None,
    factors: Mapping[str, str] | None = None,
    item_set: str = "items.json",
) -> PredictedJudgmentSourceSettings:
    return PredictedJudgmentSourceSettings(
        "runs/synthetic",
        "point-17",
        item_set,
        {"premise": "context", "hypothesis": "claim"} if inputs is None else inputs,
        {"genre": "item['genre']"} if factors is None else factors,
        "inference",
        ("entails", "contradicts", "neutral"),
        "lofi-point-17",
    )


def _item(number: int, context: str, claim: str, genre: str) -> Item:
    return Item(
        id=UUID(int=number),
        item_template_id=UUID(int=0),
        rendered_elements={"context": context, "claim": claim},
        item_metadata={"genre": genre},
    )


ITEMS_A_B = (
    _item(1, "someone admires", "someone likes", "fiction"),
    _item(2, "someone calls", "someone sleeps", "news"),
)


def test_predicted_source_reads_items_through_the_declared_roles() -> None:
    point = _Point()
    source = PredictedJudgmentSource(_settings(), ITEMS_A_B, point)
    provenance = source.provenance("predicted-inference")

    assert [(text.premise, text.hypothesis) for text in point.read] == [
        ("someone admires", "someone likes"),
        ("someone calls", "someone sleeps"),
    ]
    assert tuple(row.item for row in source.rows()) == (
        str(UUID(int=1)),
        str(UUID(int=2)),
    )
    assert tuple(row.response for row in source.rows()) == (0, 1)
    assert source.factors == ("genre",)
    assert design_of(source).factor_levels["genre"] == ("fiction", "news")
    assert source.predictions[0].response_probabilities == pytest.approx(
        (0.7, 0.2, 0.1)
    )
    assert source.predictions[0].relation_probabilities == pytest.approx(
        {
            InferenceLabel.ENTAILS: 0.6,
            InferenceLabel.CONTRADICTS: 0.3,
            InferenceLabel.NEUTRAL: 0.1,
        }
    )
    assert provenance.producer_run == "runs/synthetic"
    assert provenance.producer_point == "point-17"


def test_predicted_source_refuses_unmapped_roles_and_uncarried_factors() -> None:
    with pytest.raises(ValueError, match="reads the input roles"):
        PredictedJudgmentSource(
            _settings(inputs={"words": "context"}), ITEMS_A_B, _Point()
        )
    with pytest.raises(ValueError, match="factor 'voice'"):
        PredictedJudgmentSource(
            _settings(factors={"voice": "item['voice']"}), ITEMS_A_B, _Point()
        )


def test_the_plugin_constructs_the_declared_item_set(tmp_path: Path) -> None:
    for source in ITEMS.iterdir():
        (tmp_path / source.name).write_bytes(source.read_bytes())
    document = json.loads((ITEMS / "items.json").read_text(encoding="utf-8"))
    document["generators"] = []
    document["lexicons"].append(
        {
            "kind": "inline",
            "lexicon": {
                "name": "verbs",
                "items": [
                    {
                        "id": "0b7e2d10-2f4c-4c55-8f0e-5d1f8a3c2b09",
                        "lemma": "admired",
                        "language_code": "eng",
                        "features": {"pos": "VERB"},
                    }
                ],
            },
        }
    )
    (tmp_path / "items.json").write_text(json.dumps(document), encoding="utf-8")
    settings = _settings(
        inputs={"premise": "sentence", "hypothesis": "sentence"},
        factors={"structure": "item['structure']"},
    )

    source = PREDICTED_JUDGMENT_SOURCE_PLUGIN.construct(settings, tmp_path, _Point())

    assert design_of(source).factor_levels["structure"] == ("passive", "transitive")
    with pytest.raises(ValueError, match="no file"):
        PREDICTED_JUDGMENT_SOURCE_PLUGIN.construct(
            _settings(item_set="absent.json"), tmp_path, _Point()
        )
