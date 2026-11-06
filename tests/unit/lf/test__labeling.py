from typing import cast

import pytest

from lofi.lf import (
    CLASSICAL,
    GRADED,
    RTE,
    THREE_WAY,
    InferenceCell,
    InferenceLabel,
    InferenceLabeling,
    registered_labeling,
)


def test_three_way_labeling_preserves_the_semantic_relations() -> None:
    assert THREE_WAY.labels == (
        InferenceLabel.ENTAILS,
        InferenceLabel.CONTRADICTS,
        InferenceLabel.NEUTRAL,
    )
    assert THREE_WAY.encode(InferenceLabel.NEUTRAL) is InferenceLabel.NEUTRAL
    assert THREE_WAY.label_map(CLASSICAL) == CLASSICAL


def test_rte_labeling_coarsens_contradiction_and_neutral() -> None:
    assert RTE.labels == (
        InferenceLabel.ENTAILS,
        InferenceLabel.NON_ENTAILS,
    )
    assert RTE.encode(InferenceLabel.CONTRADICTS) is InferenceLabel.NON_ENTAILS
    assert RTE.encode(InferenceLabel.NEUTRAL) is InferenceLabel.NON_ENTAILS
    assert RTE.label_map(CLASSICAL).cells[InferenceLabel.NON_ENTAILS] == frozenset(
        {InferenceCell.CONTRADICTS, InferenceCell.NEUTRAL}
    )


def test_graded_labeling_publishes_entailment_probability_as_its_scale() -> None:
    assert GRADED.graded
    assert GRADED.labels == RTE.labels


def test_a_labeling_refuses_text_in_place_of_the_shared_label_type() -> None:
    with pytest.raises(TypeError, match="outputs are InferenceLabel"):
        InferenceLabeling(
            "invalid",
            cast("tuple[InferenceLabel, ...]", ("entails", "neutral")),
            {
                relation: relation
                for relation in (
                    InferenceLabel.ENTAILS,
                    InferenceLabel.CONTRADICTS,
                    InferenceLabel.NEUTRAL,
                )
            },
            exclusive=False,
            graded=False,
        )

    with pytest.raises(TypeError, match="inference relation is an InferenceLabel"):
        THREE_WAY.encode(cast("InferenceLabel", "neutral"))


def test_a_registered_labeling_is_found_by_its_entry_name() -> None:
    assert registered_labeling("three-way") is THREE_WAY
    assert registered_labeling("rte") is RTE
