"""Tests for the MAP refit an active-learning run scores candidates by."""

from typing import Final
from uuid import NAMESPACE_OID, UUID, uuid5

import pytest
from bead.items.item import Item

from lofi.judgment import (
    LINEAR,
    ActiveJudgment,
    ConditionalLikelihood,
    CoverageGate,
    Cumulative,
    FixedDiscrimination,
    LinkingSpec,
    LinkTerm,
    MapInference,
    PriorBlock,
    ResponseCompiler,
    ResponseSpec,
    ThresholdPriors,
)
from lofi.judgment.splits import kfold
from lofi.run._active_fit import ActiveItemColumns, MapActiveFitter
from lofi.spec import HeldOutParticipantPolicy, load_plugins
from lofi.spec.optimizers.adam import AdamOptimizerSpec

LEVELS: Final = 4
SCORE: Final = "p_gram"


GROUPS: Final = ("transitive", "intransitive")


def _key(name: str, /) -> str:
    """Return the identity of the fixture item with this name."""
    return str(uuid5(NAMESPACE_OID, name))


def _candidate(name: str, /) -> Item:
    group, index = name.split("-")
    return Item(
        id=UUID(_key(name)),
        item_template_id=UUID(int=0),
        item_metadata={"group": group, "lemma": f"verb-{index}"},
    )


def _items() -> dict[str, ActiveItemColumns]:
    """Return two strata of four items whose score rises with their index."""
    return {
        _key(f"{group}-{index}"): ActiveItemColumns(
            group,
            {"group": group, "lemma": f"verb-{index}"},
            {SCORE: -1.5 + index},
        )
        for group in GROUPS
        for index in range(4)
    }


def _fitter(
    policy: HeldOutParticipantPolicy | None = HeldOutParticipantPolicy.PLUG_IN,
    threshold: float = 0.0,
    factors: tuple[str, ...] = ("group", "lemma"),
) -> MapActiveFitter:
    load_plugins()
    response = ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(LinkTerm(readout=SCORE, linking=LinkingSpec(family=LINEAR)),),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )
    return MapActiveFitter(
        ResponseCompiler.installed().compile(response),
        MapInference(
            optimizer=AdamOptimizerSpec(learning_rate=0.1),
            steps=200,
            initial_scale=0.1,
        ),
        levels=LEVELS,
        factors=factors,
        columns=(SCORE,),
        items=_items(),
        splitter=kfold("item", folds=2, seed=5),
        gate=CoverageGate(threshold, policy=policy),
        unseen=policy,
        seed=13,
    )


def _judgments(*items: str) -> tuple[ActiveJudgment, ...]:
    """Return two annotators' responses, each the index the item ends in."""
    return tuple(
        ActiveJudgment(annotator, _key(item), int(item[-1]))
        for item in items
        for annotator in ("annotator-a", "annotator-b")
    )


JUDGED: Final = (
    "transitive-0",
    "transitive-1",
    "transitive-3",
    "intransitive-0",
    "intransitive-2",
    "intransitive-3",
)


def test_a_refit_scores_unjudged_candidates_in_the_order_of_their_predictor() -> None:
    fit = _fitter().fit(_judgments(*JUDGED))
    candidates = (_candidate("intransitive-1"), _candidate("transitive-2"))

    scores = fit.acquisition_scores(candidates)

    assert tuple(score.item_id for score in scores) == (
        _key("intransitive-1"),
        _key("transitive-2"),
    )
    assert 0.0 <= scores[0].expected < scores[1].expected <= LEVELS - 1


def test_cross_validated_ranks_and_stratum_performance_cover_every_judged_item() -> (
    None
):
    fit = _fitter().fit(_judgments(*JUDGED))
    observed, predicted = fit.validation_ranks

    assert sorted(observed) == sorted((0.0, 1.0, 3.0, 0.0, 2.0, 3.0))
    assert len(predicted) == len(observed)
    assert set(fit.performance) == set(GROUPS)
    assert all(value == pytest.approx(1.0) for value in fit.performance.values())


def test_a_refit_is_reproducible_from_its_seed() -> None:
    first = _fitter().fit(_judgments(*JUDGED)).validation_ranks
    second = _fitter().fit(_judgments(*JUDGED)).validation_ranks

    assert first == second


def test_a_judgment_outside_the_pool_is_refused() -> None:
    with pytest.raises(ValueError, match="not in the candidate pool"):
        _fitter().fit(_judgments(*JUDGED, "ditransitive-9"))


def test_a_stratum_with_too_few_judged_items_to_rank_is_refused() -> None:
    with pytest.raises(ValueError, match="stratum 'intransitive' cannot be ranked"):
        _fitter().fit(_judgments(*JUDGED[:3], "intransitive-0", "intransitive-2"))


def test_pool_items_must_record_the_declared_factors_and_columns() -> None:
    with pytest.raises(ValueError, match="records the factors"):
        _fitter(factors=("group",))


def test_unseen_participants_the_gate_admits_without_a_rule_are_refused() -> None:
    judgments = (
        *_judgments(*JUDGED),
        ActiveJudgment("annotator-c", _key("transitive-0"), 0),
    )

    with pytest.raises(ValueError, match="no unseen-participant rule"):
        _fitter(None, 0.5).fit(judgments)
