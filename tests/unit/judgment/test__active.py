"""Tests for quantitative active judgment collection."""

from typing import TYPE_CHECKING, NamedTuple, cast
from uuid import UUID

import pytest
from bead.items.item import Item

from lofi.judgment._acquisition import AcquisitionScore, RC21Acquisition
from lofi.judgment._active import (
    COLLECTOR_FAMILY,
    ActiveJudgment,
    AwaitedBatch,
    CollectorContext,
    active_collector_for,
    collect_initial,
    draw_initial_batch,
    run_active_learning,
)
from lofi.judgment._attributes import AttributeExpression
from lofi.registry import UnknownTagError
from lofi.spec.acquisition import (
    ActiveLearningSettings,
    AgreementStopSettings,
    AsymptoteStopSettings,
    BeadCollectorSettings,
    CollectedInitialSettings,
    CrossValidationSettings,
    PointReadoutSettings,
    RC21AcquisitionSettings,
    StoppingSettings,
    TokenizerSettings,
    TrainedPredictorsSettings,
    UniquenessSettings,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from lofi.judgment import AcquisitionBatch, ActiveLearningResult

KEY = {letter: str(UUID(int=index + 1)) for index, letter in enumerate("abcd")}


class _Unknown(NamedTuple):
    kind: str


def _settings(
    batch: int, stopping: StoppingSettings, *, max_rounds: int = 4
) -> ActiveLearningSettings:
    return ActiveLearningSettings(
        RC21AcquisitionSettings(
            "rc2.1",
            batch,
            "item['group']",
            (UniquenessSettings("uniqueness", "item['lemma']"),),
            (),
            0.0,
        ),
        stopping,
        max_rounds,
        0.95,
        20,
        "items.json",
        {"group": "item['group']"},
        CollectedInitialSettings("collected", "item['group']", 1),
        "fit.json",
        TrainedPredictorsSettings(
            "run",
            {"words": "sentence"},
            TokenizerSettings("whitespace", "en", None),
            {"score": PointReadoutSettings("spans", "log-mass")},
        ),
        BeadCollectorSettings("bead"),
        CrossValidationSettings("item", "kfold", 2, 0.0, None),
        "encoding.json",
    )


class _Fit:
    @property
    def performance(self) -> Mapping[str, float]:
        return {"transitive": 1.0}

    @property
    def validation_ranks(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return (0.0, 1.0, 2.0, 3.0), (0.0, 1.0, 2.0, 3.0)

    def acquisition_scores(
        self, items: Sequence[Item], /
    ) -> tuple[AcquisitionScore, ...]:
        values = dict(zip(KEY.values(), (0.0, 0.4, 0.6, 1.0), strict=True))
        return tuple(
            AcquisitionScore(str(item.id), values[str(item.id)]) for item in items
        )


class _Fitter:
    def __init__(self) -> None:
        self.fits: list[tuple[ActiveJudgment, ...]] = []

    def fit(self, judgments: Sequence[ActiveJudgment], /) -> _Fit:
        self.fits.append(tuple(judgments))
        return _Fit()


class _Collector:
    def __init__(self, *, answered: int = 4) -> None:
        self._answered = answered
        self.batches: list[tuple[str, ...]] = []

    @property
    def sources(self) -> tuple[Path, ...]:
        return ()

    def collect(
        self, round_index: int, batch: AcquisitionBatch, /
    ) -> tuple[ActiveJudgment, ...] | None:
        self.batches.append(batch.item_ids)
        if round_index > self._answered:
            return None
        return tuple(
            ActiveJudgment(f"{participant}-{round_index}", item_id, index)
            for index, item_id in enumerate(batch.item_ids)
            for participant in ("annotator-a", "annotator-b")
        )


def _item(identity: str, lemma: str, group: str) -> Item:
    return Item(
        id=UUID(identity),
        item_template_id=UUID(int=0),
        item_metadata={"lemma": lemma, "group": group},
    )


def _candidates() -> tuple[Item, ...]:
    return tuple(
        _item(KEY[letter], lemma, "transitive")
        for letter, lemma in zip(
            "abcd", ("admire", "call", "help", "visit"), strict=True
        )
    )


def test_each_round_fits_selects_one_batch_refits_and_stops_on_agreement() -> None:
    settings = _settings(2, AgreementStopSettings(0.9, "estimate"))
    fitter = _Fitter()

    result = run_active_learning(
        settings,
        _candidates(),
        (),
        fitter,
        _Collector(),
        RC21Acquisition(settings.acquisition),
        bootstrap_seed=11,
    )

    assert len(result.rounds) == 1
    assert len(fitter.fits) == 2
    assert result.rounds[0].batch.item_ids == (KEY["a"], KEY["d"])
    assert result.rounds[0].agreement == 1.0
    assert result.rounds[0].agreement_interval == (1.0, 1.0)
    assert result.rounds[0].rank_correlation == 1.0
    assert result.rounds[0].rank_interval == (1.0, 1.0)
    assert result.rounds[0].stop_reason == "agreement"
    assert result.stop_reason == "agreement"
    assert result.awaiting is None


def test_declared_asymptote_stops_after_the_quantitative_window() -> None:
    settings = _settings(1, AsymptoteStopSettings(2, 0.0))
    fitter = _Fitter()

    result = run_active_learning(
        settings,
        _candidates(),
        (),
        fitter,
        _Collector(),
        RC21Acquisition(settings.acquisition),
        bootstrap_seed=11,
    )

    assert len(result.rounds) == 2
    assert len(fitter.fits) == 3
    assert all(len(round_.batch.item_ids) == 1 for round_ in result.rounds)
    assert result.rounds[-1].stop_reason == "asymptote"


def test_an_unanswered_batch_stops_the_run_awaiting_that_round() -> None:
    settings = _settings(1, AsymptoteStopSettings(3, 0.0))
    collector = _Collector(answered=1)

    result = run_active_learning(
        settings,
        _candidates(),
        (),
        _Fitter(),
        collector,
        RC21Acquisition(settings.acquisition),
        bootstrap_seed=11,
    )

    assert len(result.rounds) == 1
    assert result.awaiting is not None
    assert result.awaiting.index == 2
    assert result.awaiting.batch.item_ids == collector.batches[-1]
    assert not result.stopped
    assert result.stop_reason is None
    assert len(result.judgments) == 2


def test_initially_judged_candidates_are_not_offered_again() -> None:
    settings = _settings(1, AsymptoteStopSettings(2, 0.0))
    collector = _Collector()
    initial = (ActiveJudgment("annotator-z", KEY["a"], 1),)

    result = run_active_learning(
        settings,
        _candidates(),
        initial,
        _Fitter(),
        collector,
        RC21Acquisition(settings.acquisition),
        bootstrap_seed=11,
    )

    assert all(KEY["a"] not in batch for batch in collector.batches)
    assert result.judgments[0] == initial[0]


def test_an_initial_judgment_outside_the_pool_is_refused() -> None:
    settings = _settings(1, AsymptoteStopSettings(2, 0.0))

    with pytest.raises(ValueError, match="outside the pool"):
        run_active_learning(
            settings,
            _candidates(),
            (ActiveJudgment("annotator-z", "z", 1),),
            _Fitter(),
            _Collector(),
            RC21Acquisition(settings.acquisition),
            bootstrap_seed=11,
        )


def _run(
    settings: ActiveLearningSettings,
    collector: _Collector,
    /,
    initial: Sequence[ActiveJudgment] = (),
) -> ActiveLearningResult:
    return run_active_learning(
        settings,
        _candidates(),
        initial,
        _Fitter(),
        collector,
        RC21Acquisition(settings.acquisition),
        bootstrap_seed=11,
    )


def test_a_run_that_judges_the_whole_pool_completes_as_pool_exhausted() -> None:
    collector = _Collector()

    result = _run(_settings(2, AsymptoteStopSettings(5, 0.0)), collector)

    assert collector.batches == [(KEY["a"], KEY["d"]), (KEY["b"], KEY["c"])]
    assert [round_.stop_reason for round_ in result.rounds] == [
        None,
        "pool-exhausted",
    ]
    assert result.rounds[-1].stopped
    assert result.stop_reason == "pool-exhausted"
    assert result.awaiting is None


def test_a_remainder_too_small_for_a_batch_exhausts_the_pool() -> None:
    collector = _Collector()

    result = _run(_settings(3, AsymptoteStopSettings(5, 0.0)), collector)

    assert collector.batches == [(KEY["a"], KEY["d"], KEY["b"])]
    assert result.rounds[-1].stop_reason == "pool-exhausted"
    assert result.stop_reason == "pool-exhausted"


def test_a_pool_the_initial_judgments_exhaust_records_no_round() -> None:
    collector = _Collector()
    initial = tuple(ActiveJudgment("annotator-z", KEY[item], 1) for item in "abc")

    result = _run(_settings(2, AsymptoteStopSettings(5, 0.0)), collector, initial)

    assert collector.batches == []
    assert result.rounds == ()
    assert result.stop_reason == "pool-exhausted"
    assert result.judgments == initial


def test_a_run_reaching_its_round_bound_records_it() -> None:
    result = _run(
        _settings(1, AsymptoteStopSettings(5, 0.0), max_rounds=2), _Collector()
    )

    assert [round_.stop_reason for round_ in result.rounds] == [None, "max-rounds"]
    assert result.stop_reason == "max-rounds"


def test_a_stopping_rule_holding_as_the_pool_empties_is_recorded() -> None:
    result = _run(_settings(4, AgreementStopSettings(0.9, "estimate")), _Collector())

    assert result.rounds[-1].stop_reason == "agreement"
    assert result.stop_reason == "agreement"


GROUPS = ("transitive", "intransitive")


def _grouped() -> tuple[Item, ...]:
    return tuple(
        _item(str(UUID(int=100 + 10 * position + index)), f"{group}-{index}", group)
        for position, group in enumerate(GROUPS)
        for index in range(5)
    )


STRATUM = AttributeExpression("item['group']", role="a stratum")


def test_the_initial_draw_takes_the_declared_items_from_every_stratum() -> None:
    batch = draw_initial_batch(_grouped(), STRATUM, 2, seed=4)
    again = draw_initial_batch(_grouped(), STRATUM, 2, seed=4)
    drawn = {
        group: [item for item in batch.items if item.item_metadata["group"] == group]
        for group in GROUPS
    }

    assert batch.item_ids == again.item_ids
    assert {group: len(items) for group, items in drawn.items()} == {
        "transitive": 2,
        "intransitive": 2,
    }
    assert batch.shares == {"intransitive": 0.5, "transitive": 0.5}
    assert batch.variance is None
    assert batch.performance == {}
    assert {
        draw_initial_batch(_grouped(), STRATUM, 2, seed=seed).item_ids
        for seed in range(6)
    } != {batch.item_ids}


def test_an_initial_draw_larger_than_a_stratum_is_refused() -> None:
    with pytest.raises(ValueError, match="'intransitive' holds 5"):
        draw_initial_batch(_grouped(), STRATUM, 6, seed=0)


def test_the_collector_answers_the_initial_draw_as_round_zero() -> None:
    collector = _Collector()

    collected = collect_initial(
        CollectedInitialSettings("collected", "item['group']", 1),
        _grouped(),
        collector,
        seed=4,
    )

    assert not isinstance(collected, AwaitedBatch)
    assert len(collector.batches) == 1
    assert {judgment.item_id for judgment in collected} == set(collector.batches[0])
    assert {judgment.participant_id for judgment in collected} == {
        "annotator-a-0",
        "annotator-b-0",
    }


def test_an_unanswered_initial_draw_awaits_round_zero() -> None:
    collected = collect_initial(
        CollectedInitialSettings("collected", "item['group']", 1),
        _grouped(),
        _Collector(answered=-1),
        seed=4,
    )

    assert isinstance(collected, AwaitedBatch)
    assert collected.index == 0
    assert len(collected.batch.item_ids) == 2


class _Stray:
    @property
    def sources(self) -> tuple[Path, ...]:
        return ()

    def collect(
        self, round_index: int, batch: AcquisitionBatch, /
    ) -> tuple[ActiveJudgment, ...]:
        del round_index, batch
        return (ActiveJudgment("annotator-a", "elsewhere", 0),)


def test_an_initial_answer_outside_the_draw_is_refused() -> None:
    with pytest.raises(ValueError, match="unselected items"):
        collect_initial(
            CollectedInitialSettings("collected", "item['group']", 1),
            _grouped(),
            _Stray(),
            seed=4,
        )


def test_a_collector_is_built_through_its_installed_family_entry(
    tmp_path: Path,
) -> None:
    context = CollectorContext(
        tmp_path / "out",
        tmp_path,
        _candidates(),
        {KEY["a"]: {"score": -1.0}},
        levels=2,
        seed=3,
    )

    assert COLLECTOR_FAMILY == "active_collector"
    assert context.directory == tmp_path / "out"
    assert context.root == tmp_path
    assert tuple(str(item.id) for item in context.candidates) == tuple(KEY.values())
    assert context.readouts == {KEY["a"]: {"score": -1.0}}
    assert context.levels == 2
    assert context.seed == 3
    assert context.encoding is None

    with pytest.raises(UnknownTagError):
        active_collector_for(
            cast("BeadCollectorSettings", _Unknown("telepathy")), context
        )
