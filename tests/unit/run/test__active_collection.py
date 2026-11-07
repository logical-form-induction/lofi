"""Tests for the bead and simulated collectors of an active-learning run."""

import json
from typing import TYPE_CHECKING

import pytest
from tests.unit.run.test__active_inputs import (
    annotation_lines,
    encoding_document,
    fixture_generator,
    item_document,
)

from lofi.judgment import AcquisitionBatch, CollectorContext, active_collector_for
from lofi.judgment.synthetic_designs import (
    KNOWN_RANDOM_EFFECTS,
    KNOWN_RANDOM_EFFECTS_TAG,
    KnownEffectsDesign,
)
from lofi.registry import UnknownTagError
from lofi.run._active_collection import BeadCollector, SimulatedCollector
from lofi.run._active_inputs import read_encoding, read_pool
from lofi.run._boundary.active import BATCH_NAME, RESPONSES_NAME, round_directory
from lofi.spec import load_plugins
from lofi.spec.acquisition import (
    BeadCollectorSettings,
    DrawnEffectsCollectorSettings,
    LatentRangeSettings,
    PredictorTruthCollectorSettings,
    RegisteredDesignCollectorSettings,
)

if TYPE_CHECKING:
    from pathlib import Path

    from bead.items.item import Item
    from bead.protocol import ResponseEncoding

    from lofi.judgment import ActiveJudgment

RANGE = LatentRangeSettings(-3.0, 3.0)
"""The latent range every drawn design of these tests divides."""


def _items(tmp_path: Path, /) -> tuple[Item, ...]:
    document = tmp_path / "items.json"
    document.write_text(json.dumps(item_document()), encoding="utf-8", newline="\n")
    return read_pool(document, generator_for=fixture_generator).items


def _encoding(tmp_path: Path, /) -> ResponseEncoding:
    path = tmp_path / "encoding.json"
    path.write_text(json.dumps(encoding_document()), encoding="utf-8", newline="\n")
    return read_encoding(path)


def _batch(items: tuple[Item, ...], /) -> AcquisitionBatch:
    return AcquisitionBatch(
        items, {"transitive": 1.0}, 0.25, performance={"transitive": 0.5}, floored=()
    )


def _context(
    tmp_path: Path, items: tuple[Item, ...], /, *, levels: int, seed: int
) -> CollectorContext:
    """Return a context whose predictor rises one unit per pool position."""
    return CollectorContext(
        tmp_path / "run",
        tmp_path,
        items,
        {
            str(item.id): {"p_gram": -6.0 + position}
            for position, item in enumerate(items)
        },
        levels=levels,
        seed=seed,
    )


def _triples(
    collected: tuple[ActiveJudgment, ...] | None, /
) -> tuple[tuple[str, str, int], ...]:
    assert collected is not None
    return tuple(
        (judgment.participant_id, judgment.item_id, judgment.response)
        for judgment in collected
    )


def test_a_new_batch_is_written_as_bead_items_and_awaits_its_responses(
    tmp_path: Path,
) -> None:
    items = _items(tmp_path)[:2]
    collector = BeadCollector(tmp_path / "run", _encoding(tmp_path))

    collected = collector.collect(1, _batch(items))
    written = round_directory(tmp_path / "run", 1) / BATCH_NAME
    lines = written.read_text(encoding="utf-8").splitlines()

    assert collected is None
    assert [json.loads(line)["id"] for line in lines] == [
        str(item.id) for item in items
    ]
    assert json.loads(lines[0])["rendered_elements"] == {
        "sentence": "word-0 word-1 word-4"
    }
    assert json.loads(lines[0])["item_metadata"] == dict(items[0].item_metadata)


def test_a_written_batch_awaits_until_its_responses_are_ingested(
    tmp_path: Path,
) -> None:
    items = _items(tmp_path)[:2]
    collector = BeadCollector(tmp_path / "run", _encoding(tmp_path))
    collector.collect(1, _batch(items))

    assert collector.collect(1, _batch(items)) is None

    responses = round_directory(tmp_path / "run", 1) / RESPONSES_NAME
    responses.write_text(
        annotation_lines({str(items[0].id): 2, str(items[1].id): 0}, ("a", "b")),
        encoding="utf-8",
        newline="\n",
    )
    collected = collector.collect(1, _batch(items))

    assert collected is not None
    assert tuple(
        (judgment.participant_id, judgment.item_id, judgment.response)
        for judgment in collected
    ) == (
        ("a", str(items[0].id), 2),
        ("b", str(items[0].id), 2),
        ("a", str(items[1].id), 0),
        ("b", str(items[1].id), 0),
    )


def test_a_batch_selected_again_for_other_items_is_refused(tmp_path: Path) -> None:
    items = _items(tmp_path)
    collector = BeadCollector(tmp_path / "run", _encoding(tmp_path))
    collector.collect(1, _batch(items[:2]))

    with pytest.raises(ValueError, match="other items"):
        collector.collect(1, _batch(items[2:4]))


def test_responses_to_another_question_are_refused(tmp_path: Path) -> None:
    items = _items(tmp_path)[:1]
    collector = BeadCollector(tmp_path / "run", _encoding(tmp_path))
    collector.collect(1, _batch(items))
    responses = round_directory(tmp_path / "run", 1) / RESPONSES_NAME
    responses.write_text(
        json.dumps(
            {
                "annotator_id": "a",
                "item_id": str(items[0].id),
                "question_name": "naturalness",
                "response_label": "yes",
            }
        ),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="answering a question other than"):
        collector.collect(1, _batch(items))


def test_the_bead_collector_is_built_through_its_installed_entry(
    tmp_path: Path,
) -> None:
    load_plugins()
    items = _items(tmp_path)
    collector = active_collector_for(
        BeadCollectorSettings("bead"),
        CollectorContext(
            tmp_path / "run",
            tmp_path,
            items,
            {},
            levels=3,
            seed=0,
            encoding=_encoding(tmp_path),
        ),
    )

    assert isinstance(collector, BeadCollector)
    assert collector.sources == ()
    assert collector.collect(1, _batch(items[:1])) is None


def test_an_initial_draw_is_written_as_round_zero(tmp_path: Path) -> None:
    items = _items(tmp_path)[:2]
    collector = BeadCollector(tmp_path / "run", _encoding(tmp_path))

    assert (
        collector.collect(
            0, AcquisitionBatch(items, {}, None, performance={}, floored=())
        )
        is None
    )
    assert (round_directory(tmp_path / "run", 0) / BATCH_NAME).is_file()


def test_a_bead_collector_without_the_run_encoding_is_refused(
    tmp_path: Path,
) -> None:
    load_plugins()
    items = _items(tmp_path)

    with pytest.raises(ValueError, match="no response encoding"):
        active_collector_for(
            BeadCollectorSettings("bead"),
            CollectorContext(tmp_path / "run", tmp_path, items, {}, levels=3, seed=0),
        )


def test_a_drawn_effects_collector_answers_each_candidate_as_its_design_item(
    tmp_path: Path,
) -> None:
    load_plugins()
    items = _items(tmp_path)
    collector = active_collector_for(
        DrawnEffectsCollectorSettings("drawn-effects", 3, RANGE),
        _context(tmp_path, items, levels=3, seed=11),
    )
    design = KnownEffectsDesign(
        "drawn-effects",
        participants=3,
        items=len(items),
        levels=3,
        seed=11,
        latent_range=(RANGE.lower, RANGE.upper),
    )()
    expected = {
        (
            design.participant_levels[int(participant)],
            str(items[int(design.item_levels[int(item)].removeprefix("item-"))].id),
            int(response),
        )
        for participant, item, response in zip(
            design.participants, design.items, design.responses, strict=True
        )
    }

    collected = _triples(collector.collect(1, _batch(items[2:4])))

    assert isinstance(collector, SimulatedCollector)
    assert collector.sources == ()
    assert set(collected) == {
        triple
        for triple in expected
        if triple[1] in {str(items[2].id), str(items[3].id)}
    }
    assert len(collected) == 2 * 3
    assert _triples(collector.collect(1, _batch(items[2:4]))) == collected
    assert not (tmp_path / "run").exists()


def test_a_drawn_effects_collector_draws_again_under_another_seed(
    tmp_path: Path,
) -> None:
    load_plugins()
    items = _items(tmp_path)
    settings = DrawnEffectsCollectorSettings("drawn-effects", 8, RANGE)
    first = active_collector_for(settings, _context(tmp_path, items, levels=5, seed=1))
    other = active_collector_for(settings, _context(tmp_path, items, levels=5, seed=2))

    assert _triples(first.collect(1, _batch(items))) != _triples(
        other.collect(1, _batch(items))
    )


def test_a_predictor_truth_collector_takes_the_column_as_the_item_scores(
    tmp_path: Path,
) -> None:
    load_plugins()
    items = _items(tmp_path)
    collector = active_collector_for(
        PredictorTruthCollectorSettings("predictor-truth", "p_gram", 4, "raw", RANGE),
        _context(tmp_path, items, levels=3, seed=5),
    )
    design = KnownEffectsDesign(
        "predictor-truth",
        participants=4,
        items=len(items),
        levels=3,
        seed=5,
        latent_range=(RANGE.lower, RANGE.upper),
        scores=tuple(-6.0 + position for position in range(len(items))),
    )()

    collected = _triples(collector.collect(1, _batch(items)))

    assert sorted(collected) == sorted(
        (
            design.participant_levels[int(participant)],
            str(items[int(design.item_levels[int(item)].removeprefix("item-"))].id),
            int(response),
        )
        for participant, item, response in zip(
            design.participants, design.items, design.responses, strict=True
        )
    )


def test_a_standardized_predictor_truth_centers_and_scales_the_column(
    tmp_path: Path,
) -> None:
    load_plugins()
    items = _items(tmp_path)
    count = len(items)
    center = sum(range(count)) / count
    spread = (sum((index - center) ** 2 for index in range(count)) / count) ** 0.5
    collector = active_collector_for(
        PredictorTruthCollectorSettings(
            "predictor-truth", "p_gram", 4, "standardized", RANGE
        ),
        _context(tmp_path, items, levels=3, seed=5),
    )
    design = KnownEffectsDesign(
        "predictor-truth",
        participants=4,
        items=count,
        levels=3,
        seed=5,
        latent_range=(RANGE.lower, RANGE.upper),
        scores=tuple((index - center) / spread for index in range(count)),
    )()

    collected = _triples(collector.collect(1, _batch(items)))

    assert sorted(response for _, _, response in collected) == sorted(
        int(response) for response in design.responses
    )


def test_a_predictor_truth_column_the_point_does_not_fill_is_refused(
    tmp_path: Path,
) -> None:
    load_plugins()
    items = _items(tmp_path)

    with pytest.raises(ValueError, match=r"'p_lex'.*'p_gram'"):
        active_collector_for(
            PredictorTruthCollectorSettings(
                "predictor-truth", "p_lex", 2, "raw", RANGE
            ),
            _context(tmp_path, items, levels=3, seed=0),
        )


def test_a_constant_column_cannot_be_standardized(tmp_path: Path) -> None:
    load_plugins()
    items = _items(tmp_path)
    context = CollectorContext(
        tmp_path / "run",
        tmp_path,
        items,
        {str(item.id): {"p_gram": 1.0} for item in items},
        levels=3,
        seed=0,
    )

    with pytest.raises(ValueError, match="constant"):
        active_collector_for(
            PredictorTruthCollectorSettings(
                "predictor-truth", "p_gram", 2, "standardized", RANGE
            ),
            context,
        )


def test_a_registered_design_answers_each_candidate_as_the_item_it_joins(
    tmp_path: Path,
) -> None:
    load_plugins()
    design = KNOWN_RANDOM_EFFECTS()
    count = len(design.item_levels)
    items = tuple(
        item.with_(
            item_metadata={
                **item.item_metadata,
                "design_item": design.item_levels[(count - 1 - index) % count],
            }
        )
        for index, item in enumerate(_items(tmp_path))
    )
    collector = active_collector_for(
        RegisteredDesignCollectorSettings(
            "registered-design", KNOWN_RANDOM_EFFECTS_TAG, "item['design_item']"
        ),
        _context(tmp_path, items, levels=KNOWN_RANDOM_EFFECTS.levels, seed=0),
    )
    third = design.item_levels.index(str(items[2].item_metadata["design_item"]))
    expected = tuple(
        (design.participant_levels[int(participant)], str(items[2].id), int(response))
        for participant, item, response in zip(
            design.participants, design.items, design.responses, strict=True
        )
        if int(item) == third
    )

    assert _triples(collector.collect(1, _batch(items[2:3]))) == expected
    assert len(expected) == KNOWN_RANDOM_EFFECTS.participants


@pytest.mark.parametrize(
    ("levels", "tag", "error", "reason"),
    [
        (3, KNOWN_RANDOM_EFFECTS_TAG, ValueError, "7 levels.*3"),
        (7, "known-random-effectz", UnknownTagError, "known-random-effectz"),
    ],
)
def test_a_registered_design_the_run_cannot_answer_from_is_refused(
    tmp_path: Path, levels: int, tag: str, error: type[Exception], reason: str
) -> None:
    load_plugins()
    items = _items(tmp_path)

    with pytest.raises(error, match=reason):
        active_collector_for(
            RegisteredDesignCollectorSettings("registered-design", tag, None),
            _context(tmp_path, items, levels=levels, seed=0),
        )


def test_a_candidate_joined_to_no_design_item_is_refused(tmp_path: Path) -> None:
    load_plugins()
    items = _items(tmp_path)

    with pytest.raises(ValueError, match="rates no item"):
        active_collector_for(
            RegisteredDesignCollectorSettings(
                "registered-design", KNOWN_RANDOM_EFFECTS_TAG, None
            ),
            _context(tmp_path, items, levels=KNOWN_RANDOM_EFFECTS.levels, seed=0),
        )


def test_a_design_rating_fewer_items_than_the_pool_holds_is_refused(
    tmp_path: Path,
) -> None:
    items = _items(tmp_path)
    design = KnownEffectsDesign(
        "small",
        participants=2,
        items=len(items) - 1,
        levels=3,
        seed=0,
        latent_range=(RANGE.lower, RANGE.upper),
    )()

    with pytest.raises(ValueError, match=rf"{len(items) - 1} items.*{len(items)}"):
        SimulatedCollector.over(design, items)


def test_a_drawn_design_on_three_levels_answers_on_every_level(
    tmp_path: Path,
) -> None:
    load_plugins()
    items = _items(tmp_path)
    collector = active_collector_for(
        DrawnEffectsCollectorSettings("drawn-effects", 6, RANGE),
        _context(tmp_path, items, levels=3, seed=4),
    )

    assert {
        response for _, _, response in _triples(collector.collect(1, _batch(items)))
    } == {
        0,
        1,
        2,
    }
