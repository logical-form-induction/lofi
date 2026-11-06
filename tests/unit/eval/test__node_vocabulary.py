"""Tests for the node-vocabulary comparison reducer."""

import json
from typing import TYPE_CHECKING, cast

import pytest
from tests.unit.probe.test__node_vocabulary import ARMS, PAIR_IDS, _report

from lofi.eval._node_vocabulary import PLUGIN
from lofi.eval._reducers import ReductionContext
from lofi.probe._node_vocabulary import node_vocabulary_row
from lofi.probe._probing import PROBE_RECORDS_NAME, write_probe_records
from lofi.spec.reducers import NodeVocabularyReducerSpec

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue

MEANS = (0.5, 0.6, 0.4, 0.2)


def _arms(
    root: Path, *, pairs: tuple[str, ...] = PAIR_IDS
) -> list[dict[str, JsonValue]]:
    declared: list[dict[str, JsonValue]] = []

    for index, (arm, mean) in enumerate(zip(ARMS, MEANS, strict=True)):
        directory = root / "arms" / arm.value
        directory.mkdir(parents=True)
        row = node_vocabulary_row(
            arm,
            _report(f"component-{index}", (mean - 0.1, mean + 0.1)),
            pairs if index == len(ARMS) - 1 else PAIR_IDS,
        )
        write_probe_records(
            directory / PROBE_RECORDS_NAME, {"causal": {"pass-0": row.to_record()}}
        )
        declared.append(
            {
                "representation": arm.value,
                "source": f"arms/{arm.value}/{PROBE_RECORDS_NAME}",
                "probe": "causal",
                "record": "pass-0",
            }
        )

    return declared


def _reduce(
    root: Path, arms: list[dict[str, JsonValue]]
) -> tuple[dict[str, JsonValue], ...]:
    declaration = NodeVocabularyReducerSpec.model_validate(
        {"kind": "node-vocabulary", "arms": cast("list[JsonValue]", arms)}
    )
    reducer = PLUGIN.construct(declaration, ReductionContext(root, seed=0))

    return tuple(dict(row) for row in reducer.reduce(()).rows)


def test_the_reducer_compares_the_four_persisted_arm_sweeps(tmp_path: Path) -> None:
    rows = _reduce(tmp_path, _arms(tmp_path))

    assert [row["arm"] for row in rows] == [arm.value for arm in ARMS]
    assert [row["mean_iia"] for row in rows] == pytest.approx(list(MEANS))
    assert {row["reducer"] for row in rows} == {"node-vocabulary"}
    assert {row["cell_count"] for row in rows} == {2}
    assert len({row["pair_digest"] for row in rows}) == 1
    assert json.loads(json.dumps(rows, allow_nan=False)) == list(rows)


def test_the_reducer_refuses_arms_over_different_pair_populations(
    tmp_path: Path,
) -> None:
    arms = _arms(tmp_path, pairs=(*PAIR_IDS, "pair-d"))

    with pytest.raises(ValueError, match="pair digest"):
        _reduce(tmp_path, arms)


def test_the_reducer_refuses_a_record_of_another_arm_or_an_absent_one(
    tmp_path: Path,
) -> None:
    arms = _arms(tmp_path)
    swapped = [arms[0], {**arms[1], "source": arms[0]["source"]}, *arms[2:]]

    with pytest.raises(ValueError, match="records arm"):
        _reduce(tmp_path, swapped)

    with pytest.raises(ValueError, match="no record 'pass-1'"):
        _reduce(tmp_path, [{**arms[0], "record": "pass-1"}, *arms[1:]])

    with pytest.raises(ValueError, match="cannot be read"):
        _reduce(tmp_path, [{**arms[0], "source": "arms/none.json"}, *arms[1:]])
