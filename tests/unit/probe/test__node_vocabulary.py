import json
from hashlib import sha256
from typing import Final, cast

import pytest

from lofi.nn.sites import ComponentId, Role, Site
from lofi.probe._causal import IIAResult, SplitAt, TypeAt
from lofi.probe._das import AlignmentObjective, Assessment, FixedRankSearch, Schedule
from lofi.probe._node_vocabulary import (
    NodeVocabularyArm,
    NodeVocabularyCell,
    NodeVocabularyComparison,
    NodeVocabularyRow,
    node_vocabulary_row,
)
from lofi.probe._pricing import CostModel, ProbeUnit, UnitPrice
from lofi.probe._sweep import CellResult, Preregistration, Subgrid, SweepReport
from lofi.stats import LabelPermutation, PValue, p_value_against_replicates

VARIABLES: Final = (TypeAt((0, 2)),)
LAYERS: Final = (0, 1)
POSITIONS: Final = (0,)
PAIR_IDS: Final = ("pair-c", "pair-a", "pair-b")
ARMS: Final = tuple(NodeVocabularyArm)


def _price(unit: ProbeUnit) -> UnitPrice:
    return UnitPrice(
        unit,
        seconds_per_run=0.01,
        seconds_fastest=0.005,
        warmup=1,
        repeats=2,
        device="cpu",
    )


COSTS: Final = CostModel(
    [
        _price(ProbeUnit.FIXED_ALIGNMENT_FIT),
        _price(ProbeUnit.FIXED_PERMUTATION_REFIT),
    ]
)


def _p_value(seed: int) -> PValue:
    return p_value_against_replicates(
        0.5,
        (0.1, 0.3, 0.7),
        tail="greater",
        null="cell permutation",
        family="permutation",
        exact=False,
        seed=seed,
        matching={"layer": seed},
    )


def _report(
    component: str,
    accuracies: tuple[float, ...] = (0.25, 0.75),
    *,
    variables: tuple[TypeAt | SplitAt, ...] = VARIABLES,
    layers: tuple[int, ...] = LAYERS,
    null_offset: int = 0,
    mismatched_result_null: bool = False,
    complete: bool = True,
) -> SweepReport:
    sites = {
        (layer, position): Site(
            ComponentId(component), Role.RESIDUAL_STREAM, (layer, position)
        )
        for layer in layers
        for position in POSITIONS
    }
    subgrid = Subgrid(variables, layers, POSITIONS, sites)
    search = FixedRankSearch(
        rank=1,
        schedule=Schedule(epochs=1, learning_rate=0.1, batch_size=2, seed=0),
        null=LabelPermutation(replicates=3, seed=0),
        objective=AlignmentObjective.CROSS_ENTROPY,
    )
    preregistration = Preregistration(subgrid, search, costs=COSTS, budget_hours=1.0)
    results: list[CellResult] = []

    for index, (cell, accuracy) in enumerate(
        zip(subgrid.cells, accuracies, strict=True)
    ):
        p_value = _p_value(index + null_offset)
        result_null = (
            _p_value(index + null_offset + 100).provenance
            if mismatched_result_null and index == 0
            else p_value.provenance
        )
        result = IIAResult(accuracy, cell.variable, 1, result_null)
        results.append(
            CellResult(
                cell,
                sites[cell.coordinate],
                Assessment(result, p_value),
                preregistration.digest,
            )
        )

    return SweepReport(preregistration, results if complete else results[:-1])


def _rows(
    means: tuple[float, ...] = (0.5, 0.6, 0.4, 0.2),
) -> tuple[NodeVocabularyRow, ...]:
    return tuple(
        node_vocabulary_row(
            arm,
            _report(
                f"component-{index}",
                (mean - 0.1, mean + 0.1),
            ),
            tuple(reversed(PAIR_IDS)) if index % 2 else PAIR_IDS,
        )
        for index, (arm, mean) in enumerate(zip(ARMS, means, strict=True))
    )


def test_the_four_fixed_arms_have_distinct_wire_names() -> None:
    assert tuple(arm.value for arm in NodeVocabularyArm) == (
        "derivation-variables",
        "trained-transformer-sae-latents",
        "dense-linear-probe",
        "random-transformer-sae-control",
    )


def test_a_row_canonicalizes_pairs_and_derives_a_versioned_stable_digest() -> None:
    row = node_vocabulary_row(ARMS[0], _report("derivations"), PAIR_IDS)
    payload = {
        "domain": "lofi.node-vocabulary-pairs",
        "version": 1,
        "pair_ids": sorted(PAIR_IDS),
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    reordered = node_vocabulary_row(
        ARMS[0], _report("derivations"), tuple(reversed(PAIR_IDS))
    )

    assert row.pair_ids == tuple(sorted(PAIR_IDS))
    assert row.pair_digest == sha256(encoded).hexdigest()
    assert reordered.pair_digest == row.pair_digest
    assert tuple(cell.accuracy for cell in row.cells) == (0.25, 0.75)
    assert reordered.to_record() == row.to_record()
    assert repr(row).startswith("NodeVocabularyRow(")


def test_the_hand_computed_comparison_emits_exactly_four_scalar_rows() -> None:
    comparison = NodeVocabularyComparison(tuple(reversed(_rows())))
    summary = comparison.to_json()

    assert tuple(row.arm for row in comparison.rows) == ARMS
    assert [row["arm"] for row in summary] == [arm.value for arm in ARMS]
    assert [row["mean_iia"] for row in summary] == pytest.approx([0.5, 0.6, 0.4, 0.2])
    assert {row["cell_count"] for row in summary} == {2}
    assert {row["pair_digest"] for row in summary} == {comparison.rows[0].pair_digest}
    assert len(summary) == 4
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary
    assert all(len(row.cells) == 2 for row in comparison.rows)
    assert all("p_value" not in row for row in summary)


def test_arm_specific_sites_are_permitted_when_logical_cells_match() -> None:
    comparison = NodeVocabularyComparison(_rows())

    assert len({tuple(cell.cell for cell in row.cells) for row in comparison.rows}) == 1
    assert all("component" not in cell.cell for cell in comparison.rows[0].cells)


def test_a_row_record_round_trips_through_json() -> None:
    row = _rows()[1]
    record = json.loads(json.dumps(row.to_record(), allow_nan=False))

    assert NodeVocabularyRow.from_record(record).to_record() == row.to_record()


def test_a_row_record_refuses_a_digest_its_pairs_do_not_derive() -> None:
    record = _rows()[0].to_record()
    record["pair_digest"] = "0" * 64

    with pytest.raises(ValueError, match="pair digest"):
        NodeVocabularyRow.from_record(record)

    with pytest.raises(ValueError, match="fields"):
        NodeVocabularyRow.from_record({"arm": "dense-linear-probe"})

    with pytest.raises(ValueError, match="accuracy lies outside"):
        NodeVocabularyCell('{"variable":"v"}', 1.5, '{"name":"null"}')


def test_a_row_refuses_incomplete_or_incoherent_evidence() -> None:
    with pytest.raises(ValueError, match="unassessed"):
        node_vocabulary_row(ARMS[0], _report("partial", complete=False), PAIR_IDS)

    with pytest.raises(ValueError, match="p-value names"):
        node_vocabulary_row(
            ARMS[0],
            _report("mismatch", mismatched_result_null=True),
            PAIR_IDS,
        )


def test_a_row_refuses_empty_repeated_or_untyped_pair_identities() -> None:
    report = _report("pairs")

    with pytest.raises(ValueError, match="at least one pair"):
        node_vocabulary_row(ARMS[0], report, ())

    with pytest.raises(ValueError, match="repeats a pair identity"):
        node_vocabulary_row(ARMS[0], report, ("pair-a", "pair-a"))

    with pytest.raises(TypeError, match="not text"):
        node_vocabulary_row(ARMS[0], report, "pair-a")

    with pytest.raises(TypeError, match="not a string"):
        node_vocabulary_row(ARMS[0], report, (cast("str", 7),))


def test_rows_refuse_untyped_arms_and_reports() -> None:
    report = _report("types")

    with pytest.raises(TypeError, match="NodeVocabularyArm"):
        node_vocabulary_row(cast("NodeVocabularyArm", "dense"), report, PAIR_IDS)

    with pytest.raises(TypeError, match="SweepReport"):
        node_vocabulary_row(ARMS[0], cast("SweepReport", report.results), PAIR_IDS)


def test_a_comparison_refuses_missing_duplicate_or_untyped_rows() -> None:
    rows = _rows()

    with pytest.raises(ValueError, match="missing arms"):
        NodeVocabularyComparison(rows[:-1])

    with pytest.raises(ValueError, match="repeats arm"):
        NodeVocabularyComparison((*rows[:-1], rows[0]))

    with pytest.raises(TypeError, match="non-row"):
        NodeVocabularyComparison((*rows[:-1], cast("NodeVocabularyRow", rows[0].cells)))


def test_a_comparison_refuses_unequal_evaluated_pair_populations() -> None:
    rows = list(_rows())
    rows[-1] = node_vocabulary_row(
        rows[-1].arm, _report("component-3", (0.1, 0.3)), (*PAIR_IDS, "pair-d")
    )

    with pytest.raises(ValueError, match="pair digest"):
        NodeVocabularyComparison(rows)


def test_a_comparison_refuses_unequal_logical_cell_sequences() -> None:
    rows = list(_rows())
    rows[-1] = node_vocabulary_row(
        rows[-1].arm,
        _report("reordered", (0.1, 0.3), layers=tuple(reversed(LAYERS))),
        PAIR_IDS,
    )

    with pytest.raises(ValueError, match="logical cells"):
        NodeVocabularyComparison(rows)


def test_a_comparison_refuses_unequal_per_cell_null_provenance() -> None:
    rows = list(_rows())
    rows[-1] = node_vocabulary_row(
        rows[-1].arm,
        _report("other-null", (0.1, 0.3), null_offset=10),
        PAIR_IDS,
    )

    with pytest.raises(ValueError, match="different per-cell null provenance"):
        NodeVocabularyComparison(rows)
