import json
from typing import TYPE_CHECKING, Final

import pytest
import torch
from scripts.probe.acdc_calibration import (
    COMPLETED,
    DECLARED_FIELDS,
    MANIFEST_NAME,
    OVER_BUDGET,
    ROWS_NAME,
    CalibrationCell,
    calibrate_cell,
    declaration_of,
    main,
    run,
)
from tests.unit.nn.parsers.test__scorer import build, leaves

from lofi.deduce import SpanIndex
from lofi.probe import receiver_circuit
from lofi.run import read_manifest

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue

DECLARED: Final[dict[str, JsonValue]] = {
    "scorer": "parser",
    "grid": [
        {
            "name": "three-words",
            "clean": ["word-0", "word-2", "word-4"],
            "counterfactual": ["word-4", "word-6", "word-0"],
        },
        {
            "name": "five-words",
            "clean": ["word-3", "word-0", "word-2", "word-7", "word-4"],
            "counterfactual": ["word-7", "word-4", "word-6", "word-3", "word-0"],
        },
    ],
    "steps": 2,
    "threshold": {"kind": "absolute", "bound": 0.0},
    "maximum_trials": 6,
    "divergence": "mean-absolute",
    "metric": "log-partition",
    "budget_seconds": 600.0,
    "price_warmup": 0,
    "price_repeats": 1,
}

#: The words in each declared cell.
WORDS: Final = {"three-words": 3, "five-words": 5}

#: Row fields that are wall-clock measurements rather than decisions.
TIMED: Final = ("wall_seconds", "price")


def cells() -> tuple[CalibrationCell, ...]:
    return tuple(
        CalibrationCell(
            name,
            leaves(1, words, seed=5, dtype=torch.float64),
            leaves(1, words, seed=17, dtype=torch.float64),
        )
        for name, words in WORDS.items()
    )


@pytest.fixture
def unchecked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Price units without the debug checks, which a price must not time."""
    monkeypatch.setattr("lofi._checks.CHECKS", False)


def untimed(row: dict[str, JsonValue]) -> dict[str, JsonValue]:
    kept: dict[str, JsonValue] = {}

    for key, value in row.items():
        if isinstance(value, dict):
            kept[key] = {
                field: held for field, held in value.items() if field not in TIMED
            }
        else:
            kept[key] = value

    return kept


def test_every_declared_field_is_required() -> None:
    for field in DECLARED_FIELDS:
        partial = {key: value for key, value in DECLARED.items() if key != field}

        with pytest.raises(ValueError, match=field):
            declaration_of(partial)

    with pytest.raises(ValueError, match="unknown"):
        declaration_of({**DECLARED, "seed": 1})


@pytest.mark.parametrize(
    ("field", "value", "kind"),
    [
        ("steps", 0, ValueError),
        ("steps", True, TypeError),
        ("threshold", 0.01, TypeError),
        ("threshold", {"kind": "relative", "fraction": -1.0}, ValueError),
        ("threshold", {"kind": "absolute", "bound": -1.0}, ValueError),
        ("threshold", {"kind": "relative"}, ValueError),
        ("threshold", {"kind": "scaled", "fraction": 0.01}, ValueError),
        ("maximum_trials", -1, ValueError),
        ("divergence", "kl", ValueError),
        ("metric", "inside", ValueError),
        ("budget_seconds", 0.0, ValueError),
        ("price_repeats", 0, ValueError),
        ("grid", [], TypeError),
        ("grid", ["three-words"], TypeError),
        (
            "grid",
            [
                {"name": "a", "clean": ["x"], "counterfactual": ["y"]},
                {"name": "a", "clean": ["x"], "counterfactual": ["y"]},
            ],
            ValueError,
        ),
        ("grid", [{"name": "a", "clean": ["x"], "counterfactual": []}], ValueError),
        (
            "grid",
            [{"name": "a", "clean": ["x"], "counterfactual": ["y", "z"]}],
            ValueError,
        ),
        ("grid", [{"name": "a", "clean": ["x"]}], ValueError),
        ("scorer", "", ValueError),
    ],
)
def test_an_invalid_declared_value_is_refused(
    field: str, value: JsonValue, kind: type[Exception]
) -> None:
    with pytest.raises(kind):
        declaration_of({**DECLARED, field: value})


@pytest.mark.usefixtures("unchecked")
def test_a_cell_row_records_both_estimators_over_the_receiver_order() -> None:
    scorer = build(dtype=torch.float64)
    cell = cells()[1]
    declaration = declaration_of(DECLARED)
    row = calibrate_cell(scorer, cell, declaration)
    circuit = receiver_circuit(scorer, SpanIndex(WORDS[cell.name]), reading=cell.name)
    order = [circuit.targets[edge].index["edge"] for edge in circuit.order]
    acdc = row["acdc"]
    eap_ig = row["eap_ig"]

    assert isinstance(acdc, dict)
    assert isinstance(eap_ig, dict)
    assert row["edge_order"] == order
    assert row["edges"] == len(order)
    assert acdc["tested"] == order[: declaration.maximum_trials]
    assert acdc["scheduling_residual"] == pytest.approx(
        (len(order) - declaration.maximum_trials) / len(order)
    )
    retained = acdc["retained"]
    matched = eap_ig["matched_circuit"]
    effects = eap_ig["effects"]

    assert isinstance(retained, list)
    assert isinstance(matched, list)
    assert isinstance(effects, list)
    assert len(matched) == len(retained)
    assert len(set(effects)) == len(effects)
    jaccard = row["matched_cardinality_jaccard"]

    assert isinstance(jaccard, float)
    assert 0.0 <= jaccard <= 1.0
    eap_price = eap_ig["price"]
    acdc_price = acdc["price"]

    assert isinstance(eap_price, dict)
    assert isinstance(acdc_price, dict)
    assert eap_price["unit"] == "attribution"
    assert acdc_price["unit"] == "pruning"


@pytest.mark.usefixtures("unchecked")
def test_a_relative_threshold_prunes_against_the_cell_s_full_corruption() -> None:
    scorer = build(dtype=torch.float64)
    relative: JsonValue = {"kind": "relative", "fraction": 0.05}
    declaration = declaration_of({**DECLARED, "threshold": relative})
    row = calibrate_cell(scorer, cells()[1], declaration)
    absolute = calibrate_cell(scorer, cells()[1], declaration_of(DECLARED))
    acdc = row["acdc"]

    assert isinstance(acdc, dict)
    corrupted = acdc["corrupted_divergence"]

    assert isinstance(corrupted, float)
    assert acdc["threshold"] == relative
    assert acdc["cutoff"] == pytest.approx(0.05 * corrupted)
    held = absolute["acdc"]

    assert isinstance(held, dict)
    assert held["threshold"] == DECLARED["threshold"]
    assert held["cutoff"] == 0.0


@pytest.mark.usefixtures("unchecked")
def test_each_circuit_records_its_faithfulness_share_and_raw_divergences() -> None:
    scorer = build(dtype=torch.float64)
    row = calibrate_cell(scorer, cells()[1], declaration_of(DECLARED))

    for estimator in ("acdc", "eap_ig"):
        record = row[estimator]

        assert isinstance(record, dict)
        circuit = record["circuit_divergence"]
        corrupted = record["corrupted_divergence"]
        share = record["faithfulness"]

        assert isinstance(circuit, float)
        assert isinstance(corrupted, float)
        assert isinstance(share, float)
        assert circuit >= 0.0
        assert corrupted > 0.0
        assert share == pytest.approx(1.0 - circuit / corrupted)


@pytest.mark.usefixtures("unchecked")
def test_a_campaign_writes_its_manifest_and_reruns_deterministically(
    tmp_path: Path,
) -> None:
    scorer = build(dtype=torch.float64)
    declaration = declaration_of(DECLARED)
    written: list[list[dict[str, JsonValue]]] = []

    for attempt in ("first", "second"):
        out = tmp_path / attempt

        assert run(scorer, cells(), declaration, out, point="unit") == COMPLETED

        manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
        lines = (out / ROWS_NAME).read_text(encoding="utf-8").splitlines()

        assert manifest["declaration"] == DECLARED
        assert manifest["point"] == "unit"
        assert manifest["calibrated"] == list(WORDS)
        assert manifest["outcome"] == COMPLETED
        written.append([untimed(json.loads(line)) for line in lines])

    assert written[0] == written[1]

    with pytest.raises(FileExistsError):
        run(scorer, cells(), declaration, tmp_path / "first", point="unit")


def test_a_campaign_stops_at_its_compute_budget(tmp_path: Path) -> None:
    scorer = build(dtype=torch.float64)
    tight = declaration_of({**DECLARED, "budget_seconds": 1e-9})

    assert run(scorer, cells(), tight, tmp_path / "tight", point="unit") == OVER_BUDGET

    manifest = json.loads(
        (tmp_path / "tight" / MANIFEST_NAME).read_text(encoding="utf-8")
    )

    assert manifest["outcome"] == OVER_BUDGET
    assert len(manifest["calibrated"]) < len(WORDS)


def test_supplied_cells_must_be_the_declared_grid(tmp_path: Path) -> None:
    scorer = build(dtype=torch.float64)

    with pytest.raises(ValueError, match="declared grid"):
        run(
            scorer,
            cells()[::-1],
            declaration_of(DECLARED),
            tmp_path / "out",
            point="unit",
        )


@pytest.mark.usefixtures("unchecked")
def test_main_calibrates_the_declared_cells_of_a_restored_trained_point(
    trained_run: Path, tmp_path: Path
) -> None:
    declared = tmp_path / "calibration.json"
    declared.write_text(json.dumps(DECLARED), encoding="utf-8", newline="\n")
    out = tmp_path / "calibrated"

    status = main(
        ["--run", str(trained_run), "--declaration", str(declared), "--out", str(out)]
    )
    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (out / ROWS_NAME).read_text(encoding="utf-8").splitlines()
    ]
    recorded = read_manifest(trained_run)

    assert status == 0
    assert manifest["outcome"] == COMPLETED
    assert manifest["declaration"] == DECLARED
    assert recorded is not None
    assert manifest["point"] == recorded.seed_derivation.point_id
    assert [row["cell"] for row in rows] == ["three-words", "five-words"]
    assert all(row["edges"] > 0 for row in rows)
    assert all(isinstance(row["acdc"]["circuit_divergence"], float) for row in rows)


def test_main_refuses_a_scorer_role_the_trained_point_does_not_hold(
    trained_run: Path, tmp_path: Path
) -> None:
    declared = tmp_path / "calibration.json"
    declared.write_text(
        json.dumps({**DECLARED, "scorer": "token_encoder"}),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(TypeError, match="token_encoder"):
        main(
            [
                "--run",
                str(trained_run),
                "--declaration",
                str(declared),
                "--out",
                str(tmp_path / "out"),
            ]
        )
