import json
import math
import shutil
from datetime import date
from math import fsum
from typing import TYPE_CHECKING, Final, NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts.judgment.marginal_likelihood import (
    ADVANTAGE_METRIC,
    AGREEMENT_METRIC,
    CONDITIONAL,
    CONTRAST,
    DENSITY_METRIC,
    MANIFEST_NAME,
    MARGINAL,
    NEW_UNIT,
    OBJECTIVES,
    RECOVERY_METRIC,
    REPORT_NAME,
    SEEN_UNIT,
    StopRuleError,
    analyze,
    main,
    read_declaration,
)
from tests.unit.run.test__experiment import (
    a_judgment_experiment_document,
    write_experiment,
)

from lofi.judgment import (
    ComparisonSpec,
    ConditionalLikelihood,
    MapInference,
    ModelComparisonCriterion,
    ModelComparisonSpec,
    ModelWeighting,
    ResponseCompiler,
    TruncatedMarginalLikelihood,
)
from lofi.judgment.compiler import PARTICIPANT
from lofi.judgment.synthetic_designs import SCORE, KnownEffectsDesign
from lofi.run import (
    ExecutionEnvironment,
    ExperimentSeed,
    ExperimentStore,
    ManifestStatus,
    RunStore,
    SweepPoint,
    installed_packages,
    read_experiment,
)
from lofi.run.fits import (
    HELD_OUT_PARTICIPANT_FRACTION_METRIC,
    HELD_OUT_PREDICTIVE_METRIC,
    POINT_ESTIMATE_NAME,
    FitExecutor,
)
from lofi.run.results import MANIFESTS_NAME, ROWS_NAME, write_results
from lofi.spec import HeldOutParticipantPolicy
from lofi.spec.optimizers.adam import AdamOptimizerSpec

if TYPE_CHECKING:
    from pathlib import Path

    import didactic.api as dx

    from lofi._types import JsonValue
    from lofi.judgment import ReadDesign

FREEZE: Final = "freeze/1"
SEEDS: Final = (0, 1)
SITES: Final = ("participant_intercept_scale", "participant_cutpoint_spread")
TOLERANCE: Final = 0.5
DESIGN: Final = KnownEffectsDesign(
    "known-random-effects-reduced",
    participants=6,
    items=5,
    levels=4,
    seed=11,
    latent_range=(-2.0, 2.0),
)
OTHER: Final = KnownEffectsDesign(
    "known-random-effects-reduced",
    participants=6,
    items=5,
    levels=4,
    seed=12,
    latent_range=(-2.0, 2.0),
)


class ReducedDesigns:
    def read(self, dataset: str, /) -> ReadDesign:
        assert dataset == DESIGN.tag
        return DESIGN.read()


def dumped(model: dx.Model) -> JsonValue:
    return json.loads(model.model_dump_json())


def a_cell_fit(objective: str, unit: str) -> dict[str, JsonValue]:
    likelihood = (
        TruncatedMarginalLikelihood(
            group=PARTICIPANT,
            radius=3.0,
            relative_tolerance=1e-5,
            absolute_tolerance=1e-7,
            quadrature_points=5,
            quadrature_seed=19,
        )
        if objective == MARGINAL
        else ConditionalLikelihood()
    )
    comparison = ComparisonSpec(
        held_out_unit=unit,
        readout_units={SCORE: "item"},
        splitter="kfold",
        folds=2,
        seed=23,
        coverage_threshold=0.0 if unit == SEEN_UNIT else 0.5,
        unseen_participant=(
            HeldOutParticipantPolicy.MARGINAL
            if objective == MARGINAL
            else HeldOutParticipantPolicy.PLUG_IN
        ),
        models=ModelComparisonSpec(
            candidates=OBJECTIVES,
            candidate=objective,
            criterion=ModelComparisonCriterion.PSIS_LOO,
            weighting=ModelWeighting.STACKING,
            pareto_fraction=0.1,
            pareto_threshold=0.7,
        ),
    )

    return {
        "dataset": DESIGN.tag,
        "response": {
            "ResponseDeclaration_responsespec": dumped(
                DESIGN.spec.with_(objective=likelihood)
            )
        },
        "inference": dumped(
            MapInference(
                optimizer=AdamOptimizerSpec(learning_rate=0.05),
                steps=20,
                initial_scale=0.1,
            )
        ),
        "elicitation": {"method": "participants", "version": "1"},
        "calibration": None,
        "comparison": dumped(comparison),
        "dynamic_range": None,
        "ablation": [],
        "interannotator": None,
    }


def a_declaration(*, margin: float = 0.5) -> dict[str, JsonValue]:
    return {
        "freeze_id": FREEZE,
        "grammar_fingerprint": "grammar:none",
        "suite": "judgment-models",
        "task": "marginal-likelihood",
        "split": "test",
        "seeds": list(SEEDS),
        "bootstrap": {"replicates": 199, "seed": 5, "level": 0.95, "interval": "bca"},
        "agreement_margin": margin,
        "recovery": {"sites": list(SITES), "tolerance": TOLERANCE},
    }


class Campaign(NamedTuple):
    results: Path
    runs: Path
    declaration: Path


def executed(root: Path) -> Campaign:
    document = a_judgment_experiment_document()
    document["seed_policy"] = {"root": 7}
    document["freeze_id"] = FREEZE
    experiment = read_experiment(
        write_experiment(root / "repository", document, name="marginal-likelihood")
    )
    store = RunStore(root / "store", freeze_id=FREEZE, seed=ExperimentSeed(7))
    executor = FitExecutor(
        ExperimentStore(store.root, experiment),
        designs=ReducedDesigns(),
        compiler=ResponseCompiler.installed(),
        environment=ExecutionEnvironment(installed_packages(), "0123abcd", "cpu"),
        infrastructure_attempts=3,
    )
    points = tuple(
        SweepPoint(a_cell_fit(objective, unit), seed=seed)
        for objective in OBJECTIVES
        for unit in (SEEN_UNIT, NEW_UNIT)
        for seed in SEEDS
    )

    for point in points:
        assert executor(point).status is ManifestStatus.COMPLETED

    results = root / "results"
    write_results(results, points=points, store=executor.store, today=date(2026, 11, 1))
    declaration = root / "declaration.json"
    declaration.write_text(json.dumps(a_declaration()), encoding="utf-8", newline="\n")

    return Campaign(results, executor.store.directory(points[0]).parent, declaration)


@pytest.fixture(scope="module")
def campaign(tmp_path_factory: pytest.TempPathFactory) -> Campaign:
    return executed(tmp_path_factory.mktemp("campaign"))


def copied(campaign: Campaign, root: Path) -> Campaign:
    results = root / "results"
    shutil.copytree(campaign.results, results)
    runs = root / "runs"
    shutil.copytree(campaign.runs, runs)

    return Campaign(results, runs, campaign.declaration)


def run(campaign: Campaign, out: Path, *, truth: KnownEffectsDesign = DESIGN) -> None:
    analyze(
        campaign.results,
        campaign.declaration,
        out,
        runs=campaign.runs,
        truth=truth,
        table=ROWS_NAME,
    )


def report_rows(out: Path) -> list[dict[str, JsonValue]]:
    return pq.read_table(out / REPORT_NAME).to_pylist()


def manifest_of(out: Path) -> dict[str, JsonValue]:
    return json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))


def result_rows(campaign: Campaign) -> list[dict[str, JsonValue]]:
    return pq.read_table(campaign.results / ROWS_NAME).to_pylist()


def participant_totals(campaign: Campaign, point: str) -> dict[str, tuple[float, int]]:
    totals: dict[str, tuple[float, int]] = {}

    for row in result_rows(campaign):
        if row["metric"] == HELD_OUT_PREDICTIVE_METRIC and row["point_id"] == point:
            participant = row["participant"]
            value = row["value"]
            n = row["n"]
            assert isinstance(participant, str)
            assert isinstance(value, float)
            assert isinstance(n, int)
            total, count = totals.get(participant, (0.0, 0))
            totals[participant] = (total + value * n, count + n)

    return totals


def points_by_cell(out: Path) -> dict[tuple[str, str, int], str]:
    points = manifest_of(out)["points"]
    assert isinstance(points, list)
    found: dict[tuple[str, str, int], str] = {}

    for entry in points:
        assert isinstance(entry, dict)
        objective = entry["objective"]
        unit = entry["held_out_unit"]
        seed = entry["seed"]
        point = entry["point_id"]
        assert isinstance(objective, str)
        assert isinstance(unit, str)
        assert isinstance(seed, int)
        assert isinstance(point, str)
        found[objective, unit, seed] = point

    return found


def test_each_candidates_density_pools_its_participants_held_out_responses(
    campaign: Campaign, tmp_path: Path
) -> None:
    run(campaign, tmp_path)
    cells = points_by_cell(tmp_path)
    densities = [
        row for row in report_rows(tmp_path) if row["metric"] == DENSITY_METRIC
    ]

    assert len(cells) == len(densities) == 8
    assert all(row["convention_hash"] is None for row in report_rows(tmp_path))

    for row in densities:
        method = row["method"]
        rung = row["rung"]
        seed = row["seed"]
        assert isinstance(method, str)
        assert isinstance(rung, str)
        assert isinstance(seed, int)
        objective, unit = rung.split("/")
        assert objective == method
        totals = participant_totals(campaign, cells[method, unit, seed])

        assert len(totals) == DESIGN.participants == row["n"]
        assert row["value"] == pytest.approx(
            fsum(total for total, _ in totals.values())
            / sum(count for _, count in totals.values())
        )
        assert row["resampling_unit"] == "participant"
        assert isinstance(row["ci_low"], float)
        assert isinstance(row["ci_high"], float)
        assert row["ci_low"] <= row["ci_high"]
        assert row["held_out_convention"] == (
            None
            if unit == SEEN_UNIT
            else ("marginal" if method == MARGINAL else "plug_in")
        )


def test_every_seed_contrasts_the_two_objectives_under_both_held_out_units(
    campaign: Campaign, tmp_path: Path
) -> None:
    run(campaign, tmp_path)
    cells = points_by_cell(tmp_path)
    contrasts = {
        (row["metric"], row["seed"]): row
        for row in report_rows(tmp_path)
        if row["method"] == CONTRAST
    }

    assert set(contrasts) == {
        (metric, seed)
        for metric in (AGREEMENT_METRIC, ADVANTAGE_METRIC)
        for seed in SEEDS
    }

    for (metric, seed), row in contrasts.items():
        assert isinstance(seed, int)
        unit = SEEN_UNIT if metric == AGREEMENT_METRIC else NEW_UNIT
        marginal = participant_totals(campaign, cells[MARGINAL, unit, seed])
        conditional = participant_totals(campaign, cells[CONDITIONAL, unit, seed])

        assert {key: count for key, (_, count) in marginal.items()} == {
            key: count for key, (_, count) in conditional.items()
        }
        assert row["value"] == pytest.approx(
            (
                fsum(total for total, _ in marginal.values())
                - fsum(total for total, _ in conditional.values())
            )
            / sum(count for _, count in marginal.values())
        )
        assert row["n"] == DESIGN.participants


def test_the_manifest_applies_each_declared_criterion_to_the_reported_interval(
    campaign: Campaign, tmp_path: Path
) -> None:
    run(campaign, tmp_path)
    rows = {
        (row["metric"], row["seed"]): row
        for row in report_rows(tmp_path)
        if row["method"] == CONTRAST
    }
    contrasts = manifest_of(tmp_path)["contrasts"]
    margin = a_declaration()["agreement_margin"]
    assert isinstance(contrasts, list)
    assert isinstance(margin, float)

    for entry in contrasts:
        assert isinstance(entry, dict)
        seed = entry["seed"]
        agreement = entry[AGREEMENT_METRIC]
        advantage = entry[ADVANTAGE_METRIC]
        assert isinstance(agreement, dict)
        assert isinstance(advantage, dict)
        seen = rows[AGREEMENT_METRIC, seed]
        new = rows[ADVANTAGE_METRIC, seed]
        low, high = seen["ci_low"], seen["ci_high"]
        assert isinstance(low, float)
        assert isinstance(high, float)
        new_low, new_high, value = new["ci_low"], new["ci_high"], new["value"]
        assert isinstance(new_low, float)
        assert isinstance(new_high, float)
        assert isinstance(value, float)

        assert agreement["holds"] is (-margin <= low and high <= margin)
        assert advantage["holds"] is (value > (new_high - new_low) / 2.0)
        assert advantage["read"] is agreement["holds"]


def test_each_fitted_site_is_compared_with_the_value_it_was_drawn_at(
    campaign: Campaign, tmp_path: Path
) -> None:
    run(campaign, tmp_path)
    recoveries = [
        row for row in report_rows(tmp_path) if row["metric"] == RECOVERY_METRIC
    ]
    recorded = manifest_of(tmp_path)["recovery"]
    assert isinstance(recorded, list)

    assert len(recoveries) == len(recorded) == 8 * len(SITES)
    assert {str(row["method"]).split(":")[1] for row in recoveries} == set(SITES)

    for entry in recorded:
        assert isinstance(entry, dict)
        error = entry["mean_absolute_error"]
        assert isinstance(error, float)
        assert math.isfinite(error)
        assert error >= 0.0
        assert entry["recovered"] is (error <= TOLERANCE)


def test_a_fit_of_another_synthetic_draw_stops_the_comparison(
    campaign: Campaign, tmp_path: Path
) -> None:
    with pytest.raises(StopRuleError, match="drawn from"):
        run(campaign, tmp_path, truth=OTHER)

    assert not (tmp_path / REPORT_NAME).exists()


def test_a_seen_participant_fold_holding_out_a_new_participant_stops_it(
    campaign: Campaign, tmp_path: Path
) -> None:
    tampered = copied(campaign, tmp_path / "tampered")
    seen = {
        point
        for (objective, unit, _), point in points_by_cell_of(campaign, tmp_path).items()
        if unit == SEEN_UNIT and objective == CONDITIONAL
    }
    rows = result_rows(tampered)

    for row in rows:
        if (
            row["metric"] == HELD_OUT_PARTICIPANT_FRACTION_METRIC
            and row["point_id"] in seen
        ):
            row["value"] = 0.25

    pq.write_table(pa.Table.from_pylist(rows), tampered.results / ROWS_NAME)

    with pytest.raises(StopRuleError, match="holds out new participants"):
        run(tampered, tmp_path / "out")


def points_by_cell_of(
    campaign: Campaign, root: Path
) -> dict[tuple[str, str, int], str]:
    out = root / "reference"
    run(campaign, out)

    return points_by_cell(out)


def test_a_tampered_point_estimate_is_refused(
    campaign: Campaign, tmp_path: Path
) -> None:
    tampered = copied(campaign, tmp_path / "tampered")
    point = next(iter(points_by_cell_of(campaign, tmp_path).values()))
    (tampered.runs / point / POINT_ESTIMATE_NAME).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        run(tampered, tmp_path / "out")


def test_a_candidate_predicting_under_another_convention_is_refused(
    campaign: Campaign, tmp_path: Path
) -> None:
    tampered = copied(campaign, tmp_path / "tampered")
    cells = points_by_cell_of(campaign, tmp_path)
    path = tampered.results / MANIFESTS_NAME / f"{cells[MARGINAL, NEW_UNIT, 0]}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["fit"]["comparison"]["unseen_participant"] = "plug_in"
    path.write_text(json.dumps(manifest), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="its own convention"):
        run(tampered, tmp_path / "out")


def test_a_missing_cell_seed_is_recorded_and_its_contrast_is_not_read(
    campaign: Campaign, tmp_path: Path
) -> None:
    tampered = copied(campaign, tmp_path / "tampered")
    cells = points_by_cell_of(campaign, tmp_path)
    missing = cells[CONDITIONAL, NEW_UNIT, 1]
    (tampered.results / MANIFESTS_NAME / f"{missing}.json").unlink()
    out = tmp_path / "out"
    run(tampered, out)
    manifest = manifest_of(out)
    contrasts = manifest["contrasts"]
    recorded = manifest["cells"]
    assert isinstance(contrasts, list)
    assert isinstance(recorded, list)

    assert {
        "objective": CONDITIONAL,
        "held_out_unit": NEW_UNIT,
        "completed_seeds": [0],
        "missing_seeds": [1],
    } in recorded
    assert [
        entry[ADVANTAGE_METRIC] is None
        for entry in contrasts
        if isinstance(entry, dict)
    ] == [False, True]
    assert (ADVANTAGE_METRIC, 1) not in {
        (row["metric"], row["seed"])
        for row in report_rows(out)
        if row["method"] == CONTRAST
    }


def test_the_command_refuses_results_that_fit_no_single_dataset(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    (results / MANIFESTS_NAME).mkdir(parents=True)
    declaration = tmp_path / "declaration.json"
    declaration.write_text(json.dumps(a_declaration()), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="exactly one dataset"):
        main(
            [
                str(results),
                "--runs",
                str(tmp_path / "runs"),
                "--declaration",
                str(declaration),
                "--out",
                str(tmp_path / "out"),
            ]
        )


def test_a_declaration_without_a_positive_agreement_margin_is_refused(
    tmp_path: Path,
) -> None:
    path = tmp_path / "declaration.json"
    path.write_text(
        json.dumps(a_declaration(margin=0.0)), encoding="utf-8", newline="\n"
    )

    with pytest.raises(ValueError, match="agreement margin"):
        read_declaration(path)


def test_a_declaration_naming_parsing_conventions_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "declaration.json"
    declared = {**a_declaration(), "conventions": {"punctuation": "keep"}}
    path.write_text(json.dumps(declared), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="conventions"):
        read_declaration(path)
