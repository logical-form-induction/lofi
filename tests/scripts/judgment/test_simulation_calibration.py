import json
from typing import TYPE_CHECKING, Final, TypedDict

import pytest
from scripts.judgment.simulation_calibration import (
    COMPLETED,
    FAILED,
    MANIFEST_NAME,
    METHODS,
    ROWS_NAME,
    CampaignSettings,
    main,
    run,
    settings_of,
)

from lofi.judgment import (
    CalibrationCampaign,
    CalibrationSeeds,
    InferenceMethod,
    ReplicateRecord,
    ResponseCompiler,
    spec_digest,
)
from lofi.judgment.synthetic_designs import KnownEffectsDesign
from lofi.run import source_revision

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


class NutsJson(TypedDict):
    divergences: list[int]
    acceptance_rates: list[float]
    r_hat: dict[str, float]
    effective_sample_sizes: dict[str, float]
    max_rhat: float
    min_ess: float


class ReplicateJson(TypedDict):
    replicate: int
    simulation: int
    inference: dict[str, int]
    retained: bool
    nuts: NutsJson | None
    evidence_bounds: dict[str, float]
    failures: dict[str, str]


TINY: Final = KnownEffectsDesign(
    "tiny-calibration",
    participants=2,
    items=3,
    levels=3,
    seed=11,
    latent_range=(-1.5, 1.5),
)

DECLARED: Final[dict[str, int | float]] = {
    "replicates": 2,
    "warmup": 2,
    "retained": 4,
    "chains": 2,
    "guide_iterations": 10,
    "guide_learning_rate": 0.05,
    "simulation_seed": 1,
    "inference_seed": 2,
    "rank_seed": 3,
}


def settings() -> CampaignSettings:
    return settings_of(
        replicates=int(DECLARED["replicates"]),
        warmup=int(DECLARED["warmup"]),
        retained=int(DECLARED["retained"]),
        chains=int(DECLARED["chains"]),
        guide_iterations=int(DECLARED["guide_iterations"]),
        guide_learning_rate=float(DECLARED["guide_learning_rate"]),
        simulation_seed=int(DECLARED["simulation_seed"]),
        inference_seed=int(DECLARED["inference_seed"]),
        rank_seed=int(DECLARED["rank_seed"]),
    )


def arguments(out: Path, **changed: float) -> list[str]:
    declared = DECLARED | changed

    return [
        *(
            token
            for name, value in declared.items()
            for token in (f"--{name.replace('_', '-')}", str(value))
        ),
        "--out",
        str(out),
    ]


def recorded(directory: Path, name: str) -> dict[str, JsonValue]:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def rows(directory: Path) -> list[dict[str, JsonValue]]:
    lines = (directory / ROWS_NAME).read_text(encoding="utf-8").splitlines()

    return [json.loads(line) for line in lines]


STREAMS: Final = CalibrationSeeds(
    simulation=int(DECLARED["simulation_seed"]),
    inference=int(DECLARED["inference_seed"]),
    rank=int(DECLARED["rank_seed"]),
)


def derived_seeds() -> list[tuple[int, dict[str, int]]]:
    return [
        (
            STREAMS.simulation(replicate),
            {method.value: STREAMS.inference(method, replicate) for method in METHODS},
        )
        for replicate in range(int(DECLARED["replicates"]))
    ]


def replicate_records(directory: Path) -> list[ReplicateJson]:
    text = (directory / MANIFEST_NAME).read_text(encoding="utf-8")

    return json.loads(text)["replicate_records"]


def recorded_seeds(directory: Path) -> list[tuple[int, dict[str, int]]]:
    return [
        (record["simulation"], record["inference"])
        for record in replicate_records(directory)
    ]


FLOOR: Final = "lofi.judgment.calibration._RECOVERY_FLOOR"

FLOOR_FIELDS: Final = frozenset({"floor", "clears_floor"})


@pytest.fixture(scope="module")
def completed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("campaign")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(FLOOR, -1.0)

        assert run(settings(), directory, source=TINY) == 0

    return directory


def test_a_completed_campaign_reduces_every_replicate_fitted_by_every_method(
    completed: Path,
) -> None:
    reduced = rows(completed)
    compiled = ResponseCompiler.installed().compile(TINY.spec)
    summaries = {row["summary"] for row in reduced}
    sbc = [row for row in reduced if row["summary"] == "sbc"]
    coverage = [row for row in reduced if row["summary"] == "coverage"]
    recovery = [row for row in reduced if row["summary"] == "cutpoint_recovery"]

    assert summaries == {"sbc", "sbc_rank_bin", "coverage", "cutpoint_recovery"}
    assert {row["replicates"] for row in reduced} == {DECLARED["replicates"]}
    assert {row["rank_seed"] for row in reduced} == {DECLARED["rank_seed"]}
    assert sorted(str(row["block"]) for row in sbc) == sorted(
        draw.site for draw in compiled.draws
    )
    assert {row["posterior_samples"] for row in sbc} == {
        DECLARED["chains"] * DECLARED["retained"]
    }
    assert {row["method"] for row in coverage} == {method.value for method in METHODS}
    assert {row["method"] for row in recovery} == {method.value for method in METHODS}
    assert {row["cutpoint"] for row in recovery} == set(range(TINY.levels - 1))
    assert {row["participants"] for row in recovery} == {
        DECLARED["replicates"] * TINY.participants
    }


def test_a_completed_manifest_records_the_declaration_the_rows_were_reduced_under(
    completed: Path,
) -> None:
    manifest = recorded(completed, MANIFEST_NAME)

    assert manifest["outcome"] == COMPLETED
    assert manifest["retained_replicates"] == DECLARED["replicates"]
    assert manifest["failed_replicates"] == []
    assert manifest["clears_floor"] is True
    assert manifest["replicates"] == DECLARED["replicates"]
    assert manifest["nuts"] == {
        "warmup": DECLARED["warmup"],
        "retained": DECLARED["retained"],
        "chains": DECLARED["chains"],
    }
    assert manifest["guides"] == {
        "methods": [method.value for method in METHODS[1:]],
        "iterations": DECLARED["guide_iterations"],
        "learning_rate": DECLARED["guide_learning_rate"],
        "draws": DECLARED["chains"] * DECLARED["retained"],
    }
    assert manifest["seeds"] == {
        "simulation": DECLARED["simulation_seed"],
        "inference": DECLARED["inference_seed"],
        "rank": DECLARED["rank_seed"],
    }
    assert manifest["specification_digest"] == spec_digest(TINY.spec)
    assert (
        manifest["program_digest"]
        == ResponseCompiler.installed().compile(TINY.spec).digest
    )
    assert manifest["design"] == json.loads(TINY.read().provenance.model_dump_json())
    assert manifest["git_revision"] == source_revision()
    assert isinstance(manifest["elapsed_seconds"], float)
    assert manifest["elapsed_seconds"] > 0.0


def unfloored(reduced: list[dict[str, JsonValue]]) -> list[dict[str, JsonValue]]:
    return [
        {key: value for key, value in row.items() if key not in FLOOR_FIELDS}
        for row in reduced
    ]


def test_a_campaign_missing_the_floor_writes_everything_then_exits_nonzero(
    completed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(FLOOR, 2.0)

    assert run(settings(), tmp_path, source=TINY) == 1

    manifest = recorded(tmp_path, MANIFEST_NAME)

    assert manifest["outcome"] == COMPLETED
    assert manifest["clears_floor"] is False
    assert manifest["retained_replicates"] == DECLARED["replicates"]
    assert recorded_seeds(tmp_path) == derived_seeds()
    assert unfloored(rows(tmp_path)) == unfloored(rows(completed))


def test_the_manifest_records_each_seed_the_package_derives_for_each_replicate(
    completed: Path,
) -> None:
    assert recorded_seeds(completed) == derived_seeds()


def test_the_manifest_records_each_replicate_s_diagnostics(completed: Path) -> None:
    records = replicate_records(completed)
    sites = {
        draw.site for draw in ResponseCompiler.installed().compile(TINY.spec).draws
    }

    assert [record["replicate"] for record in records] == [0, 1]

    for record in records:
        nuts = record["nuts"]

        assert nuts is not None
        assert record["retained"] is True
        assert record["failures"] == {}
        assert len(nuts["divergences"]) == DECLARED["chains"]
        assert len(nuts["acceptance_rates"]) == DECLARED["chains"]
        assert set(nuts["r_hat"]) == sites
        assert set(nuts["effective_sample_sizes"]) == sites
        assert nuts["max_rhat"] == max(nuts["r_hat"].values())
        assert nuts["min_ess"] == min(nuts["effective_sample_sizes"].values())
        assert set(record["evidence_bounds"]) == {
            method.value for method in METHODS[1:]
        }


def test_a_directory_holding_a_campaign_is_not_written_again(completed: Path) -> None:
    before = (completed / MANIFEST_NAME).read_bytes()

    with pytest.raises(FileExistsError, match="already records a campaign"):
        run(settings(), completed, source=TINY)

    assert (completed / MANIFEST_NAME).read_bytes() == before


MESSAGE: Final = "NUTS posterior fitting failed: every retained transition diverged"


def _refused(*arguments: object, **keywords: object) -> CalibrationCampaign:
    del arguments, keywords

    return CalibrationCampaign(
        tuple(
            ReplicateRecord(
                replicate=replicate,
                seed=STREAMS.simulation(replicate),
                nuts=None,
                evidence_bounds={
                    InferenceMethod.STRUCTURED: -1.0,
                    InferenceMethod.MEAN_FIELD: -2.0,
                },
                failures={InferenceMethod.NUTS: MESSAGE},
            )
            for replicate in range(int(DECLARED["replicates"]))
        ),
        None,
    )


def test_a_campaign_retaining_no_replicate_records_every_failure_and_no_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "scripts.judgment.simulation_calibration.prior_predictive_calibration",
        _refused,
    )

    assert run(settings(), tmp_path, source=TINY) == 1

    manifest = recorded(tmp_path, MANIFEST_NAME)
    records = replicate_records(tmp_path)

    assert manifest["outcome"] == FAILED
    assert manifest["retained_replicates"] == 0
    assert manifest["failed_replicates"] == [0, 1]
    assert manifest["clears_floor"] is None
    assert recorded_seeds(tmp_path) == derived_seeds()
    assert [record["failures"] for record in records] == [{"nuts": MESSAGE}] * 2
    assert [record["nuts"] for record in records] == [None, None]
    assert [record["evidence_bounds"] for record in records] == [
        {"structured": -1.0, "mean_field": -2.0}
    ] * 2
    assert not (tmp_path / ROWS_NAME).exists()


@pytest.mark.parametrize(
    ("changed", "match"),
    [
        ({"replicates": 0}, "at least one replicate"),
        ({"guide_iterations": 0}, "at least one sweep"),
        ({"guide_learning_rate": 0.0}, "learning rate"),
        ({"chains": 1}, "at least two chains"),
        ({"retained": 3}, "at least four draws"),
        ({"rank_seed": -1}, "rank seed"),
    ],
    ids=[
        "replicates",
        "guide-iterations",
        "learning-rate",
        "chains",
        "retained",
        "seed",
    ],
)
def test_an_invalid_declaration_is_refused_before_anything_is_written(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    changed: dict[str, float],
    match: str,
) -> None:
    out = tmp_path / "campaign"

    with pytest.raises(SystemExit) as refused:
        main(arguments(out, **changed))

    assert refused.value.code == 2
    assert match in capsys.readouterr().err
    assert not out.exists()
