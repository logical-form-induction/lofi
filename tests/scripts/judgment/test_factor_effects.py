import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final

import didactic.api as dx
import pytest
import torch
from safetensors.torch import load_file
from scripts.judgment.factor_effects import (
    EVERY_ROW,
    HELD_OUT,
    MANIFEST_NAME,
    REAL,
    SHUFFLED,
    main,
    permutation,
    read_comparison,
    read_datasets,
    run,
    scale_site,
    shuffled,
    split,
    verdicts,
)

from lofi.judgment import (
    ConditionalLikelihood,
    Cumulative,
    DesignCodec,
    DesignProvenance,
    FixedDiscrimination,
    JudgmentRow,
    LinkingSpec,
    LinkTerm,
    PriorBlock,
    RandomIntercept,
    ReadDesign,
    ResponseCompiler,
    ResponseSpec,
    RowColumns,
    ScaleNormalization,
    ThresholdPriors,
    posterior_variance_row,
    predictive_density_row,
    spec_digest,
)
from lofi.registry import UnknownTagError

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.judgment import PosteriorVarianceRow, ReadCorpusDesign

LADDER: Final = Path(__file__).resolve().parents[3] / "tests" / "data" / "judgment"
LEVELS: Final = 4
SCORE: Final = "score"
FRAME: Final = "frame"
PARTICIPANTS: Final = 3
ITEMS: Final = 8
FRAMES: Final = 2


def ladder_rung(name: str) -> ResponseSpec:
    return ResponseSpec.model_validate_json(
        (LADDER / "ladder" / f"{name}.json").read_text(encoding="utf-8")
    )


def scored(*groups: str) -> ResponseSpec:
    return ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(
            LinkTerm(readout=SCORE, linking=LinkingSpec(family="linear")),
            *(RandomIntercept(group=group) for group in groups),
        ),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
        normalization=ScaleNormalization.IDENTITY,
    )


def rows(*, response_shift: int = 0) -> tuple[JudgmentRow, ...]:
    return tuple(
        JudgmentRow(
            f"participant-{participant}",
            f"item-{item}",
            min(LEVELS - 1, (item % FRAMES) * 2 + (participant + response_shift) % 2),
            RowColumns(
                {FRAME: f"frame-{item % FRAMES}", "verb": f"verb-{item // FRAMES}"},
                {SCORE: 0.25 * (item % 3) - 0.25},
            ),
        )
        for item in range(ITEMS)
        for participant in range(PARTICIPANTS)
    )


def provenance(tag: str) -> DesignProvenance:
    return DesignProvenance(
        tag=tag,
        checkpoint=None,
        checkpoint_dtype=None,
        unigram_digest=None,
        corpus_digests={f"{tag}.tsv": sha256(tag.encode()).hexdigest()},
        seed=None,
        participants=None,
        items=None,
        levels=None,
        population_digest=None,
        producer_run=None,
        producer_point=None,
    )


class Designs:
    def __init__(self, records: dict[str, tuple[JudgmentRow, ...]]) -> None:
        self.records = records
        self.reads: list[str] = []

    def read(self, dataset: str, /) -> ReadCorpusDesign:
        self.reads.append(dataset)
        codec = DesignCodec(LEVELS, factors=(FRAME, "verb"), readouts=(SCORE,))

        return ReadDesign(codec.encode(self.records[dataset]), provenance(dataset))


def declaration(
    rungs: tuple[tuple[str, str, ResponseSpec], ...],
    *,
    unit: str = "item",
    floor: float = 0.05,
) -> dict[str, JsonValue]:
    return {
        "rungs": [
            {
                "name": name,
                "dataset": dataset,
                "response": json.loads(response.model_dump_json()),
            }
            for name, dataset, response in rungs
        ],
        "held_out": {
            "unit": unit,
            "splitter": "kfold",
            "folds": 2,
            "seed": 11,
            "coverage_threshold": 0.0,
            "unseen_participant": "plug_in",
            "samples": 2,
        },
        "inference": {"inference": "nuts", "warmup": 20, "retained": 4, "chains": 2},
        "seed": 5,
        "control": {"factor": FRAME, "seed": 3},
        "criterion": {"level": 0.9, "floor": floor},
    }


def encoded(document: dict[str, JsonValue]) -> bytes:
    return json.dumps(document).encode()


GROUPED: Final = (
    ("score_only", "plain", scored()),
    ("score_by_frame", "grouped", scored(FRAME)),
)


def test_each_grouped_ladder_rung_scales_one_frame_intercept() -> None:
    compiler = ResponseCompiler.installed()
    sites = {
        name: scale_site(compiler.compile(ladder_rung(name)), FRAME)
        for name in ("morcela_only", "morcela_by_frame", "morcela_by_verb_by_frame")
    }
    verb = scale_site(compiler.compile(ladder_rung("morcela_by_verb_by_frame")), "verb")

    assert sites["morcela_only"] is None
    assert sites["morcela_by_frame"] == sites["morcela_by_verb_by_frame"]
    assert sites["morcela_by_frame"] is not None
    assert verb is not None
    assert verb != sites["morcela_by_frame"]


def test_the_ladder_rungs_compile_into_one_comparison_with_their_digests() -> None:
    names = ("morcela_only", "morcela_by_frame", "morcela_by_verb_by_frame")
    comparison = read_comparison(
        encoded(
            declaration(tuple((name, "corpus", ladder_rung(name)) for name in names))
        )
    )

    assert tuple(rung.name for rung in comparison.rungs) == names
    assert tuple(rung.digest for rung in comparison.rungs) == tuple(
        spec_digest(ladder_rung(name)) for name in names
    )
    assert len({rung.digest for rung in comparison.rungs}) == len(names)
    assert tuple(rung.reads_factor for rung in comparison.rungs) == (False, True, True)


def test_a_comparison_without_an_intercept_over_the_factor_is_refused() -> None:
    with pytest.raises(ValueError, match="no declared rung carries"):
        read_comparison(encoded(declaration((("score_only", "plain", scored()),))))


def test_holding_out_the_shuffled_factor_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="also the held-out unit"):
        read_comparison(encoded(declaration(GROUPED, unit=FRAME)))


def test_a_repeated_rung_is_refused() -> None:
    repeated = (GROUPED[1], ("score_by_frame", "plain", scored(FRAME)))

    with pytest.raises(dx.ValidationError, match="repeats a rung"):
        read_comparison(encoded(declaration(repeated)))


def test_a_rung_outside_the_cumulative_link_is_refused() -> None:
    metric = ("p_gram_metric", "plain", ladder_rung("p_gram_metric"))

    with pytest.raises(dx.ValidationError, match="cumulative-link"):
        read_comparison(encoded(declaration((*GROUPED, metric))))


def test_the_permutation_keeps_each_level_s_item_count_and_follows_its_seed() -> None:
    recorded = rows()
    levels = permutation(recorded, FRAME, seed=3)
    counts = Counter(level for level, _ in levels.values())
    permuted = Counter(level for _, level in levels.values())
    orders = {
        tuple(level for _, level in permutation(recorded, FRAME, seed=seed).values())
        for seed in range(10)
    }

    assert tuple(levels) == tuple(dict.fromkeys(row.item for row in recorded))
    assert counts == permuted
    assert levels == permutation(recorded, FRAME, seed=3)
    assert len(orders) > 1


def test_an_item_recorded_under_two_levels_is_refused() -> None:
    recorded = rows()
    first = recorded[0]
    moved = JudgmentRow(
        first.participant,
        recorded[1].item,
        first.response,
        RowColumns({**first.factors, FRAME: "frame-elsewhere"}, dict(first.readouts)),
    )

    with pytest.raises(ValueError, match="not an item factor"):
        permutation((*recorded, moved), FRAME, seed=0)


def test_shuffled_rows_change_only_the_factor_level_of_each_item() -> None:
    recorded = rows()
    levels = permutation(recorded, FRAME, seed=3)
    relabeled = shuffled(recorded, FRAME, levels)

    for before, after in zip(recorded, relabeled, strict=True):
        assert (after.participant, after.item, after.response) == (
            before.participant,
            before.item,
            before.response,
        )
        assert after.readouts == before.readouts
        assert after.factors["verb"] == before.factors["verb"]
        assert after.factors[FRAME] == levels[before.item][1]


def test_datasets_with_the_same_responses_share_their_folds() -> None:
    comparison = read_comparison(encoded(declaration(GROUPED)))
    designs = Designs({"plain": rows(), "grouped": rows()})
    datasets = read_datasets(comparison, designs)
    folds = split(comparison, datasets)

    assert designs.reads == ["plain", "grouped"]
    assert tuple(fold.units for fold in folds["plain"]) == tuple(
        fold.units for fold in folds["grouped"]
    )
    assert sum(len(fold.held_out) for fold in folds["plain"]) == len(rows())


def test_datasets_with_different_responses_are_refused() -> None:
    comparison = read_comparison(encoded(declaration(GROUPED)))
    designs = Designs({"plain": rows(), "grouped": rows(response_shift=1)})

    with pytest.raises(ValueError, match="do not record the same responses"):
        read_datasets(comparison, designs)


def variance(fit: str, draws: tuple[float, ...]) -> PosteriorVarianceRow:
    return posterior_variance_row(
        fit, FRAME, torch.tensor(draws, dtype=torch.float64), level=0.9
    )


def test_the_criterion_reads_each_interval_against_the_floor() -> None:
    high = (0.4, 0.5, 0.6, 0.7)
    low = (0.0, 0.01, 0.02, 0.03)
    floor = 0.1

    (met,) = verdicts((variance("a", high),), (variance("a", low),), floor=floor)
    (unmet,) = verdicts((variance("a", low),), (variance("a", high),), floor=floor)

    assert (met.bounded, met.degenerate) == (True, True)
    assert (unmet.bounded, unmet.degenerate) == (False, False)


def test_the_criterion_refuses_conditions_that_report_different_rungs() -> None:
    with pytest.raises(ValueError, match="different rungs"):
        verdicts((variance("a", (0.1, 0.2)),), (variance("b", (0.1, 0.2)),), floor=0.0)


def test_main_resolves_each_dataset_through_the_registry(tmp_path: Path) -> None:
    path = tmp_path / "declaration.json"
    path.write_bytes(encoded(declaration(GROUPED)))

    with pytest.raises(UnknownTagError):
        main([str(path), "--out", str(tmp_path / "out")])


class Ran:
    def __init__(self, out: Path, comparison_document: bytes) -> None:
        self.out = out
        self.document = comparison_document
        self.report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ran(tmp_path_factory: pytest.TempPathFactory) -> Ran:
    out = tmp_path_factory.mktemp("factor-effects")
    document = encoded(declaration(GROUPED))
    run(document, out, designs=Designs({"plain": rows(), "grouped": rows()}))

    return Ran(out, document)


def fits_of(ran: Ran, condition: str, rung: str) -> list[dict[str, JsonValue]]:
    return [
        fit
        for fit in ran.manifest["fits"]
        if (fit["condition"], fit["rung"]) == (condition, rung)
    ]


def test_the_shuffled_condition_refits_only_the_rungs_reading_the_factor(
    ran: Ran,
) -> None:
    fitted = {
        (fit["condition"], fit["rung"], fit["fold"]) for fit in ran.manifest["fits"]
    }

    assert fitted == {
        (REAL, "score_only", "0"),
        (REAL, "score_only", "1"),
        (REAL, "score_by_frame", "0"),
        (REAL, "score_by_frame", "1"),
        (REAL, "score_by_frame", EVERY_ROW),
        (SHUFFLED, "score_by_frame", "0"),
        (SHUFFLED, "score_by_frame", "1"),
        (SHUFFLED, "score_by_frame", EVERY_ROW),
    }


def test_every_fit_is_persisted_under_its_recorded_digest(ran: Ran) -> None:
    for fit in ran.manifest["fits"]:
        data = (ran.out / fit["path"]).read_bytes()

        assert sha256(data).hexdigest() == fit["sha256"]
        assert (fit["warmup"], fit["retained"], fit["chains"]) == (20, 4, 2)


def test_the_shuffled_fits_reuse_every_recorded_seed(ran: Ran) -> None:
    def seeds(condition: str) -> dict[JsonValue, tuple[JsonValue, JsonValue]]:
        return {
            fit["fold"]: (fit["nuts_seed"], fit.get("prediction_seed"))
            for fit in fits_of(ran, condition, "score_by_frame")
        }

    assert seeds(REAL) == seeds(SHUFFLED)


def test_the_manifest_records_the_declaration_rungs_folds_and_permutation(
    ran: Ran,
) -> None:
    manifest = ran.manifest
    items = manifest["permutation"]["items"]

    assert manifest["declaration_sha256"] == sha256(ran.document).hexdigest()
    assert manifest["declaration"] == json.loads(ran.document)
    assert set(manifest["datasets"]) == {"plain", "grouped"}
    assert [rung["spec_digest"] for rung in manifest["rungs"]] == [
        spec_digest(response) for _, _, response in GROUPED
    ]
    assert sum(fold["held_out"] for fold in manifest["folds"]) == len(rows())
    assert items == {
        item: {"recorded": recorded, "permuted": permuted}
        for item, (recorded, permuted) in permutation(rows(), FRAME, seed=3).items()
    }


def test_each_density_is_reduced_from_the_persisted_held_out_rows(ran: Ran) -> None:
    for condition, names in (
        (REAL, ("score_only", "score_by_frame")),
        (SHUFFLED, ("score_by_frame",)),
    ):
        reported = ran.report[condition]["predictive_density"]

        assert [row["fit"] for row in reported] == list(names)

        for row, name in zip(reported, names, strict=True):
            held_out = torch.cat(
                [
                    load_file(ran.out / str(fit["path"]))[HELD_OUT]
                    for fit in fits_of(ran, condition, name)
                    if fit["fold"] != EVERY_ROW
                ]
            )
            expected = predictive_density_row(name, held_out)

            assert row == json.loads(expected.model_dump_json())
            assert row["predictive_density"]["n"] == len(rows())


def test_each_variance_is_reduced_from_the_persisted_scale_draws(ran: Ran) -> None:
    (rung,) = (rung for rung in ran.manifest["rungs"] if rung["scale"] is not None)

    for condition in (REAL, SHUFFLED):
        (every,) = (
            fit
            for fit in fits_of(ran, condition, rung["name"])
            if fit["fold"] == EVERY_ROW
        )
        scale = load_file(ran.out / str(every["path"]))[f"draws.{rung['scale']}"]
        expected = posterior_variance_row(
            rung["name"], FRAME, scale.reshape(-1).square(), level=0.9
        )
        (row,) = ran.report[condition]["posterior_variance"]

        assert row == json.loads(expected.model_dump_json())
        assert row["posterior_variance"]["n"] == 2 * 4


def test_the_verdict_is_the_declared_criterion_applied_to_the_reported_rows(
    ran: Ran,
) -> None:
    report = ran.report
    floor = report["criterion"]["floor"]
    (real,) = report[REAL]["posterior_variance"]
    (control,) = report[SHUFFLED]["posterior_variance"]
    (verdict,) = report["verdicts"]

    assert verdict == {
        "rung": "score_by_frame",
        "bounded": real["posterior_variance"]["lower"] > floor,
        "degenerate": control["posterior_variance"]["upper"] <= floor,
    }
    assert report["criterion_met"] == (verdict["bounded"] and verdict["degenerate"])
