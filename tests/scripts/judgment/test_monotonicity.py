import json
import math
import random
from hashlib import sha256
from typing import TYPE_CHECKING, Final, NamedTuple

import didactic.api as dx
import pyarrow.parquet as pq
import pytest
import torch
from scripts.judgment.monotonicity import (
    FREE_SIGNED_SLOPE,
    HELD_OUT_R_SQUARED,
    IMPROVEMENT_STREAM,
    LATEX_NAME,
    MANIFEST_NAME,
    MARKDOWN_NAME,
    PERMUTATION_P_VALUE,
    PERMUTATION_STREAM,
    PREDICTIVE_DENSITY,
    PREDICTIVE_DENSITY_IMPROVEMENT,
    REPORT_NAME,
    CompletedSeed,
    RootFeatures,
    main,
    read_document,
    residual_statistic,
    residual_structure,
    root_features,
    slope_summary,
)

from lofi.eval import PooledJudgment
from lofi.judgment import PosteriorDraws, predictive_density_row
from lofi.run.reporting import DuplicateReportRowError
from lofi.spec import derive_seed
from lofi.stats import CrossedBootstrap, LabelPermutation

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from lofi._types import JsonValue

TYPES: Final = ("S[dcl]", "S[q]", "NP", "S[b]")
TARGETS: Final = frozenset({"S[dcl]"})
LEVELS: Final = ("S[q]", "NP", "S[b]")
FOLDS: Final = 3
ITEMS: Final = 24
REPLICATES: Final = 49
SEED_ROOT: Final = 11
RESPONSES: Final = 10
PARTICIPANTS: Final = 4
ITEMS_PER_RESPONSE_CYCLE: Final = 5
BOOTSTRAP_REPLICATES: Final = 99


class Item(NamedTuple):
    name: str
    log_weights: tuple[float, ...]
    derivation_count: int
    fold: int

    @property
    def features(self) -> RootFeatures:
        return root_features(self.log_weights, TYPES, TARGETS, self.derivation_count)


def synthetic_items(seed: int = 3) -> tuple[Item, ...]:
    rng = random.Random(seed)

    return tuple(
        Item(
            f"item-{index}",
            tuple(rng.gauss(0.0, 1.5) for _ in TYPES),
            rng.randint(1, 40),
            index % FOLDS,
        )
        for index in range(ITEMS)
    )


def entropy_effect(item: Item, rng: random.Random) -> float:
    del rng
    return 2.0 * item.features.entropy - 1.0


def constant(item: Item, rng: random.Random) -> float:
    del item, rng
    return 0.5


def scalar_null(item: Item, rng: random.Random) -> float:
    del item
    return rng.gauss(0.0, 1.0)


def completed(
    seed: int,
    items: tuple[Item, ...],
    residual: Callable[[Item, random.Random], float],
    *,
    slope: float = 0.5,
    folds: int = FOLDS,
) -> dict[str, JsonValue]:
    rng = random.Random(seed)

    return {
        "outcome": "completed",
        "seed": seed,
        "residuals": [
            {"item": item.name, "fold": item.fold, "residual": residual(item, rng)}
            for item in items
        ],
        "folds": [
            {
                "fold": fold,
                "one_scalar_log_densities": [
                    math.log(rng.uniform(0.05, 0.95)) for _ in range(RESPONSES)
                ],
                "two_scalar_log_densities": [
                    math.log(rng.uniform(0.05, 0.95)) for _ in range(RESPONSES)
                ],
                "free_slope_log_densities": [
                    math.log(rng.uniform(0.05, 0.95)) for _ in range(RESPONSES)
                ],
                "slope_draws": [rng.gauss(slope, 0.1) for _ in range(40)],
                "response_participants": [
                    f"participant-{response % PARTICIPANTS}"
                    for response in range(RESPONSES)
                ],
                "response_items": [
                    held_out[response % ITEMS_PER_RESPONSE_CYCLE]
                    for response in range(RESPONSES)
                ],
            }
            for fold in range(folds)
            for held_out in [[item.name for item in items if item.fold == fold]]
        ],
    }


def declaration(**changes: JsonValue) -> dict[str, JsonValue]:
    declared: dict[str, JsonValue] = {
        "freeze_id": "freeze/3",
        "grammar_fingerprint": "grammar:synthetic",
        "suite": "judgment-models",
        "task": "monotone-scalar",
        "dataset": "synthetic-acceptability",
        "split": "test",
        "normal_form": None,
        "held_out_convention": "marginal",
        "one_scalar": {"name": "p_gram_only", "specification_hash": "spec-one"},
        "two_scalar": {"name": "p_gram_entropy", "specification_hash": "spec-two"},
        "free_slope": {"name": "p_gram_signed", "specification_hash": "spec-free"},
        "improvement_baseline": "p_gram_signed",
        "root_types": [*TYPES],
        "target_types": [*sorted(TARGETS)],
        "derivation_count_scale": "log",
        "level": 0.9,
        "replicates": REPLICATES,
        "improvement": {
            "procedure": "two-way-cluster-bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "level": 0.9,
            "interval": "percentile",
        },
        "seed_root": SEED_ROOT,
    }
    declared.update(changes)

    return declared


def document(
    items: tuple[Item, ...],
    seeds: list[JsonValue],
    **changes: JsonValue,
) -> dict[str, JsonValue]:
    return {
        "declaration": declaration(**changes),
        "items": [
            {
                "item": item.name,
                "log_weights": list(item.log_weights),
                "derivation_count": item.derivation_count,
            }
            for item in items
        ],
        "seeds": seeds,
    }


def written(root: Path, content: dict[str, JsonValue]) -> Path:
    path = root / "analysis.json"
    path.write_text(json.dumps(content), encoding="utf-8", newline="\n")

    return path


def only_completed(path: Path) -> CompletedSeed:
    [seed] = read_document(path).seeds
    assert type(seed) is CompletedSeed

    return seed


def test_a_uniform_distribution_has_the_entropy_of_its_inventory() -> None:
    features = root_features((0.25, 0.25, 0.25, 0.25), TYPES, TARGETS, 7)

    assert features.entropy == pytest.approx(math.log(len(TYPES)))
    assert features.derivation_count == 7


def test_the_non_target_identity_ignores_targets_and_breaks_ties_in_order() -> None:
    heavy_target = root_features((5.0, 1.0, 2.0, 2.0), TYPES, TARGETS, 1)
    shifted = root_features((6.0, 2.0, 3.0, 3.0), TYPES, TARGETS, 1)

    assert heavy_target.non_target == "NP"
    assert shifted == heavy_target


def test_the_pooled_statistic_is_the_out_of_fold_r_squared(tmp_path: Path) -> None:
    rng = random.Random(5)
    items = tuple(
        Item(f"item-{index}", (0.0, rng.uniform(-2.0, 2.0), -9.0, -9.0), 4, index % 2)
        for index in range(12)
    )
    path = written(
        tmp_path, document(items, [completed(1, items, scalar_null, folds=2)])
    )
    seed = only_completed(path)
    entropies = [feature.entropy for feature in seed.features]
    error = 0.0
    baseline = 0.0

    for fold in (0, 1):
        train = [
            (x, y)
            for x, y, held in zip(
                entropies, seed.residuals, seed.held_out_folds, strict=True
            )
            if held != fold
        ]
        test = [
            (x, y)
            for x, y, held in zip(
                entropies, seed.residuals, seed.held_out_folds, strict=True
            )
            if held == fold
        ]
        mean_x = sum(x for x, _ in train) / len(train)
        mean_y = sum(y for _, y in train) / len(train)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in train) / sum(
            (x - mean_x) ** 2 for x, _ in train
        )
        error += sum((y - mean_y - slope * (x - mean_x)) ** 2 for x, y in test)
        baseline += sum((y - mean_y) ** 2 for _, y in test)

    statistic = residual_statistic(seed, ("S[q]",), "log")

    assert {feature.non_target for feature in seed.features} == {"S[q]"}
    assert statistic(seed.features) == pytest.approx(1.0 - error / baseline)


def test_an_entropy_effect_separates_from_the_joint_permutation_null(
    tmp_path: Path,
) -> None:
    items = synthetic_items()
    effect = only_completed(
        written(tmp_path, document(items, [completed(1, items, entropy_effect)]))
    )
    null = only_completed(
        written(tmp_path, document(items, [completed(1, items, scalar_null)]))
    )
    permutation: LabelPermutation[RootFeatures] = LabelPermutation(
        replicates=REPLICATES, seed=SEED_ROOT
    )
    effect_r_squared, effect_p = residual_structure(effect, LEVELS, "log", permutation)
    null_r_squared, null_p = residual_structure(null, LEVELS, "log", permutation)

    assert effect_r_squared == pytest.approx(1.0)
    assert effect_p.value == pytest.approx(1.0 / (REPLICATES + 1))
    assert null_r_squared < effect_r_squared
    assert null_p.value > effect_p.value


def test_the_permutation_is_reproducible_and_leaves_global_state_alone(
    tmp_path: Path,
) -> None:
    items = synthetic_items()
    seed = only_completed(
        written(tmp_path, document(items, [completed(1, items, scalar_null)]))
    )
    random.seed(17)
    torch.manual_seed(17)
    python_state = random.getstate()
    torch_state = torch.get_rng_state()

    def run() -> float:
        permutation: LabelPermutation[RootFeatures] = LabelPermutation(
            replicates=REPLICATES, seed=SEED_ROOT
        )
        return residual_structure(seed, LEVELS, "log", permutation)[1].value

    first = run()

    assert run() == first
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_the_count_scale_is_the_declared_one(tmp_path: Path) -> None:
    items = synthetic_items()
    seed = only_completed(
        written(tmp_path, document(items, [completed(1, items, scalar_null)]))
    )

    assert residual_statistic(seed, LEVELS, "log")(seed.features) != pytest.approx(
        residual_statistic(seed, LEVELS, "identity")(seed.features)
    )


def test_residuals_constant_within_their_folds_are_refused(tmp_path: Path) -> None:
    items = synthetic_items()
    seed = only_completed(
        written(
            tmp_path,
            document(items, [completed(1, items, constant)]),
        )
    )

    with pytest.raises(ValueError, match="do not vary"):
        residual_statistic(seed, LEVELS, "log")


@pytest.mark.parametrize(
    ("center", "sign"),
    [(2.0, "positive"), (-2.0, "negative"), (0.0, "unresolved")],
)
def test_the_slope_interval_establishes_a_sign_only_when_it_excludes_zero(
    center: float, sign: str
) -> None:
    rng = random.Random(9)
    draws = tuple(rng.gauss(center, 0.5) for _ in range(200))
    measurement, established = slope_summary(draws, 0.9)
    intervals = PosteriorDraws(torch.tensor(draws).unsqueeze(1)).intervals(0.9)

    assert established == sign
    assert measurement.lower == pytest.approx(float(intervals.lower[0]))
    assert measurement.upper == pytest.approx(float(intervals.upper[0]))
    assert measurement.lower is not None
    assert measurement.upper is not None
    assert measurement.value is not None
    assert measurement.lower <= measurement.value <= measurement.upper
    assert measurement.n == len(draws)


def duplicate_item(content: dict[str, JsonValue]) -> None:
    items = content["items"]
    assert isinstance(items, list)
    items.append(items[0])


def short_weights(content: dict[str, JsonValue]) -> None:
    items = content["items"]
    assert isinstance(items, list)
    first = items[0]
    assert isinstance(first, dict)
    first["log_weights"] = [0.0]


def nonpositive_count(content: dict[str, JsonValue]) -> None:
    items = content["items"]
    assert isinstance(items, list)
    first = items[0]
    assert isinstance(first, dict)
    first["derivation_count"] = 0


def missing_residual(content: dict[str, JsonValue]) -> None:
    seeds = content["seeds"]
    assert isinstance(seeds, list)
    seed = seeds[0]
    assert isinstance(seed, dict)
    residuals = seed["residuals"]
    assert isinstance(residuals, list)
    residuals.pop()


def single_fold(content: dict[str, JsonValue]) -> None:
    seeds = content["seeds"]
    assert isinstance(seeds, list)
    seed = seeds[0]
    assert isinstance(seed, dict)
    residuals = seed["residuals"]
    folds = seed["folds"]
    assert isinstance(residuals, list)
    assert isinstance(folds, list)

    for residual in residuals:
        assert isinstance(residual, dict)
        residual["fold"] = 0

    del folds[1:]


def unpaired(field: str) -> Callable[[dict[str, JsonValue]], None]:
    def change(content: dict[str, JsonValue]) -> None:
        seeds = content["seeds"]
        assert isinstance(seeds, list)
        seed = seeds[0]
        assert isinstance(seed, dict)
        folds = seed["folds"]
        assert isinstance(folds, list)
        fold = folds[0]
        assert isinstance(fold, dict)
        densities = fold[field]
        assert isinstance(densities, list)
        densities.pop()

    return change


def unaligned(field: str) -> Callable[[dict[str, JsonValue]], None]:
    def change(content: dict[str, JsonValue]) -> None:
        seeds = content["seeds"]
        assert isinstance(seeds, list)
        seed = seeds[0]
        assert isinstance(seed, dict)
        folds = seed["folds"]
        assert isinstance(folds, list)
        fold = folds[0]
        assert isinstance(fold, dict)
        units = fold[field]
        assert isinstance(units, list)
        units.pop()

    return change


def unlabeled_item(content: dict[str, JsonValue]) -> None:
    seeds = content["seeds"]
    assert isinstance(seeds, list)
    seed = seeds[0]
    assert isinstance(seed, dict)
    folds = seed["folds"]
    assert isinstance(folds, list)
    fold = folds[0]
    assert isinstance(fold, dict)
    labels = fold["response_items"]
    assert isinstance(labels, list)
    labels[0] = ""


def undeclared_improvement(content: dict[str, JsonValue]) -> None:
    held = content["declaration"]
    assert isinstance(held, dict)
    del held["improvement"]


def undeclared_baseline(content: dict[str, JsonValue]) -> None:
    held = content["declaration"]
    assert isinstance(held, dict)
    del held["improvement_baseline"]


def repeated_seed(content: dict[str, JsonValue]) -> None:
    seeds = content["seeds"]
    assert isinstance(seeds, list)
    seeds.append({"outcome": "failed", "seed": 1, "reason": "diverged"})


def declared(**changes: JsonValue) -> Callable[[dict[str, JsonValue]], None]:
    def change(content: dict[str, JsonValue]) -> None:
        held = content["declaration"]
        assert isinstance(held, dict)
        held.update(changes)

    return change


@pytest.mark.parametrize(
    "malformation",
    [
        duplicate_item,
        short_weights,
        nonpositive_count,
        missing_residual,
        single_fold,
        unpaired("two_scalar_log_densities"),
        unpaired("free_slope_log_densities"),
        unaligned("response_participants"),
        unaligned("response_items"),
        unlabeled_item,
        undeclared_improvement,
        undeclared_baseline,
        repeated_seed,
        declared(target_types=["VP"]),
        declared(target_types=list(TYPES)),
        declared(root_types=["S[dcl]", "S[dcl]", "NP", "S[b]"]),
        declared(level=1.0),
        declared(improvement_baseline="p_gram_entropy"),
        declared(improvement_baseline="p_gram_free"),
        declared(replicates=0),
        declared(derivation_count_scale="sqrt"),
        declared(seed_root=-1),
        declared(
            improvement={
                "procedure": "two-way-cluster-bootstrap",
                "replicates": BOOTSTRAP_REPLICATES,
                "level": 1.5,
                "interval": "percentile",
            }
        ),
        declared(
            improvement={
                "procedure": "bootstrap",
                "unit": "participant",
                "replicates": BOOTSTRAP_REPLICATES,
                "level": 0.9,
                "interval": "percentile",
            }
        ),
        declared(
            improvement={
                "procedure": "sign-flip",
                "replicates": BOOTSTRAP_REPLICATES,
            }
        ),
        declared(conventions={"punctuation": "keep"}),
    ],
)
def test_a_malformed_document_is_refused_at_its_boundary(
    tmp_path: Path, malformation: Callable[[dict[str, JsonValue]], None]
) -> None:
    items = synthetic_items()
    content = document(items, [completed(1, items, scalar_null)])
    malformation(content)

    with pytest.raises(dx.ValidationError):
        read_document(written(tmp_path, content))


def test_the_report_tables_and_manifest_come_from_one_run(tmp_path: Path) -> None:
    items = synthetic_items()
    content = document(
        items,
        [
            completed(1, items, entropy_effect, slope=0.8),
            {"outcome": "failed", "seed": 2, "reason": "divergent transitions"},
            completed(3, items, scalar_null, slope=0.0),
        ],
    )
    path = written(tmp_path, content)
    out = tmp_path / "out"

    assert main([str(path), "--out", str(out)]) == 0

    stored = pq.read_table(out / REPORT_NAME).to_pylist()
    analysis = read_document(path)
    completed_seeds = [seed for seed in analysis.seeds if type(seed) is CompletedSeed]
    per_seed = FOLDS * 4 + 3

    assert len(stored) == per_seed * len(completed_seeds)
    assert {row["seed"] for row in stored} == {1, 3}
    assert all(row["convention_hash"] is None for row in stored)

    for seed, offset in zip(
        completed_seeds, range(0, len(stored), per_seed), strict=True
    ):
        rows = stored[offset : offset + per_seed]

        for fold in seed.folds:
            one, two, free, slope = rows[4 * fold.fold : 4 * fold.fold + 4]

            assert (one["rung"], one["metric"], one["fold"]) == (
                analysis.one_scalar.name,
                PREDICTIVE_DENSITY,
                fold.fold,
            )
            assert (
                one["value"]
                == predictive_density_row(
                    "one", torch.tensor(fold.one_scalar, dtype=torch.float64)
                ).predictive_density.value
            )
            assert (two["rung"], two["specification_hash"]) == (
                analysis.two_scalar.name,
                analysis.two_scalar.specification_hash,
            )
            assert (
                two["value"]
                == predictive_density_row(
                    "two", torch.tensor(fold.two_scalar, dtype=torch.float64)
                ).predictive_density.value
            )
            assert (free["rung"], free["metric"], free["specification_hash"]) == (
                analysis.free_slope.name,
                PREDICTIVE_DENSITY,
                analysis.free_slope.specification_hash,
            )
            assert (
                free["value"]
                == predictive_density_row(
                    "free", torch.tensor(fold.free_slope, dtype=torch.float64)
                ).predictive_density.value
            )
            summary, _ = slope_summary(fold.slope, analysis.level)

            assert (slope["rung"], slope["metric"]) == (
                analysis.free_slope.name,
                FREE_SIGNED_SLOPE,
            )
            assert (slope["value"], slope["ci_low"], slope["ci_high"]) == (
                summary.value,
                summary.lower,
                summary.upper,
            )
            assert slope["ci_level"] == analysis.level

        r_squared, p_value, improvement = rows[-3:]
        permutation_seed = derive_seed(SEED_ROOT, f"{PERMUTATION_STREAM}:{seed.seed}")
        permutation: LabelPermutation[RootFeatures] = LabelPermutation(
            replicates=REPLICATES, seed=permutation_seed
        )
        expected_r_squared, expected_p = residual_structure(
            seed, LEVELS, "log", permutation
        )
        folds = tuple(fold.fold for fold in seed.folds)
        residual_key = PooledJudgment(
            rungs=(analysis.one_scalar.name,),
            specification_hashes=(analysis.one_scalar.specification_hash,),
            folds=folds,
        )

        for pooled in (r_squared, p_value):
            assert (
                pooled["specification_hash"],
                pooled["fold"],
                pooled["rung"],
                pooled["held_out_convention"],
            ) == (
                residual_key.specification_hash,
                None,
                analysis.one_scalar.name,
                "marginal",
            )

        assert (r_squared["metric"], r_squared["method"]) == (
            HELD_OUT_R_SQUARED,
            analysis.one_scalar.name,
        )
        assert r_squared["value"] == expected_r_squared
        assert r_squared["n"] == ITEMS
        assert p_value["metric"] == PERMUTATION_P_VALUE
        assert p_value["value"] == expected_p.value
        assert p_value["resampling_seed"] == permutation_seed
        assert p_value["ci_low"] is None

        bootstrap_seed = derive_seed(SEED_ROOT, f"{IMPROVEMENT_STREAM}:{seed.seed}")
        differences = [
            two - free
            for fold in seed.folds
            for free, two in zip(fold.free_slope, fold.two_scalar, strict=True)
        ]
        participants = [unit for fold in seed.folds for unit in fold.participants]
        responded = [unit for fold in seed.folds for unit in fold.items]
        expected_improvement = CrossedBootstrap(
            units=("participant", "item"),
            rows=participants,
            columns=responded,
            replicates=BOOTSTRAP_REPLICATES,
            seed=bootstrap_seed,
            level=0.9,
        ).interval(differences, lambda sample: math.fsum(sample) / len(sample))
        cells = len(set(zip(participants, responded, strict=True)))
        improvement_key = PooledJudgment(
            rungs=(analysis.two_scalar.name, analysis.free_slope.name),
            specification_hashes=(
                analysis.two_scalar.specification_hash,
                analysis.free_slope.specification_hash,
            ),
            folds=folds,
        )

        assert (
            improvement["metric"],
            improvement["method"],
            improvement["rung"],
            improvement["specification_hash"],
            improvement["fold"],
        ) == (
            PREDICTIVE_DENSITY_IMPROVEMENT,
            analysis.two_scalar.name,
            analysis.two_scalar.name,
            improvement_key.specification_hash,
            None,
        )
        assert improvement["value"] == expected_improvement.observed
        assert (
            improvement["ci_low"],
            improvement["ci_high"],
            improvement["ci_level"],
        ) == (
            expected_improvement.percentile.low,
            expected_improvement.percentile.high,
            0.9,
        )
        assert (improvement["n"], improvement["resampling_unit"]) == (
            cells,
            "participant-by-item",
        )
        assert improvement["resampling_seed"] == bootstrap_seed

    markdown = (out / MARKDOWN_NAME).read_text(encoding="utf-8")
    latex = (out / LATEX_NAME).read_text(encoding="utf-8")

    assert len(markdown.splitlines()) == 2 + len(stored)
    assert latex.count(r"\\") == 1 + len(stored)

    for row in stored:
        assert str(row["value"]) in markdown

    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    by_seed = {entry["seed"]: entry for entry in manifest["seeds"]}

    assert manifest["source_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert manifest["config_hash"] == analysis.identity.config_hash
    assert manifest["rows"] == len(stored)
    assert by_seed[2] == {
        "seed": 2,
        "outcome": "failed",
        "reason": "divergent transitions",
    }
    assert [entry["sign"] for entry in by_seed[1]["slope_signs"]] == [
        "positive"
    ] * FOLDS
    assert by_seed[3]["permutation"]["replicates"] == REPLICATES
    assert by_seed[3]["permutation"]["seed"] == derive_seed(
        SEED_ROOT, f"{PERMUTATION_STREAM}:3"
    )
    assert by_seed[1]["permutation"]["seed"] != by_seed[3]["permutation"]["seed"]
    assert by_seed[3]["improvement"]["seed"] == derive_seed(
        SEED_ROOT, f"{IMPROVEMENT_STREAM}:3"
    )
    assert by_seed[3]["improvement"]["procedure"] == "two-way-cluster-bootstrap"
    assert by_seed[3]["improvement"]["units"] == ["participant", "item"]
    assert by_seed[3]["improvement"]["participants"] == PARTICIPANTS
    assert by_seed[3]["improvement"]["items"] == FOLDS * ITEMS_PER_RESPONSE_CYCLE
    assert by_seed[3]["improvement"]["cells"] == FOLDS * RESPONSES
    assert {tuple(entry["folds"]) for entry in by_seed[3]["pooled"].values()} == {
        tuple(range(FOLDS))
    }
    assert {
        tuple(rung["rung"] for rung in entry["rungs"])
        for entry in by_seed[3]["pooled"].values()
    } == {
        (analysis.one_scalar.name,),
        (analysis.two_scalar.name, analysis.free_slope.name),
    }
    assert {row["specification_hash"] for row in stored if row["fold"] is None} == set(
        by_seed[1]["pooled"]
    ) | set(by_seed[3]["pooled"])


@pytest.mark.parametrize("baseline", ["p_gram_only", "p_gram_signed"])
def test_the_improvement_is_measured_against_the_declared_baseline(
    tmp_path: Path, baseline: str
) -> None:
    items = synthetic_items()
    path = written(
        tmp_path,
        document(
            items,
            [completed(1, items, scalar_null)],
            improvement_baseline=baseline,
        ),
    )
    out = tmp_path / "out"

    assert main([str(path), "--out", str(out)]) == 0

    analysis = read_document(path)
    [seed] = analysis.seeds
    assert type(seed) is CompletedSeed
    rungs = {
        rung.name: (rung, densities)
        for rung, densities in (
            (analysis.one_scalar, [fold.one_scalar for fold in seed.folds]),
            (analysis.free_slope, [fold.free_slope for fold in seed.folds]),
        )
    }
    reference, densities = rungs[baseline]
    differences = [
        two - base
        for fold, scores in zip(seed.folds, densities, strict=True)
        for base, two in zip(scores, fold.two_scalar, strict=True)
    ]
    [improvement] = [
        row
        for row in pq.read_table(out / REPORT_NAME).to_pylist()
        if row["metric"] == PREDICTIVE_DENSITY_IMPROVEMENT
    ]
    key = PooledJudgment(
        rungs=(analysis.two_scalar.name, reference.name),
        specification_hashes=(
            analysis.two_scalar.specification_hash,
            reference.specification_hash,
        ),
        folds=tuple(fold.fold for fold in seed.folds),
    )

    assert improvement["value"] == pytest.approx(
        math.fsum(differences) / len(differences), rel=1e-12
    )
    assert improvement["specification_hash"] == key.specification_hash


def test_a_second_run_into_the_same_report_is_refused(tmp_path: Path) -> None:
    items = synthetic_items()
    path = written(tmp_path, document(items, [completed(1, items, scalar_null)]))
    out = tmp_path / "out"

    assert main([str(path), "--out", str(out)]) == 0

    with pytest.raises(DuplicateReportRowError):
        main([str(path), "--out", str(out)])


def test_a_run_whose_every_seed_failed_reports_no_rows(tmp_path: Path) -> None:
    items = synthetic_items()
    path = written(
        tmp_path,
        document(items, [{"outcome": "failed", "seed": 4, "reason": "timed out"}]),
    )
    out = tmp_path / "out"

    assert main([str(path), "--out", str(out)]) == 0
    assert not (out / REPORT_NAME).exists()
    assert len((out / MARKDOWN_NAME).read_text(encoding="utf-8").splitlines()) == 2
    assert json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))["seeds"] == [
        {"seed": 4, "outcome": "failed", "reason": "timed out"}
    ]
