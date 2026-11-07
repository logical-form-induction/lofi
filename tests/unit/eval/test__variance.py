from statistics import NormalDist, fmean
from typing import TYPE_CHECKING, Final, cast

import pytest

from lofi.eval._values import Resampling, convention_crossing
from lofi.eval._variance import (
    VARIANCE_SOURCES,
    Crossing,
    FactorScore,
    FactorVarianceComponent,
    ReportedScore,
    UnidentifiedDesignError,
    VarianceComponent,
    VarianceDecomposition,
    _fit,
    factor_variance_decomposition,
    variance_decomposition,
)
from lofi.stats import Interval

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.eval._values import ConventionVector
    from lofi.eval._variance import VarianceSource

VECTORS: Final = convention_crossing()

# target variances for the synthetic designs
CONVENTION_VARIANCE: Final = 0.09
SEED_VARIANCE: Final = 0.04
INTERACTION_VARIANCE: Final = 0.02
RESIDUAL_SD: Final = 0.2

# per-method offsets removed before decomposition
METHOD_EFFECTS: Final = (0.0, 0.4, -0.3, 0.2, -0.3)

_KNOWN: Final[Mapping[VarianceSource, float]] = {
    "convention": CONVENTION_VARIANCE,
    "seed": SEED_VARIANCE,
    "interaction": INTERACTION_VARIANCE,
    "residual": RESIDUAL_SD**2,
}


def _levels(count: int, variance: float) -> tuple[float, ...]:
    spread = tuple(index - (count - 1) / 2 for index in range(count))
    squares = sum(value * value for value in spread)
    scale = (variance * (count - 1) / squares) ** 0.5

    return tuple(value * scale for value in spread)


def _design(
    *,
    conventions: int = 10,
    seeds: int = 10,
    methods: int = 5,
    seed: int,
    keep: frozenset[tuple[int, int]] | None = None,
) -> tuple[ReportedScore, ...]:
    convention_effect = _levels(conventions, CONVENTION_VARIANCE)
    seed_effect = _levels(seeds, SEED_VARIANCE)
    row = _levels(conventions, 1.0)
    column = _levels(seeds, 1.0)
    weight = (
        INTERACTION_VARIANCE
        * (conventions - 1)
        * (seeds - 1)
        / (sum(v * v for v in row) * sum(v * v for v in column))
    ) ** 0.5
    noise = iter(
        NormalDist(0.0, RESIDUAL_SD).samples(conventions * seeds * methods, seed=seed)
    )
    built: list[ReportedScore] = []

    for i in range(conventions):
        for j in range(seeds):
            for index in range(methods):
                drawn = next(noise)

                if keep is not None and (i, j) not in keep:
                    continue

                built.append(
                    ReportedScore(
                        method=f"method-{index}",
                        conventions=VECTORS[i],
                        seed=j,
                        score=(
                            0.5
                            + METHOD_EFFECTS[index]
                            + convention_effect[i]
                            + seed_effect[j]
                            + weight * row[i] * column[j]
                            + drawn
                        ),
                    )
                )

    return tuple(built)


def _textbook(scores: Sequence[ReportedScore]) -> dict[str, float]:
    methods = sorted({score.method for score in scores})
    conventions = sorted({score.conventions for score in scores}, key=repr)
    seeds = sorted({score.seed for score in scores})
    replicates = len(methods)
    per_method: dict[str, list[float]] = {name: [] for name in methods}

    for score in scores:
        per_method[score.method].append(score.score)

    means = {name: fmean(values) for name, values in per_method.items()}
    centered = [(score, score.score - means[score.method]) for score in scores]
    grand = fmean(value for _, value in centered)

    def _mean(chosen: Sequence[float]) -> float:
        return fmean(chosen)

    cell_means = {
        (conventions.index(score.conventions), score.seed): _mean(
            [
                value
                for other, value in centered
                if other.conventions == score.conventions and other.seed == score.seed
            ]
        )
        for score, _ in centered
    }
    convention_means = [
        _mean([value for other, value in centered if other.conventions == vector])
        for vector in conventions
    ]
    seed_means = [
        _mean([value for other, value in centered if other.seed == seed])
        for seed in seeds
    ]
    between_conventions = sum(
        len(seeds) * replicates * (mean - grand) ** 2 for mean in convention_means
    )
    between_seeds = sum(
        len(conventions) * replicates * (mean - grand) ** 2 for mean in seed_means
    )
    between_cells = sum(
        replicates * (mean - grand) ** 2 for mean in cell_means.values()
    )
    within = sum(
        (value - cell_means[(conventions.index(score.conventions), score.seed)]) ** 2
        for score, value in centered
    )
    cells = len(conventions) * len(seeds)
    mean_square_conventions = between_conventions / (len(conventions) - 1)
    mean_square_seeds = between_seeds / (len(seeds) - 1)
    mean_square_interaction = (between_cells - between_conventions - between_seeds) / (
        (len(conventions) - 1) * (len(seeds) - 1)
    )
    mean_square_residual = within / ((cells - 1) * (replicates - 1))

    return {
        "convention": (mean_square_conventions - mean_square_interaction)
        / (len(seeds) * replicates),
        "seed": (mean_square_seeds - mean_square_interaction)
        / (len(conventions) * replicates),
        "interaction": (mean_square_interaction - mean_square_residual) / replicates,
        "residual": mean_square_residual,
    }


def _fitted(
    scores: Sequence[ReportedScore], *, replicates: int = 19
) -> VarianceDecomposition:
    return variance_decomposition(scores, seed=5, replicates=replicates)


def test_a_balanced_fit_agrees_with_the_textbook_two_way_analysis() -> None:
    scores = _design(conventions=3, seeds=4, methods=3, seed=0)
    decomposition = _fitted(scores)
    expected = _textbook(scores)

    for source in VARIANCE_SOURCES:
        assert decomposition.component(source).variance == pytest.approx(
            expected[source]
        )
        assert expected[source] > 0.0


def test_five_hundred_cells_recover_the_variances_they_were_built_with() -> None:
    decomposition = _fitted(_design(seed=0), replicates=99)

    assert decomposition.crossing.scores == 500

    for source, truth in _KNOWN.items():
        component = decomposition.component(source)
        interval = component.ci

        assert component.variance == pytest.approx(truth, rel=0.2)
        assert interval is not None
        assert interval.covers(component.variance)
        assert interval.covers(truth)


def test_the_fit_carries_no_bias_across_repeated_designs() -> None:
    estimates = [_fit(_design(seed=seed)) for seed in range(24)]

    for source, truth in _KNOWN.items():
        average = fmean(estimate[source] for estimate in estimates)

        assert average == pytest.approx(truth, rel=0.1)


def test_the_four_components_sum_to_the_total_and_their_shares_to_one() -> None:
    decomposition = _fitted(_design(conventions=5, seeds=5, methods=3, seed=0))
    variances = [component.variance for component in decomposition.components]
    shares = [decomposition.share(source) for source in VARIANCE_SOURCES]

    assert len(variances) == 4
    assert sum(variances) == pytest.approx(decomposition.total)
    assert sum(share for share in shares if share is not None) == pytest.approx(1.0)
    assert [component.source for component in decomposition.components] == list(
        VARIANCE_SOURCES
    )


def test_every_component_carries_an_interval_around_its_own_estimate() -> None:
    decomposition = _fitted(
        _design(conventions=5, seeds=5, methods=3, seed=0), replicates=49
    )

    for component in decomposition.components:
        assert component.ci is not None
        assert component.ci.level == 0.95
        assert component.ci.low <= component.variance <= component.ci.high

    assert decomposition.resampling == Resampling("convention-seed cell", 5)


def test_dropping_forty_percent_of_the_cells_widens_the_intervals() -> None:
    cells = [(i, j) for i in range(10) for j in range(10)]
    keep = frozenset(cells[(7 * step) % len(cells)] for step in range(60))
    whole = _fitted(_design(seed=0), replicates=99)
    partial = _fitted(_design(seed=0, keep=keep), replicates=99)

    assert partial.crossing.cells == 60
    assert not partial.crossing.is_complete
    assert whole.crossing.is_complete

    widths = {
        source: (
            _width(whole.component(source).ci),
            _width(partial.component(source).ci),
        )
        for source in VARIANCE_SOURCES
    }

    assert widths["convention"][1] > widths["convention"][0]
    assert widths["seed"][1] > widths["seed"][0]
    assert sum(pair[1] for pair in widths.values()) > sum(
        pair[0] for pair in widths.values()
    )

    for source in VARIANCE_SOURCES:
        assert partial.component(source).variance != whole.component(source).variance


def _width(interval: Interval | None) -> float:
    assert interval is not None

    return interval.width


def test_the_same_seed_draws_the_same_intervals() -> None:
    scores = _design(conventions=4, seeds=4, methods=3, seed=2)

    assert _fitted(scores).components == _fitted(scores).components


def test_a_sweep_over_one_convention_or_one_seed_is_refused() -> None:
    one_convention = [
        ReportedScore(
            method=f"m-{m}", conventions=VECTORS[0], seed=j, score=0.1 * m + j
        )
        for j in range(4)
        for m in range(3)
    ]
    one_seed = [
        ReportedScore(
            method=f"m-{m}", conventions=VECTORS[i], seed=0, score=0.1 * m + i
        )
        for i in range(4)
        for m in range(3)
    ]

    with pytest.raises(UnidentifiedDesignError, match="cannot separate the two"):
        _fitted(one_convention)

    with pytest.raises(UnidentifiedDesignError, match="cannot separate the two"):
        _fitted(one_seed)


def test_one_number_per_cell_leaves_the_interaction_and_the_residual_as_one() -> None:
    alone = [
        ReportedScore(
            method="ccg", conventions=VECTORS[i], seed=j, score=0.3 * i + 0.1 * j
        )
        for i in range(4)
        for j in range(4)
    ]

    with pytest.raises(UnidentifiedDesignError, match="leave nothing inside a cell"):
        _fitted(alone)


def test_two_numbers_for_one_method_convention_and_seed_are_refused() -> None:
    scores = _design(conventions=3, seeds=3, methods=3, seed=0)

    with pytest.raises(ValueError, match="reports two numbers"):
        _fitted([*scores, scores[0]])


def test_a_table_holding_something_that_is_not_a_reported_score_is_refused() -> None:
    with pytest.raises(TypeError, match="not a reported score"):
        _fitted([cast("ReportedScore", 0.4)])


def test_a_table_with_no_spread_at_all_apportions_nothing() -> None:
    flat = tuple(
        ReportedScore(method=f"m-{m}", conventions=VECTORS[i], seed=j, score=0.7)
        for i in range(3)
        for j in range(3)
        for m in range(3)
    )
    decomposition = _fitted(flat)

    assert decomposition.total == 0.0
    assert decomposition.share("convention") is None
    assert all(component.variance == 0.0 for component in decomposition.components)


def test_a_reported_score_is_equal_on_its_four_fields() -> None:
    one = ReportedScore(method="ccg", conventions=VECTORS[0], seed=3, score=0.71)
    same = ReportedScore(method="ccg", conventions=VECTORS[0], seed=3, score=0.71)
    other = ReportedScore(method="ccg", conventions=VECTORS[1], seed=3, score=0.71)

    assert one == same
    assert hash(one) == hash(same)
    assert one != other
    assert one.cell == (VECTORS[0], 3)
    assert one.method == "ccg"
    assert one.score == 0.71

    for field in ("method", "conventions", "seed", "score"):
        with pytest.raises(AttributeError):
            setattr(one, field, None)


def test_a_reported_score_refuses_a_convention_that_is_not_a_vector() -> None:
    with pytest.raises(TypeError, match="not a vector"):
        ReportedScore(
            method="ccg",
            conventions=cast("ConventionVector", "strip"),
            seed=0,
            score=0.5,
        )

    with pytest.raises(ValueError, match="is negative"):
        ReportedScore(method="ccg", conventions=VECTORS[0], seed=-1, score=0.5)

    with pytest.raises(ValueError, match="is empty"):
        ReportedScore(method="", conventions=VECTORS[0], seed=0, score=0.5)


def test_a_crossing_reports_whether_every_convention_met_every_seed() -> None:
    whole = Crossing(scores=500, methods=5, conventions=10, seeds=10, cells=100)
    partial = Crossing(scores=300, methods=5, conventions=10, seeds=10, cells=60)

    assert whole.is_complete
    assert not partial.is_complete
    assert whole != partial

    with pytest.raises(ValueError, match="cell holds no score"):
        Crossing(scores=4, methods=1, conventions=2, seeds=2, cells=5)


def test_a_component_refuses_a_source_nobody_declared_and_a_negative_variance() -> None:
    with pytest.raises(TypeError, match="not a declared one"):
        VarianceComponent(cast("VarianceSource", "method"), 0.1, ci=None)

    with pytest.raises(ValueError, match="is negative"):
        VarianceComponent("seed", -0.1, ci=None)

    with pytest.raises(TypeError, match="interval is not one"):
        VarianceComponent("seed", 0.1, ci=cast("Interval", (0.0, 1.0)))


def test_a_decomposition_names_every_source_exactly_once() -> None:
    crossing = Crossing(scores=8, methods=2, conventions=2, seeds=2, cells=4)
    scheme = Resampling("convention-seed cell", 0)
    components = [
        VarianceComponent(source, 0.1, ci=None) for source in VARIANCE_SOURCES
    ]

    with pytest.raises(ValueError, match="names no variance for"):
        VarianceDecomposition(components[:3], crossing=crossing, resampling=scheme)

    with pytest.raises(ValueError, match="holds 'seed' twice"):
        VarianceDecomposition(
            [*components, components[1]], crossing=crossing, resampling=scheme
        )

    held = VarianceDecomposition(
        reversed(components), crossing=crossing, resampling=scheme
    )

    assert [component.source for component in held.components] == list(VARIANCE_SOURCES)
    assert held.share("residual") == pytest.approx(0.25)

    with pytest.raises(TypeError, match="not a declared one"):
        held.component(cast("VarianceSource", "method"))


def test_declared_factors_include_a_working_data_subsample_component() -> None:
    convention = (-0.3, 0.3)
    seed_effect = (-0.2, 0.2)
    subsample = (-0.1, 0.1)
    rows = tuple(
        FactorScore(
            method=f"method-{method}",
            levels={
                "convention": convention_index,
                "seed": seed_index,
                "data-subsample": subsample_index,
            },
            score=(
                convention[convention_index]
                + seed_effect[seed_index]
                + subsample[subsample_index]
                + (0.01 if method == 0 else -0.01)
            ),
        )
        for convention_index in range(2)
        for seed_index in range(2)
        for subsample_index in range(2)
        for method in range(2)
    )
    decomposition = factor_variance_decomposition(
        rows,
        ("convention", "seed", "data-subsample"),
        estimator="moments",
        seed=7,
        replicates=19,
        level=0.9,
    )

    assert decomposition.factors == ("convention", "seed", "data-subsample")
    assert decomposition.component("convention").variance == pytest.approx(0.18)
    assert decomposition.component("seed").variance == pytest.approx(0.08)
    assert decomposition.component("data-subsample").variance == pytest.approx(0.02)
    assert decomposition.component("interaction").variance == pytest.approx(0.0)
    assert decomposition.component("residual").variance == pytest.approx(0.0)
    assert decomposition.resampling == Resampling("method", 7)
    assert decomposition.scores == 16


def test_declared_factor_decomposition_requires_a_complete_crossing() -> None:
    rows = tuple(
        FactorScore(
            method=f"method-{method}",
            levels={"seed": seed, "data-subsample": subsample},
            score=float(seed + subsample + method),
        )
        for seed in range(2)
        for subsample in range(2)
        for method in range(2)
    )

    with pytest.raises(UnidentifiedDesignError, match="one score per method"):
        factor_variance_decomposition(
            rows[:-1],
            ("seed", "data-subsample"),
            estimator="moments",
            seed=0,
            replicates=9,
            level=0.9,
        )

    with pytest.raises(ValueError, match="at least two distinct"):
        factor_variance_decomposition(
            rows,
            ("seed",),
            estimator="moments",
            seed=0,
            replicates=9,
            level=0.9,
        )

    with pytest.raises(TypeError, match="nonscalar"):
        FactorScore(method="m", levels={"seed": cast("int", [])}, score=0.0)


def test_a_declared_factor_component_refuses_a_negative_variance() -> None:
    with pytest.raises(ValueError, match="negative"):
        FactorVarianceComponent("data-subsample", -0.1, ci=Interval(0.0, 1.0, 0.9))
