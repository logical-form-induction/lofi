import json
import subprocess
import sys
from math import log, nextafter, pi
from typing import TYPE_CHECKING, cast

import pytest
import torch
from quivers.diagnostics import to_datatree_from_svi

from lofi.eval.elpd import (
    WEIGHTING_METHODS,
    ModelComparison,
    ModelRow,
    PairwiseDelta,
    ParetoGate,
    _column,
    compare_declared_models,
    compare_models,
)
from lofi.judgment import (
    ModelComparisonCriterion,
    ModelComparisonSpec,
    ModelWeighting,
    log_likelihood,
)
from lofi.judgment.synthetic_designs import KNOWN_RANDOM_EFFECTS

if TYPE_CHECKING:
    from xarray import DataTree

    from lofi.eval.elpd import WeightingMethod


def _normal_log_likelihood(observed: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    """Return unit-variance normal log-likelihoods, one row per posterior draw."""
    return -0.5 * (observed[None, :] - means[:, None]) ** 2 - 0.5 * log(2 * pi)


def _fit(means: torch.Tensor, log_likelihood: torch.Tensor) -> DataTree:
    return to_datatree_from_svi({"mu": means}, log_likelihood={"y": log_likelihood})


@pytest.fixture(scope="module")
def observed() -> torch.Tensor:
    """Return sixty draws from a unit-variance normal with mean two."""
    generator = torch.Generator().manual_seed(0)

    return 2.0 + torch.randn(60, generator=generator)


def _posterior_means(observed: torch.Tensor, center: float, seed: int) -> torch.Tensor:
    """Return four hundred draws of a mean, spread as a posterior over the sample."""
    generator = torch.Generator().manual_seed(seed)

    return center + torch.randn(400, generator=generator) / len(observed) ** 0.5


@pytest.fixture(scope="module")
def fits(observed: torch.Tensor) -> dict[str, DataTree]:
    """Return three fits of one sample: the right mean, a wrong mean, and a heavy tail.

    The full model's posterior sits on the sample mean, the null model's on zero,
    and the heavy model subtracts a cubed normal from every log-likelihood so its
    importance weights have no finite variance.
    """
    full_means = _posterior_means(observed, float(observed.mean()), seed=1)
    null_means = _posterior_means(observed, 0.0, seed=2)
    generator = torch.Generator().manual_seed(3)
    tail = torch.randn(400, len(observed), generator=generator).abs() ** 3

    return {
        "full": _fit(full_means, _normal_log_likelihood(observed, full_means)),
        "null": _fit(null_means, _normal_log_likelihood(observed, null_means)),
        "heavy": _fit(full_means, _normal_log_likelihood(observed, full_means) - tail),
    }


def _standard_error_of_sum(pointwise: torch.Tensor) -> float:
    """Return the standard error of the sum of independent pointwise terms."""
    return float(len(pointwise) ** 0.5 * pointwise.std(correction=0))


@pytest.fixture(scope="module")
def nested(fits: dict[str, DataTree]) -> ModelComparison:
    return compare_models(
        {"null": fits["null"], "full": fits["full"]}, gate=ParetoGate(fraction=0.1)
    )


@pytest.fixture(scope="module")
def gated(fits: dict[str, DataTree]) -> ModelComparison:
    with pytest.warns(UserWarning, match="Pareto"):
        return compare_models(fits, gate=ParetoGate(fraction=0.1))


def test_a_declared_comparison_lowers_every_setting_to_the_elpd_adapter(
    fits: dict[str, DataTree],
) -> None:
    declaration = ModelComparisonSpec(
        candidates=("null", "full"),
        candidate="full",
        criterion=ModelComparisonCriterion.PSIS_LOO,
        weighting=ModelWeighting.PSEUDO_BMA,
        pareto_fraction=1.0,
        pareto_threshold=1_000.0,
    )
    compared = compare_declared_models(
        {"null": fits["null"], "full": fits["full"]},
        declaration,
    )

    assert compared.method == "pseudo-BMA"
    assert compared.gate == ParetoGate(fraction=1.0, threshold=1_000.0)
    assert tuple(row.name for row in compared.models) == ("full", "null")


@pytest.mark.parametrize(
    ("selected", "missing", "unexpected"),
    [
        (("null",), "full", ""),
        (("null", "full", "heavy"), "", "heavy"),
    ],
)
def test_a_declared_comparison_requires_exactly_its_candidate_set(
    fits: dict[str, DataTree],
    selected: tuple[str, ...],
    missing: str,
    unexpected: str,
) -> None:
    declaration = ModelComparisonSpec(
        candidates=("null", "full"),
        candidate="full",
        criterion=ModelComparisonCriterion.PSIS_LOO,
        weighting=ModelWeighting.STACKING,
        pareto_fraction=0.1,
        pareto_threshold=0.7,
    )

    with pytest.raises(
        ValueError,
        match=(
            "fitted models do not match the declared candidates: "
            f"missing=.*{missing}.*unexpected=.*{unexpected}"
        ),
    ):
        compare_declared_models({name: fits[name] for name in selected}, declaration)


def test_the_generating_response_beats_both_twins_on_known_random_effects() -> None:
    program = KNOWN_RANDOM_EFFECTS.program()
    generating = KNOWN_RANDOM_EFFECTS.generating()

    def without(*sites: str) -> torch.Tensor:
        changed = generating.clone()
        start = 0

        for slot in program.layout:
            if slot.site in sites:
                changed[start : start + slot.size] = 0.0

            start += slot.size

        return changed

    def around(center: torch.Tensor, seed: int) -> DataTree:
        generator = torch.Generator().manual_seed(seed)
        draws = center[None, :] + 0.02 * torch.randn(
            200,
            program.size,
            generator=generator,
            dtype=torch.float64,
        )
        likelihood = torch.stack(
            [log_likelihood(program, parameters) for parameters in draws]
        )

        return to_datatree_from_svi(
            {"parameters": draws},
            log_likelihood={"response": likelihood},
        )

    fits = {
        "generating": around(generating, 1),
        "no-participant": around(
            without(
                "participant_intercept_offsets",
                "participant_cutpoint_offsets",
            ),
            2,
        ),
        "no-score": around(without("score_slope"), 3),
    }
    declaration = ModelComparisonSpec(
        candidates=("generating", "no-participant", "no-score"),
        candidate="generating",
        criterion=ModelComparisonCriterion.PSIS_LOO,
        weighting=ModelWeighting.STACKING,
        pareto_fraction=1.0,
        pareto_threshold=1_000.0,
    )
    compared = compare_declared_models(fits, declaration)

    assert compared.ranking[0] == "generating"
    assert all(
        delta.left == "generating" and delta.delta > 0.0
        for delta in compared.deltas
        if delta.left == "generating"
    )


# ---------------------------------------------------------------------------
# the lazy boundary
# ---------------------------------------------------------------------------


def test_the_package_import_does_not_load_the_substrate() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, lofi.eval; "
                "print(sorted(name for name in ('arviz', 'quivers', 'torch') "
                "if name in sys.modules))"
            ),
        ],
        capture_output=True,
        encoding="utf-8",
        check=True,
    )

    assert completed.stdout.strip() == "[]"


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def test_gate_defaults_the_threshold_and_passes_at_most_the_fraction() -> None:
    gate = ParetoGate(fraction=0.25)

    assert gate.threshold == 0.7
    assert gate.fraction == 0.25
    assert gate.passes(0, 4)
    assert gate.passes(1, 4)
    assert not gate.passes(2, 4)
    assert ParetoGate(fraction=0.0).passes(0, 4)
    assert not ParetoGate(fraction=0.0).passes(1, 4)
    assert ParetoGate(fraction=1.0).passes(4, 4)


def test_gate_finds_a_shape_untrustworthy_above_the_threshold_or_unestimated() -> None:
    gate = ParetoGate(fraction=0.25)

    assert not gate.exceeds(0.0)
    assert not gate.exceeds(0.7)
    assert gate.exceeds(nextafter(0.7, 1.0))
    assert gate.exceeds(float("inf"))
    assert gate.exceeds(float("nan"))
    assert ParetoGate(fraction=0.25, threshold=1.0).exceeds(1.5)
    assert not ParetoGate(fraction=0.25, threshold=1.0).exceeds(0.9)


@pytest.mark.parametrize(
    ("fraction", "threshold", "error"),
    [
        (-0.1, 0.7, ValueError),
        (1.1, 0.7, ValueError),
        (0.5, 0.0, ValueError),
        (0.5, -0.7, ValueError),
        (0.5, float("inf"), ValueError),
        ("0.5", 0.7, TypeError),
        (0.5, True, TypeError),
    ],
)
def test_gate_rejects_a_malformed_rule(
    fraction: float, threshold: float, error: type[Exception]
) -> None:
    with pytest.raises(error):
        ParetoGate(fraction=fraction, threshold=threshold)


def test_gate_rejects_counts_it_cannot_judge() -> None:
    gate = ParetoGate(fraction=0.5)

    with pytest.raises(ValueError, match="no observations"):
        gate.passes(0, 0)

    with pytest.raises(ValueError, match="5 observations exceed"):
        gate.passes(5, 4)

    with pytest.raises(ValueError, match="negative"):
        gate.passes(-1, 4)

    with pytest.raises(TypeError, match="not an integer"):
        gate.passes(cast("int", 1.0), 4)


def test_gate_equality_covers_both_parts() -> None:
    gate = ParetoGate(fraction=0.2, threshold=0.5)

    assert gate == ParetoGate(fraction=0.2, threshold=0.5)
    assert hash(gate) == hash(ParetoGate(fraction=0.2, threshold=0.5))
    assert gate != ParetoGate(fraction=0.3, threshold=0.5)
    assert gate != ParetoGate(fraction=0.2, threshold=0.7)
    assert gate != (0.2, 0.5)
    assert repr(gate) == "ParetoGate(fraction=0.2, threshold=0.5)"


# ---------------------------------------------------------------------------
# the adapter
# ---------------------------------------------------------------------------


def test_nested_pair_recovers_the_known_ordering(
    nested: ModelComparison, observed: torch.Tensor
) -> None:
    assert nested.ranking == ("full", "null")
    assert nested.method == "stacking"
    assert nested.unreliable == ()

    full, null = nested.models
    at_sample_mean = -0.5 * (observed - observed.mean()) ** 2

    assert (full.rank, null.rank) == (0, 1)
    assert full.elpd > null.elpd
    assert full.se == pytest.approx(_standard_error_of_sum(at_sample_mean), rel=0.1)
    assert null.se > full.se
    assert full.effective_parameters == pytest.approx(1.0, abs=0.25)
    assert null.effective_parameters > full.effective_parameters
    assert (full.observations, null.observations) == (60, 60)
    assert (full.exceeding, null.exceeding) == (0, 0)
    assert full.reliable
    assert null.reliable


def test_grammar_and_language_model_fits_share_one_complete_table(
    fits: dict[str, DataTree],
) -> None:
    with pytest.warns(UserWarning, match="Pareto"):
        comparison = compare_models(
            {
                "grammar-full": fits["null"],
                "grammar-ablated": fits["heavy"],
                "language-model": fits["full"],
            },
            gate=ParetoGate(fraction=0.1),
        )

    assert comparison.ranking == (
        "language-model",
        "grammar-full",
        "grammar-ablated",
    )
    assert len(comparison.models) == 3
    assert len(comparison.deltas) == 3
    assert {
        frozenset((delta.left, delta.right))
        for delta in comparison.deltas
        if "language-model" in (delta.left, delta.right)
    } == {
        frozenset(("language-model", "grammar-full")),
        frozenset(("language-model", "grammar-ablated")),
    }

    rows = comparison.rows()[: len(comparison.models)]

    for model, row in zip(comparison.models, rows, strict=True):
        expected = model.elpd / model.observations

        assert model.elpd_per_observation == pytest.approx(expected)
        assert row["model"] == model.name
        assert row["rank"] == model.rank
        assert row["elpd_per_observation"] == pytest.approx(expected)


def test_weights_sum_to_one_and_favor_the_better_model(
    nested: ModelComparison,
) -> None:
    full, null = nested.models

    assert full.weight + null.weight == pytest.approx(1.0)
    assert full.weight > null.weight
    # Stacking weights are an SLSQP optimum, settled to its default tolerance.
    assert full.weight == pytest.approx(1.0, abs=1e-6)


def test_pairwise_delta_is_signed_from_the_higher_ranked_model(
    nested: ModelComparison, observed: torch.Tensor
) -> None:
    (delta,) = nested.deltas
    full_over_null = -0.5 * (observed - observed.mean()) ** 2 + 0.5 * observed**2

    assert (delta.left, delta.right) == ("full", "null")
    assert delta.delta == pytest.approx(float(full_over_null.sum()), rel=0.05)
    assert delta.se == pytest.approx(_standard_error_of_sum(full_over_null), rel=0.1)
    assert delta.delta > 4.0 * delta.se
    assert delta.probability_left_better > 0.99
    assert delta.flagged is False


def test_an_exact_tie_reports_an_even_chance_and_no_difference(
    fits: dict[str, DataTree],
) -> None:
    tied = compare_models(
        {"a": fits["full"], "b": fits["full"].copy()}, gate=ParetoGate(fraction=0.1)
    )
    a, b = tied.models
    (delta,) = tied.deltas

    assert tied.ranking == ("a", "b")
    assert a.elpd == b.elpd
    assert a.weight + b.weight == pytest.approx(1.0)
    assert (delta.left, delta.right) == ("a", "b")
    assert delta.delta == 0.0
    assert delta.se == 0.0
    assert delta.probability_left_better == 0.5
    assert delta.flagged is False


def test_three_comparable_fits_have_weights_that_sum_to_one(
    observed: torch.Tensor,
) -> None:
    comparable: dict[str, DataTree] = {}

    for index, offset in enumerate((0.0, 0.001, 0.002)):
        means = _posterior_means(observed, float(observed.mean()) + offset, seed=1)
        comparable[f"m{index}"] = _fit(means, _normal_log_likelihood(observed, means))

    comparison = compare_models(comparable, gate=ParetoGate(fraction=0.1))
    weights = [row.weight for row in comparison.models]

    assert max(weights) - min(weights) < 0.05
    assert sum(weights) == pytest.approx(1.0)


def test_a_fit_over_the_declared_fraction_is_unreliable_and_flagged(
    gated: ModelComparison,
) -> None:
    assert gated.ranking == ("full", "null", "heavy")
    assert gated.unreliable == ("heavy",)

    heavy = gated.models[2]

    assert heavy.exceeding == heavy.observations == 60
    assert heavy.reliable is False
    assert {(delta.left, delta.right): delta.flagged for delta in gated.deltas} == {
        ("full", "null"): False,
        ("full", "heavy"): True,
        ("null", "heavy"): True,
    }
    assert all(delta.delta > 0.0 for delta in gated.deltas)


def test_the_same_fits_pass_a_gate_that_allows_every_observation(
    fits: dict[str, DataTree],
) -> None:
    with pytest.warns(UserWarning, match="Pareto"):
        lenient = compare_models(fits, gate=ParetoGate(fraction=1.0))

    assert lenient.unreliable == ()
    assert not any(delta.flagged for delta in lenient.deltas)

    with pytest.warns(UserWarning, match="Pareto"):
        strict = compare_models(fits, gate=ParetoGate(fraction=0.99))

    assert strict.unreliable == ("heavy",)


def test_a_fit_whose_shapes_cannot_be_estimated_is_unreliable(
    fits: dict[str, DataTree], observed: torch.Tensor
) -> None:
    means = _posterior_means(observed, float(observed.mean()), seed=4)
    log_likelihood = _normal_log_likelihood(observed, means)
    log_likelihood[0] = -1_000.0

    comparison = compare_models(
        {"full": fits["full"], "spiked": _fit(means, log_likelihood)},
        gate=ParetoGate(fraction=0.5, threshold=1_000.0),
    )
    spiked = comparison.models[1]

    assert spiked.exceeding == spiked.observations == 60
    assert spiked.reliable is False
    assert comparison.unreliable == ("spiked",)
    assert comparison.deltas[0].flagged is True


def test_a_higher_threshold_admits_the_heavy_fit(fits: dict[str, DataTree]) -> None:
    with pytest.warns(UserWarning, match="Pareto"):
        lenient = compare_models(fits, gate=ParetoGate(fraction=0.1, threshold=1_000.0))

    assert lenient.models[2].exceeding == 0
    assert lenient.unreliable == ()


def test_the_weighting_method_is_forwarded(fits: dict[str, DataTree]) -> None:
    pseudo = compare_models(
        {"null": fits["null"], "full": fits["full"]},
        gate=ParetoGate(fraction=0.1),
        method="pseudo-BMA",
    )

    assert pseudo.method == "pseudo-BMA"
    assert pseudo.ranking == ("full", "null")
    assert pseudo.models[0].weight + pseudo.models[1].weight == pytest.approx(1.0)


def test_the_observed_variable_can_be_named(fits: dict[str, DataTree]) -> None:
    named = compare_models(
        {"null": fits["null"], "full": fits["full"]},
        gate=ParetoGate(fraction=0.1),
        var_name="y",
    )

    assert named.ranking == ("full", "null")


def test_the_adapter_refuses_malformed_requests(fits: dict[str, DataTree]) -> None:
    with pytest.raises(ValueError, match="at least two fits"):
        compare_models({"full": fits["full"]}, gate=ParetoGate(fraction=0.1))

    with pytest.raises(ValueError, match="empty"):
        compare_models(
            {"": fits["full"], "null": fits["null"]}, gate=ParetoGate(fraction=0.1)
        )

    with pytest.raises(TypeError, match="gate is not one"):
        compare_models(fits, gate=cast("ParetoGate", 0.7))

    with pytest.raises(ValueError, match="weighting method is unknown"):
        compare_models(
            fits, gate=ParetoGate(fraction=0.1), method=cast("WeightingMethod", "waic")
        )


def test_a_table_figure_that_is_not_finite_is_refused() -> None:
    assert _column({"elpd": -81.0}, "full", "elpd") == -81.0

    with pytest.raises(ValueError, match="model 'full' has no finite se: nan"):
        _column({"se": float("nan")}, "full", "se")

    with pytest.raises(ValueError, match="no finite dse: inf"):
        _column({"dse": float("inf")}, "full", "dse")


def test_rows_are_long_format_and_serializable(gated: ModelComparison) -> None:
    rows = gated.rows()

    assert json.loads(json.dumps(rows)) == rows
    assert [row["kind"] for row in rows] == ["model"] * 3 + ["delta"] * 3
    assert {row["method"] for row in rows} == {"stacking"}
    assert {row["threshold"] for row in rows} == {0.7}
    assert {row["fraction"] for row in rows} == {0.1}
    assert [row["model"] for row in rows[:3]] == ["full", "null", "heavy"]
    assert [row["reliable"] for row in rows[:3]] == [True, True, False]
    assert [(row["left"], row["right"]) for row in rows[3:]] == [
        ("full", "null"),
        ("full", "heavy"),
        ("null", "heavy"),
    ]
    assert [row["flagged"] for row in rows[3:]] == [False, True, True]
    assert rows[0] == {
        "kind": "model",
        "method": "stacking",
        "threshold": 0.7,
        "fraction": 0.1,
        **gated.models[0].to_spec(),
    }
    assert rows[3] == {
        "kind": "delta",
        "method": "stacking",
        "threshold": 0.7,
        "fraction": 0.1,
        **gated.deltas[0].to_spec(),
    }


@pytest.mark.parametrize(
    ("elpd", "observations", "error", "message"),
    [
        (float("inf"), 10, ValueError, "ELPD"),
        (-10.0, 0, ValueError, "no observations"),
        (-10.0, -1, ValueError, "negative"),
        (-10.0, 1.0, TypeError, "not an integer"),
    ],
)
def test_a_model_row_refuses_an_invalid_per_observation_density(
    elpd: float,
    observations: int,
    error: type[Exception],
    message: str,
) -> None:
    row = ModelRow("bad", 0, elpd, 1.0, 2.0, 1.0, observations, 0, reliable=True)

    with pytest.raises(error, match=message):
        row.to_spec()

    with pytest.raises(error, match=message):
        ModelComparison(
            (row, _row("good", 1)),
            (_delta("bad", "good"),),
            gate=ParetoGate(fraction=0.1),
            method="stacking",
        )


# ---------------------------------------------------------------------------
# the comparison value
# ---------------------------------------------------------------------------


def _row(name: str, rank: int, *, reliable: bool = True) -> ModelRow:
    return ModelRow(name, rank, -10.0 * (rank + 1), 1.0, 2.0, 0.5, 10, 0, reliable)


def _delta(left: str, right: str, *, flagged: bool = False) -> PairwiseDelta:
    return PairwiseDelta(left, right, 10.0, 1.0, 0.99, flagged)


def test_comparison_value_holds_its_parts_and_compares_on_all_of_them() -> None:
    rows = (_row("a", 0), _row("b", 1))
    deltas = (_delta("a", "b"),)
    gate = ParetoGate(fraction=0.1)

    comparison = ModelComparison(rows, deltas, gate=gate, method="stacking")

    assert comparison.models == rows
    assert comparison.deltas == deltas
    assert comparison.gate == gate
    assert comparison.ranking == ("a", "b")
    assert comparison.unreliable == ()
    assert comparison == ModelComparison(rows, deltas, gate=gate, method="stacking")
    assert hash(comparison) == hash(
        ModelComparison(rows, deltas, gate=gate, method="stacking")
    )
    assert comparison != ModelComparison(rows, deltas, gate=gate, method="pseudo-BMA")
    assert comparison != ModelComparison(
        rows, deltas, gate=ParetoGate(fraction=0.2), method="stacking"
    )
    assert comparison != (rows, deltas)
    assert repr(comparison).startswith("ModelComparison((ModelRow(name='a', ")
    assert repr(comparison).endswith(
        "gate=ParetoGate(fraction=0.1, threshold=0.7), method='stacking')"
    )
    assert WEIGHTING_METHODS == ("stacking", "BB-pseudo-BMA", "pseudo-BMA")


def test_comparison_value_refuses_inconsistent_parts() -> None:
    gate = ParetoGate(fraction=0.1)

    with pytest.raises(ValueError, match="not consecutive"):
        ModelComparison(
            (_row("a", 0), _row("b", 2)),
            (_delta("a", "b"),),
            gate=gate,
            method="stacking",
        )

    with pytest.raises(ValueError, match="not consecutive"):
        ModelComparison(
            (_row("b", 1), _row("a", 0)),
            (_delta("b", "a"),),
            gate=gate,
            method="stacking",
        )

    with pytest.raises(ValueError, match="do not cover"):
        ModelComparison((_row("a", 0), _row("b", 1)), (), gate=gate, method="stacking")

    with pytest.raises(ValueError, match="do not cover"):
        ModelComparison(
            (_row("a", 0), _row("b", 1)),
            (_delta("b", "a"),),
            gate=gate,
            method="stacking",
        )

    with pytest.raises(ValueError, match="flag disagrees"):
        ModelComparison(
            (_row("a", 0), _row("b", 1, reliable=False)),
            (_delta("a", "b"),),
            gate=gate,
            method="stacking",
        )

    with pytest.raises(ValueError, match="flag disagrees"):
        ModelComparison(
            (_row("a", 0), _row("b", 1)),
            (_delta("a", "b", flagged=True),),
            gate=gate,
            method="stacking",
        )

    with pytest.raises(ValueError, match="unknown"):
        ModelComparison(
            (_row("a", 0), _row("b", 1)),
            (_delta("a", "b"),),
            gate=gate,
            method=cast("WeightingMethod", "waic"),
        )

    with pytest.raises(TypeError, match="gate is not one"):
        ModelComparison(
            (_row("a", 0), _row("b", 1)),
            (_delta("a", "b"),),
            gate=cast("ParetoGate", 0.1),
            method="stacking",
        )

    with pytest.raises(TypeError, match="model row is not one"):
        ModelComparison(
            (_row("a", 0), cast("ModelRow", ("b", 1))),
            (_delta("a", "b"),),
            gate=gate,
            method="stacking",
        )

    with pytest.raises(TypeError, match="delta is not one"):
        ModelComparison(
            (_row("a", 0), _row("b", 1)),
            (cast("PairwiseDelta", ("a", "b")),),
            gate=gate,
            method="stacking",
        )
