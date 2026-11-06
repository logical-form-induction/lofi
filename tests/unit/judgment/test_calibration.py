from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

from lofi.judgment._spec import InferenceMethod
from lofi.judgment.calibration import (
    CalibrationReplicate,
    CalibrationReport,
    CoverageSummary,
    CutpointRecovery,
    SbcBlockSummary,
    calibration_report,
)

METHODS = (
    InferenceMethod.NUTS,
    InferenceMethod.STRUCTURED,
    InferenceMethod.MEAN_FIELD,
)


def _cutpoint_truth() -> Tensor:
    return torch.tensor(
        [[-2.0, 3.0], [-0.5, 1.0], [1.0, -1.0], [3.0, -2.0]],
        dtype=torch.float64,
    )


def _replicate(
    seed: int,
    *,
    truth: Tensor | None = None,
    nuts: Tensor | None = None,
    blocks: Mapping[str, Tensor] | None = None,
    reverse_nuts_cutpoint: bool = False,
) -> CalibrationReplicate:
    held_truth = torch.tensor([0.0], dtype=torch.float64) if truth is None else truth
    truths = {"coefficient": held_truth} if blocks is None else dict(blocks)
    if nuts is None:
        nuts = (
            held_truth.unsqueeze(0).unsqueeze(0).repeat(2, 5, *([1] * held_truth.ndim))
        )
        nuts = nuts + torch.linspace(-1.0, 1.0, 10, dtype=torch.float64).reshape(
            2, 5, *([1] * held_truth.ndim)
        )

    posterior_draws: dict[InferenceMethod, dict[str, Tensor]] = {}
    for method in METHODS:
        method_blocks: dict[str, Tensor] = {}
        for block, value in truths.items():
            if block == "coefficient":
                base = nuts.clone()
            else:
                base = value.unsqueeze(0).unsqueeze(0).repeat(2, 5, *([1] * value.ndim))
                base += torch.linspace(-1.0, 1.0, 10, dtype=torch.float64).reshape(
                    2, 5, *([1] * value.ndim)
                )
            if method is InferenceMethod.MEAN_FIELD:
                base += 10.0
            method_blocks[block] = base
        posterior_draws[method] = method_blocks

    cutpoint_truth = _cutpoint_truth()
    chains, draws = nuts.shape[:2]
    cutpoint_draws = {
        method: cutpoint_truth.unsqueeze(0).unsqueeze(0).repeat(chains, draws, 1, 1)
        for method in METHODS
    }
    cutpoint_draws[InferenceMethod.STRUCTURED] += torch.tensor(
        [0.1, -0.1], dtype=torch.float64
    )
    cutpoint_draws[InferenceMethod.MEAN_FIELD][:, :, :, 1] = torch.flip(
        cutpoint_truth[:, 1], dims=(0,)
    )
    if reverse_nuts_cutpoint:
        cutpoint_draws[InferenceMethod.NUTS][:, :, :, 0] = torch.flip(
            cutpoint_truth[:, 0], dims=(0,)
        )

    return CalibrationReplicate(
        seed=seed,
        truths=truths,
        posterior_draws=posterior_draws,
        true_cutpoints=cutpoint_truth,
        cutpoint_draws=cutpoint_draws,
    )


def test_randomized_sbc_ranks_break_ties_from_the_caller_owned_stream() -> None:
    truth = torch.tensor([0.0], dtype=torch.float64)
    tied = torch.tensor([-1.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64).reshape(
        1, 5, 1
    )

    report = calibration_report((_replicate(3, truth=truth, nuts=tied),), rank_seed=17)

    assert report.sbc[0].ranks == (4,)
    assert report.sbc[0].histogram == (0, 0, 0, 0, 1, 0)


def test_uniform_ranks_pass_and_a_biased_reference_fails_the_ks_test() -> None:
    truth = torch.zeros(101, dtype=torch.float64)
    uniform = torch.empty((1, 100, 101), dtype=torch.float64)
    for rank in range(101):
        uniform[0, :rank, rank] = -1.0
        uniform[0, rank:, rank] = 1.0
    biased = torch.ones((1, 100, 101), dtype=torch.float64)

    calibrated = calibration_report(
        (_replicate(1, truth=truth, nuts=uniform),), rank_seed=0
    )
    misspecified = calibration_report(
        (_replicate(1, truth=truth, nuts=biased),), rank_seed=0
    )

    assert calibrated.sbc[0].passes
    assert calibrated.sbc[0].ks_p_value > 0.99
    assert not misspecified.sbc[0].passes
    assert misspecified.sbc[0].ks_p_value < 0.05


def test_coverage_compares_all_three_methods_over_the_same_truths() -> None:
    report = calibration_report((_replicate(1), _replicate(2)), rank_seed=4)
    coverage = {(row.method, row.block): row for row in report.coverage}

    assert tuple(row.method for row in report.coverage) == METHODS
    assert coverage[InferenceMethod.NUTS, "coefficient"].fraction == 1.0
    assert coverage[InferenceMethod.STRUCTURED, "coefficient"].fraction == 1.0
    assert coverage[InferenceMethod.MEAN_FIELD, "coefficient"].fraction == 0.0
    assert all(row.level == 0.9 for row in report.coverage)


def test_cutpoint_recovery_is_separate_by_cutpoint_and_never_flattened() -> None:
    report = calibration_report((_replicate(7),), rank_seed=5)
    recovered = {(row.method, row.cutpoint): row for row in report.cutpoints}

    assert recovered[InferenceMethod.NUTS, 0].pearson == pytest.approx(1.0)
    assert recovered[InferenceMethod.NUTS, 1].spearman == pytest.approx(1.0)
    assert recovered[InferenceMethod.MEAN_FIELD, 0].pearson == pytest.approx(1.0)
    assert recovered[InferenceMethod.MEAN_FIELD, 1].pearson < 0.0
    assert tuple(row.cutpoint for row in report.cutpoints) == (0, 1, 0, 1, 0, 1)


def test_a_nuts_cutpoint_below_the_floor_is_reported_rather_than_refused() -> None:
    cleared = calibration_report((_replicate(1),), rank_seed=0)
    missed = calibration_report(
        (_replicate(1, reverse_nuts_cutpoint=True),), rank_seed=0
    )
    nuts = {
        row.cutpoint: row
        for row in missed.cutpoints
        if row.method is InferenceMethod.NUTS
    }
    rows = {
        row["cutpoint"]: row["clears_floor"]
        for row in missed.long_rows
        if row["summary"] == "cutpoint_recovery" and row["method"] == "nuts"
    }

    assert cleared.clears_floor
    assert not missed.clears_floor
    assert not nuts[0].clears_floor
    assert nuts[1].clears_floor
    assert rows == {0: False, 1: True}


def test_replicate_and_mapping_order_do_not_change_the_report() -> None:
    left = _replicate(
        1,
        blocks={
            "zeta": torch.tensor([1.0], dtype=torch.float64),
            "alpha": torch.tensor([0.0], dtype=torch.float64),
        },
    )
    right = _replicate(
        2,
        blocks={
            "alpha": torch.tensor([0.0], dtype=torch.float64),
            "zeta": torch.tensor([1.0], dtype=torch.float64),
        },
    )

    first = calibration_report((right, left), rank_seed=9)
    second = calibration_report((left, right), rank_seed=9)

    assert tuple(summary.block for summary in first.sbc) == ("alpha", "zeta")
    assert first.long_rows == second.long_rows
    assert tuple(summary.ranks for summary in first.sbc) == tuple(
        summary.ranks for summary in second.sbc
    )


def test_rank_randomization_never_touches_torch_global_random_state() -> None:
    state = torch.random.get_rng_state().clone()

    calibration_report((_replicate(1),), rank_seed=1234)

    assert torch.equal(torch.random.get_rng_state(), state)


def test_report_rows_are_deterministic_json_scalars() -> None:
    report = calibration_report((_replicate(1),), rank_seed=3)
    scalars = (str, int, float, bool, type(None))
    rank_rows = tuple(
        row for row in report.long_rows if row["summary"] == "sbc_rank_bin"
    )

    assert isinstance(report, CalibrationReport)
    assert isinstance(report.sbc[0], SbcBlockSummary)
    assert isinstance(report.coverage[0], CoverageSummary)
    assert isinstance(report.cutpoints[0], CutpointRecovery)
    assert report.long_rows == report.long_rows
    assert all(
        isinstance(value, scalars) for row in report.long_rows for value in row.values()
    )
    assert tuple((row["rank"], row["count"]) for row in rank_rows) == tuple(
        enumerate(report.sbc[0].histogram)
    )


def test_a_direct_report_refuses_duplicate_or_noncanonical_sbc_blocks() -> None:
    report = calibration_report((_replicate(1),), rank_seed=3)
    summary = report.sbc[0]

    with pytest.raises(ValueError, match="distinct blocks in canonical order"):
        CalibrationReport(
            replicates=1,
            rank_seed=3,
            sbc=(summary, summary),
            coverage=report.coverage,
            cutpoints=report.cutpoints,
        )

    other = SbcBlockSummary(
        "zeta",
        (0,),
        summary.posterior_samples + 1,
        0.5,
        0.5,
    )
    with pytest.raises(ValueError, match="unequal posterior sample counts"):
        CalibrationReport(
            replicates=1,
            rank_seed=3,
            sbc=(summary, other),
            coverage=report.coverage,
            cutpoints=report.cutpoints,
        )


def test_a_direct_report_requires_every_canonical_coverage_cell_once() -> None:
    report = calibration_report((_replicate(1),), rank_seed=3)

    for malformed in (
        report.coverage[:-1],
        (*report.coverage, report.coverage[-1]),
        tuple(reversed(report.coverage)),
    ):
        with pytest.raises(ValueError, match="every method-by-block cell"):
            CalibrationReport(
                replicates=1,
                rank_seed=3,
                sbc=report.sbc,
                coverage=malformed,
                cutpoints=report.cutpoints,
            )


def test_a_direct_report_requires_every_coherent_cutpoint_cell_once() -> None:
    report = calibration_report((_replicate(1),), rank_seed=3)

    for malformed in (
        report.cutpoints[:-1],
        (*report.cutpoints, report.cutpoints[-1]),
        tuple(reversed(report.cutpoints)),
    ):
        with pytest.raises(
            ValueError, match=r"method-by-cutpoint cell|canonical contiguous"
        ):
            CalibrationReport(
                replicates=1,
                rank_seed=3,
                sbc=report.sbc,
                coverage=report.coverage,
                cutpoints=malformed,
            )

    last = report.cutpoints[-1]
    unequal = (
        *report.cutpoints[:-1],
        CutpointRecovery(
            last.method,
            last.cutpoint,
            last.participants + 1,
            last.pearson,
            last.spearman,
        ),
    )
    with pytest.raises(ValueError, match="unequal participant counts"):
        CalibrationReport(
            replicates=1,
            rank_seed=3,
            sbc=report.sbc,
            coverage=report.coverage,
            cutpoints=unequal,
        )


def test_replicates_snapshot_mutable_tensor_inputs() -> None:
    truth = torch.tensor([0.0], dtype=torch.float64)
    replicate = _replicate(1, truth=truth)

    truth.fill_(99.0)
    exposed = replicate.truths["coefficient"]
    exposed.fill_(42.0)

    assert replicate.truths["coefficient"].item() == 0.0


def test_empty_replicates_and_repeated_seeds_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one replicate"):
        calibration_report((), rank_seed=0)

    replicate = _replicate(1)
    with pytest.raises(ValueError, match="not unique"):
        calibration_report((replicate, replicate), rank_seed=0)


def test_unequal_blocks_and_shapes_across_replicates_are_refused() -> None:
    one = _replicate(1)
    extra = _replicate(
        2,
        blocks={
            "coefficient": torch.tensor([0.0], dtype=torch.float64),
            "scale": torch.tensor([1.0], dtype=torch.float64),
        },
    )
    vector = _replicate(3, truth=torch.tensor([0.0, 1.0], dtype=torch.float64))

    with pytest.raises(ValueError, match="identical block and tensor shapes"):
        calibration_report((one, extra), rank_seed=0)
    with pytest.raises(ValueError, match="identical block and tensor shapes"):
        calibration_report((one, vector), rank_seed=0)


def test_missing_or_extra_inference_methods_are_refused() -> None:
    truth = torch.tensor([0.0], dtype=torch.float64)
    draws = torch.zeros((1, 2, 1), dtype=torch.float64)
    posterior = {method: {"x": draws} for method in METHODS[:-1]}
    cutpoints = _cutpoint_truth()
    cutpoint_draws = {
        method: cutpoints.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
        for method in METHODS
    }

    with pytest.raises(ValueError, match="must contain exactly"):
        CalibrationReplicate(
            seed=0,
            truths={"x": truth},
            posterior_draws=posterior,
            true_cutpoints=cutpoints,
            cutpoint_draws=cutpoint_draws,
        )


def test_nonfinite_values_and_malformed_draw_axes_are_refused() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        _replicate(1, truth=torch.tensor([float("nan")], dtype=torch.float64))

    truth = torch.tensor([0.0], dtype=torch.float64)
    malformed = torch.zeros((5, 1), dtype=torch.float64)
    with pytest.raises(ValueError, match=r"not \(chain, draw"):
        _replicate(1, truth=truth, nuts=malformed)


def test_empty_blocks_and_mismatched_method_blocks_are_refused() -> None:
    cutpoints = _cutpoint_truth()
    cutpoint_draws = {
        method: cutpoints.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
        for method in METHODS
    }
    with pytest.raises(ValueError, match="no parameter blocks"):
        CalibrationReplicate(
            seed=0,
            truths={},
            posterior_draws={method: {} for method in METHODS},
            true_cutpoints=cutpoints,
            cutpoint_draws=cutpoint_draws,
        )

    truth = torch.tensor([0.0], dtype=torch.float64)
    draws = torch.zeros((1, 2, 1), dtype=torch.float64)
    posterior = {method: {"x": draws} for method in METHODS}
    posterior[InferenceMethod.MEAN_FIELD] = {"other": draws}
    with pytest.raises(ValueError, match="do not have the truth blocks"):
        CalibrationReplicate(
            seed=0,
            truths={"x": truth},
            posterior_draws=posterior,
            true_cutpoints=cutpoints,
            cutpoint_draws=cutpoint_draws,
        )
