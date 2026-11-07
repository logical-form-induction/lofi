from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import pytest
import torch

from lofi.judgment import (
    ConstraintCost,
    FixedDiscrimination,
    InferenceMethod,
    Measurement,
    PosteriorFit,
    ScaleProfile,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping


def fit() -> PosteriorFit:
    return PosteriorFit(
        draws={"slope": torch.arange(16, dtype=torch.float64).reshape(2, 4, 2)},
        log_densities=torch.zeros((2, 4), dtype=torch.float64),
        acceptance_rates=torch.tensor([0.8, 0.9], dtype=torch.float64),
        divergence_counts=torch.tensor([1, 2], dtype=torch.int64),
        r_hat={"slope": torch.tensor([1.01, 1.02], dtype=torch.float64)},
        effective_sample_sizes={"slope": torch.tensor([6.0, 7.0], dtype=torch.float64)},
    )


def test_a_posterior_fit_preserves_chain_and_draw_identity() -> None:
    result = fit()

    assert result.chains == 2
    assert result.retained == 4
    assert result.draws["slope"].shape == (2, 4, 2)
    assert result.log_densities.shape == (2, 4)
    assert result.divergence_counts.tolist() == [1, 2]


def test_a_posterior_fit_exposes_the_declared_nuts_diagnostics() -> None:
    diagnostics = fit().diagnostics

    assert diagnostics.method is InferenceMethod.NUTS
    assert diagnostics.divergences == 3
    assert diagnostics.max_rhat == pytest.approx(1.02)
    assert diagnostics.min_ess == pytest.approx(6.0)


def test_a_posterior_fit_builds_the_existing_summary_from_downstream_metadata() -> None:
    summary = fit().summary(
        spec_hash="spec-digest",
        location="fits/reference.nc",
        identification=FixedDiscrimination(value=1.0),
        scale_profile=ScaleProfile(
            scales=(0.5, 1.0, 2.0),
            log_posterior=(-3.0, 0.0, -3.0),
            minimum_tail_drop=2.0,
        ),
        constraint_cost=ConstraintCost(
            constrained=Measurement(value=-3.0, n=8, unit="judgment"),
            unconstrained=Measurement(value=-2.5, n=8, unit="judgment"),
        ),
    )

    assert summary.spec_hash == "spec-digest"
    assert summary.posterior is not None
    assert summary.posterior.location == "fits/reference.nc"
    assert summary.posterior.chains == 2
    assert summary.posterior.draws == 4
    assert summary.diagnostics.method is InferenceMethod.NUTS
    assert summary.diagnostics.divergences == 3


def test_a_posterior_fit_keeps_immutable_snapshots() -> None:
    source = torch.arange(16, dtype=torch.float64).reshape(2, 4, 2)
    result = PosteriorFit(
        draws={"slope": source},
        log_densities=torch.zeros((2, 4), dtype=torch.float64),
        acceptance_rates=torch.tensor([0.8, 0.9], dtype=torch.float64),
        divergence_counts=torch.zeros(2, dtype=torch.int64),
        r_hat={"slope": torch.ones(2, dtype=torch.float64)},
        effective_sample_sizes={"slope": torch.full((2,), 7.0)},
    )
    source.zero_()
    returned = result.draws
    returned["slope"].zero_()

    assert isinstance(returned, MappingProxyType)
    assert torch.equal(
        result.draws["slope"],
        torch.arange(16, dtype=torch.float64).reshape(2, 4, 2),
    )
    mutable = cast("MutableMapping[str, torch.Tensor]", returned)
    with pytest.raises(TypeError):
        mutable["other"] = torch.zeros(1)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({}, "posterior draws is empty"),
        (
            {"slope": torch.zeros((2, 4, 2), dtype=torch.float64)},
            "posterior R-hat is empty",
        ),
    ],
)
def test_a_posterior_fit_refuses_missing_draws_or_diagnostics(
    replacement: dict[str, torch.Tensor], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PosteriorFit(
            draws=replacement,
            log_densities=torch.zeros((2, 4), dtype=torch.float64),
            acceptance_rates=torch.tensor([0.8, 0.9], dtype=torch.float64),
            divergence_counts=torch.zeros(2, dtype=torch.int64),
            r_hat={},
            effective_sample_sizes={},
        )


def test_a_posterior_fit_refuses_a_nonfinite_result() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        PosteriorFit(
            draws={"slope": torch.full((2, 4, 1), torch.nan)},
            log_densities=torch.zeros((2, 4)),
            acceptance_rates=torch.ones(2),
            divergence_counts=torch.zeros(2, dtype=torch.int64),
            r_hat={"slope": torch.ones(1)},
            effective_sample_sizes={"slope": torch.ones(1)},
        )


def test_a_posterior_fit_refuses_a_floating_divergence_count() -> None:
    with pytest.raises(TypeError, match="not an integer tensor"):
        PosteriorFit(
            draws={"slope": torch.zeros((2, 4, 1))},
            log_densities=torch.zeros((2, 4)),
            acceptance_rates=torch.ones(2),
            divergence_counts=torch.zeros(2),
            r_hat={"slope": torch.ones(1)},
            effective_sample_sizes={"slope": torch.ones(1)},
        )
