"""Immutable posterior-fit plans, draws and sampler diagnostics."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import final

import torch
from torch import Tensor

from lofi._checks import ensure_name, ensure_non_negative, ensure_positive
from lofi._frozen import FrozenValue
from lofi.judgment._spec import (
    ConstraintCost,
    FitDiagnostics,
    FitSummary,
    InferenceMethod,
    PosteriorHandle,
    ScaleIdentification,
    ScaleProfile,
)

__all__ = ["NutsPlan", "PosteriorFit"]


@final
class NutsPlan(FrozenValue):
    """A complete deterministic plan for one multi-chain NUTS fit."""

    _noun = "NUTS plan"

    __slots__ = ("_chains", "_retained", "_seed", "_warmup")

    def __init__(
        self,
        *,
        warmup: int,
        retained: int,
        chains: int,
        seed: int,
    ) -> None:
        held_warmup = ensure_positive("a NUTS plan's warmup count", warmup)
        held_retained = ensure_positive("a NUTS plan's retained draw count", retained)
        held_chains = ensure_positive("a NUTS plan's chain count", chains)
        held_seed = ensure_non_negative("a NUTS plan's seed", seed)
        if held_retained < 4:
            msg = "a NUTS plan retains at least four draws per chain for diagnostics"
            raise ValueError(msg)
        if held_chains < 2:
            msg = "a NUTS plan runs at least two chains for convergence diagnostics"
            raise ValueError(msg)

        object.__setattr__(self, "_warmup", held_warmup)
        object.__setattr__(self, "_retained", held_retained)
        object.__setattr__(self, "_chains", held_chains)
        object.__setattr__(self, "_seed", held_seed)

    @property
    def warmup(self) -> int:
        """Return the number of discarded adaptation iterations per chain."""
        return self._warmup

    @property
    def retained(self) -> int:
        """Return the number of retained draws per chain."""
        return self._retained

    @property
    def chains(self) -> int:
        """Return the number of independently adapted chains."""
        return self._chains

    @property
    def seed(self) -> int:
        """Return the root seed from which Quivers derives chain streams."""
        return self._seed


def _floating_tensor(
    field: str,
    value: Tensor,
    *,
    dimensions: int | None = None,
) -> Tensor:
    """Return a finite floating-point CPU snapshot of a tensor."""
    if not isinstance(value, Tensor):
        msg = f"{field} is not a tensor: {value!r}"
        raise TypeError(msg)
    if dimensions is not None and value.ndim != dimensions:
        msg = f"{field} has shape {tuple(value.shape)!r}, not {dimensions} dimensions"
        raise ValueError(msg)
    if value.numel() == 0:
        msg = f"{field} is empty"
        raise ValueError(msg)
    if not value.is_floating_point() or value.is_complex():
        msg = f"{field} is not a real floating-point tensor"
        raise TypeError(msg)
    if not bool(torch.isfinite(value).all()):
        msg = f"{field} contains a nonfinite value"
        raise ValueError(msg)
    return value.detach().to(device="cpu").clone()


def _divergences(field: str, value: Tensor) -> Tensor:
    """Return a nonnegative integer CPU snapshot of divergence counts."""
    if not isinstance(value, Tensor):
        msg = f"{field} is not a tensor: {value!r}"
        raise TypeError(msg)
    if value.ndim != 1:
        msg = f"{field} has shape {tuple(value.shape)!r}, not one chain axis"
        raise ValueError(msg)
    if value.numel() == 0:
        msg = f"{field} is empty"
        raise ValueError(msg)
    if value.is_floating_point() or value.is_complex() or value.dtype is torch.bool:
        msg = f"{field} is not an integer tensor"
        raise TypeError(msg)
    if bool((value < 0).any()):
        msg = f"{field} contains a negative count"
        raise ValueError(msg)
    return value.detach().to(device="cpu").clone()


def _site_snapshots(field: str, values: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Return named finite snapshots in stable name order."""
    if not isinstance(values, Mapping):
        msg = f"{field} is not a mapping"
        raise TypeError(msg)
    if not values:
        msg = f"{field} is empty"
        raise ValueError(msg)

    held: dict[str, Tensor] = {}
    for raw_name, value in values.items():
        name = ensure_name(f"{field} site", raw_name)
        held[name] = _floating_tensor(f"{field} for site {name!r}", value)
    return {name: held[name] for name in sorted(held)}


@final
class PosteriorFit(FrozenValue):
    """Chain-preserving posterior draws and NUTS diagnostics.

    Draw tensors have shape ``(chain, draw, *site_shape)``. R-hat and
    effective-sample-size tensors retain ``site_shape`` and thus never hide
    chain identity by flattening a posterior sample.
    """

    _noun = "posterior fit"

    __slots__ = (
        "_acceptance_rates",
        "_divergence_counts",
        "_draws",
        "_effective_sample_sizes",
        "_log_densities",
        "_r_hat",
    )

    def __init__(
        self,
        *,
        draws: Mapping[str, Tensor],
        log_densities: Tensor,
        acceptance_rates: Tensor,
        divergence_counts: Tensor,
        r_hat: Mapping[str, Tensor],
        effective_sample_sizes: Mapping[str, Tensor],
    ) -> None:
        held_draws = _site_snapshots("posterior draws", draws)
        leading: tuple[int, int] | None = None
        for name, site_draws in held_draws.items():
            if site_draws.ndim < 2:
                msg = (
                    f"posterior draws for site {name!r} have shape "
                    f"{tuple(site_draws.shape)!r}, without chain and draw axes"
                )
                raise ValueError(msg)
            current = (site_draws.shape[0], site_draws.shape[1])
            if current[0] < 1 or current[1] < 1:
                msg = f"posterior draws for site {name!r} have an empty axis"
                raise ValueError(msg)
            if leading is not None and current != leading:
                msg = "posterior sites have unequal chain and draw axes"
                raise ValueError(msg)
            leading = current

        if leading is None:  # pragma: no cover - _site_snapshots already refuses this
            msg = "posterior draws are empty"
            raise ValueError(msg)

        densities = _floating_tensor(
            "posterior log densities", log_densities, dimensions=2
        )
        if tuple(densities.shape) != leading:
            msg = (
                f"posterior log densities have shape {tuple(densities.shape)!r}, "
                f"not {leading!r}"
            )
            raise ValueError(msg)

        acceptance = _floating_tensor(
            "posterior acceptance rates", acceptance_rates, dimensions=1
        )
        if acceptance.shape[0] != leading[0]:
            msg = "posterior acceptance rates do not have one value per chain"
            raise ValueError(msg)
        if bool(((acceptance < 0.0) | (acceptance > 1.0)).any()):
            msg = "posterior acceptance rates fall outside [0, 1]"
            raise ValueError(msg)

        divergences = _divergences("posterior divergence counts", divergence_counts)
        if divergences.shape[0] != leading[0]:
            msg = "posterior divergence counts do not have one value per chain"
            raise ValueError(msg)

        held_r_hat = _site_snapshots("posterior R-hat", r_hat)
        held_ess = _site_snapshots(
            "posterior effective sample sizes", effective_sample_sizes
        )
        sites = frozenset(held_draws)
        if frozenset(held_r_hat) != sites or frozenset(held_ess) != sites:
            msg = "posterior diagnostics do not name exactly the drawn sites"
            raise ValueError(msg)

        for name, site_draws in held_draws.items():
            shape = tuple(site_draws.shape[2:])
            if tuple(held_r_hat[name].shape) != shape:
                msg = f"posterior R-hat for site {name!r} does not match its shape"
                raise ValueError(msg)
            if tuple(held_ess[name].shape) != shape:
                msg = (
                    f"posterior effective sample size for site {name!r} does not "
                    "match its shape"
                )
                raise ValueError(msg)
            if bool((held_r_hat[name] < 0.0).any()):
                msg = f"posterior R-hat for site {name!r} is negative"
                raise ValueError(msg)
            if bool((held_ess[name] < 0.0).any()):
                msg = f"posterior effective sample size for site {name!r} is negative"
                raise ValueError(msg)

        object.__setattr__(self, "_draws", MappingProxyType(held_draws))
        object.__setattr__(self, "_log_densities", densities)
        object.__setattr__(self, "_acceptance_rates", acceptance)
        object.__setattr__(self, "_divergence_counts", divergences)
        object.__setattr__(self, "_r_hat", MappingProxyType(held_r_hat))
        object.__setattr__(self, "_effective_sample_sizes", MappingProxyType(held_ess))

    @property
    def draws(self) -> Mapping[str, Tensor]:
        """Return posterior snapshots by drawn-site name."""
        return MappingProxyType(
            {name: value.clone() for name, value in self._draws.items()}
        )

    @property
    def log_densities(self) -> Tensor:
        """Return one log density per chain and retained draw."""
        return self._log_densities.clone()

    @property
    def acceptance_rates(self) -> Tensor:
        """Return the mean acceptance rate of every chain."""
        return self._acceptance_rates.clone()

    @property
    def divergence_counts(self) -> Tensor:
        """Return the number of divergent transitions in every chain."""
        return self._divergence_counts.clone()

    @property
    def r_hat(self) -> Mapping[str, Tensor]:
        """Return rank-normalized folded R-hat by drawn site."""
        return MappingProxyType(
            {name: value.clone() for name, value in self._r_hat.items()}
        )

    @property
    def effective_sample_sizes(self) -> Mapping[str, Tensor]:
        """Return cross-chain bulk effective sample size by drawn site."""
        return MappingProxyType(
            {
                name: value.clone()
                for name, value in self._effective_sample_sizes.items()
            }
        )

    @property
    def chains(self) -> int:
        """Return the number of preserved chains."""
        return self._log_densities.shape[0]

    @property
    def retained(self) -> int:
        """Return the number of post-warmup draws in every chain."""
        return self._log_densities.shape[1]

    @property
    def diagnostics(self) -> FitDiagnostics:
        """Return the declared fit diagnostics for this NUTS result."""
        return FitDiagnostics(
            method=InferenceMethod.NUTS,
            divergences=int(self._divergence_counts.sum().item()),
            max_rhat=max(float(value.max().item()) for value in self._r_hat.values()),
            min_ess=min(
                float(value.min().item())
                for value in self._effective_sample_sizes.values()
            ),
        )

    def summary(
        self,
        *,
        spec_hash: str,
        location: str,
        identification: ScaleIdentification,
        scale_profile: ScaleProfile,
        constraint_cost: ConstraintCost,
    ) -> FitSummary:
        """Return the persisted fit declaration around this sampled result.

        The caller supplies the storage location and the identification
        analyses it owns. The sampled result supplies only the handle axes and
        diagnostics it measured.

        Parameters
        ----------
        spec_hash : str
            Content hash of the fitted response specification.
        location : str
            Path or URI where the caller persisted these posterior draws.
        identification : ScaleIdentification
            Constraint identifying the fitted latent scale.
        scale_profile : ScaleProfile
            Downstream scale-profile analysis.
        constraint_cost : ConstraintCost
            Downstream predictive cost of the scale constraint.

        Returns
        -------
        FitSummary
            Existing declared summary with an exact posterior handle.
        """
        return FitSummary(
            spec_hash=spec_hash,
            posterior=PosteriorHandle(
                location=location,
                chains=self.chains,
                draws=self.retained,
            ),
            diagnostics=self.diagnostics,
            identification=identification,
            scale_profile=scale_profile,
            constraint_cost=constraint_cost,
        )
