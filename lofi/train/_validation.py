"""Deterministic occurrence splitting and item-weighted epoch validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from lofi.train._steps import EvalResult

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import nn

    from lofi.nn.policy import DevicePolicy
    from lofi.train._losses import LossAssembly
    from lofi.train._ports import Forward

__all__ = ["validate_epoch"]


class _ValidationBatch(Protocol):
    """A batch whose item count weights validation metrics."""

    def __len__(self) -> int: ...


def _modules(roots: Mapping[str, nn.Module], /) -> tuple[nn.Module, ...]:
    """Return each reachable module once with parents before their children."""
    found: list[nn.Module] = []
    seen: set[int] = set()

    for root in roots.values():
        for module in root.modules():
            if id(module) not in seen:
                found.append(module)
                seen.add(id(module))

    return tuple(found)


def validate_epoch[BatchT: _ValidationBatch](
    forward: Forward[BatchT],
    assembly: LossAssembly[BatchT],
    batches: Sequence[BatchT],
    modules: Mapping[str, nn.Module],
    policy: DevicePolicy,
    /,
    *,
    step: int,
) -> EvalResult:
    """Evaluate one epoch without gradients and aggregate by item occurrence.

    Parameters
    ----------
    forward : Forward[BatchT]
        Configured parser forward pass.
    assembly : LossAssembly[BatchT]
        Weighted objectives evaluated during training.
    batches : Sequence[BatchT]
        Validation batches in their seeded order.
    modules : Mapping[str, nn.Module]
        Complete configured module roles.
    policy : DevicePolicy
        Run-wide autocast policy.
    step : int
        The count of optimizer steps the run has taken, which sets the weight
        each scheduled term takes.

    Raises
    ------
    NumericalFailureError
        If an aggregated metric is not finite.
    """
    reachable = _modules(modules)
    modes = tuple(module.training for module in reachable)
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    items = 0

    try:
        for module in reachable:
            module.eval()

        with torch.no_grad(), policy.autocast():
            for batch in batches:
                count = len(batch)
                result = assembly.evaluate(forward.publish(batch), step=step)
                items += count

                for name, value in result.metrics.items():
                    totals[name] = totals.get(name, 0.0) + value * count
                    counts[name] = counts.get(name, 0) + count

    finally:
        for module, training in zip(reachable, modes, strict=True):
            module.train(training)

    return EvalResult(
        items,
        metrics={name: total / counts[name] for name, total in totals.items()},
    )
