"""Interference among type encodings accumulated by superposition."""

from typing import TYPE_CHECKING, ClassVar, Final

import torch

from lofi.nn.sites import DTypeClass, Role
from lofi.probe._diagnose import DiagnosticResult, Measurement

if TYPE_CHECKING:
    from lofi.nn.sites import Site
    from lofi.probe._diagnose import DiagnosticContext
    from lofi.probe._values import Observation, Trace

__all__ = ["SuperpositionInterference"]

_AXES: Final[tuple[str, str]] = ("term", "width")
_TERMS: Final[str] = "terms"
_SCORES: Final[tuple[str, ...]] = tuple(
    f"interference-at-{size}" for size in range(1, 6)
)


def _type_readings(
    context: DiagnosticContext, trace: Trace, /
) -> tuple[tuple[Site, Observation], ...]:
    """Return observed type encodings in publication order."""
    taken = dict(context.readings(trace))
    published = context.sites_for(Role.TERM_ENCODING)

    for site in published:
        spec = context.spec_at(site)

        if spec.axes != _AXES:
            msg = (
                f"TERM_ENCODING at {site!r} is published over {spec.axes!r}, "
                f"not {_AXES!r}"
            )
            raise ValueError(msg)

        if spec.dtype_class is not DTypeClass.REAL:
            msg = (
                f"TERM_ENCODING at {site!r} is published as "
                f"{spec.dtype_class.value!r}, not real"
            )
            raise ValueError(msg)

    missing = tuple(site for site in published if site not in taken)

    if missing:
        names = ", ".join(repr(site) for site in missing)
        msg = f"missing TERM_ENCODING observation at published site(s): {names}"
        raise ValueError(msg)

    return tuple((site, taken[site]) for site in published)


def _live_vectors(site: Site, reading: Observation, /) -> tuple[torch.Tensor, int]:
    """Return the live vectors at one type-encoding site and their count."""
    if reading.axes != _AXES:
        msg = (
            f"the TERM_ENCODING observation at {site!r} runs over "
            f"{reading.axes!r}, not {_AXES!r}"
        )
        raise ValueError(msg)

    values = reading.tensor

    if not values.is_floating_point():
        msg = (
            f"the TERM_ENCODING observation at {site!r} is floating point, "
            f"not {values.dtype}"
        )
        raise TypeError(msg)

    if values.shape[1] < 2:
        msg = (
            f"the TERM_ENCODING observation at {site!r} has width "
            f"{values.shape[1]}, which is less than two"
        )
        raise ValueError(msg)

    live = reading.covering().all(dim=-1)

    if not bool(live.any()):
        msg = f"every term read at TERM_ENCODING site {site!r} is masked off"
        raise ValueError(msg)

    rows = live.nonzero(as_tuple=False).flatten()
    vectors = values[live].to(torch.float64)
    norms = vectors.norm(dim=-1)
    zero = norms.eq(0.0).nonzero(as_tuple=False)

    if zero.numel():
        row = int(rows[int(zero[0, 0])])
        msg = f"TERM_ENCODING site {site!r} has a live zero-norm vector at row {row}"
        raise ValueError(msg)

    return vectors, int(rows.numel())


def _measure(site: Site, reading: Observation, /) -> tuple[Measurement, ...]:
    """Return interference measurements for one observed encoding matrix."""
    vectors, count = _live_vectors(site, reading)
    measured = [Measurement(site, _TERMS, float(count))]
    prefix = torch.zeros_like(vectors[0])

    for size, vector in enumerate(vectors[:5], start=1):
        prefix = prefix + vector
        prefix_norm = prefix.norm()

        if bool(prefix_norm.eq(0.0)):
            continue

        members = vectors[:size]
        similarities = (members @ prefix) / (members.norm(dim=-1) * prefix_norm)
        score = ((1.0 - similarities.mean()) / 2.0).clamp(0.0, 1.0)
        measured.append(Measurement(site, _SCORES[size - 1], float(score)))

    return tuple(measured)


class SuperpositionInterference:
    """Measures angular interference as type encodings are superposed in row order."""

    __slots__ = ()

    tag: ClassVar[str] = "superposition-interference"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.TERM_ENCODING})
    scores: ClassVar[frozenset[str]] = frozenset(_SCORES)

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return prefix interference for every observed type-encoding site."""
        measured = [
            measurement
            for site, reading in _type_readings(context, trace)
            for measurement in _measure(site, reading)
        ]

        return DiagnosticResult(self.tag, measured)
