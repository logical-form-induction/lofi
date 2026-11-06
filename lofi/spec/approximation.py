"""One run's residual mass budget and its component overrides."""

from json import dumps
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, Protocol, final, runtime_checkable

import didactic.api as dx

from lofi._checks import ensure_finite
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import ReportValue

__all__ = [
    "ApproximationConsumer",
    "ApproximationPlan",
    "ApproximationRequest",
    "ApproximationSpec",
    "residual_mass_bound",
]


def residual_mass_bound(value: float, /) -> float:
    """Return a finite residual mass probability strictly below one."""
    bound = ensure_finite("a residual mass bound", value)

    if not 0.0 <= bound < 1.0:
        msg = f"a residual mass bound lies in [0, 1), not at {value!r}"
        raise ValueError(msg)

    return bound


class ApproximationSpec(dx.Model, extra="forbid"):
    """Declare the residual mass every truncating component may leave."""

    residual_mass: float

    @dx.validates("residual_mass")
    @classmethod
    def _residual_mass_is_a_bound(cls, value: float) -> float:
        """Validate the run level residual mass budget."""
        return residual_mass_bound(value)


class ApproximationRequest(NamedTuple):
    """A truncating component's optional explicit residual mass override."""

    residual_mass: float | None


@runtime_checkable
class ApproximationConsumer(Protocol):
    """A component declaration that may consume a residual mass budget."""

    def approximation_request(self) -> ApproximationRequest | None:
        """Return this component's request, or none when it does not truncate."""
        ...


@final
class ApproximationPlan(FrozenValue):
    """Resolved residual mass bounds keyed by structural component path."""

    _noun = "approximation plan"
    __slots__ = ("_bounds", "_measurements")

    def __init__(
        self,
        declared: ApproximationSpec | None,
        requests: Mapping[str, ApproximationRequest],
        /,
    ) -> None:
        default = None if declared is None else declared.residual_mass
        bounds: dict[str, float] = {}
        overrides: dict[str, float] = {}

        for path, request in sorted(requests.items()):
            override = request.residual_mass

            if override is None:
                if default is None:
                    msg = (
                        f"{path}: a truncating component requires the run's "
                        "approximation.residual_mass bound or an explicit "
                        "residual_mass override"
                    )
                    raise ValueError(msg)

                bounds[path] = default
            else:
                bounds[path] = residual_mass_bound(override)
                overrides[path] = override

        measurements: dict[str, ReportValue] = {
            "approximation-overrides": dumps(
                overrides,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        }

        if default is not None:
            measurements["approximation-residual-mass"] = default

        object.__setattr__(self, "_bounds", MappingProxyType(bounds))
        object.__setattr__(self, "_measurements", MappingProxyType(measurements))

    def bound(self, path: str, /) -> float:
        """Return the resolved bound for one truncating component path."""
        return self._bounds[path]

    def measurements(self) -> Mapping[str, ReportValue]:
        """Return the compile metrics describing the bound and its overrides."""
        return self._measurements
