"""Constructor-generic type-diagnostic evaluator declaration."""

from itertools import pairwise
from math import isfinite
from typing import Literal, NamedTuple, Self, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec._ports import Dependency
from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.conventions import ConventionSettings, ConventionSpec

__all__ = [
    "ConstructorDiagnosticSettings",
    "ConstructorDiagnosticSpec",
    "ConstructorViewSettings",
    "ConstructorViewSpec",
]

type BeamNormalizationName = Literal["none", "depth", "nodes", "terminals"]
"""A declared normalization of candidate log weights."""


class ConstructorViewSettings(NamedTuple):
    """The checked constructor view consumed by diagnostic projections."""

    representative: str
    roles: tuple[tuple[str, tuple[str, ...]], ...]


class ConstructorViewSpec(dx.Model, extra="forbid"):
    """Declare one representative leaf and each constructor's child roles."""

    representative: str
    roles: dict[str, tuple[str, ...]]

    @dx.validates("representative")
    @classmethod
    def _representative_is_named(cls, value: str) -> str:
        """Return a nonempty representative category."""
        return ensure_name("a constructor diagnostic representative", value)

    @dx.model_validator()
    def _roles_are_named(self) -> Self:
        """Return a nonempty view with one or more named roles per constructor."""
        if not self.roles:
            msg = "a constructor diagnostic view names at least one constructor"
            raise ValueError(msg)

        for position, (constructor, roles) in enumerate(self.roles.items()):
            ensure_name(f"constructor view name {position}", constructor)

            if not roles:
                msg = f"constructor {constructor!r} needs at least one child role"
                raise ValueError(msg)

            for slot, role in enumerate(roles):
                ensure_name(f"constructor {constructor!r} role {slot}", role)

        return self

    def settings(self) -> ConstructorViewSettings:
        """Return the immutable constructor view settings."""
        return ConstructorViewSettings(
            self.representative,
            tuple((name, tuple(roles)) for name, roles in self.roles.items()),
        )


class ConstructorDiagnosticSettings(NamedTuple):
    """The settings consumed by constructor-generic held-out diagnostics."""

    readout: str
    view: ConstructorViewSettings
    normalization: BeamNormalizationName
    widths: tuple[int, ...]
    depths: tuple[int, ...]
    collapse_margin: float
    conventions: ConventionSettings


class ConstructorDiagnosticSpec(EvaluatorSpec):
    """Declare beam, projection, and reconstruction rows over constructor roles.

    ``collapse_margin`` is how far the mean depth of the best-ranked candidates
    may fall below the mean gold depth before the ``depth-collapse`` row flags
    a collapse to shallow types.
    """

    kind: Literal["constructor-diagnostics"] = "constructor-diagnostics"
    readout: str
    view: ConstructorViewSpec
    normalization: BeamNormalizationName
    widths: tuple[int, ...]
    depths: tuple[int, ...]
    collapse_margin: float
    conventions: ConventionSpec

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        """Return a nonempty complete term readout role."""
        return ensure_name("a constructor diagnostic term-readout role", value)

    @dx.validates("collapse_margin")
    @classmethod
    def _margin_is_nonnegative(cls, value: float) -> float:
        """Return a finite, nonnegative collapse margin."""
        if not isfinite(value) or value < 0:
            msg = f"a collapse margin is finite and nonnegative, not {value}"
            raise ValueError(msg)

        return value

    @dx.model_validator()
    def _bounds_ascend(self) -> Self:
        """Return positive, strictly ascending beam widths and projection depths."""
        for field, values in (("widths", self.widths), ("depths", self.depths)):
            if not values:
                msg = f"constructor diagnostic {field} cannot be empty"
                raise ValueError(msg)
            if values[0] <= 0 or any(right <= left for left, right in pairwise(values)):
                msg = f"constructor diagnostic {field} must ascend from one"
                raise ValueError(msg)

        return self

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the declared complete term readout dependency."""
        return (Dependency(self.readout, "terms-readout"),)

    @override
    def metrics(self, labels: tuple[str, ...] = (), /) -> tuple[str, ...]:
        """Return every row this declaration emits, in report order."""
        del labels
        return (
            *(f"beam-accuracy@{width}" for width in self.widths),
            *(f"typeless-accuracy@{width}" for width in self.widths),
            *(f"truncated-accuracy@{depth}" for depth in self.depths),
            "mean-reciprocal-rank",
            "primitive-induced-count",
            "primitive-gold-count",
            "primitive-inventory-exact",
            "primitive-f1",
            "constructor-induced-count",
            "constructor-gold-count",
            "constructor-inventory-exact",
            "constructor-role-information",
            "lexical-type-f1",
            "lexical-token-accuracy",
            "depth-gap",
            "depth-collapse",
        )

    def settings(self) -> ConstructorDiagnosticSettings:
        """Return immutable evaluator settings."""
        return ConstructorDiagnosticSettings(
            self.readout,
            self.view.settings(),
            self.normalization,
            self.widths,
            self.depths,
            self.collapse_margin,
            self.conventions.settings(),
        )
