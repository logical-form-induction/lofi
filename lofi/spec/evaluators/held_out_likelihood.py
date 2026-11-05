"""Held-out string and derivational likelihood declaration."""

from typing import Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec._ports import Dependency
from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.conventions import ConventionSettings, ConventionSpec
from lofi.spec.losses.simplicity import SymbolCostSpec, UniformSymbolCostSpec

__all__ = [
    "HeldOutLikelihoodMeasurement",
    "HeldOutLikelihoodSettings",
    "HeldOutLikelihoodSpec",
]


type HeldOutLikelihoodMeasurement = Literal[
    "string", "derivation", "description-length"
]
"""A family of held-out likelihood measurements computed together."""

_MEASUREMENT_ORDER: tuple[HeldOutLikelihoodMeasurement, ...] = (
    "string",
    "derivation",
    "description-length",
)


class HeldOutLikelihoodSettings(NamedTuple):
    """Validated settings for held-out parser likelihood."""

    readout: str
    output: str
    symbol_cost: SymbolCostSpec
    conventions: ConventionSettings
    measurements: tuple[HeldOutLikelihoodMeasurement, ...]


class HeldOutLikelihoodSpec(EvaluatorSpec):
    """Declare held-out string and normal-form derivational likelihood."""

    kind: Literal["held-out-likelihood"] = "held-out-likelihood"
    readout: str
    output: str
    symbol_cost: SymbolCostSpec = dx.field(default_factory=UniformSymbolCostSpec)
    conventions: ConventionSpec
    measurements: tuple[HeldOutLikelihoodMeasurement, ...]

    @dx.validates("readout", "output")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        """Return a nonempty span-readout role or output name."""
        return ensure_name("a held-out likelihood span-readout role", value)

    @dx.validates("measurements")
    @classmethod
    def _measurements_are_nonempty_and_distinct(
        cls, values: tuple[HeldOutLikelihoodMeasurement, ...]
    ) -> tuple[HeldOutLikelihoodMeasurement, ...]:
        """Return a nonempty set of measurement families in report order."""
        if not values:
            msg = "held-out likelihood declares no measurements"
            raise ValueError(msg)
        if len(set(values)) != len(values):
            msg = f"held-out likelihood measurements repeat: {values!r}"
            raise ValueError(msg)

        selected = frozenset(values)
        return tuple(value for value in _MEASUREMENT_ORDER if value in selected)

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return only the readouts consumed by the selected measurements."""
        selected = frozenset(self.measurements)
        return tuple(
            dependency
            for measurement, dependency in (
                ("string", Dependency(self.readout, "span-readout")),
                (
                    "description-length",
                    Dependency(self.readout, "terms-readout"),
                ),
            )
            if measurement in selected
        )

    @override
    def metrics(self, labels: tuple[str, ...] = (), /) -> tuple[str, ...]:
        """Return rows for exactly the selected measurement families."""
        del labels
        selected = frozenset(self.measurements)
        return tuple(
            name
            for measurement, names in (
                (
                    "string",
                    (
                        "string-log-likelihood-per-sentence",
                        "string-log-likelihood-per-token",
                    ),
                ),
                (
                    "derivation",
                    (
                        "derivational-log-likelihood-per-sentence",
                        "derivational-log-likelihood-per-token",
                        "derivational-log-likelihood-per-sentence",
                        "derivational-log-likelihood-per-token",
                    ),
                ),
                ("description-length", ("description-length-nats",)),
            )
            if measurement in selected
            for name in names
        )

    def settings(self) -> HeldOutLikelihoodSettings:
        """Return immutable evaluator settings."""
        return HeldOutLikelihoodSettings(
            self.readout,
            self.output,
            self.symbol_cost,
            self.conventions.settings(),
            self.measurements,
        )
