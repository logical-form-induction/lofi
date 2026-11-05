"""Declared category-to-logical-form coherence-square evaluation."""

from typing import TYPE_CHECKING, Literal, NamedTuple, cast, override

import didactic.api as dx

from lofi._checks import ensure_name, ensure_non_negative
from lofi.spec._ports import Dependency
from lofi.spec.components.run import EvaluatorSpec, ParseSpec
from lofi.spec.evaluators.conventions import ConventionSettings, ConventionSpec
from lofi.spec.evaluators.decode import (
    AStarSpec,
    DecodeMethodDeclaration,
    DecodeSpec,
    DerivationDecision,
    astar_parse_refusal,
    decode_decision,
)

if TYPE_CHECKING:
    from lofi.deduce import AStarOptions
    from lofi.spec._errors import DeclarationError

__all__ = ["CoherenceSquareSettings", "CoherenceSquareSpec"]

type CoherenceMode = Literal["off", "sample", "all"]

_METRICS = (
    "coherence-square-rate",
    "coherence-square-items",
    "coherence-square-instances",
    "coherence-square-strict-rate",
    "coherence-square-collapse-rate",
)


class CoherenceSquareSettings(NamedTuple):
    """Resolved coherence-square settings."""

    readout: str
    astar: AStarOptions
    decode: DerivationDecision
    mode: CoherenceMode
    sample_items: int | None
    seed: int
    conventions: ConventionSettings
    metrics: tuple[str, ...]


class CoherenceSquareSpec(EvaluatorSpec):
    """Declare whether no, sampled, or every decoded chart item is checked."""

    kind: Literal["coherence-square"] = "coherence-square"
    readout: str
    astar: AStarSpec
    decode: DecodeSpec
    mode: CoherenceMode
    sample_items: int | None
    seed: int
    conventions: ConventionSpec

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        return ensure_name("a coherence-square term-readout role", value)

    @dx.validates("seed")
    @classmethod
    def _seed_is_nonnegative(cls, value: int) -> int:
        return ensure_non_negative("a coherence-square seed", value)

    @dx.model_validator()
    def _sampling_is_explicit(self) -> CoherenceSquareSpec:
        if self.mode == "sample":
            if self.sample_items is None or self.sample_items < 1:
                msg = "sampled coherence declares a positive item count"
                raise ValueError(msg)
        elif self.sample_items is not None:
            msg = f"coherence mode {self.mode!r} does not declare a sample size"
            raise ValueError(msg)
        return self

    def parse_refusal(self, parse: ParseSpec, /) -> DeclarationError | None:
        """Return why the run parse cannot serve the declared A* decode."""
        decode = cast("DecodeMethodDeclaration", self.decode)
        return astar_parse_refusal(
            self.astar, parse, marginals=decode.requires_marginals
        )

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the complete-term readout that supplies lexical categories."""
        return (Dependency(self.readout, "terms-readout"),)

    @override
    def metrics(self, labels: tuple[str, ...] = (), /) -> tuple[str, ...]:
        """Return the coherence and collapse report rows."""
        del labels
        return _METRICS

    def settings(self) -> CoherenceSquareSettings:
        """Return immutable plugin settings."""
        return CoherenceSquareSettings(
            self.readout,
            self.astar.build(),
            decode_decision(self.decode),
            self.mode,
            self.sample_items,
            self.seed,
            self.conventions.settings(),
            self.metrics(),
        )
