"""Declarations of tensor log weight reductions."""

from typing import Final, Literal, NamedTuple, Protocol

import didactic.api as dx

__all__ = [
    "LogSumExpReductionSettings",
    "LogSumExpReductionSpec",
    "MaximumReductionSettings",
    "MaximumReductionSpec",
    "ReductionDeclaration",
    "ReductionResolver",
    "ReductionSettings",
    "ReductionSpec",
]


class ReductionResolver[R](Protocol):
    """Build one runtime value for each declared reduction."""

    def log_sum_exp(self) -> R:
        """Return the runtime log-sum-exp reduction."""
        ...

    def maximum(self) -> R:
        """Return the runtime maximum reduction."""
        ...


class LogSumExpReductionSettings(NamedTuple):
    """Resolved log-sum-exp reduction settings."""

    def resolve[R](self, resolver: ReductionResolver[R], /) -> R:
        """Return the resolver's log-sum-exp implementation."""
        return resolver.log_sum_exp()


class MaximumReductionSettings(NamedTuple):
    """Resolved maximum reduction settings."""

    def resolve[R](self, resolver: ReductionResolver[R], /) -> R:
        """Return the resolver's maximum implementation."""
        return resolver.maximum()


type ReductionSettings = LogSumExpReductionSettings | MaximumReductionSettings
"""Resolved settings of one tensor log weight reduction."""


class ReductionDeclaration(Protocol):
    """A tensor reduction declaration, of either variant."""

    def settings(self) -> ReductionSettings:
        """Return the declared reduction's settings."""
        ...


class ReductionSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare how a tensor chart combines alternative derivations.

    Each variant states ``probabilistic``: whether the reduction defines posterior
    marginals, which outside passes and stacked charts read.
    """

    probabilistic = False


class LogSumExpReductionSpec(ReductionSpec):
    """Sum alternative probabilities in log space."""

    kind: Literal["log-sum-exp"] = "log-sum-exp"
    probabilistic = True

    def settings(self) -> LogSumExpReductionSettings:
        """Return the resolved log-sum-exp settings."""
        return LogSumExpReductionSettings()


class MaximumReductionSpec(ReductionSpec):
    """Keep only the largest alternative log weight."""

    kind: Literal["maximum"] = "maximum"

    def settings(self) -> MaximumReductionSettings:
        """Return the resolved maximum settings."""
        return MaximumReductionSettings()


_DEFAULT_REDUCTION: Final = dx.field(default_factory=LogSumExpReductionSpec)
"""The shared default reduction field of tensor chart declarations."""
