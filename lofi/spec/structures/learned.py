"""Boundary declaration of structures fitted to a language theory."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple, Self

import didactic.api as dx

from lofi.spec.structures.sampled import StructureSourceSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["LearnedSettings", "LearnedStructureSourceSpec"]


class LearnedSettings(NamedTuple):
    """Resolved learned-source settings consumed by the neural factory.

    ``initialization_scale`` is the standard deviation of the normal draw each
    denotation logit starts from.
    """

    structures: int
    carriers: Mapping[str, int]
    full_function_space_cap: int
    henkin_pool: int
    denotations: Literal["free", "from-vocabulary"]
    seed: int
    initialization_scale: float


class LearnedStructureSourceSpec(StructureSourceSpec):
    """Declare relaxed finite structures fitted only to declared theory."""

    kind: Literal["learned"] = "learned"
    structures: int
    carriers: dict[str, int]
    full_function_space_cap: int
    henkin_pool: int
    denotations: Literal["free", "from-vocabulary"]
    seed: int
    initialization_scale: float

    __axioms__ = (
        dx.axiom(
            "initialization_scale > 0.0",
            message="denotation logits start at a positive scale",
        ),
    )

    @dx.model_validator()
    def _settings_are_bounded(self) -> Self:
        """Validate counts and carriers once at the declaration boundary."""
        if self.structures < 1:
            msg = "a learned source fits at least one structure"
            raise ValueError(msg)

        if self.full_function_space_cap < 1 or self.henkin_pool < 1:
            msg = "function-space cap and Henkin pool are positive"
            raise ValueError(msg)

        if self.seed < 0:
            msg = "a learned-structure seed is nonnegative"
            raise ValueError(msg)

        if not self.carriers:
            msg = "a learned source declares at least one atomic carrier"
            raise ValueError(msg)

        if any(not name or size < 1 for name, size in self.carriers.items()):
            msg = "every named atomic carrier has positive size"
            raise ValueError(msg)

        return self

    def settings(self) -> LearnedSettings:
        """Return immutable interior settings for the learned source."""
        return LearnedSettings(
            self.structures,
            MappingProxyType(dict(self.carriers)),
            self.full_function_space_cap,
            self.henkin_pool,
            self.denotations,
            self.seed,
            self.initialization_scale,
        )
