"""Generated-term distribution declaration shared by term pretraining."""

from typing import TYPE_CHECKING, Self

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name, ensure_positive
from lofi.spec.type_samplers.distribution import (
    _DEFAULT_BRANCHING,
    _DEFAULT_CHILD_DEPTH,
    ArityBranchingSpec,
    ChildDepthSpec,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "TermGenerationSpec",
    "autoencoding_objective",
    "combinator_objective",
    "terms_metrics",
]


class TermGenerationSpec(dx.Model, extra="forbid"):
    """Declare the terms and combinator pairs term pretraining draws.

    Parameters
    ----------
    max_depth : int
        The deepest generated term and rule instance.
    child_depth : ChildDepthSpec
        How the depth of each child of a generated term is drawn.
    branching : ArityBranchingSpec
        Which children of each arity reach the deepest level.
    constructor_weights : dict[str, float]
        The relative weight of each type constructor, by name; an unnamed
        constructor keeps its sampler's weight.
    teacher_forcing_saturation : float | None
        The fraction of the horizon after which combinator decoding is no
        longer teacher-forced, or None to teacher-force throughout.
    """

    max_depth: int
    child_depth: ChildDepthSpec = _DEFAULT_CHILD_DEPTH
    branching: ArityBranchingSpec = _DEFAULT_BRANCHING
    constructor_weights: dict[str, float] = dx.field(default_factory=dict)
    teacher_forcing_saturation: float | None = None

    @dx.validates("max_depth")
    @classmethod
    def _depth_is_positive(cls, value: int) -> int:
        """Validate the generated-term depth bound."""
        return ensure_positive("a generated-term depth", value)

    @dx.validates("constructor_weights")
    @classmethod
    def _constructor_weights_are_nonnegative(
        cls, values: dict[str, float]
    ) -> dict[str, float]:
        """Validate finite nonnegative constructor weights."""
        checked: dict[str, float] = {}

        for name, value in values.items():
            held_name = ensure_name("a constructor weight", name)
            held = ensure_finite(f"constructor_weights[{name!r}]", value)

            if held < 0.0:
                msg = f"constructor_weights[{name!r}] is negative: {value!r}"
                raise ValueError(msg)

            checked[held_name] = held

        return checked

    @dx.model_validator()
    def _saturation_is_a_fraction(self) -> Self:
        """Require the teacher-forcing saturation to lie in (0, 1]."""
        saturation = self.teacher_forcing_saturation

        if saturation is None:
            return self

        checked = ensure_finite("teacher_forcing_saturation", saturation)

        if not 0.0 < checked <= 1.0:
            msg = f"teacher_forcing_saturation is outside (0, 1]: {saturation!r}"
            raise ValueError(msg)

        return self


def autoencoding_objective(space: str, /) -> str:
    """Return the name a represented space's term autoencoding is reported under.

    Parameters
    ----------
    space : str
        The represented space's role.
    """
    return f"{space}/autoencoding"


def combinator_objective(space: str, /) -> str:
    """Return the name a represented space's combinator decoding is reported under.

    Parameters
    ----------
    space : str
        The represented space's role.
    """
    return f"{space}/combinators"


def terms_metrics(trained: Iterable[tuple[str, bool]], /) -> frozenset[str]:
    """Return the held-out metrics term pretraining evaluates.

    Parameters
    ----------
    trained : Iterable[tuple[str, bool]]
        Each trained represented space, beside whether its combinator decoders
        are trained with it.
    """
    metrics: set[str] = set()

    for space, combinators in trained:
        metrics.add(f"loss/{autoencoding_objective(space)}/exact_match")

        if combinators:
            metrics.add(f"loss/{combinator_objective(space)}/exact_match")

    return frozenset(metrics)
