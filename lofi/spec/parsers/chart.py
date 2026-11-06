"""Differentiable chart parser declaration."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Self, cast, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_positive
from lofi._types import ModuleSharing, SubmoduleDeclaration
from lofi.spec._ports import Dependency
from lofi.spec.chart_compositions import (
    ChartCompositionDeclaration,
    ChartCompositionSpec,
    DeclaredChartComposition,
    default_chart_composition,
)
from lofi.spec.components.model import ParserSpec
from lofi.spec.parsers.reduction import (
    _DEFAULT_REDUCTION,
    ReductionDeclaration,
    ReductionSettings,
    ReductionSpec,
)
from lofi.spec.parsers.split import (
    _DEFAULT_SPLIT_WEIGHER,
    DeclaredSplitWeigher,
    SplitWeigherDeclaration,
    SplitWeigherSpec,
)
from lofi.spec.span_functions import (
    DeclaredSpanFunction,
    SpanFunctionDeclaration,
    SpanFunctionSpec,
    default_span_function,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "InsideOutsideParserSettings",
    "InsideOutsideParserSpec",
    "OutsideWeights",
    "RootOutside",
]

type OutsideWeights = Literal["reuse", "contextual"]
"""How the vector outside pass weighs a span's possible contexts."""

type RootOutside = Literal["zero", "learned"]
"""The boundary vector placed at each item's root span."""

_DEPENDENCIES: Final[tuple[Dependency, ...]] = (
    Dependency("subword_pooler", "leaves"),
    Dependency("controller", "rules"),
)
"""The pooled leaves the chart reads and the controller that weighs its actions."""

_DEFAULT_COMPOSITION: Final = dx.field(default_factory=default_chart_composition)
"""The shared composition field of neural chart declarations."""

_DEFAULT_SPLIT_FUNCTION: Final = dx.field(default_factory=default_span_function)
"""The shared child-projection field of neural chart declarations."""

_DEFAULT_SHARE_SPLITS: Final = False
"""Whether neural charts share their child projections by default."""

_DEFAULT_OUTSIDE_WEIGHTS: Final[OutsideWeights] = "contextual"
"""How neural charts weight their outside pass by default."""

_DEFAULT_ROOT_OUTSIDE: Final[RootOutside] = "learned"
"""The outside state neural charts start their root from by default."""

_DEFAULT_ROOT_INIT_GAIN: Final = 1.0
"""The gain a learned root's draw has over one over the root width's square root."""


def root_init_gain(value: float | None, /) -> float | None:
    """Return a finite, positive learned-root initialization gain, or None.

    Raises
    ------
    ValueError
        If the gain is not finite or not above zero.
    """
    if value is None:
        return None

    held = ensure_finite("a root initialization gain", value)

    if held <= 0.0:
        msg = f"a root initialization gain is above zero, not {value!r}"
        raise ValueError(msg)

    return held


class InsideOutsideParserSettings(NamedTuple):
    """Resolved settings of an inside-outside chart."""

    width: int
    composition: ChartCompositionDeclaration
    reduction: ReductionSettings
    attend: SplitWeigherDeclaration
    split_left: SpanFunctionDeclaration
    split_right: SpanFunctionDeclaration
    share_splits: bool
    transition: SpanFunctionDeclaration
    depth: int
    layer_sharing: Literal["independent", "shared"]
    outside_weights: OutsideWeights
    root_outside: RootOutside
    root_init_gain: float


class InsideOutsideParserSpec(ParserSpec):
    """Declares a chart scored by an inside and an outside sweep.

    The declared width owns the chart cell representation. Pooled leaves enter
    through the selected composition's leaf function, and the controller wired to
    ``rules`` must consume that same width.

    Attributes
    ----------
    kind : Literal["inside-outside"]
        Registry tag.
    """

    kind: Literal["inside-outside"] = "inside-outside"
    width: int
    composition: ChartCompositionSpec = _DEFAULT_COMPOSITION
    reduction: ReductionSpec = _DEFAULT_REDUCTION
    attend: SplitWeigherSpec = _DEFAULT_SPLIT_WEIGHER
    split_left: SpanFunctionSpec = _DEFAULT_SPLIT_FUNCTION
    split_right: SpanFunctionSpec = _DEFAULT_SPLIT_FUNCTION
    share_splits: bool = _DEFAULT_SHARE_SPLITS
    transition: SpanFunctionSpec = dx.field(default_factory=default_span_function)
    depth: int = 1
    layer_sharing: Literal["independent", "shared"] = "independent"
    outside_weights: OutsideWeights = _DEFAULT_OUTSIDE_WEIGHTS
    root_outside: RootOutside = _DEFAULT_ROOT_OUTSIDE
    root_init_gain: float | None = None

    @dx.validates("root_init_gain")
    @classmethod
    def _root_gain_is_positive(cls, value: float | None) -> float | None:
        """Return a finite, positive learned-root initialization gain."""
        return root_init_gain(value)

    @dx.model_validator()
    def _shared_split_declarations_agree(self) -> Self:
        """Return a coherent declaration of split and recurrent chart layers."""
        if self.root_init_gain is not None and self.root_outside != "learned":
            msg = "a root initialization gain is declared only for a learned root"
            raise ValueError(msg)

        if self.share_splits and self.split_left != self.split_right:
            msg = "shared split slots must declare the same span function"
            raise ValueError(msg)

        if self.depth > 1 and not self.reduction.probabilistic:
            msg = "a stacked neural chart requires log-sum-exp inside-outside readouts"
            raise ValueError(msg)

        return self

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the pooled leaves and the action weights this chart reads."""
        return _DEPENDENCIES

    @override
    def sharing(self) -> Mapping[str, SubmoduleDeclaration]:
        """Return the controller this chart weighs actions with and saves with it."""
        return MappingProxyType(
            {
                "rules": SubmoduleDeclaration(
                    "policy", ModuleSharing(in_parameters=True, in_state_dict=True)
                )
            }
        )

    @override
    def publications(self) -> frozenset[str]:
        """Return the chart cell combination represented spaces may read."""
        return frozenset({"combine"})

    def chart_outputs(self) -> frozenset[str]:
        """Return the runtime chart readouts this declaration provides."""
        outputs = {"inside", "log_z", "identity", "cells", "actions"}

        if self.reduction.probabilistic:
            outputs.update({"outside", "alpha", "mu", "spans"})

            if self.outside_weights == "contextual":
                outputs.add("outside_cells")

        return frozenset(outputs)

    @override
    def controller_inputs(self) -> frozenset[str]:
        """Return the controller representations available under this chart."""
        available = {"inside"}

        if self.attend.reads_children and self.reduction.probabilistic:
            available.add("span")

        return frozenset(available)

    def settings(self) -> InsideOutsideParserSettings:
        """Return the settings the plugin builds this component from."""
        reduction = cast("ReductionDeclaration", self.reduction)
        attend = cast("DeclaredSplitWeigher", self.attend)
        composition = cast("DeclaredChartComposition", self.composition)
        split_left = cast("DeclaredSpanFunction", self.split_left)
        split_right = cast("DeclaredSpanFunction", self.split_right)
        transition = cast("DeclaredSpanFunction", self.transition)

        return InsideOutsideParserSettings(
            self.width,
            composition.declaration(),
            reduction.settings(),
            attend.declaration(),
            split_left.declaration(),
            split_right.declaration(),
            self.share_splits,
            transition.declaration(),
            self.depth,
            self.layer_sharing,
            self.outside_weights,
            self.root_outside,
            _DEFAULT_ROOT_INIT_GAIN
            if self.root_init_gain is None
            else self.root_init_gain,
        )

    @dx.validates("width")
    @classmethod
    def _width_is_positive(cls, value: int) -> int:
        """Return a valid chart cell width."""
        return ensure_positive("a chart's node width", value)

    @dx.validates("depth")
    @classmethod
    def _depth_is_positive(cls, value: int) -> int:
        """Return a valid recurrent chart depth."""
        return ensure_positive("a chart's depth", value)
