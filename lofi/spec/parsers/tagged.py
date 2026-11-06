"""Normal-form-refined chart parser declaration."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Self, cast, override

import didactic.api as dx

from lofi._checks import ensure_non_negative, ensure_positive
from lofi._types import ModuleSharing, SubmoduleDeclaration
from lofi.spec._ports import Dependency
from lofi.spec.chart_compositions import (
    ChartCompositionDeclaration,
    ChartCompositionSpec,
    DeclaredChartComposition,
)
from lofi.spec.components.model import ParserSpec
from lofi.spec.parsers.chart import (
    _DEFAULT_COMPOSITION,
    _DEFAULT_OUTSIDE_WEIGHTS,
    _DEFAULT_ROOT_INIT_GAIN,
    _DEFAULT_ROOT_OUTSIDE,
    _DEFAULT_SHARE_SPLITS,
    _DEFAULT_SPLIT_FUNCTION,
    OutsideWeights,
    RootOutside,
    root_init_gain,
)
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
)

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["TaggedParserSettings", "TaggedParserSpec"]

_DEPENDENCIES: Final[tuple[Dependency, ...]] = (
    Dependency("subword_pooler", "leaves"),
    Dependency("controller", "rules"),
)
"""The pooled leaves and controller action weights the tagged chart reads."""


class TaggedParserSettings(NamedTuple):
    """Resolved settings of a normal-form-refined chart."""

    width: int
    composition: ChartCompositionDeclaration
    reduction: ReductionSettings
    attend: SplitWeigherDeclaration
    unary_depth: int
    k_best: int
    split_left: SpanFunctionDeclaration
    split_right: SpanFunctionDeclaration
    share_splits: bool
    outside_weights: OutsideWeights
    root_outside: RootOutside
    root_init_gain: float


class TaggedParserSpec(ParserSpec):
    """Declare a category- and normal-form-refined neural chart."""

    kind: Literal["tagged"] = "tagged"
    width: int
    composition: ChartCompositionSpec = _DEFAULT_COMPOSITION
    reduction: ReductionSpec = _DEFAULT_REDUCTION
    attend: SplitWeigherSpec = _DEFAULT_SPLIT_WEIGHER
    unary_depth: int
    k_best: int = 1
    split_left: SpanFunctionSpec = _DEFAULT_SPLIT_FUNCTION
    split_right: SpanFunctionSpec = _DEFAULT_SPLIT_FUNCTION
    share_splits: bool = _DEFAULT_SHARE_SPLITS
    outside_weights: OutsideWeights = _DEFAULT_OUTSIDE_WEIGHTS
    root_outside: RootOutside = _DEFAULT_ROOT_OUTSIDE
    root_init_gain: float | None = None

    @dx.validates("unary_depth")
    @classmethod
    def _depth_is_non_negative(cls, value: int) -> int:
        """Return a valid number of unary closure rounds."""
        return ensure_non_negative("a tagged chart's unary depth", value)

    @dx.validates("k_best")
    @classmethod
    def _k_is_positive(cls, value: int) -> int:
        """Return a valid number of derivations to retain."""
        return ensure_positive("a tagged chart's k-best width", value)

    @dx.validates("root_init_gain")
    @classmethod
    def _root_gain_is_positive(cls, value: float | None) -> float | None:
        """Return a finite, positive learned-root initialization gain."""
        return root_init_gain(value)

    @dx.model_validator()
    def _shared_split_declarations_agree(self) -> Self:
        """Return a declaration whose shared split slots specify one function."""
        if self.root_init_gain is not None and self.root_outside != "learned":
            msg = "a root initialization gain is declared only for a learned root"
            raise ValueError(msg)

        if self.share_splits and self.split_left != self.split_right:
            msg = "shared split slots must declare the same span function"
            raise ValueError(msg)

        return self

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the pooled leaves and action weights this chart reads."""
        return _DEPENDENCIES

    @override
    def sharing(self) -> Mapping[str, SubmoduleDeclaration]:
        """Return the controller this chart trains and saves with itself."""
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
            outputs.update({"alpha", "mu", "spans"})

            if self.outside_weights == "contextual":
                outputs.add("outside_cells")

        return frozenset(outputs)

    @override
    def controller_inputs(self) -> frozenset[str]:
        """Return the inside representation consumed during tagged search."""
        return frozenset({"inside"})

    def unary_closure_depth(self) -> int:
        """Return the unary rule rounds applied within each chart cell."""
        return self.unary_depth

    def category_transition_reachability(self) -> bool:
        """Return that chart categories follow compiled grammar transitions."""
        return True

    def settings(self) -> TaggedParserSettings:
        """Return the settings the plugin builds this component from."""
        reduction = cast("ReductionDeclaration", self.reduction)
        attend = cast("DeclaredSplitWeigher", self.attend)
        composition = cast("DeclaredChartComposition", self.composition)
        split_left = cast("DeclaredSpanFunction", self.split_left)
        split_right = cast("DeclaredSpanFunction", self.split_right)

        return TaggedParserSettings(
            self.width,
            composition.declaration(),
            reduction.settings(),
            attend.declaration(),
            unary_depth=self.unary_depth,
            k_best=self.k_best,
            split_left=split_left.declaration(),
            split_right=split_right.declaration(),
            share_splits=self.share_splits,
            outside_weights=self.outside_weights,
            root_outside=self.root_outside,
            root_init_gain=_DEFAULT_ROOT_INIT_GAIN
            if self.root_init_gain is None
            else self.root_init_gain,
        )

    @dx.validates("width")
    @classmethod
    def _width_is_positive(cls, value: int) -> int:
        """Return a valid chart cell width."""
        return ensure_positive("a tagged chart's node width", value)
