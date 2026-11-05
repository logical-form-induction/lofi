"""Open declarations for synthetic corruption and ill-typing operators."""

from typing import TYPE_CHECKING, Literal, Protocol, cast

import didactic.api as dx

from lofi._checks import ensure_finite
from lofi._frozen import FrozenValue
from lofi.data.synthetic._values import WeightedOperator
from lofi.registry import resolve_plugin

if TYPE_CHECKING:
    from lofi.data.synthetic._corrupt import CorruptionContext, CorruptionOperator
    from lofi.data.synthetic._injection import IllTypingContext, IllTypingOperator

__all__ = [
    "ArgumentSwapSpec",
    "CorruptionOperatorFactory",
    "CorruptionOperatorPlugin",
    "CorruptionOperatorSpec",
    "DeletionSpec",
    "IllTypingOperatorPlugin",
    "IllTypingOperatorSpec",
    "InsertionSpec",
    "OverApplicationSpec",
    "SortMismatchSpec",
    "SubstitutionSpec",
    "SwapSpec",
    "WeightedCorruptionOperatorFactory",
    "WeightedCorruptionOperatorSpec",
    "WeightedIllTypingOperatorSpec",
]


class CorruptionOperatorSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """One installed token-sequence corruption operator."""

    edits: int = 1

    __axioms__ = (dx.axiom("edits > 0", message="edits must be positive"),)

    def compile(self) -> CorruptionOperatorFactory:
        """Discard this declaration into a context-dependent factory."""
        return corruption_operator_plugin(self).compile(self)

    def build(self, context: CorruptionContext, /) -> CorruptionOperator:
        """Build the selected installed operator in a corpus context."""
        return self.compile()(context)


class SwapSpec(CorruptionOperatorSpec, extra="forbid"):
    """Exchange adjacent distinct forms."""

    kind: Literal["swap"] = "swap"


class DeletionSpec(CorruptionOperatorSpec, extra="forbid"):
    """Delete forms from a string."""

    kind: Literal["deletion"] = "deletion"


class InsertionSpec(CorruptionOperatorSpec, extra="forbid"):
    """Insert forms drawn from the declared lexicon."""

    kind: Literal["insertion"] = "insertion"


class SubstitutionSpec(CorruptionOperatorSpec, extra="forbid"):
    """Replace forms by category-disjoint lexicon forms."""

    kind: Literal["substitution"] = "substitution"


class IllTypingOperatorSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """One installed operator that makes a logical form fail type checking."""

    edits: int = 1

    __axioms__ = (dx.axiom("edits > 0", message="edits must be positive"),)

    def build(self, context: IllTypingContext, /) -> IllTypingOperator:
        """Build the selected installed operator in an interpretation context."""
        return ill_typing_operator_plugin(self).construct(self, context)


class ArgumentSwapSpec(IllTypingOperatorSpec, extra="forbid"):
    """Exchange the function and argument of applications."""

    kind: Literal["argument-swap"] = "argument-swap"


class OverApplicationSpec(IllTypingOperatorSpec, extra="forbid"):
    """Apply saturated applications to a further argument."""

    kind: Literal["over-application"] = "over-application"


class SortMismatchSpec(IllTypingOperatorSpec, extra="forbid"):
    """Rebind a node at a variable of an incompatible sort."""

    kind: Literal["sort-mismatch"] = "sort-mismatch"


class WeightedCorruptionOperatorSpec(dx.Model, extra="forbid"):
    """A corruption operator declaration and its relative draw weight."""

    operator: CorruptionOperatorSpec
    weight: float = 1.0

    @dx.validates("weight")
    @classmethod
    def _positive_weight(cls, weight: float) -> float:
        """Return a finite positive draw weight."""
        held = ensure_finite("a corruption operator weight", weight)
        if held <= 0.0:
            msg = "a corruption operator weight must be positive"
            raise ValueError(msg)
        return held

    def compile(self) -> WeightedCorruptionOperatorFactory:
        """Discard this declaration into a weighted context factory."""
        return WeightedCorruptionOperatorFactory(self.operator.compile(), self.weight)

    def build(
        self, context: CorruptionContext, /
    ) -> WeightedOperator[CorruptionOperator]:
        """Build the weighted installed operator."""
        return self.compile().build(context)


class WeightedIllTypingOperatorSpec(dx.Model, extra="forbid"):
    """An ill-typing operator declaration and its relative draw weight."""

    operator: IllTypingOperatorSpec
    weight: float = 1.0

    @dx.validates("weight")
    @classmethod
    def _positive_weight(cls, weight: float) -> float:
        """Return a finite positive draw weight."""
        held = ensure_finite("an ill-typing operator weight", weight)
        if held <= 0.0:
            msg = "an ill-typing operator weight must be positive"
            raise ValueError(msg)
        return held

    def build(
        self, context: IllTypingContext, /
    ) -> WeightedOperator[IllTypingOperator]:
        """Build the weighted installed operator."""
        return WeightedOperator(self.operator.build(context), self.weight)


class CorruptionOperatorFactory(Protocol):
    """Build one token corruption operator in an empirical corpus context."""

    def __call__(self, context: CorruptionContext, /) -> CorruptionOperator:
        """Build the operator."""
        ...


class CorruptionOperatorCompiler[Declaration: CorruptionOperatorSpec](Protocol):
    """Discard one validated declaration into a context factory."""

    def __call__(self, declaration: Declaration, /) -> CorruptionOperatorFactory:
        """Compile the operator factory."""
        ...


class CorruptionOperatorPlugin[Declaration: CorruptionOperatorSpec](FrozenValue):
    """Pair one open corruption declaration with its interior compiler."""

    _noun = "corruption-operator plugin"
    __slots__ = ("_compile", "_spec")

    def __init__(
        self,
        *,
        spec: type[Declaration],
        compiler: CorruptionOperatorCompiler[Declaration],
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_compile", compiler)

    @property
    def spec(self) -> type[Declaration]:
        """Return the declaration class this plugin builds."""
        return self._spec

    def compile(self, declaration: Declaration, /) -> CorruptionOperatorFactory:
        """Discard one validated declaration into an interior factory."""
        return self._compile(declaration)


class WeightedCorruptionOperatorFactory(FrozenValue):
    """A weighted operator factory containing no boundary declaration."""

    _noun = "weighted corruption-operator factory"
    __slots__ = ("_factory", "_weight")

    def __init__(self, factory: CorruptionOperatorFactory, weight: float) -> None:
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_weight", ensure_finite("weight", weight))

    def build(
        self, context: CorruptionContext, /
    ) -> WeightedOperator[CorruptionOperator]:
        """Build the weighted operator in an empirical corpus context."""
        return WeightedOperator(self._factory(context), self._weight)


class IllTypingOperatorBuilder[Declaration: IllTypingOperatorSpec](Protocol):
    """Build one ill-typing operator from its declaration and context."""

    def __call__(
        self, declaration: Declaration, context: IllTypingContext, /
    ) -> IllTypingOperator:
        """Build the operator."""
        ...


class IllTypingOperatorPlugin[Declaration: IllTypingOperatorSpec](FrozenValue):
    """Pair one open ill-typing declaration with its runtime builder."""

    _noun = "ill-typing-operator plugin"
    __slots__ = ("_build", "_spec")

    def __init__(
        self,
        *,
        spec: type[Declaration],
        build: IllTypingOperatorBuilder[Declaration],
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_build", build)

    @property
    def spec(self) -> type[Declaration]:
        """Return the declaration class this plugin builds."""
        return self._spec

    def construct(
        self, declaration: Declaration, context: IllTypingContext, /
    ) -> IllTypingOperator:
        """Build one operator from a validated declaration."""
        return self._build(declaration, context)


def corruption_operator_plugin(
    declaration: CorruptionOperatorSpec, /
) -> CorruptionOperatorPlugin[CorruptionOperatorSpec]:
    """Return the installed corruption plugin selected by a declaration."""
    return cast(
        "CorruptionOperatorPlugin[CorruptionOperatorSpec]",
        resolve_plugin(
            "corruption",
            cast("str", declaration.kind),
            CorruptionOperatorPlugin,
            type(declaration),
        ),
    )


def ill_typing_operator_plugin(
    declaration: IllTypingOperatorSpec, /
) -> IllTypingOperatorPlugin[IllTypingOperatorSpec]:
    """Return the installed ill-typing plugin selected by a declaration."""
    return cast(
        "IllTypingOperatorPlugin[IllTypingOperatorSpec]",
        resolve_plugin(
            "ill_typing",
            cast("str", declaration.kind),
            IllTypingOperatorPlugin,
            type(declaration),
        ),
    )
