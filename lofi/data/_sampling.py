"""Values shared by synthetic type and rule-pair samplers."""

from math import exp, isfinite, log
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, override

from lofi._checks import ensure_non_negative, ensure_positive
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.terms import Term

__all__ = [
    "ArityBranching",
    "ChildDepthDistribution",
    "ExhaustiveWeighting",
    "PairDistribution",
    "PairSampleReport",
    "PowerChildDepth",
    "SampledPair",
    "SamplingExhaustedError",
    "SizeExhaustiveWeighting",
    "TypeDistribution",
    "TypeSampleReport",
    "UniformChildDepth",
    "UniformExhaustiveWeighting",
    "term_depth",
]


class ChildDepthDistribution(Protocol):
    """Supply relative mass over the feasible depths of one child."""

    def weights(self, depths: Sequence[int], /) -> tuple[float, ...]:
        """Return one nonnegative relative weight per feasible depth."""
        ...


class ExhaustiveWeighting(Protocol):
    """Supply relative mass for one member of an exhaustive inventory."""

    def weights(self, terms: Sequence[Term], /) -> tuple[float, ...]:
        """Return one nonnegative relative weight per candidate."""
        ...


class UniformChildDepth(FrozenValue):
    """Give every feasible child depth equal mass."""

    _noun = "child-depth distribution"
    __slots__ = ()

    def weights(self, depths: Sequence[int], /) -> tuple[float, ...]:
        """Return equal mass for every feasible child depth."""
        return (1.0,) * len(depths)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether the other value is also uniform."""
        return type(other) is UniformChildDepth

    @override
    def __hash__(self) -> int:
        """Return the uniform distribution's stable hash."""
        return hash(UniformChildDepth)


class PowerChildDepth(FrozenValue):
    """Weight each feasible child depth by a declared power.

    Positive exponents favor deeper children, negative exponents favor
    shallower children, and zero assigns equal mass.
    """

    _noun = "child-depth distribution"
    __slots__ = ("_exponent",)

    def __init__(self, exponent: float) -> None:
        if isinstance(exponent, bool) or not isinstance(exponent, int | float):
            msg = f"a child-depth exponent is not numeric: {exponent!r}"
            raise TypeError(msg)

        held = float(exponent)

        if not isfinite(held):
            msg = f"a child-depth exponent is not finite: {exponent!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_exponent", held)

    @property
    def exponent(self) -> float:
        """Return the power applied to each positive depth."""
        return self._exponent

    def weights(self, depths: Sequence[int], /) -> tuple[float, ...]:
        """Return numerically stable power weights for positive depths."""
        held = tuple(
            ensure_positive("a feasible child depth", depth) for depth in depths
        )

        if not held:
            return ()

        logits = tuple(self._exponent * log(depth) for depth in held)
        peak = max(logits)

        return tuple(exp(logit - peak) for logit in logits)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether the other distribution has the same exponent."""
        return type(other) is PowerChildDepth and self._exponent == other._exponent

    @override
    def __hash__(self) -> int:
        """Return a hash over the declared exponent."""
        return hash((PowerChildDepth, self._exponent))


class ArityBranching(FrozenValue):
    """Relative deepest-child weights keyed by constructor arity.

    An omitted arity receives uniform weights. Unary constructors have only
    one possible deepest child and therefore need no declaration.
    """

    _noun = "arity branching distribution"
    __slots__ = ("_by_arity",)

    def __init__(self, by_arity: Mapping[int, Sequence[float]]) -> None:
        checked: dict[int, tuple[float, ...]] = {}

        for arity, weights in by_arity.items():
            held_arity = ensure_positive("a branching arity", arity)

            if held_arity < 2:
                msg = f"a branching arity is less than two: {arity!r}"
                raise ValueError(msg)

            held_weights = tuple(
                _nonnegative_weight(f"branching[{held_arity}][{at}]", weight)
                for at, weight in enumerate(weights)
            )

            if len(held_weights) != held_arity:
                msg = (
                    f"branching arity {held_arity} has {len(held_weights)} "
                    "deepest-child weights"
                )
                raise ValueError(msg)

            if sum(held_weights) <= 0.0:
                msg = f"branching arity {held_arity} has no positive mass"
                raise ValueError(msg)

            checked[held_arity] = held_weights

        object.__setattr__(self, "_by_arity", MappingProxyType(checked))

    @property
    def by_arity(self) -> Mapping[int, tuple[float, ...]]:
        """Return the copied declared deepest-child weights."""
        return self._by_arity

    def weights(self, arity: int, /) -> tuple[float, ...]:
        """Return declared or uniform weights for one branching arity."""
        held = ensure_positive("a constructor arity", arity)

        if held == 1:
            return (1.0,)

        return self._by_arity.get(held, (1.0,) * held)

    @property
    def _fields(self) -> tuple[tuple[int, tuple[float, ...]], ...]:
        """Return stable fields for equality and hashing."""
        return tuple(sorted(self._by_arity.items()))

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether every per-arity declaration matches."""
        return type(other) is ArityBranching and self._fields == other._fields

    @override
    def __hash__(self) -> int:
        """Return a hash over every per-arity declaration."""
        return hash((ArityBranching, self._fields))


class UniformExhaustiveWeighting(FrozenValue):
    """Give every member of an exhaustive inventory equal mass."""

    _noun = "exhaustive weighting"
    __slots__ = ()

    def weights(self, terms: Sequence[Term], /) -> tuple[float, ...]:
        """Return unit mass for every candidate."""
        return (1.0,) * len(terms)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether the other weighting is also uniform."""
        return type(other) is UniformExhaustiveWeighting

    @override
    def __hash__(self) -> int:
        """Return the uniform weighting's stable hash."""
        return hash(UniformExhaustiveWeighting)


class SizeExhaustiveWeighting(FrozenValue):
    """Weight exhaustive candidates by a power of their node count."""

    _noun = "exhaustive weighting"
    __slots__ = ("_exponent",)

    def __init__(self, exponent: float = 1.0) -> None:
        if isinstance(exponent, bool) or not isinstance(exponent, int | float):
            msg = f"an exhaustive size exponent is not numeric: {exponent!r}"
            raise TypeError(msg)

        held = float(exponent)

        if not isfinite(held):
            msg = f"an exhaustive size exponent is not finite: {exponent!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_exponent", held)

    @property
    def exponent(self) -> float:
        """Return the power applied to candidate size."""
        return self._exponent

    def weights(self, terms: Sequence[Term], /) -> tuple[float, ...]:
        """Return stable size-power mass for every candidate."""
        if not terms:
            return ()

        logits = tuple(self._exponent * log(term.size) for term in terms)
        peak = max(logits)

        return tuple(exp(logit - peak) for logit in logits)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether the other weighting has the same exponent."""
        return (
            type(other) is SizeExhaustiveWeighting and self._exponent == other._exponent
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the declared exponent."""
        return hash((SizeExhaustiveWeighting, self._exponent))


_UNIFORM_BRANCHING: Final = ArityBranching({})
_UNIFORM_CHILD_DEPTH: Final = UniformChildDepth()


def _probability(name: str, value: float) -> float:
    """Validate a probability."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{name} is not numeric: {value!r}"
        raise TypeError(msg)

    held = float(value)

    if not isfinite(held) or not 0.0 <= held <= 1.0:
        msg = f"{name} is not between zero and one: {value!r}"
        raise ValueError(msg)

    return held


def _nonnegative_weight(name: str, value: float) -> float:
    """Validate a finite nonnegative relative weight."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{name} is not numeric: {value!r}"
        raise TypeError(msg)

    held = float(value)

    if not isfinite(held) or held < 0.0:
        msg = f"{name} is negative or nonfinite: {value!r}"
        raise ValueError(msg)

    return held


def _counts[K](name: str, values: Mapping[K, int]) -> Mapping[K, int]:
    """Copy and validate a count mapping."""
    return MappingProxyType(
        {
            key: ensure_non_negative(f"{name}[{key!r}]", count)
            for key, count in values.items()
        }
    )


def _rates[K](counts: Mapping[K, int]) -> Mapping[K, float]:
    """Convert counts to observed proportions."""
    total = sum(counts.values())

    if total == 0:
        return MappingProxyType(dict.fromkeys(counts, 0.0))

    return MappingProxyType({key: count / total for key, count in counts.items()})


class SamplingExhaustedError(RuntimeError):
    """Raised when a sampler cannot emit the requested number of values."""


class TypeDistribution(FrozenValue):
    """A bounded distribution over term types.

    Parameters
    ----------
    target_depth : int
        Preferred root-to-leaf depth.
    constructor_weights : Mapping[str, float]
        Nonnegative weights keyed by operator name.
    child_depth : ChildDepthDistribution
        Relative mass over the feasible depths of each nondeepest child.
    branching : ArityBranching
        Deepest-child weights keyed by constructor arity.
    bias_types : Sequence[Term]
        Interned terms that share the reserved bias mass.
    bias_mass : float
        Probability reserved for ``bias_types``.
    max_depth : int
        Hard depth bound.

    Raises
    ------
    TypeError
        If a depth is not an integer, a probability or weight is not numeric,
        or a constructor name is not text.
    ValueError
        If a depth is nonpositive, ``target_depth`` exceeds ``max_depth``, a
        probability is outside $[0, 1]$, a weight is negative or nonfinite,
        or positive ``bias_mass`` has no bias type.
    """

    _noun = "sampling value"

    __slots__ = (
        "_bias_mass",
        "_bias_types",
        "_branching",
        "_child_depth",
        "_constructor_weights",
        "_max_depth",
        "_target_depth",
    )

    def __init__(
        self,
        target_depth: int,
        constructor_weights: Mapping[str, float],
        bias_types: Sequence[Term] = (),
        *,
        bias_mass: float = 0.0,
        branching: ArityBranching = _UNIFORM_BRANCHING,
        child_depth: ChildDepthDistribution = _UNIFORM_CHILD_DEPTH,
        max_depth: int,
    ) -> None:
        held_target = ensure_positive("target_depth", target_depth)
        held_maximum = ensure_positive("max_depth", max_depth)

        if held_target > held_maximum:
            msg = f"target_depth {held_target} exceeds max_depth {held_maximum}"
            raise ValueError(msg)

        weights: dict[str, float] = {}

        for name, weight in constructor_weights.items():
            if not isinstance(name, str):
                msg = f"a constructor name is not a string: {name!r}"
                raise TypeError(msg)

            weights[name] = _nonnegative_weight(f"the weight for {name!r}", weight)

        held_bias_types = tuple(bias_types)
        held_bias_mass = _probability("bias_mass", bias_mass)

        if held_bias_mass > 0.0 and not held_bias_types:
            msg = "positive bias_mass requires at least one bias type"
            raise ValueError(msg)

        object.__setattr__(self, "_target_depth", held_target)
        object.__setattr__(self, "_child_depth", child_depth)
        object.__setattr__(self, "_branching", branching)
        object.__setattr__(self, "_constructor_weights", MappingProxyType(weights))
        object.__setattr__(self, "_bias_types", held_bias_types)
        object.__setattr__(self, "_bias_mass", held_bias_mass)
        object.__setattr__(self, "_max_depth", held_maximum)

    @property
    def target_depth(self) -> int:
        """Return the preferred depth."""
        return self._target_depth

    @property
    def child_depth(self) -> ChildDepthDistribution:
        """Return the child-depth distribution."""
        return self._child_depth

    @property
    def branching(self) -> ArityBranching:
        """Return deepest-child weights keyed by constructor arity."""
        return self._branching

    @property
    def constructor_weights(self) -> Mapping[str, float]:
        """Return the copied constructor weights."""
        return self._constructor_weights

    @property
    def bias_types(self) -> tuple[Term, ...]:
        """Return the reserved-mass types."""
        return self._bias_types

    @property
    def bias_mass(self) -> float:
        """Return the probability reserved for bias types."""
        return self._bias_mass

    @property
    def max_depth(self) -> int:
        """Return the hard depth bound."""
        return self._max_depth

    @property
    def _fields(self) -> tuple[object, ...]:
        """Return immutable fields for equality and hashing."""
        return (
            self._target_depth,
            self._child_depth,
            self._branching,
            tuple(sorted(self._constructor_weights.items())),
            self._bias_types,
            self._bias_mass,
            self._max_depth,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether every declared distribution field matches."""
        return type(other) is TypeDistribution and self._fields == other._fields

    @override
    def __hash__(self) -> int:
        """Return a hash over every declared distribution field."""
        return hash((TypeDistribution, self._fields))


class PairDistribution(FrozenValue):
    """A term-depth bound for sampled rule instances.

    Parameters
    ----------
    max_depth : int
        Hard depth bound for premises and conclusions.

    Raises
    ------
    TypeError
        If the bound is not an integer.
    ValueError
        If the depth bound is nonpositive.
    """

    _noun = "sampling value"

    __slots__ = ("_max_depth",)

    def __init__(
        self,
        *,
        max_depth: int,
    ) -> None:
        object.__setattr__(self, "_max_depth", ensure_positive("max_depth", max_depth))

    @property
    def max_depth(self) -> int:
        """Return the hard depth bound."""
        return self._max_depth

    @property
    def _fields(self) -> tuple[object, ...]:
        """Return immutable fields for equality and hashing."""
        return (self._max_depth,)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether every declared distribution field matches."""
        return type(other) is PairDistribution and self._fields == other._fields

    @override
    def __hash__(self) -> int:
        """Return a hash over every declared distribution field."""
        return hash((PairDistribution, self._fields))


class TypeSampleReport(FrozenValue):
    """Observed type-sampling counts and distributions.

    Parameters
    ----------
    draws : int
        Number of emitted types.
    depths : Mapping[int, int]
        Emitted root depths by count.
    constructors : Mapping[str, int]
        Emitted constructor occurrences by count.

    Raises
    ------
    TypeError
        If a draw or count is not an integer.
    ValueError
        If a draw or count is negative, or depth counts do not sum to draws.
    """

    _noun = "sampling value"

    __slots__ = ("_constructors", "_depths", "_draws")

    def __init__(
        self,
        draws: int,
        depths: Mapping[int, int],
        constructors: Mapping[str, int],
    ) -> None:
        object.__setattr__(self, "_draws", ensure_non_negative("draws", draws))
        object.__setattr__(self, "_depths", _counts("depths", depths))
        object.__setattr__(self, "_constructors", _counts("constructors", constructors))

        if sum(self._depths.values()) != self._draws:
            msg = "depth counts must sum to emitted types"
            raise ValueError(msg)

    @property
    def draws(self) -> int:
        """Return the number of emitted types."""
        return self._draws

    @property
    def depths(self) -> Mapping[int, int]:
        """Return emitted root depths by count."""
        return self._depths

    @property
    def constructors(self) -> Mapping[str, int]:
        """Return all emitted constructor occurrences by count."""
        return self._constructors

    @property
    def depth_distribution(self) -> Mapping[int, float]:
        """Return observed root-depth proportions."""
        return _rates(self.depths)

    @property
    def constructor_distribution(self) -> Mapping[str, float]:
        """Return observed constructor proportions."""
        return _rates(self.constructors)

    @property
    def _fields(self) -> tuple[object, ...]:
        """Return immutable fields for equality and hashing."""
        return (
            self._draws,
            tuple(sorted(self._depths.items())),
            tuple(sorted(self._constructors.items())),
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether every observed report field matches."""
        return type(other) is TypeSampleReport and self._fields == other._fields

    @override
    def __hash__(self) -> int:
        """Return a hash over every observed report field."""
        return hash((TypeSampleReport, self._fields))


class PairSampleReport(FrozenValue):
    """Observed rule-instance emissions by premise count.

    Parameters
    ----------
    emitted : int
        Number of emitted labeled pairs.
    arities : Mapping[int, int]
        Emitted rule arities by count.

    Raises
    ------
    TypeError
        If a count is not an integer.
    ValueError
        If a count is negative or arity counts do not sum to emissions.
    """

    _noun = "sampling value"

    __slots__ = ("_arities", "_emitted")

    def __init__(
        self,
        emitted: int,
        arities: Mapping[int, int],
    ) -> None:
        held_arities = {
            ensure_non_negative("a reported rule arity", arity): count
            for arity, count in arities.items()
        }
        object.__setattr__(self, "_emitted", ensure_non_negative("emitted", emitted))
        object.__setattr__(self, "_arities", _counts("arities", held_arities))

        if sum(self._arities.values()) != self._emitted:
            msg = "arity counts must sum to emitted rule instances"
            raise ValueError(msg)

    @property
    def emitted(self) -> int:
        """Return emitted labeled pairs."""
        return self._emitted

    @property
    def arities(self) -> Mapping[int, int]:
        """Return emitted rule arities by count."""
        return self._arities

    @property
    def arity_distribution(self) -> Mapping[int, float]:
        """Return observed rule-arity proportions."""
        return _rates(self.arities)

    @property
    def _fields(self) -> tuple[object, ...]:
        """Return immutable fields for equality and hashing."""
        return (
            self._emitted,
            tuple(sorted(self._arities.items())),
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether every observed report field matches."""
        return type(other) is PairSampleReport and self._fields == other._fields

    @override
    def __hash__(self) -> int:
        """Return a hash over every observed report field."""
        return hash((PairSampleReport, self._fields))


class SampledPair[T, Label](NamedTuple):
    """A labeled rule instance of arbitrary arity.

    Attributes
    ----------
    inputs : tuple[T, ...]
        The premise values in rule order.
    output : T
        The conclusion value.
    actions : tuple[Label, ...]
        All labels whose rules admit the instance.
    """

    inputs: tuple[T, ...]
    output: T
    actions: tuple[Label, ...]


def term_depth(term: Term, /) -> int:
    """Return a term's root-to-leaf depth."""
    return 1 + max((term_depth(child) for child in term.args), default=0)
