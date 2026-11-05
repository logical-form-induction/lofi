"""Values a synthetic corpus is built from and reported with."""

from collections import Counter
from hashlib import sha256
from math import isclose, isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, final, override

from lofi._checks import (
    ensure_finite,
    ensure_name,
    ensure_non_negative,
    ensure_positive,
)
from lofi._frozen import FrozenValue
from lofi.data._sampling import SamplingExhaustedError
from lofi.data._supervision import Categorical, Example
from lofi.data.synthetic._frequency import log_log_slope, spearman
from lofi.data.synthetic._illtyped import (
    FORM_TYPING_LABELS,
    FormTypingExample,
    FormTypingLabel,
    IllTyped,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from random import Random

    from lofi._types import ItemId
    from lofi.data._supervision import GoldDerivation
    from lofi.data._values import Item
    from lofi.data.synthetic._corrupt import CorruptionOperator
    from lofi.data.synthetic._entailment import GoldForm
    from lofi.data.synthetic._injection import IllTypingOperator
    from lofi.data.synthetic._interpretation import InterpretationFunction
    from lofi.data.synthetic._lengths import LengthDistribution
    from lofi.data.synthetic._scope import UndecidedScopePolicy
    from lofi.data.synthetic._semantics import SemanticReport
    from lofi.spec.grammar import GrammarAssembly
    from lofi.typecheck import TypingFailureKind

__all__ = [
    "GRAMMATICALITY_LABELS",
    "NEGATIVE_DATA_REGIMES",
    "AmbiguityReport",
    "AmbiguityTarget",
    "AmbiguityUnreachableError",
    "Corrupted",
    "Corruption",
    "CorruptionDraw",
    "CorruptionExhaustedError",
    "CorruptionReport",
    "GrammarKnobs",
    "GrammaticalityLabel",
    "Histogram",
    "IllTypedFormError",
    "IllTypingReport",
    "InventoryReport",
    "LabelNoise",
    "LengthReport",
    "LexiconReport",
    "NegativeDataPolicy",
    "NegativeDataRegime",
    "NoiseReport",
    "NonDerivability",
    "ObservedLabel",
    "ReadingCount",
    "ScopeReport",
    "ScopeTarget",
    "ScopeUnreachableError",
    "ShapeMeasurement",
    "ShapeReport",
    "ShapeSchedule",
    "ShapeTarget",
    "ShapeUnreachableError",
    "SymbolInventory",
    "SyntheticCorpus",
    "SyntheticDataset",
    "Target",
    "UncertifiedDerivationError",
    "UninterpretableDerivationError",
    "UnproductiveGrammarError",
    "WeightedOperator",
]


class UnproductiveGrammarError(SamplingExhaustedError):
    """Raised when no goal category is derivable within the length bounds."""


class UncertifiedDerivationError(RuntimeError):
    """Raised when the parser does not rebuild a sampled derivation."""


class AmbiguityUnreachableError(SamplingExhaustedError):
    """Raised when the draws do not reach an ambiguity target within budget."""


class ScopeUnreachableError(SamplingExhaustedError):
    """Raised when the draws do not reach a readings target within budget."""


class ShapeUnreachableError(SamplingExhaustedError):
    """Raised when the draws do not reach a derivation-shape target."""


class CorruptionExhaustedError(SamplingExhaustedError):
    """Raised when no corruption within budget yields a non-derivable string."""


class UninterpretableDerivationError(RuntimeError):
    """Raised when the declared interpretation assigns a derivation no form."""


class IllTypedFormError(RuntimeError):
    """Raised when a gold logical form does not type check as a formula."""


type GrammaticalityLabel = Literal["grammatical", "ungrammatical"]
"""Whether the parser derives a string under the grammar it was drawn from."""

type NegativeDataRegime = Literal["labeled-negatives", "positive-only", "label-flips"]
"""How a certified negative is presented to a learner."""

NEGATIVE_DATA_REGIMES: Final[tuple[NegativeDataRegime, ...]] = (
    "labeled-negatives",
    "positive-only",
    "label-flips",
)
"""The negative-data regimes in declaration order."""

GRAMMATICALITY_LABELS: Final[tuple[GrammaticalityLabel, ...]] = (
    "grammatical",
    "ungrammatical",
)
"""The grammaticality labels in report order."""

_LABEL_SET: Final = frozenset(GRAMMATICALITY_LABELS)
"""The label inventory an observed grammaticality response is chosen from."""


def _proportion(field: str, value: float) -> float:
    """Return a number between zero and one inclusive."""
    held = ensure_finite(field, value)

    if not 0.0 <= held <= 1.0:
        msg = f"{field} is not between zero and one: {value!r}"
        raise ValueError(msg)

    return held


class NegativeDataPolicy(FrozenValue):
    """How certified negatives are observed, including stochastic label flips.

    Parameters
    ----------
    regime : NegativeDataRegime
        Whether negatives retain their label, enter positive-only data, or
        undergo independent label flips.
    noise : float
        The flip probability in the ``label-flips`` regime. The fixed-label
        regimes require zero.
    """

    _noun = "negative-data policy"

    __slots__ = ("_noise", "_regime")

    def __init__(self, regime: NegativeDataRegime, *, noise: float = 0.0) -> None:
        if regime not in NEGATIVE_DATA_REGIMES:
            msg = f"an unknown negative-data regime was declared: {regime!r}"
            raise ValueError(msg)

        held_noise = _proportion("a negative-data noise rate", noise)

        if regime != "label-flips" and held_noise != 0.0:
            msg = f"negative-data regime {regime!r} does not take label noise"
            raise ValueError(msg)

        object.__setattr__(self, "_regime", regime)
        object.__setattr__(self, "_noise", held_noise)

    @property
    def regime(self) -> NegativeDataRegime:
        """Return how certified negatives are presented."""
        return self._regime

    @property
    def noise(self) -> float:
        """Return the flip probability, zero outside the label-flips regime."""
        return self._noise

    def flips(self, draw: float, /) -> bool:
        """Return whether one certified negative is observed as positive.

        Parameters
        ----------
        draw : float
            A reproducible draw in the half-open unit interval.
        """
        held = ensure_finite("a negative-data label draw", draw)

        if not 0.0 <= held < 1.0:
            msg = f"a negative-data label draw is not in [0, 1): {draw!r}"
            raise ValueError(msg)

        if self._regime == "positive-only":
            return True

        return self._regime == "label-flips" and held < self._noise

    def _fields(self) -> tuple[NegativeDataRegime, float]:
        """Return the fields the policy is equal on."""
        return (self._regime, self._noise)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another policy presents negatives the same way."""
        return type(other) is NegativeDataPolicy and self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the regime and noise rate."""
        return hash((NegativeDataPolicy, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this policy."""
        return f"NegativeDataPolicy({self._regime!r}, noise={self._noise!r})"


def _by_kind[K](
    field: str, counts: Mapping[K, int], kinds: tuple[K, ...]
) -> Mapping[K, int]:
    """Return a count per requested operator, in request order, every one present."""
    unknown = [kind for kind in counts if kind not in kinds]

    if unknown:
        msg = f"{field} counts operators the request does not name: {unknown!r}"
        raise ValueError(msg)

    return MappingProxyType(
        {
            kind: ensure_non_negative(f"{field}[{kind!r}]", counts.get(kind, 0))
            for kind in kinds
        }
    )


def _by_failure(
    counts: Mapping[TypingFailureKind, int],
) -> Mapping[TypingFailureKind, int]:
    """Return a positive count per certificate kind, in kind order."""
    return MappingProxyType(
        {
            kind: ensure_positive(f"certificates[{kind!r}]", counts[kind])
            for kind in sorted(counts)
        }
    )


def _by_form_label(
    counts: Mapping[FormTypingLabel, int],
) -> Mapping[FormTypingLabel, int]:
    """Return a count per logical-form typing label in report order."""
    unknown = [label for label in counts if label not in FORM_TYPING_LABELS]

    if unknown:
        msg = f"observed forms count unknown typing labels: {unknown!r}"
        raise ValueError(msg)

    return MappingProxyType(
        {
            label: ensure_non_negative(
                f"observed_forms[{label!r}]", counts.get(label, 0)
            )
            for label in FORM_TYPING_LABELS
        }
    )


def _by_label(
    field: str, counts: Mapping[GrammaticalityLabel, int]
) -> Mapping[GrammaticalityLabel, int]:
    """Return a count per grammaticality label, in report order, both present."""
    unknown = [label for label in counts if label not in GRAMMATICALITY_LABELS]

    if unknown:
        msg = f"{field} counts unknown labels: {unknown!r}"
        raise ValueError(msg)

    return MappingProxyType(
        {
            label: ensure_non_negative(f"{field}[{label!r}]", counts.get(label, 0))
            for label in GRAMMATICALITY_LABELS
        }
    )


def _ensure_label(field: str, label: GrammaticalityLabel) -> GrammaticalityLabel:
    """Return a grammaticality label."""
    if label not in GRAMMATICALITY_LABELS:
        msg = f"{field} is not a grammaticality label: {label!r}"
        raise ValueError(msg)

    return label


def _other(label: GrammaticalityLabel) -> GrammaticalityLabel:
    """Return the grammaticality label a flip turns a label into."""
    return "ungrammatical" if label == "grammatical" else "grammatical"


class Histogram:
    """Counts of an integer measurement, one bucket per value observed.

    Parameters
    ----------
    counts : Mapping[int, int]
        How often each value was observed. A value observed zero times has
        no bucket.

    Raises
    ------
    TypeError
        If a value or a count is not an integer.
    ValueError
        If a count is negative.
    """

    __slots__ = ("_counts",)

    def __init__(self, counts: Mapping[int, int]) -> None:
        held: dict[int, int] = {}

        for value, count in sorted(counts.items()):
            if isinstance(value, bool) or not isinstance(value, int):
                msg = f"a histogram value is not an integer: {value!r}"
                raise TypeError(msg)

            seen = ensure_non_negative(f"the count of {value}", count)

            if seen:
                held[value] = seen

        self._counts = MappingProxyType(held)

    @classmethod
    def of(cls, values: Iterable[int]) -> Histogram:
        """Return the histogram of a sequence of observations.

        Parameters
        ----------
        values : Iterable[int]
            The observations, in any order.
        """
        return cls(Counter(values))

    @property
    def counts(self) -> Mapping[int, int]:
        """Return the buckets in ascending order of value."""
        return self._counts

    @property
    def total(self) -> int:
        """Return how many observations the buckets hold between them."""
        return sum(self._counts.values())

    @property
    def mean(self) -> float:
        """Return the mean observation.

        Raises
        ------
        ValueError
            If nothing was observed.
        """
        total = self.total

        if total == 0:
            msg = "an empty histogram has no mean"
            raise ValueError(msg)

        return sum(value * count for value, count in self._counts.items()) / total

    @property
    def variance(self) -> float:
        """Return the variance of the observations about their mean.

        Raises
        ------
        ValueError
            If nothing was observed.
        """
        mean = self.mean

        return (
            sum((value - mean) ** 2 * count for value, count in self._counts.items())
            / self.total
        )

    def __getitem__(self, value: int) -> int:
        """Return how often a value was observed, zero when never."""
        return self._counts.get(value, 0)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another histogram holds the same buckets."""
        if type(other) is not Histogram:
            return NotImplemented

        return self._counts == other._counts

    @override
    def __hash__(self) -> int:
        """Return a hash over the buckets."""
        return hash((Histogram, tuple(self._counts.items())))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this histogram."""
        return f"Histogram({dict(self._counts)!r})"


class Target:
    """A value to aim at and the half-width of the band that counts as a hit.

    Parameters
    ----------
    value : float
        The aimed-at value.
    tolerance : float
        How far from the value a measurement may fall and still be admitted.

    Raises
    ------
    TypeError
        If either number is not numeric.
    ValueError
        If either number is not finite, or the tolerance is negative.
    """

    __slots__ = ("_tolerance", "_value")

    def __init__(self, value: float, tolerance: float) -> None:
        self._value = ensure_finite("a target value", value)
        self._tolerance = ensure_finite("a target tolerance", tolerance)

        if self._tolerance < 0:
            msg = f"a target tolerance is negative: {tolerance!r}"
            raise ValueError(msg)

    @property
    def value(self) -> float:
        """Return the aimed-at value."""
        return self._value

    @property
    def tolerance(self) -> float:
        """Return the half-width of the admitting band."""
        return self._tolerance

    def admits(self, measured: float) -> bool:
        """Return whether a measurement falls within the band, edges included.

        A measurement within a relative distance of one part in a billion of
        an edge counts as on it, so a band whose edge is not exactly
        representable, such as ``Target(1.3, 0.3)``, admits the measurement
        at that edge.

        Parameters
        ----------
        measured : float
            The measurement to test.
        """
        if not isfinite(measured):
            return False

        lower = self._value - self._tolerance
        upper = self._value + self._tolerance

        return (lower <= measured or isclose(measured, lower)) and (
            measured <= upper or isclose(measured, upper)
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another target has the same value and tolerance."""
        if type(other) is not Target:
            return NotImplemented

        return (self._value, self._tolerance) == (other._value, other._tolerance)

    @override
    def __hash__(self) -> int:
        """Return a hash over the value and the tolerance."""
        return hash((Target, self._value, self._tolerance))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this target."""
        return f"Target({self._value!r}, {self._tolerance!r})"


class AmbiguityTarget:
    """What a sample aims at in lexical and derivational ambiguity.

    Parameters
    ----------
    lexical : Target | None
        The mean number of lexical categories per word token of a string, or
        nothing to leave lexical ambiguity uncontrolled.
    derivations : Target | None
        The number of root derivations of a string, or nothing to leave the
        derivation count uncontrolled.
    attempts : int
        How many draws one accepted string may take before the target is
        declared unreachable.

    Raises
    ------
    TypeError
        If the attempt budget is not an integer.
    ValueError
        If the attempt budget is not positive.
    """

    __slots__ = ("_attempts", "_derivations", "_lexical")

    def __init__(
        self,
        *,
        lexical: Target | None = None,
        derivations: Target | None = None,
        attempts: int,
    ) -> None:
        self._lexical = lexical
        self._derivations = derivations
        self._attempts = ensure_positive("attempts", attempts)

    @property
    def lexical(self) -> Target | None:
        """Return the lexical target, when one is set."""
        return self._lexical

    @property
    def derivations(self) -> Target | None:
        """Return the derivation-count target, when one is set."""
        return self._derivations

    @property
    def attempts(self) -> int:
        """Return how many draws one accepted string may take."""
        return self._attempts

    def _fields(self) -> tuple[Target | None, Target | None, int]:
        """Return the fields the target is equal on."""
        return (self._lexical, self._derivations, self._attempts)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value sets the same targets and budget."""
        if type(other) is not AmbiguityTarget:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the targets and the budget."""
        return hash((AmbiguityTarget, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this target."""
        return (
            f"AmbiguityTarget(lexical={self._lexical!r}, "
            f"derivations={self._derivations!r}, attempts={self._attempts!r})"
        )


class AmbiguityReport:
    """The ambiguity a sample achieved, measured from the sample itself.

    Parameters
    ----------
    lexical : Histogram
        Over the word tokens of the grammatical strings: how many lexical
        categories each token's form has.
    derivations : Histogram
        Over the grammatical strings: how many root derivations the parser
        finds for each.
    rejected : int
        How many drawn strings fell outside a target band and were redrawn.
    target : AmbiguityTarget | None
        The target the sample aimed at, or nothing when it aimed at none.

    Raises
    ------
    TypeError
        If the rejection count is not an integer.
    ValueError
        If the rejection count is negative.
    """

    __slots__ = ("_derivations", "_lexical", "_rejected", "_target")

    def __init__(
        self,
        lexical: Histogram,
        derivations: Histogram,
        *,
        rejected: int,
        target: AmbiguityTarget | None,
    ) -> None:
        self._lexical = lexical
        self._derivations = derivations
        self._rejected = ensure_non_negative("rejected", rejected)
        self._target = target

    @property
    def lexical(self) -> Histogram:
        """Return the lexical categories per word token, by count of tokens."""
        return self._lexical

    @property
    def derivations(self) -> Histogram:
        """Return the root derivations per string, by count of strings."""
        return self._derivations

    @property
    def rejected(self) -> int:
        """Return how many drawn strings fell outside a target band."""
        return self._rejected

    @property
    def target(self) -> AmbiguityTarget | None:
        """Return the target aimed at, when there was one."""
        return self._target

    def _fields(self) -> tuple[Histogram, Histogram, int, AmbiguityTarget | None]:
        """Return the fields the report is equal on."""
        return (self._lexical, self._derivations, self._rejected, self._target)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report measured the same sample at the same target."""
        if type(other) is not AmbiguityReport:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the histograms, the rejections and the target."""
        return hash((AmbiguityReport, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this report."""
        return (
            f"AmbiguityReport({self._lexical!r}, {self._derivations!r}, "
            f"rejected={self._rejected!r}, target={self._target!r})"
        )


class ShapeTarget(FrozenValue):
    """Bands a sampled derivation's structural measurements must meet."""

    __slots__ = (
        "_arity",
        "_attempts",
        "_depth",
        "_right_branching",
        "_slash_balance",
    )

    _noun = "derivation-shape target"

    def __init__(
        self,
        *,
        depth: Target | None = None,
        right_branching: Target | None = None,
        slash_balance: Target | None = None,
        arity: Target | None = None,
        attempts: int,
    ) -> None:
        if all(
            target is None for target in (depth, right_branching, slash_balance, arity)
        ):
            msg = "a derivation-shape target names no measurement"
            raise ValueError(msg)

        for name, target in (
            ("right_branching", right_branching),
            ("slash_balance", slash_balance),
        ):
            if target is not None and not 0.0 <= target.value <= 1.0:
                msg = f"{name} target is not between zero and one: {target.value!r}"
                raise ValueError(msg)

        if depth is not None and depth.value <= 0.0:
            msg = f"depth target is not positive: {depth.value!r}"
            raise ValueError(msg)

        if arity is not None and arity.value < 1.0:
            msg = f"arity target is less than one: {arity.value!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_depth", depth)
        object.__setattr__(self, "_right_branching", right_branching)
        object.__setattr__(self, "_slash_balance", slash_balance)
        object.__setattr__(self, "_arity", arity)
        object.__setattr__(self, "_attempts", ensure_positive("attempts", attempts))

    @property
    def depth(self) -> Target | None:
        """Return the root-to-leaf depth target, when set."""
        return self._depth

    @property
    def right_branching(self) -> Target | None:
        """Return the share of binary nodes deeper on the right to aim at."""
        return self._right_branching

    @property
    def slash_balance(self) -> Target | None:
        """Return the share of slash rules using the forward slash to aim at."""
        return self._slash_balance

    @property
    def arity(self) -> Target | None:
        """Return the mean rule arity target, when set."""
        return self._arity

    @property
    def attempts(self) -> int:
        """Return how many derivations one accepted string may take."""
        return self._attempts

    def admits(self, measured: ShapeMeasurement, /) -> bool:
        """Return whether every declared band admits its measurement."""
        return all(
            target is None or target.admits(value)
            for target, value in (
                (self._depth, float(measured.depth)),
                (self._right_branching, measured.right_branching),
                (self._slash_balance, measured.slash_balance),
                (self._arity, measured.arity),
            )
        )

    def _fields(
        self,
    ) -> tuple[Target | None, Target | None, Target | None, Target | None, int]:
        """Return the fields the target is equal on."""
        return (
            self._depth,
            self._right_branching,
            self._slash_balance,
            self._arity,
            self._attempts,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another target declares the same bands and budget."""
        return type(other) is ShapeTarget and self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the bands and budget."""
        return hash((ShapeTarget, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this target."""
        return (
            f"ShapeTarget(depth={self._depth!r}, "
            f"right_branching={self._right_branching!r}, "
            f"slash_balance={self._slash_balance!r}, arity={self._arity!r}, "
            f"attempts={self._attempts!r})"
        )


class ShapeSchedule(FrozenValue):
    """Select one derivation-shape target from an item's declared split."""

    __slots__ = ("_default", "_splits")

    _noun = "derivation-shape schedule"

    def __init__(
        self,
        default: ShapeTarget | None,
        splits: Mapping[str, ShapeTarget],
        /,
    ) -> None:
        held = {
            ensure_name("a shape-schedule split", name): target
            for name, target in sorted(splits.items())
        }
        if default is None and not held:
            msg = "a derivation-shape schedule names no target"
            raise ValueError(msg)
        object.__setattr__(self, "_default", default)
        object.__setattr__(self, "_splits", MappingProxyType(held))

    @property
    def default(self) -> ShapeTarget | None:
        """Return the target used when a split has no override."""
        return self._default

    @property
    def splits(self) -> Mapping[str, ShapeTarget]:
        """Return the exact target overrides by split name."""
        return self._splits

    def target(self, split: str | None, /) -> ShapeTarget | None:
        """Return the target selected for one item's split."""
        return (
            self._default if split is None else self._splits.get(split, self._default)
        )

    def _fields(
        self,
    ) -> tuple[ShapeTarget | None, tuple[tuple[str, ShapeTarget], ...]]:
        """Return the immutable values that identify the schedule."""
        return self._default, tuple(self._splits.items())

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another schedule selects the same targets."""
        return type(other) is ShapeSchedule and self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the default and split targets."""
        return hash((ShapeSchedule, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds the schedule."""
        return f"ShapeSchedule({self._default!r}, {dict(self._splits)!r})"


class ShapeMeasurement(FrozenValue):
    """The structural measurements of one sampled derivation."""

    __slots__ = ("_arity", "_depth", "_right_branching", "_slash_balance")

    _noun = "derivation-shape measurement"

    def __init__(
        self,
        depth: int,
        right_branching: float,
        slash_balance: float,
        arity: float,
    ) -> None:
        held_arity = ensure_finite("a derivation's mean arity", arity)

        if held_arity < 0.0:
            msg = f"a derivation's mean arity is negative: {arity!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_depth", ensure_positive("depth", depth))
        object.__setattr__(
            self,
            "_right_branching",
            _proportion("right_branching", right_branching),
        )
        object.__setattr__(
            self,
            "_slash_balance",
            _proportion("slash_balance", slash_balance),
        )
        object.__setattr__(self, "_arity", held_arity)

    @property
    def depth(self) -> int:
        """Return the root-to-leaf node depth."""
        return self._depth

    @property
    def right_branching(self) -> float:
        """Return the share of binary nodes deeper on the right."""
        return self._right_branching

    @property
    def slash_balance(self) -> float:
        """Return the share of slash rule applications using the forward slash."""
        return self._slash_balance

    @property
    def arity(self) -> float:
        """Return the mean premise count of rule applications."""
        return self._arity

    def _fields(self) -> tuple[int, float, float, float]:
        """Return the fields the measurement is equal on."""
        return (self._depth, self._right_branching, self._slash_balance, self._arity)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another measurement has the same values."""
        return type(other) is ShapeMeasurement and self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the measurements."""
        return hash((ShapeMeasurement, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this measurement."""
        return (
            f"ShapeMeasurement({self._depth!r}, {self._right_branching!r}, "
            f"{self._slash_balance!r}, {self._arity!r})"
        )


class ShapeReport(FrozenValue):
    """Accepted derivation shapes beside the target and rejection count."""

    __slots__ = ("_measurements", "_rejected", "_target")

    _noun = "derivation-shape report"

    def __init__(
        self,
        measurements: tuple[ShapeMeasurement, ...],
        *,
        rejected: int,
        target: ShapeSchedule,
    ) -> None:
        object.__setattr__(self, "_measurements", measurements)
        object.__setattr__(self, "_rejected", ensure_non_negative("rejected", rejected))
        object.__setattr__(self, "_target", target)

    @property
    def measurements(self) -> tuple[ShapeMeasurement, ...]:
        """Return the accepted per-string measurements in draw order."""
        return self._measurements

    @property
    def rejected(self) -> int:
        """Return how many candidates missed at least one declared band."""
        return self._rejected

    @property
    def target(self) -> ShapeSchedule:
        """Return the split-conditioned targets the sample aimed at."""
        return self._target

    @property
    def depths(self) -> Histogram:
        """Return accepted root-to-leaf depths as a histogram."""
        return Histogram.of(measurement.depth for measurement in self._measurements)

    def _mean(self, field: str) -> float:
        """Return the mean of one measured floating-point field."""
        if not self._measurements:
            return 0.0

        return sum(float(getattr(value, field)) for value in self._measurements) / len(
            self._measurements
        )

    @property
    def right_branching(self) -> float:
        """Return the mean achieved right-branching share."""
        return self._mean("right_branching")

    @property
    def slash_balance(self) -> float:
        """Return the mean achieved forward-slash share."""
        return self._mean("slash_balance")

    @property
    def arity(self) -> float:
        """Return the mean achieved rule arity."""
        return self._mean("arity")

    def _fields(self) -> tuple[tuple[ShapeMeasurement, ...], int, ShapeSchedule]:
        """Return the fields the report is equal on."""
        return (self._measurements, self._rejected, self._target)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report holds the same measurements and target."""
        return type(other) is ShapeReport and self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the measurements, rejections and target."""
        return hash((ShapeReport, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this report."""
        return (
            f"ShapeReport({self._measurements!r}, rejected={self._rejected!r}, "
            f"target={self._target!r})"
        )


class ScopeTarget:
    """What a sample aims at in logical-form readings per string.

    Each equivalence decision runs under the clock of the interpretation the
    readings are read under.

    Parameters
    ----------
    readings : Target | None
        How many logically distinct forms a string has, or nothing to measure
        the readings without banding them.
    derivations : int
        The most derivations one string's readings are enumerated from. A
        drawn string with more is discarded and drawn again.
    attempts : int
        How many draws one accepted string may take before the target is
        declared unreachable.

    Raises
    ------
    TypeError
        If the derivation bound or the attempt budget is not an integer.
    ValueError
        If the derivation bound or the attempt budget is not positive.
    """

    __slots__ = ("_attempts", "_derivations", "_readings", "_undecided")

    def __init__(
        self,
        *,
        readings: Target | None = None,
        derivations: int,
        attempts: int,
        undecided: UndecidedScopePolicy = "distinct",
    ) -> None:
        self._readings = readings
        self._derivations = ensure_positive("derivations", derivations)
        self._attempts = ensure_positive("attempts", attempts)
        if undecided not in ("distinct", "equivalent"):
            msg = f"an undecided-scope policy is not declared: {undecided!r}"
            raise ValueError(msg)
        self._undecided = undecided

    @property
    def readings(self) -> Target | None:
        """Return the readings-per-string target, when one is set."""
        return self._readings

    @property
    def derivations(self) -> int:
        """Return the most derivations one string's readings are read from."""
        return self._derivations

    @property
    def attempts(self) -> int:
        """Return how many draws one accepted string may take."""
        return self._attempts

    @property
    def undecided(self) -> UndecidedScopePolicy:
        """Return how undecided equivalences contribute to the reading count."""
        return self._undecided

    def _fields(self) -> tuple[Target | None, int, int, UndecidedScopePolicy]:
        """Return the fields the target is equal on."""
        return (self._readings, self._derivations, self._attempts, self._undecided)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value sets the same target and bounds."""
        if type(other) is not ScopeTarget:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the target and the bounds."""
        return hash((ScopeTarget, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this target."""
        return (
            f"ScopeTarget(readings={self._readings!r}, "
            f"derivations={self._derivations!r}, attempts={self._attempts!r}, "
            f"undecided={self._undecided!r})"
        )


class ReadingCount:
    """The two ambiguities of one string, counted side by side.

    Parameters
    ----------
    item_id : ItemId
        The item the counts belong to.
    derivations : int
        How many root derivations the parser finds for the string.
    readings : int
        How many of those derivations' logical forms remain once forms that
        entail each other are identified.

    Raises
    ------
    TypeError
        If either count is not an integer.
    ValueError
        If either count is not positive, or there are more readings than
        derivations.
    """

    __slots__ = ("_derivations", "_item_id", "_readings")

    def __init__(self, item_id: ItemId, *, derivations: int, readings: int) -> None:
        self._item_id = item_id
        self._derivations = ensure_positive(
            f"the derivations of {item_id!r}", derivations
        )
        self._readings = ensure_positive(f"the readings of {item_id!r}", readings)

        if self._readings > self._derivations:
            msg = (
                f"item {item_id!r} has {self._readings} readings from "
                f"{self._derivations} derivations"
            )
            raise ValueError(msg)

    @property
    def item_id(self) -> ItemId:
        """Return the item the counts belong to."""
        return self._item_id

    @property
    def derivations(self) -> int:
        """Return how many root derivations the string has."""
        return self._derivations

    @property
    def readings(self) -> int:
        """Return how many logically distinct forms the string has."""
        return self._readings

    def _fields(self) -> tuple[ItemId, int, int]:
        """Return the fields the record is equal on."""
        return (self._item_id, self._derivations, self._readings)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another record counts the same item the same way."""
        if type(other) is not ReadingCount:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the item and the two counts."""
        return hash((ReadingCount, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this record."""
        return (
            f"ReadingCount({self._item_id!r}, derivations={self._derivations!r}, "
            f"readings={self._readings!r})"
        )


class ScopeReport:
    """The semantic ambiguity a sample achieved, measured from the sample itself.

    Parameters
    ----------
    counts : tuple[ReadingCount, ...]
        The derivation and reading counts of each grammatical item, in draw
        order.
    rejected : int
        How many drawn strings fell outside the readings band and were
        redrawn.
    exceeded : int
        How many drawn strings had more derivations than the bound and were
        redrawn without their readings being counted.
    undecided : int
        How many equivalence questions the oracle left undecided, each of
        which leaves two forms counted as two readings rather than one.
    target : ScopeTarget
        The target the sample was drawn under.

    Raises
    ------
    TypeError
        If a count is not an integer.
    ValueError
        If a count is negative, or an item is counted twice.
    """

    __slots__ = ("_counts", "_exceeded", "_rejected", "_target", "_undecided")

    def __init__(
        self,
        counts: tuple[ReadingCount, ...],
        *,
        rejected: int,
        exceeded: int,
        undecided: int,
        target: ScopeTarget,
    ) -> None:
        items = [count.item_id for count in counts]

        if len(set(items)) != len(items):
            msg = "a scope report counts an item twice"
            raise ValueError(msg)

        self._counts = counts
        self._rejected = ensure_non_negative("rejected", rejected)
        self._exceeded = ensure_non_negative("exceeded", exceeded)
        self._undecided = ensure_non_negative("undecided", undecided)
        self._target = target

    @property
    def counts(self) -> tuple[ReadingCount, ...]:
        """Return the two counts of each grammatical item, in draw order."""
        return self._counts

    @property
    def readings(self) -> Histogram:
        """Return the logically distinct forms per string, by count of strings."""
        return Histogram.of(count.readings for count in self._counts)

    @property
    def derivations(self) -> Histogram:
        """Return the root derivations per string, by count of strings."""
        return Histogram.of(count.derivations for count in self._counts)

    @property
    def rejected(self) -> int:
        """Return how many drawn strings fell outside the readings band."""
        return self._rejected

    @property
    def exceeded(self) -> int:
        """Return how many drawn strings had more derivations than the bound."""
        return self._exceeded

    @property
    def undecided(self) -> int:
        """Return how many equivalence questions the oracle left undecided."""
        return self._undecided

    @property
    def target(self) -> ScopeTarget:
        """Return the target the sample was drawn under."""
        return self._target

    def _fields(self) -> tuple[tuple[ReadingCount, ...], int, int, int, ScopeTarget]:
        """Return the fields the report is equal on."""
        return (
            self._counts,
            self._rejected,
            self._exceeded,
            self._undecided,
            self._target,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report measured the same sample at the same target."""
        if type(other) is not ScopeReport:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the counts, the discards and the target."""
        return hash((ScopeReport, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this report."""
        return (
            f"ScopeReport({self._counts!r}, rejected={self._rejected!r}, "
            f"exceeded={self._exceeded!r}, undecided={self._undecided!r}, "
            f"target={self._target!r})"
        )


class LengthReport:
    """The token counts a sample achieved against the distribution it drew from.

    Parameters
    ----------
    achieved : Histogram
        Over the grammatical strings: how many tokens each holds.
    resampled : int
        How many drawn token counts were discarded and drawn again, each
        because the grammar derives no goal at it or it lies beyond the
        tabulated bound.
    distribution : LengthDistribution
        The distribution the counts were drawn from.

    Raises
    ------
    TypeError
        If the resample count is not an integer.
    ValueError
        If the resample count is negative.
    """

    __slots__ = ("_achieved", "_distribution", "_resampled")

    def __init__(
        self,
        achieved: Histogram,
        *,
        resampled: int,
        distribution: LengthDistribution,
    ) -> None:
        self._achieved = achieved
        self._resampled = ensure_non_negative("resampled", resampled)
        self._distribution = distribution

    @property
    def achieved(self) -> Histogram:
        """Return the tokens per grammatical string, by count of strings."""
        return self._achieved

    @property
    def resampled(self) -> int:
        """Return how many drawn token counts were discarded and drawn again."""
        return self._resampled

    @property
    def distribution(self) -> LengthDistribution:
        """Return the distribution the counts were drawn from."""
        return self._distribution

    def _fields(self) -> tuple[Histogram, int, LengthDistribution]:
        """Return the fields the report is equal on."""
        return (self._achieved, self._resampled, self._distribution)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report measured the same counts of the same draw."""
        if type(other) is not LengthReport:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the histogram, the resamples and the distribution."""
        return hash((LengthReport, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this report."""
        return (
            f"LengthReport({self._achieved!r}, resampled={self._resampled!r}, "
            f"distribution={self._distribution!r})"
        )


class NamedOperator(Protocol):
    """An installed operator whose entry-point name identifies report counts."""

    @property
    def name(self) -> str:
        """Return the installed operator name."""
        ...


class WeightedOperator[O: NamedOperator](FrozenValue):
    """One built operator and its relative probability of being drawn."""

    _noun = "weighted operator"
    __slots__ = ("_name", "_operator", "_weight")

    def __init__(self, operator: O, weight: float) -> None:
        name = ensure_name("an operator name", operator.name)
        held_weight = ensure_finite("an operator weight", weight)
        if held_weight <= 0.0:
            msg = f"an operator weight is not positive: {weight!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_operator", operator)
        object.__setattr__(self, "_weight", held_weight)

    @property
    def operator(self) -> O:
        """Return the installed operator."""
        return self._operator

    @property
    def name(self) -> str:
        """Return the installed operator name."""
        return self._name

    @property
    def weight(self) -> float:
        """Return the relative draw weight."""
        return self._weight

    def _fields(self) -> tuple[O, float]:
        """Return the fields the weighted operator is equal on."""
        return self._operator, self._weight

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another entry holds the same operator and weight."""
        return type(other) is WeightedOperator and self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the operator and weight."""
        return hash((WeightedOperator, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds the entry."""
        return f"WeightedOperator({self._operator!r}, {self._weight!r})"


type CorruptionDraw = Literal["exact", "bernoulli"]
"""Whether a corruption count is rounded once or drawn independently per item."""


def _rounded_share(proportion: float, count: int) -> int:
    """Return one share of a count to the nearest integer, with halves up."""
    return int(proportion * count + 0.5)


class Corruption[O: NamedOperator](FrozenValue):
    """How much of a sample to corrupt with weighted installed operators."""

    _noun = "corruption request"
    __slots__ = (
        "_attempts",
        "_draw",
        "_operators",
        "_policy",
        "_proportion",
    )

    def __init__(
        self,
        proportion: float,
        *,
        operators: tuple[WeightedOperator[O], ...],
        attempts: int,
        draw: CorruptionDraw,
        policy: NegativeDataPolicy | None = None,
    ) -> None:
        if not operators:
            msg = "a corruption names no operator"
            raise ValueError(msg)

        names = tuple(operator.name for operator in operators)
        if len(set(names)) != len(names):
            msg = f"a corruption names one operator twice: {names!r}"
            raise ValueError(msg)

        if draw not in ("exact", "bernoulli"):
            msg = f"an unknown corruption draw was declared: {draw!r}"
            raise ValueError(msg)

        object.__setattr__(
            self, "_proportion", _proportion("a corruption proportion", proportion)
        )
        object.__setattr__(self, "_operators", operators)
        object.__setattr__(self, "_attempts", ensure_positive("attempts", attempts))
        object.__setattr__(self, "_draw", draw)
        object.__setattr__(
            self,
            "_policy",
            NegativeDataPolicy("labeled-negatives") if policy is None else policy,
        )

    @property
    def proportion(self) -> float:
        """Return the share of the sample to emit as corrupted items."""
        return self._proportion

    @property
    def operators(self) -> tuple[WeightedOperator[O], ...]:
        """Return the weighted installed operators in declaration order."""
        return self._operators

    @property
    def operator_names(self) -> tuple[str, ...]:
        """Return operator names in report order."""
        return tuple(operator.name for operator in self._operators)

    @property
    def attempts(self) -> int:
        """Return how many candidates one corrupted item may take."""
        return self._attempts

    @property
    def draw(self) -> CorruptionDraw:
        """Return how the number of corrupted items is drawn."""
        return self._draw

    @property
    def policy(self) -> NegativeDataPolicy:
        """Return how accepted corruptions are presented to a learner."""
        return self._policy

    def choose(self, random: Random, /) -> O:
        """Draw one operator at its declared relative weight."""
        return random.choices(
            tuple(weighted.operator for weighted in self._operators),
            weights=tuple(weighted.weight for weighted in self._operators),
            k=1,
        )[0]

    def count(self, population: int, random: Random, /) -> int:
        """Return how many items to corrupt under the declared draw policy."""
        held = ensure_non_negative("a corruption population", population)
        if self._draw == "exact":
            return _rounded_share(self._proportion, held)
        return sum(random.random() < self._proportion for _ in range(held))

    def _fields(
        self,
    ) -> tuple[
        float,
        tuple[WeightedOperator[O], ...],
        int,
        CorruptionDraw,
        NegativeDataPolicy,
    ]:
        """Return the fields the corruption is equal on."""
        return (
            self._proportion,
            self._operators,
            self._attempts,
            self._draw,
            self._policy,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value requests the same corruption."""
        return type(other) is Corruption and self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the share, operators, draw, policy and budget."""
        return hash((Corruption, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this corruption."""
        policy = (
            ""
            if self._policy == NegativeDataPolicy("labeled-negatives")
            else f", policy={self._policy!r}"
        )
        return (
            f"Corruption({self._proportion!r}, operators={self._operators!r}, "
            f"attempts={self._attempts!r}, draw={self._draw!r}{policy})"
        )


class NonDerivability:
    """The grammar and budget under which the parser found no root derivation.

    Parameters
    ----------
    declaration_digest : str
        The SHA-256 of the declaration bytes of the grammar parsed under.
    theory_digest : str
        The SHA-256 of the compiled theory bytes of that grammar.
    unary_rounds : int
        How many unary rounds the parse closed each cell with.

    Raises
    ------
    TypeError
        If the unary rounds are not an integer.
    ValueError
        If a digest is empty or the unary rounds are negative.
    """

    __slots__ = ("_declaration_digest", "_theory_digest", "_unary_rounds")

    def __init__(
        self, declaration_digest: str, theory_digest: str, *, unary_rounds: int
    ) -> None:
        for field, digest in (
            ("declaration", declaration_digest),
            ("theory", theory_digest),
        ):
            if not digest:
                msg = f"a non-derivability certificate names no {field} digest"
                raise ValueError(msg)

        self._declaration_digest = declaration_digest
        self._theory_digest = theory_digest
        self._unary_rounds = ensure_non_negative("unary_rounds", unary_rounds)

    @property
    def declaration_digest(self) -> str:
        """Return the digest of the declaration parsed under."""
        return self._declaration_digest

    @property
    def theory_digest(self) -> str:
        """Return the digest of the compiled theory parsed under."""
        return self._theory_digest

    @property
    def unary_rounds(self) -> int:
        """Return how many unary rounds the parse closed each cell with."""
        return self._unary_rounds

    def _fields(self) -> tuple[str, str, int]:
        """Return the fields the certificate is equal on."""
        return (self._declaration_digest, self._theory_digest, self._unary_rounds)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another certificate names the same grammar and budget."""
        if type(other) is not NonDerivability:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the digests and the budget."""
        return hash((NonDerivability, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this certificate."""
        return (
            f"NonDerivability({self._declaration_digest!r}, "
            f"{self._theory_digest!r}, unary_rounds={self._unary_rounds!r})"
        )


class Corrupted:
    """A string the parser found non-derivable, with the grammatical one it came from.

    Parameters
    ----------
    item : Item
        The corrupted string as a corpus item.
    kind : str
        The operator that produced it.
    source : tuple[str, ...]
        The surface forms of the grammatical string the operator was applied
        to.
    certificate : NonDerivability
        The grammar and budget under which the parser found no derivation.

    Raises
    ------
    ValueError
        If the kind is unknown or the source is empty.
    """

    __slots__ = ("_certificate", "_item", "_kind", "_source")

    def __init__(
        self,
        item: Item,
        kind: str,
        source: tuple[str, ...],
        certificate: NonDerivability,
    ) -> None:
        if not source:
            msg = f"corrupted item {item.item_id!r} has no source string"
            raise ValueError(msg)

        self._item = item
        self._kind = ensure_name("a corruption operator", kind)
        self._source = source
        self._certificate = certificate

    @property
    def item(self) -> Item:
        """Return the corrupted string as a corpus item."""
        return self._item

    @property
    def kind(self) -> str:
        """Return the operator that produced the string."""
        return self._kind

    @property
    def source(self) -> tuple[str, ...]:
        """Return the surface forms of the grammatical string corrupted."""
        return self._source

    @property
    def certificate(self) -> NonDerivability:
        """Return the grammar and budget the string was found non-derivable under."""
        return self._certificate

    def _fields(self) -> tuple[Item, str, tuple[str, ...], NonDerivability]:
        """Return the fields the record is equal on."""
        return (self._item, self._kind, self._source, self._certificate)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another record holds the same corruption of one string."""
        if type(other) is not Corrupted:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the item, the kind, the source and the certificate."""
        return hash((Corrupted, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this record."""
        return (
            f"Corrupted({self._item!r}, {self._kind!r}, {self._source!r}, "
            f"{self._certificate!r})"
        )


class CorruptionReport[O: NamedOperator]:
    """What corrupting a sample requested and what it achieved, by operator.

    One shape serves both arms: the syntactic report counts string operators
    and the semantic one counts ill-typing operators.

    Parameters
    ----------
    requested : Corruption[O]
        The proportion, operators and budget the sample was drawn with.
    accepted : Mapping[str, int]
        How many corrupted items each operator produced.
    rejected : Mapping[str, int]
        How many candidates each operator produced that the certifier
        accepted as sound, which were discarded and redrawn.
    inapplicable : Mapping[str, int]
        How many draws named an operator the drawn item did not admit.
    sample_size : int
        How many items the corruption was drawn against, corrupted ones
        included.

    Raises
    ------
    TypeError
        If a count is not an integer.
    ValueError
        If a count is keyed by an operator the request does not name or is
        negative, or the accepted items outnumber the sample.
    """

    __slots__ = ("_accepted", "_inapplicable", "_rejected", "_requested", "_size")

    def __init__(
        self,
        requested: Corruption[O],
        *,
        accepted: Mapping[str, int],
        rejected: Mapping[str, int],
        inapplicable: Mapping[str, int],
        sample_size: int,
    ) -> None:
        kinds = requested.operator_names
        self._requested = requested
        self._accepted = _by_kind("accepted", accepted, kinds)
        self._rejected = _by_kind("rejected", rejected, kinds)
        self._inapplicable = _by_kind("inapplicable", inapplicable, kinds)
        self._size = ensure_non_negative("sample_size", sample_size)

        if sum(self._accepted.values()) > self._size:
            msg = (
                f"{sum(self._accepted.values())} corrupted items outnumber "
                f"a sample of {self._size}"
            )
            raise ValueError(msg)

    @property
    def requested(self) -> Corruption[O]:
        """Return the proportion, operators and budget the sample was drawn with."""
        return self._requested

    @property
    def accepted(self) -> Mapping[str, int]:
        """Return how many corrupted items each operator produced."""
        return self._accepted

    @property
    def rejected(self) -> Mapping[str, int]:
        """Return how many sound candidates each operator produced."""
        return self._rejected

    @property
    def inapplicable(self) -> Mapping[str, int]:
        """Return how many draws named an operator the item did not admit."""
        return self._inapplicable

    @property
    def sample_size(self) -> int:
        """Return how many items the corruption was drawn against."""
        return self._size

    @property
    def corrupted(self) -> int:
        """Return how many corrupted items the sample holds."""
        return sum(self._accepted.values())

    @property
    def achieved_proportion(self) -> float:
        """Return the share of the sample that is corrupted, zero when it is empty."""
        return self.corrupted / self._size if self._size else 0.0

    @property
    def rejection_rate(self) -> float:
        """Return the share of applicable candidates the certifier refused.

        Raises
        ------
        ValueError
            If no applicable candidate was produced.
        """
        rejected = sum(self._rejected.values())
        tried = rejected + self.corrupted

        if tried == 0:
            msg = "no applicable corruption was tried, so there is no rejection rate"
            raise ValueError(msg)

        return rejected / tried

    def _fields(
        self,
    ) -> tuple[
        Corruption[O],
        tuple[tuple[str, int], ...],
        tuple[tuple[str, int], ...],
        tuple[tuple[str, int], ...],
        int,
    ]:
        """Return the fields the report is equal on."""
        return (
            self._requested,
            tuple(self._accepted.items()),
            tuple(self._rejected.items()),
            tuple(self._inapplicable.items()),
            self._size,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report records the same counts for one request."""
        if type(other) is not CorruptionReport:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the request, the counts and the sample size."""
        return hash((CorruptionReport, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this report."""
        return (
            f"CorruptionReport({self._requested!r}, "
            f"accepted={dict(self._accepted)!r}, rejected={dict(self._rejected)!r}, "
            f"inapplicable={dict(self._inapplicable)!r}, sample_size={self._size!r})"
        )


class IllTypingReport:
    """What injecting ill-typed forms achieved, beside the refusals it collected.

    Parameters
    ----------
    injection : CorruptionReport[IllTypingOperator]
        What the injection requested and achieved, by operator, in the shape
        the syntactic corruption report takes.
    certificates : Mapping[TypingFailureKind, int]
        How many unsatisfied constraints of each kind the certificates hold,
        counted over every injected form. A kind no certificate names has no
        entry.
    observed : Mapping[FormTypingLabel, int]
        How many certified negatives training observes under each typing label.

    Raises
    ------
    TypeError
        If a count is not an integer.
    ValueError
        If a count is not positive.
    """

    __slots__ = ("_certificates", "_injection", "_observed")

    def __init__(
        self,
        injection: CorruptionReport[IllTypingOperator],
        *,
        certificates: Mapping[TypingFailureKind, int],
        observed: Mapping[FormTypingLabel, int],
    ) -> None:
        self._injection = injection
        self._certificates = _by_failure(certificates)
        self._observed = _by_form_label(observed)

        if sum(self._observed.values()) != injection.corrupted:
            msg = (
                f"{sum(self._observed.values())} observed logical-form labels do not "
                f"cover {injection.corrupted} injected forms"
            )
            raise ValueError(msg)

    @classmethod
    def of(
        cls,
        injection: CorruptionReport[IllTypingOperator],
        forms: Iterable[IllTyped],
    ) -> IllTypingReport:
        """Return the report measured over the injected forms.

        Parameters
        ----------
        injection : CorruptionReport[IllTypingOperator]
            What the injection requested and achieved, by operator.
        forms : Iterable[IllTyped]
            The injected forms with their certificates.
        """
        held = tuple(forms)
        certificates: Counter[TypingFailureKind] = Counter(
            kind for form in held for kind in form.certificate.kinds
        )
        observed: Counter[FormTypingLabel] = Counter(
            form.observed_label for form in held
        )

        return cls(injection, certificates=certificates, observed=observed)

    @property
    def injection(self) -> CorruptionReport[IllTypingOperator]:
        """Return what the injection requested and achieved, by operator."""
        return self._injection

    @property
    def certificates(self) -> Mapping[TypingFailureKind, int]:
        """Return how many unsatisfied constraints of each kind were certified."""
        return self._certificates

    @property
    def observed(self) -> Mapping[FormTypingLabel, int]:
        """Return how many injected forms training sees under each label."""
        return self._observed

    @property
    def flipped(self) -> int:
        """Return how many certified negatives training observes as well-typed."""
        return self._observed["well-typed"]

    @property
    def achieved_noise(self) -> float:
        """Return the share of injected forms whose negative label was flipped."""
        total = self._injection.corrupted

        return self.flipped / total if total else 0.0

    def _fields(
        self,
    ) -> tuple[
        CorruptionReport[IllTypingOperator],
        tuple[tuple[TypingFailureKind, int], ...],
        tuple[tuple[FormTypingLabel, int], ...],
    ]:
        """Return the fields the report is equal on."""
        return (
            self._injection,
            tuple(self._certificates.items()),
            tuple(self._observed.items()),
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report records the same injection and refusals."""
        if type(other) is not IllTypingReport:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the injection counts and the certificate counts."""
        return hash((IllTypingReport, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this report."""
        return (
            f"IllTypingReport({self._injection!r}, "
            f"certificates={dict(self._certificates)!r}, "
            f"observed={dict(self._observed)!r})"
        )


class LabelNoise:
    """The rates at which items' grammaticality labels are flipped, by true label.

    Equal rates flip every item alike. A positive ungrammatical rate with no
    grammatical flips contaminates the grammatical data with corrupted items
    observed as grammatical, and at a rate of one every corrupted item is
    observed as grammatical, so the sample is positive-only data holding the
    corruption share of bad strings.

    Parameters
    ----------
    grammatical : float
        The probability that an item the grammar derives is observed as
        ungrammatical.
    ungrammatical : float
        The probability that a corrupted item is observed as grammatical.

    Raises
    ------
    TypeError
        If a rate is not numeric.
    ValueError
        If a rate is outside the unit interval.
    """

    __slots__ = ("_grammatical", "_ungrammatical")

    def __init__(self, *, grammatical: float, ungrammatical: float) -> None:
        self._grammatical = _proportion("a grammatical label-noise rate", grammatical)
        self._ungrammatical = _proportion(
            "an ungrammatical label-noise rate", ungrammatical
        )

    @property
    def grammatical(self) -> float:
        """Return the probability that a derivable item's label is flipped."""
        return self._grammatical

    @property
    def ungrammatical(self) -> float:
        """Return the probability that a corrupted item's label is flipped."""
        return self._ungrammatical

    def rate(self, label: GrammaticalityLabel, /) -> float:
        """Return the probability that an item bearing a true label is flipped.

        Parameters
        ----------
        label : GrammaticalityLabel
            The item's true label.
        """
        return self._grammatical if label == "grammatical" else self._ungrammatical

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value requests the same rates."""
        if type(other) is not LabelNoise:
            return NotImplemented

        return (self._grammatical, self._ungrammatical) == (
            other._grammatical,
            other._ungrammatical,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the rates."""
        return hash((LabelNoise, self._grammatical, self._ungrammatical))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this request."""
        return (
            f"LabelNoise(grammatical={self._grammatical!r}, "
            f"ungrammatical={self._ungrammatical!r})"
        )


class ObservedLabel:
    """One item's true grammaticality label beside the label emitted for it.

    Parameters
    ----------
    item_id : ItemId
        The item the labels belong to.
    true_label : GrammaticalityLabel
        Whether the parser derives the item under the grammar drawn from.
    observed_label : GrammaticalityLabel
        The label a consumer is shown.
    flipped : bool
        Whether the observed label is the other one from the true label.

    Raises
    ------
    ValueError
        If a label is not a grammaticality label, or the indicator disagrees
        with the two labels.
    """

    __slots__ = ("_flipped", "_item_id", "_observed_label", "_true_label")

    def __init__(
        self,
        item_id: ItemId,
        true_label: GrammaticalityLabel,
        observed_label: GrammaticalityLabel,
        *,
        flipped: bool,
    ) -> None:
        held_true = _ensure_label(f"the true label of {item_id!r}", true_label)
        held_observed = _ensure_label(
            f"the observed label of {item_id!r}", observed_label
        )

        if flipped != (held_true != held_observed):
            msg = (
                f"item {item_id!r} is labeled {held_true!r} and observed "
                f"{held_observed!r} but its flip indicator is {flipped!r}"
            )
            raise ValueError(msg)

        self._item_id = item_id
        self._true_label = held_true
        self._observed_label = held_observed
        self._flipped = flipped

    @classmethod
    def of(
        cls, item_id: ItemId, true_label: GrammaticalityLabel, *, flipped: bool
    ) -> ObservedLabel:
        """Return the record of a true label observed as is or flipped.

        Parameters
        ----------
        item_id : ItemId
            The item the labels belong to.
        true_label : GrammaticalityLabel
            Whether the parser derives the item.
        flipped : bool
            Whether the observed label is the other one.
        """
        observed = _other(true_label) if flipped else true_label

        return cls(item_id, true_label, observed, flipped=flipped)

    @property
    def item_id(self) -> ItemId:
        """Return the item the labels belong to."""
        return self._item_id

    @property
    def true_label(self) -> GrammaticalityLabel:
        """Return whether the parser derives the item."""
        return self._true_label

    @property
    def observed_label(self) -> GrammaticalityLabel:
        """Return the label a consumer is shown."""
        return self._observed_label

    @property
    def flipped(self) -> bool:
        """Return whether the observed label differs from the true one."""
        return self._flipped

    def _fields(self) -> tuple[ItemId, GrammaticalityLabel, GrammaticalityLabel, bool]:
        """Return the fields the record is equal on."""
        return (self._item_id, self._true_label, self._observed_label, self._flipped)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another record labels the same item the same way."""
        if type(other) is not ObservedLabel:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the item, the two labels and the indicator."""
        return hash((ObservedLabel, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this record."""
        return (
            f"ObservedLabel({self._item_id!r}, {self._true_label!r}, "
            f"{self._observed_label!r}, flipped={self._flipped!r})"
        )


class NoiseReport:
    """What flipping a sample's labels requested and what it achieved, by true label.

    Parameters
    ----------
    requested : LabelNoise
        The rates the labels were flipped at.
    items : Mapping[GrammaticalityLabel, int]
        How many items bear each true label.
    flipped : Mapping[GrammaticalityLabel, int]
        How many items bearing each true label are observed as the other.

    Raises
    ------
    TypeError
        If a count is not an integer.
    ValueError
        If a count is keyed by an unknown label or is negative, or more
        items of a label are flipped than bear it.
    """

    __slots__ = ("_flipped", "_items", "_requested")

    def __init__(
        self,
        requested: LabelNoise,
        *,
        items: Mapping[GrammaticalityLabel, int],
        flipped: Mapping[GrammaticalityLabel, int],
    ) -> None:
        self._requested = requested
        self._items = _by_label("items", items)
        self._flipped = _by_label("flipped", flipped)

        for label in GRAMMATICALITY_LABELS:
            if self._flipped[label] > self._items[label]:
                msg = (
                    f"{self._flipped[label]} {label} items are flipped but only "
                    f"{self._items[label]} bear the label"
                )
                raise ValueError(msg)

    @classmethod
    def of(cls, requested: LabelNoise, labels: Iterable[ObservedLabel]) -> NoiseReport:
        """Return the report measured over the emitted label records.

        Parameters
        ----------
        requested : LabelNoise
            The rate the labels were flipped at.
        labels : Iterable[ObservedLabel]
            The records, one per item.
        """
        items: Counter[GrammaticalityLabel] = Counter()
        flipped: Counter[GrammaticalityLabel] = Counter()

        for label in labels:
            items[label.true_label] += 1

            if label.flipped:
                flipped[label.true_label] += 1

        return cls(requested, items=items, flipped=flipped)

    @property
    def requested(self) -> LabelNoise:
        """Return the rate the labels were flipped at."""
        return self._requested

    @property
    def items(self) -> Mapping[GrammaticalityLabel, int]:
        """Return how many items bear each true label."""
        return self._items

    @property
    def flipped(self) -> Mapping[GrammaticalityLabel, int]:
        """Return how many items of each true label are observed as the other."""
        return self._flipped

    @property
    def observed(self) -> Mapping[GrammaticalityLabel, int]:
        """Return how many items are observed under each label."""
        return MappingProxyType(
            {
                label: self._items[label]
                - self._flipped[label]
                + self._flipped[_other(label)]
                for label in GRAMMATICALITY_LABELS
            }
        )

    @property
    def sample_size(self) -> int:
        """Return how many items the whole sample holds."""
        return sum(self._items.values())

    @property
    def flip_count(self) -> int:
        """Return how many items are observed under the other label."""
        return sum(self._flipped.values())

    @property
    def achieved_rate(self) -> float:
        """Return the share of the items whose label is flipped, zero when empty."""
        size = self.sample_size

        return self.flip_count / size if size else 0.0

    @property
    def achieved_rates(self) -> Mapping[GrammaticalityLabel, float]:
        """Return the share of each true label's items that are flipped.

        A label no item bears has a share of zero.
        """
        return MappingProxyType(
            {
                label: (
                    self._flipped[label] / self._items[label]
                    if self._items[label]
                    else 0.0
                )
                for label in GRAMMATICALITY_LABELS
            }
        )

    def _fields(
        self,
    ) -> tuple[
        LabelNoise,
        tuple[tuple[GrammaticalityLabel, int], ...],
        tuple[tuple[GrammaticalityLabel, int], ...],
    ]:
        """Return the fields the report is equal on."""
        return (
            self._requested,
            tuple(self._items.items()),
            tuple(self._flipped.items()),
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report records the same counts for one request."""
        if type(other) is not NoiseReport:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the request and the counts."""
        return hash((NoiseReport, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this report."""
        return (
            f"NoiseReport({self._requested!r}, items={dict(self._items)!r}, "
            f"flipped={dict(self._flipped)!r})"
        )


@final
class SymbolInventory(FrozenValue):
    """The non-logical symbols a declared interpretation draws its forms from.

    Parameters
    ----------
    roles : Mapping[str, Mapping[str, int]] | None
        Each declared inventory role with its symbols and their curried
        arities, in declaration order, or nothing for an empty inventory.

    Raises
    ------
    TypeError
        If an arity is not an integer.
    ValueError
        If a role or symbol name is empty, or an arity is negative.
    """

    __slots__ = ("_names", "_roles")

    _noun = "symbol inventory"

    def __init__(
        self,
        roles: Mapping[str, Mapping[str, int]] | None = None,
    ) -> None:
        held: dict[str, Mapping[str, int]] = {}
        names: list[str] = []
        arities: dict[str, int] = {}

        for role, symbols in (roles or {}).items():
            name = ensure_name("an inventory role", role)
            members = MappingProxyType(
                {
                    ensure_name(f"a symbol in role {name!r}", symbol): (
                        ensure_non_negative(f"the arity of {symbol!r}", arity)
                    )
                    for symbol, arity in symbols.items()
                }
            )
            for symbol, arity in members.items():
                previous = arities.get(symbol)
                if previous is not None and previous != arity:
                    msg = (
                        f"symbol {symbol!r} has arity {previous} in one inventory "
                        f"role and {arity} in {name!r}"
                    )
                    raise ValueError(msg)
                arities[symbol] = arity
            held[name] = members
            names.extend(symbol for symbol in members if symbol not in names)

        object.__setattr__(self, "_roles", MappingProxyType(held))
        object.__setattr__(self, "_names", tuple(names))

    @property
    def roles(self) -> Mapping[str, Mapping[str, int]]:
        """Return each declared inventory role with its symbols and arities."""
        return self._roles

    @property
    def names(self) -> tuple[str, ...]:
        """Return every symbol once, in first role-membership order."""
        return self._names

    def in_role(self, role: str, /) -> Mapping[str, int]:
        """Return the symbols bearing one role, or an empty map when absent."""
        return self._roles.get(role, MappingProxyType({}))

    def __len__(self) -> int:
        """Return how many distinct symbols the inventory declares."""
        return len(self._names)

    def _fields(
        self,
    ) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
        """Return the fields the inventory is equal on."""
        return tuple(
            (role, tuple(symbols.items())) for role, symbols in self._roles.items()
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another inventory declares the same symbols and arities."""
        if type(other) is not SymbolInventory:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the declared roles, symbols and arities."""
        return hash((SymbolInventory, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this inventory."""
        roles = {role: dict(symbols) for role, symbols in self._roles.items()}
        return f"SymbolInventory({roles!r})"


class GrammarKnobs:
    """The dimensions a synthetic grammar varies along beside its lexicon.

    Parameters
    ----------
    primitive_count : int
        How many primitive categories the signature declares.
    constructors : tuple[str, ...]
        The category constructors, in signature order.
    combinators : tuple[str, ...]
        The rule names, in declaration order.
    inventory : SymbolInventory | None
        The non-logical symbols of the declared interpretation, or nothing
        when the grammar is drawn from without one.

    Raises
    ------
    TypeError
        If the count is not an integer.
    ValueError
        If the count is negative or a constructor or combinator is named twice.
    """

    __slots__ = ("_combinators", "_constructors", "_inventory", "_primitive_count")

    def __init__(
        self,
        primitive_count: int,
        constructors: tuple[str, ...],
        combinators: tuple[str, ...],
        *,
        inventory: SymbolInventory | None = None,
    ) -> None:
        for field, names in (
            ("constructors", constructors),
            ("combinators", combinators),
        ):
            if len(set(names)) != len(names):
                msg = f"the {field} name one member twice: {names!r}"
                raise ValueError(msg)

        self._primitive_count = ensure_non_negative("primitive_count", primitive_count)
        self._constructors = constructors
        self._combinators = combinators
        self._inventory = inventory

    @property
    def primitive_count(self) -> int:
        """Return how many primitive categories the signature declares."""
        return self._primitive_count

    @property
    def constructors(self) -> tuple[str, ...]:
        """Return the category constructors, in signature order."""
        return self._constructors

    @property
    def combinators(self) -> tuple[str, ...]:
        """Return the rule names, in declaration order."""
        return self._combinators

    @property
    def inventory(self) -> SymbolInventory | None:
        """Return the interpretation's symbols, or nothing without an interpretation."""
        return self._inventory

    def with_inventory(self, inventory: SymbolInventory | None) -> GrammarKnobs:
        """Return the same grammar dimensions beside another symbol inventory.

        Parameters
        ----------
        inventory : SymbolInventory | None
            The symbols to record, or nothing to record none.
        """
        return GrammarKnobs(
            self._primitive_count,
            self._constructors,
            self._combinators,
            inventory=inventory,
        )

    def _fields(
        self,
    ) -> tuple[int, tuple[str, ...], tuple[str, ...], SymbolInventory | None]:
        """Return the knobs in declaration order."""
        return (
            self._primitive_count,
            self._constructors,
            self._combinators,
            self._inventory,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value sets the same knobs."""
        if type(other) is not GrammarKnobs:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the knobs."""
        return hash((GrammarKnobs, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds these knobs."""
        return (
            f"GrammarKnobs({self._primitive_count!r}, {self._constructors!r}, "
            f"{self._combinators!r}, inventory={self._inventory!r})"
        )


class LexiconReport:
    """The joint of category and token frequency measured over a sample's lexicon.

    Parameters
    ----------
    classes : Mapping[str, Histogram]
        Over each category of the lexicon, by its rendered text: how many of
        the forms bearing it were drawn as a token of that category each
        number of times, a form never drawn in the zero bucket. A category's
        class size is the histogram's total and its token count the
        histogram's sum.
    forms : Histogram
        Over the forms of the lexicon: how many were drawn each number of
        times, whatever category they bore, a form never drawn in the zero
        bucket.
    per_type_ambiguity : Histogram
        Over the forms of the lexicon: how many categories each form bears.
    ambiguity_frequency_correlation : float | None
        The achieved rank correlation between a form's ambiguity and token
        count, or nothing when either quantity does not vary.

    Raises
    ------
    ValueError
        If a category bears no form, the tokens counted by category differ
        from the tokens counted by form, the form and ambiguity counts differ,
        or a supplied correlation lies outside [-1, 1].
    """

    __slots__ = (
        "_ambiguity_frequency_correlation",
        "_classes",
        "_forms",
        "_per_type_ambiguity",
    )

    def __init__(
        self,
        classes: Mapping[str, Histogram],
        forms: Histogram,
        per_type_ambiguity: Histogram,
        ambiguity_frequency_correlation: float | None,
    ) -> None:
        held: dict[str, Histogram] = {}

        for category, histogram in classes.items():
            if histogram.total == 0:
                msg = f"category {category!r} bears no form"
                raise ValueError(msg)

            held[category] = histogram

        by_category = sum(_tokens(histogram) for histogram in held.values())
        by_form = _tokens(forms)

        if by_category != by_form:
            msg = (
                f"the categories count {by_category} tokens but the forms "
                f"count {by_form}"
            )
            raise ValueError(msg)

        if forms.total != per_type_ambiguity.total:
            msg = (
                f"the report counts {forms.total} forms but measures ambiguity "
                f"for {per_type_ambiguity.total}"
            )
            raise ValueError(msg)

        correlation = (
            None
            if ambiguity_frequency_correlation is None
            else ensure_finite(
                "an ambiguity-frequency correlation",
                ambiguity_frequency_correlation,
            )
        )

        if correlation is not None and not -1.0 <= correlation <= 1.0:
            msg = (
                "an ambiguity-frequency correlation lies outside [-1, 1]: "
                f"{correlation!r}"
            )
            raise ValueError(msg)

        self._classes = MappingProxyType(held)
        self._forms = forms
        self._per_type_ambiguity = per_type_ambiguity
        self._ambiguity_frequency_correlation = correlation

    @property
    def classes(self) -> Mapping[str, Histogram]:
        """Return the per-category token counts of the forms bearing each category."""
        return self._classes

    @property
    def forms(self) -> Histogram:
        """Return the token counts of the forms, whatever category they bore."""
        return self._forms

    @property
    def realized_size(self) -> int:
        """Return how many form types the lexicon holds."""
        return self._forms.total

    @property
    def realized_vocabulary_size(self) -> int:
        """Return how many form types occur in the generated strings."""
        return sum(
            count for frequency, count in self._forms.counts.items() if frequency > 0
        )

    @property
    def per_type_ambiguity(self) -> Histogram:
        """Return the categories-per-form histogram over form types."""
        return self._per_type_ambiguity

    @property
    def ambiguity_frequency_correlation(self) -> float | None:
        """Return the achieved ambiguity-frequency rank correlation."""
        return self._ambiguity_frequency_correlation

    @property
    def type_token_ratio(self) -> float:
        """Return observed form types divided by token count, zero without tokens."""
        tokens = _tokens(self._forms)

        if tokens == 0:
            return 0.0

        observed = sum(
            count for value, count in self._forms.counts.items() if value > 0
        )

        return observed / tokens

    @property
    def class_sizes(self) -> Mapping[str, int]:
        """Return how many forms bear each category."""
        return MappingProxyType(
            {category: histogram.total for category, histogram in self._classes.items()}
        )

    @property
    def token_counts(self) -> Mapping[str, int]:
        """Return how many tokens bore each category."""
        return MappingProxyType(
            {
                category: _tokens(histogram)
                for category, histogram in self._classes.items()
            }
        )

    @property
    def token_shares(self) -> Mapping[str, float]:
        """Return the fraction of the tokens that bore each category, zero when none."""
        counts = self.token_counts
        total = sum(counts.values())

        return MappingProxyType(
            {
                category: count / total if total else 0.0
                for category, count in counts.items()
            }
        )

    @property
    def rank_frequency_exponent(self) -> float | None:
        """Return the exponent fitted to the rank-frequency curve of the forms.

        The forms drawn at least once are ranked by token count, and the
        negated least-squares slope of log count on log rank is returned, or
        nothing when fewer than two forms were drawn.
        """
        counts = sorted(
            (
                value
                for value, count in self._forms.counts.items()
                if value > 0
                for _ in range(count)
            ),
            reverse=True,
        )
        slope = log_log_slope(
            tuple(float(rank) for rank in range(1, len(counts) + 1)),
            tuple(float(count) for count in counts),
        )

        return None if slope is None else -slope

    @property
    def coupling(self) -> float | None:
        """Return the coupling fitted to the categories' sizes and token counts.

        Over the categories drawn at least once, the negated least-squares
        slope of the log of tokens per form on the log of class size is
        returned, or nothing when fewer than two distinct class sizes were
        drawn.
        """
        drawn = [
            (histogram.total, _tokens(histogram) / histogram.total)
            for histogram in self._classes.values()
            if _tokens(histogram) > 0
        ]
        slope = log_log_slope(
            tuple(float(size) for size, _ in drawn),
            tuple(per_form for _, per_form in drawn),
        )

        return None if slope is None else -slope

    @property
    def concordance(self) -> float | None:
        """Return the rank correlation of entry token count with inverse class size.

        Every form-category pair of the lexicon is one unit. The correlation
        is near one when the most frequent forms belong to the smallest
        classes, near zero when frequency says nothing about class size, and
        near minus one when the most frequent forms belong to the largest
        classes, or nothing when the counts or the sizes do not vary.
        """
        frequencies: list[float] = []
        inverse_sizes: list[float] = []

        for histogram in self._classes.values():
            for value, count in histogram.counts.items():
                frequencies.extend([float(value)] * count)
                inverse_sizes.extend([1.0 / histogram.total] * count)

        return spearman(frequencies, inverse_sizes)

    def _fields(
        self,
    ) -> tuple[tuple[tuple[str, Histogram], ...], Histogram, Histogram, float | None]:
        """Return the fields the report is equal on."""
        return (
            tuple(self._classes.items()),
            self._forms,
            self._per_type_ambiguity,
            self._ambiguity_frequency_correlation,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report measured the same joint."""
        if type(other) is not LexiconReport:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the per-category and per-form histograms."""
        return hash((LexiconReport, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this report."""
        return (
            f"LexiconReport({dict(self._classes)!r}, {self._forms!r}, "
            f"{self._per_type_ambiguity!r}, "
            f"{self._ambiguity_frequency_correlation!r})"
        )


class InventoryReport:
    """The lexical category inventory a sample was drawn over.

    Parameters
    ----------
    categories : Mapping[str, int]
        Each category of the lexicon by its rendered text, in the order the
        lexicon first offers it, with how deeply constructors nest in it: a
        primitive has depth zero and a constructor one more than its deepest
        argument.
    argument_counts : Mapping[str, int]
        Each category's functor-spine argument count, in category order.
    composition_orders : Mapping[str, int]
        Each category's order under the grammar's declared functor roles, in
        category order.
    max_arguments : int | None
        The declared argument-count bound, or no bound.
    max_order : int | None
        The declared category-order bound, or no bound.
    primitives : tuple[str, ...]
        The primitive categories occurring in some lexical category, a
        feature-marked primitive counted as its bare primitive, in signature
        order.

    Raises
    ------
    TypeError
        If a depth is not an integer.
    ValueError
        If the inventory is empty, the measurement maps name different
        categories, a measurement is negative or a primitive is named twice.
    """

    __slots__ = (
        "_argument_counts",
        "_categories",
        "_composition_orders",
        "_max_arguments",
        "_max_order",
        "_primitives",
    )

    def __init__(
        self,
        categories: Mapping[str, int],
        argument_counts: Mapping[str, int],
        composition_orders: Mapping[str, int],
        primitives: tuple[str, ...],
        *,
        max_arguments: int | None,
        max_order: int | None,
    ) -> None:
        if not categories:
            msg = "a category inventory is empty"
            raise ValueError(msg)

        names = tuple(categories)

        if tuple(argument_counts) != names or tuple(composition_orders) != names:
            msg = "the inventory measurements name different lexical categories"
            raise ValueError(msg)

        if len(set(primitives)) != len(primitives):
            msg = f"the primitives name one member twice: {primitives!r}"
            raise ValueError(msg)

        self._categories = MappingProxyType(
            {
                category: ensure_non_negative(f"the depth of {category!r}", depth)
                for category, depth in categories.items()
            }
        )
        self._argument_counts = MappingProxyType(
            {
                category: ensure_non_negative(
                    f"the argument count of {category!r}", count
                )
                for category, count in argument_counts.items()
            }
        )
        self._composition_orders = MappingProxyType(
            {
                category: ensure_non_negative(
                    f"the composition order of {category!r}", order
                )
                for category, order in composition_orders.items()
            }
        )
        self._max_arguments = (
            None
            if max_arguments is None
            else ensure_non_negative("the argument-count bound", max_arguments)
        )
        self._max_order = (
            None
            if max_order is None
            else ensure_non_negative("the category-order bound", max_order)
        )
        self._primitives = primitives

    @property
    def categories(self) -> Mapping[str, int]:
        """Return each lexical category's depth, by rendered text."""
        return self._categories

    @property
    def argument_counts(self) -> Mapping[str, int]:
        """Return each lexical category's functor-spine argument count."""
        return self._argument_counts

    @property
    def composition_orders(self) -> Mapping[str, int]:
        """Return each lexical category's order under declared functor roles."""
        return self._composition_orders

    @property
    def max_arguments(self) -> int | None:
        """Return the declared argument-count bound, when one is set."""
        return self._max_arguments

    @property
    def max_order(self) -> int | None:
        """Return the declared category-order bound, when one is set."""
        return self._max_order

    @property
    def primitives(self) -> tuple[str, ...]:
        """Return the primitives occurring in the lexicon, in signature order."""
        return self._primitives

    @property
    def realized_primitive_count(self) -> int:
        """Return how many primitive types occur in the lexicon's categories."""
        return len(self._primitives)

    @property
    def size(self) -> int:
        """Return how many distinct categories the lexicon holds."""
        return len(self._categories)

    @property
    def depths(self) -> Histogram:
        """Return how many categories have each depth."""
        return Histogram.of(self._categories.values())

    @property
    def arguments(self) -> Histogram:
        """Return how many categories have each argument count."""
        return Histogram.of(self._argument_counts.values())

    @property
    def orders(self) -> Histogram:
        """Return how many categories have each composition order."""
        return Histogram.of(self._composition_orders.values())

    def _fields(
        self,
    ) -> tuple[
        tuple[tuple[str, int], ...],
        tuple[tuple[str, int], ...],
        tuple[tuple[str, int], ...],
        tuple[str, ...],
        int | None,
        int | None,
    ]:
        """Return the fields the report is equal on."""
        return (
            tuple(self._categories.items()),
            tuple(self._argument_counts.items()),
            tuple(self._composition_orders.items()),
            self._primitives,
            self._max_arguments,
            self._max_order,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report holds the same inventory."""
        if type(other) is not InventoryReport:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the categories and the primitives."""
        return hash((InventoryReport, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this report."""
        return (
            f"InventoryReport({dict(self._categories)!r}, "
            f"{dict(self._argument_counts)!r}, "
            f"{dict(self._composition_orders)!r}, {self._primitives!r}, "
            f"max_arguments={self._max_arguments!r}, max_order={self._max_order!r})"
        )


def _tokens(histogram: Histogram) -> int:
    """Return how many observations a histogram of counts sums to."""
    return sum(value * count for value, count in histogram.counts.items())


class SyntheticCorpus:
    """Sampled examples beside the bytes of the grammar they were drawn from.

    Parameters
    ----------
    declaration : bytes
        The canonical bytes of the grammar declaration.
    theory : bytes
        The canonical bytes of the compiled theory.
    seed : int
        The seed the sample was drawn with.
    length : LengthReport
        The token counts the grammatical items achieved, beside the
        distribution they were drawn from. Its bounds bound every item: a
        grammatical item on both sides, a corrupted item above only, since
        a deletion from a source of the least length is one token shorter.
    knobs : GrammarKnobs
        The dimensions of the grammar the sample was drawn from.
    examples : tuple[Example[GoldDerivation[str]], ...]
        The grammatical items, each with its gold derivation, in draw order.
    ambiguity : AmbiguityReport
        The lexical and derivational ambiguity measured over the grammatical
        items.
    lexicon : LexiconReport
        The joint of category and token frequency measured over the
        grammatical items' tokens.
    inventory : InventoryReport
        The lexical categories the sample was drawn over, with their depths
        and the primitives they are built from. It names the categories the
        lexicon report counts.
    corrupted : tuple[Corrupted, ...]
        The non-derivable items, in draw order.
    corruption : CorruptionReport[CorruptionOperator] | None
        What corrupting the sample requested and achieved, or nothing when no
        corruption was requested.
    labels : tuple[ObservedLabel, ...]
        One record per item, in draw order: its true label, which is
        grammatical for an example and ungrammatical for a corrupted item,
        beside the label emitted for it and whether the two differ.
    noise : NoiseReport | None
        What flipping the labels requested and achieved, or nothing when no
        noise was requested, in which case no label is flipped.
    forms : tuple[GoldForm, ...]
        The logical form of every grammatical item, in draw order, or none
        when the sample was drawn without an interpretation.
    ill_typed : tuple[IllTyped, ...]
        The refused logical forms injected among the grammatical items, in
        draw order, each with the well-typed form it corrupts.
    ill_typing : IllTypingReport | None
        What injecting refused forms requested and achieved, or nothing when
        no injection was requested.
    scope : ScopeReport | None
        The derivation and reading counts of every grammatical item beside
        the target they were drawn under, or nothing when the sample was
        drawn without a scope target.
    shape : ShapeReport | None
        The accepted derivation-shape measurements beside their target, or
        nothing when no shape target was declared.

    Raises
    ------
    TypeError
        If the seed is not an integer.
    ValueError
        If either byte string is empty, the inventory and the lexicon report
        name different categories, an item is longer than the upper
        bound, a grammatical item is shorter than the lower bound, the length
        report's histogram is not that of the grammatical items, corrupted
        items are held without a corruption report, the corruption report's
        sample size is not the item count, the labels do not name every item
        exactly once, a true label disagrees with the item it names, a label
        is flipped without a noise report, the noise report's counts are
        not those of the labels, the forms do not name every grammatical
        item exactly once, an injected form names an item without a
        form or does not corrupt that item's form, injected forms are held
        without an ill-typing report, the ill-typing report does not count
        the grammatical items or the certificates it was built from, or the
        scope report does not count every grammatical item exactly once in
        draw order.
    """

    __slots__ = (
        "_ambiguity",
        "_corrupted",
        "_corruption",
        "_declaration",
        "_declaration_digest",
        "_examples",
        "_forms",
        "_ill_typed",
        "_ill_typing",
        "_inventory",
        "_items",
        "_knobs",
        "_labels",
        "_length",
        "_lexicon",
        "_noise",
        "_scope",
        "_seed",
        "_shape",
        "_theory",
        "_theory_digest",
    )

    def __init__(
        self,
        declaration: bytes,
        theory: bytes,
        *,
        seed: int,
        length: LengthReport,
        knobs: GrammarKnobs,
        examples: tuple[Example[GoldDerivation[str]], ...],
        ambiguity: AmbiguityReport,
        lexicon: LexiconReport,
        inventory: InventoryReport,
        corrupted: tuple[Corrupted, ...] = (),
        corruption: CorruptionReport[CorruptionOperator] | None = None,
        labels: tuple[ObservedLabel, ...],
        noise: NoiseReport | None = None,
        forms: tuple[GoldForm, ...] = (),
        ill_typed: tuple[IllTyped, ...] = (),
        ill_typing: IllTypingReport | None = None,
        scope: ScopeReport | None = None,
        shape: ShapeReport | None = None,
    ) -> None:
        for field, held in (("declaration", declaration), ("theory", theory)):
            if not held:
                msg = f"a synthetic corpus has empty {field} bytes"
                raise ValueError(msg)

        if not isinstance(seed, int) or isinstance(seed, bool):
            msg = f"a synthetic corpus seed is not an integer: {seed!r}"
            raise TypeError(msg)

        if set(inventory.categories) != set(lexicon.classes):
            msg = (
                f"the inventory names {sorted(inventory.categories)!r} but the "
                f"lexicon report counts {sorted(lexicon.classes)!r}"
            )
            raise ValueError(msg)

        lower = length.distribution.min_length
        bound = length.distribution.max_length

        for example in examples:
            width = len(example.item.tokens)

            if not lower <= width <= bound:
                msg = (
                    f"item {example.item.item_id!r} has {width} tokens, "
                    f"outside the bounds {lower} to {bound}"
                )
                raise ValueError(msg)

        achieved = Histogram.of(len(example.item.tokens) for example in examples)

        if achieved != length.achieved:
            msg = (
                f"the length report holds {length.achieved!r} but the "
                f"grammatical items measure {achieved!r}"
            )
            raise ValueError(msg)

        for record in corrupted:
            if len(record.item.tokens) > bound:
                msg = (
                    f"item {record.item.item_id!r} has "
                    f"{len(record.item.tokens)} tokens, more than the bound {bound}"
                )
                raise ValueError(msg)

        if corruption is None:
            if corrupted:
                msg = f"{len(corrupted)} corrupted items are held without a report"
                raise ValueError(msg)
        elif corruption.sample_size != len(examples) + len(corrupted):
            msg = (
                f"the corruption report counts a sample of "
                f"{corruption.sample_size} but the corpus holds "
                f"{len(examples) + len(corrupted)} items"
            )
            raise ValueError(msg)

        items = _labeled_items(examples, corrupted, labels)

        if noise is None:
            flipped = sum(1 for label in labels if label.flipped)

            if flipped:
                msg = f"{flipped} labels are flipped without a noise report"
                raise ValueError(msg)
        elif noise != NoiseReport.of(noise.requested, labels):
            msg = (
                f"the noise report holds {noise!r} but the labels measure "
                f"{NoiseReport.of(noise.requested, labels)!r}"
            )
            raise ValueError(msg)

        interpreted = _interpreted_items(examples, forms)
        _check_ill_typed(interpreted, ill_typed, ill_typing)

        if scope is not None:
            counted = tuple(count.item_id for count in scope.counts)
            drawn = tuple(example.item.item_id for example in examples)

            if counted != drawn:
                msg = (
                    f"the scope report counts {len(counted)} items but the "
                    f"corpus holds {len(drawn)} grammatical items"
                )
                raise ValueError(msg)

        if shape is not None and len(shape.measurements) != len(examples):
            msg = (
                f"the shape report measures {len(shape.measurements)} items but "
                f"the corpus holds {len(examples)} grammatical items"
            )
            raise ValueError(msg)
        if shape is not None:
            for example, measurement in zip(examples, shape.measurements, strict=True):
                target = shape.target.target(example.item.split)
                if target is None:
                    msg = (
                        f"the shape schedule has no target for item "
                        f"{example.item.item_id!r} in split {example.item.split!r}"
                    )
                    raise ValueError(msg)
                if not target.admits(measurement):
                    msg = (
                        f"item {example.item.item_id!r} measures {measurement!r}, "
                        f"outside its split's shape target {target!r}"
                    )
                    raise ValueError(msg)

        self._declaration = declaration
        self._declaration_digest = sha256(declaration).hexdigest()
        self._theory = theory
        self._theory_digest = sha256(theory).hexdigest()
        self._seed = seed
        self._length = length
        self._knobs = knobs
        self._examples = examples
        self._ambiguity = ambiguity
        self._lexicon = lexicon
        self._inventory = inventory
        self._corrupted = corrupted
        self._corruption = corruption
        self._labels = labels
        self._noise = noise
        self._items = items
        self._forms = interpreted
        self._ill_typed = ill_typed
        self._ill_typing = ill_typing
        self._scope = scope
        self._shape = shape

    @property
    def declaration(self) -> bytes:
        """Return the canonical bytes of the grammar declaration."""
        return self._declaration

    @property
    def declaration_digest(self) -> str:
        """Return the SHA-256 of the declaration bytes as lowercase hexadecimal."""
        return self._declaration_digest

    @property
    def theory(self) -> bytes:
        """Return the canonical bytes of the compiled theory."""
        return self._theory

    @property
    def theory_digest(self) -> str:
        """Return the SHA-256 of the theory bytes as lowercase hexadecimal."""
        return self._theory_digest

    @property
    def seed(self) -> int:
        """Return the seed the sample was drawn with."""
        return self._seed

    @property
    def length(self) -> LengthReport:
        """Return the token counts achieved beside the distribution drawn from."""
        return self._length

    @property
    def min_length(self) -> int:
        """Return the shortest token sequence a grammatical item may hold."""
        return self._length.distribution.min_length

    @property
    def max_length(self) -> int:
        """Return the longest token sequence the sample may hold."""
        return self._length.distribution.max_length

    @property
    def knobs(self) -> GrammarKnobs:
        """Return the dimensions of the grammar the sample was drawn from."""
        return self._knobs

    @property
    def examples(self) -> tuple[Example[GoldDerivation[str]], ...]:
        """Return the grammatical items with their gold derivations, in draw order."""
        return self._examples

    @property
    def ambiguity(self) -> AmbiguityReport:
        """Return the ambiguity measured over the grammatical items."""
        return self._ambiguity

    @property
    def lexicon(self) -> LexiconReport:
        """Return the joint of category and frequency measured over the sample."""
        return self._lexicon

    @property
    def inventory(self) -> InventoryReport:
        """Return the lexical categories the sample was drawn over."""
        return self._inventory

    @property
    def corrupted(self) -> tuple[Corrupted, ...]:
        """Return the non-derivable items, in draw order."""
        return self._corrupted

    @property
    def corruption(self) -> CorruptionReport[CorruptionOperator] | None:
        """Return what corrupting the sample requested and achieved, if requested."""
        return self._corruption

    @property
    def labels(self) -> tuple[ObservedLabel, ...]:
        """Return the true and observed label of every item, in draw order."""
        return self._labels

    @property
    def noise(self) -> NoiseReport | None:
        """Return what flipping the labels requested and achieved, if requested."""
        return self._noise

    @property
    def forms(self) -> Mapping[ItemId, GoldForm]:
        """Return the logical form of each grammatical item, keyed by item."""
        return self._forms

    @property
    def ill_typed(self) -> tuple[IllTyped, ...]:
        """Return the refused logical forms injected among the items, in draw order."""
        return self._ill_typed

    @property
    def ill_typing(self) -> IllTypingReport | None:
        """Return what injecting refused forms requested and achieved, if requested."""
        return self._ill_typing

    @property
    def form_typing(self) -> Mapping[ItemId, tuple[FormTypingExample, ...]]:
        """Return the labeled logical forms available to training, by item."""
        injected: dict[ItemId, list[FormTypingExample]] = {}

        for form in self._ill_typed:
            injected.setdefault(form.item_id, []).append(
                FormTypingExample(
                    form.item_id,
                    form.text,
                    form.true_label,
                    form.observed_label,
                )
            )

        return MappingProxyType(
            {
                item_id: (
                    FormTypingExample(
                        item_id,
                        gold.text,
                        "well-typed",
                        "well-typed",
                    ),
                    *injected.get(item_id, ()),
                )
                for item_id, gold in self._forms.items()
            }
        )

    @property
    def scope(self) -> ScopeReport | None:
        """Return the readings measured over the grammatical items, if targeted."""
        return self._scope

    @property
    def shape(self) -> ShapeReport | None:
        """Return derivation shapes measured under a target, if targeted."""
        return self._shape

    def observed(self) -> tuple[Example[Categorical], ...]:
        """Return every item with its observed label as a categorical response.

        The examples are in draw order, and the response's inventory is the
        two grammaticality labels.
        """
        return tuple(
            Example(
                self._items[label.item_id],
                Categorical(label.observed_label, _LABEL_SET),
            )
            for label in self._labels
        )

    def __len__(self) -> int:
        """Return how many items the corpus holds, corrupted ones included."""
        return len(self._examples) + len(self._corrupted)

    def _fields(
        self,
    ) -> tuple[
        bytes,
        bytes,
        int,
        LengthReport,
        GrammarKnobs,
        tuple[Example[GoldDerivation[str]], ...],
        AmbiguityReport,
        LexiconReport,
        InventoryReport,
        tuple[Corrupted, ...],
        CorruptionReport[CorruptionOperator] | None,
        tuple[ObservedLabel, ...],
        NoiseReport | None,
        tuple[GoldForm, ...],
        tuple[IllTyped, ...],
        IllTypingReport | None,
        ScopeReport | None,
        ShapeReport | None,
    ]:
        """Return the fields the corpus is equal on."""
        return (
            self._declaration,
            self._theory,
            self._seed,
            self._length,
            self._knobs,
            self._examples,
            self._ambiguity,
            self._lexicon,
            self._inventory,
            self._corrupted,
            self._corruption,
            self._labels,
            self._noise,
            tuple(self._forms.values()),
            self._ill_typed,
            self._ill_typing,
            self._scope,
            self._shape,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value holds the same sample of the same grammar."""
        if type(other) is not SyntheticCorpus:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the bytes, the settings, the items and the reports."""
        return hash((SyntheticCorpus, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the declaration, the seed and the sizes."""
        return (
            f"SyntheticCorpus(declaration={self._declaration_digest[:12]!r}, "
            f"seed={self._seed!r}, length={self._length.distribution!r}, "
            f"examples={len(self._examples)}, corrupted={len(self._corrupted)})"
        )


def _interpreted_items(
    examples: tuple[Example[GoldDerivation[str]], ...],
    forms: tuple[GoldForm, ...],
) -> Mapping[ItemId, GoldForm]:
    """Return the forms by item once found to name each grammatical item once.

    Parameters
    ----------
    examples : tuple[Example[GoldDerivation[str]], ...]
        The grammatical items.
    forms : tuple[GoldForm, ...]
        The forms to check against the items, or none.

    Raises
    ------
    ValueError
        If a form names no grammatical item or names one twice, or some
        forms are held while an item has none.
    """
    if not forms:
        return MappingProxyType({})

    grammatical = {example.item.item_id for example in examples}
    held: dict[ItemId, GoldForm] = {}

    for form in forms:
        if form.item_id not in grammatical:
            msg = f"form {form.item_id!r} names no grammatical item of the corpus"
            raise ValueError(msg)

        if form.item_id in held:
            msg = f"item {form.item_id!r} bears two logical forms"
            raise ValueError(msg)

        held[form.item_id] = form

    if len(held) != len(grammatical):
        msg = f"{len(grammatical) - len(held)} grammatical items bear no logical form"
        raise ValueError(msg)

    return MappingProxyType(held)


def _check_ill_typed(
    forms: Mapping[ItemId, GoldForm],
    ill_typed: tuple[IllTyped, ...],
    report: IllTypingReport | None,
) -> None:
    """Require every injected form to corrupt an item's form and the report to count it.

    Parameters
    ----------
    forms : Mapping[ItemId, GoldForm]
        The well-typed forms by item.
    ill_typed : tuple[IllTyped, ...]
        The injected forms.
    report : IllTypingReport | None
        The report over the injection, or nothing when none was requested.

    Raises
    ------
    ValueError
        If an injected form names an item without a form or does not corrupt
        that item's form, forms are injected without a report, or the report
        does not count the grammatical items, the operators the forms were
        drawn by, or the certificates they hold.
    """
    for record in ill_typed:
        held = forms.get(record.item_id)

        if held is None:
            msg = f"injected form of {record.item_id!r} bears no logical form"
            raise ValueError(msg)

        if held.text != record.source:
            msg = (
                f"the injected form of {record.item_id!r} corrupts "
                f"{record.source!r} but the item bears {held.text!r}"
            )
            raise ValueError(msg)

    if report is None:
        if ill_typed:
            msg = f"{len(ill_typed)} injected forms are held without a report"
            raise ValueError(msg)

        return

    if report.injection.sample_size != len(forms):
        msg = (
            f"the ill-typing report counts a sample of "
            f"{report.injection.sample_size} but the corpus holds "
            f"{len(forms)} grammatical items"
        )
        raise ValueError(msg)

    drawn: Counter[str] = Counter(record.kind for record in ill_typed)
    accepted = {name: drawn[name] for name in report.injection.requested.operator_names}

    if dict(report.injection.accepted) != accepted:
        msg = (
            f"the ill-typing report accepts {dict(report.injection.accepted)!r} "
            f"but the injected forms were drawn by {accepted!r}"
        )
        raise ValueError(msg)

    measured = IllTypingReport.of(report.injection, ill_typed)

    if report != measured:
        msg = (
            f"the ill-typing report holds {report!r} but the injected forms "
            f"measure {measured!r}"
        )
        raise ValueError(msg)


def _labeled_items(
    examples: tuple[Example[GoldDerivation[str]], ...],
    corrupted: tuple[Corrupted, ...],
    labels: tuple[ObservedLabel, ...],
) -> Mapping[ItemId, Item]:
    """Return the items by identifier once the labels are found to name each once.

    Parameters
    ----------
    examples : tuple[Example[GoldDerivation[str]], ...]
        The grammatical items.
    corrupted : tuple[Corrupted, ...]
        The non-derivable items.
    labels : tuple[ObservedLabel, ...]
        The records to check against the items.

    Raises
    ------
    ValueError
        If two items share an identifier, a label names no item or names one
        twice, an item has no label, or a true label disagrees with the item
        it names.
    """
    truths: dict[ItemId, GrammaticalityLabel] = {}
    items: dict[ItemId, Item] = {}

    for example in examples:
        truths[example.item.item_id] = "grammatical"
        items[example.item.item_id] = example.item

    for record in corrupted:
        truths[record.item.item_id] = "ungrammatical"
        items[record.item.item_id] = record.item

    if len(truths) != len(examples) + len(corrupted):
        msg = (
            f"{len(examples) + len(corrupted) - len(truths)} items share an "
            f"identifier with another"
        )
        raise ValueError(msg)

    seen: set[ItemId] = set()

    for label in labels:
        if label.item_id not in truths:
            msg = f"label record {label.item_id!r} names no item of the corpus"
            raise ValueError(msg)

        if label.item_id in seen:
            msg = f"item {label.item_id!r} is labeled twice"
            raise ValueError(msg)

        if label.true_label != truths[label.item_id]:
            msg = (
                f"item {label.item_id!r} is {truths[label.item_id]} but its "
                f"record says {label.true_label!r}"
            )
            raise ValueError(msg)

        seen.add(label.item_id)

    if len(seen) != len(truths):
        msg = f"{len(truths) - len(seen)} items have no label record"
        raise ValueError(msg)

    return MappingProxyType(items)


class SyntheticDataset:
    """A synthetic corpus beside the declaration and the grammar it was drawn under.

    Parameters
    ----------
    declaration : bytes
        The canonical bytes of the corpus declaration.
    source_grammar : GrammarAssembly
        The compiled grammar the corpus was drawn under.
    corpus : SyntheticCorpus
        The drawn items with their reports.
    source_version : str
        The declaration version the dataset was lowered from.
    target_version : str
        The native version the dataset realizes.
    source_interpretation : InterpretationFunction | None
        The interpretation the forms were read under, or nothing when the
        corpus was drawn without one.
    semantic : SemanticReport | None
        The generated semantic assignment measured over the corpus, or nothing
        when meanings were listed.

    Raises
    ------
    ValueError
        If the declaration bytes are empty, the corpus holds the bytes of
        another grammar than the assembly's, or the corpus holds forms
        without an interpretation or an interpretation without forms while
        holding grammatical items.
    """

    __slots__ = (
        "_corpus",
        "_declaration",
        "_fingerprint",
        "_semantic",
        "_source_grammar",
        "_source_interpretation",
        "_source_version",
        "_target_version",
    )

    def __init__(
        self,
        declaration: bytes,
        source_grammar: GrammarAssembly,
        corpus: SyntheticCorpus,
        *,
        source_version: str,
        target_version: str,
        source_interpretation: InterpretationFunction | None = None,
        semantic: SemanticReport | None = None,
    ) -> None:
        if not declaration:
            msg = "a synthetic dataset has empty declaration bytes"
            raise ValueError(msg)

        if corpus.declaration != source_grammar.declaration:
            msg = (
                f"the corpus was drawn under grammar "
                f"{corpus.declaration_digest[:12]!r} but the dataset holds "
                f"grammar {source_grammar.declaration_digest[:12]!r}"
            )
            raise ValueError(msg)

        if corpus.examples and source_interpretation is None and corpus.forms:
            msg = (
                f"the corpus holds {len(corpus.forms)} logical forms but the "
                f"dataset holds no interpretation"
            )
            raise ValueError(msg)

        if corpus.examples and source_interpretation is not None and not corpus.forms:
            msg = "the dataset holds an interpretation but the corpus holds no forms"
            raise ValueError(msg)

        generated = (
            None
            if source_interpretation is None
            else source_interpretation.semantic_lexicon
        )

        if (semantic is None) != (generated is None):
            msg = "a generated semantic lexicon and its report are held together"
            raise ValueError(msg)

        if semantic is not None and semantic.lexicon != generated:
            msg = "the semantic report measures another generated lexicon"
            raise ValueError(msg)

        self._declaration = declaration
        self._fingerprint = sha256(declaration).hexdigest()
        self._source_grammar = source_grammar
        self._corpus = corpus
        self._source_version = source_version
        self._target_version = target_version
        self._source_interpretation = source_interpretation
        self._semantic = semantic

    @property
    def declaration(self) -> bytes:
        """Return the canonical bytes of the corpus declaration."""
        return self._declaration

    @property
    def fingerprint(self) -> str:
        """Return the SHA-256 of the declaration bytes as lowercase hexadecimal."""
        return self._fingerprint

    @property
    def source_grammar(self) -> GrammarAssembly:
        """Return the compiled grammar the corpus was drawn under."""
        return self._source_grammar

    @property
    def corpus(self) -> SyntheticCorpus:
        """Return the drawn items with their reports."""
        return self._corpus

    @property
    def source_interpretation(
        self,
    ) -> InterpretationFunction | None:
        """Return the interpretation the forms were read under, if one was declared."""
        return self._source_interpretation

    @property
    def semantic(self) -> SemanticReport | None:
        """Return the generated semantic assignment's report, if one was drawn."""
        return self._semantic

    @property
    def source_version(self) -> str:
        """Return the declaration version the dataset was lowered from."""
        return self._source_version

    @property
    def target_version(self) -> str:
        """Return the native version the dataset realizes."""
        return self._target_version

    def __len__(self) -> int:
        """Return how many items the corpus holds, corrupted ones included."""
        return len(self._corpus)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another dataset holds the same declaration and corpus."""
        if type(other) is not SyntheticDataset:
            return NotImplemented

        return (self._declaration, self._corpus, self._semantic) == (
            other._declaration,
            other._corpus,
            other._semantic,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the fingerprint and the corpus."""
        return hash((SyntheticDataset, self._fingerprint, self._corpus, self._semantic))

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the fingerprint and the corpus."""
        return (
            f"SyntheticDataset(fingerprint={self._fingerprint[:12]!r}, "
            f"corpus={self._corpus!r})"
        )
