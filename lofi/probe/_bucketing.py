"""Deterministic partitions of distributed-alignment outcomes."""

from fractions import Fraction
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, override

from lofi._checks import (
    ensure_finite,
    ensure_name,
    ensure_non_negative,
    ensure_positive,
)
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "AbstractionBucket",
    "AbstractionCase",
    "AbstractionFeatures",
    "BucketingReport",
    "bucket_abstraction",
]

type _FeatureName = Literal[
    "sentence_length",
    "derivation_depth",
    "ambiguity",
    "span_length",
]
type _BucketLabel = Literal["well", "under"]
type _Comparison = Literal["<=", ">"]

_FEATURE_ORDER: Final[tuple[_FeatureName, ...]] = (
    "sentence_length",
    "derivation_depth",
    "ambiguity",
    "span_length",
)


class _Condition(NamedTuple):
    """One parsed integer inequality over a declared feature."""

    feature: _FeatureName
    comparison: _Comparison
    cut: int

    def accepts(self, features: AbstractionFeatures, /) -> bool:
        """Return whether a feature vector satisfies this inequality."""
        value = features.values[_FEATURE_ORDER.index(self.feature)]

        return value <= self.cut if self.comparison == "<=" else value > self.cut


def _parse_condition(text: str, /) -> _Condition:
    """Return one condition parsed under the public condition grammar.

    Parameters
    ----------
    text : str
        Proposed ``feature operator integer`` inequality.

    Returns
    -------
    _Condition
        The parsed feature, comparison and integer cut.

    Raises
    ------
    ValueError
        If the condition names another feature or comparison, or has no integer cut.
    """
    try:
        named_feature, named_comparison, named_cut = text.split(" ")
    except ValueError as malformed:
        msg = f"a bucket condition is not a declared integer inequality: {text!r}"
        raise ValueError(msg) from malformed

    for declared in _FEATURE_ORDER:
        if named_feature == declared:
            feature = declared
            break
    else:
        msg = f"a bucket condition names no declared feature: {text!r}"
        raise ValueError(msg)

    if named_comparison == "<=":
        comparison: _Comparison = "<="
    elif named_comparison == ">":
        comparison = ">"
    else:
        msg = f"a bucket condition has no declared comparison: {text!r}"
        raise ValueError(msg)

    try:
        cut = int(named_cut)
    except ValueError as noninteger:
        msg = f"a bucket condition has no integer cut: {text!r}"
        raise ValueError(msg) from noninteger

    if str(cut) != named_cut:
        msg = f"a bucket condition has no canonical integer cut: {text!r}"
        raise ValueError(msg)

    return _Condition(feature, comparison, cut)


def _unit_interval(name: str, value: float) -> float:
    """Return a finite number in the closed unit interval.

    Parameters
    ----------
    name : str
        Field name used in refusal messages.
    value : float
        Number to validate.

    Returns
    -------
    float
        Validated number.

    Raises
    ------
    TypeError
        If the value is not numeric or is a boolean.
    ValueError
        If the value is nonfinite or outside the closed unit interval.
    """
    checked = ensure_finite(name, value)

    if not 0.0 <= checked <= 1.0:
        msg = f"{name} is outside the closed unit interval: {value!r}"
        raise ValueError(msg)

    return checked


class AbstractionFeatures(FrozenValue):
    """The four integer features that may define an abstraction bucket.

    Parameters
    ----------
    sentence_length : int
        Number of tokens in the sentence.
    derivation_depth : int
        Depth of the derivational decision.
    ambiguity : int
        Number of competing analyses at the decision.
    span_length : int
        Number of tokens in the decision's span.

    Raises
    ------
    TypeError
        If a feature is not an integer or is a boolean.
    ValueError
        If a length is not positive, a count is negative, or the span is longer
        than the sentence.
    """

    _noun = "abstraction features"

    __slots__ = (
        "_ambiguity",
        "_derivation_depth",
        "_sentence_length",
        "_span_length",
    )

    def __init__(
        self,
        *,
        sentence_length: int,
        derivation_depth: int,
        ambiguity: int,
        span_length: int,
    ) -> None:
        sentence = ensure_positive("a sentence length", sentence_length)
        depth = ensure_non_negative("a derivation depth", derivation_depth)
        alternatives = ensure_non_negative("an ambiguity count", ambiguity)
        span = ensure_positive("a span length", span_length)

        if span > sentence:
            msg = f"a span length {span!r} exceeds its sentence length {sentence!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_sentence_length", sentence)
        object.__setattr__(self, "_derivation_depth", depth)
        object.__setattr__(self, "_ambiguity", alternatives)
        object.__setattr__(self, "_span_length", span)

    @property
    def sentence_length(self) -> int:
        """Return the number of tokens in the sentence."""
        return self._sentence_length

    @property
    def derivation_depth(self) -> int:
        """Return the depth of the derivational decision."""
        return self._derivation_depth

    @property
    def ambiguity(self) -> int:
        """Return the number of competing analyses at the decision."""
        return self._ambiguity

    @property
    def span_length(self) -> int:
        """Return the number of tokens in the decision's span."""
        return self._span_length

    @property
    def values(self) -> tuple[int, int, int, int]:
        """Return the four features in their declared split order."""
        return self._identity()

    def _identity(self) -> tuple[int, int, int, int]:
        """Return the four features in their declared split order."""
        return (
            self._sentence_length,
            self._derivation_depth,
            self._ambiguity,
            self._span_length,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value has the same four features."""
        if type(other) is not AbstractionFeatures:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the four features."""
        return hash((AbstractionFeatures, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds these features."""
        return (
            "AbstractionFeatures("
            f"sentence_length={self._sentence_length!r}, "
            f"derivation_depth={self._derivation_depth!r}, "
            f"ambiguity={self._ambiguity!r}, "
            f"span_length={self._span_length!r})"
        )


class AbstractionCase(FrozenValue):
    """One distributed-alignment match beside its interpretable features.

    Parameters
    ----------
    features : AbstractionFeatures
        Features under which the match may be partitioned.
    matched : bool
        Whether the distributed alignment made the counterfactual prediction.

    Raises
    ------
    TypeError
        If either value has another type.
    """

    _noun = "abstraction case"

    __slots__ = ("_features", "_matched")

    def __init__(self, features: AbstractionFeatures, *, matched: bool) -> None:
        if type(features) is not AbstractionFeatures:
            msg = f"a case has AbstractionFeatures, not {features!r}"
            raise TypeError(msg)

        if not isinstance(matched, bool):
            msg = f"a case's match is boolean, not {matched!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_features", features)
        object.__setattr__(self, "_matched", matched)

    @property
    def features(self) -> AbstractionFeatures:
        """Return the features under which the match may be partitioned."""
        return self._features

    @property
    def matched(self) -> bool:
        """Return whether the counterfactual prediction matched."""
        return self._matched

    def _identity(self) -> tuple[AbstractionFeatures, bool]:
        """Return the features and match outcome."""
        return (self._features, self._matched)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another case has the same features and outcome."""
        if type(other) is not AbstractionCase:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the features and outcome."""
        return hash((AbstractionCase, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this case."""
        return f"AbstractionCase({self._features!r}, matched={self._matched!r})"


def _case_key(case: AbstractionCase) -> tuple[int, int, int, int, bool]:
    """Return a case's canonical order key.

    Parameters
    ----------
    case : AbstractionCase
        Case to order.

    Returns
    -------
    tuple[int, int, int, int, bool]
        Four features in split order, followed by the outcome.
    """
    return (*case.features.values, case.matched)


def _checked_conditions(
    conditions: Sequence[str], cases: tuple[AbstractionCase, ...], /
) -> tuple[str, ...]:
    """Return declared conditions after checking every case against them.

    Parameters
    ----------
    conditions : Sequence[str]
        Proposed integer inequalities over declared features.
    cases : tuple[AbstractionCase, ...]
        Cases the conditions must define.

    Returns
    -------
    tuple[str, ...]
        The checked conditions in their declared order.

    Raises
    ------
    TypeError
        If a condition is not a string.
    ValueError
        If a condition is malformed or does not hold of every case.
    """
    held = tuple(
        ensure_name("a bucket condition", condition) for condition in conditions
    )
    parsed = tuple(_parse_condition(condition) for condition in held)

    for position, case in enumerate(cases):
        for text, condition in zip(held, parsed, strict=True):
            if not condition.accepts(case.features):
                msg = f"bucket case {position} does not satisfy condition {text!r}"
                raise ValueError(msg)

    return held


class AbstractionBucket(FrozenValue):
    """One terminal region of abstraction cases.

    Parameters
    ----------
    conditions : Sequence[str]
        Integer inequalities defining the region from the root.
    cases : Sequence[AbstractionCase]
        Cases in the region.
    well_cutoff : float
        Inclusive accuracy at which the region is labeled well abstracted.

    Raises
    ------
    TypeError
        If a condition or case has another type, or the cutoff is not numeric.
    ValueError
        If no case is present, a condition is malformed or does not define every
        case, or the cutoff is invalid.
    """

    _noun = "abstraction bucket"

    __slots__ = ("_accuracy", "_cases", "_conditions", "_label")

    def __init__(
        self,
        conditions: Sequence[str],
        cases: Sequence[AbstractionCase],
        *,
        well_cutoff: float,
    ) -> None:
        held_cases = tuple(cases)

        for case in held_cases:
            if type(case) is not AbstractionCase:
                msg = f"a bucket contains an AbstractionCase, not {case!r}"
                raise TypeError(msg)

        if not held_cases:
            msg = "an abstraction bucket contains at least one case"
            raise ValueError(msg)

        held_conditions = _checked_conditions(conditions, held_cases)
        cutoff = _unit_interval("a well cutoff", well_cutoff)
        ordered = tuple(sorted(held_cases, key=_case_key))
        accuracy = sum(case.matched for case in ordered) / len(ordered)
        label: _BucketLabel = "well" if accuracy >= cutoff else "under"
        object.__setattr__(self, "_conditions", held_conditions)
        object.__setattr__(self, "_cases", ordered)
        object.__setattr__(self, "_accuracy", accuracy)
        object.__setattr__(self, "_label", label)

    @property
    def conditions(self) -> tuple[str, ...]:
        """Return the inequalities defining the region from the root."""
        return self._conditions

    @property
    def cases(self) -> tuple[AbstractionCase, ...]:
        """Return the cases in canonical feature order."""
        return self._cases

    @property
    def accuracy(self) -> float:
        """Return the fraction of cases whose prediction matched."""
        return self._accuracy

    @property
    def size(self) -> int:
        """Return the number of cases in the region."""
        return len(self._cases)

    @property
    def label(self) -> _BucketLabel:
        """Return whether the region meets the declared well cutoff."""
        return self._label

    def _identity(
        self,
    ) -> tuple[tuple[str, ...], tuple[AbstractionCase, ...], float, _BucketLabel]:
        """Return the conditions, cases, accuracy and label."""
        return (self._conditions, self._cases, self._accuracy, self._label)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another bucket defines the same terminal region."""
        if type(other) is not AbstractionBucket:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the terminal region."""
        return hash((AbstractionBucket, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the conditions, accuracy, size and label."""
        return (
            f"AbstractionBucket({self._conditions!r}, accuracy={self._accuracy!r}, "
            f"size={self.size!r}, label={self._label!r})"
        )


class BucketingReport(FrozenValue):
    """A deterministic abstraction partition and its declared policy.

    Parameters
    ----------
    buckets : Sequence[AbstractionBucket]
        Terminal regions in left-to-right tree order.
    well_cutoff : float
        Inclusive accuracy at which a region is labeled well abstracted.
    min_bucket_size : int
        Fewest cases permitted on either side of a split.
    min_accuracy_gap : float
        Accuracy difference a split must exceed.

    Raises
    ------
    TypeError
        If a bucket or policy value has another type.
    ValueError
        If no bucket is present or a policy value is invalid.
    """

    _noun = "bucketing report"

    __slots__ = (
        "_buckets",
        "_min_accuracy_gap",
        "_min_bucket_size",
        "_well_cutoff",
    )

    def __init__(
        self,
        buckets: Sequence[AbstractionBucket],
        *,
        well_cutoff: float,
        min_bucket_size: int,
        min_accuracy_gap: float,
    ) -> None:
        held = tuple(buckets)
        cutoff = _unit_interval("a well cutoff", well_cutoff)

        for bucket in held:
            if type(bucket) is not AbstractionBucket:
                msg = f"a report contains an AbstractionBucket, not {bucket!r}"
                raise TypeError(msg)

            expected: _BucketLabel = "well" if bucket.accuracy >= cutoff else "under"

            if bucket.label != expected:
                msg = (
                    f"a bucket labeled {bucket.label!r} does not agree with the "
                    f"report's well cutoff {cutoff!r}"
                )
                raise ValueError(msg)

            _checked_conditions(bucket.conditions, bucket.cases)

        if not held:
            msg = "a bucketing report contains at least one bucket"
            raise ValueError(msg)

        object.__setattr__(self, "_buckets", held)
        object.__setattr__(self, "_well_cutoff", cutoff)
        object.__setattr__(
            self,
            "_min_bucket_size",
            ensure_positive("a minimum bucket size", min_bucket_size),
        )
        object.__setattr__(
            self,
            "_min_accuracy_gap",
            _unit_interval("a minimum accuracy gap", min_accuracy_gap),
        )

    @property
    def buckets(self) -> tuple[AbstractionBucket, ...]:
        """Return terminal regions in left-to-right tree order."""
        return self._buckets

    @property
    def well_cutoff(self) -> float:
        """Return the inclusive accuracy defining a well-abstracted region."""
        return self._well_cutoff

    @property
    def min_bucket_size(self) -> int:
        """Return the fewest cases permitted on either side of a split."""
        return self._min_bucket_size

    @property
    def min_accuracy_gap(self) -> float:
        """Return the accuracy difference every split exceeds."""
        return self._min_accuracy_gap

    @property
    def size(self) -> int:
        """Return the number of cases covered by all terminal regions."""
        return sum(bucket.size for bucket in self._buckets)

    def _identity(
        self,
    ) -> tuple[tuple[AbstractionBucket, ...], float, int, float]:
        """Return the partition and its declared policy."""
        return (
            self._buckets,
            self._well_cutoff,
            self._min_bucket_size,
            self._min_accuracy_gap,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report has the same partition and policy."""
        if type(other) is not BucketingReport:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the partition and its policy."""
        return hash((BucketingReport, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the bucket count, case count and declared policy."""
        return (
            f"BucketingReport(buckets={len(self._buckets)!r}, size={self.size!r}, "
            f"well_cutoff={self._well_cutoff!r}, "
            f"min_bucket_size={self._min_bucket_size!r}, "
            f"min_accuracy_gap={self._min_accuracy_gap!r})"
        )


class _Split(NamedTuple):
    """One admissible split and the cases on either side."""

    feature: _FeatureName
    cut: int
    left: tuple[AbstractionCase, ...]
    right: tuple[AbstractionCase, ...]
    gain: Fraction


def _accuracy(cases: tuple[AbstractionCase, ...]) -> Fraction:
    """Return the exact match fraction in a nonempty collection.

    Parameters
    ----------
    cases : tuple[AbstractionCase, ...]
        Cases whose outcomes are reduced.

    Returns
    -------
    Fraction
        Exact match fraction.
    """
    return Fraction(sum(case.matched for case in cases), len(cases))


def _gini(cases: tuple[AbstractionCase, ...]) -> Fraction:
    """Return exact Bernoulli Gini impurity for nonempty cases.

    Parameters
    ----------
    cases : tuple[AbstractionCase, ...]
        Cases whose outcomes define the impurity.

    Returns
    -------
    Fraction
        Twice the product of the two outcome probabilities.
    """
    accuracy = _accuracy(cases)

    return 2 * accuracy * (1 - accuracy)


def _best_split(
    cases: tuple[AbstractionCase, ...],
    *,
    min_bucket_size: int,
    min_accuracy_gap: float,
) -> _Split | None:
    """Return the best admissible split under the declared tie order.

    Parameters
    ----------
    cases : tuple[AbstractionCase, ...]
        Cases in the region being split.
    min_bucket_size : int
        Fewest cases permitted on either side.
    min_accuracy_gap : float
        Accuracy difference a split must exceed.

    Returns
    -------
    _Split | None
        Highest-gain split, or None when no candidate is admissible.
    """
    parent = _gini(cases)
    best: _Split | None = None

    for position, feature in enumerate(_FEATURE_ORDER):
        values = sorted({case.features.values[position] for case in cases})

        for cut in values[:-1]:
            left = tuple(
                case for case in cases if case.features.values[position] <= cut
            )
            right = tuple(
                case for case in cases if case.features.values[position] > cut
            )

            if len(left) < min_bucket_size or len(right) < min_bucket_size:
                continue

            gap = abs(_accuracy(left) - _accuracy(right))

            if gap <= Fraction(min_accuracy_gap):
                continue

            count = len(cases)
            children = Fraction(len(left), count) * _gini(left) + Fraction(
                len(right), count
            ) * _gini(right)
            gain = parent - children

            if gain <= 0:
                continue

            candidate = _Split(feature, cut, left, right, gain)

            if best is None or candidate.gain > best.gain:
                best = candidate

    return best


def _buckets(
    cases: tuple[AbstractionCase, ...],
    conditions: tuple[str, ...],
    *,
    well_cutoff: float,
    min_bucket_size: int,
    min_accuracy_gap: float,
) -> tuple[AbstractionBucket, ...]:
    """Return terminal regions below one recursive partition node.

    Parameters
    ----------
    cases : tuple[AbstractionCase, ...]
        Cases at the node.
    conditions : tuple[str, ...]
        Inequalities leading to the node.
    well_cutoff : float
        Inclusive accuracy defining a well-abstracted region.
    min_bucket_size : int
        Fewest cases permitted on either side.
    min_accuracy_gap : float
        Accuracy difference a split must exceed.

    Returns
    -------
    tuple[AbstractionBucket, ...]
        Terminal regions in left-to-right order.
    """
    split = _best_split(
        cases,
        min_bucket_size=min_bucket_size,
        min_accuracy_gap=min_accuracy_gap,
    )

    if split is None:
        return (AbstractionBucket(conditions, cases, well_cutoff=well_cutoff),)

    left = _buckets(
        split.left,
        (*conditions, f"{split.feature} <= {split.cut}"),
        well_cutoff=well_cutoff,
        min_bucket_size=min_bucket_size,
        min_accuracy_gap=min_accuracy_gap,
    )
    right = _buckets(
        split.right,
        (*conditions, f"{split.feature} > {split.cut}"),
        well_cutoff=well_cutoff,
        min_bucket_size=min_bucket_size,
        min_accuracy_gap=min_accuracy_gap,
    )

    return left + right


def bucket_abstraction(
    cases: Sequence[AbstractionCase],
    *,
    well_cutoff: float,
    min_bucket_size: int,
    min_accuracy_gap: float,
) -> BucketingReport:
    """Partition abstraction outcomes by deterministic Gini splits.

    Parameters
    ----------
    cases : Sequence[AbstractionCase]
        Match outcomes and the four integer features that may partition them.
    well_cutoff : float
        Inclusive accuracy defining a well-abstracted terminal region.
    min_bucket_size : int
        Fewest cases permitted on either side of a split.
    min_accuracy_gap : float
        Accuracy difference a split must exceed.

    Returns
    -------
    BucketingReport
        Terminal regions and the declared split policy.

    Raises
    ------
    TypeError
        If a case or policy value has another type.
    ValueError
        If no case is given or a policy value is invalid.
    """
    cutoff = _unit_interval("a well cutoff", well_cutoff)
    size = ensure_positive("a minimum bucket size", min_bucket_size)
    gap = _unit_interval("a minimum accuracy gap", min_accuracy_gap)
    held = tuple(cases)

    for case in held:
        if type(case) is not AbstractionCase:
            msg = f"bucketing consumes an AbstractionCase, not {case!r}"
            raise TypeError(msg)

    if not held:
        msg = "bucketing requires at least one abstraction case"
        raise ValueError(msg)

    leaves = _buckets(
        held,
        (),
        well_cutoff=cutoff,
        min_bucket_size=size,
        min_accuracy_gap=gap,
    )

    return BucketingReport(
        leaves,
        well_cutoff=cutoff,
        min_bucket_size=size,
        min_accuracy_gap=gap,
    )
