"""Aligned tree-distance and shape agreement metrics."""

from math import fsum, sqrt
from statistics import fmean
from typing import TYPE_CHECKING, final, override

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.eval._alignment import Alignment
from lofi.terms import Operator, Term, TermStore, TreeEditDistance, UnknownOperatorError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = ["TypeTreeAgreement", "normalized_tree_distance", "type_tree_agreement"]

_MINIMUM_RANK_PAIRS = 2


type _DepthHistogram = tuple[tuple[int, int], ...]
"""Counts keyed by absolute depth difference."""

type _AgreementFields = tuple[
    int,
    float | None,
    float | None,
    float | None,
    _DepthHistogram,
    float | None,
]
"""Fields needed to reconstruct a tree-agreement report."""

type _AgreementReduction = tuple[
    Callable[[_AgreementFields], TypeTreeAgreement],
    tuple[_AgreementFields],
]
"""Tree-agreement pickle reduction."""


def _rate(field: str, value: float | None) -> float | None:
    """Validate an optional proportion."""
    if value is None:
        return None

    number = ensure_finite(field, value)

    if not 0.0 <= number <= 1.0:
        msg = f"{field} is not a proportion: {value!r}"
        raise ValueError(msg)

    return number


def _correlation(value: float | None) -> float | None:
    """Validate an optional correlation."""
    if value is None:
        return None

    number = ensure_finite("a depth correlation", value)

    if not -1.0 <= number <= 1.0:
        msg = f"a depth correlation is outside [-1, 1]: {value!r}"
        raise ValueError(msg)

    return number


@final
class TypeTreeAgreement(FrozenValue):
    """A normalized tree-distance and shape-agreement report."""

    _noun = "tree-agreement value"

    __slots__ = (
        "_arity_agreement",
        "_depth_difference_histogram",
        "_depth_spearman",
        "_exact_match_rate",
        "_mean_normalized_distance",
        "_vocabulary_size",
    )

    def __init__(self, fields: _AgreementFields) -> None:
        (
            vocabulary_size,
            mean_normalized_distance,
            exact_match_rate,
            arity_agreement,
            depth_difference_histogram,
            depth_spearman,
        ) = fields
        size = ensure_non_negative("a tree-agreement vocabulary size", vocabulary_size)
        distance = _rate("a mean normalized tree distance", mean_normalized_distance)
        exact = _rate("a tree exact-match rate", exact_match_rate)
        arity = _rate("a root-arity agreement rate", arity_agreement)
        rho = _correlation(depth_spearman)
        histogram = tuple(
            (
                ensure_non_negative("an absolute depth difference", difference),
                ensure_non_negative("a depth-difference count", count),
            )
            for difference, count in depth_difference_histogram
        )

        if any(count == 0 for _, count in histogram):
            msg = "a depth-difference histogram contains an empty bin"
            raise ValueError(msg)

        if tuple(sorted(histogram)) != histogram:
            msg = "a depth-difference histogram is not in ascending order"
            raise ValueError(msg)

        if len({difference for difference, _ in histogram}) != len(histogram):
            msg = "a depth-difference histogram repeats a bin"
            raise ValueError(msg)

        if sum(count for _, count in histogram) != size:
            msg = (
                f"a depth-difference histogram counts "
                f"{sum(count for _, count in histogram)} entries for a vocabulary "
                f"of size {size}"
            )
            raise ValueError(msg)

        rates = (distance, exact, arity)

        if size == 0 and (any(value is not None for value in rates) or rho is not None):
            msg = "an empty tree-agreement report has a defined rate"
            raise ValueError(msg)

        if size > 0 and any(value is None for value in rates):
            msg = "a nonempty tree-agreement report has a missing rate"
            raise ValueError(msg)

        object.__setattr__(self, "_vocabulary_size", size)
        object.__setattr__(self, "_mean_normalized_distance", distance)
        object.__setattr__(self, "_exact_match_rate", exact)
        object.__setattr__(self, "_arity_agreement", arity)
        object.__setattr__(self, "_depth_difference_histogram", histogram)
        object.__setattr__(self, "_depth_spearman", rho)

    @property
    def vocabulary_size(self) -> int:
        """Return the number of paired vocabulary entries."""
        return self._vocabulary_size

    @property
    def mean_normalized_distance(self) -> float | None:
        """Return the mean edit distance divided by paired node count."""
        return self._mean_normalized_distance

    @property
    def exact_match_rate(self) -> float | None:
        """Return the fraction of pairs with zero edit distance."""
        return self._exact_match_rate

    @property
    def arity_agreement(self) -> float | None:
        """Return the fraction of pairs whose root arities match."""
        return self._arity_agreement

    @property
    def depth_difference_histogram(self) -> _DepthHistogram:
        """Return counts by absolute depth difference."""
        return self._depth_difference_histogram

    @property
    def depth_spearman(self) -> float | None:
        """Return Spearman correlation between paired type depths."""
        return self._depth_spearman

    def _fields(self) -> _AgreementFields:
        """Return fields in reconstruction order."""
        return (
            self._vocabulary_size,
            self._mean_normalized_distance,
            self._exact_match_rate,
            self._arity_agreement,
            self._depth_difference_histogram,
            self._depth_spearman,
        )

    @override
    def __copy__(self) -> TypeTreeAgreement:
        """Reconstruct the tree-agreement report."""
        return _restore_agreement(self._fields())

    @override
    def __deepcopy__(self, memo: dict[int, TypeTreeAgreement]) -> TypeTreeAgreement:
        """Reconstruct the tree-agreement report."""
        del memo

        return _restore_agreement(self._fields())

    @override
    def __reduce__(self) -> _AgreementReduction:
        """Return the tree-agreement reconstruction call."""
        return _restore_agreement, (self._fields(),)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report has the same values."""
        if type(other) is not TypeTreeAgreement:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over every report value."""
        return hash((TypeTreeAgreement, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return (
            f"TypeTreeAgreement(({self._vocabulary_size!r}, "
            f"{self._mean_normalized_distance!r}, "
            f"{self._exact_match_rate!r}, {self._arity_agreement!r}, "
            f"{self._depth_difference_histogram!r}, {self._depth_spearman!r}))"
        )


def _restore_agreement(fields: _AgreementFields) -> TypeTreeAgreement:
    """Reconstruct a tree-agreement report from frozen fields."""
    return TypeTreeAgreement(fields)


def normalized_tree_distance(
    distance: TreeEditDistance,
    left: Term,
    right: Term,
    /,
) -> float:
    """Return the edit distance between two trees divided by their node count."""
    return distance(left, right) / (left.size + right.size)


def _depth(term: Term) -> int:
    """Return a term's maximum root-to-leaf edge count."""
    return max((_depth(child) + 1 for child in term.args), default=0)


def _average_ranks(values: Sequence[int]) -> tuple[float, ...]:
    """Return average ranks with equal values tied."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0

    while start < len(order):
        stop = start + 1

        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1

        shared = (start + stop + 1) / 2.0

        for position in range(start, stop):
            ranks[order[position]] = shared

        start = stop

    return tuple(ranks)


def _spearman(left: Sequence[int], right: Sequence[int]) -> float | None:
    """Return Spearman correlation or None where ranks cannot vary."""
    if len(left) < _MINIMUM_RANK_PAIRS or len(set(left)) == 1 or len(set(right)) == 1:
        return None

    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = fmean(left_ranks)
    right_mean = fmean(right_ranks)
    left_spread = fsum((value - left_mean) ** 2 for value in left_ranks)
    right_spread = fsum((value - right_mean) ** 2 for value in right_ranks)
    covariance = fsum(
        (value - left_mean) * (other - right_mean)
        for value, other in zip(left_ranks, right_ranks, strict=True)
    )
    correlation = covariance / sqrt(left_spread * right_spread)

    return min(1.0, max(-1.0, correlation))


def _payload(
    source: TermStore,
    target: TermStore,
    term: Term,
    operator: Operator,
) -> int | None:
    """Translate a payload into a comparison store by name.

    Parameters
    ----------
    source : TermStore
        The store the term is interned in.
    target : TermStore
        The comparison store.
    term : Term
        The node whose payload is translated.
    operator : Operator
        The operator the node is rebuilt with.
    """
    payload = term.payload
    kind = term.op.payload_sort

    if payload is None or kind not in ("free", "constant"):
        return payload

    if kind == "constant":
        name = source.signature.vocabulary(term.op.sort).name(payload)

        return target.signature.vocabulary(operator.sort).id(name)

    return target.atom(operator.sort, source.atom_name(payload))


def _items(
    vocabulary: Mapping[str, Term],
    store: TermStore,
    *,
    side: str,
) -> tuple[tuple[str, Term], ...]:
    """Validate and freeze one side of a type vocabulary."""
    held: list[tuple[str, Term]] = []

    for position, (label, term) in enumerate(vocabulary.items()):
        name = ensure_name(f"{side} vocabulary label {position}", label)

        if not isinstance(term, Term):
            msg = f"{side} vocabulary entry {name!r} is not an interned term"
            raise TypeError(msg)

        if not store.contains(term):
            msg = f"{side} vocabulary entry {name!r} is outside its supplied store"
            raise ValueError(msg)

        held.append((name, term))

    return tuple(held)


def _primitive_feature(
    operator: Operator,
    core: Operator,
    predicted_store: TermStore,
) -> str | None:
    """Resolve a featured primitive's declared feature."""
    if operator == core:
        return None

    for feature in predicted_store.signature.features.get(core.name, ()):
        if predicted_store.signature.featured(core.name, feature) == operator:
            return feature

    msg = f"predicted primitive {operator.name!r} has an undeclared decoration"
    raise ValueError(msg)


def _primitive_target(
    operator: Operator,
    stores: tuple[TermStore, TermStore],
    alignment: Alignment,
) -> Operator:
    """Resolve one primitive through the supplied alignment."""
    predicted_store, gold_store = stores
    core = predicted_store.signature.erase(operator)
    image = alignment.apply(core.name)

    if image is None:
        msg = f"primitive {core.name!r} has no image in the supplied alignment"
        raise ValueError(msg)

    try:
        declared_target = gold_store.signature.by_name(image)
    except UnknownOperatorError as error:
        msg = f"aligned primitive {image!r} is absent from the gold signature"
        raise ValueError(msg) from error

    target = gold_store.signature.erase(declared_target)

    if declared_target != target:
        msg = f"aligned primitive {image!r} is not an erasure core"
        raise ValueError(msg)

    if target.arity != 0:
        msg = f"aligned primitive {image!r} has arity {target.arity}, not zero"
        raise ValueError(msg)

    if target.sort != core.sort:
        msg = (
            f"aligned primitive {image!r} has result sort {target.sort!r}, "
            f"not {core.sort!r}"
        )
        raise ValueError(msg)

    if target.payload_sort != core.payload_sort:
        msg = (
            f"aligned primitive {image!r} has payload sort "
            f"{target.payload_sort!r}, not {core.payload_sort!r}"
        )
        raise ValueError(msg)

    feature = _primitive_feature(operator, core, predicted_store)

    if feature is None:
        return target

    try:
        return gold_store.signature.featured(target.name, feature)
    except UnknownOperatorError as error:
        msg = (
            f"aligned primitive {image!r} has no feature {feature!r} required by "
            f"predicted primitive {operator.name!r}"
        )
        raise ValueError(msg) from error


def _constructor_target(
    operator: Operator,
    gold_store: TermStore,
) -> Operator:
    """Resolve one structurally compatible constructor declaration."""
    try:
        target = gold_store.signature.by_name(operator.name)
    except UnknownOperatorError as error:
        msg = (
            f"predicted constructor {operator.name!r} is absent from the gold signature"
        )
        raise ValueError(msg) from error

    if target != operator:
        msg = (
            f"predicted constructor {operator.name!r} is incompatible with "
            f"the gold declaration"
        )
        raise ValueError(msg)

    return target


def _translation(
    predicted: Sequence[Term],
    stores: tuple[TermStore, TermStore],
    alignment: Alignment,
) -> Mapping[Operator, Operator]:
    """Validate and build the predicted-to-gold operator translation."""
    _, gold_store = stores
    translated: dict[Operator, Operator] = {}
    visited: set[Term] = set()

    def visit(term: Term) -> None:
        if term in visited:
            return

        visited.add(term)
        operator = term.op
        target = (
            _primitive_target(operator, stores, alignment)
            if operator.arity == 0
            else _constructor_target(operator, gold_store)
        )

        translated[operator] = target

        for child in term.args:
            visit(child)

    for root in predicted:
        visit(root)

    return translated


def _clone(
    source: TermStore,
    target: TermStore,
    term: Term,
    operators: Mapping[Operator, Operator] | None,
    memo: dict[Term, Term],
) -> Term:
    """Rebuild one term in the comparison store."""
    held = memo.get(term)

    if held is not None:
        return held

    operator = term.op if operators is None else operators[term.op]
    arguments = tuple(
        _clone(source, target, child, operators, memo) for child in term.args
    )
    rebuilt = target.op(
        operator, *arguments, payload=_payload(source, target, term, operator)
    )
    memo[term] = rebuilt

    return rebuilt


def type_tree_agreement(
    predicted: Mapping[str, Term],
    gold: Mapping[str, Term],
    stores: tuple[TermStore, TermStore],
    *,
    alignment: Alignment,
) -> TypeTreeAgreement:
    """Measure aligned tree distance and shape agreement across a vocabulary.

    Raises
    ------
    TypeError
        If ``alignment`` is not an ``Alignment``.
    ValueError
        If the vocabularies have different keys or contain invalid terms.
    """
    if not isinstance(alignment, Alignment):
        msg = f"a tree-agreement alignment is not frozen Alignment: {alignment!r}"
        raise TypeError(msg)

    predicted_store, gold_store = stores
    predicted_items = _items(predicted, predicted_store, side="predicted")
    gold_items = _items(gold, gold_store, side="gold")
    predicted_by_label = dict(predicted_items)
    gold_by_label = dict(gold_items)

    if predicted_by_label.keys() != gold_by_label.keys():
        only_predicted = sorted(predicted_by_label.keys() - gold_by_label.keys())
        only_gold = sorted(gold_by_label.keys() - predicted_by_label.keys())
        msg = (
            f"type vocabularies have mismatched keys; predicted only "
            f"{only_predicted!r}, gold only {only_gold!r}"
        )
        raise ValueError(msg)

    size = len(predicted_items)

    if size == 0:
        return TypeTreeAgreement((0, None, None, None, (), None))

    predicted_terms = tuple(term for _, term in predicted_items)
    translations = _translation(predicted_terms, stores, alignment)
    comparison = TermStore(gold_store.signature, max_nodes=gold_store.max_nodes)
    predicted_memo: dict[Term, Term] = {}
    gold_memo: dict[Term, Term] = {}
    distance: TreeEditDistance = TreeEditDistance()
    normalized: list[float] = []
    exact = 0
    arity_matches = 0
    depth_counts: dict[int, int] = {}
    predicted_depths: list[int] = []
    gold_depths: list[int] = []

    for label, predicted_term in predicted_items:
        gold_term = gold_by_label[label]
        aligned = _clone(
            predicted_store,
            comparison,
            predicted_term,
            translations,
            predicted_memo,
        )
        reference = _clone(
            gold_store,
            comparison,
            gold_term,
            None,
            gold_memo,
        )
        edits = normalized_tree_distance(distance, aligned, reference)
        normalized.append(edits)
        exact += edits == 0.0
        arity_matches += predicted_term.op.arity == gold_term.op.arity
        predicted_depth = _depth(predicted_term)
        gold_depth = _depth(gold_term)
        difference = abs(predicted_depth - gold_depth)
        depth_counts[difference] = depth_counts.get(difference, 0) + 1
        predicted_depths.append(predicted_depth)
        gold_depths.append(gold_depth)

    return TypeTreeAgreement(
        (
            size,
            fmean(normalized),
            exact / size,
            arity_matches / size,
            tuple(sorted(depth_counts.items())),
            _spearman(predicted_depths, gold_depths),
        )
    )
