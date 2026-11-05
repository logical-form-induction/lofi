"""Logical-vocabulary recovery up to relabeling with argument permutation."""

from itertools import permutations as orderings
from itertools import product
from math import ceil, factorial
from statistics import fmean
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.eval._alignment import Contingency
from lofi.eval._logical_form import normalized_form
from lofi.eval._values import MetricResult, Resampling
from lofi.lf import Graded, Proved
from lofi.lf.provers import ProofSite
from lofi.stats import Interval
from lofi.terms import Normalization, Term

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from random import Random

    from lofi.eval._alignment import Alignment, Split
    from lofi.eval._ports import AlignmentPolicy
    from lofi.lf.provers import ProofBudget, ProofService
    from lofi.terms import Calculus, TermStore

__all__ = [
    "FIBER_MEASURES",
    "FiberMeasure",
    "FiberRecovery",
    "LogicalVocabulary",
    "Predicate",
    "fiber_floor",
    "fiber_match",
    "fiber_recovery",
    "fiber_size",
    "permuted_body",
]

type FiberMeasure = Literal["up-to-fiber", "fixed-alphabet"]
"""A headline number a vocabulary recovery reports."""

FIBER_MEASURES: Final[tuple[FiberMeasure, ...]] = ("up-to-fiber", "fixed-alphabet")
"""Recovery through the symmetry group first, then under the gold alphabet."""

_UNIT: Final = "predicate"


_APPLICATION_ARITY: Final = 2
_PROOF_SITE: Final = ProofSite("evaluation.fiber")


def _check_normalization(
    calculus: Calculus | None,
    normalization: Normalization | None,
) -> None:
    """Require the optional calculus and normalization treatment together."""
    if (calculus is None) != (normalization is None):
        msg = "a calculus and normalization treatment are supplied together"
        raise ValueError(msg)


class Predicate(FrozenValue):
    """A named relation and the formula that gives its meaning.

    The body is a closed formula in which the relation's argument positions
    appear as the first ``arity`` slot names its vocabulary declares.

    Parameters
    ----------
    symbol : str
        The relation's name.
    arity : int
        How many argument positions the relation takes.
    body : Term
        The formula the relation means.

    Raises
    ------
    TypeError
        If the body is not a term or the arity is not an integer.
    ValueError
        If the symbol is empty or the arity is negative.
    """

    _noun = "vocabulary value"

    __slots__ = ("_arity", "_body", "_symbol")

    def __init__(self, symbol: str, arity: int, body: Term) -> None:
        if not isinstance(body, Term):
            msg = f"a predicate's body is not a term: {body!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_symbol", ensure_name("a predicate's symbol", symbol))
        object.__setattr__(
            self, "_arity", ensure_non_negative("a predicate's arity", arity)
        )
        object.__setattr__(self, "_body", body)

    @property
    def symbol(self) -> str:
        """Return the relation's name."""
        return self._symbol

    @property
    def arity(self) -> int:
        """Return how many argument positions the relation takes."""
        return self._arity

    @property
    def body(self) -> Term:
        """Return the formula the relation means."""
        return self._body

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value names the same relation with the same body."""
        if type(other) is not Predicate:
            return NotImplemented

        return (
            self._symbol == other._symbol
            and self._arity == other._arity
            and self._body is other._body
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the symbol, the arity and the body."""
        return hash((Predicate, self._symbol, self._arity, self._body))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this predicate."""
        return f"Predicate({self._symbol!r}, {self._arity!r}, {self._body!r})"


class LogicalVocabulary(FrozenValue):
    """A set of relations over one shared alphabet of argument slots.

    Parameters
    ----------
    predicates : Iterable[Predicate]
        The relations the vocabulary declares.
    slots : Sequence[str]
        The constant names standing for argument positions, in order.

    Raises
    ------
    TypeError
        If a predicate is not one.
    ValueError
        If a symbol or a slot repeats, a slot name is empty, or a predicate
        takes more arguments than there are slots.
    """

    _noun = "vocabulary value"

    __slots__ = ("_predicates", "_slots", "_symbols")

    def __init__(
        self, predicates: Iterable[Predicate], *, slots: Sequence[str]
    ) -> None:
        held = tuple(
            sorted(
                (_predicate(entry) for entry in predicates),
                key=lambda entry: entry.symbol,
            )
        )
        names = tuple(ensure_name("a slot", slot) for slot in slots)

        if len(set(names)) != len(names):
            msg = f"a vocabulary's slots repeat a name: {names!r}"
            raise ValueError(msg)

        symbols = tuple(entry.symbol for entry in held)

        if len(set(symbols)) != len(symbols):
            msg = f"a vocabulary declares a symbol twice: {symbols!r}"
            raise ValueError(msg)

        for entry in held:
            if entry.arity > len(names):
                msg = (
                    f"{entry.symbol!r} takes {entry.arity} arguments and the "
                    f"vocabulary declares {len(names)} slots"
                )
                raise ValueError(msg)

        object.__setattr__(self, "_predicates", held)
        object.__setattr__(self, "_slots", names)
        object.__setattr__(self, "_symbols", symbols)

    @property
    def predicates(self) -> tuple[Predicate, ...]:
        """Return the relations, in ascending symbol order."""
        return self._predicates

    @property
    def slots(self) -> tuple[str, ...]:
        """Return the constant names standing for argument positions."""
        return self._slots

    @property
    def symbols(self) -> tuple[str, ...]:
        """Return the relation names, in ascending order."""
        return self._symbols

    def predicate(self, symbol: str, /) -> Predicate:
        """Return the relation a symbol names.

        Parameters
        ----------
        symbol : str
            The relation's name.

        Returns
        -------
        Predicate
            The relation the vocabulary declares under that name.

        Raises
        ------
        KeyError
            If the vocabulary declares no such relation.
        """
        for entry in self._predicates:
            if entry.symbol == symbol:
                return entry

        raise KeyError(symbol)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value declares the same relations and slots."""
        if type(other) is not LogicalVocabulary:
            return NotImplemented

        return self._predicates == other._predicates and self._slots == other._slots

    @override
    def __hash__(self) -> int:
        """Return a hash over the relations and the slots."""
        return hash((LogicalVocabulary, self._predicates, self._slots))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this vocabulary."""
        return f"LogicalVocabulary({self._predicates!r}, slots={self._slots!r})"


def _predicate(value: Predicate) -> Predicate:
    """Return a validated vocabulary entry.

    Parameters
    ----------
    value : Predicate
        Candidate relation.

    Returns
    -------
    Predicate
        The relation.

    Raises
    ------
    TypeError
        If ``value`` is not a predicate.
    """
    if not isinstance(value, Predicate):
        msg = f"a vocabulary entry is not a predicate: {value!r}"
        raise TypeError(msg)

    return value


class FiberRecovery(NamedTuple):
    """Recovery of an induced logical vocabulary against a gold one.

    Attributes
    ----------
    up_to_fiber : MetricResult
        The share of gold relations identified under the fitted relabeling and
        the argument permutation chosen for each aligned relation.
    fixed_alphabet : MetricResult
        The share identified when each induced relation must keep its own name
        and its own argument order.
    alignment : Alignment
        The relabeling the assignment policy fitted.
    permutations : tuple[tuple[str, tuple[int, ...]], ...]
        The argument permutation chosen for each aligned induced relation,
        in ascending symbol order.
    floor : MetricResult | None
        The chance recovery at this signature size, when one was supplied.
    """

    up_to_fiber: MetricResult
    fixed_alphabet: MetricResult
    alignment: Alignment
    permutations: tuple[tuple[str, tuple[int, ...]], ...]
    floor: MetricResult | None = None

    def measure(self, measure: FiberMeasure, /) -> MetricResult:
        """Return one of the two headline results.

        Parameters
        ----------
        measure : FiberMeasure
            Which recovery to read.

        Returns
        -------
        MetricResult
            The named recovery.
        """
        return self.up_to_fiber if measure == "up-to-fiber" else self.fixed_alphabet

    def permutation(self, symbol: str, /) -> tuple[int, ...] | None:
        """Return the argument permutation chosen for an induced relation.

        Parameters
        ----------
        symbol : str
            An induced relation's name.

        Returns
        -------
        tuple[int, ...] | None
            The chosen permutation, or ``None`` if the relation was not
            aligned.
        """
        for name, order in self.permutations:
            if name == symbol:
                return order

        return None

    @property
    def beats_floor(self) -> bool | None:
        """Return whether recovery up to the fiber lies above its chance floor.

        A recovery at or below the floor is the flag this reports as ``False``.
        """
        if self.floor is None or self.up_to_fiber.value is None:
            return None

        if self.floor.value is None:
            return None

        bound = self.floor.value if self.floor.ci is None else self.floor.ci.high

        return self.up_to_fiber.value > bound


def _rebuilt(store: TermStore, original: Term, arguments: tuple[Term, ...]) -> Term:
    """Return the original node, or a fresh one when an argument changed.

    Parameters
    ----------
    store : TermStore
        The store the node is interned in.
    original : Term
        The node the arguments came from.
    arguments : tuple[Term, ...]
        The arguments to rebuild with.

    Returns
    -------
    Term
        The rebuilt node.
    """
    if arguments == original.args:
        return original

    return store.op(original.op, *arguments, payload=original.payload)


def _symbol_of(store: TermStore, term: Term) -> str | None:
    """Return the name a symbol node carries, or None for any other node.

    Parameters
    ----------
    store : TermStore
        The store the node is interned in.
    term : Term
        The node to read.

    Returns
    -------
    str | None
        The symbol's name, or ``None``.
    """
    if term.payload is None or not term.op.is_constant:
        return None

    return store.signature.vocabulary(term.op.sort).name(term.payload)


def _renamed(store: TermStore, term: Term, rename: Mapping[str, str]) -> Term:
    """Return a formula with its symbol constants renamed.

    Parameters
    ----------
    store : TermStore
        The store the formula is interned in.
    term : Term
        The formula to rewrite.
    rename : Mapping[str, str]
        The new name for each symbol that moves.

    Returns
    -------
    Term
        The rewritten formula.
    """
    name = _symbol_of(store, term)

    if name is not None:
        moved = rename.get(name)

        if moved is None:
            return term

        return store.const(term.op.sort, moved)

    if term.op.is_free_variable:
        moved = rename.get(store.payload_name(term))

        if moved is not None:
            return store.fvar(term.op.sort, moved)

    if not term.args:
        return term

    return _rebuilt(
        store, term, tuple(_renamed(store, argument, rename) for argument in term.args)
    )


def _order(permutation: Sequence[int], arity: int) -> tuple[int, ...]:
    """Return a validated argument permutation.

    Parameters
    ----------
    permutation : Sequence[int]
        Candidate ordering of argument positions.
    arity : int
        How many argument positions the relation takes.

    Returns
    -------
    tuple[int, ...]
        The validated ordering.

    Raises
    ------
    ValueError
        If the ordering is not a permutation of the argument positions.
    """
    held = tuple(permutation)

    if sorted(held) != list(range(arity)):
        msg = (
            f"{held!r} is not a permutation of the {arity} argument positions "
            "of the relation it was given for"
        )
        raise ValueError(msg)

    return held


def permuted_body(
    store: TermStore,
    vocabulary: LogicalVocabulary,
    symbol: str,
    permutation: Sequence[int],
    /,
) -> Term:
    """Return a relation's body with its argument positions permuted.

    The returned body reads its argument at position ``i`` off the slot the
    original read its argument at position ``permutation[i]`` off, so the
    identity permutation returns the body unchanged.

    Parameters
    ----------
    store : TermStore
        The store the body is interned in.
    vocabulary : LogicalVocabulary
        The vocabulary whose slots name the argument positions.
    symbol : str
        The relation to permute.
    permutation : Sequence[int]
        The new order of the argument positions.

    Returns
    -------
    Term
        The permuted body.

    Raises
    ------
    KeyError
        If the vocabulary declares no such relation.
    ValueError
        If the permutation does not order the relation's arguments.
    """
    predicate = vocabulary.predicate(symbol)
    held = _order(permutation, predicate.arity)
    slots = vocabulary.slots

    return _renamed(
        store,
        predicate.body,
        {slots[held[position]]: slots[position] for position in range(len(held))},
    )


def _same(
    store: TermStore,
    left: Term,
    right: Term,
    calculus: Calculus | None,
    normalization: Normalization | None,
    service: ProofService | None,
    budget: ProofBudget | None,
) -> bool:
    """Report whether two formulas are the same one in the ambient theory.

    Parameters
    ----------
    store : TermStore
        The store both formulas are interned in.
    left : Term
        One formula.
    right : Term
        The other formula.
    calculus : Calculus | None
        The abstraction and application operators to normalize under, or
        ``None`` to compare the formulas as they stand.
    normalization : Normalization | None
        The declared normalization treatment, or ``None`` with no calculus.
    service : ProofService | None
        The shared proof service consulted when the two do not intern alike.
    budget : ProofBudget | None
        The use-site proof budget, required when a service is supplied.

    Returns
    -------
    bool
        Whether the two are the same formula.
    """
    first = left
    second = right

    if calculus is not None and normalization is not None:
        first = normalized_form(store, left, calculus, normalization=normalization)
        second = normalized_form(store, right, calculus, normalization=normalization)

    if first is second:
        return True

    if service is None:
        return False

    if budget is None:
        msg = "logical equivalence through a proof service requires a proof budget"
        raise ValueError(msg)

    verdict = service.equivalent(first, second, budget=budget, site=_PROOF_SITE)
    return isinstance(verdict, Proved) or (
        isinstance(verdict, Graded) and verdict.degree == 1.0
    )


def _matching_order(
    store: TermStore,
    predicted: LogicalVocabulary,
    gold: LogicalVocabulary,
    pair: tuple[str, str],
    calculus: Calculus | None,
    normalization: Normalization | None,
    service: ProofService | None,
    budget: ProofBudget | None,
) -> tuple[int, ...] | None:
    """Return the least argument permutation identifying two relations, or None.

    Parameters
    ----------
    store : TermStore
        The store both bodies are interned in.
    predicted : LogicalVocabulary
        The induced vocabulary.
    gold : LogicalVocabulary
        The reference vocabulary.
    pair : tuple[str, str]
        The induced and gold relation names.
    calculus : Calculus | None
        The operators to normalize under, if any.
    normalization : Normalization | None
        The declared normalization treatment, or ``None`` with no calculus.
    service : ProofService | None
        The shared proof service for bodies that do not intern alike.
    budget : ProofBudget | None
        The use-site proof budget, required when a service is supplied.

    Returns
    -------
    tuple[int, ...] | None
        The permutation, or ``None`` when no permutation identifies them.
    """
    induced, reference = pair
    left = predicted.predicate(induced)
    right = gold.predicate(reference)

    if left.arity != right.arity:
        return None

    for candidate in orderings(range(left.arity)):
        moved = permuted_body(store, predicted, induced, candidate)

        if _same(
            store,
            moved,
            right.body,
            calculus,
            normalization,
            service,
            budget,
        ):
            return candidate

    return None


def _spine(term: Term, calculus: Calculus) -> tuple[Term, tuple[Term, ...]]:
    """Unwind an application spine into its head and its arguments.

    Parameters
    ----------
    term : Term
        The formula whose spine to unwind.
    calculus : Calculus
        The abstraction and application operators.

    Returns
    -------
    tuple[Term, tuple[Term, ...]]
        The head and the arguments, outermost last.
    """
    head = term
    arguments: list[Term] = []

    while head.op.name == calculus.app and len(head.args) == _APPLICATION_ARITY:
        head, argument = head.args
        arguments.append(argument)

    arguments.reverse()

    return head, tuple(arguments)


def _applied(
    store: TermStore,
    head: Term,
    arguments: tuple[Term, ...],
    calculus: Calculus,
) -> Term:
    """Rebuild an application spine over a head.

    Parameters
    ----------
    store : TermStore
        The store the spine is interned in.
    head : Term
        The head the arguments apply to.
    arguments : tuple[Term, ...]
        The arguments, outermost last.
    calculus : Calculus
        The abstraction and application operators.

    Returns
    -------
    Term
        The rebuilt spine.
    """
    operator = store.signature.by_name(calculus.app)
    held = head

    for argument in arguments:
        held = store.op(operator, held, argument)

    return held


def _permuted_spines(
    store: TermStore,
    term: Term,
    calculus: Calculus,
    chosen: Mapping[str, tuple[int, ...]],
) -> Term:
    """Reorder the arguments of every spine a chosen symbol heads.

    Parameters
    ----------
    store : TermStore
        The store the formula is interned in.
    term : Term
        The formula to rewrite.
    calculus : Calculus
        The abstraction and application operators.
    chosen : Mapping[str, tuple[int, ...]]
        The argument permutation picked for each symbol.

    Returns
    -------
    Term
        The rewritten formula.
    """
    head, arguments = _spine(term, calculus)
    moved = tuple(
        _permuted_spines(store, argument, calculus, chosen) for argument in arguments
    )
    name = _symbol_of(store, head)

    if name is None:
        inner = tuple(
            _permuted_spines(store, argument, calculus, chosen)
            for argument in head.args
        )

        return _applied(store, _rebuilt(store, head, inner), moved, calculus)

    order = chosen.get(name)

    if order is not None and len(order) == len(moved):
        moved = tuple(moved[position] for position in order)

    return _applied(store, head, moved, calculus)


def _symbols_in(store: TermStore, term: Term, found: dict[str, None]) -> None:
    """Record every symbol a formula names.

    Parameters
    ----------
    store : TermStore
        The store the formula is interned in.
    term : Term
        The formula to walk.
    found : dict[str, None]
        The symbols recorded so far.
    """
    name = _symbol_of(store, term)

    if name is not None:
        found[name] = None

    for argument in term.args:
        _symbols_in(store, argument, found)


def fiber_match(
    store: TermStore,
    gold: Term,
    predicted: Term,
    permutations: Mapping[str, Sequence[Sequence[int]]],
    /,
    *,
    calculus: Calculus,
    normalization: Normalization,
    service: ProofService | None = None,
    budget: ProofBudget | None = None,
) -> bool:
    """Report whether two formulas agree under some declared argument permutation.

    Every symbol the predicted formula names and the permutations declare is
    given one of its declared orders, independently of the others, and the
    reordered formula is compared with the gold one. The comparison is the
    identity of declared normal forms, and falls back to
    the proof service when one is supplied and the two do not intern alike.

    Parameters
    ----------
    store : TermStore
        The store both formulas are interned in.
    gold : Term
        The reference formula.
    predicted : Term
        The formula to reorder and compare.
    permutations : Mapping[str, Sequence[Sequence[int]]]
        The argument orders declared for each symbol.
    calculus : Calculus
        The abstraction and application operators, which name the spines whose
        arguments move and the redexes normalization contracts.
    normalization : Normalization
        The declared normalization treatment.
    service : ProofService | None
        The shared proof service consulted for a reordering that does not
        intern alike.
    budget : ProofBudget | None
        The use-site proof budget, required when a service is supplied.

    Returns
    -------
    bool
        Whether some choice of declared order makes the two the same formula.

    Raises
    ------
    ValueError
        If a declared order is not a permutation of the positions it is
        declared over.
    """
    if service is not None:
        service.open_step(_PROOF_SITE)

    found: dict[str, None] = {}
    _symbols_in(store, predicted, found)
    moving = tuple(name for name in sorted(found) if name in permutations)
    declared = tuple(
        tuple(_order(candidate, len(candidate)) for candidate in permutations[name])
        for name in moving
    )

    for combination in product(*declared):
        reordered = _permuted_spines(
            store, predicted, calculus, dict(zip(moving, combination, strict=True))
        )

        if _same(
            store,
            reordered,
            gold,
            calculus,
            normalization,
            service,
            budget,
        ):
            return True

    return False


def _identifications(
    store: TermStore,
    predicted: LogicalVocabulary,
    gold: LogicalVocabulary,
    calculus: Calculus | None,
    normalization: Normalization | None,
    service: ProofService | None,
    budget: ProofBudget | None,
) -> dict[tuple[str, str], tuple[int, ...]]:
    """Return the least identifying permutation for every relation pair that has one.

    Parameters
    ----------
    store : TermStore
        The store both vocabularies are interned in.
    predicted : LogicalVocabulary
        The induced vocabulary.
    gold : LogicalVocabulary
        The reference vocabulary.
    calculus : Calculus | None
        The operators to normalize under, if any.
    service : ProofService | None
        The shared proof service for bodies that do not intern alike.
    budget : ProofBudget | None
        The use-site proof budget, required when a service is supplied.

    Returns
    -------
    dict[tuple[str, str], tuple[int, ...]]
        The permutation identifying each identifiable pair.
    """
    found: dict[tuple[str, str], tuple[int, ...]] = {}

    for induced in predicted.symbols:
        for reference in gold.symbols:
            order = _matching_order(
                store,
                predicted,
                gold,
                (induced, reference),
                calculus,
                normalization,
                service,
                budget,
            )

            if order is not None:
                found[induced, reference] = order

    return found


def _vocabulary(value: LogicalVocabulary) -> LogicalVocabulary:
    """Return a validated vocabulary.

    Parameters
    ----------
    value : LogicalVocabulary
        Candidate vocabulary.

    Returns
    -------
    LogicalVocabulary
        The vocabulary.

    Raises
    ------
    TypeError
        If ``value`` is not a vocabulary.
    """
    if not isinstance(value, LogicalVocabulary):
        msg = f"a scored vocabulary is not one: {value!r}"
        raise TypeError(msg)

    return value


def _comparable(predicted: LogicalVocabulary, gold: LogicalVocabulary) -> None:
    """Refuse two vocabularies that do not name their argument positions alike.

    Parameters
    ----------
    predicted : LogicalVocabulary
        The induced vocabulary.
    gold : LogicalVocabulary
        The reference vocabulary.

    Raises
    ------
    TypeError
        If either value is not a vocabulary.
    ValueError
        If the two declare different argument slots.
    """
    if _vocabulary(predicted).slots != _vocabulary(gold).slots:
        msg = (
            f"an induced vocabulary declares slots {predicted.slots!r} and its "
            f"gold vocabulary declares {gold.slots!r}, so their argument "
            "positions cannot be compared"
        )
        raise ValueError(msg)


def _result(matched: int, total: int, *, induced: int) -> MetricResult:
    """Return one recovery share and the counts behind it.

    Parameters
    ----------
    matched : int
        How many gold relations were identified.
    total : int
        How many gold relations there are.
    induced : int
        How many induced relations there are.

    Returns
    -------
    MetricResult
        The share, the counts and the inventory sizes.
    """
    return MetricResult(
        None if total == 0 else matched / total,
        total,
        Resampling(_UNIT),
        components={
            "matched": float(matched),
            "induced_labels": float(induced),
            "gold_labels": float(total),
        },
    )


def fiber_recovery(
    store: TermStore,
    predicted: LogicalVocabulary,
    gold: LogicalVocabulary,
    /,
    *,
    policy: AlignmentPolicy,
    split: Split,
    calculus: Calculus | None = None,
    normalization: Normalization | None = None,
    service: ProofService | None = None,
    budget: ProofBudget | None = None,
    floor: MetricResult | None = None,
) -> FiberRecovery:
    """Measure recovery of an induced logical vocabulary against a gold one.

    A relation pair counts once when some permutation of the induced
    relation's arguments makes its body the gold relation's body in the
    ambient theory, and the assignment policy chooses the relabeling from
    those counts. Recovery up to the fiber reads the policy's relabeling;
    recovery under the fixed alphabet reads the identity relabeling with the
    identity argument order, so a vocabulary that is gold under a relabeling
    scores one on the first and less on the second.

    Parameters
    ----------
    store : TermStore
        The store both vocabularies are interned in.
    predicted : LogicalVocabulary
        The induced vocabulary.
    gold : LogicalVocabulary
        The reference vocabulary.
    policy : AlignmentPolicy
        The rule that chooses the relabeling.
    split : Split
        The split the counts were accumulated on.
    calculus : Calculus | None
        The operators to normalize bodies under, or ``None`` to compare them
        as they stand.
    normalization : Normalization | None
        The declared normalization treatment, or ``None`` with no calculus.
    service : ProofService | None
        The shared proof service for bodies that do not intern alike.
    budget : ProofBudget | None
        The use-site proof budget, required when a service is supplied.
    floor : MetricResult | None
        The chance recovery at this signature size, against which the reported
        recovery is flagged.

    Returns
    -------
    FiberRecovery
        Both recoveries, the fitted relabeling and the chosen permutations.

    Raises
    ------
    TypeError
        If a vocabulary is not one.
    ValueError
        If the two vocabularies declare different argument slots.
    """
    if service is not None:
        service.open_step(_PROOF_SITE)

    _check_normalization(calculus, normalization)
    _comparable(predicted, gold)
    identified = _identifications(
        store, predicted, gold, calculus, normalization, service, budget
    )
    counts = {
        (induced, reference): int((induced, reference) in identified)
        for induced in predicted.symbols
        for reference in gold.symbols
    }
    alignment = policy.fit(Contingency(counts, split=split))
    chosen = tuple(
        (induced, identified[induced, reference])
        for induced, reference in sorted(alignment.mapping.items())
    )
    reached = set(alignment.mapping.values())
    standing = _under_fixed_alphabet(
        store, predicted, gold, calculus, normalization, service, budget
    )
    total = len(gold.symbols)
    induced_labels = len(predicted.symbols)

    return FiberRecovery(
        _result(len(reached), total, induced=induced_labels),
        _result(standing, total, induced=induced_labels),
        alignment,
        chosen,
        floor,
    )


def _under_fixed_alphabet(
    store: TermStore,
    predicted: LogicalVocabulary,
    gold: LogicalVocabulary,
    calculus: Calculus | None,
    normalization: Normalization | None,
    service: ProofService | None,
    budget: ProofBudget | None,
) -> int:
    """Count the gold relations an induced relation of the same name already is.

    Parameters
    ----------
    store : TermStore
        The store both vocabularies are interned in.
    predicted : LogicalVocabulary
        The induced vocabulary.
    gold : LogicalVocabulary
        The reference vocabulary.
    calculus : Calculus | None
        The operators to normalize bodies under, if any.
    normalization : Normalization | None
        The declared normalization treatment, or ``None`` with no calculus.
    service : ProofService | None
        The shared proof service for bodies that do not intern alike.
    budget : ProofBudget | None
        The use-site proof budget, required when a service is supplied.

    Returns
    -------
    int
        How many gold relations keep both their name and their argument order.
    """
    shared = set(predicted.symbols) & set(gold.symbols)
    matched = 0

    for symbol in sorted(shared):
        left = predicted.predicate(symbol)
        right = gold.predicate(symbol)

        if left.arity != right.arity:
            continue

        if _same(
            store,
            left.body,
            right.body,
            calculus,
            normalization,
            service,
            budget,
        ):
            matched += 1

    return matched


def fiber_size(
    store: TermStore,
    vocabulary: LogicalVocabulary,
    /,
    *,
    calculus: Calculus | None = None,
    normalization: Normalization | None = None,
    service: ProofService | None = None,
    budget: ProofBudget | None = None,
    limit: int,
) -> int:
    """Return how many relabelings with argument permutations fix a vocabulary.

    The count is the product of the factorials of the classes of relations
    that some argument permutation identifies with one another and the sizes
    of each relation's own symmetry group, which is the closed form of the
    stabilizer whenever the identification relation is an equivalence, as it
    is for term identity and for logical equivalence.

    Parameters
    ----------
    store : TermStore
        The store the vocabulary is interned in.
    vocabulary : LogicalVocabulary
        The vocabulary whose symmetries to count.
    calculus : Calculus | None
        The operators to normalize bodies under, if any.
    normalization : Normalization | None
        The declared normalization treatment, or ``None`` with no calculus.
    service : ProofService | None
        The shared proof service for bodies that do not intern alike.
    budget : ProofBudget | None
        The use-site proof budget, required when a service is supplied.
    limit : int
        The largest number of argument permutations one relation may be
        enumerated over.

    Returns
    -------
    int
        How many group elements leave the vocabulary as it was.

    Raises
    ------
    TypeError
        If ``vocabulary`` is not one.
    ValueError
        If a relation takes more arguments than ``limit`` permits.
    """
    if service is not None:
        service.open_step(_PROOF_SITE)

    _check_normalization(calculus, normalization)
    _vocabulary(vocabulary)

    for entry in vocabulary.predicates:
        if factorial(entry.arity) > limit:
            msg = (
                f"{entry.symbol!r} takes {entry.arity} arguments, whose "
                f"{factorial(entry.arity)} permutations exceed the limit of {limit}"
            )
            raise ValueError(msg)

    identified = _identifications(
        store,
        vocabulary,
        vocabulary,
        calculus,
        normalization,
        service,
        budget,
    )
    size = 1
    classes: list[list[str]] = []

    for symbol in vocabulary.symbols:
        size *= sum(
            1
            for candidate in orderings(range(vocabulary.predicate(symbol).arity))
            if _same(
                store,
                permuted_body(store, vocabulary, symbol, candidate),
                vocabulary.predicate(symbol).body,
                calculus,
                normalization,
                service,
                budget,
            )
        )
        joined = next(
            (held for held in classes if (symbol, held[0]) in identified), None
        )

        if joined is None:
            classes.append([symbol])
        else:
            joined.append(symbol)

    for held in classes:
        size *= factorial(len(held))

    return size


def _drawn(
    store: TermStore,
    pool: Sequence[Predicate],
    slots: tuple[str, ...],
    size: int,
    rng: Random,
) -> LogicalVocabulary:
    """Return a vocabulary whose relations carry meanings drawn from a pool.

    Parameters
    ----------
    store : TermStore
        The store the pool's bodies are interned in.
    pool : Sequence[Predicate]
        The relations meanings are drawn from, with replacement.
    slots : tuple[str, ...]
        The argument slots the drawn vocabulary declares.
    size : int
        How many relations to draw.
    rng : Random
        The stream the draw reads.

    Returns
    -------
    LogicalVocabulary
        The drawn vocabulary, whose relations are named by their position.
    """
    held: list[Predicate] = []
    source = LogicalVocabulary(pool, slots=slots)

    for position in range(size):
        taken = pool[rng.randrange(len(pool))]
        order = list(range(taken.arity))
        rng.shuffle(order)
        held.append(
            Predicate(
                str(position),
                taken.arity,
                permuted_body(store, source, taken.symbol, order),
            )
        )

    return LogicalVocabulary(held, slots=slots)


def _interval(scores: Sequence[float], level: float) -> Interval:
    """Return the order-statistic interval over a sequence of draws.

    Parameters
    ----------
    scores : Sequence[float]
        One score per draw.
    level : float
        Nominal coverage proportion.

    Returns
    -------
    Interval
        The interval the draws cut out.
    """
    ordered = sorted(scores)
    count = len(ordered)
    tail = (1.0 - level) / 2.0
    low = min(count - 1, int(tail * count))
    high = max(low, min(count - 1, ceil((1.0 - tail) * count) - 1))

    return Interval(ordered[low], ordered[high], level)


def fiber_floor(
    store: TermStore,
    gold: LogicalVocabulary,
    pool: Sequence[Predicate],
    /,
    *,
    size: int,
    policy: AlignmentPolicy,
    split: Split,
    rng: Random,
    draws: int,
    calculus: Calculus | None = None,
    normalization: Normalization | None = None,
    service: ProofService | None = None,
    budget: ProofBudget | None = None,
    level: float,
) -> MetricResult:
    """Return the recovery a vocabulary of a given size reaches by chance.

    Each draw builds a vocabulary of ``size`` relations whose meanings come
    from the pool at random under a random argument permutation, and scores it
    against the gold vocabulary through the same assignment policy the
    observed recovery runs through, so the floor carries the optimism of
    maximizing over the group rather than only the chance of a single guess.

    Parameters
    ----------
    store : TermStore
        The store the vocabularies are interned in.
    gold : LogicalVocabulary
        The reference vocabulary each draw is scored against.
    pool : Sequence[Predicate]
        The relations each draw takes its meanings from.
    size : int
        How many relations a drawn vocabulary declares.
    policy : AlignmentPolicy
        The rule that chooses each draw's relabeling.
    split : Split
        The split the counts are accumulated on.
    rng : Random
        The stream the draws read, whose state this advances.
    draws : int
        How many vocabularies to draw.
    calculus : Calculus | None
        The operators to normalize bodies under, if any.
    normalization : Normalization | None
        The declared normalization treatment, or ``None`` with no calculus.
    service : ProofService | None
        The shared proof service for bodies that do not intern alike.
    budget : ProofBudget | None
        The use-site proof budget, required when a service is supplied.
    level : float
        Nominal coverage of the reported interval.

    Returns
    -------
    MetricResult
        The mean recovery over the draws, its interval, and the signature size
        it was computed at.

    Raises
    ------
    TypeError
        If ``gold`` is not a vocabulary.
    ValueError
        If the size, the draw count or the pool is empty.
    """
    _check_normalization(calculus, normalization)
    _vocabulary(gold)

    if size < 1:
        msg = f"a floor is computed at a signature size of at least one, not {size}"
        raise ValueError(msg)

    if draws < 1:
        msg = f"a floor needs at least one draw, not {draws}"
        raise ValueError(msg)

    if not pool:
        msg = "a floor draws its meanings from a pool, which is empty"
        raise ValueError(msg)

    scores: list[float] = []

    for _ in range(draws):
        recovered = fiber_recovery(
            store,
            _drawn(store, pool, gold.slots, size, rng),
            gold,
            policy=policy,
            split=split,
            calculus=calculus,
            normalization=normalization,
            service=service,
            budget=budget,
        ).up_to_fiber.value
        scores.append(0.0 if recovered is None else recovered)

    return MetricResult(
        fmean(scores),
        len(gold.symbols),
        Resampling(_UNIT),
        components={
            "induced_labels": float(size),
            "gold_labels": float(len(gold.symbols)),
            "draws": float(draws),
        },
        ci=_interval(scores, level),
    )
