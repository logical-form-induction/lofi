r"""Gold CCGbank derivations projected to dependencies and scored against PARG.

A CCGbank AUTO derivation states every node's category and head daughter but
not the combinatory rule that built it. :func:`gold_sentence` classifies each
step structurally and returns a :class:`~lofi.data.Derivation` whose lexical
labels are the predicate-argument categories CCGbank prints beside each word,
such as ``(S[dcl]\NP_10)/(S[b]_11\NP_10:B)_11``. The numbered markup tells
:class:`MarkedCategoryAlgebra` which functors pass their argument's head to
their result.

Scores are micro-aggregated over a corpus: matched, predicted and gold arc
counts are summed across sentences before any ratio is taken. A labeled arc is
the 4-tuple of head position, argument position, the head's unmarked lexical
category and the argument slot, compared as a multiset. Each unmatched arc is
attributed to a rule name, so a shortfall names the projection that produced
it: a spurious arc to the rule that emitted it, and a missed arc to the rule at
the smallest derivation node spanning both of its words. A sentence whose
projection raises loses all of its gold arcs, and those are attributed to the
step that raised, named together with the steps that built its daughters.
"""

import re
from collections import Counter
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, final, override

from lofi._frozen import FrozenValue
from lofi._types import HeadSelection
from lofi.data import Derivation, DerivationNode, Token, TokenSpan
from lofi.data.corpora.ccgbank._auto_parse import AutoLeaf
from lofi.data.deps import (
    CCGDependencyProjector,
    CoindexedPath,
    project_dependencies,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from lofi.data.corpora.ccgbank._auto_parse import AutoDerivation, AutoNode
    from lofi.data.corpora.ccgbank._parg_parse import PargSentence
    from lofi.data.deps import (
        CategoryDirection,
        DecoratedCategory,
        DependencyProjector,
        HeadProjection,
        ProjectedDependency,
    )

__all__ = [
    "CCGBANK_HEAD_SELECTIONS",
    "UNEVALUATED_DEPENDENCIES",
    "GoldSentence",
    "LabeledArc",
    "MarkedCategoryAlgebra",
    "RoundTripCounts",
    "ccgbank_round_trip_projector",
    "evaluated",
    "gold_sentence",
    "round_trip",
    "score_arcs",
]

type LabeledArc = tuple[int, int, str, int]
"""A head position, argument position, unmarked head category and slot."""

type _Projector = DependencyProjector[
    str, ProjectedDependency[str], DecoratedCategory[str]
]
"""A dependency projector over CCGbank category strings."""

CCGBANK_HEAD_SELECTIONS: Final[Mapping[str, HeadSelection]] = MappingProxyType(
    {
        "forward-application": HeadSelection("functor", 0),
        "backward-application": HeadSelection("functor", 1),
        "forward-composition": HeadSelection("functor", 0),
        "backward-composition": HeadSelection("functor", 1),
        "coordination": HeadSelection("left-conjunct", 0),
        "conjunct": HeadSelection("non-absorbing", 1),
        "punctuation-left": HeadSelection("non-absorbing", 1),
        "punctuation-right": HeadSelection("non-absorbing", 0),
        "binary-type-changing-left": HeadSelection("non-absorbing", 0),
        "binary-type-changing-right": HeadSelection("non-absorbing", 1),
        "type-raising": HeadSelection("pass-through", 0),
        "type-changing": HeadSelection("pass-through", 0),
    }
)
"""The head selection for every rule name :func:`gold_sentence` assigns.

A binary step that is neither application, composition, coordination nor
punctuation is named by the daughter AUTO marks as its head, and that daughter
is the one whose head and shape the step keeps.
"""

_FEATURE: Final = re.compile(r"\[[^\]]*\]|_[0-9]+|:[A-Z]")
_INDEX: Final = re.compile(r"_([0-9]+)(?::[A-Z])?")
_CONJ: Final = "[conj]"
_PUNCTUATION: Final = frozenset((",", ".", ":", ";", "LRB", "RRB", "LQU", "RQU"))


def _closing(category: str) -> int:
    """Return the position of the parenthesis closing the one at position zero."""
    depth = 0

    for position, character in enumerate(category):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1

            if depth == 0:
                return position

    msg = f"a CCGbank category has unbalanced parentheses: {category!r}"
    raise ValueError(msg)


def _unwrapped(category: str) -> str:
    """Return a category without parentheses enclosing all of it."""
    while category.startswith("(") and _closing(category) == len(category) - 1:
        category = category[1:-1]

    return category


@cache
def _parts(category: str) -> tuple[str, str, str] | None:
    """Return a functor's result, slash and argument, or ``None`` for an atom."""
    main = None
    depth = 0

    for position, character in enumerate(category):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character in "/\\" and depth == 0:
            main = position

    if main is not None:
        return (
            _unwrapped(category[:main]),
            category[main],
            _unwrapped(category[main + 1 :]),
        )

    if category.startswith("("):
        end = _closing(category)

        if end == len(category) - 1 or _INDEX.fullmatch(category[end + 1 :]):
            return _parts(category[1:end])

    return None


@cache
def _erased(category: str) -> str:
    """Return a category without features, head indices or markup letters."""
    parts = _parts(_unwrapped(category))

    if parts is None:
        return _FEATURE.sub("", _unwrapped(category))

    result, slash, argument = parts
    return f"({_erased(result)}{slash}{_erased(argument)})"


def _head_index(category: str) -> str | None:
    """Return the head index a marked category states for itself, if any."""
    if category.startswith("("):
        end = _closing(category)
        stated = _INDEX.fullmatch(category[end + 1 :])
        return None if stated is None else stated.group(1)

    if _parts(category) is not None:
        return None

    stated = _INDEX.search(category)
    return None if stated is None else stated.group(1)


@cache
def _coindexed(category: str) -> tuple[CoindexedPath, ...]:
    """Return the argument paths sharing a numbered index within a category."""
    found: dict[int, list[tuple[CategoryDirection, ...]]] = {}
    pending: list[tuple[str, tuple[CategoryDirection, ...]]] = [(category, ())]

    while pending:
        current, path = pending.pop()

        if path and path[-1] == "argument":
            stated = _head_index(current)

            if stated is not None:
                found.setdefault(int(stated), []).append(path)

        parts = _parts(current)

        if parts is not None:
            result, _, argument = parts
            pending.append((result, (*path, "result")))
            pending.append((argument, (*path, "argument")))

    return tuple(
        CoindexedPath(path, index)
        for index, paths in sorted(found.items())
        if len(paths) > 1
        for path in sorted(paths)
    )


def _strip_conj(category: str) -> str:
    """Return a category without its trailing conjunct marker."""
    return category.removesuffix(_CONJ)


@final
class MarkedCategoryAlgebra(FrozenValue):
    r"""Structural operations over CCGbank category strings.

    A category may carry CCGbank's numbered markup. Equivalence ignores
    features, numbered indices and markup letters, so ``S[dcl]\NP_8`` is
    equivalent to ``S\NP``. A marked functor shares its head when its result
    and argument state the same index, as ``N_73/N_73`` does and
    ``(S[dcl]\NP_10)/(S[b]_11\NP_10:B)_11`` does not. An unmarked functor
    shares its head when its result and argument are equivalent or when it is
    equivalent to one of the declared head-sharing categories.

    Parameters
    ----------
    head_sharing : Iterable[str]
        Unmarked categories whose result takes the argument's head although the
        two differ, such as a determiner category ``NP/N``.
    """

    _noun = "marked category algebra"

    __slots__ = ("_head_sharing",)

    def __init__(self, head_sharing: Iterable[str] = ()) -> None:
        object.__setattr__(
            self,
            "_head_sharing",
            frozenset(_erased(category) for category in head_sharing),
        )

    @property
    def head_sharing(self) -> frozenset[str]:
        """Return the declared head-sharing categories, erased."""
        return self._head_sharing

    def is_functor(self, category: str, /) -> bool:
        """Return whether a category has a result and an argument."""
        return _parts(category) is not None

    def result(self, category: str, /) -> str:
        """Return a functor's result, keeping its markup.

        Raises
        ------
        ValueError
            If the category is atomic.
        """
        return self._split(category)[0]

    def argument(self, category: str, /) -> str:
        """Return a functor's argument, keeping its markup.

        Raises
        ------
        ValueError
            If the category is atomic.
        """
        return self._split(category)[2]

    def coindices(self, category: str, /) -> tuple[CoindexedPath, ...]:
        r"""Return the argument paths that share a numbered index.

        In ``(S[dcl]\NP_10)/(S[b]_11\NP_10:B)_11`` the subject of the result and
        the subject of the argument share index 10. An index stated on only one
        argument node coindexes nothing, and an unmarked category has none.
        """
        return _coindexed(category)

    def equivalent(self, left: str, right: str, /) -> bool:
        """Return whether two categories agree once features and markup are erased."""
        return _erased(left) == _erased(right)

    def shares_head(self, category: str, /) -> bool:
        """Return whether a functor's result takes its argument's head."""
        parts = _parts(category)

        if parts is None:
            return False

        result, _, argument = parts

        if _INDEX.search(category):
            stated = _head_index(result)
            return stated is not None and stated == _head_index(argument)

        return (
            _erased(result) == _erased(argument)
            or _erased(category) in self._head_sharing
        )

    @staticmethod
    def _split(category: str) -> tuple[str, str, str]:
        """Return a functor's parts or raise for an atom."""
        parts = _parts(category)

        if parts is None:
            msg = f"category {category!r} is not a CCG functor"
            raise ValueError(msg)

        return parts

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another algebra declares the same head-sharing categories."""
        if type(other) is not MarkedCategoryAlgebra:
            return NotImplemented

        return self._head_sharing == other._head_sharing

    @override
    def __hash__(self) -> int:
        """Return a hash over the declared head-sharing categories."""
        return hash((MarkedCategoryAlgebra, self._head_sharing))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"MarkedCategoryAlgebra({tuple(sorted(self._head_sharing))!r})"


def _applies(functor: str, slash: str, other: str, parent: str) -> bool:
    """Return whether a functor applies across a slash to another category."""
    parts = _parts(functor)
    return (
        parts is not None
        and parts[1] == slash
        and _erased(parts[2]) == _erased(other)
        and _erased(parts[0]) == _erased(parent)
    )


def _composes(functor: str, slash: str, other: str, parent: str) -> bool:
    """Return whether a functor composes across a slash into another category."""
    parts = _parts(functor)

    if parts is None or parts[1] != slash:
        return False

    result, _, argument = parts
    outside: list[tuple[str, str]] = []
    current = other

    while (inner := _parts(current)) is not None:
        current, inner_slash, inner_argument = inner
        outside.append((inner_slash, inner_argument))

        if _erased(current) == _erased(argument):
            built = result

            for outer_slash, outer_argument in reversed(outside):
                built = f"({built}){outer_slash}({outer_argument})"

            return _erased(built) == _erased(parent)

    return False


def _binary_rule(parent: str, left: str, right: str, head: int) -> str:
    """Name the rule that builds a binary AUTO node from its daughters."""
    rules: tuple[tuple[str, Callable[[], bool]], ...] = (
        ("conjunct", lambda: parent.endswith(_CONJ)),
        (
            "coordination",
            lambda: (
                right.endswith(_CONJ) and _erased(_strip_conj(right)) == _erased(parent)
            ),
        ),
        ("forward-application", lambda: _applies(left, "/", right, parent)),
        ("backward-application", lambda: _applies(right, "\\", left, parent)),
        (
            "punctuation-left",
            lambda: left in _PUNCTUATION and _erased(right) == _erased(parent),
        ),
        (
            "punctuation-right",
            lambda: right in _PUNCTUATION and _erased(left) == _erased(parent),
        ),
        ("forward-composition", lambda: _composes(left, "/", right, parent)),
        ("backward-composition", lambda: _composes(right, "\\", left, parent)),
    )
    fallback = (
        "binary-type-changing-left" if head == 0 else "binary-type-changing-right"
    )
    return next((name for name, holds in rules if holds()), fallback)


def _unary_rule(parent: str, child: str) -> str:
    """Name the rule that builds a unary AUTO node from its daughter."""
    outer = _parts(parent)

    if outer is not None:
        target, slash, raised = outer
        inner = _parts(raised)

        if (
            inner is not None
            and inner[1] != slash
            and _erased(inner[0]) == _erased(target)
            and _erased(inner[2]) == _erased(child)
        ):
            return "type-raising"

    return "type-changing"


class GoldSentence(NamedTuple):
    """A gold AUTO derivation prepared for dependency projection."""

    #: The derivation, with marked lexical labels and a rule name on every step.
    derivation: Derivation[str]
    #: The words, one per lexical node, in surface order.
    tokens: tuple[Token, ...]
    #: Each word's unmarked lexical category, by position.
    categories: tuple[str, ...]
    #: The rule at the smallest step spanning each pair of positions.
    covering: Mapping[tuple[int, int], str]


def gold_sentence(tree: AutoNode) -> GoldSentence:
    """Return a gold AUTO tree as a rule-named derivation and its words.

    Parameters
    ----------
    tree : AutoNode
        The root of one parsed AUTO derivation.

    Raises
    ------
    ValueError
        If an internal node has neither one nor two daughters.
    """
    nodes: dict[int, DerivationNode[str]] = {}
    tokens: list[Token] = []
    categories: list[str] = []
    covering: dict[tuple[int, int], str] = {}

    def visit(node: AutoNode) -> tuple[int, int, int, str]:
        if isinstance(node, AutoLeaf):
            position = len(tokens)
            tokens.append(Token(position, node.word))
            categories.append(node.category)
            identifier = len(nodes)
            nodes[identifier] = DerivationNode(
                TokenSpan(position, position + 1),
                node.slotted or node.category,
            )
            return identifier, position, position + 1, node.category

        daughters = tuple(visit(child) for child in node.children)

        if len(daughters) == 1:
            rule = _unary_rule(node.category, daughters[0][3])
        elif len(daughters) == 2:
            rule = _binary_rule(
                node.category, daughters[0][3], daughters[1][3], node.head
            )
        else:
            msg = f"an AUTO node has {len(daughters)} daughters"
            raise ValueError(msg)

        start, end = daughters[0][1], daughters[-1][2]

        if len(daughters) > 1:
            boundary = daughters[0][2]

            for head in range(start, boundary):
                for argument in range(boundary, end):
                    covering[head, argument] = rule
                    covering[argument, head] = rule

        identifier = len(nodes)
        nodes[identifier] = DerivationNode(
            TokenSpan(start, end),
            _strip_conj(node.category),
            rule=rule,
            children=tuple(daughter[0] for daughter in daughters),
        )
        return identifier, start, end, node.category

    root = visit(tree)[0]
    return GoldSentence(
        Derivation(root, nodes),
        tuple(tokens),
        tuple(categories),
        MappingProxyType(covering),
    )


def _tallied(pairs: Iterable[tuple[str, int]]) -> tuple[tuple[str, int], ...]:
    """Return summed positive rule counts, largest first and then by name."""
    counts = Counter[str]()

    for rule, count in pairs:
        counts[rule] += count

    return tuple(
        sorted(
            ((rule, count) for rule, count in counts.items() if count > 0),
            key=lambda entry: (-entry[1], entry[0]),
        )
    )


@final
class RoundTripCounts(FrozenValue):
    """Micro-aggregated labeled dependency counts with per-rule failures.

    Adding two counts sums every field, which is how sentence counts become
    corpus counts. Every gold arc that is not matched is counted exactly once,
    either in ``missed`` or in ``raised``.

    Parameters
    ----------
    matched : int
        Predicted arcs that match a gold arc, as a multiset intersection.
    predicted : int
        Predicted arcs.
    gold : int
        Gold arcs.
    spurious : Iterable[tuple[str, int]]
        Unmatched predicted arcs, counted by the rule that emitted them.
    missed : Iterable[tuple[str, int]]
        Unmatched gold arcs of projected sentences, counted by the rule at the
        smallest step spanning both words. A rule named more than once has its
        counts summed.
    raised : Iterable[tuple[str, int]]
        Gold arcs of sentences whose projection raised, counted by the step
        that raised, written as its rule followed by its daughters' rules in
        parentheses, such as ``forward-application(lexical, type-changing)``.
    unprojected : int
        Sentences whose projection raised and so predicted nothing.
    """

    _noun = "round-trip count"

    __slots__ = (
        "_gold",
        "_matched",
        "_missed",
        "_predicted",
        "_raised",
        "_spurious",
        "_unprojected",
    )

    def __init__(
        self,
        matched: int = 0,
        predicted: int = 0,
        gold: int = 0,
        *,
        spurious: Iterable[tuple[str, int]] = (),
        missed: Iterable[tuple[str, int]] = (),
        raised: Iterable[tuple[str, int]] = (),
        unprojected: int = 0,
    ) -> None:
        if min(matched, predicted, gold, unprojected) < 0:
            msg = "a round-trip count is negative"
            raise ValueError(msg)

        if matched > min(predicted, gold):
            msg = f"{matched} matched arcs exceed {predicted} predicted or {gold} gold"
            raise ValueError(msg)

        object.__setattr__(self, "_matched", matched)
        object.__setattr__(self, "_predicted", predicted)
        object.__setattr__(self, "_gold", gold)
        object.__setattr__(self, "_spurious", _tallied(spurious))
        object.__setattr__(self, "_missed", _tallied(missed))
        object.__setattr__(self, "_raised", _tallied(raised))
        object.__setattr__(self, "_unprojected", unprojected)

    @property
    def matched(self) -> int:
        """Return the matched arc count."""
        return self._matched

    @property
    def predicted(self) -> int:
        """Return the predicted arc count."""
        return self._predicted

    @property
    def gold(self) -> int:
        """Return the gold arc count."""
        return self._gold

    @property
    def spurious(self) -> tuple[tuple[str, int], ...]:
        """Return unmatched predicted arcs by rule, largest count first."""
        return self._spurious

    @property
    def missed(self) -> tuple[tuple[str, int], ...]:
        """Return unmatched gold arcs by rule, largest count first."""
        return self._missed

    @property
    def raised(self) -> tuple[tuple[str, int], ...]:
        """Return gold arcs lost to a raising projection by step, largest first."""
        return self._raised

    @property
    def unprojected(self) -> int:
        """Return the number of sentences whose projection raised."""
        return self._unprojected

    @property
    def labeled_precision(self) -> float:
        """Return matched over predicted arcs, or zero when nothing was predicted."""
        return self._matched / self._predicted if self._predicted else 0.0

    @property
    def labeled_recall(self) -> float:
        """Return matched over gold arcs, or zero when there are no gold arcs."""
        return self._matched / self._gold if self._gold else 0.0

    def __add__(self, other: RoundTripCounts) -> RoundTripCounts:
        """Return the counts of both corpora together."""
        return RoundTripCounts(
            self._matched + other._matched,
            self._predicted + other._predicted,
            self._gold + other._gold,
            spurious=(*self._spurious, *other._spurious),
            missed=(*self._missed, *other._missed),
            raised=(*self._raised, *other._raised),
            unprojected=self._unprojected + other._unprojected,
        )

    def _fields(
        self,
    ) -> tuple[
        int,
        int,
        int,
        tuple[tuple[str, int], ...],
        tuple[tuple[str, int], ...],
        tuple[tuple[str, int], ...],
        int,
    ]:
        """Return fields in declaration order."""
        return (
            self._matched,
            self._predicted,
            self._gold,
            self._spurious,
            self._missed,
            self._raised,
            self._unprojected,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether other counts have the same fields."""
        if type(other) is not RoundTripCounts:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((RoundTripCounts, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return (
            f"RoundTripCounts({self._matched!r}, {self._predicted!r}, "
            f"{self._gold!r}, spurious={self._spurious!r}, "
            f"missed={self._missed!r}, raised={self._raised!r}, "
            f"unprojected={self._unprojected!r})"
        )


def score_arcs(
    predicted: Iterable[tuple[LabeledArc, str]],
    gold: Iterable[tuple[LabeledArc, str]],
) -> RoundTripCounts:
    """Score one sentence's labeled arcs as multisets.

    Arcs whose head and argument positions coincide are dropped from both
    sides before matching.

    Parameters
    ----------
    predicted : Iterable[tuple[LabeledArc, str]]
        Projected arcs, each with the rule that emitted it.
    gold : Iterable[tuple[LabeledArc, str]]
        Gold arcs, each with the rule a miss is attributed to.
    """
    held_predicted = tuple(entry for entry in predicted if entry[0][0] != entry[0][1])
    held_gold = tuple(entry for entry in gold if entry[0][0] != entry[0][1])
    predicted_arcs = Counter(arc for arc, _ in held_predicted)
    gold_arcs = Counter(arc for arc, _ in held_gold)
    matched = (predicted_arcs & gold_arcs).total()

    def unmatched(
        entries: tuple[tuple[LabeledArc, str], ...],
        excess: Counter[LabeledArc],
    ) -> Counter[str]:
        by_rule = Counter[str]()

        for arc, rule in entries:
            if excess[arc] > 0:
                excess[arc] -= 1
                by_rule[rule] += 1

        return by_rule

    return RoundTripCounts(
        matched,
        predicted_arcs.total(),
        gold_arcs.total(),
        spurious=unmatched(held_predicted, predicted_arcs - gold_arcs).items(),
        missed=unmatched(held_gold, gold_arcs - predicted_arcs).items(),
    )


UNEVALUATED_DEPENDENCIES: Final[frozenset[tuple[str, int, str | None, str | None]]] = (
    frozenset(
        (
            ("conj", 1, None, None),
            ("(S[to]\\NP)/(S[b]\\NP)", 1, None, None),
            ("(S[pt]\\NP)/(S[ng]\\NP)", 1, "been", None),
            ("(S[pt]\\NP)/(S[pss]\\NP)", 1, "been", None),
            ("(S[pt]\\NP)/(S[adj]\\NP)", 1, "been", None),
            ("(S[pt]\\NP)/NP", 1, "been", "there"),
            ("(S[pt]\\NP)/NP", 1, "been", "There"),
            ("(S[b]\\NP)/NP", 1, "be", "there"),
            ("(S[b]\\NP)/NP", 1, "be", "There"),
            ("(S[b]\\NP)/(S[pss]\\NP)", 1, "be", None),
            ("(S[b]\\NP)/(S[adj]\\NP)", 1, "be", None),
            ("(S[b]\\NP)/(S[ng]\\NP)", 1, "be", None),
            ("(S[b]\\NP)/(S[pt]\\NP)", 1, "have", None),
            ("(S[b]\\NP)/(S[to]\\NP)", 1, "have", None),
            ("(S[ng]\\NP)/(S[to]\\NP)", 1, "going", None),
            ("S[adj]\\NP", 1, "Here", None),
            ("((NP\\NP)/(NP\\NP))/NP", 1, "from", None),
        )
    )
)
r"""Predicted dependencies the C&C CCGbank evaluation leaves unscored.

Each entry is a head's unmarked lexical category, a slot, and the head and
argument words it is restricted to, where ``None`` means any word. They are
the category entries of the ``%noeval`` table in C&C's
``evaluateDepsCCGbank.pl``: the subject of infinitival ``to``, the first
conjunct of ``conj``, the subjects of nonfinite ``be``, ``been``, ``have``
and ``going``, the subject of ``Here`` as ``S[adj]\NP``, and the first
slot of ``from`` as ``((NP\NP)/(NP\NP))/NP``. C&C keys each entry to one
of its own rule ids as well; no derivation here has those ids, so an entry
matches whatever step emitted the arc, and the table's entries that name a
rule id alone are absent.
"""


def evaluated(arc: LabeledArc, words: Sequence[str]) -> bool:
    """Return whether a labeled arc is scored rather than left unevaluated.

    Parameters
    ----------
    arc : LabeledArc
        The arc. A position outside ``words`` matches only entries that name
        no word for it.
    words : Sequence[str]
        The sentence's words.
    """
    head, argument, category, slot = arc
    predicates = (None, *words[head : head + 1])
    fillers = (None, *words[argument : argument + 1])
    return not any(
        (category, slot, predicate, filler) in UNEVALUATED_DEPENDENCIES
        for predicate in predicates
        for filler in fillers
    )


def ccgbank_round_trip_projector() -> CCGDependencyProjector[str]:
    r"""Return the dependency projector the gold round trip validates.

    The algebra reads CCGbank's numbered markup, so each lexical category
    states its own head passing and coindices, and also treats an unmarked
    determiner ``NP/N`` as passing its noun's head up.
    """
    return CCGDependencyProjector(MarkedCategoryAlgebra(("NP/N",)))


type _Head = HeadProjection[ProjectedDependency[str], DecoratedCategory[str]]
"""One projected subtree over CCGbank category strings."""


class _Witness:
    """A projector that delegates and records the first step that raises."""

    def __init__(self, projector: _Projector, derivation: Derivation[str]) -> None:
        self._projector = projector
        self._derivation = derivation
        self.failure: str | None = None

    def lexical(self, node: DerivationNode[str], token: Token, /) -> _Head:
        try:
            return self._projector.lexical(node, token)
        except ValueError:
            self.failure = self.failure or "lexical"
            raise

    def step(
        self,
        node: DerivationNode[str],
        rule_name: str,
        children: tuple[_Head, ...],
        selection: HeadSelection,
        /,
    ) -> _Head:
        try:
            return self._projector.step(node, rule_name, children, selection)
        except ValueError:
            if self.failure is None:
                daughters = ", ".join(
                    self._derivation.node(child).rule or "lexical"
                    for child in node.children
                )
                self.failure = f"{rule_name}({daughters})"

            raise


def round_trip(
    derivation: AutoDerivation,
    sentence: PargSentence,
    projector: _Projector,
) -> RoundTripCounts:
    """Project one gold derivation and score it against its PARG sentence.

    Predicted arcs in :data:`UNEVALUATED_DEPENDENCIES` are removed before
    scoring, and every gold arc is scored. A sentence whose projection raises
    a ``ValueError`` predicts no arcs and is counted as unprojected. Its gold
    arcs remain in the recall denominator and are counted as raised by the
    step that raised. Any other exception propagates.

    Parameters
    ----------
    derivation : AutoDerivation
        The gold AUTO derivation, whose header names its sentence.
    sentence : PargSentence
        The released dependencies for the same sentence.
    projector : _Projector
        The CCG dependency projector under test.

    Raises
    ------
    ValueError
        If the derivation and the PARG sentence name different sentences.
    """
    identifier = dict(derivation.header).get("ID")

    if identifier != sentence.identifier:
        msg = (
            f"AUTO derivation {identifier!r} is scored against PARG sentence "
            f"{sentence.identifier!r}"
        )
        raise ValueError(msg)

    prepared = gold_sentence(derivation.tree)
    words = tuple(token.surface for token in prepared.tokens)
    gold = tuple(
        (
            (
                dependency.head_index,
                dependency.target_index,
                dependency.head_category,
                dependency.argument_slot,
            ),
            prepared.covering.get(
                (dependency.head_index, dependency.target_index), "unspanned"
            ),
        )
        for dependency in sentence.dependencies
    )
    witness = _Witness(projector, prepared.derivation)

    try:
        projection = project_dependencies(
            prepared.derivation,
            prepared.tokens,
            CCGBANK_HEAD_SELECTIONS,
            witness,
        )
    except ValueError:
        lost = score_arcs((), gold)
        return RoundTripCounts(
            gold=lost.gold,
            raised=((witness.failure or "projection", lost.gold),),
            unprojected=1,
        )

    predicted = tuple(
        (
            (
                dependency.predicate_position,
                dependency.argument_position,
                prepared.categories[dependency.predicate_position],
                dependency.slot,
            ),
            dependency.rule_id,
        )
        for dependency in projection.dependencies
    )
    return score_arcs(
        (entry for entry in predicted if evaluated(entry[0], words)), gold
    )
