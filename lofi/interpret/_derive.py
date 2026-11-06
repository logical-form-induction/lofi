"""Derivation extraction and packed forest statistics."""

from heapq import heappop, heappush
from itertools import islice, product
from math import exp, fsum, inf
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, Protocol, final

from lofi._checks import ensure_non_negative, ensure_positive
from lofi.deduce import LEXICAL_TAG_PREFIX, Derivation, KBest
from lofi.interpret._errors import CyclicForestError

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from lofi.deduce import Chart, Edge, ItemKey, Ranked
    from lofi.interpret._ports import Interpretation, Lexicalization, Ordering
    from lofi.judge import Rule
    from lofi.lexicon import Entry

__all__ = [
    "BestOfRanked",
    "ConstituentUtility",
    "LexicalTable",
    "LogProbability",
    "MbrUtility",
    "StepContext",
    "axiom_entries",
    "constituent_gain",
    "constituents",
    "derivation_entropy",
    "is_axiom",
    "k_best_scores",
    "mbr",
    "read",
    "step_context",
    "unfold",
]


class StepContext[Category, Weight = float](NamedTuple):
    """Local inputs for interpreting a rule application.

    Attributes
    ----------
    rule : Rule[Category]
        Applied rule.
    conclusion : Category
        Rule conclusion.
    weight : Weight
        Local rule-application weight.
    order : int
        Stable application order.
    premises : tuple[Category, ...]
        Exact premise categories in physical order.
    """

    rule: Rule[Category]
    conclusion: Category
    weight: Weight
    order: int
    premises: tuple[Category, ...]


@final
class LexicalTable[Category, Meaning = Category]:
    """Rules by name and the lexical entries admitted at each token position.

    Parameters
    ----------
    rules : Mapping[str, Rule[Category]]
        Rules keyed by the names derivation edges use.
    lexicon : Sequence[Sequence[Entry[Category, Meaning]]]
        The entries admitted at each token position, in position order.
    """

    __slots__ = ("_lexicon", "_rules")

    def __init__(
        self,
        rules: Mapping[str, Rule[Category]],
        lexicon: Sequence[Sequence[Entry[Category, Meaning]]],
    ) -> None:
        self._rules = MappingProxyType(dict(rules))
        self._lexicon = tuple(tuple(entries) for entries in lexicon)

    def rule(self, name: str, /) -> Rule[Category]:
        """Return the rule an edge names.

        Raises
        ------
        ValueError
            If the table holds no rule of that name.
        """
        rule = self._rules.get(name)

        if rule is None:
            msg = f"the rule table names no rule {name!r}"
            raise ValueError(msg)

        return rule

    def admitted(
        self, position: int, form: str, category: Category, /
    ) -> tuple[Entry[Category, Meaning], ...]:
        """Return the entries with a form and category at one position.

        Raises
        ------
        ValueError
            If the position lies outside the lexicon or admits no such entry.
        """
        if position >= len(self._lexicon):
            msg = (
                f"axiom {form!r} stands at position {position}, beyond the "
                f"{len(self._lexicon)} lexicon positions"
            )
            raise ValueError(msg)

        admitted = tuple(
            entry
            for entry in self._lexicon[position]
            if entry.form == form and entry.category == category
        )

        if not admitted:
            msg = (
                f"no entry at position {position} has the form {form!r} and the "
                f"category of its axiom"
            )
            raise ValueError(msg)

        return admitted


def step_context[Category, Weight](
    derivation: Derivation[Category, Weight],
    lexicalization: Lexicalization[Category, object],
    /,
) -> StepContext[Category, Weight]:
    """Return the local interpretation inputs of a derivation's root step.

    Parameters
    ----------
    derivation : Derivation[Category, Weight]
        Derivation whose root edge applies a rule.
    lexicalization : Lexicalization[Category, object]
        Where the edge's rule is found by name.
    """
    edge = derivation.edge

    return StepContext(
        lexicalization.rule(edge.rule_name),
        edge.output,
        edge.weight,
        edge.order,
        tuple(child.edge.output for child in derivation.children),
    )


def is_axiom[T, W](derivation: Derivation[T, W], /) -> bool:
    """Return whether a derivation is one lexical axiom."""
    return derivation.edge.rule_name.startswith(LEXICAL_TAG_PREFIX)


def axiom_entries[Category, Meaning](
    derivation: Derivation[Category, object],
    lexicalization: Lexicalization[Category, Meaning],
    /,
) -> tuple[Entry[Category, Meaning], ...]:
    """Return the lexical entries one axiom stands for.

    Raises
    ------
    ValueError
        If the axiom's edge has premises.
    """
    edge = derivation.edge

    if derivation.children:
        msg = f"lexical edge {edge.rule_name!r} has premises"
        raise ValueError(msg)

    return lexicalization.admitted(
        derivation.key.span[0],
        edge.rule_name.removeprefix(LEXICAL_TAG_PREFIX),
        edge.output,
    )


def read[Category, Value, Meaning, Weight](
    derivation: Derivation[Category, Weight],
    interpretation: Interpretation[Category, Value, Meaning, Weight],
    lexicalization: Lexicalization[Category, Meaning],
    /,
) -> tuple[Value | None, ...]:
    """Fold an interpretation over a derivation under each lexical choice.

    A lexical axiom stands for every entry its lexicalization admits with the
    axiom's form and category at its position, so a derivation is read once
    per combination of those choices.

    Parameters
    ----------
    derivation : Derivation[Category, Weight]
        Derivation to interpret.
    interpretation : Interpretation[Category, Value, Meaning, Weight]
        Homomorphism applied to axioms and rule applications.
    lexicalization : Lexicalization[Category, Meaning]
        The rules and lexical entries the derivation's edges name.

    Returns
    -------
    tuple[Value | None, ...]
        One root value per lexical choice, lexical choices varying fastest at
        the rightmost premise; ``None`` where the interpretation rejects a
        node.

    Raises
    ------
    ValueError
        If a lexical edge has premises, an axiom admits no entry, or an edge
        names a rule the lexicalization lacks.
    """
    if is_axiom(derivation):
        return tuple(
            interpretation.axiom(entry)
            for entry in axiom_entries(derivation, lexicalization)
        )

    context = step_context(derivation, lexicalization)
    below = tuple(
        read(child, interpretation, lexicalization) for child in derivation.children
    )
    values: list[Value | None] = []

    for choice in product(*below):
        held = tuple(value for value in choice if value is not None)
        values.append(
            None if len(held) != len(choice) else interpretation.step(context, held)
        )

    return tuple(values)


class LogProbability:
    """Ranks derivations by log probability."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the ordering name."""
        return "log-probability"

    def rank(self, weight: float, /) -> float:
        """Return the step's own log probability.

        Parameters
        ----------
        weight : float
            Step log probability.

        Returns
        -------
        float
            The input weight.
        """
        return weight


class BestOfRanked:
    """Ranks derivations by their best stored score."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the ordering name."""
        return "best-of-ranked"

    def rank(self, weight: Ranked, /) -> float:
        """Return the best entry of a step's score list.

        Parameters
        ----------
        weight : Ranked
            Descending step-score sequence.

        Returns
        -------
        float
            Best score, or negative infinity for an empty sequence.
        """
        return weight[0] if weight else -inf


def unfold[T, W](
    chart: Chart[T, W],
    key: ItemKey[T],
    /,
    *,
    limit: int = 1,
    order: Ordering[W] | None = None,
) -> Iterator[Derivation[T, W]]:
    """Return up to ``limit`` derivations for an item.

    Parameters
    ----------
    chart : Chart[T, W]
        Packed forest.
    key : ItemKey[T]
        Item whose derivations to extract.
    limit : int
        Maximum number of derivations.
    order : Ordering[W] | None
        Best-first ordering, or ``None`` for forest order.

    Returns
    -------
    Iterator[Derivation[T, W]]
        Extracted derivations in the requested order.

    Raises
    ------
    ValueError
        If ``limit`` is negative.
    CyclicForestError
        If best-first extraction encounters a forest cycle.
    """
    ensure_non_negative("limit", limit)

    if limit == 0:
        return iter(())

    if order is None:
        return islice(chart.derivations(key), limit)

    return _BestFirst(chart, order).take(key, limit)


def constituents[T, W](derivation: Derivation[T, W], /) -> frozenset[ItemKey[T]]:
    """Return all items in a derivation, including its root.

    Parameters
    ----------
    derivation : Derivation[T, W]
        Tree to traverse.

    Returns
    -------
    frozenset[ItemKey[T]]
        Distinct items concluded by the tree.
    """
    seen: set[ItemKey[T]] = set()
    stack: list[Derivation[T, W]] = [derivation]

    while stack:
        node = stack.pop()
        seen.add(node.key)
        stack.extend(node.children)

    return frozenset(seen)


def constituent_gain[T, W](
    derivation: Derivation[T, W],
    /,
    *,
    marginals: Mapping[ItemKey[T], float],
) -> float:
    """Return the posterior mass of the items used by a derivation.

    Parameters
    ----------
    derivation : Derivation[T, W]
        Candidate tree.
    marginals : Mapping[ItemKey[T], float]
        Posterior mass keyed by item; missing items contribute zero.

    Returns
    -------
    float
        Sum of constituent posterior masses.
    """
    return fsum(marginals.get(item, 0.0) for item in constituents(derivation))


class MbrUtility(Protocol):
    """Scores one candidate derivation against posterior item marginals."""

    @property
    def name(self) -> str:
        """Return the utility's declared name."""

    def gain[T, W](
        self,
        derivation: Derivation[T, W],
        /,
        *,
        marginals: Mapping[ItemKey[T], float],
    ) -> float:
        """Return the utility of one candidate derivation."""


@final
class ConstituentUtility:
    """Score candidates by their posterior constituent mass."""

    @property
    def name(self) -> str:
        """Return the utility's declared name."""
        return "constituent"

    def gain[T, W](
        self,
        derivation: Derivation[T, W],
        /,
        *,
        marginals: Mapping[ItemKey[T], float],
    ) -> float:
        """Return the candidate's posterior constituent mass."""
        return constituent_gain(derivation, marginals=marginals)


def mbr[T, W](
    chart: Chart[T, W],
    key: ItemKey[T],
    /,
    *,
    marginals: Mapping[ItemKey[T], float],
    utility: MbrUtility,
    candidates: int,
    order: Ordering[W] | None = None,
) -> Derivation[T, W] | None:
    """Return the candidate with the greatest constituent posterior mass.

    Parameters
    ----------
    chart : Chart[T, W]
        Packed forest.
    key : ItemKey[T]
        Item whose candidates to compare.
    marginals : Mapping[ItemKey[T], float]
        Posterior mass keyed by item.
    utility : MbrUtility
        Declared gain function used to compare candidates.
    candidates : int
        Maximum number of candidates.
    order : Ordering[W] | None
        Candidate ordering, or ``None`` for forest order.

    Returns
    -------
    Derivation[T, W] | None
        Highest-gain candidate, or ``None`` when no candidate exists.

    Raises
    ------
    ValueError
        If ``candidates`` is not positive.
    CyclicForestError
        If ordered extraction encounters a forest cycle.
    """
    best: Derivation[T, W] | None = None
    highest = -inf
    limit = ensure_positive("an MBR candidate count", candidates)

    for candidate in unfold(chart, key, limit=limit, order=order):
        gain = utility.gain(candidate, marginals=marginals)

        if gain > highest:
            best = candidate
            highest = gain

    return best


def k_best_scores[T, W](
    chart: Chart[T, W],
    key: ItemKey[T],
    /,
    *,
    order: Ordering[W],
    k: int,
) -> Ranked:
    """Return the ``k`` best derivation scores, best first.

    Parameters
    ----------
    chart : Chart[T, W]
        Packed forest.
    key : ItemKey[T]
        Item whose derivations to score.
    order : Ordering[W]
        Step-score ordering.
    k : int
        Number of scores to retain.

    Returns
    -------
    Ranked
        At most ``k`` scores in descending order.

    Raises
    ------
    ValueError
        If ``k`` is less than one.
    CyclicForestError
        If score folding encounters a forest cycle.
    """
    return _KBestFold(chart, order, KBest(k)).of(key)


def derivation_entropy[T](chart: Chart[T, float], key: ItemKey[T], /) -> float:
    """Return the entropy of an item's derivation distribution.

    Parameters
    ----------
    chart : Chart[T, float]
        Packed forest with log probability weights.
    key : ItemKey[T]
        Item whose derivation entropy to compute.

    Returns
    -------
    float
        Entropy in nats, or zero when no derivation exists.

    Raises
    ------
    CyclicForestError
        If entropy folding encounters a forest cycle.
    """
    return _Entropy(chart).of(key)


class _Ranks[T, W]:
    """Tracks best-first search state for one item.

    Parameters
    ----------
    edges : tuple[Edge[T, W], ...]
        Steps concluding the item.
    """

    __slots__ = ("candidates", "edges", "found", "offered", "scores")

    def __init__(self, edges: tuple[Edge[T, W], ...]) -> None:
        self.edges: tuple[Edge[T, W], ...] = edges
        self.found: list[Derivation[T, W]] = []
        self.scores: list[float] = []
        self.candidates: list[tuple[float, int, tuple[int, ...]]] = []
        self.offered: set[tuple[int, tuple[int, ...]]] = set()


class _BestFirst[T, W]:
    """Enumerates an item's derivations in best-first order.

    Parameters
    ----------
    chart : Chart[T, W]
        Packed forest.
    order : Ordering[W]
        Step-score ordering.
    """

    __slots__ = ("_chart", "_opening", "_order", "_table")

    def __init__(self, chart: Chart[T, W], order: Ordering[W]) -> None:
        self._chart: Chart[T, W] = chart
        self._order: Ordering[W] = order
        self._table: dict[ItemKey[T], _Ranks[T, W]] = {}
        self._opening: set[ItemKey[T]] = set()

    def take(self, key: ItemKey[T], limit: int) -> Iterator[Derivation[T, W]]:
        """Yield an item's best derivations, best first.

        Parameters
        ----------
        key : ItemKey[T]
            Item whose derivations to extract.
        limit : int
            Maximum number of derivations.

        Yields
        ------
        Derivation[T, W]
            Derivations in descending score order.
        """
        for rank in range(limit):
            tree = self._nth(key, rank)

            if tree is None:
                return

            yield tree

    def _nth(self, key: ItemKey[T], rank: int) -> Derivation[T, W] | None:
        """Return the derivation at a rank, expanding the search as needed.

        Parameters
        ----------
        key : ItemKey[T]
            Item to search.
        rank : int
            Zero-based derivation rank.

        Returns
        -------
        Derivation[T, W] | None
            Ranked derivation, or ``None`` if the rank does not exist.
        """
        ranks = self._table.get(key)

        if ranks is None:
            ranks = self._open(key)

        while len(ranks.found) <= rank:
            if not ranks.candidates:
                return None

            score, at, choice = heappop(ranks.candidates)
            edge = ranks.edges[at]
            below = tuple(
                self._settled(child, choice[premise])
                for premise, child in enumerate(edge.children)
            )

            ranks.found.append(Derivation(key, edge, below))
            ranks.scores.append(-score)
            self._grow(ranks, at, choice)

        return ranks.found[rank]

    def _settled(self, key: ItemKey[T], rank: int) -> Derivation[T, W]:
        """Return an already-settled derivation.

        Parameters
        ----------
        key : ItemKey[T]
            Settled item.
        rank : int
            Zero-based settled rank.

        Returns
        -------
        Derivation[T, W]
            Previously materialized derivation.
        """
        return self._table[key].found[rank]

    def _open(self, key: ItemKey[T]) -> _Ranks[T, W]:
        """Seed an item's search with the best candidate from each step.

        Parameters
        ----------
        key : ItemKey[T]
            Item to open.

        Returns
        -------
        _Ranks[T, W]
            Initialized search state.

        Raises
        ------
        CyclicForestError
            If the item is already being opened.
        """
        if key in self._opening:
            raise CyclicForestError(item=repr(key))

        self._opening.add(key)

        try:
            ranks = _Ranks[T, W](self._chart.edges(key))

            for at, edge in enumerate(ranks.edges):
                choice = (0,) * len(edge.children)

                if self._reaches(edge, choice):
                    self._offer(ranks, at, choice)
        finally:
            self._opening.discard(key)

        self._table[key] = ranks

        return ranks

    def _reaches(self, edge: Edge[T, W], choice: tuple[int, ...]) -> bool:
        """Return whether each premise has a derivation at its chosen rank.

        Parameters
        ----------
        edge : Edge[T, W]
            Candidate step.
        choice : tuple[int, ...]
            Chosen rank for each premise.

        Returns
        -------
        bool
            Whether all chosen premise derivations exist.
        """
        return all(
            self._nth(child, choice[premise]) is not None
            for premise, child in enumerate(edge.children)
        )

    def _offer(self, ranks: _Ranks[T, W], at: int, choice: tuple[int, ...]) -> None:
        """Push one candidate onto an item's frontier.

        Parameters
        ----------
        ranks : _Ranks[T, W]
            Item search state.
        at : int
            Index of the candidate step.
        choice : tuple[int, ...]
            Chosen rank for each premise.
        """
        edge = ranks.edges[at]
        score = self._order.rank(edge.weight) + fsum(
            self._table[child].scores[choice[premise]]
            for premise, child in enumerate(edge.children)
        )

        ranks.offered.add((at, choice))
        heappush(ranks.candidates, (-score, at, choice))

    def _grow(self, ranks: _Ranks[T, W], at: int, choice: tuple[int, ...]) -> None:
        """Offer successors of a candidate that was just taken.

        Parameters
        ----------
        ranks : _Ranks[T, W]
            Item search state.
        at : int
            Index of the selected step.
        choice : tuple[int, ...]
            Selected rank for each premise.
        """
        edge = ranks.edges[at]

        for premise in range(len(choice)):
            after = (*choice[:premise], choice[premise] + 1, *choice[premise + 1 :])

            if (at, after) in ranks.offered or not self._reaches(edge, after):
                continue

            self._offer(ranks, at, after)


class _KBestFold[T, W]:
    """Computes k-best scores over a packed forest.

    Parameters
    ----------
    chart : Chart[T, W]
        Packed forest.
    order : Ordering[W]
        Step-score ordering.
    algebra : KBest
        K-best semiring defining the retained score count.
    """

    __slots__ = ("_algebra", "_chart", "_folding", "_order", "_table")

    def __init__(self, chart: Chart[T, W], order: Ordering[W], algebra: KBest) -> None:
        self._chart: Chart[T, W] = chart
        self._order: Ordering[W] = order
        self._algebra = algebra
        self._table: dict[ItemKey[T], Ranked] = {}
        self._folding: set[ItemKey[T]] = set()

    def of(self, key: ItemKey[T]) -> Ranked:
        """Return the best few scores of one item's derivations.

        Parameters
        ----------
        key : ItemKey[T]
            Item to score.

        Returns
        -------
        Ranked
            Derivation scores in descending order.

        Raises
        ------
        CyclicForestError
            If the item is already being folded.
        """
        settled = self._table.get(key)

        if settled is not None:
            return settled

        if key in self._folding:
            raise CyclicForestError(item=repr(key))

        self._folding.add(key)

        try:
            total = self._algebra.zero

            for edge in self._chart.edges(key):
                weight: Ranked = (self._order.rank(edge.weight),)

                for child in edge.children:
                    weight = self._algebra.times(weight, self.of(child))

                total = self._algebra.plus(total, weight)
        finally:
            self._folding.discard(key)

        self._table[key] = total

        return total


class _Entropy[T]:
    """Computes derivation entropy over a packed forest.

    Parameters
    ----------
    chart : Chart[T, float]
        Packed forest with log probability weights.
    """

    __slots__ = ("_chart", "_folding", "_table")

    def __init__(self, chart: Chart[T, float]) -> None:
        self._chart: Chart[T, float] = chart
        self._table: dict[ItemKey[T], float] = {}
        self._folding: set[ItemKey[T]] = set()

    def of(self, key: ItemKey[T]) -> float:
        """Return the entropy in nats of one item's derivations.

        Parameters
        ----------
        key : ItemKey[T]
            Item to evaluate.

        Returns
        -------
        float
            Entropy in nats, or zero when no derivation exists.

        Raises
        ------
        CyclicForestError
            If the item is already being folded.
        """
        settled = self._table.get(key)

        if settled is not None:
            return settled

        if key in self._folding:
            raise CyclicForestError(item=repr(key))

        self._folding.add(key)

        try:
            entropy = self._spend(key)
        finally:
            self._folding.discard(key)

        self._table[key] = entropy

        return entropy

    def _spend(self, key: ItemKey[T]) -> float:
        """Return one item's entropy contribution.

        Parameters
        ----------
        key : ItemKey[T]
            Item to evaluate.

        Returns
        -------
        float
            Local and descendant entropy in nats.
        """
        total = self._chart.inside(key)

        if total == -inf:
            return 0.0

        entropy = 0.0

        for edge in self._chart.edges(key):
            weight = edge.weight

            for child in edge.children:
                weight += self._chart.inside(child)

            if weight == -inf:
                continue

            share = weight - total
            below = fsum(self.of(child) for child in edge.children)
            entropy += exp(share) * (below - share)

        return entropy
