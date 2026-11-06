"""Deduction protocols."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from lofi._native import (
        Boolean,
        Counting,
        Expectation,
        KBest,
        LogInside,
        PythonCarrier,
        SpanDependent,
        SpanFree,
        UnitEdges,
        Viterbi,
    )
    from lofi._types import JsonValue
    from lofi.deduce._spans import Cell, SpanIndex
    from lofi.deduce._values import Derivation, Edge, EdgeProduct, ItemKey
    from lofi.judge import Rule, SchematicSequent
    from lofi.lexicon import Entry
    from lofi.terms import Matching, TermRuleKernel

    type SemiringLowering = (
        Boolean | Counting | Expectation | KBest | LogInside | PythonCarrier | Viterbi
    )
    type ScorerLowering = SpanDependent | SpanFree | UnitEdges

__all__ = [
    "CategoryMeasure",
    "CellPruner",
    "Chart",
    "EdgeScorer",
    "Goal",
    "NamedRoots",
    "NativeTermRule",
    "Normalizer",
    "Proposer",
    "Semiring",
]


@runtime_checkable
class NativeTermRule(Protocol):
    """Term-rule capabilities required by the native chart compiler."""

    @property
    def lowering(self) -> TermRuleKernel:
        """Return the rule's native term kernel."""

    @property
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        """Return premise and conclusion patterns for root-directed parsing."""


@runtime_checkable
class CategoryMeasure[T](Protocol):
    """Measure categories under the formalism that owns their constructors."""

    @property
    def constructors(self) -> tuple[tuple[str, int], ...]:
        """Return functor constructor names and their result slots."""

    def size(self, term: T, /) -> int:
        """Return the category's exact node count."""

    def arity(self, term: T, /) -> int:
        """Return the category's functor spine arity."""


class Chart[T, W](Protocol):
    """Specifies packed forest operations."""

    @property
    def index(self) -> SpanIndex:
        """Return the chart's span index."""

    @property
    def semiring(self) -> Semiring[W]:
        """Return the chart's weight algebra."""

    def add(self, key: ItemKey[T], edge: Edge[T, W], /) -> None:
        """Record one derivation of one item.

        Parameters
        ----------
        key : ItemKey[T]
            Concluded item.
        edge : Edge[T, W]
            Derivation edge with chart premises.
        """

    def items_at(self, span: Cell, /) -> tuple[ItemKey[T], ...]:
        """Return the items covering one span.

        Parameters
        ----------
        span : Cell
            Half-open token span.

        Returns
        -------
        tuple[ItemKey[T], ...]
            Snapshot of items at the span.
        """

    def inside(self, key: ItemKey[T], /) -> W:
        """Return an item's aggregated inside weight.

        Parameters
        ----------
        key : ItemKey[T]
            The item to read.

        Returns
        -------
        W
            The semiring sum of edge contributions, or zero if the item is absent.
        """

    def edges(self, key: ItemKey[T], /) -> tuple[Edge[T, W], ...]:
        """Return every step that concluded an item.

        Parameters
        ----------
        key : ItemKey[T]
            The item to read.

        Returns
        -------
        tuple[Edge[T, W], ...]
            Snapshot of edges in insertion order.
        """

    def term(self, key: ItemKey[T], /) -> T:
        """Return an item's term.

        Parameters
        ----------
        key : ItemKey[T]
            The item to read.

        Returns
        -------
        T
            The conclusion recorded for the item.

        Raises
        ------
        KeyError
            If the item is absent.
        """

    def derivations(self, key: ItemKey[T], /) -> Iterator[Derivation[T, W]]:
        """Enumerate an item's derivations, one tree at a time.

        Parameters
        ----------
        key : ItemKey[T]
            The item to read.

        Returns
        -------
        Iterator[Derivation[T, W]]
            An iterator over the trees concluding that item.
        """


class ProductChart[T, W](Chart[T, W], Protocol):
    """Specifies a chart that stores exact Cartesian edge products."""

    def add_product(
        self,
        key: ItemKey[T],
        product: EdgeProduct[T, W],
        /,
    ) -> None:
        """Record one exact Cartesian product of derivation edges."""

    def products(self, key: ItemKey[T], /) -> tuple[EdgeProduct[T, W], ...]:
        """Return exact Cartesian edge products in insertion order."""


@runtime_checkable
class CellPruner[T, W](Protocol):
    """Selects the items a closed cell keeps."""

    def keep(
        self, span: Cell, items: Sequence[tuple[ItemKey[T], W]], /
    ) -> Sequence[ItemKey[T]]:
        """Return the items the cell keeps.

        Parameters
        ----------
        span : Cell
            The half-open token interval the cell covers.
        items : Sequence[tuple[ItemKey[T], W]]
            Every item of the cell with its inside weight, in arrival order.

        Returns
        -------
        Sequence[ItemKey[T]]
            The items to keep, in arrival order and without repetition. The
            chart drops the rest, along with any step that used one of them.
        """


@runtime_checkable
class Proposer[T](Protocol):
    """Proposes conclusions for check-only rules."""

    def propose(
        self, rule: Rule[T], inputs: tuple[T, ...], span: Cell, /
    ) -> Iterator[T]:
        """Enumerate the conclusions worth trying for one rule at one span.

        Parameters
        ----------
        rule : Rule[T]
            The rule to be checked against each candidate.
        inputs : tuple[T, ...]
            The premises, in order.
        span : Cell
            The half-open token interval the conclusion would cover.

        Returns
        -------
        Iterator[T]
            An iterator over candidate conclusions, best first where the supplier
            ranks them.
        """


@runtime_checkable
class EdgeScorer[T, W](Protocol):
    """Scores lexical axioms and rule applications."""

    @property
    def lowering(self) -> ScorerLowering:
        """Return the declared rule-edge weight dependency."""

    def axiom_weight(self, position: int, entry: Entry[T], /) -> W:
        """Return a lexical axiom's weight.

        Parameters
        ----------
        position : int
            The index of the token the entry was drawn for.
        entry : Entry[T]
            The entry admitted there.

        Returns
        -------
        W
            The weight, in the algebra the sweep accumulates in.
        """

    def edge_weight(
        self,
        rule: Rule[T],
        inputs: tuple[T, ...],
        output: T,
        span: Cell,
        split: int | None,
        /,
    ) -> W:
        """Return a rule application's weight.

        Parameters
        ----------
        rule : Rule[T]
            The rule that fired.
        inputs : tuple[T, ...]
            The premises, in order.
        output : T
            The conclusion drawn.
        span : Cell
            The half-open token interval the conclusion covers.
        split : int | None
            Binary split point, or ``None`` for another arity.

        Returns
        -------
        W
            Local edge weight excluding premise inside weights.
        """


@runtime_checkable
class Goal[T](Protocol):
    """Specifies successful parse items."""

    @property
    def settings(self) -> Mapping[str, JsonValue]:
        """Return the goal's stable run-identity settings."""

    def accepts[W](self, chart: Chart[T, W], key: ItemKey[T], /) -> bool:
        """Decide whether one item is a successful parse on its own.

        Parameters
        ----------
        chart : Chart[T, W]
            The finished chart.
        key : ItemKey[T]
            The item to judge.
        """

    def roots[W](self, chart: Chart[T, W], /) -> Iterator[ItemKey[T]]:
        """Enumerate the items a successful parse ends at.

        Parameters
        ----------
        chart : Chart[T, W]
            The finished chart.

        Returns
        -------
        Iterator[ItemKey[T]]
            An iterator over accepted items, with none where the input has no parse.
        """

    def root_groups[W](
        self,
        chart: Chart[T, W],
        /,
    ) -> tuple[tuple[ItemKey[T], ...], ...]:
        """Return the alternative roots for each required goal component."""

    def root_weight(self, root: ItemKey[T], /) -> float:
        """Return the local log weight of one root alternative."""

    def combination_weight(self, group_count: int, /) -> float:
        """Return the shared log weight of a complete root combination."""

    def admits_edge[W](
        self,
        chart: Chart[T, W],
        parent: ItemKey[T],
        edge: Edge[T, W],
        /,
    ) -> bool:
        """Return whether an edge may contribute under this goal."""


@runtime_checkable
class NamedRoots[T](Protocol):
    """A goal naming the categories a whole-input derivation may conclude."""

    @property
    def categories(self) -> tuple[T, ...]:
        """Return the root categories."""

    @property
    def matching(self) -> Matching[T]:
        """Return the congruence a conclusion is compared with them under."""


@runtime_checkable
class Normalizer[T](Protocol):
    """Normalizes rule outputs before chart insertion."""

    def normalize(self, term: T, /) -> T:
        """Return the normal form of a term.

        Parameters
        ----------
        term : T
            The term produced by a rule.

        Returns
        -------
        T
            The term to record in the chart.
        """


@runtime_checkable
class Semiring[W](Protocol):
    """Specifies a chart-weight algebra."""

    @property
    def name(self) -> str:
        """Return the name the algebra is selected by."""

    @property
    def zero(self) -> W:
        """Return the additive identity, which annihilates under times."""

    @property
    def one(self) -> W:
        """Return the multiplicative identity."""

    @property
    def log_probability(self) -> bool:
        """Report whether weights are log probabilities summed by log-sum-exp.

        Goal-relative outside marginals and pruned goal mass read a weight as
        the logarithm of a probability, so they require this capability.
        """

    @property
    def lowering(self) -> SemiringLowering:
        """Return the carrier used by the native chart kernel."""

    def plus(self, a: W, b: W, /) -> W:
        """Combine the weights of two alternative derivations of one item.

        Parameters
        ----------
        a : W
            One weight.
        b : W
            The other weight.

        Returns
        -------
        W
            The combined weight.
        """

    def times(self, a: W, b: W, /) -> W:
        """Combine the weights of two antecedents of one inference.

        Parameters
        ----------
        a : W
            One weight.
        b : W
            The other weight.

        Returns
        -------
        W
            The combined weight.
        """

    def is_zero(self, a: W, /) -> bool:
        """Return whether a weight is the additive identity.

        Parameters
        ----------
        a : W
            The weight to test.

        Returns
        -------
        bool
            Whether the weight is the additive identity.
        """
