"""Derivation-count reports."""

from typing import TYPE_CHECKING, Final, Literal, NamedTuple

from lofi._checks import ensure_non_negative, ensure_positive
from lofi._native import UnitEdges
from lofi.deduce._engine import parse
from lofi.deduce._policy import ParsePolicy, Reachability, Retain
from lofi.deduce._semiring import Counting
from lofi.judge import NoNormalForm, RuleSet

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lofi.deduce._ports import Chart, Goal, Normalizer, Proposer
    from lofi.deduce._spans import Cell
    from lofi.judge import NormalForm, Rule
    from lofi.lexicon import Clamp, Entry, LexicalSource

__all__ = ["DerivationCountRow", "derivation_count_report"]

_FOREST_ALL: Final[ParsePolicy[Literal[Retain.FOREST]]] = ParsePolicy(
    Retain.FOREST, Reachability.ALL
)


class DerivationCountRow(NamedTuple):
    """Derivation counts for one inventory and sentence.

    Attributes
    ----------
    inventory_name : str
        Inventory name.
    sentence_position : int
        Zero-based position in the supplied sentences.
    tokens : tuple[str, ...]
        Sentence tokens.
    normal_form_count : int
        Derivations admitted by the declared normal form.
    unconstrained_count : int
        Derivations admitted without a normal-form restriction.
    retention_ratio : float | None
        Normal-form count divided by the unconstrained count, or ``None`` when
        both counts are zero.
    """

    inventory_name: str
    sentence_position: int
    tokens: tuple[str, ...]
    normal_form_count: int
    unconstrained_count: int
    retention_ratio: float | None


class _UnitSource[T, W]:
    """A lexical source that assigns unit weights.

    Parameters
    ----------
    source : LexicalSource[T, W]
        Wrapped lexical source.
    """

    __slots__ = ("_source",)

    def __init__(self, source: LexicalSource[T, W]) -> None:
        self._source = source

    @property
    def is_re_callable(self) -> bool:
        """Return whether the wrapped source may be read more than once."""
        return self._source.is_re_callable

    @property
    def clamps(self) -> tuple[Clamp[T], ...]:
        """Return the clamps enforced by the wrapped source."""
        return self._source.clamps

    def axioms(
        self,
        tokens: Sequence[str],
        /,
    ) -> Iterable[tuple[int, Entry[T], int]]:
        """Enumerate lexical axioms with unit weights.

        Parameters
        ----------
        tokens : Sequence[str]
            Surface tokens in position order.

        Returns
        -------
        Iterable[tuple[int, Entry[T], int]]
            Position, entry, and unit weight for each lexical axiom.
        """
        return (
            (position, entry, 1) for position, entry, _ in self._source.axioms(tokens)
        )


class _UnitScorer[T]:
    """Assigns unit weight to lexical and rule edges."""

    __slots__ = ()

    @property
    def lowering(self) -> UnitEdges:
        """Declare that every rule edge has unit weight."""
        return UnitEdges()

    def axiom_weight(self, position: int, entry: Entry[T], /) -> int:
        """Return unit weight for a lexical axiom.

        Parameters
        ----------
        position : int
            Token position.
        entry : Entry[T]
            Lexical entry.
        """
        del position, entry

        return 1

    def edge_weight(
        self,
        rule: Rule[T],
        inputs: tuple[T, ...],
        output: T,
        span: Cell,
        split: int | None,
        /,
    ) -> int:
        """Return unit weight for a rule edge.

        Parameters
        ----------
        rule : Rule[T]
            Applied rule.
        inputs : tuple[T, ...]
            Premise terms.
        output : T
            Rule conclusion.
        span : Cell
            Conclusion span.
        split : int | None
            Binary split point, if present.
        """
        del rule, inputs, output, span, split

        return 1


def _with_normal_form[T](
    inventory: RuleSet[T],
    normal_form: NormalForm[T],
) -> RuleSet[T]:
    """Copy a rule inventory with a selected normal form.

    Parameters
    ----------
    inventory : RuleSet[T]
        Rule inventory to copy.
    normal_form : NormalForm[T]
        Normal form for the copy.
    """
    return RuleSet(
        inventory.name,
        inventory.rules,
        inventory.matching,
        normal_form,
        inventory.restriction_mode,
        introduces=inventory.introduces,
        consumes=inventory.consumes,
        index=inventory.index,
    )


def _goal_count[T](
    chart: Chart[T, int],
    goal: Goal[T],
) -> int:
    """Count derivations satisfying a chart goal.

    Parameters
    ----------
    chart : Chart[T, int]
        Parsed counting chart.
    goal : Goal[T]
        Goal selecting the chart roots.
    """
    groups = goal.root_groups(chart)

    if not groups:
        return 0

    total = 1

    for group in groups:
        alternatives = sum(chart.inside(key) for key in group)
        total *= alternatives

    return total


def derivation_count_report[T, W](
    inventories: Iterable[RuleSet[T]],
    sentences: Iterable[Sequence[str]],
    *,
    source: LexicalSource[T, W],
    goal: Goal[T],
    proposer: Proposer[T] | None = None,
    normalizer: Normalizer[T] | None = None,
    unary_depth: int,
    threads: int,
) -> tuple[DerivationCountRow, ...]:
    """Count derivations with and without each inventory's normal form.

    Parameters
    ----------
    inventories : Iterable[RuleSet[T]]
        Rule inventories in report order.
    sentences : Iterable[Sequence[str]]
        Token sequences in report order.
    source : LexicalSource[T, W]
        Repeatable lexical source shared by all parses.
    goal : Goal[T]
        Parse goal used to select result items.
    proposer : Proposer[T] | None
        Optional producer of conclusions for check-only rules.
    normalizer : Normalizer[T] | None
        Optional rule-output normalizer.
    unary_depth : int
        Maximum unary closure depth.
    threads : int
        How many chart cells of one width each parse may fill concurrently.

    Returns
    -------
    tuple[DerivationCountRow, ...]
        One immutable row per inventory-sentence pair.

    Raises
    ------
    ValueError
        If inventory names repeat, the source is not repeatable, the unary
        depth is negative, or the thread count is not positive.
    """
    held_inventories = tuple(inventories)
    held_sentences = tuple(tuple(sentence) for sentence in sentences)
    names = tuple(inventory.name for inventory in held_inventories)

    if len(set(names)) != len(names):
        msg = "derivation-count inventories have duplicate names"
        raise ValueError(msg)

    if not source.is_re_callable:
        msg = "a derivation-count lexical source must support repeated enumeration"
        raise ValueError(msg)

    checked_depth = ensure_non_negative("unary_depth", unary_depth)
    checked_threads = ensure_positive("threads", threads)
    unit_source = _UnitSource(source)
    scorer = _UnitScorer()
    semiring = Counting()
    rows: list[DerivationCountRow] = []

    for inventory in held_inventories:
        unrestricted = _with_normal_form(inventory, NoNormalForm[T]())

        for sentence_position, tokens in enumerate(held_sentences):
            normal_chart = parse(
                tokens,
                rules=inventory,
                source=unit_source,
                semiring=semiring,
                scorer=scorer,
                goal=goal,
                proposer=proposer,
                normalizer=normalizer,
                unary_depth=checked_depth,
                policy=_FOREST_ALL,
                threads=checked_threads,
            )
            unconstrained_chart = parse(
                tokens,
                rules=unrestricted,
                source=unit_source,
                semiring=semiring,
                scorer=scorer,
                goal=goal,
                proposer=proposer,
                normalizer=normalizer,
                unary_depth=checked_depth,
                policy=_FOREST_ALL,
                threads=checked_threads,
            )
            normal_form_count = _goal_count(normal_chart, goal)
            unconstrained_count = _goal_count(unconstrained_chart, goal)
            retention_ratio = (
                None
                if unconstrained_count == 0
                else normal_form_count / unconstrained_count
            )
            rows.append(
                DerivationCountRow(
                    inventory.name,
                    sentence_position,
                    tokens,
                    normal_form_count,
                    unconstrained_count,
                    retention_ratio,
                )
            )

    return tuple(rows)
