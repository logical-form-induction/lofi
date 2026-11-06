"""A* parsing over supertag beams."""

from math import fsum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Protocol, cast, final

from lofi._checks import ensure_non_negative
from lofi.deduce._engine import parse
from lofi.lexicon import BeamSource, GrowingBeamSource

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.deduce._forest import PackedForest
    from lofi.deduce._policy import ParsePolicy, ParseSettings, Retain
    from lofi.deduce._ports import (
        CellPruner,
        Chart,
        EdgeScorer,
        Goal,
        Normalizer,
        Proposer,
    )
    from lofi.deduce._spans import Cell
    from lofi.deduce._values import Edge, ItemKey
    from lofi.judge import Rule, RuleSet

__all__ = [
    "BINARY_WIDENING",
    "FREE_ROOTS",
    "GOAL_ROOTS",
    "LINEAR_WIDENING",
    "SUPERTAG_BEAM_PROPOSER",
    "SUPERTAG_OUTSIDE",
    "ZERO_HEURISTIC",
    "AStarContext",
    "AStarHeuristic",
    "AStarOptions",
    "AStarProposerFactory",
    "AStarResult",
    "AStarRootMode",
    "WideningStrategy",
    "astar_parse",
    "supertag_outside",
]


class AStarHeuristic(Protocol):
    """Supplies an admissible remaining lexical-score bound."""

    @property
    def name(self) -> str:
        """Return the heuristic's declared name."""

    def estimate[T](self, source: BeamSource[T], /) -> tuple[float, ...]:
        """Return one remaining-score bound per lexical position."""


class AStarRootMode(Protocol):
    """Selects the roots that finish an A* search."""

    @property
    def name(self) -> str:
        """Return the mode's declared name."""

    def goal[T](self, grammar_goal: Goal[T], /) -> Goal[T]:
        """Return the goal that selects roots under this mode."""


class WideningStrategy(Protocol):
    """Chooses which global score thresholds an A* search probes."""

    @property
    def name(self) -> str:
        """Return the strategy's declared name."""

    def search[T](
        self,
        source: BeamSource[T],
        outside: tuple[float, ...],
        probe: _AStarProbe[T],
        /,
    ) -> tuple[_AStarAttempt[T], int]:
        """Return the narrowest successful attempt and the attempt count."""


class AStarProposerFactory(Protocol):
    """Builds a checked-rule proposer from one sentence's supertag beams."""

    @property
    def name(self) -> str:
        """Return the factory's declared name."""

    def build[T](
        self,
        source: BeamSource[T] | GrowingBeamSource[T],
        /,
    ) -> Proposer[T]:
        """Return the checked-rule proposer for one active lexical source."""


class AStarContext[T](NamedTuple):
    """Deduction services used by an A* search.

    Parameters
    ----------
    rules : RuleSet[T]
        The grammar's rules, their matching and their normal form.
    scorer : EdgeScorer[T, float]
        The local weight of each axiom and each rule application.
    goal : Goal[T]
        The grammar's goal.
    unary_depth : int
        How many rounds of unary closure a cell runs.
    threads : int
        How many chart cells of one width each probed frontier's parse may
        fill concurrently.
    settings : ParseSettings[float]
        The declared chart components every probed frontier is parsed with.
        Its policy retains a forest, which the search reads its roots from,
        and its semiring weighs derivations by float log probabilities.
    normalizer : Normalizer[T] | None
        The normal form rule outputs take before insertion, or None.
    """

    rules: RuleSet[T]
    scorer: EdgeScorer[T, float]
    goal: Goal[T]
    unary_depth: int
    threads: int
    settings: ParseSettings[float]
    normalizer: Normalizer[T] | None = None


class AStarOptions(NamedTuple):
    """A* widening, root-selection, heuristic and proposer options."""

    strategy: WideningStrategy
    root_mode: AStarRootMode
    heuristic: AStarHeuristic
    proposer: AStarProposerFactory | None


class AStarResult[T](NamedTuple):
    """A completed supertag-factored search."""

    chart: PackedForest[T, float]
    roots: tuple[ItemKey[T], ...]
    source: GrowingBeamSource[T]
    outside: tuple[float, ...]
    attempts: int
    options: AStarOptions

    @property
    def parsed(self) -> bool:
        """Return whether the search found a spanning root."""
        return bool(self.roots)

    @property
    def widenings(self) -> int:
        """Return the number of candidates added after seeding."""
        return self.source.widenings

    @property
    def root_edges(self) -> tuple[Edge[T, float], ...]:
        """Return every edge concluding a selected spanning root."""
        return tuple(edge for root in self.roots for edge in self.chart.edges(root))


def supertag_outside[T](source: BeamSource[T], /) -> tuple[float, ...]:
    """Return each position's best-other-position log weight.

    Raises
    ------
    ValueError
        If a position is empty or a candidate has a positive log weight.
    """
    best: list[float] = []

    for position in range(source.positions):
        candidates = source.candidates(position)

        if not candidates:
            msg = f"the supertag beam at position {position} is empty"
            raise ValueError(msg)

        if any(candidate.weight > 0.0 for candidate in candidates):
            msg = f"the supertag beam at position {position} has a positive log weight"
            raise ValueError(msg)

        best.append(candidates[0].weight)

    return tuple(
        fsum((*best[:position], *best[position + 1 :])) for position in range(len(best))
    )


@final
class SupertagOutside:
    """Bound one position by the best tags at all other positions."""

    @property
    def name(self) -> str:
        """Return the heuristic's declared name."""
        return "supertag-outside"

    def estimate[T](self, source: BeamSource[T], /) -> tuple[float, ...]:
        """Return the sum of the best other-position log weights."""
        return supertag_outside(source)


@final
class ZeroHeuristic:
    """Use the neutral admissible bound at every lexical position."""

    @property
    def name(self) -> str:
        """Return the heuristic's declared name."""
        return "zero"

    def estimate[T](self, source: BeamSource[T], /) -> tuple[float, ...]:
        """Return one zero bound per lexical position."""
        return (0.0,) * source.positions


class _SupertagBeamProposer[T]:
    """Propose active categories ranked across the sentence."""

    __slots__ = ("_ranked",)

    def __init__(
        self,
        source: BeamSource[T] | GrowingBeamSource[T],
        /,
    ) -> None:
        weights: dict[T, float] = {}

        for position in range(source.positions):
            for entry in source.candidates(position):
                held = weights.get(entry.category)

                if held is None or entry.weight > held:
                    weights[entry.category] = entry.weight

        self._ranked = tuple(sorted(weights, key=lambda candidate: -weights[candidate]))

    def propose(
        self,
        rule: Rule[T],
        inputs: tuple[T, ...],
        span: Cell,
        /,
    ) -> Iterator[T]:
        """Yield active decoded categories from highest score to lowest."""
        del rule, inputs, span
        yield from self._ranked


@final
class SupertagBeamProposerFactory:
    """Build checked-rule candidates from the sentence's supertag beams."""

    @property
    def name(self) -> str:
        """Return the factory's declared name."""
        return "supertag-beam"

    def build[T](
        self,
        source: BeamSource[T] | GrowingBeamSource[T],
        /,
    ) -> Proposer[T]:
        """Return a stable score-ranked proposer over the active categories."""
        return _SupertagBeamProposer(source)


@final
class GoalRoots:
    """Finish only at roots accepted by the grammar's goal."""

    @property
    def name(self) -> str:
        """Return the root mode's declared name."""
        return "goal"

    def goal[T](self, grammar_goal: Goal[T], /) -> Goal[T]:
        """Return the grammar's declared goal."""
        return grammar_goal


_FREE_GOAL_SETTINGS: Final[Mapping[str, JsonValue]] = MappingProxyType({"kind": "free"})


@final
class _FreeGoal:
    """Treat every whole-input chart item as one root alternative."""

    @property
    def settings(self) -> Mapping[str, JsonValue]:
        """Return the free-root identity settings."""
        return _FREE_GOAL_SETTINGS

    def accepts[T, W](
        self,
        chart: Chart[T, W],
        key: ItemKey[T],
        /,
    ) -> bool:
        """Return whether an item spans the whole input."""
        return key.span == (0, chart.index.n)

    def roots[T, W](self, chart: Chart[T, W], /) -> Iterator[ItemKey[T]]:
        """Yield every whole-input item."""
        yield from chart.items_at((0, chart.index.n))

    def root_groups[T, W](
        self,
        chart: Chart[T, W],
        /,
    ) -> tuple[tuple[ItemKey[T], ...], ...]:
        """Return whole-input items as one group of alternatives."""
        roots = tuple(self.roots(chart))
        return () if not roots else (roots,)

    def root_weight[T](self, root: ItemKey[T], /) -> float:
        """Return the unit log weight for a free root."""
        del root
        return 0.0

    def combination_weight(self, group_count: int, /) -> float:
        """Return the unit log weight for the one free-root group."""
        del group_count
        return 0.0

    def admits_edge[T, W](
        self,
        chart: Chart[T, W],
        parent: ItemKey[T],
        edge: Edge[T, W],
        /,
    ) -> bool:
        """Admit every edge in the free-root forest."""
        del chart, parent, edge
        return True


_FREE_GOAL: Final = _FreeGoal()


@final
class FreeRoots:
    """Finish at every item spanning the whole sentence."""

    @property
    def name(self) -> str:
        """Return the root mode's declared name."""
        return "free"

    def goal[T](self, grammar_goal: Goal[T], /) -> Goal[T]:
        """Return a goal over every spanning chart item."""
        del grammar_goal
        return _FREE_GOAL


class _AStarAttempt[T](NamedTuple):
    """One parsed beam frontier."""

    chart: PackedForest[T, float]
    roots: tuple[ItemKey[T], ...]
    source: GrowingBeamSource[T]


class _AStarProbe[T]:
    """Parses beam frontiers under one declared A* setup."""

    __slots__ = ("_context", "_options", "_tokens")

    def __init__(
        self,
        tokens: tuple[str, ...],
        context: AStarContext[T],
        options: AStarOptions,
    ) -> None:
        self._tokens = tokens
        self._context = context
        self._options = options

    def parse(self, source: GrowingBeamSource[T], /) -> _AStarAttempt[T]:
        """Parse one active beam frontier and select its roots."""
        context = self._context
        proposer = (
            None
            if self._options.proposer is None
            else self._options.proposer.build(source)
        )
        settings = context.settings
        chart = parse(
            self._tokens,
            rules=context.rules,
            source=source,
            semiring=settings.semiring,
            scorer=context.scorer,
            goal=context.goal,
            proposer=proposer,
            normalizer=context.normalizer,
            unary_depth=ensure_non_negative("unary_depth", context.unary_depth),
            pruner=cast("CellPruner[T, float] | None", settings.pruner),
            prune_lexical=settings.prune_lexical,
            measure_pruned_mass=settings.measure_pruned_mass,
            strategy=settings.strategy,
            span_index=settings.span_index,
            chart_store=settings.chart_store,
            policy=cast("ParsePolicy[Literal[Retain.FOREST]]", settings.policy),
            threads=context.threads,
        )
        selected_goal = self._options.root_mode.goal(context.goal)
        roots = tuple(selected_goal.roots(chart))

        return _AStarAttempt(chart, roots, source)


def _validate_sentence[T](tokens: tuple[str, ...], source: BeamSource[T]) -> None:
    """Reject a sentence and beam with different position counts."""
    if len(tokens) != source.positions:
        msg = (
            f"the sentence has {len(tokens)} tokens but the supertag source has "
            f"{source.positions} positions"
        )
        raise ValueError(msg)


def _threshold_source[T](
    source: BeamSource[T], outside: tuple[float, ...], threshold: float
) -> GrowingBeamSource[T]:
    """Return the beam prefix admitted by a global A* threshold."""
    frontier = tuple(
        max(
            1,
            sum(
                candidate.weight + outside[position] >= threshold
                for candidate in source.candidates(position)
            ),
        )
        for position in range(source.positions)
    )

    return GrowingBeamSource(source, frontier=frontier)


@final
class LinearWidening:
    """Probe global lexical-score levels from best to worst."""

    @property
    def name(self) -> str:
        """Return the strategy's declared name."""
        return "linear"

    def search[T](
        self,
        source: BeamSource[T],
        outside: tuple[float, ...],
        probe: _AStarProbe[T],
        /,
    ) -> tuple[_AStarAttempt[T], int]:
        """Return the first successful global-score level."""
        current = GrowingBeamSource(source)
        attempts = 0

        while True:
            attempted = probe.parse(current)
            attempts += 1

            if attempted.roots:
                return attempted, attempts

            offers = tuple(
                candidate.weight + outside[position]
                for position in range(source.positions)
                if (candidate := current.next_best(position)) is not None
            )

            if not offers:
                return attempted, attempts

            current = _threshold_source(source, outside, max(offers))


@final
class BinaryWidening:
    """Binary-search the ordered global lexical-score levels."""

    @property
    def name(self) -> str:
        """Return the strategy's declared name."""
        return "binary"

    def search[T](
        self,
        source: BeamSource[T],
        outside: tuple[float, ...],
        probe: _AStarProbe[T],
        /,
    ) -> tuple[_AStarAttempt[T], int]:
        """Return the narrowest successful global-score level."""
        initial = probe.parse(GrowingBeamSource(source))
        attempts = 1

        if initial.roots:
            return initial, attempts

        levels = tuple(
            sorted(
                {
                    candidate.weight + outside[position]
                    for position in range(source.positions)
                    for candidate in source.candidates(position)[1:]
                },
                reverse=True,
            )
        )

        if not levels:
            return initial, attempts

        selected = probe.parse(_threshold_source(source, outside, levels[-1]))
        attempts += 1

        if not selected.roots:
            return selected, attempts

        low = 0
        high = len(levels) - 1

        while low <= high:
            middle = (low + high) // 2
            attempted = (
                selected
                if middle == len(levels) - 1
                else probe.parse(_threshold_source(source, outside, levels[middle]))
            )

            if middle != len(levels) - 1:
                attempts += 1

            if attempted.roots:
                selected = attempted
                high = middle - 1
            else:
                low = middle + 1

        return selected, attempts


LINEAR_WIDENING = LinearWidening()
"""Built-in linear global-threshold widening."""

BINARY_WIDENING = BinaryWidening()
"""Built-in binary global-threshold widening."""

GOAL_ROOTS = GoalRoots()
"""Built-in grammar-goal root selection."""

FREE_ROOTS = FreeRoots()
"""Built-in free spanning-root selection."""

SUPERTAG_OUTSIDE = SupertagOutside()
"""Built-in supertag-outside heuristic."""

ZERO_HEURISTIC = ZeroHeuristic()
"""Built-in zero heuristic."""

SUPERTAG_BEAM_PROPOSER = SupertagBeamProposerFactory()
"""Built-in checked proposer over decoded supertag beams."""


def astar_parse[T](
    sentence: Sequence[str],
    source: BeamSource[T],
    context: AStarContext[T],
    options: AStarOptions,
    /,
) -> AStarResult[T]:
    """Parse a sentence by widening a supertag-factored lexical source."""
    tokens = tuple(sentence)
    _validate_sentence(tokens, source)
    outside = options.heuristic.estimate(source)

    if len(outside) != source.positions:
        msg = (
            f"the {options.heuristic.name!r} heuristic returned {len(outside)} "
            f"bounds for {source.positions} positions"
        )
        raise ValueError(msg)

    attempted, attempts = options.strategy.search(
        source,
        outside,
        _AStarProbe(tokens, context, options),
    )
    return AStarResult(
        attempted.chart,
        attempted.roots,
        attempted.source,
        outside,
        attempts,
        options,
    )
