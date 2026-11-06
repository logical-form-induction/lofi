"""Bisk--Hockenmaier CCG induction over packed derivation forests.

The induction core grows a POS-conditioned lexicon only from categories in
complete parses, recomputes bounded top-k derivations at each EM iteration, and
obtains the smoothing distribution directly from packed inside/outside edge
marginals.  Thus the full forest term is exact without materializing the
forest's complete derivation set.
"""

from math import exp, fsum, inf, log
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast, final, override

from lofi._checks import (
    ensure_finite,
    ensure_name,
    ensure_positive,
)
from lofi._frozen import FrozenValue
from lofi._types import HeadSelection, ItemId, JsonValue
from lofi.data import AttachmentTree, DerivationNode, Token, TokenSpan, attachment_tree
from lofi.data import Derivation as DataDerivation
from lofi.data.deps import (
    BiskHockenmaierDependencyProjector,
    project_bisk_hockenmaier_dependencies,
)
from lofi.deduce import (
    LEXICAL_TAG_PREFIX,
    Edge,
    LogInside,
    PackedForest,
    outside_from_edges,
)
from lofi.eval._identity import config_digest
from lofi.eval._supertagging import (
    CategoryProbability,
    SupertagPrediction,
    TokenSupertags,
)
from lofi.interpret import LogProbability, unfold
from lofi.terms import Term

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.deduce import Chart, Goal, ItemKey
    from lofi.deduce import Derivation as ForestDerivation
    from lofi.terms import Functors, TermStore, TermSyntax

__all__ = [
    "BiskCandidate",
    "BiskEvent",
    "BiskHockenmaierInducer",
    "BiskHockenmaierModel",
    "BiskHockenmaierSettings",
    "BiskItem",
    "BiskPrediction",
    "bisk_candidates",
]

type BiskEstimator = Literal["smoothed-k-best-em"]
type BiskEventFamily = Literal["expansion", "word", "head", "sister"]
type _EventContext = tuple[BiskEventFamily, tuple[str, ...]]


def _data_derivation(
    source: ForestDerivation[Term, float],
) -> DataDerivation[Term]:
    """Translate one unpacked forest tree to the corpus derivation interface."""
    nodes: dict[int, DerivationNode[Term]] = {}

    def visit(node: ForestDerivation[Term, float]) -> int:
        children = tuple(visit(child) for child in node.children)
        identifier = len(nodes)
        start, end = node.key.span
        nodes[identifier] = DerivationNode(
            TokenSpan(start, end),
            node.key.reading,
            rule=None if not children else node.edge.rule_name,
            children=children,
        )
        return identifier

    root = visit(source)
    return DataDerivation(root, nodes)


def _lexical_categories(
    derivation: ForestDerivation[Term, float], token_count: int
) -> tuple[Term, ...]:
    """Return one exact lexical category per token in position order."""
    categories: dict[int, Term] = {}
    pending = [derivation]

    while pending:
        node = pending.pop()

        if node.edge.rule_name.startswith(LEXICAL_TAG_PREFIX):
            start, end = node.key.span

            if end != start + 1 or start in categories:
                msg = "a candidate derivation does not tile tokens with lexical axioms"
                raise ValueError(msg)

            categories[start] = node.key.reading

        pending.extend(node.children)

    if set(categories) != set(range(token_count)):
        msg = "a candidate derivation has no exact lexical category for every token"
        raise ValueError(msg)

    return tuple(categories[position] for position in range(token_count))


def _events(
    derivation: ForestDerivation[Term, float],
    tags: tuple[str, ...],
    syntax: TermSyntax,
    store: TermStore,
    head_selections: Mapping[str, HeadSelection],
) -> tuple[BiskEvent, ...]:
    """Read the published head-outward event factorization from one tree."""
    events: list[BiskEvent] = []

    def rendered(term: Term) -> str:
        return syntax.render(store, term)

    def visit(node: ForestDerivation[Term, float]) -> None:
        parent = rendered(node.key.reading)
        children = node.children

        if not children:
            start, end = node.key.span

            if end != start + 1:
                msg = f"a lexical event covers the non-token span {node.key.span!r}"
                raise ValueError(msg)

            events.append(BiskEvent("expansion", (parent,), "lexical"))
            events.append(BiskEvent("word", (parent, "lexical"), tags[start]))
            return

        if len(children) == 1:
            head = rendered(children[0].key.reading)
            events.append(BiskEvent("expansion", (parent,), "unary"))
            events.append(BiskEvent("head", (parent, "unary"), head))
        elif len(children) == 2:
            try:
                selection = head_selections[node.edge.rule_name]
            except KeyError:
                msg = f"no head selection is declared for rule {node.edge.rule_name!r}"
                raise KeyError(msg) from None

            slot = selection.slot

            if slot not in (0, 1):
                msg = f"binary rule {node.edge.rule_name!r} selects head slot {slot!r}"
                raise ValueError(msg)

            direction = "left" if slot == 0 else "right"
            head = rendered(children[slot].key.reading)
            sister = rendered(children[1 - slot].key.reading)
            events.append(BiskEvent("expansion", (parent,), direction))
            events.append(BiskEvent("head", (parent, direction), head))
            events.append(BiskEvent("sister", (parent, head, direction), sister))
        else:
            msg = f"the published inducer does not admit arity {len(children)}"
            raise ValueError(msg)

        for child in children:
            visit(child)

    visit(derivation)
    return tuple(events)


@final
class BiskEvent(FrozenValue):
    """One conditional event in the head-outward generative model."""

    _noun = "Bisk--Hockenmaier event"
    __slots__ = ("_conditioning", "_family", "_outcome")

    def __init__(
        self,
        family: BiskEventFamily,
        conditioning: Sequence[str],
        outcome: str,
    ) -> None:
        if family not in ("expansion", "word", "head", "sister"):
            msg = f"an event has an undeclared family: {family!r}"
            raise ValueError(msg)

        held = tuple(
            ensure_name(f"event conditioning field {position}", value)
            for position, value in enumerate(conditioning)
        )

        if not held:
            msg = "an event has no conditioning context"
            raise ValueError(msg)

        object.__setattr__(self, "_family", family)
        object.__setattr__(self, "_conditioning", held)
        object.__setattr__(self, "_outcome", ensure_name("an event outcome", outcome))

    @property
    def family(self) -> BiskEventFamily:
        """Return the factor family."""
        return self._family

    @property
    def conditioning(self) -> tuple[str, ...]:
        """Return the variables the event is conditioned on."""
        return self._conditioning

    @property
    def outcome(self) -> str:
        """Return the predicted value."""
        return self._outcome

    @property
    def context(self) -> _EventContext:
        """Return the normalization context."""
        return self._family, self._conditioning

    def _fields(self) -> tuple[BiskEventFamily, tuple[str, ...], str]:
        return self._family, self._conditioning, self._outcome

    @override
    def __reduce__(self) -> tuple[type[BiskEvent], tuple[object, ...]]:
        return BiskEvent, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not BiskEvent:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        return hash((BiskEvent, self._fields()))

    @override
    def __repr__(self) -> str:
        return f"BiskEvent{self._fields()!r}"


@final
class BiskCandidate(FrozenValue):
    """One complete derivation admitted for an induction sentence."""

    _noun = "Bisk--Hockenmaier candidate"
    __slots__ = ("_categories", "_dependencies", "_events")

    def __init__(
        self,
        categories: Sequence[Term],
        dependencies: AttachmentTree,
        events: Sequence[BiskEvent],
    ) -> None:
        held_categories = tuple(categories)
        held_events = tuple(events)

        if not held_categories:
            msg = "a candidate has no lexical categories"
            raise ValueError(msg)

        for position, category in enumerate(held_categories):
            if not isinstance(category, Term):
                msg = f"candidate category {position} is not a term: {category!r}"
                raise TypeError(msg)

        if not isinstance(dependencies, AttachmentTree):
            msg = f"candidate dependencies are not a tree: {dependencies!r}"
            raise TypeError(msg)

        if dependencies.length != len(held_categories):
            msg = (
                f"a candidate has {len(held_categories)} categories and a "
                f"{dependencies.length}-token dependency tree"
            )
            raise ValueError(msg)

        for position, event in enumerate(held_events):
            if not isinstance(event, BiskEvent):
                msg = f"candidate event {position} is not one: {event!r}"
                raise TypeError(msg)

        if not held_events:
            msg = "a candidate has no generative events"
            raise ValueError(msg)

        object.__setattr__(self, "_categories", held_categories)
        object.__setattr__(self, "_dependencies", dependencies)
        object.__setattr__(self, "_events", held_events)

    @property
    def categories(self) -> tuple[Term, ...]:
        """Return lexical categories in token order."""
        return self._categories

    @property
    def dependencies(self) -> AttachmentTree:
        """Return the projected dependency tree."""
        return self._dependencies

    @property
    def events(self) -> tuple[BiskEvent, ...]:
        """Return the derivation's generative events."""
        return self._events

    def _fields(self) -> tuple[tuple[Term, ...], AttachmentTree, tuple[BiskEvent, ...]]:
        return self._categories, self._dependencies, self._events

    @override
    def __reduce__(self) -> tuple[type[BiskCandidate], tuple[object, ...]]:
        return BiskCandidate, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not BiskCandidate:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        return hash((BiskCandidate, self._fields()))

    @override
    def __repr__(self) -> str:
        return (
            f"BiskCandidate(categories={len(self._categories)}, "
            f"events={len(self._events)})"
        )


@final
class BiskItem(FrozenValue):
    """One POS-tagged sentence and its packed derivation forest.

    ``root_feeders`` names the only categories from which a unary edge may
    derive a complete root of the item, or is ``None`` when every unary edge
    the grammar licenses into a root is admitted.
    """

    _noun = "Bisk--Hockenmaier item"
    __slots__ = (
        "_chart",
        "_forms",
        "_goal",
        "_head_selections",
        "_item_id",
        "_projector",
        "_raising_rules",
        "_raising_targets",
        "_root_feeders",
        "_store",
        "_syntax",
        "_tags",
        "_tokens",
    )

    def __init__(
        self,
        item_id: ItemId,
        forms: Sequence[str],
        tags: Sequence[str],
        tokens: Sequence[Token],
        chart: Chart[Term, float],
        goal: Goal[Term],
        syntax: TermSyntax,
        store: TermStore,
        head_selections: Mapping[str, HeadSelection],
        raising_rules: frozenset[str],
        raising_targets: Mapping[tuple[str, Term], Term],
        projector: BiskHockenmaierDependencyProjector[Term],
        *,
        root_feeders: frozenset[Term] | None,
    ) -> None:
        held_forms = tuple(
            ensure_name(f"form {position}", form) for position, form in enumerate(forms)
        )
        held_tags = tuple(
            ensure_name(f"POS tag {position}", tag) for position, tag in enumerate(tags)
        )
        held_tokens = tuple(tokens)

        if not held_forms:
            msg = "an induction item has no tokens"
            raise ValueError(msg)

        if len(held_forms) != len(held_tags):
            msg = f"an item has {len(held_forms)} forms and {len(held_tags)} POS tags"
            raise ValueError(msg)

        if len(held_tokens) != len(held_forms) or chart.index.n != len(held_forms):
            msg = "an item's forms, tokens, and chart do not have one common length"
            raise ValueError(msg)
        if tuple(token.surface for token in held_tokens) != held_forms:
            msg = "an item's token surfaces differ from its forms"
            raise ValueError(msg)
        groups = goal.root_groups(chart)
        if len(groups) > 1:
            msg = "Bisk--Hockenmaier induction requires one complete-root group"
            raise ValueError(msg)
        held_rules = frozenset(
            ensure_name("a type-raising rule", rule_name) for rule_name in raising_rules
        )
        held_raising: dict[tuple[str, Term], Term] = {}
        for (rule_name, output), target in raising_targets.items():
            held_rule = ensure_name("a type-raising rule", rule_name)
            if held_rule not in held_rules:
                msg = f"type-raising relation names undeclared rule {held_rule!r}"
                raise ValueError(msg)
            if not isinstance(output, Term) or not isinstance(target, Term):
                msg = "a type-raising target relation must contain terms"
                raise TypeError(msg)
            held_raising[(held_rule, output)] = target
        for parent in _chart_items(chart):
            for edge in chart.edges(parent):
                if (
                    edge.rule_name in held_rules
                    and (edge.rule_name, parent.reading) not in held_raising
                ):
                    msg = (
                        f"type-raising edge {edge.rule_name!r} has no declared "
                        f"target for {parent.reading!r}"
                    )
                    raise ValueError(msg)

        object.__setattr__(
            self, "_item_id", ItemId(ensure_name("an induction item id", item_id))
        )
        object.__setattr__(self, "_forms", held_forms)
        object.__setattr__(self, "_tags", held_tags)
        object.__setattr__(self, "_tokens", held_tokens)
        object.__setattr__(self, "_chart", chart)
        object.__setattr__(self, "_goal", goal)
        object.__setattr__(self, "_syntax", syntax)
        object.__setattr__(self, "_store", store)
        object.__setattr__(
            self, "_head_selections", MappingProxyType(dict(head_selections))
        )
        object.__setattr__(self, "_raising_rules", held_rules)
        object.__setattr__(self, "_raising_targets", MappingProxyType(held_raising))
        object.__setattr__(self, "_projector", projector)
        object.__setattr__(
            self,
            "_root_feeders",
            None if root_feeders is None else frozenset(root_feeders),
        )

    @property
    def item_id(self) -> ItemId:
        return self._item_id

    @property
    def forms(self) -> tuple[str, ...]:
        return self._forms

    @property
    def tags(self) -> tuple[str, ...]:
        return self._tags

    @property
    def tokens(self) -> tuple[Token, ...]:
        return self._tokens

    @property
    def chart(self) -> Chart[Term, float]:
        return self._chart

    @property
    def goal(self) -> Goal[Term]:
        return self._goal

    @property
    def syntax(self) -> TermSyntax:
        return self._syntax

    @property
    def store(self) -> TermStore:
        return self._store

    @property
    def head_selections(self) -> Mapping[str, HeadSelection]:
        return self._head_selections

    @property
    def projector(self) -> BiskHockenmaierDependencyProjector[Term]:
        return self._projector

    @property
    def raising_targets(self) -> Mapping[tuple[str, Term], Term]:
        """Return each type raising output's required attested result."""
        return self._raising_targets

    @property
    def root_feeders(self) -> frozenset[Term] | None:
        """Return the only categories a unary edge may derive a root from."""
        return self._root_feeders


@final
class BiskHockenmaierSettings(FrozenValue):
    """Declared controls for lexicon induction and parameter estimation."""

    _noun = "Bisk--Hockenmaier settings"
    __slots__ = (
        "_category_induction_steps",
        "_conventions",
        "_coordination_tags",
        "_derived_constituent_pass",
        "_gerund_tag",
        "_iterations",
        "_k",
        "_max_arity",
        "_nominal_tags",
        "_other_tags",
        "_probability_floor",
        "_second_order_modifiers",
        "_tolerance",
        "_verbal_tags",
    )

    def __init__(
        self,
        *,
        nominal_tags: frozenset[str],
        verbal_tags: frozenset[str],
        coordination_tags: frozenset[str],
        other_tags: frozenset[str],
        gerund_tag: str,
        max_arity: int,
        category_induction_steps: int,
        derived_constituent_pass: bool,
        second_order_modifiers: bool,
        iterations: int,
        k: int,
        probability_floor: float,
        tolerance: float,
        conventions: Mapping[str, JsonValue],
    ) -> None:
        classes = (
            ("nominal", frozenset(nominal_tags)),
            ("verbal", frozenset(verbal_tags)),
            ("coordination", frozenset(coordination_tags)),
            ("other", frozenset(other_tags)),
        )

        for label, tags in classes:
            for tag in tags:
                ensure_name(f"a {label} POS tag", tag)

        repeated = set().union(
            *(
                left[1] & right[1]
                for at, left in enumerate(classes)
                for right in classes[at + 1 :]
            )
        )

        if repeated:
            msg = f"POS classes overlap: {sorted(repeated)!r}"
            raise ValueError(msg)

        held_tolerance = ensure_finite("EM convergence tolerance", tolerance)
        held_floor = ensure_finite("EM probability floor", probability_floor)

        if held_tolerance < 0.0:
            msg = f"EM convergence tolerance is negative: {tolerance!r}"
            raise ValueError(msg)
        if not 0.0 < held_floor <= 1.0:
            msg = f"EM probability floor is outside (0, 1]: {probability_floor!r}"
            raise ValueError(msg)
        held_gerund = ensure_name("the gerund POS tag", gerund_tag)
        if held_gerund not in classes[1][1]:
            msg = f"the gerund tag {held_gerund!r} is not verbal"
            raise ValueError(msg)

        object.__setattr__(self, "_nominal_tags", classes[0][1])
        object.__setattr__(self, "_verbal_tags", classes[1][1])
        object.__setattr__(self, "_coordination_tags", classes[2][1])
        object.__setattr__(self, "_other_tags", classes[3][1])
        object.__setattr__(self, "_gerund_tag", held_gerund)
        object.__setattr__(
            self, "_max_arity", ensure_positive("maximum arity", max_arity)
        )
        object.__setattr__(
            self, "_iterations", ensure_positive("EM iterations", iterations)
        )
        object.__setattr__(
            self,
            "_category_induction_steps",
            ensure_positive("category induction steps", category_induction_steps),
        )
        object.__setattr__(self, "_derived_constituent_pass", derived_constituent_pass)
        object.__setattr__(self, "_second_order_modifiers", second_order_modifiers)
        object.__setattr__(self, "_k", ensure_positive("k-best width", k))
        object.__setattr__(self, "_probability_floor", held_floor)
        object.__setattr__(self, "_tolerance", held_tolerance)
        object.__setattr__(self, "_conventions", MappingProxyType(dict(conventions)))

    @property
    def nominal_tags(self) -> frozenset[str]:
        return self._nominal_tags

    @property
    def verbal_tags(self) -> frozenset[str]:
        return self._verbal_tags

    @property
    def coordination_tags(self) -> frozenset[str]:
        return self._coordination_tags

    @property
    def other_tags(self) -> frozenset[str]:
        return self._other_tags

    @property
    def gerund_tag(self) -> str:
        return self._gerund_tag

    @property
    def max_arity(self) -> int:
        return self._max_arity

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def category_induction_steps(self) -> int:
        return self._category_induction_steps

    @property
    def derived_constituent_pass(self) -> bool:
        return self._derived_constituent_pass

    @property
    def second_order_modifiers(self) -> bool:
        return self._second_order_modifiers

    @property
    def k(self) -> int:
        return self._k

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @property
    def probability_floor(self) -> float:
        return self._probability_floor

    @property
    def conventions(self) -> Mapping[str, JsonValue]:
        return self._conventions

    def to_spec(self) -> dict[str, JsonValue]:
        """Return canonical JSON-shaped controls."""
        return {
            "nominal_tags": sorted(self._nominal_tags),
            "verbal_tags": sorted(self._verbal_tags),
            "coordination_tags": sorted(self._coordination_tags),
            "other_tags": sorted(self._other_tags),
            "gerund_tag": self._gerund_tag,
            "max_arity": self._max_arity,
            "category_induction_steps": self._category_induction_steps,
            "derived_constituent_pass": self._derived_constituent_pass,
            "second_order_modifiers": self._second_order_modifiers,
            "iterations": self._iterations,
            "estimator": "smoothed-k-best-em",
            "k": self._k,
            "probability_floor": self._probability_floor,
            "tolerance": self._tolerance,
            "conventions": dict(self._conventions),
        }


def _term_order(term: Term) -> tuple[str, int | str | None, tuple[object, ...]]:
    key = term.key()
    return key[0], key[1], cast("tuple[object, ...]", key[2])


def _chart_items(chart: Chart[Term, float], /) -> tuple[ItemKey[Term], ...]:
    """Return every packed item in deterministic narrow-to-wide order."""
    return tuple(
        key for span in chart.index.spans_by_length() for key in chart.items_at(span)
    )


def _edge_events(
    item: BiskItem,
    parent: ItemKey[Term],
    edge: Edge[Term, float],
    /,
) -> tuple[BiskEvent, ...]:
    """Return the local generative events owned by one packed edge."""
    rendered = item.syntax.render(item.store, parent.reading)
    events: list[BiskEvent] = []

    children = edge.children
    if not children:
        start, end = parent.span
        if end != start + 1 or not edge.rule_name.startswith(LEXICAL_TAG_PREFIX):
            msg = f"a lexical edge does not cover one token: {parent.span!r}"
            raise ValueError(msg)
        events.append(BiskEvent("expansion", (rendered,), "lexical"))
        events.append(BiskEvent("word", (rendered, "lexical"), item.tags[start]))
        return tuple(events)

    if len(children) == 1:
        head = item.syntax.render(item.store, children[0].reading)
        events.append(BiskEvent("expansion", (rendered,), "unary"))
        events.append(BiskEvent("head", (rendered, "unary"), head))
        return tuple(events)
    if len(children) != 2:
        msg = f"the published inducer does not admit arity {len(children)}"
        raise ValueError(msg)

    try:
        slot = item.head_selections[edge.rule_name].slot
    except KeyError:
        msg = f"no head selection is declared for rule {edge.rule_name!r}"
        raise KeyError(msg) from None
    if slot not in (0, 1):
        msg = f"binary rule {edge.rule_name!r} selects head slot {slot!r}"
        raise ValueError(msg)
    direction = "left" if slot == 0 else "right"
    head = item.syntax.render(item.store, children[slot].reading)
    sister = item.syntax.render(item.store, children[1 - slot].reading)
    events.append(BiskEvent("expansion", (rendered,), direction))
    events.append(BiskEvent("head", (rendered, direction), head))
    events.append(BiskEvent("sister", (rendered, head, direction), sister))
    return tuple(events)


def _copy_forest(
    item: BiskItem,
    admitted: Mapping[str, tuple[Term, ...]],
    probabilities: Mapping[BiskEvent, float] | None,
    /,
) -> PackedForest[Term, float]:
    """Return the item's packed forest filtered and weighted by one model."""
    source = item.chart
    target = PackedForest(source.index, LogInside())
    roots = frozenset(root for group in item.goal.root_groups(source) for root in group)
    attested = {category for categories in admitted.values() for category in categories}
    feeders = item.root_feeders
    rows: list[tuple[ItemKey[Term], Edge[Term, float]]] = []
    for parent in _chart_items(source):
        for edge in source.edges(parent):
            if not edge.children and edge.rule_name.startswith(LEXICAL_TAG_PREFIX):
                start, end = parent.span
                if end != start + 1:
                    msg = f"a lexical edge does not cover one token: {parent.span!r}"
                    raise ValueError(msg)
                if parent.reading not in admitted.get(item.tags[start], ()):
                    continue
            raising_target = item.raising_targets.get((edge.rule_name, parent.reading))
            if raising_target is not None and raising_target not in attested:
                continue
            if (
                feeders is not None
                and parent in roots
                and len(edge.children) == 1
                and edge.children[0].reading not in feeders
            ):
                continue
            if probabilities is not None and any(
                event not in probabilities for event in _edge_events(item, parent, edge)
            ):
                continue
            rows.append((parent, edge))

    live: set[ItemKey[Term]] = set()
    retained_edges: set[int] = set()
    changed = True
    while changed:
        changed = False
        for position, (parent, edge) in enumerate(rows):
            if position in retained_edges or any(
                child not in live for child in edge.children
            ):
                continue
            retained_edges.add(position)
            if parent not in live:
                live.add(parent)
            changed = True
    reachable = set(roots & live)
    changed = True
    while changed:
        changed = False
        for position, (parent, edge) in enumerate(rows):
            if position not in retained_edges or parent not in reachable:
                continue
            for child in edge.children:
                if child not in reachable:
                    reachable.add(child)
                    changed = True
    for position, (parent, edge) in enumerate(rows):
        if position not in retained_edges or parent not in reachable:
            continue
        events = _edge_events(item, parent, edge)
        weight = (
            0.0
            if probabilities is None
            else fsum(log(probabilities[event]) for event in events)
        )
        target.add(
            parent,
            Edge(
                edge.rule_name,
                edge.children,
                edge.output,
                weight,
                edge.split,
                edge.order,
            ),
        )

    return target


def _tree_weight(derivation: ForestDerivation[Term, float], /) -> float:
    """Return one derivation's summed local log weight."""
    return derivation.edge.weight + fsum(
        _tree_weight(child) for child in derivation.children
    )


def bisk_candidates(
    item: BiskItem,
    admitted: Mapping[str, tuple[Term, ...]],
    probabilities: Mapping[BiskEvent, float],
    /,
    *,
    limit: int,
) -> tuple[BiskCandidate, ...]:
    """Materialize only the reranked top-k derivations of one packed forest."""
    width = ensure_positive("a k-best width", limit)
    chart = _copy_forest(item, admitted, probabilities)
    groups = item.goal.root_groups(chart)
    if not groups:
        return ()
    if len(groups) != 1:
        msg = "Bisk--Hockenmaier induction requires one complete-root group"
        raise ValueError(msg)
    ranked = sorted(
        (
            derivation
            for root in groups[0]
            for derivation in unfold(
                chart,
                root,
                limit=width,
                order=LogProbability(),
            )
        ),
        key=_tree_weight,
        reverse=True,
    )[:width]
    candidates: list[BiskCandidate] = []
    for derivation in ranked:
        data = _data_derivation(derivation)
        projection = project_bisk_hockenmaier_dependencies(
            data,
            item.tokens,
            item.head_selections,
            item.projector,
        )
        candidates.append(
            BiskCandidate(
                _lexical_categories(derivation, chart.index.n),
                attachment_tree(projection, item.tokens),
                _events(
                    derivation,
                    item.tags,
                    item.syntax,
                    item.store,
                    item.head_selections,
                ),
            )
        )
    return tuple(candidates)


def _normalized(
    counts: Mapping[BiskEvent, float],
    support: Mapping[_EventContext, tuple[BiskEvent, ...]],
    floor: float,
) -> dict[BiskEvent, float]:
    probabilities: dict[BiskEvent, float] = {}

    for events in support.values():
        if len(events) * floor > 1.0:
            msg = (
                f"probability floor {floor!r} is infeasible for {len(events)} outcomes"
            )
            raise ValueError(msg)
        total = fsum(counts.get(event, 0.0) for event in events)
        raw = (
            {event: 1.0 / len(events) for event in events}
            if total == 0.0
            else {event: counts.get(event, 0.0) / total for event in events}
        )
        if all(probability >= floor for probability in raw.values()):
            probabilities.update(raw)
            continue

        remaining = list(events)
        remaining_mass = 1.0
        projected: dict[BiskEvent, float] = {}
        while remaining:
            weight = fsum(raw[event] for event in remaining)
            proposed = {
                event: (
                    remaining_mass / len(remaining)
                    if weight == 0.0
                    else remaining_mass * raw[event] / weight
                )
                for event in remaining
            }
            below = tuple(event for event in remaining if proposed[event] < floor)
            if not below:
                projected.update(proposed)
                break
            for event in below:
                projected[event] = floor
            remaining_mass -= floor * len(below)
            rejected = frozenset(below)
            remaining = [event for event in remaining if event not in rejected]

        anchor = max(events, key=lambda event: projected[event])
        projected[anchor] += 1.0 - fsum(projected.values())
        probabilities.update(projected)

    return probabilities


def _posterior(
    candidates: Sequence[BiskCandidate], probabilities: Mapping[BiskEvent, float]
) -> tuple[float, ...]:
    if not candidates:
        return ()

    scores = tuple(
        fsum(log(probabilities[event]) for event in candidate.events)
        for candidate in candidates
    )
    peak = max(scores)
    masses = tuple(exp(score - peak) for score in scores)
    total = fsum(masses)
    return tuple(mass / total for mass in masses)


def _counts(
    items: Sequence[BiskItem],
    inventory: Mapping[str, tuple[Term, ...]],
    probabilities: Mapping[BiskEvent, float],
    *,
    k: int,
) -> dict[BiskEvent, float]:
    full: dict[BiskEvent, float] = {}
    narrow: dict[BiskEvent, float] = {}

    for item in items:
        candidates = bisk_candidates(item, inventory, probabilities, limit=k)
        posterior = _posterior(candidates, probabilities)
        for candidate, weight in zip(candidates, posterior, strict=True):
            for event in candidate.events:
                narrow[event] = narrow.get(event, 0.0) + weight

        chart = _copy_forest(item, inventory, probabilities)
        outside = outside_from_edges(chart, item.goal)
        for parent in _chart_items(chart):
            for edge, marginal in outside.rule_marginals(parent):
                for event in _edge_events(item, parent, edge):
                    full[event] = full.get(event, 0.0) + marginal

    mixed: dict[BiskEvent, float] = {}
    contexts = {event.context for event in (*narrow, *full)}

    for context in contexts:
        k_total = fsum(
            count for event, count in narrow.items() if event.context == context
        )
        full_total = fsum(
            count for event, count in full.items() if event.context == context
        )
        coefficient = (
            0.0 if k_total + full_total == 0.0 else k_total / (k_total + full_total)
        )
        outcomes = {event for event in (*narrow, *full) if event.context == context}

        for event in outcomes:
            k_probability = 0.0 if k_total == 0.0 else narrow.get(event, 0.0) / k_total
            full_probability = (
                0.0 if full_total == 0.0 else full.get(event, 0.0) / full_total
            )
            mixed[event] = (
                coefficient * k_probability + (1.0 - coefficient) * full_probability
            )

    return mixed


@final
class BiskPrediction(FrozenValue):
    """Supertag and dependency predictions from one fitted model."""

    _noun = "Bisk--Hockenmaier prediction"
    __slots__ = ("_dependencies", "_supertags")

    def __init__(
        self,
        supertags: SupertagPrediction,
        dependencies: AttachmentTree | None,
    ) -> None:
        object.__setattr__(self, "_supertags", supertags)
        object.__setattr__(self, "_dependencies", dependencies)

    @property
    def supertags(self) -> SupertagPrediction:
        return self._supertags

    @property
    def dependencies(self) -> AttachmentTree | None:
        return self._dependencies


@final
class BiskHockenmaierModel(FrozenValue):
    """A fitted POS lexicon and normalized generative event table."""

    _noun = "Bisk--Hockenmaier model"
    __slots__ = (
        "_config_hash",
        "_convention_hash",
        "_inventory",
        "_k",
        "_probabilities",
    )

    def __init__(
        self,
        inventory: Mapping[str, tuple[Term, ...]],
        probabilities: Mapping[BiskEvent, float],
        settings: BiskHockenmaierSettings,
    ) -> None:
        object.__setattr__(
            self,
            "_inventory",
            MappingProxyType(
                {tag: tuple(categories) for tag, categories in inventory.items()}
            ),
        )
        object.__setattr__(
            self, "_probabilities", MappingProxyType(dict(probabilities))
        )
        object.__setattr__(self, "_k", settings.k)
        convention_hash = config_digest(settings.conventions)
        object.__setattr__(self, "_convention_hash", convention_hash)
        object.__setattr__(
            self,
            "_config_hash",
            config_digest(
                {
                    **settings.to_spec(),
                    "convention_hash": convention_hash,
                }
            ),
        )

    @property
    def config_hash(self) -> str:
        return self._config_hash

    @property
    def convention_hash(self) -> str:
        return self._convention_hash

    def categories(self, tag: str, /) -> tuple[Term, ...]:
        """Return the induced categories of one POS tag."""
        return self._inventory.get(tag, ())

    def event_probability(self, event: BiskEvent, /) -> float:
        """Return one fitted conditional probability, zero outside support."""
        return self._probabilities.get(event, 0.0)

    def predict(self, item: BiskItem, /) -> BiskPrediction:
        """Return posterior supertags and the Viterbi dependency tree."""
        candidates = bisk_candidates(
            item,
            self._inventory,
            self._probabilities,
            limit=self._k,
        )

        if not candidates:
            tokens = tuple(
                TokenSupertags(
                    tuple(
                        CategoryProbability(category, 1.0 / len(categories))
                        for category in categories
                    )
                )
                for tag in item.tags
                if (categories := self.categories(tag))
            )

            if len(tokens) != len(item.tags):
                msg = f"item {item.item_id!r} has a POS tag outside the induced lexicon"
                raise ValueError(msg)

            return BiskPrediction(SupertagPrediction(tokens, None), None)

        posterior = _posterior(candidates, self._probabilities)
        token_rows: list[TokenSupertags] = []

        for position in range(len(item.forms)):
            masses: dict[Term, float] = {}

            for candidate, probability in zip(candidates, posterior, strict=True):
                category = candidate.categories[position]
                masses[category] = masses.get(category, 0.0) + probability

            total = fsum(masses.values())
            ordered = sorted(
                (
                    (category, probability / total)
                    for category, probability in masses.items()
                ),
                key=lambda pair: (-pair[1], _term_order(pair[0])),
            )
            token_rows.append(
                TokenSupertags(
                    tuple(
                        CategoryProbability(category, probability)
                        for category, probability in ordered
                    )
                )
            )

        best = max(range(len(candidates)), key=lambda position: posterior[position])
        selected = candidates[best]
        return BiskPrediction(
            SupertagPrediction(tuple(token_rows), selected.categories),
            selected.dependencies,
        )


@final
class BiskHockenmaierInducer(FrozenValue):
    """Induce the bounded category inventory and estimate its event model."""

    _noun = "Bisk--Hockenmaier inducer"
    __slots__ = (
        "_coordination",
        "_functors",
        "_headed_arguments",
        "_nominal",
        "_sentence",
        "_sentence_arguments",
        "_settings",
    )

    def __init__(
        self,
        settings: BiskHockenmaierSettings,
        functors: Functors[Term],
        *,
        nominal: Term,
        sentence: Term,
        coordination: Term,
        headed_arguments: Sequence[Term],
        sentence_arguments: Sequence[Term],
    ) -> None:
        if not isinstance(settings, BiskHockenmaierSettings):
            msg = f"induction settings are not Bisk--Hockenmaier settings: {settings!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_settings", settings)
        object.__setattr__(self, "_functors", functors)
        object.__setattr__(self, "_nominal", nominal)
        object.__setattr__(self, "_sentence", sentence)
        object.__setattr__(self, "_coordination", coordination)
        object.__setattr__(self, "_headed_arguments", tuple(headed_arguments))
        object.__setattr__(self, "_sentence_arguments", tuple(sentence_arguments))

    def _arity(self, category: Term) -> int:
        arity = 0
        current = category

        while self._functors.is_functor(current):
            arity += 1
            current = current.args[self._functors.result_path[0]]

        return arity

    def _base(self, category: Term) -> Term:
        current = category

        while self._functors.is_functor(current):
            current = current.args[self._functors.result_path[0]]

        return current

    def _parts(self, category: Term) -> tuple[Term, Term]:
        return (
            category.args[self._functors.result_path[0]],
            category.args[self._functors.argument_path[0]],
        )

    def _modifier(self, category: Term) -> bool:
        if not self._functors.is_functor(category):
            return False
        result, argument = self._parts(category)
        return result.key() == argument.key()

    def _arguments(self, category: Term) -> tuple[Term, ...]:
        """Return the functor spine arguments from outermost inward."""
        arguments: list[Term] = []
        current = category
        while self._functors.is_functor(current):
            result, argument = self._parts(current)
            arguments.append(argument)
            current = result
        return tuple(arguments)

    def _canonical_transitive(self, category: Term) -> bool:
        r"""Reject the unattested `(S/N)\N` ordering from the description."""
        if (
            self._arity(category) != 2
            or self._base(category).key() != self._sentence.key()
        ):
            return True

        outer_result, outer_argument = self._parts(category)

        if outer_argument.key() != self._nominal.key() or not self._functors.is_functor(
            outer_result
        ):
            return True

        _, inner_argument = self._parts(outer_result)

        if inner_argument.key() != self._nominal.key():
            return True

        return not (
            self._functors.slash(category) == self._functors.backward_slash
            and self._functors.slash(outer_result) == self._functors.forward_slash
        )

    def _proposable(
        self,
        tag: str,
        category: Term,
        inventory: Mapping[str, set[Term]],
        derived: frozenset[Term],
    ) -> bool:
        standing = inventory.get(tag, set())

        if category in standing:
            return True
        if not self._structurally_admitted(tag, category):
            return False
        base = self._base(category)
        modifier = self._modifier(category)
        allowed = (
            self._sentence_arguments
            if base.key() == self._sentence.key() and not modifier
            else self._headed_arguments
        )
        attested = {
            term for categories in inventory.values() for term in categories
        } | set(derived)
        allowed_keys = {term.key() for term in (*allowed, *attested)}
        arguments = self._arguments(category)
        rejected_argument = any(
            argument.key() not in allowed_keys for argument in arguments
        )
        result, _ = self._parts(category)
        unattested_raised_result = (
            self._functors.is_functor(result) and result not in attested
        )
        return not (rejected_argument or unattested_raised_result)

    def _structurally_admitted(self, tag: str, category: Term, /) -> bool:
        """Return whether a category can enter at any declared induction stage."""
        base = self._base(category)
        structurally_rejected = (
            self._arity(category) > self._settings.max_arity
            or not self._canonical_transitive(category)
            or not self._functors.is_functor(category)
            or (
                tag in self._settings.verbal_tags and base.key() != self._sentence.key()
            )
        )
        if structurally_rejected:
            return False
        modifier = self._modifier(category)
        allowed = (
            self._sentence_arguments
            if base.key() == self._sentence.key() and not modifier
            else self._headed_arguments
        )
        allowed_keys = {term.key() for term in allowed}
        arguments = self._arguments(category)
        rejected_argument = any(
            argument.key() not in allowed_keys
            and not (
                modifier
                and self._settings.second_order_modifiers
                and self._modifier(argument)
                and self._base(argument).key() in allowed_keys
            )
            for argument in arguments
        )
        rejected_second_order = (
            not self._settings.second_order_modifiers
            and modifier
            and any(self._functors.is_functor(argument) for argument in arguments)
        )
        return not (rejected_argument or rejected_second_order)

    @staticmethod
    def _held(inventory: Mapping[str, set[Term]]) -> dict[str, tuple[Term, ...]]:
        """Freeze a working POS inventory in deterministic order."""
        return {
            tag: tuple(sorted(categories, key=_term_order))
            for tag, categories in sorted(inventory.items())
        }

    def _grow(
        self,
        items: Sequence[BiskItem],
        inventory: dict[str, set[Term]],
        derived: frozenset[Term],
        /,
    ) -> None:
        """Run one whole-corpus successful-parse induction pass."""
        proposals = {tag: set(categories) for tag, categories in inventory.items()}
        for item in items:
            for position, tag in enumerate(item.tags):
                for key in item.chart.items_at((position, position + 1)):
                    if self._proposable(tag, key.reading, inventory, derived):
                        proposals.setdefault(tag, set()).add(key.reading)
        held = self._held(proposals)
        additions: dict[str, set[Term]] = {}
        for item in items:
            chart = _copy_forest(item, held, None)
            outside = outside_from_edges(chart, item.goal)
            for position, tag in enumerate(item.tags):
                for parent in chart.items_at((position, position + 1)):
                    if any(
                        marginal > 0.0 for _, marginal in outside.rule_marginals(parent)
                    ):
                        additions.setdefault(tag, set()).add(parent.reading)
        for tag, categories in additions.items():
            inventory.setdefault(tag, set()).update(categories)

    def _derived(
        self,
        items: Sequence[BiskItem],
        inventory: Mapping[str, set[Term]],
        /,
    ) -> frozenset[Term]:
        """Return categories of derived constituents in successful parses."""
        held = self._held(inventory)
        categories: set[Term] = set()
        for item in items:
            chart = _copy_forest(item, held, None)
            outside = outside_from_edges(chart, item.goal)
            for span in chart.index.spans_by_length():
                if span[1] == span[0] + 1:
                    continue
                for key in outside.intersection(span):
                    if outside.constituent_marginal(key) > 0.0:
                        categories.add(key.reading)
        return frozenset(categories)

    def _inventory(self, items: Sequence[BiskItem]) -> dict[str, tuple[Term, ...]]:
        inventory = self._initial_inventory()

        for _ in range(self._settings.category_induction_steps):
            self._grow(items, inventory, frozenset())
        if self._settings.derived_constituent_pass:
            self._grow(items, inventory, self._derived(items, inventory))
        return self._held(inventory)

    def _initial_inventory(self) -> dict[str, set[Term]]:
        """Return the paper's POS-conditioned atomic seeds."""
        inventory: dict[str, set[Term]] = {}

        for tag in self._settings.nominal_tags:
            inventory.setdefault(tag, set()).add(self._nominal)
        for tag in self._settings.verbal_tags:
            inventory.setdefault(tag, set()).add(self._sentence)
        for tag in self._settings.coordination_tags:
            inventory.setdefault(tag, set()).add(self._coordination)
        return inventory

    def lexical_categories(
        self, tag: str, candidates: Sequence[Term], /
    ) -> tuple[Term, ...]:
        """Return a parser superset containing every stage-admissible category."""
        inventory = self._initial_inventory()
        held = tuple(candidates)
        standing = inventory.get(tag, set())
        return tuple(
            category
            for category in held
            if category in standing or self._structurally_admitted(tag, category)
        )

    def fit(self, items: Sequence[BiskItem], /) -> BiskHockenmaierModel:
        """Fit a bounded induced lexicon and its generative parameters."""
        held_items = tuple(items)

        if not held_items:
            msg = "Bisk--Hockenmaier induction needs at least one training item"
            raise ValueError(msg)

        if len({item.item_id for item in held_items}) != len(held_items):
            msg = "Bisk--Hockenmaier training repeats an item id"
            raise ValueError(msg)
        declared_tags = (
            self._settings.nominal_tags
            | self._settings.verbal_tags
            | self._settings.coordination_tags
            | self._settings.other_tags
        )
        unknown = sorted(
            {tag for item in held_items for tag in item.tags} - declared_tags
        )
        if unknown:
            msg = (
                "training items contain POS tags outside the declared groups: "
                f"{unknown!r}"
            )
            raise ValueError(msg)

        inventory = self._inventory(held_items)
        live: list[BiskItem] = []
        events: list[BiskEvent] = []
        for item in held_items:
            chart = _copy_forest(item, inventory, None)
            outside = outside_from_edges(chart, item.goal)
            if outside.partition == -inf:
                continue
            live.append(item)
            for parent in _chart_items(chart):
                for edge, marginal in outside.rule_marginals(parent):
                    if marginal > 0.0:
                        events.extend(_edge_events(item, parent, edge))

        if not live:
            msg = "the induced lexicon licenses no complete training derivation"
            raise ValueError(msg)
        unique_events = tuple(dict.fromkeys(events))
        support: dict[_EventContext, tuple[BiskEvent, ...]] = {}

        for context in dict.fromkeys(event.context for event in unique_events):
            support[context] = tuple(
                event for event in unique_events if event.context == context
            )

        probabilities = _normalized({}, support, self._settings.probability_floor)

        for _ in range(self._settings.iterations):
            counted = _counts(
                live,
                inventory,
                probabilities,
                k=self._settings.k,
            )
            updated = _normalized(
                counted,
                support,
                self._settings.probability_floor,
            )
            change = max(
                abs(updated[event] - probabilities[event]) for event in unique_events
            )
            probabilities = updated

            if change <= self._settings.tolerance:
                break

        return BiskHockenmaierModel(inventory, probabilities, self._settings)
