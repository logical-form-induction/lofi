"""Logical-form match metrics of increasing leniency."""

from random import Random
from statistics import fmean
from typing import TYPE_CHECKING, Final, NamedTuple, final

from lofi._checks import ensure_non_negative
from lofi.eval._tree_agreement import normalized_tree_distance
from lofi.eval._values import MetricResult, Resampling
from lofi.spec.evaluators.logical_form import LogicalFormMeasure
from lofi.terms import (
    Calculus,
    Nominal,
    Normalization,
    NotLocallyClosedError,
    Term,
    TermStore,
    TreeEditDistance,
    normalize,
    to_core,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import ItemId
    from lofi.eval._values import ConventionVector
    from lofi.terms import Signature, Sort

__all__ = [
    "LOGICAL_FORM_MEASURES",
    "LogicalFormMeasure",
    "LogicalFormMetric",
    "LogicalFormScores",
    "SmatchScore",
    "alpha_match",
    "exact_match",
    "logical_form_scores",
    "normal_form_match",
    "normalized_form",
    "smatch_score",
]

LOGICAL_FORM_MEASURES: Final[tuple[LogicalFormMeasure, ...]] = (
    "exact",
    "alpha",
    "normal-form",
    "smatch",
    "nted",
)
"""The measures in increasing leniency, with normalized distance last."""

type _Label = tuple[str, int | None]
"""An operator name and payload identifying a node's concept."""

type _Edge = tuple[str, int, int]
"""A relation from one node to another."""


class _Flattening(NamedTuple):
    """The relation triples of one logical form.

    Attributes
    ----------
    labels : tuple[_Label, ...]
        Concept of each node, indexed by node.
    edges : tuple[_Edge, ...]
        Relations between nodes.
    variables : int
        How many nodes are bound variables and so have no concept.
    """

    labels: tuple[_Label, ...]
    edges: tuple[_Edge, ...]
    variables: int

    @property
    def triples(self) -> int:
        """Return how many triples the flattening has in all."""
        return len(self.labels) - self.variables + len(self.edges)


_VARIABLE: Final[_Label] = ("", None)
"""The empty concept a bound-variable node has."""


def _flatten(term: Term) -> _Flattening:
    """Flatten a closed term into concept and relation triples.

    Every operator node has a concept triple, its operator and payload, and one
    relation triple per argument. A binder additionally has one relation triple
    per variable it binds, pointing at a concept-free variable node, and an
    occurrence of that variable is a relation from the occurrence's parent to
    the variable node rather than a node of its own.

    Raises
    ------
    NotLocallyClosedError
        If ``term`` has an open bound index.
    """
    if not term.is_closed:
        raise NotLocallyClosedError(max_index=term.maxidx, depth=0)

    labels: list[_Label] = []
    edges: list[_Edge] = []
    scopes: dict[Sort, list[int]] = {}
    variables = 0

    def visit(node: Term) -> int:
        nonlocal variables

        op = node.op

        if op.is_bound_variable and node.payload is not None:
            return scopes[op.sort][-1 - node.payload]

        this = len(labels)
        labels.append((op.name, node.payload))
        bound: list[int] = []

        for slot, _ in enumerate(op.binding):
            variable = len(labels)
            labels.append(_VARIABLE)
            variables += 1
            bound.append(variable)
            edges.append((f"binds{slot}", this, variable))

        for position, child in enumerate(node.args):
            pushed: list[Sort] = []

            # the first site of a sort is the innermost, so it is pushed last.
            for site, variable in reversed(tuple(zip(op.binding, bound, strict=True))):
                if position in site.arguments:
                    scopes.setdefault(site.sort, []).append(variable)
                    pushed.append(site.sort)

            at = len(edges)
            target = visit(child)
            edges.insert(at, (f"arg{position}", this, target))

            for sort in pushed:
                scopes[sort].pop()

        return this

    visit(term)

    return _Flattening(tuple(labels), tuple(edges), variables)


class SmatchScore(NamedTuple):
    """A Smatch-style triple overlap between two logical forms.

    Attributes
    ----------
    matched : int
        Predicted triples with a gold image under the best alignment found.
    predicted : int
        Triples in the predicted flattening.
    gold : int
        Triples in the gold flattening.
    """

    matched: int
    predicted: int
    gold: int

    @property
    def precision(self) -> float:
        """Return the matched fraction of predicted triples."""
        return self.matched / self.predicted

    @property
    def recall(self) -> float:
        """Return the matched fraction of gold triples."""
        return self.matched / self.gold

    @property
    def f1(self) -> float:
        """Return the harmonic mean of precision and recall."""
        return 2 * self.matched / (self.predicted + self.gold)


@final
class _Search:
    """A hill-climbing search over node alignments of two flattenings."""

    __slots__ = (
        "_gold",
        "_gold_edges",
        "_gold_outgoing",
        "_incident",
        "_inverse",
        "_mapping",
        "_predicted",
        "_predicted_outgoing",
    )

    def __init__(self, predicted: _Flattening, gold: _Flattening) -> None:
        self._predicted = predicted
        self._gold = gold
        self._gold_edges = frozenset(gold.edges)
        self._predicted_outgoing = _outgoing(predicted)
        self._gold_outgoing = _outgoing(gold)
        incident: list[list[int]] = [[] for _ in predicted.labels]

        for index, (_, source, target) in enumerate(predicted.edges):
            incident[source].append(index)

            if target != source:
                incident[target].append(index)

        self._incident = tuple(tuple(found) for found in incident)
        self._mapping: list[int | None] = [None] * len(predicted.labels)
        self._inverse: list[int | None] = [None] * len(gold.labels)

    def start_greedy(self) -> None:
        """Align each predicted node to the first free gold node with its concept."""
        self._clear()

        for node, label in enumerate(self._predicted.labels):
            if label == _VARIABLE:
                continue

            for candidate, held in enumerate(self._gold.labels):
                if held == label and self._inverse[candidate] is None:
                    self._assign(node, candidate)

                    break

    def start_structural(self) -> None:
        """Align the roots and then each argument to the argument at its position."""
        self._clear()
        pending = [(0, 0)]

        while pending:
            node, candidate = pending.pop()

            if self._mapping[node] is not None or self._inverse[candidate] is not None:
                continue

            self._assign(node, candidate)
            targets = self._gold_outgoing[candidate]
            pending.extend(
                (target, targets[relation])
                for relation, target in self._predicted_outgoing[node].items()
                if relation in targets
            )

    def start_random(self, rng: Random) -> None:
        """Align a random injection of predicted nodes into gold nodes."""
        self._clear()
        count = min(len(self._predicted.labels), len(self._gold.labels))
        sources = rng.sample(range(len(self._predicted.labels)), count)
        targets = rng.sample(range(len(self._gold.labels)), count)

        for node, candidate in zip(sources, targets, strict=True):
            self._assign(node, candidate)

    def climb(self) -> int:
        """Apply improving moves until none remains and return the score."""
        while True:
            best_gain = 0
            best_move: tuple[int, int] | None = None

            for node in range(len(self._predicted.labels)):
                for candidate in range(len(self._gold.labels)):
                    if self._mapping[node] == candidate:
                        continue

                    gain = self._gain(node, candidate)

                    if gain > best_gain:
                        best_gain = gain
                        best_move = (node, candidate)

            if best_move is None:
                return self._score()

            self._move(*best_move)

    def _clear(self) -> None:
        """Forget every alignment."""
        self._mapping = [None] * len(self._predicted.labels)
        self._inverse = [None] * len(self._gold.labels)

    def _assign(self, node: int, candidate: int) -> None:
        """Align one predicted node to one free gold node."""
        self._mapping[node] = candidate
        self._inverse[candidate] = node

    def _move(self, node: int, candidate: int) -> None:
        """Realign one node, swapping with the node holding the candidate."""
        displaced = self._inverse[candidate]
        previous = self._mapping[node]

        if displaced is not None:
            self._mapping[displaced] = previous

        if previous is not None:
            self._inverse[previous] = displaced

        self._assign(node, candidate)

    def _gain(self, node: int, candidate: int) -> int:
        """Return the score change of realigning one node."""
        displaced = self._inverse[candidate]
        affected = (node,) if displaced is None else (node, displaced)
        before = self._local(affected, {})
        override: dict[int, int | None] = {node: candidate}

        if displaced is not None:
            override[displaced] = self._mapping[node]

        return self._local(affected, override) - before

    def _local(self, nodes: tuple[int, ...], override: Mapping[int, int | None]) -> int:
        """Return the matched triples touching some nodes under an override."""
        matched = 0
        seen: set[int] = set()

        for node in nodes:
            image = override.get(node, self._mapping[node])

            if (
                image is not None
                and self._predicted.labels[node] != _VARIABLE
                and self._gold.labels[image] == self._predicted.labels[node]
            ):
                matched += 1

            for index in self._incident[node]:
                if index in seen:
                    continue

                seen.add(index)
                relation, source, target = self._predicted.edges[index]
                source_image = override.get(source, self._mapping[source])
                target_image = override.get(target, self._mapping[target])

                if (
                    source_image is not None
                    and target_image is not None
                    and (relation, source_image, target_image) in self._gold_edges
                ):
                    matched += 1

        return matched

    def _score(self) -> int:
        """Return the matched triples under the current alignment."""
        return self._local(tuple(range(len(self._predicted.labels))), {})


def _outgoing(flattening: _Flattening) -> tuple[Mapping[str, int], ...]:
    """Return each node's relation targets keyed by relation."""
    outgoing: list[dict[str, int]] = [{} for _ in flattening.labels]

    for relation, source, target in flattening.edges:
        outgoing[source][relation] = target

    return tuple(outgoing)


def smatch_score(
    predicted: Term,
    gold: Term,
    /,
    *,
    restarts: int = 4,
    seed: int = 0,
) -> SmatchScore:
    """Return the Smatch-style triple overlap of two closed terms.

    Each term is flattened into concept and relation triples over its nodes,
    and the node alignment maximizing the number of predicted triples with a
    gold image is searched by hill climbing from a concept-matching start, a
    position-matching start, and ``restarts`` further random starts. The
    score measures surface overlap between the two flattenings and not
    logical equivalence: two logically equivalent forms with different
    structure score below one, and two forms sharing most of their structure
    score high whatever they mean.

    Parameters
    ----------
    predicted : Term
        The predicted form.
    gold : Term
        The gold form.
    restarts : int
        Random starts searched after the greedy start.
    seed : int
        Seed of the random starts.

    Raises
    ------
    NotLocallyClosedError
        If either term has an open bound index.
    TypeError
        If ``restarts`` or ``seed`` is not an integer.
    ValueError
        If ``restarts`` or ``seed`` is negative.
    """
    rounds = ensure_non_negative("a Smatch restart count", restarts)
    rng = Random(ensure_non_negative("a Smatch seed", seed))
    left = _flatten(predicted)
    right = _flatten(gold)
    search = _Search(left, right)
    search.start_greedy()
    best = search.climb()
    search.start_structural()
    best = max(best, search.climb())

    for _ in range(rounds):
        search.start_random(rng)
        best = max(best, search.climb())

    return SmatchScore(best, left.triples, right.triples)


def normalized_form(
    store: TermStore,
    term: Term,
    calculus: Calculus,
    /,
    *,
    normalization: Normalization,
) -> Term:
    """Return a term's declared normal form.

    Raises
    ------
    NonTerminatingError
        If the term has no normal form within the declared fuel.
    """
    return normalize(store, term, calculus, normalization=normalization)


def exact_match(predicted: Nominal, gold: Nominal, /) -> bool:
    """Return whether two named forms have the same canonical rendering.

    The rendering is the generic prefix form with binder names written out,
    so two forms differing only in a bound variable's name do not match.
    """
    return repr(predicted) == repr(gold)


def alpha_match(
    store: TermStore,
    predicted: Nominal,
    gold: Nominal,
    /,
) -> bool:
    """Return whether two named forms are alpha-equivalent.

    Each form is converted in one traversal to the store's locally nameless
    term, in which a bound variable is the de Bruijn index of its binder, so
    two forms are alpha-equivalent exactly when they intern as one term.

    Raises
    ------
    UnknownOperatorError
        If a form uses an operator the store's signature does not declare.
    """
    return to_core(store, predicted) is to_core(store, gold)


def normal_form_match(
    store: TermStore,
    calculus: Calculus,
    predicted: Nominal,
    gold: Nominal,
    /,
    *,
    normalization: Normalization,
) -> bool:
    """Return whether two named forms have one declared normal form.

    Raises
    ------
    NonTerminatingError
        If a form has no normal form within the declared fuel.
    UnknownOperatorError
        If a form uses an operator the store's signature does not declare.
    """
    left = normalized_form(
        store,
        to_core(store, predicted),
        calculus,
        normalization=normalization,
    )
    right = normalized_form(
        store,
        to_core(store, gold),
        calculus,
        normalization=normalization,
    )

    return left is right


class LogicalFormScores(NamedTuple):
    """Every match measure of one predicted form against its gold form.

    Attributes
    ----------
    exact : bool
        Whether the canonical renderings of the two named forms are equal.
    alpha : bool
        Whether the two forms are the same term after de Bruijn conversion.
    normal_form : bool
        Whether the two forms have the same declared normal form.
    smatch : SmatchScore
        Triple overlap of the two normalized forms.
    nted : float
        Normalized tree edit distance between the two de Bruijn forms.
    """

    exact: bool
    alpha: bool
    normal_form: bool
    smatch: SmatchScore
    nted: float

    def measure(self, measure: LogicalFormMeasure) -> float:
        """Return one headline measure as a number."""
        return self.components()[measure]

    def components(self) -> dict[str, float]:
        """Return every measure, with the Smatch precision and recall beside it."""
        return {
            "exact": float(self.exact),
            "alpha": float(self.alpha),
            "normal-form": float(self.normal_form),
            "smatch": self.smatch.f1,
            "nted": self.nted,
            "smatch-precision": self.smatch.precision,
            "smatch-recall": self.smatch.recall,
        }


def logical_form_scores(
    store: TermStore,
    calculus: Calculus,
    predicted: Nominal,
    gold: Nominal,
    /,
    *,
    distance: TreeEditDistance | None = None,
    restarts: int = 4,
    seed: int = 0,
    normalization: Normalization,
) -> LogicalFormScores:
    """Score one predicted named form against its gold named form.

    Exact match compares the canonical prefix renderings of the two named
    forms, so a renamed bound variable is a mismatch. Alpha match converts
    both forms to the store's locally nameless terms, where bound variables
    are de Bruijn indices, and compares the interned terms. Normal-form match
    compares those terms under the declared normalization treatment.
    The Smatch score flattens the normalized terms, and the normalized tree
    edit distance is taken between the locally nameless terms.

    Parameters
    ----------
    store : TermStore
        Store the forms are interned in.
    calculus : Calculus
        Abstraction and application operators of the forms.
    predicted : Nominal
        The predicted form.
    gold : Nominal
        The gold form.
    distance : TreeEditDistance | None
        Edit-distance kernel to reuse, or ``None`` for a fresh one.
    restarts : int
        Random starts of the Smatch alignment search after the greedy start.
    seed : int
        Seed of the Smatch random starts.
    normalization : Normalization
        Declared reduction order, eta treatment, quotient and fuel.

    Raises
    ------
    NonTerminatingError
        If a form has no normal form within the declared fuel.
    UnknownOperatorError
        If a form uses an operator the store's signature does not declare.
    """
    kernel: TreeEditDistance = TreeEditDistance() if distance is None else distance
    predicted_core = to_core(store, predicted)
    gold_core = to_core(store, gold)
    predicted_normal = normalized_form(
        store,
        predicted_core,
        calculus,
        normalization=normalization,
    )
    gold_normal = normalized_form(
        store,
        gold_core,
        calculus,
        normalization=normalization,
    )

    return LogicalFormScores(
        exact_match(predicted, gold),
        predicted_core is gold_core,
        predicted_normal is gold_normal,
        smatch_score(predicted_normal, gold_normal, restarts=restarts, seed=seed),
        normalized_tree_distance(kernel, predicted_core, gold_core),
    )


@final
class LogicalFormMetric:
    """Mean of one logical-form measure over scored items.

    Parameters
    ----------
    signature : Signature
        Signature the forms are written in.
    calculus : Calculus
        Abstraction and application operators of the forms.
    measure : LogicalFormMeasure
        The headline measure this metric reports.
    conventions : ConventionVector | None
        Grammatical conventions under which the forms were decoded, if supplied.
    seed : int
        Seed of the Smatch random starts.
    restarts : int
        Random starts of the Smatch alignment search after the greedy start.
    normalization : Normalization
        Declared reduction order, eta treatment, quotient and fuel.

    Raises
    ------
    TypeError
        If ``seed`` or ``restarts`` is not an integer.
    ValueError
        If ``measure`` is unknown or an integer argument is negative.
    UnknownOperatorError
        If the signature does not declare the calculus operators.
    """

    __slots__ = (
        "_calculus",
        "_conventions",
        "_distance",
        "_items",
        "_measure",
        "_normalization",
        "_resampling",
        "_restarts",
        "_seed",
        "_signature",
        "_store",
    )

    def __init__(
        self,
        signature: Signature,
        calculus: Calculus,
        measure: LogicalFormMeasure,
        *,
        normalization: Normalization,
        store_nodes: int,
        conventions: ConventionVector | None = None,
        seed: int = 0,
        restarts: int = 4,
    ) -> None:
        if measure not in LOGICAL_FORM_MEASURES:
            msg = f"a logical-form measure is not one of {LOGICAL_FORM_MEASURES!r}"
            raise ValueError(msg)

        signature.by_name(calculus.lam)
        signature.by_name(calculus.app)
        self._signature = signature
        self._calculus = calculus
        self._measure: LogicalFormMeasure = measure
        self._conventions = conventions
        self._restarts = ensure_non_negative("a Smatch restart count", restarts)
        self._normalization = normalization
        self._seed = ensure_non_negative("a metric seed", seed)
        self._resampling = Resampling("item", self._seed)
        self._store: TermStore = TermStore(signature, max_nodes=store_nodes)
        self._distance: TreeEditDistance = TreeEditDistance()
        self._items: dict[
            ItemId, tuple[Nominal | None, Nominal, LogicalFormScores | None]
        ] = {}

    @property
    def name(self) -> str:
        """Return the metric name."""
        return f"logical-form-{self._measure}"

    @property
    def measure(self) -> LogicalFormMeasure:
        """Return the headline measure."""
        return self._measure

    @property
    def conventions(self) -> ConventionVector | None:
        """Return the conventions these scores are reported under, if supplied."""
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        """Return whether update order affects the aggregate."""
        return True

    def reset(self) -> None:
        """Forget every item scored so far."""
        self._items = {}
        self._store = TermStore(
            self._signature,
            max_nodes=self._store.max_nodes,
        )
        self._distance = TreeEditDistance()

    def update(self, pred: Nominal | None, gold: Nominal, *, key: ItemId) -> None:
        """Score one predicted form against its gold form.

        Parameters
        ----------
        pred : Nominal | None
            The predicted named form, or no usable form.
        gold : Nominal
            The gold named form.
        key : ItemId
            Item identifier.

        Raises
        ------
        TypeError
            If either form is not a named term.
        ValueError
            If the key was already scored against different forms.
        NonTerminatingError
            If a form has no normal form within the declared fuel.
        UnknownOperatorError
            If a form uses an operator the signature does not declare.
        """
        if pred is not None and not isinstance(pred, Nominal):
            msg = f"a predicted logical form is not a named term: {pred!r}"
            raise TypeError(msg)

        if not isinstance(gold, Nominal):
            msg = f"a gold logical form is not a named term: {gold!r}"
            raise TypeError(msg)

        if key in self._items:
            standing = self._items[key]
            if (standing[0], standing[1]) != (pred, gold):
                msg = f"the item {key!r} was already scored against a different form"
                raise ValueError(msg)

            return

        scores = (
            None
            if pred is None
            else logical_form_scores(
                self._store,
                self._calculus,
                pred,
                gold,
                distance=self._distance,
                restarts=self._restarts,
                seed=self._seed,
                normalization=self._normalization,
            )
        )
        self._items[key] = (pred, gold, scores)

    def compute(self) -> MetricResult:
        """Return the mean measure over every item scored so far.

        Returns
        -------
        MetricResult
            The mean headline measure with every measure's mean as a
            component, or an absent result when no item is scored.
        """
        scored = tuple(
            scores for _, _, scores in self._items.values() if scores is not None
        )
        missing = sum(scores is None for _, _, scores in self._items.values())

        if not scored and (
            not missing
            or self._conventions is None
            or self._conventions.coverage == "ignore"
        ):
            return MetricResult(None, 0, self._resampling)

        values = tuple(scores.measure(self._measure) for scores in scored)
        components = tuple(scores.components() for scores in scored)
        if self._conventions is not None and self._conventions.coverage == "zero":
            values = (*values, *(0.0 for _ in range(missing)))
            zero = {
                "exact": 0.0,
                "alpha": 0.0,
                "normal-form": 0.0,
                "smatch": 0.0,
                "nted": 0.0,
                "smatch-precision": 0.0,
                "smatch-recall": 0.0,
            }
            components = (*components, *(zero for _ in range(missing)))

        return MetricResult(
            fmean(values),
            len(values),
            self._resampling,
            components={
                name: fmean(row[name] for row in components) for name in components[0]
            },
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return the measure of each item scored so far.

        Returns
        -------
        Mapping[ItemId, MetricResult]
            One single-item result per key.
        """
        results: dict[ItemId, MetricResult] = {}
        for key, (_, _, scores) in self._items.items():
            if scores is None:
                if (
                    self._conventions is not None
                    and self._conventions.coverage == "zero"
                ):
                    results[key] = MetricResult(0.0, 1, self._resampling)
                continue
            results[key] = MetricResult(
                scores.measure(self._measure),
                1,
                self._resampling,
                components=scores.components(),
            )
        return results


def _mean_components(scored: Sequence[LogicalFormScores]) -> dict[str, float]:
    """Return the mean of every component over scored items."""
    rows = tuple(scores.components() for scores in scored)

    return {name: fmean(row[name] for row in rows) for name in rows[0]}
