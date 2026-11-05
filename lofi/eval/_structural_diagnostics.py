"""Held-out structural reports from one supertag-factored decode."""

from collections.abc import Mapping as RuntimeMapping
from math import fsum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, cast, override

from lofi.data import AttachmentTree
from lofi.deduce import Derivation as ForestDerivation
from lofi.deduce import derivation_count_report
from lofi.eval._attachment import AttachmentScorer, LengthBand
from lofi.eval._bracketing import (
    Bracketing,
    BracketingF1,
    Constituent,
    GoldBracketing,
    balanced,
    left_branching,
    right_branching,
)
from lofi.eval._decode_evaluator import DecodeEvaluator, DecodePrediction
from lofi.eval._dependencies import (
    DependencyArc,
    DependencyPrediction,
    DependencyScorer,
    DependencyToken,
)
from lofi.eval._held_out import derivation_dataset
from lofi.eval._values import ConventionVector, MetricResult, Resampling
from lofi.lexicon import BeamSource, Entry
from lofi.registry import ComponentPlugin
from lofi.spec.evaluators.structural_diagnostics import (
    StructuralDiagnosticsSettings,
    StructuralDiagnosticsSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from lofi._types import ItemId
    from lofi.data import Derivation, EvalDataset, Item
    from lofi.deduce import AStarProposerFactory
    from lofi.eval import HeldOutEvaluationContext, HeldOutItems
    from lofi.eval._ports import HeldOutData
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.terms import Term

__all__ = ["PLUGIN", "StructuralDiagnosticsEvaluator"]


class StructuralPrediction(NamedTuple):
    """All structural projections of one selected derivation."""

    bracketing: Bracketing | None
    attachment: AttachmentTree | None
    dependency: DependencyPrediction | None
    normal_form_count: int | None
    unconstrained_count: int | None
    retention: float | None


class StructuralGold(NamedTuple):
    """Gold projections scored by the structural report."""

    bracketing: GoldBracketing
    attachment: AttachmentTree
    dependency: DependencyPrediction


def _gold_bracketing(derivation: Derivation[str]) -> GoldBracketing:
    depths = {derivation.root: 0}
    for node_id in derivation.preorder:
        for child in derivation.node(node_id).children:
            depths[child] = depths[node_id] + 1
    return GoldBracketing(
        derivation.node(derivation.root).span.end,
        (
            Constituent(
                node.span.start,
                node.span.end,
                label=node.label,
                depth=depths[node_id],
            )
            for node_id, node in derivation.nodes.items()
        ),
        punctuation=(),
    )


def _gold_links(
    derivation: Derivation[str],
    node_id: int,
    heads: Mapping[str, int],
    words: tuple[str, ...],
) -> tuple[int, list[DependencyArc]]:
    node = derivation.node(node_id)
    if not node.children:
        return node.span.start, []
    below = [_gold_links(derivation, child, heads, words) for child in node.children]
    if len(below) == 1:
        return below[0]
    rule = node.rule
    if rule is None or rule not in heads:
        msg = f"the dependency interpretation names no head for gold rule {rule!r}"
        raise ValueError(msg)
    at = heads[rule]
    if at >= len(below):
        msg = f"gold rule {rule!r} selects premise {at} of {len(below)}"
        raise ValueError(msg)
    head = below[at][0]
    arcs = [arc for _, child_arcs in below for arc in child_arcs]
    arcs.extend(
        DependencyArc(
            DependencyToken(head, words[head]),
            DependencyToken(child_head, words[child_head]),
            node.label,
            premise + 1,
            rule,
        )
        for premise, (child_head, _) in enumerate(below)
        if premise != at
    )
    return head, arcs


def _pred_links(
    derivation: ForestDerivation[Term, float],
    heads: Mapping[str, int],
    words: tuple[str, ...],
    render: Callable[[Term], str],
) -> tuple[int, list[DependencyArc]]:
    children = derivation.children
    if not children:
        return derivation.key.span[0], []
    below = [_pred_links(child, heads, words, render) for child in children]
    if len(below) == 1:
        return below[0]
    rule = derivation.edge.rule_name
    at = heads.get(rule)
    if at is None:
        msg = f"the dependency interpretation names no head for rule {rule!r}"
        raise ValueError(msg)
    if at >= len(below):
        msg = f"rule {rule!r} selects premise {at} of {len(below)}"
        raise ValueError(msg)
    head = below[at][0]
    arcs = [arc for _, child_arcs in below for arc in child_arcs]
    arcs.extend(
        DependencyArc(
            DependencyToken(head, words[head]),
            DependencyToken(child_head, words[child_head]),
            render(derivation.key.reading),
            premise + 1,
            rule,
        )
        for premise, (child_head, _) in enumerate(below)
        if premise != at
    )
    return head, arcs


def _tree(
    words: tuple[str, ...], root: int, arcs: list[DependencyArc]
) -> AttachmentTree:
    heads = {root + 1: 0}
    for arc in arcs:
        heads[arc.argument_position + 1] = arc.predicate_position + 1
    return AttachmentTree(
        words, tuple(heads[position] for position in range(1, len(words) + 1))
    )


def _forest_bracketing(derivation: ForestDerivation[Term, float]) -> Bracketing:
    pending = [derivation]
    spans: list[tuple[int, int]] = []
    while pending:
        node = pending.pop()
        spans.append(node.key.span)
        pending.extend(node.children)
    return Bracketing(derivation.key.span[1], spans)


type _Pairs = Mapping[ItemId, tuple[StructuralPrediction, StructuralGold]]
"""Every held-out item's structural prediction beside its gold projection."""

type _Reading = Callable[[_Pairs, ConventionVector], MetricResult]
"""Computes one report row from the accumulated pairs under its conventions."""

type _Baseline = Callable[[int], Bracketing]
"""Brackets a sentence of one length without reading the model."""

type _BracketingRow = Callable[[BracketingF1], MetricResult]
"""Selects one row from a filled bracketing accumulator."""


def _bracketing(baseline: _Baseline | None, row: _BracketingRow, /) -> _Reading:
    """Return a bracketing row over the model's or a fixed baseline's brackets."""

    def reading(items: _Pairs, conventions: ConventionVector, /) -> MetricResult:
        metric = BracketingF1(conventions)
        for key, (pred, gold) in items.items():
            prediction = (
                pred.bracketing
                if baseline is None
                else baseline(gold.bracketing.length)
            )
            metric.update(prediction, gold.bracketing, key=key)
        return row(metric)

    return reading


def _overall(metric: BracketingF1, /) -> MetricResult:
    """Return unlabeled bracketing F1."""
    return metric.compute()


def _absent() -> MetricResult:
    """Return the row of a breakdown no gold constituent reached."""
    return MetricResult(None, 0, Resampling("constituent"))


def _label_recall(label: str, /) -> _BracketingRow:
    """Return the recall row of one declared constituent label."""
    return lambda metric: metric.per_label_recall().get(label, _absent())


def _length_recall(length: int, /) -> _BracketingRow:
    """Return the recall row of one declared constituent length."""
    return lambda metric: metric.recall_by_length().get(length, _absent())


def _depth_recall(depth: int, /) -> _BracketingRow:
    """Return the recall row of one declared constituent depth."""
    return lambda metric: metric.recall_by_depth().get(depth, _absent())


def _attachment(name: str, band: LengthBand, /) -> _Reading:
    """Return the declared attachment row over one length band."""

    def reading(items: _Pairs, conventions: ConventionVector, /) -> MetricResult:
        scorer = AttachmentScorer(conventions, (band,))
        for key, (pred, gold) in items.items():
            scorer.update(pred.attachment, gold.attachment, key=key)
        return _named_row(name, ((row.name, row.result) for row in scorer.rows()))

    return reading


def _dependency(name: str, /) -> _Reading:
    """Return the declared labeled-dependency row."""

    def reading(items: _Pairs, conventions: ConventionVector, /) -> MetricResult:
        del conventions
        scorer = DependencyScorer()
        for key, (pred, gold) in items.items():
            scorer.update(pred.dependency, gold.dependency, key=key)
        return _named_row(name, ((row.name, row.result) for row in scorer.rows()))

    return reading


def _named_row(name: str, rows: Iterable[tuple[str, MetricResult]], /) -> MetricResult:
    """Return the scorer row of one declared name, refusing a name it lacks."""
    produced = dict(rows)
    found = produced.get(name)
    if found is None:
        msg = f"the scorer reports {sorted(produced)}, not the declared row {name!r}"
        raise ValueError(msg)
    return found


def _sentence_mean(
    value: Callable[[StructuralPrediction], float | None], /
) -> _Reading:
    """Return the sentence mean of one per-prediction census value."""

    def reading(items: _Pairs, conventions: ConventionVector, /) -> MetricResult:
        del conventions
        defined = tuple(
            held for pred, _ in items.values() if (held := value(pred)) is not None
        )
        return MetricResult(
            None if not defined else fsum(defined) / len(defined),
            len(defined),
            Resampling("sentence"),
        )

    return reading


def _normal_form_count(prediction: StructuralPrediction, /) -> float | None:
    """Return the normal-form derivation count, when the census ran."""
    count = prediction.normal_form_count
    return None if count is None else float(count)


def _unconstrained_count(prediction: StructuralPrediction, /) -> float | None:
    """Return the unconstrained derivation count, when the census ran."""
    count = prediction.unconstrained_count
    return None if count is None else float(count)


def _retention(prediction: StructuralPrediction, /) -> float | None:
    """Return the normal-form retention, when the census ran."""
    return prediction.retention


def _structural_rows(
    settings: StructuralDiagnosticsSettings, band: LengthBand, /
) -> tuple[tuple[str, _Reading], ...]:
    """Return every declared structural row beside the reading that computes it.

    Parameters
    ----------
    settings : StructuralDiagnosticsSettings
        The resolved declaration.
    band : LengthBand
        The declared attachment length band.

    Returns
    -------
    tuple[tuple[str, _Reading], ...]
        One named reading per declared row, in declaration order.

    Raises
    ------
    ValueError
        If the declaration names a row no reading computes.
    """
    rows: tuple[tuple[str, _Reading], ...] = (
        ("unlabeled-bracketing-f1", _bracketing(None, _overall)),
        *(
            (
                f"bracketing-recall-label:{label}",
                _bracketing(None, _label_recall(label)),
            )
            for label in settings.labels
        ),
        *(
            (
                f"bracketing-recall-length:{length}",
                _bracketing(None, _length_recall(length)),
            )
            for length in settings.lengths
        ),
        *(
            (
                f"bracketing-recall-depth:{depth}",
                _bracketing(None, _depth_recall(depth)),
            )
            for depth in settings.depths
        ),
        ("bracketing-baseline:left", _bracketing(left_branching, _overall)),
        ("bracketing-baseline:right", _bracketing(right_branching, _overall)),
        ("bracketing-baseline:balanced", _bracketing(balanced, _overall)),
        (settings.attachment_row, _attachment(settings.attachment_row, band)),
        (settings.dependency_row, _dependency(settings.dependency_row)),
    )
    if settings.derivation_counts:
        rows = (
            *rows,
            ("normal-form-derivation-count", _sentence_mean(_normal_form_count)),
            ("unconstrained-derivation-count", _sentence_mean(_unconstrained_count)),
            ("normal-form-retention", _sentence_mean(_retention)),
        )
    computed = dict(rows)
    undeclared = tuple(name for name in settings.metrics if name not in computed)
    if undeclared:
        msg = (
            f"the declaration names {undeclared!r}, which no structural reading "
            f"computes; the readings compute {tuple(computed)!r}"
        )
        raise ValueError(msg)
    return tuple((name, computed[name]) for name in settings.metrics)


class _StructuralMetric:
    """Accumulate one declared row from the shared structural projections."""

    __slots__ = ("_conventions", "_items", "_name", "_reading")

    def __init__(
        self, name: str, reading: _Reading, conventions: ConventionVector
    ) -> None:
        self._name = name
        self._reading = reading
        self._conventions = conventions
        self._items: dict[ItemId, tuple[StructuralPrediction, StructuralGold]] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def conventions(self) -> ConventionVector:
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        return True

    def reset(self) -> None:
        self._items = {}

    def update(
        self, pred: StructuralPrediction, gold: StructuralGold, *, key: ItemId
    ) -> None:
        value = (pred, gold)
        standing = self._items.get(key)
        if standing is not None and standing != value:
            msg = f"item {key!r} has two structural predictions"
            raise ValueError(msg)
        self._items[key] = value

    def compute(self) -> MetricResult:
        return self._reading(self._items, self._conventions)

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        return MappingProxyType(
            {key: self._single(key, value) for key, value in self._items.items()}
        )

    def _single(
        self, key: ItemId, value: tuple[StructuralPrediction, StructuralGold]
    ) -> MetricResult:
        standing = self._items
        self._items = {key: value}
        try:
            return self.compute()
        finally:
            self._items = standing


class StructuralDiagnosticsEvaluator:
    """Project one decoded derivation into every structural report family."""

    __slots__ = (
        "_band",
        "_conventions",
        "_decode",
        "_derivation_counts",
        "_goal",
        "_grammar",
        "_heads",
        "_metrics",
        "_proposer",
        "_render",
    )

    def __init__(
        self,
        context: HeldOutEvaluationContext,
        settings: StructuralDiagnosticsSettings,
        /,
    ) -> None:
        declaration = context.grammar.grammar.interpretations.get(
            settings.dependency_interpretation
        )
        if declaration is None or declaration.settings.get("kind") != "dependency":
            msg = (
                f"{settings.dependency_interpretation!r} is not a dependency "
                "interpretation"
            )
            raise ValueError(msg)
        raw_heads = declaration.settings.get("heads")
        if not isinstance(raw_heads, RuntimeMapping):
            msg = (
                f"dependency interpretation {settings.dependency_interpretation!r} "
                "has no head table"
            )
            raise TypeError(msg)
        heads: dict[str, int] = {}
        for rule, at in raw_heads.items():
            if isinstance(at, bool) or not isinstance(at, int):
                msg = f"the head of rule {rule!r} is not an integer: {at!r}"
                raise TypeError(msg)
            heads[str(rule)] = at
        assembly = context.grammar
        self._heads = heads
        self._grammar = assembly.grammar
        self._goal = settings.astar.root_mode.goal(assembly.grammar.goal)
        self._render = lambda category: assembly.syntax.render(assembly.store, category)
        base_conventions = ConventionVector.from_settings(settings.conventions)
        proposer = cast("AStarProposerFactory", settings.astar.proposer)
        self._proposer = proposer
        self._conventions = base_conventions.extended(
            {
                "structural_readout": settings.readout,
                "astar_strategy": settings.astar.strategy.name,
                "astar_root_mode": settings.astar.root_mode.name,
                "parse_semiring": context.parse.semiring.name,
                "astar_heuristic": settings.astar.heuristic.name,
                "astar_proposer": proposer.name,
                **settings.decode.conventions,
                "derivation_counts": settings.derivation_counts,
                "derivation_count_parse": "exhaustive",
                "derivation_count_semiring": "counting",
                "derivation_count_uses_run_parse": False,
            }
        )
        self._band = LengthBand(settings.attachment_band, settings.attachment_limit)
        self._decode = DecodeEvaluator(
            context,
            readout=settings.readout,
            output="lexical-types",
            astar=settings.astar,
            decode=settings.decode,
            conventions=base_conventions,
        )
        self._derivation_counts = settings.derivation_counts
        self._metrics = tuple(
            _StructuralMetric(name, reading, self._conventions)
            for name, reading in _structural_rows(settings, self._band)
        )

    @property
    def conventions(self) -> ConventionVector:
        return self._conventions

    @property
    def metrics(self) -> tuple[_StructuralMetric, ...]:
        return self._metrics

    def held_out(self, data: HeldOutData, split: str, /) -> EvalDataset[StructuralGold]:
        """Return a retained partition's derivations as this evaluator reads them."""
        return derivation_dataset(self.annotate, data.derivations(split))

    def annotate(self, item: Item, derivation: Derivation[str], /) -> StructuralGold:
        words = tuple(token.surface for token in item.tokens)
        root, arcs = _gold_links(derivation, derivation.root, self._heads, words)
        return StructuralGold(
            _gold_bracketing(derivation),
            _tree(words, root, arcs),
            DependencyPrediction(arcs),
        )

    def predict[BatchT: HeldOutItems](
        self, outputs: ObjectiveOutputs[BatchT], /
    ) -> Mapping[ItemId, StructuralPrediction]:
        decoded = self._decode.predict(outputs)
        grammar = self._grammar
        predictions: dict[ItemId, StructuralPrediction] = {}
        for item_id, words in zip(
            outputs.batch.item_ids, outputs.batch.words, strict=True
        ):
            found: DecodePrediction = decoded[item_id]
            derivation = found.derivation
            if self._derivation_counts:
                source = BeamSource(
                    tuple(
                        tuple(
                            Entry(word, term, weight=float(weight.detach()))
                            for term, weight in zip(
                                beam.terms, beam.log_weights, strict=True
                            )
                        )
                        for word, beam in zip(words, found.beams, strict=True)
                    )
                )
                (counts,) = derivation_count_report(
                    (grammar.rules,),
                    (words,),
                    source=source,
                    goal=self._goal,
                    unary_depth=grammar.budget.unary_rounds,
                    threads=grammar.budget.threads,
                    proposer=self._proposer.build(source),
                )
                normal_form_count: int | None = counts.normal_form_count
                unconstrained_count: int | None = counts.unconstrained_count
                retention = counts.retention_ratio
            else:
                normal_form_count = None
                unconstrained_count = None
                retention = None
            if derivation is None:
                bracketing = None
                attachment = None
                dependency = None
            else:
                root, arcs = _pred_links(
                    derivation, self._heads, tuple(words), self._render
                )
                bracketing = _forest_bracketing(derivation)
                attachment = _tree(tuple(words), root, arcs)
                dependency = DependencyPrediction(arcs)
            predictions[item_id] = StructuralPrediction(
                bracketing,
                attachment,
                dependency,
                normal_form_count,
                unconstrained_count,
                retention,
            )
        return MappingProxyType(predictions)

    @override
    def __repr__(self) -> str:
        return f"StructuralDiagnosticsEvaluator(metrics={len(self._metrics)})"


def _build(
    settings: StructuralDiagnosticsSettings,
    context: HeldOutEvaluationContext,
    /,
) -> StructuralDiagnosticsEvaluator:
    return StructuralDiagnosticsEvaluator(context, settings)


PLUGIN: Final = ComponentPlugin(spec=StructuralDiagnosticsSpec, build=_build)
