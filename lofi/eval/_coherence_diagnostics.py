"""Held-out coherence, entropy, collapse, usage, and mass diagnostics."""

from collections import Counter
from math import exp, fsum, isfinite, log
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, cast, override

import torch

from lofi._stats import spearman
from lofi._types import ReadoutId
from lofi.deduce import outside_from_edges, target_mass
from lofi.eval._decode_evaluator import DecodeEvaluator
from lofi.eval._held_out import derivation_dataset
from lofi.eval._values import ConventionVector, MetricResult, Resampling
from lofi.nn.loss._parser import (
    SPAN_AXES,
    SPLIT_AXES,
)
from lofi.nn.loss._ports import ScheduledBatch
from lofi.nn.loss._readout import log_mass
from lofi.registry import ComponentPlugin
from lofi.spec.evaluators.coherence_diagnostics import (
    CoherenceDiagnosticsSettings,
    CoherenceDiagnosticsSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import ItemId
    from lofi.data import Derivation, EvalDataset, Item
    from lofi.eval import HeldOutEvaluationContext, HeldOutItems
    from lofi.eval._ports import HeldOutData
    from lofi.nn.forms import WeightedTerms
    from lofi.nn.loss._ports import CoherenceEvaluation
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.terms import Matching, Term

__all__ = ["PLUGIN", "CoherenceDiagnosticsEvaluator"]


class _SpanPeak(NamedTuple):
    """One decoded span's entropy, top category, and top probability."""

    entropy: float
    top: str
    probability: float


class CoherenceDiagnosticPrediction(NamedTuple):
    """Detached diagnostics for one held-out sentence."""

    words: tuple[str, ...]
    token_tops: tuple[Term | None, ...]
    evaluation: CoherenceEvaluation
    spans: tuple[_SpanPeak, ...]
    controller_entropies: tuple[float, ...]
    predicted_actions: tuple[tuple[str, float], ...]
    vector_root_mass: float | None
    symbolic_root_mass: float | None


class CoherenceDiagnosticGold(NamedTuple):
    """Gold token categories and combinator uses for one sentence."""

    categories: tuple[Term, ...]
    actions: tuple[str, ...]


type _Pair = tuple[CoherenceDiagnosticPrediction, CoherenceDiagnosticGold]


def _posterior(beam: WeightedTerms, /) -> tuple[tuple[Term, float], ...]:
    """Return a normalized finite posterior in decoder order."""
    finite = tuple(
        (term, float(weight))
        for term, weight in zip(beam.terms, beam.log_weights.detach(), strict=True)
        if isfinite(float(weight))
    )
    if not finite:
        return ()
    peak = max(weight for _, weight in finite)
    masses = tuple(exp(weight - peak) for _, weight in finite)
    total = fsum(masses)
    return tuple(
        (term, mass / total) for (term, _), mass in zip(finite, masses, strict=True)
    )


def _span_peak(beam: WeightedTerms, /) -> _SpanPeak | None:
    """Return entropy and peak information for one finite decoded span."""
    posterior = _posterior(beam)
    if not posterior:
        return None
    term, probability = max(posterior, key=lambda pair: pair[1])
    entropy = -fsum(mass * log(mass) for _, mass in posterior if mass > 0.0)
    return _SpanPeak(entropy, repr(term.key()), probability)


def _expanded_action(
    tensor: torch.Tensor,
    axes: tuple[str, ...],
    *,
    spans: int,
    splits: int,
) -> torch.Tensor:
    """Return one action's logits over item, span, and split."""
    if axes == SPAN_AXES:
        return tensor.unsqueeze(-1).expand(-1, spans, splits)
    if axes == SPLIT_AXES:
        return tensor
    msg = f"a coherence action runs over {SPAN_AXES} or {SPLIT_AXES}, not {axes}"
    raise ValueError(msg)


def _expanded_mask(
    mask: torch.Tensor | None,
    shape: tuple[int, int, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return one action's cell mask over the expanded split shape."""
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    widened = mask.reshape((*mask.shape, *(1 for _ in range(3 - mask.ndim))))
    return widened.expand(shape)


def _controller(
    outputs: ObjectiveOutputs[HeldOutItems], /
) -> tuple[tuple[tuple[float, ...], tuple[tuple[str, float], ...]], ...]:
    """Return per-item entropy samples and predicted rule masses."""
    actions = tuple(
        (action, weights)
        for action, weights in outputs.actions.items()
        if action.rule is not None
    )
    items = len(outputs.batch.item_ids)
    if not actions:
        return tuple(((), ()) for _ in range(items))
    spans = outputs.spans.num_spans
    splits = outputs.spans.max_splits
    shape = (items, spans, splits)
    logits = torch.stack(
        [
            _expanded_action(
                weights.tensor,
                weights.axes,
                spans=spans,
                splits=splits,
            )
            for _, weights in actions
        ]
    )
    masks = tuple(
        _expanded_mask(weights.mask, shape, device=logits.device)
        for _, weights in actions
    )
    valid = torch.tensor(
        outputs.spans.split_mask,
        dtype=torch.bool,
        device=logits.device,
    ).unsqueeze(0)
    valid = valid.expand(shape)
    if outputs.splits is not None:
        valid = valid & outputs.splits.valid
    if outputs.marginals is not None:
        valid = valid & outputs.marginals.built.unsqueeze(-1)
    for mask in masks:
        valid = valid & mask
    some = valid & torch.isfinite(torch.logsumexp(logits, dim=0))
    safe = torch.where(some.unsqueeze(0), logits, logits.new_zeros(()))
    probabilities = safe.softmax(dim=0)
    probabilities = torch.where(
        some.unsqueeze(0), probabilities, probabilities.new_zeros(())
    )
    floor = torch.finfo(probabilities.dtype).tiny
    entropy = -(probabilities * probabilities.clamp_min(floor).log()).sum(0)
    emitted: list[tuple[tuple[float, ...], tuple[tuple[str, float], ...]]] = []
    for item in range(items):
        selected = some[item]
        entropies = tuple(float(value) for value in entropy[item][selected].detach())
        masses: dict[str, float] = {}
        for column, (action, _) in enumerate(actions):
            if action.rule is None:
                continue
            masses[action.rule] = masses.get(action.rule, 0.0) + float(
                probabilities[column, item][selected].sum().detach()
            )
        emitted.append((entropies, tuple(sorted(masses.items()))))
    return tuple(emitted)


def _jsd(predicted: Mapping[str, float], gold: Mapping[str, float], /) -> float | None:
    """Return normalized Jensen-Shannon divergence over the union support."""
    predicted_total = fsum(predicted.values())
    gold_total = fsum(gold.values())
    if predicted_total == 0.0 or gold_total == 0.0:
        return None
    labels = set(predicted) | set(gold)
    divergence = 0.0
    for label in labels:
        left = predicted.get(label, 0.0) / predicted_total
        right = gold.get(label, 0.0) / gold_total
        middle = (left + right) / 2.0
        if left > 0.0:
            divergence += 0.5 * left * log(left / middle)
        if right > 0.0:
            divergence += 0.5 * right * log(right / middle)
    return divergence / log(2.0)


class _CoherenceMetric:
    """One reportable view of shared held-out coherence observations."""

    __slots__ = ("_conventions", "_items", "_matching", "_name", "_threshold")

    def __init__(
        self,
        name: str,
        conventions: ConventionVector,
        matching: Matching[Term],
        threshold: float,
    ) -> None:
        self._name = name
        self._conventions = conventions
        self._matching = matching
        self._threshold = threshold
        self._items: dict[ItemId, _Pair] = {}

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
        self,
        pred: CoherenceDiagnosticPrediction,
        gold: CoherenceDiagnosticGold,
        *,
        key: ItemId,
    ) -> None:
        if len(pred.words) != len(gold.categories):
            msg = f"coherence diagnostics disagree on token count for {key!r}"
            raise ValueError(msg)
        pair = (pred, gold)
        standing = self._items.get(key)
        if standing is not None and standing != pair:
            msg = f"coherence diagnostics already saw another prediction for {key!r}"
            raise ValueError(msg)
        self._items[key] = pair

    def _sentence_rows(self) -> tuple[tuple[float, float], ...]:
        rows: list[tuple[float, float]] = []
        matching = self._matching
        for key in sorted(self._items):
            prediction, gold = self._items[key]
            coherence = fsum(
                value for _, _, value in prediction.evaluation.contributions
            ) / len(prediction.words)
            correct = sum(
                predicted is not None and matching.equivalent(predicted, expected)
                for predicted, expected in zip(
                    prediction.token_tops, gold.categories, strict=True
                )
            )
            rows.append((coherence, correct / len(gold.categories)))
        return tuple(rows)

    def compute(self) -> MetricResult:
        name = self._name
        pairs = tuple(self._items[key] for key in sorted(self._items))
        sentence_rows = self._sentence_rows()
        components: dict[str, float] = {}
        unit = "sentence"

        if name == "coherence-loss-per-token":
            values = tuple(value for value, _ in sentence_rows)
        elif name in {"coherence-loss-by-action", "coherence-loss-by-span-length"}:
            grouped: dict[str, float] = {}
            for prediction, _ in pairs:
                for action, span_length, value in prediction.evaluation.contributions:
                    key = action if name.endswith("action") else str(span_length)
                    grouped[key] = grouped.get(key, 0.0) + value / len(prediction.words)
            denominator = max(1, len(pairs))
            prefix = "action" if name.endswith("action") else "span-length"
            components = {
                f"{prefix}:{key}": value / denominator
                for key, value in sorted(grouped.items())
            }
            values = tuple(value for value, _ in sentence_rows)
        elif name == "controller-entropy-per-span":
            values = tuple(
                value
                for prediction, _ in pairs
                for value in prediction.controller_entropies
            )
            unit = "span"
        elif name == "decoder-entropy-per-span":
            values = tuple(
                span.entropy for prediction, _ in pairs for span in prediction.spans
            )
            unit = "span"
        elif name == "type-distribution-collapse-rate":
            peaks = tuple(span for prediction, _ in pairs for span in prediction.spans)
            modal = Counter(span.top for span in peaks).most_common(1)
            if not modal:
                values = ()
            else:
                label = modal[0][0]
                values = tuple(
                    float(span.top == label and span.probability > self._threshold)
                    for span in peaks
                )
                components = {
                    "modal-count": float(modal[0][1]),
                    "spans": float(len(peaks)),
                }
            unit = "span"
        elif name == "coherence-accuracy-correlation":
            correlation = spearman(
                tuple(value for value, _ in sentence_rows),
                tuple(accuracy for _, accuracy in sentence_rows),
            )
            return MetricResult(
                correlation,
                len(sentence_rows),
                Resampling("sentence"),
            )
        elif name == "combinator-use-jsd":
            predicted: Counter[str] = Counter()
            golden: Counter[str] = Counter()
            for prediction, gold in pairs:
                predicted.update(dict(prediction.predicted_actions))
                golden.update(gold.actions)
            value = _jsd(predicted, golden)
            labels = sorted(set(predicted) | set(golden))
            components = {
                field: held
                for label in labels
                for field, held in (
                    (f"predicted:{label}", float(predicted[label])),
                    (f"gold:{label}", float(golden[label])),
                    (f"delta:{label}", float(predicted[label] - golden[label])),
                )
            }
            return MetricResult(
                value,
                sum(golden.values()),
                Resampling("rule"),
                components=components,
            )
        else:
            agreements = tuple(
                1.0 - abs(prediction.vector_root_mass - prediction.symbolic_root_mass)
                for prediction, _ in pairs
                if prediction.vector_root_mass is not None
                and prediction.symbolic_root_mass is not None
            )
            values = agreements
            components = {
                "mean-absolute-difference": (
                    0.0 if not agreements else 1.0 - fsum(agreements) / len(agreements)
                )
            }

        return MetricResult(
            None if not values else fsum(values) / len(values),
            len(values),
            Resampling(unit),
            components=components,
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        complete = self._items
        results: dict[ItemId, MetricResult] = {}
        for key, pair in sorted(complete.items()):
            self._items = {key: pair}
            results[key] = self.compute()
        self._items = complete
        return MappingProxyType(results)


class CoherenceDiagnosticsEvaluator:
    """Report held-out coherence from its objective and published parser outputs."""

    __slots__ = (
        "_context",
        "_decode",
        "_metrics",
        "_objective",
        "_readout",
        "_root_mass",
        "_targets",
    )

    def __init__(
        self,
        context: HeldOutEvaluationContext,
        settings: CoherenceDiagnosticsSettings,
        /,
    ) -> None:
        try:
            objective = context.coherence[settings.objective]
        except KeyError as missing:
            known = ", ".join(sorted(context.coherence)) or "no coherence objective"
            msg = f"coherence diagnostics name {settings.objective!r}; known: {known}"
            raise ValueError(msg) from missing
        try:
            targets = context.target_sets[ReadoutId(settings.root_mass)]
        except KeyError as missing:
            known = ", ".join(sorted(context.target_sets)) or "no target mass"
            msg = (
                f"coherence diagnostics name root mass {settings.root_mass!r}; "
                f"known: {known}"
            )
            raise ValueError(msg) from missing
        conventions = ConventionVector.from_settings(settings.conventions).extended(
            {
                "coherence_objective": settings.objective,
                "coherence_readout": settings.readout,
                "coherence_root_mass": settings.root_mass,
                "collapse_threshold": settings.collapse_threshold,
            }
        )
        self._context = context
        self._objective = objective
        self._readout = ReadoutId(settings.readout)
        self._root_mass = ReadoutId(settings.root_mass)
        self._targets = targets
        self._decode = DecodeEvaluator(
            context,
            readout=settings.readout,
            output="lexical-types",
            astar=settings.astar,
            decode=settings.decode,
            conventions=conventions,
        )
        matching = context.grammar.grammar.rules.matching
        self._metrics = tuple(
            _CoherenceMetric(
                name,
                conventions,
                matching,
                settings.collapse_threshold,
            )
            for name in settings.metrics
        )

    @property
    def conventions(self) -> ConventionVector:
        return self._metrics[0].conventions

    @property
    def metrics(self) -> tuple[_CoherenceMetric, ...]:
        return self._metrics

    def held_out(
        self, data: HeldOutData, split: str, /
    ) -> EvalDataset[CoherenceDiagnosticGold]:
        """Return a retained partition's derivations as this evaluator reads them."""
        return derivation_dataset(self.annotate, data.derivations(split))

    def annotate(
        self, item: Item, derivation: Derivation[str], /
    ) -> CoherenceDiagnosticGold:
        del item
        grammar = self._context.grammar
        leaves = derivation.leaves()
        categories = tuple(
            grammar.syntax.parse(grammar.store, leaf.label) for leaf in leaves
        )
        actions = tuple(
            node.rule
            for node in derivation.nodes.values()
            if node.children and node.rule is not None
        )
        return CoherenceDiagnosticGold(categories, actions)

    def predict[BatchT: HeldOutItems](
        self, outputs: ObjectiveOutputs[BatchT], /
    ) -> Mapping[ItemId, CoherenceDiagnosticPrediction]:
        decoded = outputs.terms.get(self._readout)
        if decoded is None:
            msg = (
                f"coherence diagnostics read {self._readout!r}, which was not published"
            )
            raise ValueError(msg)
        mass = outputs.readouts.get(self._root_mass)
        if mass is None:
            msg = (
                f"coherence diagnostics read {self._root_mass!r}, "
                "which was not published"
            )
            raise ValueError(msg)
        if not isinstance(outputs.batch, ScheduledBatch):
            msg = "coherence diagnostics require an evaluation step"
            raise TypeError(msg)
        evaluations = self._objective.evaluation(
            cast("ObjectiveOutputs[ScheduledBatch]", outputs)
        )
        if len(evaluations) != len(outputs.batch.item_ids):
            msg = "coherence evaluation count differs from the held-out batch"
            raise ValueError(msg)
        if tuple(evaluation.item for evaluation in evaluations) != tuple(
            range(len(evaluations))
        ):
            msg = "coherence evaluations do not follow held-out batch order"
            raise ValueError(msg)
        controllers = _controller(outputs)
        searches = self._decode.predict(outputs)
        grammar = self._context.grammar.grammar
        predictions: dict[ItemId, CoherenceDiagnosticPrediction] = {}
        for item, (item_id, words) in enumerate(
            zip(outputs.batch.item_ids, outputs.batch.words, strict=True)
        ):
            token_posteriors = tuple(
                _posterior(
                    decoded.at(item, decoded.spans.span_id(position, position + 1))
                )
                for position in range(len(words))
            )
            span_peaks = tuple(
                peak
                for (row, _), beam in decoded.cells.items()
                if row == item and (peak := _span_peak(beam)) is not None
            )
            root = decoded.spans.span_id(0, len(words))
            vector_mass = float(exp(float(log_mass(mass.distribution)[item, root])))
            search = searches[item_id].search
            symbolic_mass = None
            if search is not None:
                goal = search.options.root_mode.goal(grammar.goal)
                outside = outside_from_edges(search.chart, goal)
                symbolic_mass = target_mass(
                    search.chart,
                    outside,
                    self._targets,
                    decoded.spans.span_of(root),
                )
            controller_entropies, action_masses = controllers[item]
            predictions[item_id] = CoherenceDiagnosticPrediction(
                tuple(words),
                tuple(
                    posterior[0][0] if posterior else None
                    for posterior in token_posteriors
                ),
                evaluations[item],
                span_peaks,
                controller_entropies,
                action_masses,
                vector_mass,
                symbolic_mass,
            )
        return MappingProxyType(predictions)

    @override
    def __repr__(self) -> str:
        return f"CoherenceDiagnosticsEvaluator(objective={self._objective.name!r})"


def _build(
    settings: CoherenceDiagnosticsSettings,
    context: HeldOutEvaluationContext,
    /,
) -> CoherenceDiagnosticsEvaluator:
    """Build the declared coherence diagnostic evaluator."""
    return CoherenceDiagnosticsEvaluator(context, settings)


PLUGIN: Final = ComponentPlugin(spec=CoherenceDiagnosticsSpec, build=_build)
