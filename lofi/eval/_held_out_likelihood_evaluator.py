"""Held-out string and derivational likelihood over a trained parser."""

from collections.abc import Hashable
from math import fsum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, cast, final

from lofi._types import ReadoutId
from lofi.data import Derivation, DerivationNode
from lofi.eval._derivational import (
    DerivationalLikelihood,
    DerivationalScoring,
    LikelihoodUnit,
    SentenceLikelihood,
)
from lofi.eval._held_out import derivation_dataset, surface_derivation_dataset
from lofi.eval._values import ConventionVector, MetricResult, Resampling
from lofi.nn.loss._description_cost import description_cost
from lofi.nn.loss._readout import (
    MASS_AXES,
    log_mass,
)
from lofi.nn.loss._simplicity import (
    expected_description_lengths,
)
from lofi.registry import ComponentPlugin
from lofi.spec.evaluators.held_out_likelihood import (
    HeldOutLikelihoodSettings,
    HeldOutLikelihoodSpec,
)
from lofi.spec.losses.simplicity import SimplicityScope

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import ItemId, JsonValue
    from lofi.data import EvalDataset, Item
    from lofi.eval import HeldOutEvaluationContext, HeldOutItems
    from lofi.eval._ports import HeldOutData
    from lofi.nn._values import DerivationScoreChart, DerivationSetScore
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.terms import Term

__all__ = ["PLUGIN", "HeldOutLikelihoodEvaluator"]


@final
class _LikelihoodPrediction:
    """One item's root mass, description length and derivation-score chart."""

    __slots__ = (
        "_chart",
        "_description_cells",
        "_description_length",
        "_item",
        "_string_score",
    )

    def __init__(
        self,
        item: int,
        string_score: float | None,
        description_length: float | None,
        description_cells: int,
        chart: DerivationScoreChart[Term] | None,
        /,
    ) -> None:
        self._item = item
        self._string_score = string_score
        self._description_length = description_length
        self._description_cells = description_cells
        self._chart = chart

    @property
    def string_score(self) -> float:
        """Return the log root target mass."""
        score = self._string_score
        if score is None:
            msg = "this prediction carries no string-likelihood score"
            raise ValueError(msg)
        return score

    @property
    def description_length(self) -> float | None:
        """Return total lexical description length, or None with no priced cell."""
        return self._description_length

    @property
    def description_cells(self) -> int:
        """Return how many lexical cells contribute to the description length."""
        return self._description_cells

    def score(self, gold: Derivation[Term], /) -> tuple[DerivationSetScore, ...]:
        """Return constrained and unrestricted gold derivation scores."""
        chart = self._chart
        if chart is None:
            msg = "this prediction carries no derivational score chart"
            raise ValueError(msg)
        scores = chart.score(self._item, gold)

        return (scores.constrained, scores.unconstrained)


class _StringLikelihoodMetric:
    """Accumulate root target mass log likelihood under one reporting unit."""

    __slots__ = ("_conventions", "_likelihood", "_name", "_unit")

    def __init__(self, unit: LikelihoodUnit, conventions: ConventionVector) -> None:
        self._unit = unit
        self._name = f"string-log-likelihood-per-{unit}"
        self._conventions = conventions
        self._likelihood: DerivationalLikelihood[Hashable] = DerivationalLikelihood(
            "string"
        )

    @property
    def name(self) -> str:
        """Return the unit-specific string-likelihood metric name."""
        return self._name

    @property
    def conventions(self) -> ConventionVector:
        """Return the evaluator's held-out scoring conventions."""
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        """Return that update order does not affect the aggregate."""
        return True

    def reset(self) -> None:
        """Forget every accumulated item."""
        self._likelihood.reset()

    def update[L: Hashable](
        self,
        pred: _LikelihoodPrediction,
        gold: Derivation[L],
        *,
        key: ItemId,
    ) -> None:
        """Accumulate one item's log root target mass."""
        self._likelihood.update(
            SentenceLikelihood(pred.string_score, 0.0),
            cast("Derivation[Hashable]", gold),
            key=key,
        )

    def compute(self) -> MetricResult:
        """Return the mean log root target mass under the declared unit."""
        return self._likelihood.result(self._unit)

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return each sentence's log root target mass."""
        return self._likelihood.per_item()


class _DerivationalLikelihoodMetric:
    """Score and accumulate one normal form side under one reporting unit."""

    __slots__ = ("_conventions", "_index", "_likelihood", "_name", "_unit")

    def __init__(
        self,
        likelihood: DerivationalLikelihood[Term],
        unit: LikelihoodUnit,
        conventions: ConventionVector,
        /,
        *,
        index: Literal[0, 1],
    ) -> None:
        self._likelihood = likelihood
        self._unit = unit
        self._index = index
        self._name = f"derivational-log-likelihood-per-{unit}"
        self._conventions = conventions

    @property
    def name(self) -> str:
        """Return the unit-specific derivational-likelihood metric name."""
        return self._name

    @property
    def normal_form(self) -> str:
        """Return the filter defining this metric's derivation set."""
        return self._likelihood.normal_form

    @property
    def conventions(self) -> ConventionVector:
        """Return the evaluator's held-out scoring conventions."""
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        """Return that update order does not affect the aggregate."""
        return True

    def reset(self) -> None:
        """Forget every accumulated item."""
        self._likelihood.reset()

    def update(
        self,
        pred: _LikelihoodPrediction,
        gold: Derivation[Term],
        *,
        key: ItemId,
    ) -> None:
        """Score and accumulate one gold derivation."""
        scored = pred.score(gold)[self._index]

        if scored.normal_form != self._likelihood.normal_form:
            msg = (
                f"the chart scored normal form {scored.normal_form!r}, not the "
                f"metric's {self._likelihood.normal_form!r}"
            )
            raise ValueError(msg)

        self._likelihood.update(
            SentenceLikelihood(scored.score, scored.log_partition), gold, key=key
        )

    def compute(self) -> MetricResult:
        """Return held-out likelihood under the declared unit."""
        return self._likelihood.result(self._unit)

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return each held-out sentence's log probability."""
        return self._likelihood.per_item()


_DESCRIPTION_RESAMPLING: Final = Resampling("sentence")
"""Description length is resampled by held-out item."""


class _DescriptionLengthMetric:
    """Accumulate exact expected lexical constructor cost as a corpus total."""

    __slots__ = ("_conventions", "_items")

    def __init__(self, conventions: ConventionVector) -> None:
        self._conventions = conventions
        self._items: dict[ItemId, tuple[float | None, int]] = {}

    @property
    def name(self) -> str:
        """Return the source comparison description length row name."""
        return "description-length-nats"

    @property
    def conventions(self) -> ConventionVector:
        """Return the evaluator's held-out scoring conventions."""
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        """Return that item order does not affect the exact sum."""
        return True

    def reset(self) -> None:
        """Forget every accumulated item."""
        self._items = {}

    def update[L: Hashable](
        self,
        pred: _LikelihoodPrediction,
        gold: Derivation[L],
        *,
        key: ItemId,
    ) -> None:
        """Accumulate one item's expected lexical constructor cost."""
        del gold
        scored = (pred.description_length, pred.description_cells)
        standing = self._items.get(key)

        if standing is not None:
            if standing != scored:
                msg = f"the item {key!r} already has a different description length"
                raise ValueError(msg)

            return

        self._items[key] = scored

    def compute(self) -> MetricResult:
        """Return total expected lexical constructor cost over eligible items."""
        eligible = tuple(
            (length, cells)
            for length, cells in self._items.values()
            if length is not None
        )
        components = {"cells": float(sum(cells for _, cells in eligible))}

        if not eligible:
            return MetricResult(
                None,
                0,
                _DESCRIPTION_RESAMPLING,
                components=components,
            )

        return MetricResult(
            fsum(length for length, _ in eligible if length is not None),
            len(eligible),
            _DESCRIPTION_RESAMPLING,
            components=components,
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return each item's total for item-level bootstrap resampling."""
        return MappingProxyType(
            {
                key: MetricResult(
                    length,
                    0 if length is None else 1,
                    _DESCRIPTION_RESAMPLING,
                    components={"cells": float(cells)},
                )
                for key, (length, cells) in self._items.items()
            }
        )


class HeldOutLikelihoodEvaluator:
    """Score root target mass and gold derivations on held-out items."""

    __slots__ = (
        "_codec",
        "_conventions",
        "_derivation",
        "_description",
        "_description_cost",
        "_grammar",
        "_metrics",
        "_readout",
        "_string",
        "_term_readout",
    )

    def __init__(
        self,
        context: HeldOutEvaluationContext,
        settings: HeldOutLikelihoodSettings,
        /,
    ) -> None:
        self._grammar = context.grammar
        self._readout = ReadoutId(settings.output)
        self._term_readout = ReadoutId(settings.readout)
        language = context.grammar.language
        self._codec = language.codec.at(language.category_sort)
        self._description_cost = description_cost(settings.symbol_cost)
        factors: dict[str, JsonValue] = {
            "readout": settings.readout,
            "readout_output": settings.output,
            "description_length_scope": "lexicon",
            "symbol_cost": str(settings.symbol_cost.kind),
            "measurements": ",".join(settings.measurements),
        }
        self._conventions = ConventionVector.from_settings(
            settings.conventions
        ).extended(factors)
        selected = frozenset(settings.measurements)
        self._string = "string" in selected
        self._derivation = "derivation" in selected
        self._description = "description-length" in selected
        metrics: list[
            _StringLikelihoodMetric
            | _DerivationalLikelihoodMetric
            | _DescriptionLengthMetric
        ] = []
        if self._string:
            metrics.extend(
                (
                    _StringLikelihoodMetric("sentence", self._conventions),
                    _StringLikelihoodMetric("token", self._conventions),
                )
            )
        if self._derivation:
            sentence: DerivationalScoring[Term] = DerivationalScoring(
                context.grammar.grammar.normal_form
            )
            token: DerivationalScoring[Term] = DerivationalScoring(
                context.grammar.grammar.normal_form
            )
            metrics.extend(
                (
                    _DerivationalLikelihoodMetric(
                        sentence.constrained,
                        "sentence",
                        self._conventions,
                        index=0,
                    ),
                    _DerivationalLikelihoodMetric(
                        token.constrained, "token", self._conventions, index=0
                    ),
                    _DerivationalLikelihoodMetric(
                        sentence.unconstrained,
                        "sentence",
                        self._conventions,
                        index=1,
                    ),
                    _DerivationalLikelihoodMetric(
                        token.unconstrained, "token", self._conventions, index=1
                    ),
                )
            )
        if self._description:
            metrics.append(_DescriptionLengthMetric(self._conventions))
        self._metrics = tuple(metrics)

    @property
    def conventions(self) -> ConventionVector:
        """Return the conventions shared by every likelihood row."""
        return self._conventions

    @property
    def metrics(
        self,
    ) -> tuple[
        _StringLikelihoodMetric
        | _DerivationalLikelihoodMetric
        | _DescriptionLengthMetric,
        ...,
    ]:
        """Return string and derivational metrics in report order."""
        return self._metrics

    def held_out(
        self, data: HeldOutData, split: str, /
    ) -> EvalDataset[Derivation[Term]] | EvalDataset[Derivation[str]]:
        """Return a retained partition with translated or surface derivations.

        Derivational likelihood reads gold derivations translated into learner
        terms; string likelihood and description length read them as surfaces.
        """
        examples = data.derivations(split)

        if self._derivation:
            return derivation_dataset(self.annotate, examples)

        return surface_derivation_dataset(examples)

    def annotate(self, item: Item, derivation: Derivation[str], /) -> Derivation[Term]:
        """Translate a gold derivation's category texts into learner terms."""
        del item
        grammar = self._grammar
        nodes = {
            node_id: DerivationNode(
                node.span,
                grammar.syntax.parse(grammar.store, node.label),
                rule=node.rule,
                children=node.children,
                features=node.features,
            )
            for node_id, node in derivation.nodes.items()
        }
        confidences = {
            node_id: confidence
            for node_id in derivation.nodes
            if (confidence := derivation.confidence_of(node_id)) is not None
        }

        return Derivation(derivation.root, nodes, confidences)

    def predict[BatchT: HeldOutItems](
        self, outputs: ObjectiveOutputs[BatchT], /
    ) -> Mapping[ItemId, _LikelihoodPrediction]:
        """Return predictions without depending on prior gold annotation."""
        count = len(outputs.batch.item_ids)

        masses: list[float | None] = [None] * count
        if self._string:
            readout = outputs.readouts.get(self._readout)
            if readout is None:
                offered = ", ".join(sorted(outputs.readouts)) or "nothing"
                msg = (
                    f"held-out likelihood reads {self._readout!r}, and the pass "
                    f"published {offered}"
                )
                raise ValueError(msg)
            if readout.axes != MASS_AXES:
                msg = (
                    f"held-out likelihood reads a mass pair over {MASS_AXES}, not "
                    f"{readout.axes}"
                )
                raise ValueError(msg)
            cells = readout.gather(
                [(row, 0, len(words)) for row, words in enumerate(outputs.batch.words)]
            )
            masses = log_mass(cells).detach().cpu().tolist()

        term_derivations: DerivationScoreChart[Term] | None = None
        if self._derivation:
            derivations = outputs.derivations
            if derivations is None:
                msg = (
                    "held-out derivational likelihood needs a tagged log-sum-exp chart"
                )
                raise ValueError(msg)
            term_derivations = cast("DerivationScoreChart[Term]", derivations)

        totals: list[float | None] = [None] * count
        counts: list[int] = [0] * count
        if self._description:
            term_readout = outputs.readouts.get(self._term_readout)
            if term_readout is None:
                offered = ", ".join(sorted(outputs.readouts)) or "nothing"
                msg = (
                    f"held-out description length reads {self._term_readout!r}, "
                    f"and the pass published {offered}"
                )
                raise ValueError(msg)
            lengths, priced = expected_description_lengths(
                term_readout,
                outputs.spans,
                self._codec,
                SimplicityScope.LEXICON,
                self._description_cost,
            )
            per_item_lengths = lengths.masked_fill(~priced, 0.0).sum(dim=1)
            per_item_cells = priced.sum(dim=1)
            raw_totals = per_item_lengths.detach().cpu().tolist()
            counts = per_item_cells.detach().cpu().tolist()
            totals = [
                None if cells == 0 else total
                for total, cells in zip(raw_totals, counts, strict=True)
            ]

        return MappingProxyType(
            {
                item_id: _LikelihoodPrediction(
                    row,
                    mass,
                    total,
                    cell_count,
                    term_derivations,
                )
                for row, (item_id, mass, total, cell_count) in enumerate(
                    zip(outputs.batch.item_ids, masses, totals, counts, strict=True)
                )
            }
        )


def _build(
    settings: HeldOutLikelihoodSettings,
    context: HeldOutEvaluationContext,
    /,
) -> HeldOutLikelihoodEvaluator:
    """Build held-out likelihood from its declared settings and grammar."""
    return HeldOutLikelihoodEvaluator(context, settings)


PLUGIN: Final = ComponentPlugin(spec=HeldOutLikelihoodSpec, build=_build)
