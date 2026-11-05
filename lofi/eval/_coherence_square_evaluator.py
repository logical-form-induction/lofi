"""Coherence-square reports over derivation-specific category/form pairs."""

from hashlib import sha256
from math import fsum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, override

from lofi._checks import LawViolationError
from lofi.deduce import LEXICAL_TAG_PREFIX
from lofi.eval._decode_evaluator import DecodeEvaluator
from lofi.eval._held_out import derivation_dataset
from lofi.eval._values import ConventionVector, MetricResult, Resampling
from lofi.interpret import LexicalTable
from lofi.lexicon import Entry
from lofi.registry import ComponentPlugin
from lofi.spec.evaluators.coherence_square import (
    CoherenceSquareSettings,
    CoherenceSquareSpec,
)
from lofi.typecheck import CoherenceFixture, CoherenceItem, coherence_square

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lofi._types import ItemId
    from lofi.data import Derivation, EvalDataset, Item
    from lofi.data.synthetic import InterpretationFunction
    from lofi.deduce import Derivation as ForestDerivation
    from lofi.deduce import ItemKey
    from lofi.eval import HeldOutEvaluationContext, HeldOutItems
    from lofi.eval._ports import HeldOutData
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.terms import Term

__all__ = ["PLUGIN", "CoherenceSquareEvaluator"]


class CoherencePrediction(NamedTuple):
    """Counts obtained by checking one decoded forest."""

    selected: int
    coherent: int
    instances: int
    strict: int
    unique_forms: int


def _selected[T](
    keys: Sequence[ItemKey[T]],
    *,
    mode: str,
    count: int | None,
    seed: int,
    item_id: ItemId,
) -> tuple[ItemKey[T], ...]:
    """Return the exact chart items named by the declared policy."""
    if mode == "off":
        return ()
    if mode == "all":
        return tuple(keys)
    if count is None:
        msg = "sampled coherence resolved without an item count"
        raise ValueError(msg)
    ranked = sorted(
        keys,
        key=lambda key: sha256(f"{seed}\0{item_id}\0{key!r}".encode()).digest(),
    )
    return tuple(ranked[:count])


def _lexicon(
    words: Sequence[str], interpretation: InterpretationFunction
) -> tuple[tuple[Entry[Term, Term], ...], ...]:
    """Return every declared meaning of every surface entry by position."""
    entries = interpretation.entries
    return tuple(
        tuple(
            Entry(word, category, meaning)
            for (form, category), meanings in entries.items()
            if form == word
            for meaning in meanings
        )
        for word in words
    )


def _readable(
    derivation: ForestDerivation[Term, float],
    lexicon: Sequence[Sequence[Entry[Term, Term]]],
    /,
) -> bool:
    """Return whether every axiom has a declared meaning-bearing entry."""
    edge = derivation.edge
    if edge.rule_name.startswith(LEXICAL_TAG_PREFIX):
        position = derivation.key.span[0]
        if position >= len(lexicon):
            return False
        form = edge.rule_name.removeprefix(LEXICAL_TAG_PREFIX)
        return any(
            entry.form == form and entry.category == edge.output
            for entry in lexicon[position]
        )
    return all(_readable(child, lexicon) for child in derivation.children)


def _square_rate(prediction: CoherencePrediction, /) -> float | None:
    """Return the share of selected items whose square commutes."""
    if prediction.selected == 0:
        return None
    return prediction.coherent / prediction.selected


def _square_items(prediction: CoherencePrediction, /) -> float | None:
    """Return how many packed items were selected."""
    return float(prediction.selected)


def _square_instances(prediction: CoherencePrediction, /) -> float | None:
    """Return how many derivation instances were checked."""
    return float(prediction.instances)


def _square_strict_rate(prediction: CoherencePrediction, /) -> float | None:
    """Return the share of instances that commute strictly."""
    if prediction.instances == 0:
        return None
    return prediction.strict / prediction.instances


def _square_collapse_rate(prediction: CoherencePrediction, /) -> float | None:
    """Return the share of instances whose meanings collapse onto one form."""
    if prediction.instances == 0:
        return None
    return 1.0 - prediction.unique_forms / prediction.instances


type _Reading = Callable[[CoherencePrediction], float | None]
"""Reads one report row's scalar off one sentence's square checks."""

_READINGS: Final[Mapping[str, _Reading]] = MappingProxyType(
    {
        "coherence-square-rate": _square_rate,
        "coherence-square-items": _square_items,
        "coherence-square-instances": _square_instances,
        "coherence-square-strict-rate": _square_strict_rate,
        "coherence-square-collapse-rate": _square_collapse_rate,
    }
)
"""The scalar each declared coherence-square row reads."""


class _CoherenceMetric:
    """Accumulate one scalar from per-sentence square checks."""

    __slots__ = ("_conventions", "_items", "_name", "_reading")

    def __init__(self, name: str, conventions: ConventionVector) -> None:
        reading = _READINGS.get(name)
        if reading is None:
            msg = (
                f"no coherence-square row is named {name!r}; the rows are "
                f"{sorted(_READINGS)}"
            )
            raise ValueError(msg)
        self._name = name
        self._reading = reading
        self._conventions = conventions
        self._items: dict[ItemId, CoherencePrediction] = {}

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

    def update(self, pred: CoherencePrediction, gold: ItemId, *, key: ItemId) -> None:
        del gold
        standing = self._items.get(key)
        if standing is not None and standing != pred:
            msg = f"item {key!r} has two coherence-square reports"
            raise ValueError(msg)
        self._items[key] = pred

    def _value(self, prediction: CoherencePrediction) -> float | None:
        return self._reading(prediction)

    def compute(self) -> MetricResult:
        values = tuple(
            value
            for prediction in self._items.values()
            if (value := self._value(prediction)) is not None
        )
        return MetricResult(
            None if not values else fsum(values) / len(values),
            len(values),
            Resampling("sentence"),
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        return MappingProxyType(
            {
                key: MetricResult(value, 1, Resampling("sentence"))
                for key, prediction in self._items.items()
                if (value := self._value(prediction)) is not None
            }
        )


class CoherenceSquareEvaluator:
    """Check every selected packed item with its own derivation meanings."""

    __slots__ = (
        "_conventions",
        "_decode",
        "_interpretation",
        "_metrics",
        "_mode",
        "_rules",
        "_sample_items",
        "_seed",
    )

    def __init__(
        self,
        context: HeldOutEvaluationContext,
        settings: CoherenceSquareSettings,
        /,
    ) -> None:
        interpretation = context.interpretation
        if interpretation is None:
            msg = "coherence-square evaluation requires a declared interpretation"
            raise ValueError(msg)
        self._interpretation = interpretation
        self._mode = settings.mode
        self._sample_items = settings.sample_items
        self._seed = settings.seed
        self._rules = {rule.name: rule for rule in context.grammar.grammar.rules.rules}
        self._conventions = ConventionVector.from_settings(
            settings.conventions
        ).extended(
            {
                "coherence_mode": settings.mode,
                "coherence_sample_items": settings.sample_items,
            }
        )
        self._decode = DecodeEvaluator(
            context,
            readout=settings.readout,
            output="lexical-types",
            astar=settings.astar,
            decode=settings.decode,
            conventions=self._conventions,
        )
        self._metrics = tuple(
            _CoherenceMetric(name, self._conventions) for name in settings.metrics
        )

    @property
    def conventions(self) -> ConventionVector:
        return self._conventions

    @property
    def metrics(self) -> tuple[_CoherenceMetric, ...]:
        return self._metrics

    def held_out(self, data: HeldOutData, split: str, /) -> EvalDataset[ItemId]:
        """Return a retained partition's derivations as this evaluator reads them."""
        return derivation_dataset(self.annotate, data.derivations(split))

    def annotate(self, item: Item, derivation: Derivation[str], /) -> ItemId:
        del derivation
        return item.item_id

    def predict[BatchT: HeldOutItems](
        self, outputs: ObjectiveOutputs[BatchT], /
    ) -> Mapping[ItemId, CoherencePrediction]:
        decoded = self._decode.predict(outputs)
        interpretation = self._interpretation
        language = interpretation.language
        predictions: dict[ItemId, CoherencePrediction] = {}
        for item_id, words in zip(
            outputs.batch.item_ids, outputs.batch.words, strict=True
        ):
            search = decoded[item_id].search
            if search is None or self._mode == "off":
                predictions[item_id] = CoherencePrediction(0, 0, 0, 0, 0)
                continue
            chart = search.chart
            keys = tuple(
                key
                for span in chart.index.spans_by_length()
                for key in chart.items_at(span)
            )
            selected = _selected(
                keys,
                mode=self._mode,
                count=self._sample_items,
                seed=self._seed,
                item_id=item_id,
            )
            lexical = _lexicon(words, interpretation)
            table = LexicalTable(self._rules, lexical)
            coherent = 0
            instances = 0
            strict = 0
            unique_forms = 0
            for key in selected:
                logical_forms = tuple(
                    meaning
                    for forest in chart.derivations(key)
                    if _readable(forest, lexical)
                    for meaning in interpretation.read(forest, table)
                )
                item = CoherenceItem(key.span, key.reading, logical_forms, key.nf_tag)
                try:
                    report = coherence_square(
                        CoherenceFixture(
                            (item,),
                            language.checker,
                            interpretation.signature,
                            language.rules,
                            interpretation.erasure,
                        )
                    )
                except LawViolationError:
                    continue
                coherent += 1
                instances += len(report.instances)
                strict += len(report.strict)
                unique_forms += len(
                    {instance.logical_form for instance in report.instances}
                )
            predictions[item_id] = CoherencePrediction(
                len(selected), coherent, instances, strict, unique_forms
            )
        return MappingProxyType(predictions)

    @override
    def __repr__(self) -> str:
        return f"CoherenceSquareEvaluator(mode={self._mode!r})"


def _build(
    settings: CoherenceSquareSettings,
    context: HeldOutEvaluationContext,
    /,
) -> CoherenceSquareEvaluator:
    return CoherenceSquareEvaluator(context, settings)


PLUGIN: Final = ComponentPlugin(spec=CoherenceSquareSpec, build=_build)
