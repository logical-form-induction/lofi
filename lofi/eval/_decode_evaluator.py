"""Held-out neural term beams and supertag-factored A* diagnostics."""

from math import fsum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, cast, override

from lofi._checks import ensure_name
from lofi._types import ReadoutId
from lofi.deduce import (
    AStarContext,
    AStarOptions,
    AStarResult,
    UnitEdges,
    astar_parse,
)
from lofi.eval._held_out import derivation_dataset
from lofi.eval._matched import InducedBracketing
from lofi.eval._values import ConventionVector, MetricResult, Resampling
from lofi.lexicon import BeamSource, Entry
from lofi.registry import ComponentPlugin
from lofi.spec.evaluators.decode import (
    DecodeEvaluatorSettings,
    DecodeEvaluatorSpec,
    DerivationDecision,
    derivation_nodes,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi._types import ItemId
    from lofi.data import Derivation, EvalDataset, Item
    from lofi.deduce import Cell, ParseSettings
    from lofi.deduce import Derivation as ForestDerivation
    from lofi.eval import HeldOutEvaluationContext, HeldOutItems
    from lofi.eval._ports import HeldOutData
    from lofi.judge import Rule
    from lofi.nn.forms import WeightedTerms
    from lofi.nn.loss._values import (
        ObjectiveOutputs,
        SpanTerms,
    )
    from lofi.terms import Term

__all__ = ["PLUGIN", "DecodeEvaluator", "DecodePrediction"]


class DecodePrediction(NamedTuple):
    """One item's published term beams and optional symbolic decode."""

    beams: tuple[WeightedTerms, ...]
    search: AStarResult[Term] | None
    derivation: ForestDerivation[Term, float] | None


class _UnitScorer:
    """Leave neural lexical weights unchanged and assign unit rule weights."""

    @property
    def lowering(self) -> UnitEdges:
        """Return the native unit-edge lowering."""
        return UnitEdges()

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> float:
        """Return the log space multiplicative identity."""
        del position, entry
        return 0.0

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        """Return the log space multiplicative identity."""
        del rule, inputs, output, span, split
        return 0.0


def _source(
    words: tuple[str, ...],
    decoded: SpanTerms,
    item: int,
    /,
) -> tuple[BeamSource[Term], tuple[WeightedTerms, ...]]:
    """Return one lexical source and its published per-token beams."""
    beams = tuple(
        decoded.at(item, decoded.spans.span_id(position, position + 1))
        for position in range(len(words))
    )
    source = BeamSource(
        tuple(
            tuple(
                Entry(word, term, weight=float(weight.detach()))
                for term, weight in zip(
                    beam.terms,
                    beam.log_weights,
                    strict=True,
                )
            )
            for word, beam in zip(words, beams, strict=True)
        )
    )

    return source, beams


def _candidates(prediction: DecodePrediction, /) -> float:
    """Return how many candidate terms the published beams hold."""
    return float(sum(len(beam.terms) for beam in prediction.beams))


def _searched(prediction: DecodePrediction, name: str, /) -> AStarResult[Term]:
    """Return the A* result a search row reads, refusing a beam-only prediction."""
    search = prediction.search
    if search is None:
        msg = f"the metric {name!r} requires an A* result"
        raise ValueError(msg)
    return search


def _parsed(prediction: DecodePrediction, /) -> float:
    """Return whether the search parsed the sentence."""
    return float(_searched(prediction, "decode-parsed").parsed)


def _refused(prediction: DecodePrediction, /) -> float:
    """Return whether the search parsed nothing."""
    return float(not _searched(prediction, "decode-refused").parsed)


def _attempts(prediction: DecodePrediction, /) -> float:
    """Return how many searches were attempted."""
    return float(_searched(prediction, "decode-attempts").attempts)


def _widenings(prediction: DecodePrediction, /) -> float:
    """Return how many times the supertag threshold widened."""
    return float(_searched(prediction, "decode-widenings").widenings)


type _Reading = Callable[[DecodePrediction], float]
"""Reads one diagnostic scalar off one item's decode."""

_BEAM_ROWS: Final[tuple[tuple[str, _Reading], ...]] = (
    ("decode-candidates", _candidates),
)
"""The rows every decode reports."""

_SEARCH_ROWS: Final[tuple[tuple[str, _Reading], ...]] = (
    ("decode-parsed", _parsed),
    ("decode-refused", _refused),
    ("decode-attempts", _attempts),
    ("decode-widenings", _widenings),
)
"""The rows a declared A* search adds."""


class _DecodeMetric:
    """Accumulate one scalar diagnostic from decode predictions."""

    __slots__ = ("_conventions", "_items", "_name", "_reading")

    def __init__(
        self, name: str, reading: _Reading, conventions: ConventionVector
    ) -> None:
        self._name = ensure_name("a decode metric", name)
        self._reading = reading
        self._conventions = conventions
        self._items: dict[ItemId, float] = {}

    @property
    def name(self) -> str:
        """Return the report-row name."""
        return self._name

    @property
    def conventions(self) -> ConventionVector:
        """Return the conventions this row is reported under."""
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        """Return that item order does not affect the mean."""
        return True

    def reset(self) -> None:
        """Forget every accumulated item."""
        self._items = {}

    def _value(self, prediction: DecodePrediction) -> float:
        """Return this metric's scalar for one prediction."""
        return self._reading(prediction)

    def update(self, pred: DecodePrediction, gold: ItemId, *, key: ItemId) -> None:
        """Accumulate one item's declared decode diagnostic."""
        del gold
        value = self._value(pred)
        held = self._items.get(key)

        if held is not None and held != value:
            msg = f"decode evaluation already saw another prediction for {key!r}"
            raise ValueError(msg)

        self._items[key] = value

    def compute(self) -> MetricResult:
        """Return the arithmetic mean across held-out sentences."""
        if not self._items:
            return MetricResult(None, 0, Resampling("sentence"))

        return MetricResult(
            fsum(self._items.values()) / len(self._items),
            len(self._items),
            Resampling("sentence"),
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return one scalar result per held-out sentence."""
        return MappingProxyType(
            {
                item_id: MetricResult(value, 1, Resampling("sentence"))
                for item_id, value in sorted(self._items.items())
            }
        )


class DecodeEvaluator:
    """Read shared neural term beams and optionally parse them with A*."""

    __slots__ = (
        "_astar",
        "_context",
        "_decode",
        "_metrics",
        "_output",
        "_readout",
    )

    def __init__(
        self,
        context: HeldOutEvaluationContext,
        /,
        *,
        readout: str,
        output: str,
        astar: AStarOptions | None,
        decode: DerivationDecision | None,
        conventions: ConventionVector,
    ) -> None:
        grammar = context.grammar.grammar

        if astar is not None and not grammar.profile.lexicalized:
            msg = (
                f"A* supertag decoding requires a lexicalized profile, not "
                f"{grammar.profile!r}"
            )
            raise ValueError(msg)

        self._context = context
        self._readout = ReadoutId(
            ensure_name("a decode evaluator's term-readout role", readout)
        )
        self._output = output
        self._astar = astar
        self._decode = decode
        declared = conventions.extended(
            {
                "decode_readout": readout,
                "decode_output": output,
                "astar_strategy": None if astar is None else astar.strategy.name,
                "astar_root_mode": None if astar is None else astar.root_mode.name,
                "parse_semiring": (
                    None if astar is None else context.parse.semiring.name
                ),
                "astar_heuristic": None if astar is None else astar.heuristic.name,
                "astar_proposer": (
                    None
                    if astar is None or astar.proposer is None
                    else astar.proposer.name
                ),
                **(
                    dict.fromkeys(
                        ("decode_method", "decode_utility", "decode_candidates")
                    )
                    if decode is None
                    else decode.conventions
                ),
            }
        )
        rows = _BEAM_ROWS if astar is None else (*_BEAM_ROWS, *_SEARCH_ROWS)
        self._metrics = tuple(
            _DecodeMetric(name, reading, declared) for name, reading in rows
        )

    @property
    def conventions(self) -> ConventionVector:
        """Return the conventions every diagnostic row carries."""
        return self._metrics[0].conventions

    @property
    def metrics(self) -> tuple[_DecodeMetric, ...]:
        """Return decode diagnostics in declaration order."""
        return self._metrics

    def held_out(self, data: HeldOutData, split: str, /) -> EvalDataset[ItemId]:
        """Return a retained partition's derivations as this evaluator reads them."""
        return derivation_dataset(self.annotate, data.derivations(split))

    def annotate(self, item: Item, derivation: Derivation[str], /) -> ItemId:
        """Return the item identifier used to count this held-out sentence."""
        del derivation
        return item.item_id

    def labeled_spans(
        self, prediction: DecodePrediction, /
    ) -> InducedBracketing | None:
        """Return the spans of the selected derivation under their categories.

        Each span is labeled by the category that concludes it, rendered in
        the grammar's own syntax, and a span a unary chain covers by the
        chain's highest node.

        Parameters
        ----------
        prediction : DecodePrediction
            One item's prediction.

        Returns
        -------
        InducedBracketing | None
            The labeled spans, or None when no derivation was selected: the
            evaluator declares no A* search, or the search parsed nothing.
        """
        derivation = prediction.derivation

        if derivation is None:
            return None

        grammar = self._context.grammar
        labels: dict[tuple[int, int], str] = {}

        for node in derivation_nodes(derivation):
            labels.setdefault(
                node.key.span, grammar.syntax.render(grammar.store, node.key.reading)
            )

        return InducedBracketing(derivation.key.span[1], labels)

    def predict[BatchT: HeldOutItems](
        self,
        outputs: ObjectiveOutputs[BatchT],
        /,
    ) -> Mapping[ItemId, DecodePrediction]:
        """Return published beams and optional A* parses by item identifier."""
        decoded = outputs.terms.get(self._readout)

        if decoded is None:
            msg = (
                f"decode evaluation reads complete terms {self._readout!r}, and "
                f"the pass published {sorted(outputs.terms)}"
            )
            raise ValueError(msg)

        grammar = self._context.grammar.grammar
        astar = self._astar
        decode = self._decode
        predictions: dict[ItemId, DecodePrediction] = {}

        for item, (item_id, words) in enumerate(
            zip(outputs.batch.item_ids, outputs.batch.words, strict=True)
        ):
            if astar is None:
                root = decoded.spans.span_id(0, len(words))
                predictions[item_id] = DecodePrediction(
                    (decoded.at(item, root),),
                    None,
                    None,
                )
                continue

            if decode is None:
                msg = "an A* evaluator resolved without a derivation decode"
                raise ValueError(msg)

            source, beams = _source(tuple(words), decoded, item)
            context = AStarContext(
                grammar.rules,
                _UnitScorer(),
                grammar.goal,
                grammar.budget.unary_rounds,
                grammar.budget.threads,
                cast("ParseSettings[float]", self._context.parse),
            )
            result = astar_parse(tuple(words), source, context, astar)
            predictions[item_id] = DecodePrediction(
                beams,
                result,
                decode.select(result, context),
            )

        return MappingProxyType(predictions)

    @override
    def __repr__(self) -> str:
        """Return a summary naming the readout and output kind."""
        return (
            f"DecodeEvaluator(readout={self._readout!r}, "
            f"output={self._output!r}, astar={self._astar is not None!r})"
        )


def _build(
    settings: DecodeEvaluatorSettings,
    context: HeldOutEvaluationContext,
    /,
) -> DecodeEvaluator:
    """Build the declared evaluator in its held-out grammar context."""
    return DecodeEvaluator(
        context,
        readout=settings.readout,
        output=settings.output,
        astar=settings.astar,
        decode=settings.decode,
        conventions=ConventionVector.from_settings(settings.conventions),
    )


PLUGIN: Final = ComponentPlugin(spec=DecodeEvaluatorSpec, build=_build)
