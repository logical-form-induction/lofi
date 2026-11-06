"""Declared constructor-generic beam and reconstruction diagnostics."""

from collections import Counter
from json import dumps
from math import fsum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, override

from lofi._checks import ensure_name
from lofi._types import ReadoutId
from lofi.eval._alignment import Alignment, Contingency
from lofi.eval._beam import (
    BeamNormalization,
    ScoredBeam,
    accuracy_at_beam_size,
    depth_truncated_accuracy,
    mean_reciprocal_rank,
    term_depth,
    typeless_accuracy,
)
from lofi.eval._held_out import derivation_dataset
from lofi.eval._reconstruction import (
    TypeSystemReconstruction,
    type_system_reconstruction,
)
from lofi.eval._values import ConventionVector, MetricResult, Resampling
from lofi.registry import ComponentPlugin
from lofi.spec.evaluators.constructor_diagnostics import (
    ConstructorDiagnosticSettings,
    ConstructorDiagnosticSpec,
    ConstructorViewSettings,
)
from lofi.terms import DeclaredConstructors, Term, typeless

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lofi._types import ItemId
    from lofi.data import Derivation, EvalDataset, Item
    from lofi.eval import HeldOutEvaluationContext, HeldOutItems
    from lofi.eval._ports import HeldOutData
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.spec.grammar import GrammarAssembly
    from lofi.terms import TermStore, TermSyntax

__all__ = [
    "PLUGIN",
    "ConstructorDiagnosticEvaluator",
    "ConstructorDiagnosticGold",
    "ConstructorDiagnosticPrediction",
]


class ConstructorDiagnosticPrediction(NamedTuple):
    """The words and per-token type beams published for one item."""

    words: tuple[str, ...]
    beams: tuple[ScoredBeam, ...]


class ConstructorDiagnosticGold(NamedTuple):
    """The words and gold lexical categories of one item."""

    words: tuple[str, ...]
    categories: tuple[Term, ...]


class _ReportContext(NamedTuple):
    """Constructor and inventory facts shared by every report row."""

    store: TermStore
    syntax: TermSyntax
    constructors: DeclaredConstructors
    representative: Term
    primitives: tuple[str, ...]
    roles: tuple[str, ...]


type _Entry = tuple[ConstructorDiagnosticPrediction, ConstructorDiagnosticGold]


def _normalization(name: str, /) -> BeamNormalization:
    """Return the declared beam normalization."""
    match name:
        case "none":
            return BeamNormalization.NONE
        case "depth":
            return BeamNormalization.DEPTH
        case "nodes":
            return BeamNormalization.NODES
        case "terminals":
            return BeamNormalization.TERMINALS
        case _:
            msg = f"unknown beam normalization {name!r}"
            raise ValueError(msg)


def _best(beam: ScoredBeam, normalization: BeamNormalization, /) -> Term | None:
    """Return a beam's highest-ranked term, or none for an empty beam."""
    ranked = normalization.ranked(beam)
    return None if not ranked else ranked[0]


def _identity_alignment(induced: Sequence[str], gold: Sequence[str], /) -> Alignment:
    """Return the declared same-name alignment over two observed inventories."""
    counts = {(left, right): 0 for left in induced for right in gold}
    return Alignment(
        {name: name for name in induced if name in frozenset(gold)},
        Contingency(counts, split="dev"),
    )


def _primitive_name(term: Term, store: TermStore, /) -> str:
    """Return the semantic label carried by a nullary category term."""
    operator = term.op
    if operator.arity != 0:
        msg = f"operator {operator.name!r} is not a primitive"
        raise ValueError(msg)
    if operator.is_constant:
        payload = term.payload
        if payload is None:
            msg = f"constant operator {operator.name!r} has no payload"
            raise ValueError(msg)
        return store.signature.vocabulary(operator.sort).name(payload)
    return store.signature.erase(operator).name


def _operator_observations(
    predicted: Term,
    gold: Term,
    store: TermStore,
    constructors: DeclaredConstructors,
    primitives: frozenset[str],
    primitive_counts: Counter[tuple[str, str]],
    constructor_counts: Counter[tuple[str, str]],
    role_pairs: list[tuple[str, str]],
) -> None:
    """Accumulate structurally paired operator and child-role observations."""
    predicted_constructor = constructors.is_constructor(predicted)
    gold_constructor = constructors.is_constructor(gold)

    if predicted_constructor and gold_constructor:
        predicted_name = constructors.constructor(predicted)
        gold_name = constructors.constructor(gold)
        constructor_counts[predicted_name, gold_name] += 1
        roles = constructors.roles(gold_name)

        for predicted_child, gold_child, role in zip(
            predicted.args, gold.args, roles, strict=False
        ):
            if constructors.is_constructor(predicted_child):
                role_pairs.append((constructors.constructor(predicted_child), role))
            _operator_observations(
                predicted_child,
                gold_child,
                store,
                constructors,
                primitives,
                primitive_counts,
                constructor_counts,
                role_pairs,
            )

        return
    if predicted.op.arity == 0 and gold.op.arity == 0:
        predicted_name = _primitive_name(predicted, store)
        gold_name = _primitive_name(gold, store)
        if predicted_name in primitives and gold_name in primitives:
            primitive_counts[predicted_name, gold_name] += 1

    for left, right in zip(predicted.args, gold.args, strict=False):
        _operator_observations(
            left,
            right,
            store,
            constructors,
            primitives,
            primitive_counts,
            constructor_counts,
            role_pairs,
        )


def _observed_operators(
    terms: Sequence[Term],
    store: TermStore,
    constructors: DeclaredConstructors,
    primitives: frozenset[str],
    /,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the primitive and constructor names occurring in terms."""
    primitive_names: set[str] = set()
    constructor_names: set[str] = set()
    pending = list(terms)
    while pending:
        term = pending.pop()
        if constructors.is_constructor(term):
            constructor_names.add(constructors.constructor(term))
        elif term.op.arity == 0:
            name = _primitive_name(term, store)
            if name in primitives:
                primitive_names.add(name)
        pending.extend(term.args)
    return tuple(sorted(primitive_names)), tuple(sorted(constructor_names))


def _reconstruction(
    entries: Sequence[_Entry],
    context: _ReportContext,
    normalization: BeamNormalization,
) -> TypeSystemReconstruction:
    """Return the reconstruction report pooled across scored items."""
    predicted_by_word: dict[str, list[Term]] = {}
    gold_by_word: dict[str, list[Term]] = {}
    predicted_tokens: list[Term] = []
    gold_tokens: list[Term] = []
    primitive_counts: Counter[tuple[str, str]] = Counter()
    constructor_counts: Counter[tuple[str, str]] = Counter()
    role_pairs: list[tuple[str, str]] = []
    primitives = frozenset(context.primitives)

    for prediction, gold in entries:
        for word in gold.words:
            predicted_by_word.setdefault(word, [])
            gold_by_word.setdefault(word, [])

        for word, beam, target in zip(
            gold.words, prediction.beams, gold.categories, strict=True
        ):
            gold_by_word[word].append(target)
            gold_tokens.append(target)
            candidate = _best(beam, normalization)

            if candidate is None:
                continue

            predicted_by_word[word].append(candidate)
            predicted_tokens.append(candidate)
            _operator_observations(
                candidate,
                target,
                context.store,
                context.constructors,
                primitives,
                primitive_counts,
                constructor_counts,
                role_pairs,
            )

    predicted_primitives, predicted_constructors = _observed_operators(
        predicted_tokens,
        context.store,
        context.constructors,
        primitives,
    )
    gold_primitives, gold_constructors = _observed_operators(
        gold_tokens,
        context.store,
        context.constructors,
        primitives,
    )
    primitive_alignment = _identity_alignment(context.primitives, context.primitives)
    constructor_alignment = _identity_alignment(
        context.constructors.names, context.constructors.names
    )

    return type_system_reconstruction(
        predicted_by_word,
        gold_by_word,
        predicted_tokens,
        gold_tokens,
        (context.store, context.store),
        primitive_alignment=primitive_alignment,
        primitive_contingency=Contingency(
            {
                (induced, gold): primitive_counts[induced, gold]
                for induced in predicted_primitives
                for gold in gold_primitives
            },
            split="test",
        ),
        constructor_alignment=constructor_alignment,
        constructor_contingency=Contingency(
            {
                (induced, gold): constructor_counts[induced, gold]
                for induced in predicted_constructors
                for gold in gold_constructors
            },
            split="test",
        ),
        role_pairs=role_pairs,
        roles=context.roles,
    )


type _Reading = Callable[[Sequence[_Entry]], MetricResult]
"""Computes one report row over an ordered set of scored sentences."""

type _Projection = Callable[[tuple[ScoredBeam, ...], tuple[Term, ...]], float | None]
"""Reads one beam or projection value off the pooled beams and gold types."""

type _ReportValue = Callable[[TypeSystemReconstruction], float | None]
"""Reads one value off the pooled type-system reconstruction."""

type _ReportComponents = Callable[[TypeSystemReconstruction], Mapping[str, float]]
"""Reads the per-symbol breakdown a reconstruction row reports beside its value."""


def _pooled(entries: Sequence[_Entry], /) -> MetricResult | None:
    """Return the absent row of an unscorable sentence set, or None to score it."""
    scheme = Resampling("sentence")
    if not entries:
        return MetricResult(None, 0, scheme)
    if not any(item.categories for _, item in entries):
        return MetricResult(None, len(entries), scheme)
    return None


def _projection(value: _Projection, /) -> _Reading:
    """Return a row read off the pooled beams and gold lexical types."""

    def reading(entries: Sequence[_Entry], /) -> MetricResult:
        absent = _pooled(entries)
        if absent is not None:
            return absent
        beams = tuple(beam for pred, _ in entries for beam in pred.beams)
        gold = tuple(category for _, item in entries for category in item.categories)
        return MetricResult(value(beams, gold), len(entries), Resampling("sentence"))

    return reading


def _no_components(report: TypeSystemReconstruction, /) -> Mapping[str, float]:
    """Return the empty breakdown of a scalar reconstruction row."""
    del report
    return MappingProxyType({})


def _reconstructed(
    context: _ReportContext,
    normalization: BeamNormalization,
    value: _ReportValue,
    components: _ReportComponents = _no_components,
    /,
) -> _Reading:
    """Return a row read off the pooled type-system reconstruction."""

    def reading(entries: Sequence[_Entry], /) -> MetricResult:
        absent = _pooled(entries)
        if absent is not None:
            return absent
        report = _reconstruction(entries, context, normalization)
        return MetricResult(
            value(report),
            len(entries),
            Resampling("sentence"),
            components=dict(components(report)),
        )

    return reading


def _primitive_f1(report: TypeSystemReconstruction, /) -> float | None:
    """Return the mean per-primitive F1, absent when no primitive was scored."""
    scored = report.primitives.per_primitive_f1
    if not scored:
        return None
    return fsum(value for _, value in scored) / len(scored)


def _depths(
    entries: Sequence[_Entry], normalization: BeamNormalization, /
) -> MetricResult:
    """Return the predicted-minus-gold mean depth of the best-ranked candidates.

    Only tokens whose beam ranks a candidate are scored, so both means range
    over the same tokens.
    """
    absent = _pooled(entries)
    if absent is not None:
        return absent
    pairs = tuple(
        (term_depth(candidate), term_depth(target))
        for prediction, gold in entries
        for beam, target in zip(prediction.beams, gold.categories, strict=True)
        if (candidate := _best(beam, normalization)) is not None
    )
    if not pairs:
        return MetricResult(None, len(entries), Resampling("sentence"))
    predicted = fsum(depth for depth, _ in pairs) / len(pairs)
    gold_depth = fsum(depth for _, depth in pairs) / len(pairs)
    return MetricResult(
        predicted - gold_depth,
        len(entries),
        Resampling("sentence"),
        components={"predicted-depth": predicted, "gold-depth": gold_depth},
    )


def _collapse(margin: float, normalization: BeamNormalization, /) -> _Reading:
    """Return the row flagging a depth gap below the negated collapse margin."""

    def reading(entries: Sequence[_Entry], /) -> MetricResult:
        gap = _depths(entries, normalization)
        if gap.value is None:
            return gap
        return MetricResult(
            float(gap.value < -margin),
            gap.n,
            gap.resampling,
            components=dict(gap.components),
        )

    return reading


def _constructor_rows(
    widths: Sequence[int],
    depths: Sequence[int],
    margin: float,
    context: _ReportContext,
    normalization: BeamNormalization,
    /,
) -> tuple[tuple[str, _Reading], ...]:
    """Return every declared constructor row beside the reading that computes it.

    Parameters
    ----------
    widths : Sequence[int]
        The declared beam widths.
    depths : Sequence[int]
        The declared projection depths.
    margin : float
        How far the predicted mean depth may fall below the gold mean depth
        before the collapse row flags it.
    context : _ReportContext
        The constructor-role view.
    normalization : BeamNormalization
        How beam terms are normalized before comparison.

    Returns
    -------
    tuple[tuple[str, _Reading], ...]
        One named reading per row, in report order.
    """

    def beam(width: int, /) -> _Projection:
        return lambda beams, gold: accuracy_at_beam_size(
            beams, gold, normalization=normalization, max_width=width
        ).at(width)

    def typeless_at(width: int, /) -> _Projection:
        return lambda beams, gold: typeless_accuracy(
            beams,
            gold,
            context.representative,
            context.constructors,
            normalization=normalization,
            max_width=width,
        ).at(width)

    def truncated_at(depth: int, /) -> _Projection:
        return lambda beams, gold: depth_truncated_accuracy(
            beams,
            gold,
            context.representative,
            context.constructors,
            normalization=normalization,
            max_depth=depth,
        ).at(depth)

    def report(
        value: _ReportValue, components: _ReportComponents = _no_components, /
    ) -> _Reading:
        return _reconstructed(context, normalization, value, components)

    return (
        *((f"beam-accuracy@{width}", _projection(beam(width))) for width in widths),
        *(
            (f"typeless-accuracy@{width}", _projection(typeless_at(width)))
            for width in widths
        ),
        *(
            (f"truncated-accuracy@{depth}", _projection(truncated_at(depth)))
            for depth in depths
        ),
        (
            "mean-reciprocal-rank",
            _projection(
                lambda beams, gold: mean_reciprocal_rank(
                    beams, gold, normalization=normalization
                )
            ),
        ),
        (
            "primitive-induced-count",
            report(lambda found: float(found.primitives.induced_size)),
        ),
        (
            "primitive-gold-count",
            report(lambda found: float(found.primitives.gold_size)),
        ),
        (
            "primitive-inventory-exact",
            report(lambda found: float(found.primitives.exact_set_match)),
        ),
        (
            "primitive-f1",
            report(
                _primitive_f1,
                lambda found: {
                    f"primitive:{name}": value
                    for name, value in found.primitives.per_primitive_f1
                },
            ),
        ),
        (
            "constructor-induced-count",
            report(lambda found: float(found.constructors.induced_size)),
        ),
        (
            "constructor-gold-count",
            report(lambda found: float(found.constructors.gold_size)),
        ),
        (
            "constructor-inventory-exact",
            report(lambda found: float(found.constructors.exact_set_match)),
        ),
        (
            "constructor-role-information",
            report(lambda found: found.constructors.role_mutual_information),
        ),
        (
            "lexical-type-f1",
            report(
                lambda found: found.lexicon.type_f1,
                lambda found: {
                    f"word:{word}": value for word, value in found.lexicon.per_word_f1
                },
            ),
        ),
        ("lexical-token-accuracy", report(lambda found: found.lexicon.token_accuracy)),
        ("depth-gap", lambda entries: _depths(entries, normalization)),
        ("depth-collapse", _collapse(margin, normalization)),
    )


class _ConstructorMetric:
    """Accumulate one declared diagnostic row over held-out sentences."""

    __slots__ = ("_conventions", "_items", "_name", "_reading")

    def __init__(
        self, name: str, reading: _Reading, conventions: ConventionVector
    ) -> None:
        self._name = ensure_name("a constructor diagnostic row", name)
        self._reading = reading
        self._conventions = conventions
        self._items: dict[ItemId, _Entry] = {}

    @property
    def name(self) -> str:
        """Return this report row's declared name."""
        return self._name

    @property
    def conventions(self) -> ConventionVector:
        """Return the conventions this row is reported under."""
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        """Return that sentence order does not affect this row."""
        return True

    def reset(self) -> None:
        """Forget every scored sentence."""
        self._items = {}

    def update(
        self,
        pred: ConstructorDiagnosticPrediction,
        gold: ConstructorDiagnosticGold,
        *,
        key: ItemId,
    ) -> None:
        """Accumulate one sentence's type beams and gold lexical categories."""
        if pred.words != gold.words or len(pred.beams) != len(gold.categories):
            msg = "constructor diagnostics need one beam and gold type per word"
            raise ValueError(msg)

        held = self._items.get(key)
        entry = (pred, gold)

        if held is not None and held != entry:
            msg = f"constructor diagnostics already saw another item for {key!r}"
            raise ValueError(msg)

        self._items[key] = entry

    def compute(self) -> MetricResult:
        """Return this row pooled over every scored sentence."""
        return self._reading(tuple(self._items[key] for key in sorted(self._items)))

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return this row separately for each scored sentence."""
        return MappingProxyType(
            {key: self._reading((entry,)) for key, entry in sorted(self._items.items())}
        )


def _view(
    grammar: GrammarAssembly, settings: ConstructorViewSettings, /
) -> _ReportContext:
    """Build and validate the report's constructor-role view."""
    roles = dict(settings.roles)
    constructors = DeclaredConstructors(grammar.store, roles)
    representative = grammar.syntax.parse(grammar.store, settings.representative)
    typeless(representative, representative, constructors)
    category_sort = grammar.language.category_sort
    signature = grammar.store.signature
    vocabulary = signature.vocabularies.get(category_sort)
    primitives = tuple(
        dict.fromkeys(
            (
                *(() if vocabulary is None else vocabulary.names),
                *(
                    signature.erase(operator).name
                    for operator in signature.operators
                    if operator.sort == category_sort
                    and operator.arity == 0
                    and not operator.is_constant
                ),
            )
        )
    )
    declared_roles = tuple(
        dict.fromkeys(role for child_roles in roles.values() for role in child_roles)
    )

    if not primitives:
        msg = "constructor diagnostics need a declared category primitive"
        raise ValueError(msg)

    return _ReportContext(
        grammar.store,
        grammar.syntax,
        constructors,
        representative,
        primitives,
        declared_roles,
    )


class ConstructorDiagnosticEvaluator:
    """Report beam, projection, and reconstruction rows over declared roles."""

    __slots__ = ("_context", "_metrics", "_readout")

    def __init__(
        self,
        grammar: GrammarAssembly,
        *,
        readout: str,
        view: ConstructorViewSettings,
        normalization: str,
        widths: Sequence[int],
        depths: Sequence[int],
        collapse_margin: float,
        conventions: ConventionVector,
    ) -> None:
        context = _view(grammar, view)
        normalized = _normalization(normalization)
        declared = conventions.extended(
            {
                "constructor_readout": readout,
                "beam_normalization": normalization,
                "beam_widths": dumps(tuple(widths), separators=(",", ":")),
                "projection_depths": dumps(tuple(depths), separators=(",", ":")),
                "collapse_margin": repr(float(collapse_margin)),
                "constructor_representative": view.representative,
                "constructor_roles": dumps(
                    dict(view.roles),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            }
        )
        self._context = context
        self._readout = ReadoutId(
            ensure_name("a constructor diagnostic term-readout role", readout)
        )
        self._metrics = tuple(
            _ConstructorMetric(name, reading, declared)
            for name, reading in _constructor_rows(
                widths, depths, collapse_margin, context, normalized
            )
        )

    @property
    def conventions(self) -> ConventionVector:
        """Return the conventions every row carries."""
        return self._metrics[0].conventions

    @property
    def metrics(self) -> tuple[_ConstructorMetric, ...]:
        """Return diagnostic rows in declaration order."""
        return self._metrics

    def held_out(
        self, data: HeldOutData, split: str, /
    ) -> EvalDataset[ConstructorDiagnosticGold]:
        """Return a retained partition's derivations as this evaluator reads them."""
        return derivation_dataset(self.annotate, data.derivations(split))

    def annotate(
        self, item: Item, derivation: Derivation[str], /
    ) -> ConstructorDiagnosticGold:
        """Return one gold lexical category per surface token."""
        leaves = sorted(derivation.leaves(), key=lambda leaf: leaf.span.start)
        words = tuple(token.surface for token in item.tokens)
        spans = tuple((leaf.span.start, leaf.span.end) for leaf in leaves)

        if spans != tuple((position, position + 1) for position in range(len(words))):
            msg = "constructor diagnostics need one gold leaf per surface token"
            raise ValueError(msg)

        return ConstructorDiagnosticGold(
            words,
            tuple(
                self._context.syntax.parse(self._context.store, leaf.label)
                for leaf in leaves
            ),
        )

    def predict[BatchT: HeldOutItems](
        self, outputs: ObjectiveOutputs[BatchT], /
    ) -> Mapping[ItemId, ConstructorDiagnosticPrediction]:
        """Return each token's published complete term beam."""
        decoded = outputs.terms.get(self._readout)

        if decoded is None:
            msg = (
                f"constructor diagnostics read complete terms {self._readout!r}, "
                f"and the pass published {sorted(outputs.terms)}"
            )
            raise ValueError(msg)

        predictions: dict[ItemId, ConstructorDiagnosticPrediction] = {}

        for row, (item_id, words) in enumerate(
            zip(outputs.batch.item_ids, outputs.batch.words, strict=True)
        ):
            beams = []

            for position in range(len(words)):
                span = decoded.spans.span_id(position, position + 1)
                weighted = decoded.at(row, span)
                beams.append(
                    tuple(
                        (term, float(weight.detach()))
                        for term, weight in zip(
                            weighted.terms, weighted.log_weights, strict=True
                        )
                    )
                )

            predictions[item_id] = ConstructorDiagnosticPrediction(
                tuple(words), tuple(beams)
            )

        return MappingProxyType(predictions)

    @override
    def __repr__(self) -> str:
        """Return a summary naming the readout and row count."""
        return (
            f"ConstructorDiagnosticEvaluator(readout={self._readout!r}, "
            f"rows={len(self._metrics)})"
        )


def _build(
    settings: ConstructorDiagnosticSettings,
    context: HeldOutEvaluationContext,
    /,
) -> ConstructorDiagnosticEvaluator:
    """Build declared constructor diagnostics in a held-out grammar context."""
    return ConstructorDiagnosticEvaluator(
        context.grammar,
        readout=settings.readout,
        view=settings.view,
        normalization=settings.normalization,
        widths=settings.widths,
        depths=settings.depths,
        collapse_margin=settings.collapse_margin,
        conventions=ConventionVector.from_settings(settings.conventions),
    )


PLUGIN: Final = ComponentPlugin(spec=ConstructorDiagnosticSpec, build=_build)
