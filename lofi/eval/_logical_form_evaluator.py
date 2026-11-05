"""Held-out logical-form matching over a complete-term readout."""

from collections import Counter
from itertools import permutations
from math import factorial
from random import Random
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final, override

from lofi._checks import ensure_name, ensure_positive
from lofi._frozen import FrozenValue
from lofi._types import ReadoutId
from lofi.eval._alignment import Contingency
from lofi.eval._declared_alignment import (
    DeclaredAlignmentScore,
    FittedDeclaredAlignment,
    RelabelingBasePrediction,
    fit_declared_alignment,
    predict_relabeling_base,
    score_declared_alignment,
    stable_declared_alignments,
)
from lofi.eval._fiber import (
    FiberRecovery,
    LogicalVocabulary,
    Predicate,
    fiber_floor,
    fiber_recovery,
    fiber_size,
    permuted_body,
)
from lofi.eval._held_out import derivation_dataset
from lofi.eval._logical_form import (
    LogicalFormMeasure,
    LogicalFormMetric,
    normalized_form,
)
from lofi.eval._policies import ManyToOne, OneToOne
from lofi.eval._values import ConventionVector, MetricResult, Resampling
from lofi.registry import ComponentPlugin
from lofi.spec.evaluators.logical_form import (
    LogicalFormEvaluatorSettings,
    LogicalFormEvaluatorSpec,
)
from lofi.terms import Nominal, UnknownOperatorError, to_core, to_nominal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import ItemId
    from lofi.data import Derivation, EvalDataset, Item
    from lofi.eval import HeldOutEvaluationContext, HeldOutItems
    from lofi.eval._floor import Floor
    from lofi.eval._ports import HeldOutData
    from lofi.eval._stability import AlignmentStability
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.spec.evaluators.alignment import AlignmentSettings
    from lofi.terms import Calculus, Normalization, Signature, Term, TermStore

__all__ = [
    "PLUGIN",
    "AlignedLogicalFormMetric",
    "FittedLogicalFormAlignment",
    "LogicalFormAlignmentScore",
    "LogicalFormEvaluator",
    "fit_logical_form_alignment",
    "logical_form_relabeling_base",
    "score_logical_form_alignment",
    "stable_logical_form_alignments",
]


_FIBER_UNIT: Final = "predicate"


@final
class FittedLogicalFormAlignment(FrozenValue):
    """A logical-symbol correspondence frozen on development."""

    _noun = "fitted logical-form alignment"
    __slots__ = ("_development", "_fitted")

    def __init__(
        self,
        fitted: FittedDeclaredAlignment,
        predictions: Mapping[ItemId, Nominal | None],
        gold: Mapping[ItemId, Nominal],
    ) -> None:
        object.__setattr__(self, "_fitted", fitted)
        object.__setattr__(
            self,
            "_development",
            tuple(
                (item_id, predictions[item_id], gold[item_id])
                for item_id in sorted(gold)
            ),
        )

    @property
    def fitted(self) -> FittedDeclaredAlignment:
        """Return the shared declared alignment value."""
        return self._fitted

    @property
    def development(
        self,
    ) -> tuple[tuple[ItemId, Nominal | None, Nominal], ...]:
        """Return the paired development forms that fixed the alignment."""
        return self._development


def _symbol(node: Nominal, calculus: Calculus) -> str | None:
    """Return the relabelable symbol named by a logical-form node."""
    if node.op.is_constant:
        return node.payload
    if (
        node.op.name in {calculus.lam, calculus.app}
        or node.op.is_bound_variable
        or node.op.is_free_variable
    ):
        return None
    return node.op.name


def _symbol_contingency(
    predictions: Mapping[ItemId, Nominal | None],
    gold: Mapping[ItemId, Nominal],
    calculus: Calculus,
    *,
    split: str,
) -> Contingency:
    """Return structurally paired logical-symbol counts for one split."""
    if predictions.keys() != gold.keys():
        missing = sorted(set(gold) - set(predictions))
        extra = sorted(set(predictions) - set(gold))
        msg = (
            "logical-form alignment requires identical item inventories; "
            f"missing {missing!r}, extra {extra!r}"
        )
        raise ValueError(msg)
    counts: dict[tuple[str, str], int] = {}

    def visit(predicted: Nominal, expected: Nominal) -> None:
        left = _symbol(predicted, calculus)
        right = _symbol(expected, calculus)
        if left is not None and right is not None:
            pair = (left, right)
            counts[pair] = counts.get(pair, 0) + 1
        for predicted_child, gold_child in zip(
            predicted.args, expected.args, strict=False
        ):
            visit(predicted_child, gold_child)

    for item_id in sorted(gold):
        predicted = predictions[item_id]
        if predicted is not None:
            visit(predicted, gold[item_id])
    if not counts:
        msg = f"logical-form alignment has no paired symbols on {split!r}"
        raise ValueError(msg)
    return Contingency(counts, split=split)


def fit_logical_form_alignment(
    predictions: Mapping[ItemId, Nominal | None],
    gold: Mapping[ItemId, Nominal],
    calculus: Calculus,
    settings: AlignmentSettings,
    /,
) -> FittedLogicalFormAlignment:
    """Fit a logical-symbol mapping from real development outputs."""
    contingency = _symbol_contingency(
        predictions, gold, calculus, split=settings.fit_on
    )
    return FittedLogicalFormAlignment(
        fit_declared_alignment(contingency, settings), predictions, gold
    )


def score_logical_form_alignment(
    fitted: FittedLogicalFormAlignment,
    predictions: Mapping[ItemId, Nominal | None],
    gold: Mapping[ItemId, Nominal],
    calculus: Calculus,
    /,
    *,
    seed: int,
) -> DeclaredAlignmentScore:
    """Score a frozen logical-symbol mapping on held-out test evidence."""
    contingency = _symbol_contingency(
        predictions,
        gold,
        calculus,
        split=fitted.fitted.settings.score_on,
    )
    return score_declared_alignment(fitted.fitted, contingency, seed=seed)


def stable_logical_form_alignments(
    fitted: Sequence[FittedLogicalFormAlignment],
    /,
    *,
    seed: int,
) -> AlignmentStability:
    """Return logical-symbol mapping stability across run seeds."""
    return stable_declared_alignments(
        tuple(entry.fitted for entry in fitted), seed=seed
    )


def logical_form_relabeling_base(
    fitted: FittedLogicalFormAlignment,
    /,
    *,
    ambiguity_budget: int,
) -> RelabelingBasePrediction:
    """Return the symbol permutation group's predicted clamp base."""
    return predict_relabeling_base(
        primitive_count=len(fitted.fitted.alignment.contingency.induced_labels),
        constructor_count=0,
        ambiguity_budget=ensure_positive("an ambiguity budget", ambiguity_budget),
    )


def _aligned_nominal(
    node: Nominal,
    fitted: FittedLogicalFormAlignment,
    signature: Signature,
    calculus: Calculus,
) -> Nominal:
    """Apply one frozen symbol mapping while preserving term structure."""
    symbol = _symbol(node, calculus)
    operator = node.op
    payload = node.payload
    if symbol is not None:
        image = fitted.fitted.alignment.apply(symbol)
        if image is None:
            if fitted.fitted.settings.kind == "identity":
                image = symbol
            else:
                msg = f"logical symbol {symbol!r} has no frozen alignment image"
                raise ValueError(msg)
        if operator.is_constant:
            payload = image
        else:
            try:
                target = signature.by_name(image)
            except UnknownOperatorError as error:
                msg = f"aligned logical symbol {image!r} is absent from the signature"
                raise ValueError(msg) from error
            source_shape = (
                operator.sort,
                operator.arg_sorts,
                operator.binding,
                operator.payload_sort,
            )
            target_shape = (
                target.sort,
                target.arg_sorts,
                target.binding,
                target.payload_sort,
            )
            if source_shape != target_shape:
                msg = (
                    f"logical symbol {symbol!r} is incompatible with aligned "
                    f"symbol {image!r}"
                )
                raise ValueError(msg)
            operator = target
    return Nominal(
        operator,
        tuple(
            _aligned_nominal(child, fitted, signature, calculus) for child in node.args
        ),
        payload=payload,
        binders=node.binders,
    )


def _alignment_covers(
    node: Nominal,
    fitted: FittedLogicalFormAlignment,
    calculus: Calculus,
) -> bool:
    """Return whether every relabelable symbol has a frozen image."""
    symbol = _symbol(node, calculus)
    covered = (
        symbol is None
        or fitted.fitted.settings.kind == "identity"
        or fitted.fitted.alignment.apply(symbol) is not None
    )
    return covered and all(
        _alignment_covers(child, fitted, calculus) for child in node.args
    )


@final
class AlignedLogicalFormMetric:
    """Apply one frozen symbol mapping before logical-form scoring."""

    __slots__ = ("_calculus", "_fitted", "_metric", "_signature")

    def __init__(
        self,
        metric: LogicalFormMetric,
        fitted: FittedLogicalFormAlignment,
        signature: Signature,
        calculus: Calculus,
    ) -> None:
        self._metric = metric
        self._fitted = fitted
        self._signature = signature
        self._calculus = calculus

    @property
    def name(self) -> str:
        """Return the wrapped metric name."""
        return self._metric.name

    @property
    def conventions(self) -> ConventionVector | None:
        """Return the wrapped scoring conventions."""
        return self._metric.conventions

    @property
    def order_invariant(self) -> bool:
        """Return whether update order affects the aggregate."""
        return self._metric.order_invariant

    def reset(self) -> None:
        """Forget all scored test forms."""
        self._metric.reset()

    def update(self, pred: Nominal | None, gold: Nominal, *, key: ItemId) -> None:
        """Align one predicted form and score it against its gold form."""
        self._metric.update(
            (
                None
                if pred is None
                or not _alignment_covers(pred, self._fitted, self._calculus)
                else _aligned_nominal(
                    pred, self._fitted, self._signature, self._calculus
                )
            ),
            gold,
            key=key,
        )

    def compute(self) -> MetricResult:
        """Return the aggregate aligned score."""
        return self._metric.compute()

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return aligned scores by item identifier."""
        return self._metric.per_item()


@final
class LogicalFormAlignmentScore(FrozenValue):
    """Logical-symbol alignment rows beside the vocabulary-fiber controls.

    Beside the declared alignment's rows, the score reports four rows read off the
    induced logical vocabulary. Each share is over the gold relations the scored
    forms use, and is None when they use none.

    ``fiber-relative-recovery``
        The share of gold relations reached by the relabeling the assignment policy
        fits, where an induced relation counts for a gold relation when some
        permutation of its arguments makes its body the gold relation's body in the
        ambient theory.
    ``fixed-vocabulary-accuracy``
        The share of gold relations whose same-named induced relation has the same
        arity and, under its own argument order, the gold relation's body in the
        ambient theory, with no relabeling.
    ``fiber-floor``
        The mean ``fiber-relative-recovery`` of random vocabularies, with an
        interval at the declared floor level. Each of the declared number of draws
        declares as many relations as the induced vocabulary, gives each one the
        body of a gold relation drawn with replacement under a random argument
        permutation, and is scored through the same assignment policy.
    ``fiber-size``
        How many relabelings of the gold vocabulary, with argument permutations,
        yield a distinct vocabulary: the order of the relabeling group divided by
        the number of its elements that fix the vocabulary.

    Parameters
    ----------
    alignment : DeclaredAlignmentScore
        The declared alignment's score and rows.
    recovery : FiberRecovery
        The induced vocabulary's recovery and its chance floor.
    fiber_size_result : MetricResult
        The ``fiber-size`` row.
    """

    _noun = "logical-form alignment score"
    __slots__ = ("_alignment", "_rows")

    def __init__(
        self,
        alignment: DeclaredAlignmentScore,
        recovery: FiberRecovery,
        fiber_size_result: MetricResult,
    ) -> None:
        object.__setattr__(self, "_alignment", alignment)
        object.__setattr__(
            self,
            "_rows",
            MappingProxyType(
                {
                    **alignment.rows,
                    "fiber-relative-recovery": recovery.up_to_fiber,
                    "fixed-vocabulary-accuracy": recovery.fixed_alphabet,
                    "fiber-floor": recovery.floor,
                    "fiber-size": fiber_size_result,
                }
            ),
        )

    @property
    def kind(self) -> str:
        """Return the fitted symbol-alignment policy."""
        return self._alignment.kind

    @property
    def observed(self) -> MetricResult:
        """Return held-out symbol recovery under the fitted alignment."""
        return self._alignment.observed

    @property
    def floor(self) -> Floor:
        """Return the symbol-relabeling floor."""
        return self._alignment.floor

    @property
    def rows(self) -> Mapping[str, MetricResult]:
        """Return alignment, floor, and fiber-size rows in report order."""
        return self._rows


def _symbol_arities(context: HeldOutEvaluationContext) -> Mapping[str, int]:
    """Return the interpretation's non-logical symbols and curried arities."""
    interpretation = context.interpretation
    if interpretation is None:
        msg = "logical-form evaluation requires a declared interpretation"
        raise ValueError(msg)
    arities: dict[str, int] = {}
    for members in interpretation.inventory.roles.values():
        arities.update(members)
    return MappingProxyType(arities)


def _gold_vocabulary(
    context: HeldOutEvaluationContext,
    symbols: frozenset[str],
) -> LogicalVocabulary:
    """Return the observed part of the declared non-logical vocabulary."""
    interpretation = context.interpretation
    if interpretation is None:
        msg = "logical-form evaluation requires a declared interpretation"
        raise ValueError(msg)
    arities = _symbol_arities(context)
    slots = tuple(
        f"x{position}" for position in range(max(arities.values(), default=0))
    )
    slot_terms = tuple(
        interpretation.store.fvar(interpretation.language.term_sort, slot)
        for slot in slots
    )
    application = interpretation.store.signature.by_name(
        interpretation.calculus.operators.app
    )

    def body(symbol: str, arity: int) -> Term:
        """Return the relation applied to distinct canonical argument slots."""
        held = interpretation.store.const(
            interpretation.language.term_sort,
            symbol,
        )
        for argument in slot_terms[:arity]:
            held = interpretation.store.op(application, held, argument)
        return held

    return LogicalVocabulary(
        (
            Predicate(
                symbol,
                arity,
                body(symbol, arity),
            )
            for symbol, arity in arities.items()
            if symbol in symbols
        ),
        slots=slots,
    )


def _spines(
    node: Nominal,
    symbol: str,
    arity: int,
    calculus: Calculus,
) -> tuple[tuple[Nominal, ...], ...]:
    """Return complete application spines headed by one symbol."""
    found: list[tuple[Nominal, ...]] = []
    pending = [node]
    while pending:
        current = pending.pop()
        head = current
        arguments: list[Nominal] = []
        while head.op.name == calculus.app and len(head.args) == 2:
            head, argument = head.args
            arguments.append(argument)
        arguments.reverse()
        if _symbol(head, calculus) == symbol and len(arguments) == arity:
            found.append(tuple(arguments))
        pending.extend(reversed(current.args))
    return tuple(found)


def _argument_order(
    fitted: FittedLogicalFormAlignment,
    induced: str,
    reference: str,
    arity: int,
    store: TermStore,
    calculus: Calculus,
    normalization: Normalization,
) -> tuple[int, ...] | None:
    """Infer one coherent argument order from paired development occurrences."""

    def key(nominal: Nominal) -> Term:
        return normalized_form(
            store,
            to_core(store, nominal),
            calculus,
            normalization=normalization,
        )

    def aligned_keys(arguments: tuple[Nominal, ...]) -> tuple[Term, ...] | None:
        aligned: list[Term] = []
        for argument in arguments:
            if not _alignment_covers(argument, fitted, calculus):
                return None
            nominal = _aligned_nominal(
                argument,
                fitted,
                store.signature,
                calculus,
            )
            aligned.append(key(nominal))
        return tuple(aligned)

    candidates: list[tuple[int, ...]] = []
    for order in permutations(range(arity)):
        observed = False
        coherent = True
        for _, predicted, gold in fitted.development:
            predicted_rows = (
                ()
                if predicted is None
                else _spines(predicted, induced, arity, calculus)
            )
            gold_rows = _spines(gold, reference, arity, calculus)
            if not predicted_rows and not gold_rows:
                continue
            observed = True
            moved: list[tuple[Term, ...]] = []
            for arguments in predicted_rows:
                aligned = aligned_keys(arguments)
                if aligned is None:
                    coherent = False
                    break
                moved.append(tuple(aligned[position] for position in order))
            if not coherent:
                break
            expected = tuple(
                tuple(key(argument) for argument in row) for row in gold_rows
            )
            if Counter(moved) != Counter(expected):
                coherent = False
                break
        if observed and coherent:
            candidates.append(tuple(order))
    return None if not candidates else candidates[0]


def _induced_vocabulary(
    fitted: FittedLogicalFormAlignment,
    gold: LogicalVocabulary,
    store: TermStore,
    calculus: Calculus,
    normalization: Normalization,
) -> LogicalVocabulary:
    """Return relations whose argument order is coherent on development."""
    predicates: list[Predicate] = []
    gold_symbols = frozenset(gold.symbols)
    for induced, reference in sorted(fitted.fitted.alignment.mapping.items()):
        if reference not in gold_symbols:
            continue
        meaning = gold.predicate(reference)
        order = _argument_order(
            fitted,
            induced,
            reference,
            meaning.arity,
            store,
            calculus,
            normalization,
        )
        if order is not None:
            predicates.append(
                Predicate(
                    induced,
                    meaning.arity,
                    permuted_body(store, gold, reference, order),
                )
            )
    return LogicalVocabulary(predicates, slots=gold.slots)


def _fiber_cardinality(
    vocabulary: LogicalVocabulary,
    stabilizer: int,
) -> tuple[int, int]:
    """Return the orbit size and full action-group order."""
    group_order = factorial(len(vocabulary.symbols))
    for predicate in vocabulary.predicates:
        group_order *= factorial(predicate.arity)
    orbit, remainder = divmod(group_order, stabilizer)
    if remainder:
        msg = (
            f"a vocabulary stabilizer of {stabilizer} does not divide its "
            f"action-group order of {group_order}"
        )
        raise ValueError(msg)
    return orbit, group_order


def _fiber_policy(settings: AlignmentSettings) -> ManyToOne | OneToOne:
    """Return the assignment policy used by the random fiber control."""
    tie_break = settings.tie_break or "least-gold-label"
    if settings.kind == "many-to-one":
        return ManyToOne(tie_break=tie_break)
    return OneToOne(tie_break=tie_break)


class LogicalFormEvaluator:
    """Score the most probable decoded form of each held-out sentence.

    The evaluator reads a complete-term readout at an item's root cell and
    selects the form with greatest decoder weight. Gold forms come from the
    corpus's declared interpretation, through the held-out evaluation context.
    """

    __slots__ = (
        "_alignment",
        "_context",
        "_conventions",
        "_fitted",
        "_measure",
        "_metrics",
        "_names",
        "_permutation_limit",
        "_readout",
        "_restarts",
    )

    def __init__(
        self,
        context: HeldOutEvaluationContext,
        /,
        *,
        readout: str,
        measure: LogicalFormMeasure,
        conventions: ConventionVector,
        alignment: AlignmentSettings,
        restarts: int,
        permutation_limit: int,
    ) -> None:
        interpretation = context.interpretation

        if interpretation is None:
            msg = "logical-form evaluation requires a declared interpretation"
            raise ValueError(msg)

        if not context.forms:
            msg = "logical-form evaluation requires held-out gold forms"
            raise ValueError(msg)

        self._context = context
        self._alignment = alignment
        self._measure = measure
        self._readout = ReadoutId(
            ensure_name("a logical-form term-readout role", readout)
        )
        self._conventions = conventions.extended(
            {
                "alignment": alignment.kind,
                "alignment_tie_break": alignment.tie_break,
                "alignment_fit_on": alignment.fit_on,
                "alignment_score_on": alignment.score_on,
                "random_floor_replicates": alignment.floor_replicates,
                "random_floor_level": alignment.floor_level,
                "stability_replicates": alignment.stability_replicates,
                "stability_level": alignment.stability_level,
                "normalization_order": interpretation.normalization.order,
                "normalization_eta": interpretation.normalization.eta,
                "normalization_fuel": interpretation.normalization.fuel,
                "normalization_policy": interpretation.normalization_policy,
                "normalization_associative_commutative": ",".join(
                    interpretation.normalization.associative_commutative
                ),
                "smatch_restarts": restarts,
                "fiber_permutation_limit": permutation_limit,
            }
        )
        self._restarts = restarts
        self._permutation_limit = permutation_limit
        self._names = tuple(
            f"x{index}"
            for index in range(
                interpretation.language.codec.at(
                    interpretation.language.term_sort
                ).capacity
            )
        )
        self._fitted: FittedLogicalFormAlignment | None = None
        self._metrics: tuple[AlignedLogicalFormMetric, ...] = ()

    @property
    def conventions(self) -> ConventionVector:
        """Return the conventions the metric rows are reported under."""
        return self._conventions

    @property
    def metrics(
        self,
    ) -> tuple[AlignedLogicalFormMetric, ...]:
        """Return the declared logical-form metric in report order."""
        if not self._metrics:
            msg = "logical-form evaluation requires alignment fitted on development"
            raise ValueError(msg)
        return self._metrics

    @property
    def alignment(self) -> AlignmentSettings:
        """Return the declared development-fitted symbol alignment settings."""
        return self._alignment

    @property
    def fitted_alignment(self) -> FittedLogicalFormAlignment | None:
        """Return the frozen development mapping, if it has been fitted."""
        return self._fitted

    def fit_alignment(
        self,
        predictions: Mapping[ItemId, Nominal | None],
        gold: Mapping[ItemId, Nominal],
        /,
    ) -> FittedLogicalFormAlignment:
        """Fit the declared logical-symbol mapping on development outputs."""
        interpretation = self._context.interpretation
        if interpretation is None:
            msg = "logical-form evaluation requires a declared interpretation"
            raise ValueError(msg)
        calculus = interpretation.calculus.operators
        fitted = fit_logical_form_alignment(
            predictions, gold, calculus, self._alignment
        )
        metric = LogicalFormMetric(
            interpretation.store.signature,
            calculus,
            self._measure,
            normalization=interpretation.normalization,
            store_nodes=self._context.grammar.store.max_nodes,
            conventions=self._conventions,
            restarts=self._restarts,
        )
        self._fitted = fitted
        self._metrics = (
            AlignedLogicalFormMetric(
                metric,
                fitted,
                interpretation.store.signature,
                calculus,
            ),
        )
        return fitted

    def score_alignment(
        self,
        predictions: Mapping[ItemId, Nominal | None],
        gold: Mapping[ItemId, Nominal],
        /,
        *,
        seed: int,
    ) -> LogicalFormAlignmentScore:
        """Return held-out symbol accuracy and the vocabulary-fiber controls."""
        if self._fitted is None:
            msg = "logical-form evaluation requires alignment fitted on development"
            raise ValueError(msg)
        interpretation = self._context.interpretation
        if interpretation is None:
            msg = "logical-form evaluation requires a declared interpretation"
            raise ValueError(msg)
        aligned = score_logical_form_alignment(
            self._fitted,
            predictions,
            gold,
            interpretation.calculus.operators,
            seed=seed,
        )
        contingency = _symbol_contingency(
            predictions,
            gold,
            interpretation.calculus.operators,
            split=self._alignment.score_on,
        )
        vocabulary = _gold_vocabulary(
            self._context,
            frozenset(contingency.gold_labels),
        )
        if vocabulary.predicates:
            predicted = _induced_vocabulary(
                self._fitted,
                vocabulary,
                interpretation.store,
                interpretation.calculus.operators,
                interpretation.normalization,
            )
            floor = fiber_floor(
                interpretation.store,
                vocabulary,
                vocabulary.predicates,
                size=max(len(predicted.predicates), 1),
                policy=_fiber_policy(self._alignment),
                split=self._alignment.fit_on,
                rng=Random(seed),
                draws=self._alignment.floor_replicates,
                calculus=interpretation.calculus.operators,
                normalization=interpretation.normalization,
                service=interpretation.service,
                budget=interpretation.budget,
                level=self._alignment.floor_level,
            )
            recovery = fiber_recovery(
                interpretation.store,
                predicted,
                vocabulary,
                policy=_fiber_policy(self._alignment),
                split=self._alignment.fit_on,
                calculus=interpretation.calculus.operators,
                normalization=interpretation.normalization,
                service=interpretation.service,
                budget=interpretation.budget,
                floor=floor,
            )
            stabilizer = fiber_size(
                interpretation.store,
                vocabulary,
                calculus=interpretation.calculus.operators,
                normalization=interpretation.normalization,
                service=interpretation.service,
                budget=interpretation.budget,
                limit=self._permutation_limit,
            )
            size, group_order = _fiber_cardinality(vocabulary, stabilizer)
            sized = MetricResult(
                float(size),
                len(vocabulary.predicates),
                Resampling(_FIBER_UNIT),
                components={
                    "gold_labels": float(len(vocabulary.predicates)),
                    "group_order": float(group_order),
                    "stabilizer_size": float(stabilizer),
                },
            )
        else:
            empty = MetricResult(None, 0, Resampling(_FIBER_UNIT))
            recovery = FiberRecovery(
                empty,
                empty,
                _fiber_policy(self._alignment).fit(
                    Contingency({}, split=self._alignment.fit_on)
                ),
                (),
                empty,
            )
            sized = MetricResult(None, 0, Resampling(_FIBER_UNIT))
        return LogicalFormAlignmentScore(aligned, recovery, sized)

    def relabeling_base(self, *, ambiguity_budget: int) -> RelabelingBasePrediction:
        """Return the logical-symbol group's predicted clamp base."""
        if self._fitted is None:
            msg = "logical-form evaluation requires alignment fitted on development"
            raise ValueError(msg)
        return logical_form_relabeling_base(
            self._fitted, ambiguity_budget=ambiguity_budget
        )

    def held_out(self, data: HeldOutData, split: str, /) -> EvalDataset[Nominal]:
        """Return a retained partition's derivations as this evaluator reads them."""
        return derivation_dataset(self.annotate, data.derivations(split))

    def annotate(self, item: Item, derivation: Derivation[str], /) -> Nominal:
        """Return the corpus's interpreted gold form for one held-out item."""
        del derivation

        try:
            text = self._context.forms[item.item_id]
        except KeyError:
            msg = f"the held-out item {item.item_id!r} has no gold logical form"
            raise ValueError(msg) from None

        interpretation = self._context.interpretation

        if interpretation is None:
            msg = "logical-form evaluation requires a declared interpretation"
            raise ValueError(msg)

        return to_nominal(
            interpretation.store,
            interpretation.read_text(text),
            self._names,
        )

    def predict[BatchT: HeldOutItems](
        self, outputs: ObjectiveOutputs[BatchT], /
    ) -> Mapping[ItemId, Nominal]:
        """Return the highest-weight decoded form at each item's root cell."""
        decoded = outputs.terms.get(self._readout)

        if decoded is None:
            msg = (
                f"logical forms read the complete terms {self._readout!r}, and "
                f"the pass published {sorted(outputs.terms)}"
            )
            raise ValueError(msg)

        interpretation = self._context.interpretation

        if interpretation is None:
            msg = "logical-form evaluation requires a declared interpretation"
            raise ValueError(msg)

        store = interpretation.store
        predictions: dict[ItemId, Nominal] = {}

        for row, (item_id, words) in enumerate(
            zip(outputs.batch.item_ids, outputs.batch.words, strict=True)
        ):
            root = decoded.spans.span_id(0, len(words))
            candidates = decoded.at(row, root)

            if not candidates.terms:
                msg = (
                    f"the complete-term readout {self._readout!r} decoded no "
                    f"logical form for {item_id!r}"
                )
                raise ValueError(msg)

            selected = int(candidates.log_weights.detach().argmax())
            predictions[item_id] = to_nominal(
                store,
                candidates.terms[selected],
                self._names,
            )

        return MappingProxyType(predictions)

    @override
    def __repr__(self) -> str:
        """Return a summary naming the readout and metric."""
        return (
            f"LogicalFormEvaluator(readout={self._readout!r}, "
            f"metric={f'logical-form-{self._measure}'!r})"
        )


def _build(
    settings: LogicalFormEvaluatorSettings,
    context: HeldOutEvaluationContext,
    /,
) -> LogicalFormEvaluator:
    """Build the declared logical-form evaluator in its held-out context."""
    return LogicalFormEvaluator(
        context,
        readout=settings.readout,
        measure=settings.measure,
        conventions=ConventionVector.from_settings(settings.conventions),
        alignment=settings.alignment,
        restarts=settings.restarts,
        permutation_limit=settings.permutation_limit,
    )


PLUGIN: Final = ComponentPlugin(spec=LogicalFormEvaluatorSpec, build=_build)
