"""Held-out lexical-type F1 over a trained parser's term readout."""

from statistics import fmean
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, cast, final, override

from lofi._checks import ensure_name, ensure_positive
from lofi._frozen import FrozenValue
from lofi._types import ReadoutId
from lofi.eval._alignment import Alignment, Contingency
from lofi.eval._declared_alignment import (
    DeclaredAlignmentScore,
    FittedDeclaredAlignment,
    RelabelingBasePrediction,
    fit_declared_alignment,
    predict_relabeling_base,
    score_declared_alignment,
    stable_declared_alignments,
)
from lofi.eval._held_out import derivation_dataset
from lofi.eval._reconstruction import LexicalTypeRecovery, lexical_type_recovery
from lofi.eval._values import ConventionVector, MetricResult, Resampling
from lofi.registry import ComponentPlugin
from lofi.spec.evaluators.lexical_types import (
    LexicalTypeEvaluatorSettings,
    LexicalTypeEvaluatorSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import ItemId
    from lofi.data import Derivation, EvalDataset, Item
    from lofi.eval._held_out import HeldOutEvaluationContext
    from lofi.eval._ports import HeldOutData, HeldOutItems
    from lofi.eval._stability import AlignmentStability
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.spec.evaluators.alignment import AlignmentSettings
    from lofi.spec.grammar import GrammarAssembly
    from lofi.terms import Term, TermStore

__all__ = [
    "PLUGIN",
    "FittedLexicalAlignment",
    "LexicalAlignmentScore",
    "LexicalAlignmentStability",
    "LexicalAnalysis",
    "LexicalTypeEvaluator",
    "LexicalTypeF1",
    "fit_lexical_alignment",
    "lexical_relabeling_base",
    "score_lexical_alignment",
    "stable_lexical_alignments",
]


class LexicalAnalysis(NamedTuple):
    """The words of one item and the lexical type of each.

    Attributes
    ----------
    words : tuple[str, ...]
        The surface form of each token.
    categories : tuple[Term, ...]
        The lexical type of each token.
    punctuation : frozenset[int]
        The positions of punctuation tokens.
    """

    words: tuple[str, ...]
    categories: tuple[Term, ...]
    punctuation: frozenset[int]


class _Scored(NamedTuple):
    """The words, predicted types and gold types one item contributes."""

    predicted_words: tuple[str, ...]
    predicted: tuple[Term, ...]
    gold_words: tuple[str, ...]
    gold: tuple[Term, ...]
    extra: tuple[Term, ...]


@final
class FittedLexicalAlignment(FrozenValue):
    """Primitive and constructor correspondences frozen on development."""

    _noun = "fitted lexical alignment"
    __slots__ = ("_constructors", "_primitives")

    def __init__(
        self,
        primitives: FittedDeclaredAlignment,
        constructors: FittedDeclaredAlignment | None,
    ) -> None:
        object.__setattr__(self, "_primitives", primitives)
        object.__setattr__(self, "_constructors", constructors)

    @property
    def primitives(self) -> FittedDeclaredAlignment:
        """Return the primitive correspondence fitted on development."""
        return self._primitives

    @property
    def constructors(self) -> FittedDeclaredAlignment | None:
        """Return the constructor correspondence, if constructors occurred."""
        return self._constructors


@final
class LexicalAlignmentScore(FrozenValue):
    """Held-out primitive and constructor scores under frozen correspondences."""

    _noun = "lexical alignment score"
    __slots__ = ("_constructors", "_primitives", "_rows")

    def __init__(
        self,
        primitives: DeclaredAlignmentScore,
        constructors: DeclaredAlignmentScore | None,
    ) -> None:
        rows = {f"primitive-{name}": result for name, result in primitives.rows.items()}
        if constructors is None:
            missing = MetricResult(None, 0, Resampling("token"))
            rows[f"constructor-{primitives.kind}-accuracy"] = missing
            rows["constructor-random-relabeling-floor"] = missing
        else:
            rows.update(
                (f"constructor-{name}", result)
                for name, result in constructors.rows.items()
            )
        object.__setattr__(self, "_primitives", primitives)
        object.__setattr__(self, "_constructors", constructors)
        object.__setattr__(self, "_rows", MappingProxyType(rows))

    @property
    def primitives(self) -> DeclaredAlignmentScore:
        """Return held-out primitive accuracy and its relabeling floor."""
        return self._primitives

    @property
    def constructors(self) -> DeclaredAlignmentScore | None:
        """Return constructor accuracy and floor, if constructors occurred."""
        return self._constructors

    @property
    def rows(self) -> Mapping[str, MetricResult]:
        """Return observed and floor rows with unambiguous metric names."""
        return self._rows


class LexicalAlignmentStability(NamedTuple):
    """Across-seed stability for the two operator inventories."""

    primitives: AlignmentStability
    constructors: AlignmentStability | None


def _scored_entry(
    pred: tuple[Term, ...],
    gold: LexicalAnalysis,
    conventions: ConventionVector,
) -> _Scored | None:
    """Return the token evidence retained by the declared conventions."""
    removed = gold.punctuation if conventions.punctuation == "strip" else frozenset()
    gold_kept = tuple(
        position for position in range(len(gold.words)) if position not in removed
    )

    if not gold_kept or len(gold_kept) < conventions.minimum_length:
        return None

    paired = tuple(position for position in gold_kept if position < len(pred))

    if conventions.coverage == "ignore":
        gold_kept = paired

        if not gold_kept:
            return None

    extras = tuple(pred[len(gold.words) :]) if conventions.coverage == "zero" else ()

    return _Scored(
        tuple(gold.words[position] for position in paired),
        tuple(pred[position] for position in paired),
        tuple(gold.words[position] for position in gold_kept),
        tuple(gold.categories[position] for position in gold_kept),
        extras,
    )


def _alignment_counts(
    scored: Sequence[_Scored],
    grammar: GrammarAssembly,
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    """Count structurally paired primitive and constructor observations."""
    primitives: dict[tuple[str, str], int] = {}
    constructors: dict[tuple[str, str], int] = {}
    signature = grammar.store.signature

    def visit(predicted: Term, gold: Term) -> None:
        if predicted.op.arity == 0 and gold.op.arity == 0:
            if (
                predicted.op.is_constant
                and gold.op.is_constant
                and predicted.payload is not None
                and gold.payload is not None
            ):
                pair = (
                    signature.vocabulary(predicted.op.sort).name(predicted.payload),
                    signature.vocabulary(gold.op.sort).name(gold.payload),
                )
            else:
                pair = (
                    signature.erase(predicted.op).name,
                    signature.erase(gold.op).name,
                )
            primitives[pair] = primitives.get(pair, 0) + 1
        elif predicted.op.arity > 0 and gold.op.arity > 0:
            pair = (predicted.op.name, gold.op.name)
            constructors[pair] = constructors.get(pair, 0) + 1

        for predicted_child, gold_child in zip(predicted.args, gold.args, strict=False):
            visit(predicted_child, gold_child)

    for entry in scored:
        for predicted, gold in zip(entry.predicted, entry.gold, strict=False):
            visit(predicted, gold)

    return primitives, constructors


def _alignment_covers(
    term: Term,
    store: TermStore,
    primitive_alignment: Alignment,
    constructor_alignment: Alignment | None,
    /,
) -> bool:
    """Return whether every operator in a term has a frozen alignment image."""
    operator = term.op

    if operator.arity == 0 and operator.is_constant:
        payload = term.payload
        if payload is None:
            msg = f"constant operator {operator.name!r} has no payload"
            raise ValueError(msg)
        name = store.signature.vocabulary(operator.sort).name(payload)
        covered = primitive_alignment.apply(name) is not None
    elif operator.arity == 0:
        covered = (
            primitive_alignment.apply(store.signature.erase(operator).name) is not None
        )
    else:
        covered = (
            constructor_alignment is not None
            and constructor_alignment.apply(operator.name) is not None
        )

    return covered and all(
        _alignment_covers(
            child,
            store,
            primitive_alignment,
            constructor_alignment,
        )
        for child in term.args
    )


def _zero_coverage_recovery(
    predicted: Mapping[str, Sequence[Term]],
    gold: Mapping[str, Sequence[Term]],
    predicted_tokens: Sequence[Term],
    gold_tokens: Sequence[Term],
    paired_tokens: Sequence[tuple[Term, Term]],
    store: TermStore,
    primitive_alignment: Alignment,
    constructor_alignment: Alignment | None,
    /,
) -> LexicalTypeRecovery:
    """Score unaligned predicted operators as false positives under zero coverage."""
    covered = {
        word: tuple(
            term
            for term in terms
            if _alignment_covers(
                term,
                store,
                primitive_alignment,
                constructor_alignment,
            )
        )
        for word, terms in predicted.items()
    }
    covered_pairs = tuple(
        (induced, expected)
        for induced, expected in paired_tokens
        if _alignment_covers(
            induced,
            store,
            primitive_alignment,
            constructor_alignment,
        )
    )
    baseline = lexical_type_recovery(
        covered,
        gold,
        tuple(induced for induced, _ in covered_pairs),
        tuple(expected for _, expected in covered_pairs),
        (store, store),
        primitive_alignment=primitive_alignment,
        constructor_alignment=constructor_alignment,
    )
    baseline_by_word = dict(baseline.per_word_f1)
    per_word: list[tuple[str, float]] = []

    for word in sorted(predicted):
        baseline_denominator = len(covered[word]) + len(gold[word])
        matched = baseline_by_word[word] * baseline_denominator / 2.0
        denominator = len(predicted[word]) + len(gold[word])
        per_word.append(
            (word, 1.0 if denominator == 0 else 2.0 * matched / denominator)
        )

    paired_accuracy = baseline.token_accuracy
    matched_tokens = (
        0.0 if paired_accuracy is None else paired_accuracy * len(covered_pairs)
    )
    token_count = max(len(predicted_tokens), len(gold_tokens))
    return LexicalTypeRecovery(
        len(per_word),
        token_count,
        None if not per_word else fmean(score for _, score in per_word),
        None if token_count == 0 else matched_tokens / token_count,
        tuple(per_word),
    )


def _alignment_evidence(
    predictions: Mapping[ItemId, tuple[Term, ...]],
    gold: Mapping[ItemId, LexicalAnalysis],
    grammar: GrammarAssembly,
    conventions: ConventionVector,
    *,
    split: str,
) -> tuple[Contingency, Contingency | None]:
    """Return primitive and constructor contingencies for one named split."""
    if predictions.keys() != gold.keys():
        missing = sorted(set(gold) - set(predictions))
        extra = sorted(set(predictions) - set(gold))
        msg = (
            "lexical alignment requires identical item inventories; "
            f"missing {missing!r}, extra {extra!r}"
        )
        raise ValueError(msg)

    scored = tuple(
        entry
        for item_id in sorted(gold)
        if (entry := _scored_entry(predictions[item_id], gold[item_id], conventions))
        is not None
    )
    primitives, constructors = _alignment_counts(scored, grammar)

    if not primitives:
        msg = f"lexical alignment has no paired primitive observations on {split!r}"
        raise ValueError(msg)

    return (
        Contingency(primitives, split=split),
        None if not constructors else Contingency(constructors, split=split),
    )


def fit_lexical_alignment(
    predictions: Mapping[ItemId, tuple[Term, ...]],
    gold: Mapping[ItemId, LexicalAnalysis],
    grammar: GrammarAssembly,
    conventions: ConventionVector,
    settings: AlignmentSettings,
    /,
) -> FittedLexicalAlignment:
    """Fit primitive and constructor mappings from real development outputs."""
    primitives, constructors = _alignment_evidence(
        predictions,
        gold,
        grammar,
        conventions,
        split=settings.fit_on,
    )
    return FittedLexicalAlignment(
        fit_declared_alignment(primitives, settings),
        (
            None
            if constructors is None
            else fit_declared_alignment(constructors, settings)
        ),
    )


def score_lexical_alignment(
    fitted: FittedLexicalAlignment,
    predictions: Mapping[ItemId, tuple[Term, ...]],
    gold: Mapping[ItemId, LexicalAnalysis],
    grammar: GrammarAssembly,
    conventions: ConventionVector,
    /,
    *,
    seed: int,
) -> LexicalAlignmentScore:
    """Score frozen mappings on held-out test evidence and emit their floors."""
    primitives, constructors = _alignment_evidence(
        predictions,
        gold,
        grammar,
        conventions,
        split=fitted.primitives.settings.score_on,
    )
    constructor_score = None
    if constructors is not None:
        if fitted.constructors is None:
            msg = "test lexical types contain constructors absent from development"
            raise ValueError(msg)
        constructor_score = score_declared_alignment(
            fitted.constructors, constructors, seed=seed
        )
    return LexicalAlignmentScore(
        score_declared_alignment(fitted.primitives, primitives, seed=seed),
        constructor_score,
    )


def stable_lexical_alignments(
    fitted: Sequence[FittedLexicalAlignment],
    /,
    *,
    seed: int,
) -> LexicalAlignmentStability:
    """Return primitive and constructor mapping stability across run seeds."""
    held = tuple(fitted)
    if not held:
        msg = "lexical alignment stability has no fitted runs"
        raise ValueError(msg)
    constructor_runs = tuple(
        entry.constructors for entry in held if entry.constructors is not None
    )
    if constructor_runs and len(constructor_runs) != len(held):
        msg = "constructors occur in only some fitted lexical alignments"
        raise ValueError(msg)
    return LexicalAlignmentStability(
        stable_declared_alignments(
            tuple(entry.primitives for entry in held), seed=seed
        ),
        (
            None
            if not constructor_runs
            else stable_declared_alignments(constructor_runs, seed=seed)
        ),
    )


def lexical_relabeling_base(
    fitted: FittedLexicalAlignment,
    /,
    *,
    ambiguity_budget: int,
) -> RelabelingBasePrediction:
    """Return the group order and clamp base implied by fitted inventories."""
    ambiguity = ensure_positive("an ambiguity budget", ambiguity_budget)
    primitives = fitted.primitives.alignment.contingency.induced_labels
    constructors = (
        ()
        if fitted.constructors is None
        else fitted.constructors.alignment.contingency.induced_labels
    )
    return predict_relabeling_base(
        primitive_count=len(primitives),
        constructor_count=len(constructors),
        ambiguity_budget=ambiguity,
    )


class LexicalTypeF1:
    """Scores per-word multiset F1 of predicted against gold lexical types.

    The value is the mean, over word types, of the multiset F1 between the types
    predicted at a word's occurrences and its gold types there, computed after the
    frozen primitive and constructor alignments. Punctuation tokens are skipped
    when the conventions strip punctuation, and an item shorter than the minimum
    length is not scored.

    Parameters
    ----------
    conventions : ConventionVector
        The conventions tokens are scored and reported under.
    store : TermStore
        The store both the predicted and the gold types are interned in.
    primitive_alignment : Alignment
        Frozen primitive alignment.
    constructor_alignment : Alignment | None
        Frozen constructor alignment, or ``None`` when development contained no
        constructor observation.
    """

    __slots__ = (
        "_constructor_alignment",
        "_conventions",
        "_items",
        "_primitive_alignment",
        "_sentences",
        "_store",
    )

    def __init__(
        self,
        conventions: ConventionVector,
        store: TermStore,
        *,
        primitive_alignment: Alignment,
        constructor_alignment: Alignment | None,
    ) -> None:
        self._conventions = conventions
        self._store = store
        self._primitive_alignment = primitive_alignment
        self._constructor_alignment = constructor_alignment
        self._sentences = Resampling("sentence")
        self._items: dict[
            ItemId, tuple[tuple[Term, ...], LexicalAnalysis, _Scored | None]
        ] = {}

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "lexical-type-f1"

    @property
    def conventions(self) -> ConventionVector:
        """Return the scoring conventions."""
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        """Return whether update order affects the aggregate."""
        return True

    def reset(self) -> None:
        """Forget every item scored so far."""
        self._items = {}

    def update(
        self, pred: tuple[Term, ...], gold: LexicalAnalysis, *, key: ItemId
    ) -> None:
        """Score one item's predicted types against its gold analysis.

        Unequal token counts are retained. Under zero-coverage scoring, absent
        types count as errors; under ignore-coverage scoring, only paired token
        positions contribute.

        Parameters
        ----------
        pred : tuple[Term, ...]
            The predicted type of each token.
        gold : LexicalAnalysis
            The gold analysis.
        key : ItemId
            Item identifier.

        Raises
        ------
        ValueError
            If the key was updated with different values.
        """
        standing = self._items.get(key)

        if standing is not None:
            if standing[:2] != (pred, gold):
                msg = (
                    f"the item {key!r} was already scored against a different analysis"
                )
                raise ValueError(msg)

            return

        self._items[key] = (pred, gold, self._scored(pred, gold))

    def compute(self) -> MetricResult:
        """Return the aggregate over every item scored so far.

        Returns
        -------
        MetricResult
            Mean word-level multiset F1 with token accuracy beside it, or an
            absent result when no item is scored.
        """
        scored = tuple(
            entry
            for _, (_, _, entry) in sorted(self._items.items())
            if entry is not None
        )

        if not scored:
            return MetricResult(None, 0, self._sentences)

        recovery = self._recovery(scored)
        per_sentence = tuple(self._recovery((entry,)).type_f1 for entry in scored)
        sentence_score = fmean(score for score in per_sentence if score is not None)
        value = (
            sentence_score
            if self._conventions.aggregation == "sentence"
            else recovery.type_f1
        )

        return MetricResult(
            value,
            len(scored),
            self._sentences,
            components={
                "token_accuracy": cast("float", recovery.token_accuracy),
                "words": float(recovery.words),
                "tokens": float(recovery.tokens),
                "dropped": float(len(self._items) - len(scored)),
            },
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return each scored item's word-level multiset F1.

        Returns
        -------
        Mapping[ItemId, MetricResult]
            Results keyed by item identifier, in ascending key order.
        """
        results: dict[ItemId, MetricResult] = {}

        for key, (_, _, entry) in sorted(self._items.items()):
            if entry is None:
                continue

            recovery = self._recovery((entry,))
            results[key] = MetricResult(
                recovery.type_f1,
                1,
                self._sentences,
                components={"token_accuracy": cast("float", recovery.token_accuracy)},
            )

        return MappingProxyType(results)

    def _scored(self, pred: tuple[Term, ...], gold: LexicalAnalysis) -> _Scored | None:
        """Return the tokens an item contributes under the conventions."""
        return _scored_entry(pred, gold, self._conventions)

    def _recovery(self, scored: tuple[_Scored, ...]) -> LexicalTypeRecovery:
        """Return the aligned lexical recovery over scored items."""
        predicted: dict[str, list[Term]] = {}
        gold: dict[str, list[Term]] = {}
        paired_tokens: list[tuple[Term, Term]] = []

        for entry in scored:
            for word in entry.gold_words:
                predicted.setdefault(word, [])
                gold.setdefault(word, [])

            for word, induced in zip(
                entry.predicted_words, entry.predicted, strict=True
            ):
                predicted.setdefault(word, []).append(induced)

            for word, expected in zip(entry.gold_words, entry.gold, strict=True):
                gold.setdefault(word, []).append(expected)

            paired_tokens.extend(zip(entry.predicted, entry.gold, strict=False))

        predicted_tokens = tuple(
            term for entry in scored for term in (*entry.predicted, *entry.extra)
        )
        gold_tokens = tuple(term for entry in scored for term in entry.gold)

        if self._conventions.coverage == "zero":
            return _zero_coverage_recovery(
                predicted,
                gold,
                predicted_tokens,
                gold_tokens,
                paired_tokens,
                self._store,
                self._primitive_alignment,
                self._constructor_alignment,
            )

        return lexical_type_recovery(
            predicted,
            gold,
            predicted_tokens,
            gold_tokens,
            (self._store, self._store),
            primitive_alignment=self._primitive_alignment,
            constructor_alignment=self._constructor_alignment,
        )

    @override
    def __repr__(self) -> str:
        """Return a summary naming the conventions and the scored item count."""
        return f"LexicalTypeF1({self._conventions!r}, {len(self._items)} items)"


class LexicalTypeEvaluator:
    """Scores the lexical type a trained parser's readout gives each token.

    The predicted type of a token is the highest-weight complete term in the
    readout's decoded frontier at the token's single-word cell. The gold type is
    the category of the gold derivation's leaf over it.

    Parameters
    ----------
    grammar : GrammarAssembly
        The grammar the parser was trained under.
    readout : str
        The term-readout role whose distribution is read.
    conventions : ConventionVector
        The conventions tokens are scored and reported under.
    """

    __slots__ = (
        "_alignment",
        "_conventions",
        "_fitted",
        "_grammar",
        "_metrics",
        "_readout",
    )

    def __init__(
        self,
        grammar: GrammarAssembly,
        *,
        readout: str,
        conventions: ConventionVector,
        alignment: AlignmentSettings,
    ) -> None:
        declared_conventions = conventions.extended(
            {
                "alignment": alignment.kind,
                "alignment_tie_break": alignment.tie_break,
                "alignment_fit_on": alignment.fit_on,
                "alignment_score_on": alignment.score_on,
                "random_floor_replicates": alignment.floor_replicates,
                "random_floor_level": alignment.floor_level,
                "stability_replicates": alignment.stability_replicates,
                "stability_level": alignment.stability_level,
            }
        )
        self._grammar = grammar
        self._readout = ReadoutId(
            ensure_name("a lexical-type term-readout role", readout)
        )
        self._alignment = alignment
        self._conventions = declared_conventions
        self._fitted: FittedLexicalAlignment | None = None
        self._metrics: tuple[LexicalTypeF1, ...] = ()

    @property
    def conventions(self) -> ConventionVector:
        """Return the conventions tokens are scored and reported under."""
        return self._conventions

    @property
    def metrics(self) -> tuple[LexicalTypeF1, ...]:
        """Return the lexical-type metric in report order."""
        if not self._metrics:
            msg = "lexical-type evaluation requires alignment fitted on development"
            raise ValueError(msg)
        return self._metrics

    @property
    def alignment(self) -> AlignmentSettings:
        """Return the declared development-fitted alignment settings."""
        return self._alignment

    @property
    def fitted_alignment(self) -> FittedLexicalAlignment | None:
        """Return the frozen development mapping, if it has been fitted."""
        return self._fitted

    def fit_alignment(
        self,
        predictions: Mapping[ItemId, tuple[Term, ...]],
        gold: Mapping[ItemId, LexicalAnalysis],
        /,
    ) -> FittedLexicalAlignment:
        """Fit the declared mapping on development and prepare test scoring."""
        fitted = fit_lexical_alignment(
            predictions,
            gold,
            self._grammar,
            self._conventions,
            self._alignment,
        )
        self._fitted = fitted
        self._metrics = (
            LexicalTypeF1(
                self._conventions,
                self._grammar.store,
                primitive_alignment=fitted.primitives.alignment,
                constructor_alignment=(
                    None
                    if fitted.constructors is None
                    else fitted.constructors.alignment
                ),
            ),
        )
        return fitted

    def score_alignment(
        self,
        predictions: Mapping[ItemId, tuple[Term, ...]],
        gold: Mapping[ItemId, LexicalAnalysis],
        /,
        *,
        seed: int,
    ) -> LexicalAlignmentScore:
        """Return held-out alignment accuracies and random-labeling floors."""
        if self._fitted is None:
            msg = "lexical-type evaluation requires alignment fitted on development"
            raise ValueError(msg)
        return score_lexical_alignment(
            self._fitted,
            predictions,
            gold,
            self._grammar,
            self._conventions,
            seed=seed,
        )

    def relabeling_base(self, *, ambiguity_budget: int) -> RelabelingBasePrediction:
        """Return the declared relabeling group's predicted clamp base."""
        if self._fitted is None:
            msg = "lexical-type evaluation requires alignment fitted on development"
            raise ValueError(msg)
        return lexical_relabeling_base(self._fitted, ambiguity_budget=ambiguity_budget)

    def held_out(
        self, data: HeldOutData, split: str, /
    ) -> EvalDataset[LexicalAnalysis]:
        """Return a retained partition's derivations as this evaluator reads them."""
        return derivation_dataset(self.annotate, data.derivations(split))

    def annotate(self, item: Item, derivation: Derivation[str], /) -> LexicalAnalysis:
        """Return the gold lexical type of every token of an item.

        Parameters
        ----------
        item : Item
            The held-out item.
        derivation : Derivation[str]
            Its gold derivation.

        Returns
        -------
        LexicalAnalysis
            The item's words beside the category of the leaf over each.

        Raises
        ------
        ValueError
            If the derivation's leaves do not cover each token exactly once.
        """
        leaves = sorted(derivation.leaves(), key=lambda leaf: leaf.span.start)
        words = tuple(token.surface for token in item.tokens)
        spans = tuple((leaf.span.start, leaf.span.end) for leaf in leaves)

        if spans != tuple((position, position + 1) for position in range(len(words))):
            msg = (
                f"the gold derivation of {item.item_id!r} has leaves {spans}, which "
                f"do not cover its {len(words)} tokens one at a time"
            )
            raise ValueError(msg)

        store = self._grammar.store
        syntax = self._grammar.syntax

        return LexicalAnalysis(
            words,
            tuple(syntax.parse(store, leaf.label) for leaf in leaves),
            frozenset(),
        )

    def predict[BatchT: HeldOutItems](
        self, outputs: ObjectiveOutputs[BatchT], /
    ) -> Mapping[ItemId, tuple[Term, ...]]:
        """Return the highest-weight decoded complete term at every token.

        Parameters
        ----------
        outputs : ObjectiveOutputs[BatchT]
            The published forward pass over one held-out batch.

        Returns
        -------
        Mapping[ItemId, tuple[Term, ...]]
            Each item's predicted type at each token.

        Raises
        ------
        ValueError
            If the readout is not decoded or a token's cell has an empty frontier.
        """
        published = outputs.terms
        decoded = published.get(self._readout)

        if decoded is None:
            msg = (
                f"lexical types read decoded terms from {self._readout!r}, and "
                f"the pass decoded {sorted(published)}"
            )
            raise ValueError(msg)

        spans = decoded.spans
        predictions: dict[ItemId, tuple[Term, ...]] = {}

        for row, (item_id, words) in enumerate(
            zip(outputs.batch.item_ids, outputs.batch.words, strict=True)
        ):
            predicted: list[Term] = []

            for position, _form in enumerate(words):
                cell = spans.span_id(position, position + 1)
                try:
                    frontier = decoded.at(row, cell)
                except KeyError:
                    msg = (
                        f"the readout {self._readout!r} did not decode the cell of "
                        f"token {position} of {item_id!r}"
                    )
                    raise ValueError(msg) from None
                if not frontier.terms:
                    msg = (
                        f"the readout {self._readout!r} decoded no type for token "
                        f"{position} of {item_id!r}"
                    )
                    raise ValueError(msg)
                predicted.append(
                    frontier.terms[int(frontier.log_weights.detach().argmax())]
                )

            predictions[item_id] = tuple(predicted)

        return MappingProxyType(predictions)

    @override
    def __repr__(self) -> str:
        """Return a summary naming the readout and the conventions."""
        return (
            f"LexicalTypeEvaluator(readout={self._readout!r}, "
            f"conventions={self._conventions!r})"
        )


def _build(
    settings: LexicalTypeEvaluatorSettings,
    context: HeldOutEvaluationContext,
    /,
) -> LexicalTypeEvaluator:
    """Build a held-out lexical-type evaluator from its declared settings.

    Parameters
    ----------
    settings : LexicalTypeEvaluatorSettings
        The declared readout role and conventions.
    context : HeldOutEvaluationContext
        The held-out resources, including the trained grammar.
    """
    return LexicalTypeEvaluator(
        context.grammar,
        readout=settings.readout,
        conventions=ConventionVector.from_settings(settings.conventions),
        alignment=settings.alignment,
    )


PLUGIN: Final = ComponentPlugin(spec=LexicalTypeEvaluatorSpec, build=_build)
