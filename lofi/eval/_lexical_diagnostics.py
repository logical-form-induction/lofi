"""Registered supertag, lexicon, case-study, and clustering diagnostics."""

from collections import Counter
from json import dumps
from math import exp, fsum, isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, override

from lofi._types import ReadoutId
from lofi.data import TokenSpan
from lofi.eval._alignment import Contingency
from lofi.eval._case_study import (
    SpanContrast,
    SpanGroup,
    induced_type_report,
)
from lofi.eval._clustering import clustering_scores
from lofi.eval._held_out import derivation_dataset
from lofi.eval._lexicon_mass import lexicon_mass_report, supertag_confusion_table
from lofi.eval._supertagging import (
    CategoryProbability,
    SupertagAccuracy,
    SupertagPrediction,
    TokenSupertags,
)
from lofi.eval._values import ConventionVector, MetricResult, Resampling
from lofi.registry import ComponentPlugin
from lofi.spec.evaluators.lexical_diagnostics import (
    CountStratumSettings,
    LexicalDiagnosticsSettings,
    LexicalDiagnosticsSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import ItemId
    from lofi.data import Derivation, EvalDataset, Item
    from lofi.eval import HeldOutEvaluationContext, HeldOutItems
    from lofi.eval._held_out import LexicalTrainingEvidence
    from lofi.eval._ports import HeldOutData
    from lofi.nn.forms import WeightedTerms
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.spec.grammar import GrammarAssembly
    from lofi.terms import Term

__all__ = ["PLUGIN", "LexicalDiagnosticsEvaluator"]


class LexicalDiagnosticPrediction(NamedTuple):
    """One sentence's words and decoded term beams at every published span."""

    words: tuple[str, ...]
    token_beams: tuple[WeightedTerms, ...]
    span_beams: tuple[tuple[TokenSpan, WeightedTerms], ...]


class LexicalDiagnosticGold(NamedTuple):
    """One source item and its gold lexical categories."""

    item: Item
    categories: tuple[Term, ...]

    @property
    def words(self) -> tuple[str, ...]:
        """Return the source item's token surfaces."""
        return tuple(token.surface for token in self.item.tokens)


type _Pair = tuple[LexicalDiagnosticPrediction, LexicalDiagnosticGold]


def _ranked(beam: WeightedTerms, /) -> tuple[tuple[Term, float], ...]:
    """Return finite candidate weights in descending shared-score order."""
    pairs = tuple(
        (term, float(weight))
        for term, weight in zip(beam.terms, beam.log_weights.detach(), strict=True)
        if isfinite(float(weight))
    )
    return tuple(sorted(pairs, key=lambda pair: -pair[1]))


def _posterior(beam: WeightedTerms, /) -> TokenSupertags | None:
    """Return a normalized posterior over the finite candidates, if any."""
    ranked = _ranked(beam)
    if not ranked:
        return None
    peak = ranked[0][1]
    masses = tuple(exp(weight - peak) for _, weight in ranked)
    total = fsum(masses)
    return TokenSupertags(
        tuple(
            CategoryProbability(term, mass / total)
            for (term, _), mass in zip(ranked, masses, strict=True)
        )
    )


def _pairs(items: Mapping[ItemId, _Pair], /) -> tuple[_Pair, ...]:
    """Return stored sentence pairs in stable item order."""
    return tuple(pair for _, pair in sorted(items.items()))


def _tokens(
    items: Mapping[ItemId, _Pair], /
) -> tuple[tuple[str, WeightedTerms, Term], ...]:
    """Flatten stored sentences to word, beam, gold triples."""
    return tuple(
        (word, beam, gold)
        for prediction, reference in _pairs(items)
        for word, beam, gold in zip(
            prediction.words,
            prediction.token_beams,
            reference.categories,
            strict=True,
        )
    )


def _ratio(numerator: float, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


class _LexicalDiagnosticMetric:
    """One independently reportable view of the shared lexical battery."""

    __slots__ = (
        "_ambiguity_strata",
        "_beta",
        "_case_study_by_metric",
        "_classes",
        "_clustering_beta",
        "_conventions",
        "_frequency_strata",
        "_grammar",
        "_items",
        "_name",
        "_top_k",
        "_training",
    )

    def __init__(
        self,
        name: str,
        grammar: GrammarAssembly,
        conventions: ConventionVector,
        *,
        beta: float,
        clustering_beta: float,
        top_k: int,
        frequency_strata: tuple[tuple[str, CountStratumSettings], ...],
        ambiguity_strata: tuple[tuple[str, CountStratumSettings], ...],
        training: Mapping[str, LexicalTrainingEvidence],
        word_classes: tuple[tuple[str, tuple[str, ...]], ...],
        case_studies: tuple[tuple[str, tuple[tuple[str, tuple[str, ...]], ...]], ...],
    ) -> None:
        self._name = name
        self._grammar = grammar
        self._conventions = conventions
        self._beta = beta
        self._clustering_beta = clustering_beta
        self._top_k = top_k
        self._frequency_strata = frequency_strata
        self._ambiguity_strata = ambiguity_strata
        self._training = training
        self._classes = word_classes
        self._case_study_by_metric = {
            f"case-study-{contrast}-{group}": (contrast, group, groups)
            for contrast, groups in case_studies
            for group, _ in groups
        }
        self._items: dict[ItemId, _Pair] = {}

    @property
    def name(self) -> str:
        """Return the report-row name."""
        return self._name

    @property
    def conventions(self) -> ConventionVector:
        """Return the complete declared convention vector."""
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        """Return that item order does not affect the report."""
        return True

    def reset(self) -> None:
        """Forget every accumulated sentence."""
        self._items = {}

    def update(
        self,
        pred: LexicalDiagnosticPrediction,
        gold: LexicalDiagnosticGold,
        *,
        key: ItemId,
    ) -> None:
        """Store one sentence after checking token alignment."""
        if pred.words != gold.words:
            msg = f"lexical diagnostic words disagree for {key!r}"
            raise ValueError(msg)
        if len(pred.token_beams) != len(gold.categories):
            msg = f"lexical diagnostic token counts disagree for {key!r}"
            raise ValueError(msg)
        pair = (pred, gold)
        standing = self._items.get(key)
        if standing is not None and standing != pair:
            msg = f"lexical diagnostics already saw another prediction for {key!r}"
            raise ValueError(msg)
        self._items[key] = pair

    def _result(
        self,
        value: float | None,
        n: int,
        *,
        unit: str = "token",
        components: Mapping[str, float] | None = None,
    ) -> MetricResult:
        return MetricResult(
            value,
            n,
            Resampling(unit),
            components=components,
        )

    def _clustering(self, name: str) -> MetricResult:
        tokens = _tokens(self._items)
        counts: Counter[tuple[str, str]] = Counter()
        for _, beam, gold in tokens:
            ranked = _ranked(beam)
            if ranked:
                counts[(repr(ranked[0][0].key()), repr(gold.key()))] += 1
        score = clustering_scores(
            Contingency(counts, split="test"),
            unit="token",
            beta=self._clustering_beta,
        )[name]
        return score.result

    def _lexicon_mass(self, word_class: str) -> MetricResult:
        forms = dict(self._classes)[word_class]
        observations = tuple(
            (word_class, posterior, gold)
            for word, beam, gold in _tokens(self._items)
            if word in forms and (posterior := _posterior(beam)) is not None
        )
        if not observations:
            return self._result(None, 0)
        row = lexicon_mass_report(
            self._grammar.store,
            self._grammar.syntax,
            observations,
            k=self._top_k,
        )[0]
        return self._result(
            row.gold_attested_mass,
            len(observations),
            components={
                "top_categories": float(len(row.top_categories)),
                **{
                    f"top:{position}:{category}": mass
                    for position, (category, mass) in enumerate(
                        row.top_categories, start=1
                    )
                },
            },
        )

    def _case_study(self) -> MetricResult:
        contrast_name, group_name, declared = self._case_study_by_metric[self._name]
        contrast = SpanContrast(
            contrast_name,
            tuple(SpanGroup(name, forms) for name, forms in declared),
        )
        entries = _pairs(self._items)
        posteriors = {
            (gold.item.item_id, span): posterior
            for prediction, gold in entries
            for span, beam in prediction.span_beams
            if (posterior := _posterior(beam)) is not None
        }
        reports = induced_type_report(
            tuple(gold.item for _, gold in entries), contrast, posteriors
        )
        report = next(row for row in reports if row.group.name == group_name)
        masses = {f"mass:{key!r}": mass for key, mass in report.distribution}
        return self._result(
            max(masses.values(), default=None),
            report.sentence_count,
            unit="sentence",
            components={
                "occurrences": float(report.occurrence_count),
                "categories": float(len(report.distribution)),
                **masses,
            },
        )

    def _supertag(self, row: str) -> MetricResult:
        scorer = SupertagAccuracy(
            self._grammar.store,
            self._grammar.syntax,
            beta=self._beta,
        )
        for key, (prediction, gold) in sorted(self._items.items()):
            posteriors = tuple(_posterior(beam) for beam in prediction.token_beams)
            if any(posterior is None for posterior in posteriors):
                msg = f"item {key!r} has a token without a finite term posterior"
                raise ValueError(msg)
            scorer.update(
                SupertagPrediction(
                    tuple(
                        posterior for posterior in posteriors if posterior is not None
                    ),
                    None,
                ),
                gold.categories,
                key=key,
            )
        return scorer.rows()[row]

    def _stratified_supertag(
        self, dimension: Literal["frequency", "ambiguity"], stratum: str
    ) -> MetricResult:
        """Return tagger accuracy for one training-evidence count band."""
        declared = dict(
            self._frequency_strata
            if dimension == "frequency"
            else self._ambiguity_strata
        )[stratum]
        scorer = SupertagAccuracy(
            self._grammar.store,
            self._grammar.syntax,
            beta=self._beta,
        )
        for key, (prediction, gold) in sorted(self._items.items()):
            selected: list[int] = []
            for position, word in enumerate(prediction.words):
                evidence = self._training.get(word)
                if evidence is None:
                    count = 0
                elif dimension == "frequency":
                    count = evidence.frequency
                else:
                    count = evidence.ambiguity
                if declared.at_least <= count and (
                    declared.at_most is None or count <= declared.at_most
                ):
                    selected.append(position)
            positions = tuple(selected)
            if not positions:
                continue
            posteriors = tuple(
                _posterior(prediction.token_beams[position]) for position in positions
            )
            if any(posterior is None for posterior in posteriors):
                msg = f"item {key!r} has a stratified token without a posterior"
                raise ValueError(msg)
            scorer.update(
                SupertagPrediction(
                    tuple(
                        posterior for posterior in posteriors if posterior is not None
                    ),
                    None,
                ),
                tuple(gold.categories[position] for position in positions),
                key=key,
            )
        return scorer.rows()["tagger-accuracy"]

    def compute(self) -> MetricResult:
        """Compute the selected library diagnostic over accumulated sentences."""
        tokens = _tokens(self._items)
        total = len(tokens)
        name = self._name
        beams = tuple(_ranked(beam) for _, beam, _ in tokens)

        if name in {
            "supertag-accuracy",
            "supertag-multi-accuracy",
            "supertag-ambiguity",
        }:
            row = {
                "supertag-accuracy": "tagger-accuracy",
                "supertag-multi-accuracy": "multi-tag-accuracy",
                "supertag-ambiguity": "ambiguity",
            }[name]
            result = self._supertag(row)

        elif name == "supertag-confusion":
            observations = tuple(
                ("all", posterior, reference)
                for (_, beam, reference) in tokens
                if (posterior := _posterior(beam)) is not None
            )
            if observations:
                table = supertag_confusion_table(
                    self._grammar.store, self._grammar.syntax, observations
                )
                diagonal = sum(
                    table.counts[index][index] for index in range(len(table.categories))
                )
                counts = {
                    f"confusion:{row}:{column}:{gold_category}->{predicted_category}": (
                        float(table.counts[row][column])
                    )
                    for row, gold_category in enumerate(table.categories)
                    for column, predicted_category in enumerate(table.categories)
                }
                result = self._result(
                    diagonal / len(observations),
                    len(observations),
                    components={
                        "categories": float(len(table.categories)),
                        "diagonal": float(diagonal),
                        **counts,
                    },
                )
            else:
                result = self._result(None, 0)

        elif name.startswith("clustering-"):
            clustering = {
                "clustering-homogeneity": "homogeneity",
                "clustering-completeness": "completeness",
                "clustering-v-measure": "v-measure",
                "clustering-adjusted-mutual-information": "adjusted-mutual-information",
                "clustering-adjusted-rand": "adjusted-rand",
            }
            result = self._clustering(clustering[name])

        elif name.startswith("supertag-accuracy-frequency-"):
            result = self._stratified_supertag(
                "frequency", name.removeprefix("supertag-accuracy-frequency-")
            )
        elif name.startswith("supertag-accuracy-ambiguity-"):
            result = self._stratified_supertag(
                "ambiguity", name.removeprefix("supertag-accuracy-ambiguity-")
            )

        elif name.startswith("lexicon-mass-"):
            result = self._lexicon_mass(name.removeprefix("lexicon-mass-"))
        elif name in self._case_study_by_metric:
            result = self._case_study()

        elif name == "collapse-rate":
            labels = tuple(beam[0][0].key() for beam in beams if beam)
            modal = max(Counter(labels).values(), default=0)
            result = self._result(
                _ratio(modal, total),
                total,
                components={"modal": float(modal)},
            )

        elif name == "decoded-residual-mass":
            residual = fsum(
                0.0
                if float(beam.log_residual) == float("-inf")
                else exp(float(beam.log_residual))
                for _, beam, _ in tokens
            )
            result = self._result(_ratio(residual, total), total)

        else:
            refused = sum(not beam for beam in beams)
            result = self._result(
                _ratio(refused, total),
                total,
                components={"refused": float(refused)},
            )
        return result

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return this diagnostic separately for each sentence."""
        complete = self._items
        results: dict[ItemId, MetricResult] = {}
        for key, pair in sorted(complete.items()):
            self._items = {key: pair}
            results[key] = self.compute()
        self._items = complete
        return MappingProxyType(results)


class LexicalDiagnosticsEvaluator:
    """Evaluate decoded lexical beams with the complete registered battery."""

    __slots__ = ("_context", "_metrics", "_readout")

    def __init__(
        self,
        context: HeldOutEvaluationContext,
        settings: LexicalDiagnosticsSettings,
        /,
    ) -> None:
        self._context = context
        self._readout = ReadoutId(settings.readout)
        conventions = ConventionVector.from_settings(settings.conventions).extended(
            {
                "diagnostic_readout": settings.readout,
                "supertag_beta": settings.beta,
                "clustering_beta": settings.clustering_beta,
                "lexicon_mass_top_k": settings.top_k,
                "frequency_strata": dumps(
                    {
                        name: {
                            "at_least": stratum.at_least,
                            "at_most": stratum.at_most,
                        }
                        for name, stratum in settings.frequency_strata
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "ambiguity_strata": dumps(
                    {
                        name: {
                            "at_least": stratum.at_least,
                            "at_most": stratum.at_most,
                        }
                        for name, stratum in settings.ambiguity_strata
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "word_classes": dumps(
                    dict(settings.word_classes),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "case_studies": dumps(
                    {
                        contrast: dict(groups)
                        for contrast, groups in settings.case_studies
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            }
        )
        self._metrics = tuple(
            _LexicalDiagnosticMetric(
                name,
                context.grammar,
                conventions,
                beta=settings.beta,
                clustering_beta=settings.clustering_beta,
                top_k=settings.top_k,
                frequency_strata=settings.frequency_strata,
                ambiguity_strata=settings.ambiguity_strata,
                training=context.lexical_training,
                word_classes=settings.word_classes,
                case_studies=settings.case_studies,
            )
            for name in settings.metrics
        )

    @property
    def conventions(self) -> ConventionVector:
        """Return the shared and battery-specific conventions."""
        return self._metrics[0].conventions

    @property
    def metrics(self) -> tuple[_LexicalDiagnosticMetric, ...]:
        """Return every diagnostic in report order."""
        return self._metrics

    def held_out(
        self, data: HeldOutData, split: str, /
    ) -> EvalDataset[LexicalDiagnosticGold]:
        """Return a retained partition's derivations as this evaluator reads them."""
        return derivation_dataset(self.annotate, data.derivations(split))

    def annotate(
        self, item: Item, derivation: Derivation[str], /
    ) -> LexicalDiagnosticGold:
        """Return the source item and gold lexical categories in token order."""
        leaves = sorted(derivation.leaves(), key=lambda leaf: leaf.span.start)
        words = tuple(token.surface for token in item.tokens)
        spans = tuple((leaf.span.start, leaf.span.end) for leaf in leaves)
        expected = tuple((position, position + 1) for position in range(len(words)))
        if spans != expected:
            msg = f"the gold derivation of {item.item_id!r} does not tile its tokens"
            raise ValueError(msg)
        grammar = self._context.grammar
        return LexicalDiagnosticGold(
            item,
            tuple(grammar.syntax.parse(grammar.store, leaf.label) for leaf in leaves),
        )

    def predict[BatchT: HeldOutItems](
        self, outputs: ObjectiveOutputs[BatchT], /
    ) -> Mapping[ItemId, LexicalDiagnosticPrediction]:
        """Return decoded token beams and every published span posterior."""
        decoded = outputs.terms.get(self._readout)
        if decoded is None:
            msg = (
                f"lexical diagnostics read {self._readout!r}, and it was not published"
            )
            raise ValueError(msg)
        return MappingProxyType(
            {
                item_id: LexicalDiagnosticPrediction(
                    tuple(words),
                    tuple(
                        decoded.at(row, decoded.spans.span_id(position, position + 1))
                        for position in range(len(words))
                    ),
                    tuple(
                        (TokenSpan(*decoded.spans.span_of(span)), beam)
                        for (item, span), beam in decoded.cells.items()
                        if item == row
                    ),
                )
                for row, (item_id, words) in enumerate(
                    zip(outputs.batch.item_ids, outputs.batch.words, strict=True)
                )
            }
        )

    @override
    def __repr__(self) -> str:
        """Return a summary naming the readout and metric count."""
        count = len(self._metrics)
        return f"LexicalDiagnosticsEvaluator({self._readout!r}, metrics={count})"


def _build(
    settings: LexicalDiagnosticsSettings,
    context: HeldOutEvaluationContext,
    /,
) -> LexicalDiagnosticsEvaluator:
    """Build the declared lexical diagnostic evaluator."""
    return LexicalDiagnosticsEvaluator(context, settings)


PLUGIN: Final = ComponentPlugin(spec=LexicalDiagnosticsSpec, build=_build)
