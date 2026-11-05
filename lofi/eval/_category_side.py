"""Category-side recovery for logical-form induction runs."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, cast, final, override

from lofi.eval._held_out import HeldOutEvaluationContext, derivation_dataset
from lofi.eval._lexical_diagnostics import (
    LexicalDiagnosticGold,
    LexicalDiagnosticPrediction,
    LexicalDiagnosticsEvaluator,
)
from lofi.eval._lexical_type_evaluator import (
    FittedLexicalAlignment,
    LexicalAlignmentScore,
    LexicalAnalysis,
    LexicalTypeEvaluator,
)
from lofi.eval._structural_diagnostics import (
    StructuralDiagnosticsEvaluator,
    StructuralGold,
    StructuralPrediction,
)
from lofi.eval._values import ConventionVector
from lofi.registry import ComponentPlugin
from lofi.spec.evaluators.category_side import (
    CategorySideEvaluatorSettings,
    CategorySideEvaluatorSpec,
)
from lofi.spec.evaluators.lexical_diagnostics import LexicalDiagnosticsSettings
from lofi.spec.evaluators.structural_diagnostics import StructuralDiagnosticsSettings
from lofi.terms import Term

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import ItemId
    from lofi.data import Derivation, EvalDataset, Item
    from lofi.deduce import AStarProposerFactory, ParseSettings
    from lofi.eval._ports import HeldOutData, HeldOutItems
    from lofi.eval._values import MetricResult
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.spec.evaluators.alignment import AlignmentSettings
    from lofi.spec.grammar import GrammarAssembly

__all__ = [
    "PLUGIN",
    "CategorySideEvaluator",
    "CategorySideGold",
    "CategorySidePrediction",
]


class CategorySidePrediction(NamedTuple):
    """Supertag, derivational, and lexical-type views of one category readout."""

    supertags: LexicalDiagnosticPrediction
    structure: StructuralPrediction
    lexical_types: tuple[Term, ...]


class CategorySideGold(NamedTuple):
    """The three category analyses projected from one retained derivation."""

    supertags: LexicalDiagnosticGold
    structure: StructuralGold
    lexical_types: LexicalAnalysis


class _CategoryBuildContext(Protocol):
    """The grammar capability shared by sentence and inference contexts."""

    @property
    def grammar(self) -> GrammarAssembly:
        """Return the grammar used to draw the scored categories."""

    @property
    def parse(self) -> ParseSettings[object]:
        """Return the run's declared symbolic parse."""


class _InnerMetric[Prediction, Gold](Protocol):
    """A score family consumed through one category-side projection."""

    @property
    def name(self) -> str:
        """Return the report row name."""

    @property
    def order_invariant(self) -> bool:
        """Return whether insertion order changes the aggregate."""

    def reset(self) -> None:
        """Forget accumulated items."""

    def update(self, pred: Prediction, gold: Gold, *, key: ItemId) -> None:
        """Accumulate one projected item."""

    def compute(self) -> MetricResult:
        """Return the aggregate score."""

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return one score per item."""


class _ProjectedMetric[Prediction, Gold]:
    """Expose an existing metric under the shared category-side conventions."""

    __slots__ = ("_conventions", "_inner")

    def __init__(
        self,
        inner: _InnerMetric[Prediction, Gold],
        conventions: ConventionVector,
    ) -> None:
        self._inner = inner
        self._conventions = conventions

    @property
    def name(self) -> str:
        """Return the wrapped report row name."""
        return self._inner.name

    @property
    def conventions(self) -> ConventionVector:
        """Return the complete category-side convention vector."""
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        """Return whether the wrapped aggregate is order invariant."""
        return self._inner.order_invariant

    def reset(self) -> None:
        """Forget every accumulated projected item."""
        self._inner.reset()

    def compute(self) -> MetricResult:
        """Return the wrapped aggregate."""
        return self._inner.compute()

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return wrapped scores by item identifier."""
        return self._inner.per_item()


@final
class _SupertagMetric(
    _ProjectedMetric[LexicalDiagnosticPrediction, LexicalDiagnosticGold]
):
    """Project a category-side item into supertagging evaluation."""

    def update(
        self,
        pred: CategorySidePrediction,
        gold: CategorySideGold,
        *,
        key: ItemId,
    ) -> None:
        """Accumulate one item's lexical beams and gold leaf categories."""
        self._inner.update(pred.supertags, gold.supertags, key=key)


@final
class _DependencyMetric(_ProjectedMetric[StructuralPrediction, StructuralGold]):
    """Project a category-side item into labeled-dependency evaluation."""

    def update(
        self,
        pred: CategorySidePrediction,
        gold: CategorySideGold,
        *,
        key: ItemId,
    ) -> None:
        """Accumulate one item's decoded and retained derivation projections."""
        self._inner.update(pred.structure, gold.structure, key=key)


@final
class _LexicalTypeMetric(_ProjectedMetric[tuple[Term, ...], LexicalAnalysis]):
    """Project a category-side item into aligned lexical-type evaluation."""

    def update(
        self,
        pred: CategorySidePrediction,
        gold: CategorySideGold,
        *,
        key: ItemId,
    ) -> None:
        """Accumulate one item's selected and gold lexical types."""
        self._inner.update(pred.lexical_types, gold.lexical_types, key=key)


type _CategoryMetric = _SupertagMetric | _DependencyMetric | _LexicalTypeMetric


@final
class CategorySideEvaluator:
    """Score one category readout against inference-side gold derivations."""

    __slots__ = (
        "_alignment",
        "_conventions",
        "_lexical_types",
        "_metrics",
        "_structure",
        "_supertags",
    )

    def __init__(
        self,
        context: _CategoryBuildContext,
        settings: CategorySideEvaluatorSettings,
        /,
    ) -> None:
        held_out = HeldOutEvaluationContext(
            context.grammar,
            interpretation=None,
            forms={},
            coherence={},
            target_sets={},
            lexical_training={},
            parse=context.parse,
            training_items=None,
        )
        proposer = cast("AStarProposerFactory", settings.astar.proposer)
        conventions = ConventionVector.from_settings(settings.conventions).extended(
            {
                "category_readout": settings.readout,
                "supertag_beta": settings.supertag_beta,
                "dependency_interpretation": settings.dependency_interpretation,
                "dependency_mode": settings.dependency_mode,
                "astar_strategy": settings.astar.strategy.name,
                "astar_root_mode": settings.astar.root_mode.name,
                "parse_semiring": context.parse.semiring.name,
                "astar_heuristic": settings.astar.heuristic.name,
                "astar_proposer": proposer.name,
                **settings.decode.conventions,
                "alignment": settings.alignment.kind,
                "alignment_tie_break": settings.alignment.tie_break,
                "alignment_fit_on": settings.alignment.fit_on,
                "alignment_score_on": settings.alignment.score_on,
                "random_floor_replicates": settings.alignment.floor_replicates,
                "random_floor_level": settings.alignment.floor_level,
                "stability_replicates": settings.alignment.stability_replicates,
                "stability_level": settings.alignment.stability_level,
            }
        )
        dependency_metric = settings.metrics[1]
        self._supertags = LexicalDiagnosticsEvaluator(
            held_out,
            LexicalDiagnosticsSettings(
                readout=settings.readout,
                beta=settings.supertag_beta,
                clustering_beta=1.0,
                top_k=1,
                frequency_strata=(),
                ambiguity_strata=(),
                word_classes=(),
                case_studies=(),
                conventions=settings.conventions,
                metrics=(settings.metrics[0],),
            ),
        )
        self._structure = StructuralDiagnosticsEvaluator(
            held_out,
            StructuralDiagnosticsSettings(
                readout=settings.readout,
                astar=settings.astar,
                decode=settings.decode,
                derivation_counts=False,
                dependency_interpretation=settings.dependency_interpretation,
                labels=(),
                lengths=(),
                depths=(),
                attachment_band="all",
                attachment_limit=None,
                attachment_direction="directed",
                dependency_mode=settings.dependency_mode,
                dependency_measure="f-score",
                conventions=settings.conventions,
                attachment_row="attachment:all:directed",
                dependency_row=dependency_metric,
                metrics=(dependency_metric,),
            ),
        )
        self._lexical_types = LexicalTypeEvaluator(
            context.grammar,
            readout=settings.readout,
            conventions=ConventionVector.from_settings(settings.conventions),
            alignment=settings.alignment,
        )
        self._alignment = settings.alignment
        self._conventions = conventions
        self._metrics: tuple[_CategoryMetric, ...] = ()

    @property
    def conventions(self) -> ConventionVector:
        """Return the complete conventions shared by every category-side row."""
        return self._conventions

    @property
    def metrics(self) -> tuple[_CategoryMetric, ...]:
        """Return the headline metrics after development alignment is frozen."""
        if not self._metrics:
            msg = "category-side evaluation requires alignment fitted on development"
            raise ValueError(msg)
        return self._metrics

    @property
    def alignment(self) -> AlignmentSettings:
        """Return the exact development and test alignment partitions."""
        return self._alignment

    @property
    def fitted_alignment(self) -> FittedLexicalAlignment | None:
        """Return the lexical alignment frozen on development, if fitted."""
        return self._lexical_types.fitted_alignment

    def fit_alignment(
        self,
        predictions: Mapping[ItemId, CategorySidePrediction],
        gold: Mapping[ItemId, CategorySideGold],
        /,
    ) -> FittedLexicalAlignment:
        """Fit lexical label correspondence and prepare all headline metrics."""
        fitted = self._lexical_types.fit_alignment(
            {
                item_id: prediction.lexical_types
                for item_id, prediction in predictions.items()
            },
            {item_id: analysis.lexical_types for item_id, analysis in gold.items()},
        )
        self._metrics = (
            _SupertagMetric(self._supertags.metrics[0], self._conventions),
            _DependencyMetric(self._structure.metrics[0], self._conventions),
            _LexicalTypeMetric(self._lexical_types.metrics[0], self._conventions),
        )
        return fitted

    def score_alignment(
        self,
        predictions: Mapping[ItemId, CategorySidePrediction],
        gold: Mapping[ItemId, CategorySideGold],
        /,
        *,
        seed: int,
    ) -> LexicalAlignmentScore:
        """Return frozen lexical alignment accuracy and random floors on test."""
        return self._lexical_types.score_alignment(
            {
                item_id: prediction.lexical_types
                for item_id, prediction in predictions.items()
            },
            {item_id: analysis.lexical_types for item_id, analysis in gold.items()},
            seed=seed,
        )

    def held_out(
        self, data: HeldOutData, split: str, /
    ) -> EvalDataset[CategorySideGold]:
        """Return a retained partition's pair-side derivations as gold categories."""
        return derivation_dataset(self.annotate, data.side_derivations(split))

    def annotate(self, item: Item, derivation: Derivation[str], /) -> CategorySideGold:
        """Project one retained derivation into all three gold analyses."""
        return CategorySideGold(
            self._supertags.annotate(item, derivation),
            self._structure.annotate(item, derivation),
            self._lexical_types.annotate(item, derivation),
        )

    def predict[BatchT: HeldOutItems](
        self,
        outputs: ObjectiveOutputs[BatchT],
        /,
    ) -> Mapping[ItemId, CategorySidePrediction]:
        """Read all category predictions from one shared published pass."""
        supertags = self._supertags.predict(outputs)
        structure = self._structure.predict(outputs)
        lexical_types = self._lexical_types.predict(outputs)
        expected = set(outputs.batch.item_ids)
        inventories = (set(supertags), set(structure), set(lexical_types))
        if any(inventory != expected for inventory in inventories):
            msg = "category-side projections disagree on their item inventory"
            raise ValueError(msg)
        return MappingProxyType(
            {
                item_id: CategorySidePrediction(
                    supertags[item_id],
                    structure[item_id],
                    lexical_types[item_id],
                )
                for item_id in outputs.batch.item_ids
            }
        )

    @override
    def __repr__(self) -> str:
        """Return a summary naming the category readout and metric count."""
        readout = self._conventions.factors["category_readout"]
        return (
            f"CategorySideEvaluator(readout={readout!r}, metrics={len(self._metrics)})"
        )


def _build(
    settings: CategorySideEvaluatorSettings,
    context: _CategoryBuildContext,
    /,
) -> CategorySideEvaluator:
    """Build the one declared category-side evaluator."""
    return CategorySideEvaluator(context, settings)


PLUGIN: Final = ComponentPlugin(spec=CategorySideEvaluatorSpec, build=_build)
