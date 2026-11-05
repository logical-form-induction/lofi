"""The metrics record of one training point: what it measured and what it achieved.

A point's run directory holds one record, ``metrics.json``, written once
preparation ends and again when the point closes. It carries the point's
measured size by component role, the wall time of each phase, the peak memory
of the run device, every report the drawn corpus achieved, the constraint
selection and the length bucketing of its batches, the approximation residual of
every truncating loss term, what fitting completed, and the reports
components and stages contribute, among them the grounding it trained under.

A point attempted again after an interruption keeps what earlier attempts
recorded: its phases add up across attempts, and the fitting it resumed carries
the epochs or steps completed before the checkpoint it resumed from.

Every block is built from the values the run computed, so equal declarations
under equal seeds write byte-equal report blocks. ``metrics_row`` flattens a
record into the one row per point that ingestion tabulates beside the report
rows.
"""

import json
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast, final

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name
from lofi.data import BucketOccupancy
from lofi.data.synthetic import NegativeDataRegime
from lofi.lexicon import ClampMode, ClampScope, UnsatisfiableAction
from lofi.run._boundary.manifest import MeasuredRole, PointManifest, measure_point
from lofi.run._manifest import replace_file

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from lofi._types import JsonValue, ReportValue
    from lofi.data import SelectionReport
    from lofi.data.synthetic import (
        AmbiguityReport,
        ComplexityReport,
        ConfusionReport,
        CorruptionReport,
        Histogram,
        IllTypingReport,
        InferenceReport,
        InventoryReport,
        LengthReport,
        LexiconReport,
        NoiseReport,
        ScopeReport,
        SemanticReport,
        SyntheticCorpus,
        SyntheticInference,
    )
    from lofi.data.synthetic._values import NamedOperator
    from lofi.lexicon import ClampPolicy, ClampReport
    from lofi.run._boundary.sweep import SweepPoint
    from lofi.run._training import TrainingBlueprint
    from lofi.run._values import RunPhase
    from lofi.terms import Matching, Term
    from lofi.train import (
        EpochReport,
        EvalResult,
        FittingReport,
        ParserFittingPlan,
        PreparedTraining,
        ResumePosition,
        TrainResult,
    )

__all__ = [
    "METRICS_NAME",
    "AmbiguityRecord",
    "BucketingRecord",
    "CategoryBudgetRecord",
    "ClampSupervisionRecord",
    "ComplexityRecord",
    "ConfusionRecord",
    "ContributedValue",
    "CorpusRecord",
    "CorruptionRecord",
    "EpochMetricsRecord",
    "FittingMetricsRecord",
    "FormTypingSplitRecord",
    "HistogramRecord",
    "IllTypingRecord",
    "InferenceDataRecord",
    "InferenceLabelRecord",
    "InventoryRecord",
    "LengthRecord",
    "LexiconRecord",
    "MetricsRecord",
    "MetricsRecorder",
    "NoiseRecord",
    "OccupancyRecord",
    "ResidualRecord",
    "ScopeRecord",
    "SelectionRecord",
    "SemanticRecord",
    "StepMetricsRecord",
    "contributed_record",
    "corpus_record",
    "inference_record",
    "metrics_row",
    "read_metrics",
    "selection_record",
    "write_metrics",
]

METRICS_NAME: Final = "metrics.json"
"""The file in a run directory holding the point's metrics record."""


class HistogramRecord(dx.Model, extra="forbid"):
    """Records the counts of an integer measurement.

    Attributes
    ----------
    counts : dict[str, int]
        How often each observed value occurred, keyed by the value's decimal
        text in ascending order of value.
    total : int
        How many observations the counts hold between them.
    mean : float | None
        The mean observation, or None when nothing was observed.
    """

    counts: dict[str, int]
    total: int
    mean: float | None


class LengthRecord(dx.Model, extra="forbid"):
    """Records the sentence lengths a corpus achieved.

    Attributes
    ----------
    achieved : HistogramRecord
        Over the drawn strings: how many tokens each has.
    resampled : int
        How many drawn lengths fell outside the declared bounds and were drawn
        again.
    """

    achieved: HistogramRecord
    resampled: int


class LexiconRecord(dx.Model, extra="forbid"):
    """Records the joint of category and token frequency a corpus's lexicon achieved.

    Attributes
    ----------
    classes : dict[str, HistogramRecord]
        Over each category, by its rendered text: how many of the forms
        bearing it were drawn as its token each number of times.
    forms : HistogramRecord
        Over the forms: how many were drawn each number of times.
    realized_size : int
        How many forms the realized lexicon holds.
    realized_vocabulary_size : int
        How many forms occur in the generated strings.
    per_type_ambiguity : HistogramRecord
        Over the forms: how many categories each bears.
    ambiguity_frequency_correlation : float | None
        The rank correlation between a form's ambiguity and its token count,
        or None when either does not vary.
    type_token_ratio : float
        Drawn forms per drawn token.
    class_sizes : dict[str, int]
        How many forms bear each category.
    token_counts : dict[str, int]
        How many tokens were drawn of each category.
    token_shares : dict[str, float]
        Each category's tokens as a share of every token.
    rank_frequency_exponent : float | None
        The exponent of the fitted rank-frequency law, or None where it cannot
        be fitted.
    coupling : float | None
        How closely a category's token share follows its class size, or None
        where either does not vary.
    concordance : float | None
        The rank concordance of class size and token share, or None where
        either does not vary.
    """

    classes: dict[str, HistogramRecord]
    forms: HistogramRecord
    realized_size: int
    realized_vocabulary_size: int
    per_type_ambiguity: HistogramRecord
    ambiguity_frequency_correlation: float | None
    type_token_ratio: float
    class_sizes: dict[str, int]
    token_counts: dict[str, int]
    token_shares: dict[str, float]
    rank_frequency_exponent: float | None
    coupling: float | None
    concordance: float | None


class SemanticRecord(dx.Model, extra="forbid"):
    """Records the achieved logical-form inventory and token-frequency joint.

    Attributes
    ----------
    joint : LexiconRecord
        Logical-form classes in place of syntactic categories, with their
        inventory sizes, token counts, shares, and frequency coupling.
    arities : HistogramRecord
        Over predicate meanings: how many arguments each takes.
    per_form_meanings : HistogramRecord
        Over surface forms: how many distinct lexical meanings each bears.
    items : dict[str, int]
        How many lexical items bear each logical-form class.
    item_shares : dict[str, float]
        Each class's share of the logical-form inventory.
    symbols_used : dict[str, int]
        How many inventory symbols of each class occur in either pair side.
    cells : dict[str, dict[str, int]]
        For each category, how many lexical items bear each class.
    concordance : float | None
        Normalized mutual information between category and logical-form class.
    """

    joint: LexiconRecord
    arities: HistogramRecord
    per_form_meanings: HistogramRecord
    items: dict[str, int]
    item_shares: dict[str, float]
    symbols_used: dict[str, int]
    cells: dict[str, dict[str, int]]
    concordance: float | None


class InventoryRecord(dx.Model, extra="forbid"):
    """Records the categories a compiled grammar's lexicon uses.

    Attributes
    ----------
    categories : dict[str, int]
        The depth of each category, by its rendered text.
    argument_counts : dict[str, int]
        The functor-spine argument count of each rendered category.
    composition_orders : dict[str, int]
        The order of each rendered category under declared functor roles.
    max_arguments : int | None
        The declared argument-count bound, or no bound.
    max_order : int | None
        The declared category-order bound, or no bound.
    primitives : tuple[str, ...]
        The primitive categories the lexicon's categories use.
    realized_primitive_count : int
        How many primitive types occur in the lexicon's categories.
    size : int
        How many categories there are.
    depths : HistogramRecord
        Over the categories: how deep each is.
    arguments : HistogramRecord
        Over the categories: how many arguments each takes.
    orders : HistogramRecord
        Over the categories: the order of each category.
    """

    categories: dict[str, int]
    argument_counts: dict[str, int]
    composition_orders: dict[str, int]
    max_arguments: int | None
    max_order: int | None
    primitives: tuple[str, ...]
    realized_primitive_count: int
    size: int
    depths: HistogramRecord
    arguments: HistogramRecord
    orders: HistogramRecord


class AmbiguityRecord(dx.Model, extra="forbid"):
    """Records the ambiguity a corpus achieved.

    Attributes
    ----------
    lexical : HistogramRecord
        Over the word tokens: how many categories each token's form has.
    derivations : HistogramRecord
        Over the grammatical strings: how many root derivations each has.
    rejected : int
        How many drawn strings fell outside a target band and were drawn again.
    """

    lexical: HistogramRecord
    derivations: HistogramRecord
    rejected: int


class CorruptionRecord(dx.Model, extra="forbid"):
    """Records the corruption a corpus achieved, by kind of corruption.

    Attributes
    ----------
    accepted : dict[str, int]
        How many corrupted items of each kind were kept.
    rejected : dict[str, int]
        How many corruptions of each kind were refused, as derivable.
    inapplicable : dict[str, int]
        How many attempts of each kind found nothing to corrupt.
    sample_size : int
        How many items the corpus holds.
    corrupted : int
        How many of them are corrupted.
    achieved_proportion : float
        The corrupted items as a share of the sample.
    rejection_rate : float
        The refused corruptions as a share of every attempt.
    """

    accepted: dict[str, int]
    rejected: dict[str, int]
    inapplicable: dict[str, int]
    sample_size: int
    corrupted: int
    achieved_proportion: float
    rejection_rate: float


class IllTypingRecord(dx.Model, extra="forbid"):
    """Records the ill-typed logical forms a corpus carries.

    Attributes
    ----------
    injection : CorruptionRecord
        How the ill-typed forms were injected, by kind.
    certificates : dict[str, int]
        How many forms the type checker refused for each reason.
    regime : NegativeDataRegime
        How the certified negatives were presented to training.
    requested_noise : float
        The declared flip probability.
    observed : dict[str, int]
        How many forms training observed under each typing label.
    flipped : int
        How many certified negatives training observed as well-typed.
    achieved_noise : float
        The achieved share of flipped negative labels.
    """

    injection: CorruptionRecord
    certificates: dict[str, int]
    regime: NegativeDataRegime
    requested_noise: float
    observed: dict[str, int]
    flipped: int
    achieved_noise: float


class NoiseRecord(dx.Model, extra="forbid"):
    """Records the label noise a corpus achieved, by true label.

    Attributes
    ----------
    items : dict[str, int]
        How many items carry each true label.
    flipped : dict[str, int]
        How many of them were observed with the other label.
    observed : dict[str, int]
        How many items were observed with each label.
    sample_size : int
        How many labeled items there are.
    flip_count : int
        How many labels were flipped.
    achieved_rate : float
        The flipped labels as a share of the sample.
    achieved_rates : dict[str, float]
        The flipped share of each true label.
    """

    items: dict[str, int]
    flipped: dict[str, int]
    observed: dict[str, int]
    sample_size: int
    flip_count: int
    achieved_rate: float
    achieved_rates: dict[str, float]


class ScopeRecord(dx.Model, extra="forbid"):
    """Records the scope ambiguity a corpus achieved.

    Attributes
    ----------
    readings : HistogramRecord
        Over the strings: how many distinct readings each has.
    derivations : HistogramRecord
        Over the strings: how many root derivations each has.
    rejected : int
        How many drawn strings fell outside the target and were drawn again.
    exceeded : int
        How many strings had more derivations than the declared limit.
    undecided : int
        How many strings had readings whose equivalence went undecided.
    """

    readings: HistogramRecord
    derivations: HistogramRecord
    rejected: int
    exceeded: int
    undecided: int


class CorpusRecord(dx.Model, extra="forbid"):
    """Records every report one drawn corpus achieved.

    Attributes
    ----------
    length : LengthRecord
        The sentence lengths.
    lexicon : LexiconRecord
        The joint of category and token frequency.
    inventory : InventoryRecord
        The categories the lexicon uses.
    ambiguity : AmbiguityRecord
        The lexical and derivational ambiguity.
    corruption : CorruptionRecord | None
        The corruption, or None for a corpus declared without one.
    noise : NoiseRecord | None
        The label noise, or None for a corpus declared without any.
    ill_typing : IllTypingRecord | None
        The ill-typed logical forms, or None for a corpus declared without any.
    scope : ScopeRecord | None
        The scope ambiguity, or None for a corpus declared without a target.
    """

    length: LengthRecord
    lexicon: LexiconRecord
    inventory: InventoryRecord
    ambiguity: AmbiguityRecord
    corruption: CorruptionRecord | None
    noise: NoiseRecord | None
    ill_typing: IllTypingRecord | None
    scope: ScopeRecord | None


class ComplexityRecord(dx.Model, extra="forbid"):
    """Records the logical-form complexity an inference dataset achieved.

    Attributes
    ----------
    premise : dict[str, HistogramRecord]
        Over the premises, for each complexity control: the values it took.
    hypothesis : dict[str, HistogramRecord]
        Over the hypotheses, for each complexity control: the values it took.
    overlap : HistogramRecord
        Over the pairs: how many symbols premise and hypothesis share.
    rejected : int
        How many drawn pairs fell outside the target band.
    satisfied : bool
        Whether every pair falls in the declared band.
    """

    premise: dict[str, HistogramRecord]
    hypothesis: dict[str, HistogramRecord]
    overlap: HistogramRecord
    rejected: int
    satisfied: bool


class InferenceLabelRecord(dx.Model, extra="forbid"):
    """Records the inference labels an inference dataset achieved.

    Attributes
    ----------
    labels : dict[str, int]
        How many pairs bear each label.
    proportions : dict[str, float]
        Each decided label's share of the decided pairs.
    total : int
        How many pairs there are.
    decided : int
        How many of them were decided.
    rejected : int
        How many drawn pairs were refused to meet the declared proportion.
    satisfied : bool
        Whether the declared proportion was met.
    """

    labels: dict[str, int]
    proportions: dict[str, float]
    total: int
    decided: int
    rejected: int
    satisfied: bool


class ConfusionRecord(dx.Model, extra="forbid"):
    """Records the label noise an inference dataset achieved.

    Attributes
    ----------
    counts : dict[str, dict[str, int]]
        For each true label, how many pairs were observed with each label.
    matrix : dict[str, dict[str, float]]
        For each true label, the share observed with each label.
    decided : int
        How many pairs carry a decided true label.
    undecided : int
        How many do not.
    flip_count : int
        How many labels were changed.
    achieved_rate : float
        The changed labels as a share of the decided pairs.
    satisfied : bool
        Whether the declared confusion was met.
    """

    counts: dict[str, dict[str, int]]
    matrix: dict[str, dict[str, float]]
    decided: int
    undecided: int
    flip_count: int
    achieved_rate: float
    satisfied: bool


class InferenceDataRecord(dx.Model, extra="forbid"):
    """Records every report one drawn inference dataset achieved.

    Attributes
    ----------
    premise : CorpusRecord
        The reports of the premise corpus.
    hypothesis : CorpusRecord
        The reports of the hypothesis corpus.
    complexity : ComplexityRecord
        The logical-form complexity of the pairs.
    inference : InferenceLabelRecord
        The inference labels of the pairs.
    confusion : ConfusionRecord | None
        The label noise, or None for a dataset declared without any.
    semantic : SemanticRecord | None
        The joint achieved logical-form inventory across both pair sides, or
        None for a fixed lexical interpretation.
    """

    premise: CorpusRecord
    hypothesis: CorpusRecord
    complexity: ComplexityRecord
    inference: InferenceLabelRecord
    confusion: ConfusionRecord | None
    semantic: SemanticRecord | None


class CategoryBudgetRecord(dx.Model, extra="forbid"):
    """Records the constraint budget a selection achieved on one category.

    Attributes
    ----------
    types_pinned : int
        How many pinned forms bear the category.
    tokens_covered : int
        How many tokens those forms cover.
    mass_share : float
        Those tokens as a share of the corpus's token mass.
    """

    types_pinned: int
    tokens_covered: int
    mass_share: float


class ClampSupervisionRecord(dx.Model, extra="forbid"):
    """Records the applied clamp policy and its observed outcomes.

    Attributes
    ----------
    mode : ClampMode
        Whether incompatible analyses were filtered or penalized.
    penalty : float | None
        The soft penalty, or None for a hard clamp.
    gradient_scale : float
        The multiplier the clamp contributes to its objective gradient.
    scope : ClampScope
        Whether clamps bound token occurrences or constituents.
    matching : str
        The resolved category congruence, including a grammar default.
    meaning_matching : str
        The resolved meaning congruence.
    require_constituent : bool
        Whether named spans were required to form constituents.
    negate : bool
        Whether named labels were excluded rather than required.
    on_unsatisfiable : UnsatisfiableAction
        The action taken when no analysis satisfied a hard clamp.
    dropped_clamps : int
        How many unsatisfiable clamps were dropped.
    skipped_items : int
        How many items were skipped because a clamp was unsatisfiable.
    """

    mode: ClampMode
    penalty: float | None
    gradient_scale: float
    scope: ClampScope
    matching: str
    meaning_matching: str
    require_constituent: bool
    negate: bool
    on_unsatisfiable: UnsatisfiableAction
    dropped_clamps: int
    skipped_items: int


class SelectionRecord(dx.Model, extra="forbid"):
    """Records the constraint budget a selection achieved on the corpus.

    Attributes
    ----------
    types_pinned : int
        How many forms the selection pinned.
    tokens_covered : int
        How many tokens those forms cover.
    mass_share : float
        Those tokens as a share of the corpus's token mass.
    by_category : dict[str, CategoryBudgetRecord]
        The same quantities for each category the pinned forms bear.
    concordance : float | None
        The rank-class concordance of the corpus, or None where it does not
        vary.
    clamp : ClampSupervisionRecord
        The resolved clamp policy and its observed outcomes.
    """

    types_pinned: int
    tokens_covered: int
    mass_share: float
    by_category: dict[str, CategoryBudgetRecord]
    concordance: float | None
    clamp: ClampSupervisionRecord


class FormTypingSplitRecord(dx.Model, extra="forbid"):
    """Records the logical-form typing labels available in one data split.

    Attributes
    ----------
    examples : int
        How many logical forms the split presents to training.
    ill_typed : int
        How many of them the declared type checker certified as ill-typed.
    observed : dict[str, int]
        How many forms training observes under each typing label.
    """

    examples: int
    ill_typed: int
    observed: dict[str, int]


class OccupancyRecord(dx.Model, extra="forbid"):
    """Records how full the length-bucketed batches of one split are.

    Attributes
    ----------
    batches : int
        How many batches the split holds.
    items : int
        How many items they hold between them.
    live : int
        How many token positions hold a token.
    capacity : int
        How many token positions the batches allocate, each batch as long as
        its longest item.
    padded : int
        How many allocated positions hold padding.
    rate : float
        The live positions as a share of the allocated ones.
    form_typing : FormTypingSplitRecord | None
        Logical-form typing examples in this split, or none for an
        uninterpreted corpus.
    """

    batches: int
    items: int
    live: int
    capacity: int
    padded: int
    rate: float
    form_typing: FormTypingSplitRecord | None


class BucketingRecord(dx.Model, extra="forbid"):
    """Records the length bucketing of a run's training and validation batches.

    Attributes
    ----------
    training : OccupancyRecord
        The training split.
    validation : OccupancyRecord
        The validation split.
    """

    training: OccupancyRecord
    validation: OccupancyRecord


class ResidualRecord(dx.Model, extra="forbid"):
    """Records the approximation residual one truncating loss term reported.

    Attributes
    ----------
    steps : int
        How many training steps reported it.
    mean : float
        Its mean over those steps.
    maximum : float
        Its largest value.
    last : float
        The value the last of them reported.
    """

    steps: int
    mean: float
    maximum: float
    last: float


class StepMetricsRecord(dx.Model, extra="forbid"):
    """Records what one training step reported.

    Attributes
    ----------
    step : int
        The optimizer steps the run had taken when the step reported.
    items : int
        How many items the step consumed.
    dropped : int
        How many items it declined to consume.
    metrics : dict[str, float]
        The numbers it reported, by name.
    """

    step: int
    items: int
    dropped: int
    metrics: dict[str, float]


class EpochMetricsRecord(dx.Model, extra="forbid"):
    """Records one completed epoch: its steps and its validation metrics.

    Attributes
    ----------
    epoch : int
        The zero-based epoch.
    steps : tuple[StepMetricsRecord, ...]
        What each training step of the epoch reported, in order.
    validation_items : int | None
        How many validation items the epoch's validation consumed, or None
        before a staged trainer reaches its validation-owning stage.
    validation : dict[str, float] | None
        The item-weighted validation metrics, or None before validation.
    """

    epoch: int
    steps: tuple[StepMetricsRecord, ...]
    validation_items: int | None
    validation: dict[str, float] | None


class FittingMetricsRecord(dx.Model, extra="forbid"):
    """Records what fitting completed across every attempt of a point.

    Attributes
    ----------
    unit : Literal["epochs", "steps"]
        The unit the fitted plan's horizon is counted in.
    epochs : tuple[EpochMetricsRecord, ...]
        Every completed epoch of a run that fits in epochs, in order.
    steps : tuple[StepMetricsRecord, ...]
        Every training step of a run that fits in steps, in order.
    optimizer_steps : int
        How many optimizer steps the run completed.
    data_position : int
        How many training occurrences it consumed.
    stop : Literal["horizon", "early-stopping", "target-met", "target-not-met"]
        Why fitting ended: ``horizon`` once the declared horizon was
        completed, ``early-stopping`` once validation patience was spent.
    stopped_epoch : int | None
        The last epoch fitting completed, or None for a run that fits in steps.
    best_checkpoint : str | None
        The validation-best checkpoint, or None for a run that keeps none.
    latest_checkpoint : str
        The latest resumable checkpoint.
    """

    unit: Literal["epochs", "steps"]
    epochs: tuple[EpochMetricsRecord, ...]
    steps: tuple[StepMetricsRecord, ...]
    optimizer_steps: int
    data_position: int
    stop: Literal["horizon", "early-stopping", "target-met", "target-not-met"]
    stopped_epoch: int | None
    best_checkpoint: str | None
    latest_checkpoint: str
    achieved_accuracy: float | None = None

    @dx.validates("achieved_accuracy")
    @classmethod
    def _accuracy_is_a_proportion(cls, value: float | None) -> float | None:
        """Validate the held-out target observation, when one was made."""
        if value is None:
            return None

        checked = ensure_finite("achieved_accuracy", value)

        if not 0.0 <= checked <= 1.0:
            msg = f"achieved_accuracy is outside [0, 1]: {value!r}"
            raise ValueError(msg)

        return checked


class ContributedValue(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """One named measurement a component contributed to the record."""


class CountValue(ContributedValue, extra="forbid"):
    """A contributed count."""

    kind: Literal["count"] = "count"
    value: int


class MeasureValue(ContributedValue, extra="forbid"):
    """A contributed real-valued measure."""

    kind: Literal["measure"] = "measure"
    value: float


class TextValue(ContributedValue, extra="forbid"):
    """A contributed text."""

    kind: Literal["text"] = "text"
    value: str


class FlagValue(ContributedValue, extra="forbid"):
    """A contributed flag."""

    kind: Literal["flag"] = "flag"
    value: bool


class CountsValue(ContributedValue, extra="forbid"):
    """A contributed table of counts, keyed by name."""

    kind: Literal["counts"] = "counts"
    value: dict[str, int]


class MeasuresValue(ContributedValue, extra="forbid"):
    """A contributed table of real-valued measures, keyed by name."""

    kind: Literal["measures"] = "measures"
    value: dict[str, float]


class MetricsRecord(dx.Model, extra="forbid"):
    """Records everything one training point measured and achieved.

    Attributes
    ----------
    stage : Literal["prepared", "closed"]
        ``prepared`` for the record written once preparation ends, ``closed``
        for the record written when the point closes.
    point : PointManifest
        The point's identity and the measured size of each component role.
    phases : dict[str, float]
        The wall time, in seconds, of each phase the point has run, summed
        across its attempts.
    peak_memory_bytes : int | None
        The most bytes the run device held at once during the latest attempt,
        or None where the device reports no peak or the point has not closed.
    corpus : CorpusRecord | None
        Every report a drawn sentence corpus achieved, or None for inference pairs.
    inference : InferenceDataRecord | None
        Every report a drawn inference dataset achieved, or None for a run
        trained on a corpus of strings.
    selection : SelectionRecord | None
        The constraint budget the selection achieved, or None for a run that
        selects no constraints.
    bucketing : BucketingRecord | None
        The length bucketing of the run's batches, or None for a run whose
        batches are drawn as it trains.
    residuals : dict[str, ResidualRecord]
        The approximation residual of every loss term that truncated a sum,
        by the term's tag.
    fitting : FittingMetricsRecord | None
        What fitting completed, or None before any fitting has.
    contributed : dict[str, dict[str, ContributedValue]]
        The named measurements each contributing component or stage reported,
        by the name it reported under: the grounding a run trains under is
        recorded as ``grounding``, and a component that reports on itself under
        its role.
    """

    stage: Literal["prepared", "closed"]
    point: PointManifest
    phases: dict[str, float]
    peak_memory_bytes: int | None
    corpus: CorpusRecord | None
    inference: InferenceDataRecord | None
    selection: SelectionRecord | None
    bucketing: BucketingRecord | None
    residuals: dict[str, ResidualRecord]
    fitting: FittingMetricsRecord | None
    contributed: dict[str, dict[str, ContributedValue]]


def _histogram(histogram: Histogram, /) -> HistogramRecord:
    """Return the record of one histogram."""
    return HistogramRecord(
        counts={str(value): count for value, count in histogram.counts.items()},
        total=histogram.total,
        mean=histogram.mean if histogram.total else None,
    )


def _sorted[K: str, V](table: Mapping[K, V], /) -> dict[str, V]:
    """Return a table with its keys in sorted order."""
    return {key: table[key] for key in sorted(table)}


def _length(report: LengthReport, /) -> LengthRecord:
    """Return the record of a length report."""
    return LengthRecord(
        achieved=_histogram(report.achieved), resampled=report.resampled
    )


def _lexicon(report: LexiconReport, /) -> LexiconRecord:
    """Return the record of a lexicon report."""
    return LexiconRecord(
        classes=_sorted(
            {
                category: _histogram(histogram)
                for category, histogram in report.classes.items()
            }
        ),
        forms=_histogram(report.forms),
        realized_size=report.realized_size,
        realized_vocabulary_size=report.realized_vocabulary_size,
        per_type_ambiguity=_histogram(report.per_type_ambiguity),
        ambiguity_frequency_correlation=report.ambiguity_frequency_correlation,
        type_token_ratio=report.type_token_ratio,
        class_sizes=_sorted(report.class_sizes),
        token_counts=_sorted(report.token_counts),
        token_shares=_sorted(report.token_shares),
        rank_frequency_exponent=report.rank_frequency_exponent,
        coupling=report.coupling,
        concordance=report.concordance,
    )


def _inventory(report: InventoryReport, /) -> InventoryRecord:
    """Return the record of an inventory report."""
    return InventoryRecord(
        categories=_sorted(report.categories),
        argument_counts=_sorted(report.argument_counts),
        composition_orders=_sorted(report.composition_orders),
        max_arguments=report.max_arguments,
        max_order=report.max_order,
        primitives=report.primitives,
        realized_primitive_count=report.realized_primitive_count,
        size=report.size,
        depths=_histogram(report.depths),
        arguments=_histogram(report.arguments),
        orders=_histogram(report.orders),
    )


def _semantic(report: SemanticReport, /) -> SemanticRecord:
    """Return the persisted joint semantic inventory of both inference sides."""
    return SemanticRecord(
        joint=_lexicon(report.joint),
        arities=_histogram(report.arities),
        per_form_meanings=_histogram(report.per_form_meanings),
        items=_sorted(report.items),
        item_shares=_sorted(report.item_shares),
        symbols_used=_sorted(report.symbols_used),
        cells=_sorted(
            {category: _sorted(counts) for category, counts in report.cells.items()}
        ),
        concordance=report.concordance,
    )


def _ambiguity(report: AmbiguityReport, /) -> AmbiguityRecord:
    """Return the record of an ambiguity report."""
    return AmbiguityRecord(
        lexical=_histogram(report.lexical),
        derivations=_histogram(report.derivations),
        rejected=report.rejected,
    )


def _corruption[O: NamedOperator](report: CorruptionReport[O], /) -> CorruptionRecord:
    """Return the record of a corruption report."""
    return CorruptionRecord(
        accepted=_sorted(report.accepted),
        rejected=_sorted(report.rejected),
        inapplicable=_sorted(report.inapplicable),
        sample_size=report.sample_size,
        corrupted=report.corrupted,
        achieved_proportion=report.achieved_proportion,
        rejection_rate=report.rejection_rate,
    )


def _ill_typing(report: IllTypingReport, /) -> IllTypingRecord:
    """Return the record of an ill-typing report."""
    policy = report.injection.requested.policy

    return IllTypingRecord(
        injection=_corruption(report.injection),
        certificates=_sorted(report.certificates),
        regime=policy.regime,
        requested_noise=policy.noise,
        observed=_sorted(report.observed),
        flipped=report.flipped,
        achieved_noise=report.achieved_noise,
    )


def _noise(report: NoiseReport, /) -> NoiseRecord:
    """Return the record of a label-noise report."""
    return NoiseRecord(
        items=_sorted(report.items),
        flipped=_sorted(report.flipped),
        observed=_sorted(report.observed),
        sample_size=report.sample_size,
        flip_count=report.flip_count,
        achieved_rate=report.achieved_rate,
        achieved_rates=_sorted(report.achieved_rates),
    )


def _scope(report: ScopeReport, /) -> ScopeRecord:
    """Return the record of a scope report."""
    return ScopeRecord(
        readings=_histogram(report.readings),
        derivations=_histogram(report.derivations),
        rejected=report.rejected,
        exceeded=report.exceeded,
        undecided=report.undecided,
    )


def corpus_record(corpus: SyntheticCorpus, /) -> CorpusRecord:
    """Return the record of every report a drawn corpus achieved.

    Parameters
    ----------
    corpus : SyntheticCorpus
        The drawn corpus.

    Returns
    -------
    CorpusRecord
        Its length, lexicon, inventory and ambiguity reports, and its
        corruption, noise, ill-typing and scope reports where it declares them.
    """
    corruption = corpus.corruption
    noise = corpus.noise
    ill_typing = corpus.ill_typing
    scope = corpus.scope

    return CorpusRecord(
        length=_length(corpus.length),
        lexicon=_lexicon(corpus.lexicon),
        inventory=_inventory(corpus.inventory),
        ambiguity=_ambiguity(corpus.ambiguity),
        corruption=None if corruption is None else _corruption(corruption),
        noise=None if noise is None else _noise(noise),
        ill_typing=None if ill_typing is None else _ill_typing(ill_typing),
        scope=None if scope is None else _scope(scope),
    )


def _complexity(report: ComplexityReport, /) -> ComplexityRecord:
    """Return the record of a complexity report."""
    return ComplexityRecord(
        premise=_sorted(
            {
                control: _histogram(histogram)
                for control, histogram in report.premise.items()
            }
        ),
        hypothesis=_sorted(
            {
                control: _histogram(histogram)
                for control, histogram in report.hypothesis.items()
            }
        ),
        overlap=_histogram(report.overlap),
        rejected=report.rejected,
        satisfied=report.satisfied(),
    )


def _inference_labels(report: InferenceReport, /) -> InferenceLabelRecord:
    """Return the record of an inference-label report."""
    return InferenceLabelRecord(
        labels=_sorted(report.labels),
        proportions=_sorted(report.proportions()),
        total=report.total,
        decided=report.decided,
        rejected=report.rejected,
        satisfied=report.satisfied(),
    )


def _confusion(report: ConfusionReport, /) -> ConfusionRecord:
    """Return the record of a confusion report."""
    return ConfusionRecord(
        counts=_sorted({label: _sorted(row) for label, row in report.counts.items()}),
        matrix=_sorted({label: _sorted(row) for label, row in report.matrix().items()}),
        decided=report.decided,
        undecided=report.undecided,
        flip_count=report.flip_count,
        achieved_rate=report.achieved_rate,
        satisfied=report.satisfied(),
    )


def inference_record(inference: SyntheticInference, /) -> InferenceDataRecord:
    """Return the record of every report a drawn inference dataset achieved.

    Parameters
    ----------
    inference : SyntheticInference
        The drawn inference dataset.

    Returns
    -------
    InferenceDataRecord
        The reports of its premise and hypothesis corpora, its complexity and
        label reports, and its confusion report where it declares noise.
    """
    noise = inference.noise
    semantic = inference.semantic

    return InferenceDataRecord(
        premise=corpus_record(inference.premises),
        hypothesis=corpus_record(inference.hypotheses),
        complexity=_complexity(inference.complexity),
        inference=_inference_labels(inference.inference),
        confusion=None if noise is None else _confusion(noise),
        semantic=None if semantic is None else _semantic(semantic),
    )


def _clamp_record(
    policy: ClampPolicy[Term],
    report: ClampReport,
    default_matching: Matching[Term],
    /,
) -> ClampSupervisionRecord:
    """Return the persisted policy and outcomes of configured clamping."""
    matching = policy.matching or default_matching

    return ClampSupervisionRecord(
        mode=policy.mode,
        penalty=policy.penalty,
        gradient_scale=1.0 if policy.penalty is None else policy.penalty,
        scope=policy.scope,
        matching=matching.name,
        meaning_matching=policy.meaning_matching.name,
        require_constituent=policy.require_constituent,
        negate=policy.negate,
        on_unsatisfiable=policy.on_unsatisfiable.action,
        dropped_clamps=report.dropped_clamps,
        skipped_items=report.skipped_items,
    )


def selection_record(
    report: SelectionReport[str],
    policy: ClampPolicy[Term],
    outcomes: ClampReport,
    default_matching: Matching[Term],
    /,
) -> SelectionRecord:
    """Return the record of the constraint budget a selection achieved.

    Parameters
    ----------
    report : SelectionReport[str]
        The selection's budget, by category text.
    policy : ClampPolicy[Term]
        The clamp policy applied to the selection.
    outcomes : ClampReport
        Counts of the unsatisfiable clamps dropped and items skipped.
    default_matching : Matching[Term]
        The grammar congruence used when the policy declares no override.

    Returns
    -------
    SelectionRecord
        The same budget as a record.
    """
    return SelectionRecord(
        types_pinned=report.types_pinned,
        tokens_covered=report.tokens_covered,
        mass_share=report.mass_share,
        by_category=_sorted(
            {
                category: CategoryBudgetRecord(
                    types_pinned=budget.types_pinned,
                    tokens_covered=budget.tokens_covered,
                    mass_share=budget.mass_share,
                )
                for category, budget in report.by_category.items()
            }
        ),
        concordance=report.concordance,
        clamp=_clamp_record(policy, outcomes, default_matching),
    )


class _TypingExample(Protocol):
    """Typing labels needed for persisted split counts."""

    @property
    def true_label(self) -> str: ...

    @property
    def observed_label(self) -> str: ...


class _OccupancyBatch(Protocol):
    """Batch fields needed for occupancy and typing counts."""

    def __len__(self) -> int: ...

    @property
    def words(self) -> Sequence[Sequence[str]]: ...

    @property
    def form_typing(self) -> Sequence[Sequence[_TypingExample]]: ...


def _occupancy(batches: Sequence[_OccupancyBatch], /) -> OccupancyRecord:
    """Return how full a split's batches are, each as long as its longest item."""
    live = sum(len(words) for batch in batches for words in batch.words)
    capacity = sum(
        len(batch.words) * max((len(words) for words in batch.words), default=0)
        for batch in batches
    )
    occupancy = BucketOccupancy(live, capacity)
    form_examples = tuple(
        example for batch in batches for row in batch.form_typing for example in row
    )
    observed = {
        label: sum(example.observed_label == label for example in form_examples)
        for label in ("well-typed", "ill-typed")
    }

    return OccupancyRecord(
        batches=len(batches),
        items=sum(len(batch) for batch in batches),
        live=occupancy.live,
        capacity=occupancy.capacity,
        padded=occupancy.padded,
        rate=occupancy.rate,
        form_typing=(
            None
            if not form_examples
            else FormTypingSplitRecord(
                examples=len(form_examples),
                ill_typed=sum(
                    example.true_label == "ill-typed" for example in form_examples
                ),
                observed=observed,
            )
        ),
    )


def _plan_reports(
    plan: ParserFittingPlan | None, /
) -> tuple[SelectionRecord | None, BucketingRecord | None]:
    """Return a parser plan's selection and the bucketing of its batches.

    A plan that holds out no parser draws its batches as fitting runs, so it
    records no selection and no bucketing.
    """
    if plan is None:
        return None, None

    selection = plan.constraint_selection
    policy = plan.clamp_policy
    outcomes = plan.clamp_report

    if selection is None:
        if policy is not None or outcomes is not None:
            msg = "a parser plan without a selection carries clamp supervision"
            raise ValueError(msg)

        selection_recorded = None

    elif policy is None or outcomes is None:
        msg = "a parser plan with a selection lacks clamp supervision"
        raise ValueError(msg)

    else:
        selection_recorded = selection_record(
            selection,
            policy,
            outcomes,
            plan.grammar.grammar.matching,
        )

    return (
        selection_recorded,
        BucketingRecord(
            training=_occupancy(plan.batches),
            validation=_occupancy(plan.fit_batches),
        ),
    )


def _plan_data_records(
    plan: ParserFittingPlan | None, /
) -> tuple[CorpusRecord | None, InferenceDataRecord | None]:
    """Return the corpus or inference report a parser plan owns, if any."""
    if plan is None:
        return None, None

    corpus = plan.corpus
    drawn = None if plan.pairs is None else plan.pairs.drawn

    return (
        None if corpus is None else corpus_record(corpus),
        None if drawn is None else inference_record(drawn),
    )


def _plan_interpretation_report(
    plan: ParserFittingPlan | None, /
) -> Mapping[str, ReportValue] | None:
    """Return the parser data's interpretation report, when declared."""
    interpretation = None if plan is None else plan.interpretation

    return None if interpretation is None else interpretation.report()


def _step(result: TrainResult, /) -> StepMetricsRecord:
    """Return the record of one training step's result."""
    return StepMetricsRecord(
        step=result.step,
        items=result.items,
        dropped=result.dropped,
        metrics=_sorted(result.metrics),
    )


def _epoch(report: EpochReport, /) -> EpochMetricsRecord:
    """Return the record of one completed epoch."""
    validation: EvalResult | None = report.validation

    return EpochMetricsRecord(
        epoch=report.epoch,
        steps=tuple(_step(result) for result in report.results),
        validation_items=None if validation is None else validation.items,
        validation=None if validation is None else _sorted(validation.metrics),
    )


def _before_resume(
    steps: Sequence[StepMetricsRecord], completed: int, /
) -> tuple[StepMetricsRecord, ...]:
    """Return the steps an earlier attempt took before the resumed position.

    A step counts the optimizer steps taken once it reported, so the steps
    before a checkpoint at ``completed`` optimizer steps are those reporting
    fewer and the first reporting exactly that many, which took the optimizer
    step that reached it.
    """
    kept = [step for step in steps if step.step < completed]
    reaching = next((step for step in steps if step.step == completed), None)

    if completed > 0 and reaching is not None:
        kept.append(reaching)

    return tuple(kept)


def _fitting(
    report: FittingReport,
    start: ResumePosition,
    earlier: FittingMetricsRecord | None,
    /,
) -> FittingMetricsRecord:
    """Return what fitting completed, carrying what earlier attempts completed."""
    epochs = tuple(_epoch(epoch) for epoch in report.epochs)
    steps = tuple(_step(result) for result in report.results)

    if earlier is not None:
        first = epochs[0].epoch if epochs else None
        epochs = (
            *(
                epoch
                for epoch in earlier.epochs
                if first is None or epoch.epoch < first
            ),
            *epochs,
        )
        steps = (
            *_before_resume(earlier.steps, start.step),
            *steps,
        )

    best = report.best_checkpoint

    return FittingMetricsRecord(
        unit=report.unit.value,
        epochs=epochs,
        steps=steps,
        optimizer_steps=report.optimizer_steps,
        data_position=report.data_position,
        stop=report.stop_reason,
        stopped_epoch=epochs[-1].epoch if epochs else None,
        best_checkpoint=None if best is None else str(best),
        latest_checkpoint=str(report.latest_checkpoint),
        achieved_accuracy=report.achieved_accuracy,
    )


def _residuals(
    keys: Mapping[str, str], fitting: FittingMetricsRecord | None, /
) -> dict[str, ResidualRecord]:
    """Return the residual every truncating term reported across the fitting.

    Parameters
    ----------
    keys : Mapping[str, str]
        The telemetry key each term's residual is logged under, by its tag.
    fitting : FittingMetricsRecord | None
        What fitting completed, if anything.
    """
    if fitting is None:
        return {}

    steps = (
        *(step for epoch in fitting.epochs for step in epoch.steps),
        *fitting.steps,
    )
    residuals: dict[str, ResidualRecord] = {}

    for tag in sorted(keys):
        key = keys[tag]
        values = [step.metrics[key] for step in steps if key in step.metrics]

        if values:
            residuals[tag] = ResidualRecord(
                steps=len(values),
                mean=sum(values) / len(values),
                maximum=max(values),
                last=values[-1],
            )

    return residuals


def _contributed_value(name: str, value: ReportValue, /) -> ContributedValue:
    """Return the record of one contributed measurement.

    Raises
    ------
    TypeError
        If the value is a table whose entries are not all numbers.
    """
    match value:
        case bool():
            return FlagValue(value=value)

        case int():
            return CountValue(value=value)

        case float():
            return MeasureValue(value=value)

        case str():
            return TextValue(value=value)

        case table:
            counts = {
                key: entry
                for key, entry in table.items()
                if isinstance(entry, int) and not isinstance(entry, bool)
            }

            if len(counts) == len(table):
                return CountsValue(value=_sorted(counts))

            measures = {
                key: float(entry)
                for key, entry in table.items()
                if isinstance(entry, float | int) and not isinstance(entry, bool)
            }

            if len(measures) == len(table):
                return MeasuresValue(value=_sorted(measures))

            msg = f"the contributed table {name!r} holds a value that is not a number"
            raise TypeError(msg)


def contributed_record(
    report: Mapping[str, ReportValue], /
) -> dict[str, ContributedValue]:
    """Return the record of one contributed report.

    Parameters
    ----------
    report : Mapping[str, ReportValue]
        The named measurements.

    Returns
    -------
    dict[str, ContributedValue]
        Each measurement as a count, a measure, a text, a flag, or a table of
        counts or of measures, in sorted order of name.

    Raises
    ------
    TypeError
        If a table holds a value that is not a number.
    ValueError
        If a name is empty.
    """
    return {
        ensure_name("a contributed measurement", name): _contributed_value(
            name, report[name]
        )
        for name in sorted(report)
    }


def write_metrics(directory: Path, record: MetricsRecord, /) -> None:
    """Write a point's metrics record into its run directory, replacing any it holds.

    Parameters
    ----------
    directory : Path
        The run directory, created if absent.
    record : MetricsRecord
        The record to write.
    """
    directory.mkdir(parents=True, exist_ok=True)
    document = json.dumps(json.loads(record.model_dump_json()), indent=2)
    replace_file(directory / METRICS_NAME, f"{document}\n".encode())


def read_metrics(directory: Path, /) -> MetricsRecord | None:
    """Return the metrics record a run directory holds, or None when it holds none.

    Parameters
    ----------
    directory : Path
        The run directory.

    Raises
    ------
    didactic.api.ValidationError
        If the file does not hold a metrics record.
    """
    path = directory / METRICS_NAME

    if not path.is_file():
        return None

    return MetricsRecord.model_validate_json(path.read_text(encoding="utf-8"))


def metrics_row(record: MetricsRecord, /) -> dict[str, JsonValue]:
    """Return a metrics record as one row of named scalar values.

    Every scalar the record holds outside a sequence becomes a column named by
    its path through the record, its parts joined by ``/``; a sequence, such as
    the per-step metrics of fitting, stays in the record.

    Parameters
    ----------
    record : MetricsRecord
        The record.

    Returns
    -------
    dict[str, JsonValue]
        The scalars by path, in the record's order.
    """
    row: dict[str, JsonValue] = {}

    def walk(path: str, value: JsonValue) -> None:
        if isinstance(value, dict):
            for name, held in value.items():
                walk(f"{path}/{name}" if path else name, held)

        elif not isinstance(value, list):
            row[path] = value

    walk("", json.loads(record.model_dump_json()))

    return row


@final
class MetricsRecorder:
    """Records one training point's measurements and reports in its run directory.

    A recorder is bound to one point and the run directory it executes in. A
    record the directory already holds, from an earlier attempt of the point, is
    read once preparation ends: the phases it timed are carried forward and
    added to, and the fitting it recorded is extended by what the resumed
    fitting completes, or kept whole when nothing remained to fit.

    Parameters
    ----------
    directory : Path
        The point's run directory.
    point : SweepPoint
        The point being executed.
    """

    __slots__ = (
        "_contributed",
        "_directory",
        "_earlier",
        "_fitting",
        "_phases",
        "_point",
        "_prepared",
        "_residual_keys",
    )

    def __init__(self, directory: Path, point: SweepPoint, /) -> None:
        self._directory = directory
        self._point = point
        self._earlier: MetricsRecord | None = None
        self._prepared: MetricsRecord | None = None
        self._phases: dict[str, float] = {}
        self._fitting: FittingMetricsRecord | None = None
        self._contributed: dict[str, dict[str, ContributedValue]] = {}
        self._residual_keys: dict[str, str] = {}

    def prepared(
        self, blueprint: TrainingBlueprint, prepared: PreparedTraining, /
    ) -> None:
        """Record what preparation built and write the prepared record.

        Parameters
        ----------
        blueprint : TrainingBlueprint
            The lowered run.
        prepared : PreparedTraining
            The model, optimization and fitting plan preparation built.
        """
        self._earlier = read_metrics(self._directory)
        earlier = self._earlier
        self._fitting = None if earlier is None else earlier.fitting
        declared = blueprint.run_spec.model.checkpointed_components()
        owned = prepared.model.role_parameters()
        point = measure_point(
            self._point,
            {
                role: MeasuredRole(cast("str", declared[role].kind), owned[role])
                for role in sorted(owned)
            },
            corpus_fingerprint=blueprint.corpus_revision,
        )
        plan = prepared.plan
        parser = plan.held_out_parser()
        selection, bucketing = _plan_reports(parser)
        corpus, inference = _plan_data_records(parser)
        interpretation = _plan_interpretation_report(parser)
        self._residual_keys = dict(plan.residual_keys)
        self._prepared = MetricsRecord(
            stage="prepared",
            point=point,
            phases={},
            peak_memory_bytes=None,
            corpus=corpus,
            inference=inference,
            selection=selection,
            bucketing=bucketing,
            residuals={},
            fitting=None,
            contributed={},
        )
        self.contribute(
            "compile",
            {
                **blueprint.compile_report.measurements(),
                **blueprint.approximation.measurements(),
            },
        )

        if interpretation is not None:
            self.contribute("interpretation", interpretation)

        self._write("prepared", peak_memory_bytes=None)

    def timed(self, phase: RunPhase, seconds: float, /) -> None:
        """Add the wall time one phase of the point took.

        Parameters
        ----------
        phase : RunPhase
            The phase.
        seconds : float
            The wall time it took, in seconds.
        """
        self._phases[phase.value] = self._phases.get(phase.value, 0.0) + seconds

    def fitted(self, report: FittingReport, start: ResumePosition, /) -> None:
        """Record what one fitting invocation completed.

        Parameters
        ----------
        report : FittingReport
            What the invocation completed.
        start : ResumePosition
            The position the invocation started from.
        """
        earlier = None if self._earlier is None else self._earlier.fitting
        self._fitting = _fitting(report, start, earlier)

    def contribute(self, name: str, report: Mapping[str, ReportValue], /) -> None:
        """Record a named report a component or a stage contributes.

        Parameters
        ----------
        name : str
            The name the report is recorded under.
        report : Mapping[str, ReportValue]
            The named measurements.

        Raises
        ------
        ValueError
            If the name is empty or a report was already contributed under it.
        """
        held = ensure_name("a contributed report", name)

        if held in self._contributed:
            msg = f"a report was already contributed under {held!r}"
            raise ValueError(msg)

        self._contributed[held] = contributed_record(report)

    def closed(self, *, peak_memory_bytes: int | None) -> None:
        """Write the closed record.

        Parameters
        ----------
        peak_memory_bytes : int | None
            The most bytes the run device held at once during the point, or
            None where the device reports no peak.
        """
        self._write("closed", peak_memory_bytes=peak_memory_bytes)

    def _write(
        self, stage: Literal["prepared", "closed"], /, *, peak_memory_bytes: int | None
    ) -> None:
        """Write the record of the point as recorded so far.

        Raises
        ------
        RuntimeError
            If preparation has not been recorded.
        """
        prepared = self._prepared

        if prepared is None:
            msg = "a metrics record is written only once preparation is recorded"
            raise RuntimeError(msg)

        phases = dict(self._phases)

        if self._earlier is not None:
            for phase, seconds in self._earlier.phases.items():
                phases[phase] = phases.get(phase, 0.0) + seconds

        record = prepared.with_(
            stage=stage,
            phases=_sorted(phases),
            peak_memory_bytes=peak_memory_bytes,
            residuals=_residuals(self._residual_keys, self._fitting),
            fitting=self._fitting,
            contributed=_sorted(self._contributed),
        )
        write_metrics(self._directory, record)
