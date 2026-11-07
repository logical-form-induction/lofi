"""Declarations for experiment-level evaluation reducers."""

from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import didactic.api as dx

from lofi._checks import ensure_name, ensure_non_negative
from lofi.spec.components.base import ComponentSpec
from lofi.spec.components.run import BootstrapIntervalSpec, IntervalSettings
from lofi.spec.evaluators.alignment import AlignmentSettings, AlignmentSpec

if TYPE_CHECKING:
    from lofi.spec.evaluators.alignment import AlignmentDeclaration

__all__ = [
    "CoherenceCollapseReducerSettings",
    "CoherenceCollapseReducerSpec",
    "ConventionGridSettings",
    "ConventionGridSpec",
    "CrossedRecoveryReducerSettings",
    "CrossedRecoveryReducerSpec",
    "CrosslingualReducerSettings",
    "CrosslingualReducerSpec",
    "EncoderAblationReducerSettings",
    "EncoderAblationReducerSpec",
    "NodeVocabularyReducerSettings",
    "NodeVocabularyReducerSpec",
    "NodeVocabularySourceSettings",
    "NodeVocabularySourceSpec",
    "RankChurnReducerSettings",
    "RankChurnReducerSpec",
    "ReducerSpec",
    "RescoringReducerSettings",
    "RescoringReducerSpec",
    "ResidualStructureReducerSettings",
    "ResidualStructureReducerSpec",
    "RestitutiveCoverageReducerSettings",
    "RestitutiveCoverageReducerSpec",
    "SourceComparisonReducerSettings",
    "SourceComparisonReducerSpec",
    "StatisticalTestSettings",
    "StatisticalTestSpec",
    "TransferReducerSettings",
    "TransferReducerSpec",
    "VarianceFactorSettings",
    "VarianceFactorSpec",
    "VarianceReducerSettings",
    "VarianceReducerSpec",
]


class VarianceFactorSettings(NamedTuple):
    """One named random factor and the row column carrying its level."""

    name: str
    column: str


class VarianceFactorSpec(dx.Model, extra="forbid"):
    """Declare one random factor in a variance decomposition."""

    name: str
    column: str

    @dx.validates("name", "column")
    @classmethod
    def _is_named(cls, value: str) -> str:
        """Return a nonempty factor or column name."""
        return ensure_name("a variance factor field", value)

    def settings(self) -> VarianceFactorSettings:
        """Return the immutable factor declaration."""
        return VarianceFactorSettings(self.name, self.column)


class ConventionGridSettings(NamedTuple):
    """Levels crossed for the eight shared scoring conventions."""

    punctuation: tuple[str, ...]
    unary_chains: tuple[str, ...]
    trivial_spans: tuple[str, ...]
    minimum_length: tuple[int, ...]
    aggregation: tuple[str, ...]
    coverage: tuple[str, ...]
    self_arcs: tuple[str, ...]
    gold_derivation: tuple[str, ...]


class ConventionGridSpec(dx.Model, extra="forbid"):
    """Declare the convention levels a rescoring reducer crosses."""

    punctuation: tuple[Literal["keep", "strip"], ...]
    unary_chains: tuple[Literal["keep", "collapse"], ...]
    trivial_spans: tuple[Literal["keep", "drop"], ...]
    minimum_length: tuple[int, ...]
    aggregation: tuple[Literal["sentence", "corpus"], ...]
    coverage: tuple[Literal["ignore", "zero"], ...]
    self_arcs: tuple[Literal["allow", "forbid"], ...]
    gold_derivation: tuple[Literal["canonical", "oracle"], ...]

    @dx.validates(
        "punctuation",
        "unary_chains",
        "trivial_spans",
        "minimum_length",
        "aggregation",
        "coverage",
        "self_arcs",
        "gold_derivation",
    )
    @classmethod
    def _levels_are_nonempty_and_distinct[V](
        cls, levels: tuple[V, ...]
    ) -> tuple[V, ...]:
        """Return a nonempty sequence without repeated levels."""
        if not levels:
            msg = "a convention grid axis declares no level"
            raise ValueError(msg)

        if len(set(levels)) != len(levels):
            msg = f"a convention grid axis repeats a level: {levels!r}"
            raise ValueError(msg)

        return levels

    @dx.validates("minimum_length")
    @classmethod
    def _lengths_are_nonnegative(cls, levels: tuple[int, ...]) -> tuple[int, ...]:
        """Return nonnegative minimum-length levels."""
        if any(level < 0 for level in levels):
            msg = f"a convention grid declares a negative minimum length: {levels!r}"
            raise ValueError(msg)

        return levels

    def settings(self) -> ConventionGridSettings:
        """Return the immutable convention grid."""
        return ConventionGridSettings(
            punctuation=self.punctuation,
            unary_chains=self.unary_chains,
            trivial_spans=self.trivial_spans,
            minimum_length=self.minimum_length,
            aggregation=self.aggregation,
            coverage=self.coverage,
            self_arcs=self.self_arcs,
            gold_derivation=self.gold_derivation,
        )


class ReducerSpec(ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare a reducer over completed experiment rows."""

    @classmethod
    def family(cls) -> str:
        """Return the reducer plugin family."""
        return "reducer"


class RestitutiveCoverageReducerSettings(NamedTuple):
    """Resolved lexicon role and declared semantic classes for a coverage query."""

    dataset: str
    change_class: str
    modifier_class: str


class RestitutiveCoverageReducerSpec(ReducerSpec):
    """Declare a restitutive scope query over one interpreted dataset lexicon."""

    kind: Literal["restitutive-coverage"] = "restitutive-coverage"
    dataset: str
    change_class: str
    modifier_class: str

    @dx.validates("dataset", "change_class", "modifier_class")
    @classmethod
    def _field_is_named(cls, value: str) -> str:
        """Return a named dataset role or semantic constant class."""
        return ensure_name("a restitutive-coverage field", value)

    @dx.model_validator()
    def _classes_are_distinct(self) -> RestitutiveCoverageReducerSpec:
        """Require the change and modifier roles to denote distinct classes."""
        if self.change_class == self.modifier_class:
            msg = "restitutive coverage needs distinct change and modifier classes"
            raise ValueError(msg)
        return self

    def settings(self) -> RestitutiveCoverageReducerSettings:
        """Return the immutable coverage-query declaration."""
        return RestitutiveCoverageReducerSettings(
            self.dataset, self.change_class, self.modifier_class
        )


type NodeVocabularyRepresentation = Literal[
    "derivation-variables",
    "trained-transformer-sae-latents",
    "dense-linear-probe",
    "random-transformer-sae-control",
]
"""The four representations a node vocabulary comparison holds one arm for."""

_NODE_VOCABULARY_ARMS: tuple[NodeVocabularyRepresentation, ...] = (
    "derivation-variables",
    "trained-transformer-sae-latents",
    "dense-linear-probe",
    "random-transformer-sae-control",
)


class NodeVocabularySourceSettings(NamedTuple):
    """Resolved location of one arm's persisted sweep record."""

    representation: NodeVocabularyRepresentation
    source: str
    probe: str
    record: str


class NodeVocabularySourceSpec(dx.Model, extra="forbid"):
    """Name one arm and the persisted causal probe record that holds its sweep.

    Parameters
    ----------
    representation : str
        The arm the record measures.
    source : str
        The probe records file, relative to the experiment root.
    probe : str
        The causal probe role the record is filed under.
    record : str
        The record's name, such as ``pass-0`` for the first held-out pass.
    """

    representation: NodeVocabularyRepresentation
    source: str
    probe: str
    record: str

    @dx.validates("source", "probe", "record")
    @classmethod
    def _field_is_named(cls, value: str) -> str:
        """Return a named source path, probe role or record."""
        return ensure_name("a node-vocabulary field", value)

    def settings(self) -> NodeVocabularySourceSettings:
        """Return the immutable source declaration."""
        return NodeVocabularySourceSettings(
            self.representation, self.source, self.probe, self.record
        )


class NodeVocabularyReducerSettings(NamedTuple):
    """Resolved persisted records of the four node vocabulary arms."""

    arms: tuple[NodeVocabularySourceSettings, ...]


class NodeVocabularyReducerSpec(ReducerSpec):
    """Declare a node vocabulary comparison over four persisted arm sweeps.

    Parameters
    ----------
    arms : tuple[NodeVocabularySourceSpec, ...]
        One persisted record for each of the four representations.
    """

    kind: Literal["node-vocabulary"] = "node-vocabulary"
    arms: tuple[NodeVocabularySourceSpec, ...]

    @dx.model_validator()
    def _each_arm_once(self) -> NodeVocabularyReducerSpec:
        """Require exactly one record for every representation."""
        declared = sorted(arm.representation for arm in self.arms)

        if declared != sorted(_NODE_VOCABULARY_ARMS):
            msg = (
                "a node-vocabulary comparison declares every arm once, "
                f"{list(_NODE_VOCABULARY_ARMS)!r}, not {declared!r}"
            )
            raise ValueError(msg)

        return self

    def settings(self) -> NodeVocabularyReducerSettings:
        """Return the immutable comparison declaration."""
        return NodeVocabularyReducerSettings(tuple(arm.settings() for arm in self.arms))


class StatisticalTestSettings(NamedTuple):
    """Resolved randomization budget and seed for one statistical test."""

    replicates: int
    seed: int


class StatisticalTestSpec(dx.Model, extra="forbid"):
    """Declare the randomization budget and seed for one statistical test."""

    replicates: int
    seed: int

    @dx.validates("replicates")
    @classmethod
    def _replicates_are_positive(cls, value: int) -> int:
        checked = ensure_non_negative("a statistical-test replicate count", value)
        if checked == 0:
            msg = "a statistical test declares no replicates"
            raise ValueError(msg)
        return checked

    @dx.validates("seed")
    @classmethod
    def _seed_is_nonnegative(cls, value: int) -> int:
        return ensure_non_negative("a statistical-test seed", value)

    def settings(self) -> StatisticalTestSettings:
        """Return the immutable randomization declaration."""
        return StatisticalTestSettings(self.replicates, self.seed)


class SourceComparisonReducerSettings(NamedTuple):
    """Resolved experiment dimensions for source-model comparison."""

    metric: str
    source_path: str
    candidate_path: str
    data_size_path: str
    condition_path: str
    direction_path: str
    twin_path: str
    value_column: str
    method_column: str
    description_length_metric: str
    capacity_field: Literal["parameter_count"]
    primary_condition: str
    reversed_ground_truth_condition: str
    lexicon_scrambled_condition: str
    observed_twin: str
    permutation: StatisticalTestSettings
    equivalence: StatisticalTestSettings
    falsifier: str
    smoke_test: bool


class SourceComparisonReducerSpec(ReducerSpec):
    """Declare source selection from held-out fit rows and point metadata."""

    kind: Literal["source-comparison"] = "source-comparison"
    metric: str
    source_path: str
    candidate_path: str
    data_size_path: str
    condition_path: str
    direction_path: str
    twin_path: str
    value_column: str = "value"
    method_column: str = "method"
    description_length_metric: str
    capacity_field: Literal["parameter_count"] = "parameter_count"
    primary_condition: str
    reversed_ground_truth_condition: str
    lexicon_scrambled_condition: str
    observed_twin: str
    permutation: StatisticalTestSpec
    equivalence: StatisticalTestSpec
    falsifier: str
    smoke_test: bool

    @dx.validates(
        "metric",
        "source_path",
        "candidate_path",
        "data_size_path",
        "condition_path",
        "direction_path",
        "twin_path",
        "value_column",
        "method_column",
        "description_length_metric",
        "primary_condition",
        "reversed_ground_truth_condition",
        "lexicon_scrambled_condition",
        "observed_twin",
        "falsifier",
    )
    @classmethod
    def _field_is_named(cls, value: str) -> str:
        """Return a named metric, arm, path, column, condition, or falsifier."""
        return ensure_name("a source-comparison reducer field", value)

    @dx.validates(
        "source_path",
        "candidate_path",
        "data_size_path",
        "condition_path",
        "direction_path",
        "twin_path",
    )
    @classmethod
    def _path_is_dotted(cls, value: str) -> str:
        """Return a dotted path whose every segment is named."""
        for segment in value.split("."):
            ensure_name("a source-comparison path segment", segment)
        return value

    @dx.model_validator()
    def _design_is_identified(self) -> SourceComparisonReducerSpec:
        conditions = (
            self.primary_condition,
            self.reversed_ground_truth_condition,
            self.lexicon_scrambled_condition,
        )
        if len(set(conditions)) != len(conditions):
            msg = f"a source-comparison reducer repeats a condition: {conditions!r}"
            raise ValueError(msg)
        if self.metric == self.description_length_metric:
            msg = (
                "a source-comparison reducer uses one metric for fit and "
                "description length"
            )
            raise ValueError(msg)
        paths = (
            self.source_path,
            self.candidate_path,
            self.data_size_path,
            self.condition_path,
            self.direction_path,
            self.twin_path,
        )
        if len(set(paths)) != len(paths):
            msg = f"a source-comparison reducer repeats a point path: {paths!r}"
            raise ValueError(msg)
        return self

    def settings(self) -> SourceComparisonReducerSettings:
        """Return the immutable source comparison declaration."""
        return SourceComparisonReducerSettings(
            metric=self.metric,
            source_path=self.source_path,
            candidate_path=self.candidate_path,
            data_size_path=self.data_size_path,
            condition_path=self.condition_path,
            direction_path=self.direction_path,
            twin_path=self.twin_path,
            value_column=self.value_column,
            method_column=self.method_column,
            description_length_metric=self.description_length_metric,
            capacity_field=self.capacity_field,
            primary_condition=self.primary_condition,
            reversed_ground_truth_condition=self.reversed_ground_truth_condition,
            lexicon_scrambled_condition=self.lexicon_scrambled_condition,
            observed_twin=self.observed_twin,
            permutation=self.permutation.settings(),
            equivalence=self.equivalence.settings(),
            falsifier=self.falsifier,
            smoke_test=self.smoke_test,
        )


class VarianceReducerSettings(NamedTuple):
    """Resolved settings for the random-factor variance reducer."""

    metric: str
    factors: tuple[VarianceFactorSettings, ...]
    estimator: Literal["moments"]
    interval: IntervalSettings
    method_column: str
    value_column: str


class VarianceReducerSpec(ReducerSpec):
    """Declare a method of moments decomposition over named row factors."""

    kind: Literal["variance"] = "variance"
    metric: str
    factors: tuple[VarianceFactorSpec, ...]
    estimator: Literal["moments"]
    interval: BootstrapIntervalSpec
    method_column: str
    value_column: str

    @dx.validates("metric", "method_column", "value_column")
    @classmethod
    def _field_is_named(cls, value: str) -> str:
        """Return a named metric or input column."""
        return ensure_name("a variance reducer field", value)

    @dx.validates("factors")
    @classmethod
    def _factors_are_identified(
        cls, factors: tuple[VarianceFactorSpec, ...]
    ) -> tuple[VarianceFactorSpec, ...]:
        """Return at least two factors with distinct names and columns."""
        if len(factors) < 2:
            msg = "a variance decomposition declares at least two factors"
            raise ValueError(msg)

        names = tuple(factor.name for factor in factors)
        columns = tuple(factor.column for factor in factors)

        if len(set(names)) != len(names) or len(set(columns)) != len(columns):
            msg = f"variance factors repeat a name or column: {factors!r}"
            raise ValueError(msg)

        return factors

    def settings(self) -> VarianceReducerSettings:
        """Return the resolved variance reducer settings."""
        return VarianceReducerSettings(
            metric=self.metric,
            factors=tuple(factor.settings() for factor in self.factors),
            estimator=self.estimator,
            interval=self.interval.settings(),
            method_column=self.method_column,
            value_column=self.value_column,
        )


class RescoringReducerSettings(NamedTuple):
    """Resolved settings for released-prediction rescoring."""

    source: str
    alignments: tuple[AlignmentSettings, ...]
    metric: str
    conventions: ConventionGridSettings
    prediction_format: str
    register: tuple[str, ...]


class RescoringReducerSpec(ReducerSpec):
    """Declare a scoring sweep over released predictions."""

    kind: Literal["rescoring"] = "rescoring"
    source: str
    alignments: tuple[AlignmentSpec, ...]
    metric: str
    conventions: ConventionGridSpec
    prediction_format: str
    register: tuple[str, ...]

    @dx.validates("source", "metric", "prediction_format")
    @classmethod
    def _field_is_named(cls, value: str) -> str:
        """Return a named source, metric, or format."""
        return ensure_name("a rescoring reducer field", value)

    @dx.validates("alignments")
    @classmethod
    def _alignments_are_distinct_and_nonempty(
        cls, alignments: tuple[AlignmentSpec, ...]
    ) -> tuple[AlignmentSpec, ...]:
        """Return one declaration per alignment kind."""
        kinds = tuple(alignment.kind for alignment in alignments)

        if not kinds:
            msg = "a rescoring reducer declares no alignment"
            raise ValueError(msg)

        if len(set(kinds)) != len(kinds):
            msg = f"a rescoring reducer repeats an alignment: {kinds!r}"
            raise ValueError(msg)

        return alignments

    @dx.validates("register")
    @classmethod
    def _register_is_distinct(cls, register: tuple[str, ...]) -> tuple[str, ...]:
        """Return distinct, named convention-provider tags."""
        held = tuple(
            ensure_name("a convention register entry", tag) for tag in register
        )

        if len(set(held)) != len(held):
            msg = f"a convention register repeats an entry: {held!r}"
            raise ValueError(msg)

        return held

    def settings(self) -> RescoringReducerSettings:
        """Return the resolved rescoring settings."""
        return RescoringReducerSettings(
            source=self.source,
            alignments=tuple(
                cast("AlignmentDeclaration", alignment).settings()
                for alignment in self.alignments
            ),
            metric=self.metric,
            conventions=self.conventions.settings(),
            prediction_format=self.prediction_format,
            register=self.register,
        )


class RankChurnReducerSettings(NamedTuple):
    """Resolved settings for a paired ranking comparison."""

    metric: str
    constrained: str
    unconstrained: str
    method_column: str
    condition_column: str
    value_column: str


class RankChurnReducerSpec(ReducerSpec):
    """Declare rank churn between two named scoring conditions."""

    kind: Literal["rank-churn"] = "rank-churn"
    metric: str
    constrained: str
    unconstrained: str
    method_column: str
    condition_column: str
    value_column: str

    @dx.validates(
        "metric",
        "constrained",
        "unconstrained",
        "method_column",
        "condition_column",
        "value_column",
    )
    @classmethod
    def _field_is_named(cls, value: str) -> str:
        """Return a named metric, condition, or input column."""
        return ensure_name("a rank-churn reducer field", value)

    @dx.model_validator()
    def _conditions_differ(self) -> RankChurnReducerSpec:
        """Return a comparison between two distinct conditions."""
        if self.constrained == self.unconstrained:
            msg = "a rank-churn reducer compares two distinct conditions"
            raise ValueError(msg)

        return self

    def settings(self) -> RankChurnReducerSettings:
        """Return the resolved rank-churn settings."""
        return RankChurnReducerSettings(
            metric=self.metric,
            constrained=self.constrained,
            unconstrained=self.unconstrained,
            method_column=self.method_column,
            condition_column=self.condition_column,
            value_column=self.value_column,
        )


def _named_distinct_nonempty(noun: str, values: tuple[str, ...], /) -> tuple[str, ...]:
    """Return a named, distinct, nonempty ordered declaration."""
    held = tuple(ensure_name(noun, value) for value in values)
    if not held:
        msg = f"{noun} declaration is empty"
        raise ValueError(msg)
    if len(set(held)) != len(held):
        msg = f"{noun} declaration repeats a value: {held!r}"
        raise ValueError(msg)
    return held


class TransferReducerSettings(NamedTuple):
    """Resolved columns and language design for zero-shot transfer."""

    metrics: tuple[str, ...]
    source_language: str
    languages: tuple[tuple[str, Literal["gold", "silver"]], ...]
    metric_column: str
    language_column: str
    value_column: str
    count_column: str


class TransferReducerSpec(ReducerSpec):
    """Declare a zero-shot transfer report over completed evaluator rows."""

    kind: Literal["transfer"] = "transfer"
    metrics: tuple[str, ...]
    source_language: str
    languages: dict[str, Literal["gold", "silver"]]
    metric_column: str
    language_column: str
    value_column: str
    count_column: str

    @dx.validates("metrics")
    @classmethod
    def _metrics_are_named_distinct_and_nonempty(
        cls, metrics: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Return the metrics in report order."""
        return _named_distinct_nonempty("a transfer metric", metrics)

    @dx.validates(
        "source_language",
        "metric_column",
        "language_column",
        "value_column",
        "count_column",
    )
    @classmethod
    def _field_is_named(cls, value: str) -> str:
        """Return a named language or input column."""
        return ensure_name("a transfer reducer field", value)

    @dx.validates("languages")
    @classmethod
    def _languages_are_named(
        cls, languages: dict[str, Literal["gold", "silver"]]
    ) -> dict[str, Literal["gold", "silver"]]:
        """Return a nonempty language-to-annotation-class declaration."""
        held = {
            ensure_name("a transfer language", language): source
            for language, source in sorted(languages.items())
        }
        if not held:
            msg = "a transfer reducer declares no language"
            raise ValueError(msg)
        return held

    @dx.model_validator()
    def _source_is_declared(self) -> TransferReducerSpec:
        """Return a design whose source language is one of its languages."""
        if self.source_language not in self.languages:
            msg = f"transfer source {self.source_language!r} is not declared"
            raise ValueError(msg)
        return self

    def settings(self) -> TransferReducerSettings:
        """Return immutable transfer reducer settings."""
        return TransferReducerSettings(
            self.metrics,
            self.source_language,
            tuple(self.languages.items()),
            self.metric_column,
            self.language_column,
            self.value_column,
            self.count_column,
        )


class CrosslingualReducerSettings(NamedTuple):
    """Resolved design for language-level aggregate diagnostics."""

    metrics: tuple[str, ...]
    source_language: str
    metric_column: str
    language_column: str
    value_column: str


class CrosslingualReducerSpec(ReducerSpec):
    """Declare macro, floor, disparity, and transfer gap language summaries."""

    kind: Literal["cross-lingual"] = "cross-lingual"
    metrics: tuple[str, ...]
    source_language: str
    metric_column: str
    language_column: str
    value_column: str

    @dx.validates("metrics")
    @classmethod
    def _metrics_are_named_distinct_and_nonempty(
        cls, values: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Return the metrics in report order."""
        return _named_distinct_nonempty("a cross-lingual metric", values)

    @dx.validates("source_language", "metric_column", "language_column", "value_column")
    @classmethod
    def _field_is_named(cls, value: str) -> str:
        """Return a named source language or input column."""
        return ensure_name("a cross-lingual reducer field", value)

    def settings(self) -> CrosslingualReducerSettings:
        """Return immutable cross-lingual reducer settings."""
        return CrosslingualReducerSettings(
            self.metrics,
            self.source_language,
            self.metric_column,
            self.language_column,
            self.value_column,
        )


class EncoderAblationReducerSettings(NamedTuple):
    """Resolved encoder-by-freeze design and input columns."""

    checkpoints: tuple[str, ...]
    freeze_states: tuple[Literal["frozen", "fine-tuned"], ...]
    metrics: tuple[str, ...]
    minimum_seeds: int
    checkpoint_column: str
    freeze_column: str
    metric_column: str
    seed_column: str
    value_column: str
    weights_column: str


class EncoderAblationReducerSpec(ReducerSpec):
    """Declare the complete encoder-by-freeze ablation table."""

    kind: Literal["encoder-ablation"] = "encoder-ablation"
    checkpoints: tuple[str, ...]
    freeze_states: tuple[Literal["frozen", "fine-tuned"], ...]
    metrics: tuple[str, ...]
    minimum_seeds: int
    checkpoint_column: str
    freeze_column: str
    metric_column: str
    seed_column: str
    value_column: str
    weights_column: str

    @dx.validates("checkpoints")
    @classmethod
    def _checkpoints_are_declared(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Return the distinct checkpoint identities in table order."""
        return _named_distinct_nonempty("an encoder checkpoint", values)

    @dx.validates("metrics")
    @classmethod
    def _metrics_are_declared(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Return the distinct metrics in table order."""
        return _named_distinct_nonempty("an encoder ablation metric", values)

    @dx.validates("freeze_states")
    @classmethod
    def _states_are_declared(
        cls, values: tuple[Literal["frozen", "fine-tuned"], ...]
    ) -> tuple[Literal["frozen", "fine-tuned"], ...]:
        """Return one or both distinct freeze states in table order."""
        if not values or len(set(values)) != len(values):
            msg = f"encoder ablation freeze states are empty or repeated: {values!r}"
            raise ValueError(msg)
        return values

    @dx.validates("minimum_seeds")
    @classmethod
    def _minimum_seeds_estimates_dispersion(cls, value: int) -> int:
        """Return a seed budget sufficient for the shared seed summary."""
        if value < 4:
            msg = f"an encoder ablation requires at least four seeds, not {value}"
            raise ValueError(msg)
        return value

    @dx.validates(
        "checkpoint_column",
        "freeze_column",
        "metric_column",
        "seed_column",
        "value_column",
        "weights_column",
    )
    @classmethod
    def _column_is_named(cls, value: str) -> str:
        """Return a named input column."""
        return ensure_name("an encoder ablation column", value)

    def settings(self) -> EncoderAblationReducerSettings:
        """Return immutable encoder-ablation settings."""
        return EncoderAblationReducerSettings(
            self.checkpoints,
            self.freeze_states,
            self.metrics,
            self.minimum_seeds,
            self.checkpoint_column,
            self.freeze_column,
            self.metric_column,
            self.seed_column,
            self.value_column,
            self.weights_column,
        )


class CrossedRecoveryReducerSettings(NamedTuple):
    """Resolved crossed-recovery factor design and input columns."""

    metrics: tuple[str, ...]
    factor_columns: tuple[str, ...]
    minimum_seeds: int
    null_replicates: int
    metric_column: str
    seed_column: str
    value_column: str
    floor_column: str


class CrossedRecoveryReducerSpec(ReducerSpec):
    """Declare recovery summaries over fully crossed factor cells and seeds."""

    kind: Literal["crossed-recovery"] = "crossed-recovery"
    metrics: tuple[str, ...]
    factor_columns: tuple[str, ...]
    minimum_seeds: int
    null_replicates: int
    metric_column: str
    seed_column: str
    value_column: str
    floor_column: str

    @dx.validates("metrics")
    @classmethod
    def _metrics_are_declared(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Return distinct recovery metrics in report order."""
        return _named_distinct_nonempty("a recovery metric", values)

    @dx.validates("factor_columns")
    @classmethod
    def _factors_are_declared(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Return distinct factor columns in cell-identity order."""
        return _named_distinct_nonempty("a recovery factor column", values)

    @dx.validates("metric_column", "seed_column", "value_column", "floor_column")
    @classmethod
    def _column_is_named(cls, value: str) -> str:
        """Return a named input column."""
        return ensure_name("a crossed-recovery column", value)

    @dx.validates("minimum_seeds")
    @classmethod
    def _minimum_seeds_is_replicated(cls, value: int) -> int:
        """Return a seed floor that can estimate dispersion."""
        if value < 4:
            msg = f"crossed recovery requires at least four seeds, not {value}"
            raise ValueError(msg)
        return value

    @dx.validates("null_replicates")
    @classmethod
    def _null_replicates_is_positive(cls, value: int) -> int:
        """Return a positive sign-flip enumeration or sampling budget."""
        if value < 1:
            msg = f"crossed recovery declares no null replicate: {value}"
            raise ValueError(msg)
        return value

    def settings(self) -> CrossedRecoveryReducerSettings:
        """Return immutable crossed-recovery settings."""
        return CrossedRecoveryReducerSettings(
            self.metrics,
            self.factor_columns,
            self.minimum_seeds,
            self.null_replicates,
            self.metric_column,
            self.seed_column,
            self.value_column,
            self.floor_column,
        )


class CoherenceCollapseReducerSettings(NamedTuple):
    """Resolved KL-versus-cross-entropy collapse comparison."""

    metric: str
    kl: str
    cross_entropy: str
    minimum_seeds: int
    metric_column: str
    divergence_column: str
    seed_column: str
    value_column: str


class CoherenceCollapseReducerSpec(ReducerSpec):
    """Declare a seed-paired collapse comparison between KL and cross-entropy."""

    kind: Literal["coherence-collapse"] = "coherence-collapse"
    metric: str
    kl: str
    cross_entropy: str
    minimum_seeds: int
    metric_column: str
    divergence_column: str
    seed_column: str
    value_column: str

    @dx.validates(
        "metric",
        "kl",
        "cross_entropy",
        "metric_column",
        "divergence_column",
        "seed_column",
        "value_column",
    )
    @classmethod
    def _field_is_named(cls, value: str) -> str:
        """Return a named metric, divergence level, or input column."""
        return ensure_name("a coherence-collapse field", value)

    @dx.validates("minimum_seeds")
    @classmethod
    def _minimum_seeds_estimates_dispersion(cls, value: int) -> int:
        """Return a seed budget sufficient for a paired dispersion estimate."""
        if value < 4:
            msg = f"coherence collapse requires at least four seeds, not {value}"
            raise ValueError(msg)
        return value

    @dx.model_validator()
    def _divergences_differ(self) -> CoherenceCollapseReducerSpec:
        """Return a comparison between two distinct divergence declarations."""
        if self.kl == self.cross_entropy:
            msg = "coherence collapse compares distinct KL and cross-entropy levels"
            raise ValueError(msg)
        return self

    def settings(self) -> CoherenceCollapseReducerSettings:
        """Return the immutable paired-collapse design."""
        return CoherenceCollapseReducerSettings(
            self.metric,
            self.kl,
            self.cross_entropy,
            self.minimum_seeds,
            self.metric_column,
            self.divergence_column,
            self.seed_column,
            self.value_column,
        )


class ResidualStructureReducerSettings(NamedTuple):
    """Resolved columns for the residual-structure regression."""

    response_column: str
    root_entropy_column: str
    non_target_column: str
    derivation_count_column: str


class ResidualStructureReducerSpec(ReducerSpec):
    """Declare the residual-structure regression over held-out row diagnostics."""

    kind: Literal["residual-structure"] = "residual-structure"
    response_column: str
    root_entropy_column: str
    non_target_column: str
    derivation_count_column: str

    @dx.validates(
        "response_column",
        "root_entropy_column",
        "non_target_column",
        "derivation_count_column",
    )
    @classmethod
    def _column_is_named(cls, value: str) -> str:
        """Return a named regression input column."""
        return ensure_name("a residual-structure column", value)

    def settings(self) -> ResidualStructureReducerSettings:
        """Return immutable residual-structure settings."""
        return ResidualStructureReducerSettings(
            self.response_column,
            self.root_entropy_column,
            self.non_target_column,
            self.derivation_count_column,
        )
