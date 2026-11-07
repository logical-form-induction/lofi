"""Declarations for quantitative active judgment collection.

An active learning document names everything one collection draws on: the
judgment item construction its candidate pool is built from, the item
attributes its response model reads as factors, how the judgments the first
round starts from are obtained, the judgment fit document whose
response model is refitted each round, the trained point whose readouts score
the candidates, the collector that answers each acquired batch, and the
cross-validation each round's performance is measured by. A document whose
start or collector reads bead responses declares the one response encoding
they are all read through, and a document with no bead reader declares none.
Every path is
relative to the directory holding the document. A run's seeds derive from the
run seed, so the document declares none, the seed of its cross-validation split
included. Every rule that groups or constrains candidates names the item
attributes it reads with a bead DSL expression over the item's
``item_metadata``, such as ``item['structure']``.
"""

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, cast, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name, ensure_positive
from lofi.spec.acquisition.collectors import ActiveCollectorSpec
from lofi.spec.acquisition.constraints import (
    BatchConstraintSpec,
    ListConstraintSpec,
    SizeConstraintSpec,
    batch_settings,
    list_settings,
)
from lofi.spec.held_out import HeldOutParticipantPolicy

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.spec.acquisition.collectors import CollectorSettings
    from lofi.spec.acquisition.constraints import (
        BatchConstraintSettings,
        ListConstraintSettings,
    )

__all__ = [
    "ActiveLearningSettings",
    "ActiveLearningSpec",
    "ActiveStopReason",
    "AgreementStopSettings",
    "AgreementStopSpec",
    "AsymptoteStopSettings",
    "AsymptoteStopSpec",
    "BeadInitialSettings",
    "BeadInitialSpec",
    "CollectedInitialSettings",
    "CollectedInitialSpec",
    "CrossValidationSettings",
    "CrossValidationSpec",
    "InitialSettings",
    "InitialSpec",
    "PointReadoutSettings",
    "PointReadoutSpec",
    "RC21AcquisitionSettings",
    "RC21AcquisitionSpec",
    "RankStopSettings",
    "RankStopSpec",
    "ReadoutMeasure",
    "StoppingSettings",
    "StoppingSpec",
    "TokenizerSettings",
    "TokenizerSpec",
    "TrainedPredictorsSettings",
    "TrainedPredictorsSpec",
    "decode_active_learning",
]

type ReadoutMeasure = Literal["log-mass", "log-complement"]
"""The root-span quantity a mass readout contributes to a predictor column."""

type ActiveStopReason = Literal[
    "agreement", "rank-correlation", "asymptote", "pool-exhausted", "max-rounds"
]
"""Why collection ended: a stopping rule's kind, an exhausted pool, or the bound.

A stopping rule that holds is recorded under its kind. Otherwise collection
ends with ``pool-exhausted`` once the remaining candidates cannot fill another
batch, or with ``max-rounds`` once the declared bound on rounds is reached.
"""


class TokenizerSettings(NamedTuple):
    """Resolved bead tokenizer configuration."""

    backend: Literal["spacy", "stanza", "whitespace"]
    language: str
    model_name: str | None


class RC21AcquisitionSettings(NamedTuple):
    """Resolved RC2.1 batch constraints."""

    kind: Literal["rc2.1"]
    batch_size: int
    allocate_by: str
    constraints: tuple[ListConstraintSettings, ...]
    batch_constraints: tuple[BatchConstraintSettings, ...]
    share_tolerance: float


class AgreementStopSettings(NamedTuple):
    """Resolved agreement stopping threshold."""

    threshold: float
    basis: Literal["estimate", "lower-bound"]


class RankStopSettings(NamedTuple):
    """Resolved cross-validated rank correlation stopping threshold."""

    threshold: float
    basis: Literal["estimate", "lower-bound"]


class AsymptoteStopSettings(NamedTuple):
    """Resolved rank correlation asymptote rule."""

    window: int
    maximum_gain: float


type StoppingSettings = AgreementStopSettings | RankStopSettings | AsymptoteStopSettings


class _StoppingDeclaration(Protocol):
    """A validated stopping declaration exposing immutable settings."""

    def settings(self) -> StoppingSettings:
        """Return the resolved quantitative stopping rule."""
        ...


class BeadInitialSettings(NamedTuple):
    """Resolved bead judgments the first round starts from."""

    kind: Literal["bead"]
    responses: str


class CollectedInitialSettings(NamedTuple):
    """Resolved per-stratum draw the run's collector answers before round 1."""

    kind: Literal["collected"]
    stratify_by: str
    items_per_stratum: int


type InitialSettings = BeadInitialSettings | CollectedInitialSettings
"""The resolved start of an active learning run."""


class _InitialDeclaration(Protocol):
    """A validated start declaration exposing immutable settings."""

    def settings(self) -> InitialSettings:
        """Return the resolved start."""
        ...


class PointReadoutSettings(NamedTuple):
    """Resolved source of one predictor column in a trained point's readouts."""

    readout: str
    measure: ReadoutMeasure


class TrainedPredictorsSettings(NamedTuple):
    """Resolved trained point, the elements it reads and the columns it fills."""

    run: str
    inputs: Mapping[str, str]
    tokenizer: TokenizerSettings
    readouts: Mapping[str, PointReadoutSettings]


class CrossValidationSettings(NamedTuple):
    """Resolved cross-validation each round's performance is measured by."""

    held_out_unit: str
    splitter: str
    folds: int | None
    coverage_threshold: float
    unseen_participant: HeldOutParticipantPolicy | None


class ActiveLearningSettings(NamedTuple):
    """Resolved controls for one active judgment-collection operation."""

    acquisition: RC21AcquisitionSettings
    stopping: StoppingSettings
    max_rounds: int
    confidence: float
    bootstrap_samples: int
    pool: str
    factors: Mapping[str, str]
    initial: InitialSettings
    fit: str
    predictors: TrainedPredictorsSettings
    collector: CollectorSettings
    cross_validation: CrossValidationSettings
    encoding: str | None


class TokenizerSpec(dx.Model, extra="forbid"):
    """Declare a bead tokenizer, in bead's ``TokenizerConfig`` shape.

    Parameters
    ----------
    backend : {"spacy", "stanza", "whitespace"}
        The bead tokenizer backend.
    language : str
        The language code the backend tokenizes.
    model_name : str | None
        The backend's model, or None for the one bead resolves.
    """

    backend: Literal["spacy", "stanza", "whitespace"]
    language: str
    model_name: str | None

    @dx.validates("language")
    @classmethod
    def _language_is_named(cls, value: str) -> str:
        return ensure_name("a tokenizer language", value)

    def settings(self) -> TokenizerSettings:
        """Discard the boundary declaration into immutable settings."""
        return TokenizerSettings(self.backend, self.language, self.model_name)


class RC21AcquisitionSpec(dx.Model, extra="forbid"):
    """Declare the RC2.1 constrained high-variance acquisition rule.

    Parameters
    ----------
    batch_size : int
        Candidates selected each round.
    allocate_by : str
        Expression naming the stratum each candidate is allocated under; each
        stratum's quota is proportional to its performance.
    constraints : tuple[ListConstraintSpec, ...]
        The bead list constraints every selected batch satisfies.
    batch_constraints : tuple[BatchConstraintSpec, ...]
        The bead batch constraints the run's acquired batches satisfy
        together, each round's batch being one list.
    share_tolerance : float
        Largest gap between a stratum's realized share of the batch and its
        target share.
    """

    kind: Literal["rc2.1"] = "rc2.1"
    batch_size: int = 500
    allocate_by: str
    constraints: tuple[ListConstraintSpec, ...]
    batch_constraints: tuple[BatchConstraintSpec, ...] = ()
    share_tolerance: float

    @dx.validates("batch_size")
    @classmethod
    def _batch_is_positive(cls, value: int) -> int:
        return ensure_positive("an acquisition batch size", value)

    @dx.validates("allocate_by")
    @classmethod
    def _allocation_is_named(cls, value: str) -> str:
        return ensure_name("an allocation expression", value)

    @dx.model_validator()
    def _the_batch_size_is_admitted(self) -> RC21AcquisitionSpec:
        """Validate that every declared size constraint admits the batch size."""
        for constraint in self.constraints:
            if isinstance(constraint, SizeConstraintSpec):
                exact, low, high = (
                    constraint.exact_size,
                    constraint.min_size,
                    constraint.max_size,
                )
                if (
                    (exact is not None and exact != self.batch_size)
                    or (low is not None and self.batch_size < low)
                    or (high is not None and self.batch_size > high)
                ):
                    msg = f"a batch of {self.batch_size} violates {constraint!r}"
                    raise ValueError(msg)
        return self

    def settings(self) -> RC21AcquisitionSettings:
        """Discard the boundary declaration into immutable settings."""
        return RC21AcquisitionSettings(
            self.kind,
            self.batch_size,
            self.allocate_by,
            tuple(list_settings(constraint) for constraint in self.constraints),
            tuple(batch_settings(constraint) for constraint in self.batch_constraints),
            self.share_tolerance,
        )


class StoppingSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare the quantitative condition that ends collection."""


class AgreementStopSpec(StoppingSpec):
    """Stop when Krippendorff agreement reaches a declared threshold."""

    kind: Literal["agreement"] = "agreement"
    threshold: float
    basis: Literal["estimate", "lower-bound"] = "lower-bound"

    @dx.validates("threshold")
    @classmethod
    def _threshold_is_a_proportion(cls, value: float) -> float:
        ensure_finite("an agreement threshold", value)
        if not 0.0 <= value <= 1.0:
            msg = f"an agreement threshold is not a proportion: {value!r}"
            raise ValueError(msg)
        return value

    def settings(self) -> AgreementStopSettings:
        """Discard the boundary declaration into immutable settings."""
        return AgreementStopSettings(self.threshold, self.basis)


class RankStopSpec(StoppingSpec):
    """Stop when cross-validated rank correlation reaches a threshold."""

    kind: Literal["rank-correlation"] = "rank-correlation"
    threshold: float
    basis: Literal["estimate", "lower-bound"] = "lower-bound"

    @dx.validates("threshold")
    @classmethod
    def _threshold_is_a_correlation(cls, value: float) -> float:
        ensure_finite("a rank-correlation threshold", value)
        if not -1.0 <= value <= 1.0:
            msg = f"a rank-correlation threshold is outside [-1, 1]: {value!r}"
            raise ValueError(msg)
        return value

    def settings(self) -> RankStopSettings:
        """Discard the boundary declaration into immutable settings."""
        return RankStopSettings(self.threshold, self.basis)


class AsymptoteStopSpec(StoppingSpec):
    """Stop when rank correlation improvement is below a declared gain."""

    kind: Literal["asymptote"] = "asymptote"
    window: int
    maximum_gain: float

    @dx.validates("window")
    @classmethod
    def _window_has_two_rounds(cls, value: int) -> int:
        value = ensure_positive("an asymptote window", value)
        if value < 2:
            msg = f"an asymptote window has fewer than two rounds: {value}"
            raise ValueError(msg)
        return value

    @dx.validates("maximum_gain")
    @classmethod
    def _gain_is_nonnegative(cls, value: float) -> float:
        ensure_finite("an asymptote maximum gain", value)
        if value < 0.0:
            msg = f"an asymptote maximum gain is negative: {value!r}"
            raise ValueError(msg)
        return value

    def settings(self) -> AsymptoteStopSettings:
        """Discard the boundary declaration into immutable settings."""
        return AsymptoteStopSettings(self.window, self.maximum_gain)


class InitialSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare how the judgments the first round starts from are obtained."""

    def reads_bead_responses(self) -> bool:
        """Return whether the start reads bead responses.

        A start that does reads them through the response encoding the
        active learning document declares.
        """
        return False


class BeadInitialSpec(InitialSpec):
    """Start from bead judgments already collected.

    The records are read through the response encoding the active learning
    document declares.

    Parameters
    ----------
    responses : str
        JSON Lines file of bead annotation records.
    """

    kind: Literal["bead"] = "bead"
    responses: str

    @dx.validates("responses")
    @classmethod
    def _responses_are_named(cls, value: str) -> str:
        return ensure_name("an initial-judgment document", value)

    @override
    def reads_bead_responses(self) -> bool:
        """Return that the start reads bead responses."""
        return True

    def settings(self) -> BeadInitialSettings:
        """Discard the boundary declaration into immutable settings."""
        return BeadInitialSettings(self.kind, self.responses)


class CollectedInitialSpec(InitialSpec):
    """Start from the run's collector answering a seeded per-stratum draw.

    Before round 1 the run draws ``items_per_stratum`` candidates from each
    stratum of the pool, without replacement, from a seed derived from the run
    seed, and the collector answers them as round 0's batch.

    Parameters
    ----------
    stratify_by : str
        Expression naming the stratum each candidate is drawn from.
    items_per_stratum : int
        Candidates drawn from every stratum.
    """

    kind: Literal["collected"] = "collected"
    stratify_by: str
    items_per_stratum: int

    @dx.validates("stratify_by")
    @classmethod
    def _stratification_is_named(cls, value: str) -> str:
        return ensure_name("an initial draw's stratification", value)

    @dx.validates("items_per_stratum")
    @classmethod
    def _items_are_positive(cls, value: int) -> int:
        return ensure_positive("an initial draw's items per stratum", value)

    def settings(self) -> CollectedInitialSettings:
        """Discard the boundary declaration into immutable settings."""
        return CollectedInitialSettings(
            self.kind, self.stratify_by, self.items_per_stratum
        )


class PointReadoutSpec(dx.Model, extra="forbid"):
    """Declare which published readout fills one predictor column.

    Parameters
    ----------
    readout : str
        Identifier of a mass readout the trained point publishes.
    measure : ReadoutMeasure
        The root-span log mass or log complement the column holds.
    """

    readout: str
    measure: ReadoutMeasure

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        return ensure_name("a trained-point readout", value)

    def settings(self) -> PointReadoutSettings:
        """Discard the boundary declaration into immutable settings."""
        return PointReadoutSettings(self.readout, self.measure)


class TrainedPredictorsSpec(dx.Model, extra="forbid"):
    """Declare the trained point whose readouts score every candidate.

    Parameters
    ----------
    run : str
        Run directory of a completed point.
    inputs : dict[str, str]
        The rendered element of every pool item each input role the point
        publishes reads, keyed by role.
    tokenizer : TokenizerSpec
        The bead tokenizer an element's text is split into the words the point
        reads by.
    readouts : dict[str, PointReadoutSpec]
        The readout filling each predictor column, keyed by the column name
        the judgment fit's response model reads, held in column-name order.
    """

    run: str
    inputs: dict[str, str]
    tokenizer: TokenizerSpec
    readouts: dict[str, PointReadoutSpec]

    @dx.validates("run")
    @classmethod
    def _run_is_named(cls, value: str) -> str:
        return ensure_name("a trained-point run directory", value)

    @dx.validates("inputs")
    @classmethod
    def _inputs_are_named(cls, value: dict[str, str]) -> dict[str, str]:
        return {
            ensure_name("a trained point's input role", role): ensure_name(
                f"input role {role!r}'s element", element
            )
            for role, element in value.items()
        }

    @dx.validates("readouts")
    @classmethod
    def _readouts_are_named(
        cls, readouts: dict[str, PointReadoutSpec]
    ) -> dict[str, PointReadoutSpec]:
        if not readouts:
            msg = "a trained point fills at least one predictor column"
            raise ValueError(msg)
        for column in readouts:
            ensure_name("a predictor column", column)
        return dict(sorted(readouts.items()))

    def settings(self) -> TrainedPredictorsSettings:
        """Discard the boundary declaration into immutable settings."""
        return TrainedPredictorsSettings(
            self.run,
            MappingProxyType(dict(self.inputs)),
            self.tokenizer.settings(),
            MappingProxyType(
                {
                    column: declared.settings()
                    for column, declared in self.readouts.items()
                }
            ),
        )


class CrossValidationSpec(dx.Model, extra="forbid"):
    """Declare the cross-validation each round's performance is measured by.

    Every fold is refitted by MAP on its retained judgments and predicts its
    held-out ones. The split's seed derives from the run seed.

    Parameters
    ----------
    held_out_unit : str
        The row column held out together: ``response``, ``participant``,
        ``item``, or one of the document's declared factors.
    splitter : str
        Tag of a splitter in the ``lofi.splitter`` entry point group.
    folds : int | None
        Number of folds, at least two, or null for one fold per level of the
        held-out unit, as the splitter requires.
    coverage_threshold : float
        Largest share of a fold's held-out participants its retained judgments
        may never show before the fold needs an unseen-participant rule.
    unseen_participant : HeldOutParticipantPolicy | None
        The rule predicting a participant a fold never retained, applied to
        every such participant, or null for none. Candidates are scored for an
        unseen participant under this rule, or at the population mean where
        none is declared.
    """

    held_out_unit: str
    splitter: str
    folds: int | None
    coverage_threshold: float
    unseen_participant: HeldOutParticipantPolicy | None

    @dx.validates("held_out_unit")
    @classmethod
    def _unit_is_named(cls, value: str) -> str:
        return ensure_name("a cross-validation's held-out unit", value)

    @dx.validates("splitter")
    @classmethod
    def _splitter_is_named(cls, value: str) -> str:
        return ensure_name("a cross-validation splitter", value)

    @dx.validates("folds")
    @classmethod
    def _folds_are_several(cls, value: int | None) -> int | None:
        if value is not None and value < 2:
            msg = f"a cross-validation fold count is below two: {value!r}"
            raise ValueError(msg)
        return value

    @dx.validates("coverage_threshold")
    @classmethod
    def _coverage_is_a_proportion(cls, value: float) -> float:
        ensure_finite("a cross-validation coverage threshold", value)
        if not 0.0 <= value <= 1.0:
            msg = (
                f"a cross-validation coverage threshold is not a proportion: {value!r}"
            )
            raise ValueError(msg)
        return value

    def settings(self) -> CrossValidationSettings:
        """Discard the boundary declaration into immutable settings."""
        return CrossValidationSettings(
            self.held_out_unit,
            self.splitter,
            self.folds,
            self.coverage_threshold,
            self.unseen_participant,
        )


class _CollectorDeclaration(Protocol):
    """A validated collector declaration exposing immutable settings."""

    def settings(self) -> CollectorSettings:
        """Return the resolved collector."""
        ...


class ActiveLearningSpec(dx.Model, extra="forbid"):
    """Declare one bounded active judgment-collection operation.

    Parameters
    ----------
    acquisition : RC21AcquisitionSpec
        The rule each round's batch is selected by.
    stopping : StoppingSpec
        The condition, checked after each round, that ends collection.
    max_rounds : int
        Bound on the number of rounds.
    confidence : float
        Level of the agreement and rank correlation intervals.
    bootstrap_samples : int
        Resamples behind the agreement interval.
    pool : str
        Judgment item construction document the candidate pool is built from.
    factors : dict[str, str]
        The bead expression each factor every judgment row records is read
        by, keyed by the factor's name; the response model's factor terms read
        these.
    initial : InitialSpec
        The bead judgments the first round starts from, or the per-stratum draw
        the collector answers before it.
    fit : str
        Judgment fit document whose response model is refitted by MAP each
        round.
    predictors : TrainedPredictorsSpec
        Trained point whose readouts are the response model's predictors.
    collector : ActiveCollectorSpec
        Collector answering each acquired batch.
    cross_validation : CrossValidationSpec
        Cross-validation each round's performance is measured by.
    encoding : str | None
        Bead response-encoding document naming the question every bead
        response answers and its labels in response order, required when the
        start or the collector reads bead responses and refused otherwise.
    """

    acquisition: RC21AcquisitionSpec
    stopping: StoppingSpec
    max_rounds: int
    confidence: float = 0.95
    bootstrap_samples: int = 1_000
    pool: str
    factors: dict[str, str]
    initial: InitialSpec
    fit: str
    predictors: TrainedPredictorsSpec
    collector: ActiveCollectorSpec
    cross_validation: CrossValidationSpec
    encoding: str | None = None

    @dx.validates("pool", "fit")
    @classmethod
    def _documents_are_named(cls, value: str) -> str:
        return ensure_name("an active-learning document", value)

    @dx.validates("factors")
    @classmethod
    def _factors_are_named(cls, value: dict[str, str]) -> dict[str, str]:
        return {
            ensure_name("an active-learning factor", name): ensure_name(
                f"factor {name!r}'s expression", expression
            )
            for name, expression in value.items()
        }

    @dx.validates("encoding")
    @classmethod
    def _encoding_is_named(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ensure_name("an active-learning response encoding", value)

    @dx.model_validator()
    def _encoding_has_a_reader(self) -> ActiveLearningSpec:
        """Validate that the encoding is declared exactly when a reader needs it."""
        readers = tuple(
            name
            for name, reads in (
                ("initial", self.initial.reads_bead_responses()),
                ("collector", self.collector.reads_bead_responses()),
            )
            if reads
        )
        if readers and self.encoding is None:
            msg = (
                "the active-learning document reads bead responses in its "
                f"{' and '.join(readers)} and declares no response encoding"
            )
            raise ValueError(msg)
        if not readers and self.encoding is not None:
            msg = (
                f"the active-learning document declares the response encoding "
                f"{self.encoding!r} and no bead reader"
            )
            raise ValueError(msg)
        return self

    @dx.validates("max_rounds", "bootstrap_samples")
    @classmethod
    def _counts_are_positive(cls, value: int) -> int:
        return ensure_positive("an active-learning count", value)

    @dx.validates("confidence")
    @classmethod
    def _confidence_is_open_proportion(cls, value: float) -> float:
        ensure_finite("an active-learning confidence level", value)
        if not 0.0 < value < 1.0:
            msg = f"an active-learning confidence level is not in (0, 1): {value!r}"
            raise ValueError(msg)
        return value

    def settings(self) -> ActiveLearningSettings:
        """Discard the boundary document into immutable runtime settings."""
        stopping = cast("_StoppingDeclaration", self.stopping)
        collector = cast("_CollectorDeclaration", self.collector)
        initial = cast("_InitialDeclaration", self.initial)
        return ActiveLearningSettings(
            self.acquisition.settings(),
            stopping.settings(),
            self.max_rounds,
            self.confidence,
            self.bootstrap_samples,
            self.pool,
            MappingProxyType(dict(self.factors)),
            initial.settings(),
            self.fit,
            self.predictors.settings(),
            collector.settings(),
            self.cross_validation.settings(),
            self.encoding,
        )


def decode_active_learning(data: bytes, /) -> ActiveLearningSpec:
    """Decode and validate one active learning JSON document."""
    return ActiveLearningSpec.model_validate_json(data)
