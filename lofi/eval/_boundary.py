"""Validated documents entering the evaluation layer."""

from typing import TYPE_CHECKING, Self

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name
from lofi._types import JsonValue

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "BudgetDocument",
    "ComparisonReportDocument",
    "ConstituentDocument",
    "EpochDocument",
    "FittingDocument",
    "HypothesisDocument",
    "ManifestDocument",
    "MetricsDocument",
    "ObjectDocument",
    "PointDocument",
    "PredictionItemDocument",
    "PredictionsDocument",
    "PreregistrationDocument",
    "ProtocolDocument",
    "ProtocolRegisterDocument",
    "PublishedComparisonDocument",
    "ResultRowDocument",
    "RootMassDocument",
    "RootMassItemDocument",
    "RowsDocument",
    "SpanDocument",
    "StepDocument",
    "ValidationError",
    "decode_comparison_report",
    "decode_manifest",
    "decode_metrics",
    "decode_object",
    "decode_predictions",
    "decode_preregistration",
    "decode_protocol_register",
    "decode_root_mass",
    "decode_rows",
    "encode_root_mass",
]

ValidationError = dx.ValidationError
"""The refusal raised when an evaluation boundary document does not validate."""


class HypothesisDocument(dx.Model, extra="forbid"):
    """Decode one preregistered hypothesis."""

    name: str
    left: str
    right: str
    family: str
    margin: float | None = None


class PreregistrationDocument(dx.Model, extra="forbid"):
    """Decode a complete preregistration."""

    grid: dict[str, tuple[JsonValue, ...]]
    hypotheses: tuple[HypothesisDocument, ...]


class ProtocolDocument(dx.Model, extra="forbid"):
    """Decode one protocol-register entry."""

    name: str
    dataset: str
    falsifier: str
    smoke_test: bool
    reason: str | None = None


class ProtocolRegisterDocument(dx.Model, extra="forbid"):
    """Decode the packaged protocol register."""

    protocols: tuple[ProtocolDocument, ...]


class PublishedComparisonDocument(dx.Model, extra="ignore"):
    """Project the falsifier fields from one published comparison."""

    protocol: str
    falsifier: str
    smoke_test: bool


class ComparisonReportDocument(dx.Model, extra="ignore"):
    """Decode the comparisons checked against the protocol register."""

    comparisons: tuple[PublishedComparisonDocument, ...]


class BudgetDocument(dx.Model, extra="forbid"):
    """Decode a released prediction budget."""

    depth: int
    beam: int
    mass: float | None


class SpanDocument(dx.Model, extra="forbid"):
    """Decode one labeled predicted span."""

    start: int
    end: int
    label: str


class ConstituentDocument(dx.Model, extra="forbid"):
    """Decode one gold constituent."""

    start: int
    end: int
    label: str
    depth: int


class PredictionItemDocument(dx.Model, extra="forbid"):
    """Decode one released prediction item."""

    item_id: str
    split: str
    length: int
    punctuation: tuple[int, ...]
    gold: tuple[ConstituentDocument, ...]
    spans: tuple[SpanDocument, ...] | None


class PredictionsDocument(dx.Model, extra="forbid"):
    """Decode a released prediction set."""

    system: str
    budget: BudgetDocument
    items: tuple[PredictionItemDocument, ...]


class ObjectDocument(dx.Model, extra="ignore"):
    """Validate that a boundary document is a JSON object."""


class ManifestDocument(dx.Model, extra="ignore"):
    """Project completion status from a run manifest."""

    status: str


class StepDocument(dx.Model, extra="ignore"):
    """Project numeric telemetry from one training step."""

    metrics: dict[str, float]


class EpochDocument(dx.Model, extra="ignore"):
    """Project the steps completed in one training epoch."""

    steps: tuple[StepDocument, ...]


class FittingDocument(dx.Model, extra="ignore"):
    """Project costs and steps from a completed fitting record."""

    optimizer_steps: int
    epochs: tuple[EpochDocument, ...]
    steps: tuple[StepDocument, ...]


class PointDocument(dx.Model, extra="ignore"):
    """Project the seed from a run's measured point."""

    seed: int


class MetricsDocument(dx.Model, extra="ignore"):
    """Project estimator inputs from a run's metrics record."""

    stage: str
    point: PointDocument
    fitting: FittingDocument


class ResultRowDocument(dx.Model, extra="ignore"):
    """Project a named value from one reported result row."""

    metric: str
    value: float | None


class RowsDocument(dx.Model, extra="ignore"):
    """Decode the result rows emitted by one run."""

    rows: tuple[ResultRowDocument, ...]


class RootMassItemDocument(dx.Model, extra="forbid"):
    """Decode one item's published root log mass pair."""

    item_id: str
    log_mass: float
    log_complement: float

    @dx.validates("item_id")
    @classmethod
    def _item_is_named(cls, value: str) -> str:
        return ensure_name("a root-mass item", value)

    @dx.validates("log_mass", "log_complement")
    @classmethod
    def _log_probability_is_finite(cls, value: float) -> float:
        held = ensure_finite("a root log probability", value)
        if held > 0.0:
            msg = f"a root log probability exceeds zero: {held!r}"
            raise ValueError(msg)
        return held


class RootMassDocument(dx.Model, extra="forbid"):
    """Decode the published root masses from one held-out pass."""

    readout: str
    items: tuple[RootMassItemDocument, ...]

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        return ensure_name("a root-mass readout", value)

    @dx.model_validator()
    def _items_are_nonempty_and_distinct(self) -> Self:
        if not self.items:
            msg = "a root-mass document has no item"
            raise ValueError(msg)
        names = tuple(item.item_id for item in self.items)
        if len(names) != len(set(names)):
            msg = f"a root-mass document repeats an item: {names!r}"
            raise ValueError(msg)
        return self


def decode_preregistration(document: str, /) -> PreregistrationDocument:
    """Validate one preregistration document at the evaluation boundary."""
    return PreregistrationDocument.model_validate_json(document)


def decode_protocol_register(document: str, /) -> ProtocolRegisterDocument:
    """Validate one protocol register at the evaluation boundary."""
    return ProtocolRegisterDocument.model_validate_json(document)


def decode_comparison_report(
    document: Mapping[str, JsonValue], /
) -> ComparisonReportDocument:
    """Validate one comparison report at the evaluation boundary."""
    return ComparisonReportDocument.model_validate(document)


def decode_predictions(document: str, /) -> PredictionsDocument:
    """Validate one released-prediction document at the evaluation boundary."""
    return PredictionsDocument.model_validate_json(document)


def decode_object(document: str, /) -> ObjectDocument:
    """Validate one JSON object at the evaluation boundary."""
    return ObjectDocument.model_validate_json(document)


def decode_manifest(document: str, /) -> ManifestDocument:
    """Validate one projected run manifest at the evaluation boundary."""
    return ManifestDocument.model_validate_json(document)


def decode_metrics(document: str, /) -> MetricsDocument:
    """Validate one projected metrics record at the evaluation boundary."""
    return MetricsDocument.model_validate_json(document)


def decode_rows(document: str, /) -> RowsDocument:
    """Validate one projected result-row record at the evaluation boundary."""
    return RowsDocument.model_validate_json(document)


def decode_root_mass(document: bytes, /) -> RootMassDocument:
    """Validate one published root mass artifact."""
    return RootMassDocument.model_validate_json(document)


def encode_root_mass(document: RootMassDocument, /) -> bytes:
    """Return one published root mass artifact as canonical JSON bytes."""
    return document.model_dump_json().encode("utf-8")
