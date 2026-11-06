"""Released-prediction rescoring under alternate conventions."""

import json
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    NamedTuple,
    Protocol,
    cast,
    final,
    override,
    runtime_checkable,
)

from lofi._checks import ensure_commit, ensure_finite, ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi._types import ItemId
from lofi.eval._alignment import apply_alignment
from lofi.eval._boundary import (
    PredictionItemDocument,
    ValidationError,
    decode_predictions,
)
from lofi.eval._bracketing import Constituent, GoldBracketing
from lofi.eval._declared_alignment import fit_declared_alignment
from lofi.eval._identity import ConfigSource, ReportIdentity
from lofi.eval._matched import InducedBracketing, RecallHomogeneity
from lofi.eval._report import ReportRow
from lofi.eval._residual import Approximation
from lofi.eval._values import ConventionVector, MetricResult
from lofi.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.eval._alignment import Contingency, Split
    from lofi.eval._residual import Residual
    from lofi.eval._values import (
        Aggregation,
        CoveragePenalty,
        GoldDerivation,
        PunctuationPolicy,
        SelfArcPolicy,
        TrivialSpanPolicy,
        UnaryChainPolicy,
    )
    from lofi.formalism import Grammar
    from lofi.spec.evaluators.alignment import AlignmentSettings
    from lofi.spec.reducers import ConventionGridSettings


__all__ = [
    "RECALL_HOMOGENEITY_METRIC",
    "CheckpointRescorer",
    "CheckpointScore",
    "ConventionProvider",
    "DeclaredBudget",
    "PredictionError",
    "PredictionItem",
    "PredictionReader",
    "RecallHomogeneityMetric",
    "ReleasedPredictions",
    "RescoredCell",
    "RescoringDelta",
    "RescoringDeltaTable",
    "RescoringMetric",
    "RescoringRow",
    "declared_conventions",
    "read_declared_predictions",
    "read_predictions",
    "rescore_declared",
    "rescore_frozen_checkpoint",
    "rescore_predictions",
    "rescoring_metric",
]


class PredictionReader(Protocol):
    """Reads one declared released-prediction wire format."""

    def __call__(self, text: str, /) -> ReleasedPredictions:
        """Return the validated released predictions encoded by ``text``."""
        ...


class CheckpointRescorer(Protocol):
    """Scores one frozen checkpoint under the code at the current head."""

    def __call__(self, checkpoint: Path, /) -> tuple[CheckpointScore, ...]:
        """Return named scalar scores for ``checkpoint``."""
        ...


class ConventionProvider(Protocol):
    """Contributes convention vectors to a declared rescoring sweep."""

    def __call__(self) -> tuple[ConventionVector, ...]:
        """Return contributed vectors in declared order."""
        ...


@runtime_checkable
class RescoringMetric(Protocol):
    """Produces the contingency a declared alignment scores."""

    @property
    def name(self) -> str:
        """Return the metric name written on report rows."""
        ...

    def contingency(
        self,
        predictions: ReleasedPredictions,
        conventions: ConventionVector,
        split: str,
        /,
    ) -> tuple[Contingency, str]:
        """Return matched counts and their resampling unit."""
        ...


class PredictionError(ValueError):
    """Signals an invalid released prediction set."""


@final
class CheckpointScore(FrozenValue):
    """One named score produced from a frozen checkpoint."""

    _noun = "checkpoint score"
    __slots__ = ("_metric", "_value")

    def __init__(self, metric: str, value: float) -> None:
        object.__setattr__(self, "_metric", ensure_name("a checkpoint metric", metric))
        object.__setattr__(
            self, "_value", ensure_finite("a checkpoint metric value", value)
        )

    @property
    def metric(self) -> str:
        """Return the metric identifier."""
        return self._metric

    @property
    def value(self) -> float:
        """Return the score."""
        return self._value


@final
class RescoringDelta(FrozenValue):
    """One frozen-to-head checkpoint score difference."""

    _noun = "rescoring delta"
    __slots__ = ("_frozen", "_head", "_metric", "_tolerance")

    def __init__(
        self, metric: str, frozen: float, head: float, *, tolerance: float
    ) -> None:
        checked_tolerance = ensure_finite("a rescoring tolerance", tolerance)

        if checked_tolerance < 0.0:
            msg = f"a rescoring tolerance is negative: {tolerance!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_metric", ensure_name("a rescored metric", metric))
        object.__setattr__(
            self, "_frozen", ensure_finite("a frozen checkpoint score", frozen)
        )
        object.__setattr__(
            self, "_head", ensure_finite("a head checkpoint score", head)
        )
        object.__setattr__(self, "_tolerance", checked_tolerance)

    @property
    def metric(self) -> str:
        """Return the metric identifier."""
        return self._metric

    @property
    def frozen(self) -> float:
        """Return the score recorded at the frozen revision."""
        return self._frozen

    @property
    def head(self) -> float:
        """Return the score produced at the new head."""
        return self._head

    @property
    def delta(self) -> float:
        """Return the signed head-minus-frozen difference."""
        return self._head - self._frozen

    @property
    def tolerance(self) -> float:
        """Return the committed absolute tolerance."""
        return self._tolerance

    @property
    def within_tolerance(self) -> bool:
        """Return whether the absolute difference meets the tolerance."""
        return abs(self.delta) <= self._tolerance

    def to_spec(self) -> dict[str, JsonValue]:
        """Return this difference as JSON-safe scalar columns."""
        return {
            "metric": self._metric,
            "frozen": self._frozen,
            "head": self._head,
            "delta": self.delta,
            "tolerance": self._tolerance,
            "within_tolerance": self.within_tolerance,
        }


@final
class RescoringDeltaTable(FrozenValue):
    """A committed checkpoint's score changes under a new source revision."""

    _noun = "rescoring delta table"
    __slots__ = (
        "_checkpoint",
        "_frozen_revision",
        "_head_revision",
        "_rows",
    )

    def __init__(
        self,
        checkpoint: str,
        frozen_revision: str,
        head_revision: str,
        rows: Sequence[RescoringDelta],
    ) -> None:
        held = tuple(rows)

        if not held:
            msg = "a rescoring delta table has no metric rows"
            raise ValueError(msg)

        if any(not isinstance(row, RescoringDelta) for row in held):
            msg = f"a rescoring delta table contains a non-delta row: {held!r}"
            raise TypeError(msg)

        metrics = tuple(row.metric for row in held)

        if tuple(sorted(metrics)) != metrics or len(set(metrics)) != len(metrics):
            msg = f"rescoring delta metrics are not unique lexical order: {metrics!r}"
            raise ValueError(msg)

        frozen_commit = ensure_commit("a frozen checkpoint revision", frozen_revision)
        head_commit = ensure_commit("a rescoring head revision", head_revision)

        if frozen_commit == head_commit:
            msg = "checkpoint rescoring requires a new head revision"
            raise ValueError(msg)

        object.__setattr__(
            self, "_checkpoint", ensure_name("a frozen checkpoint", checkpoint)
        )
        object.__setattr__(self, "_frozen_revision", frozen_commit)
        object.__setattr__(self, "_head_revision", head_commit)
        object.__setattr__(self, "_rows", held)

    @property
    def checkpoint(self) -> str:
        """Return the frozen checkpoint identifier."""
        return self._checkpoint

    @property
    def frozen_revision(self) -> str:
        """Return the source revision at which the checkpoint was frozen."""
        return self._frozen_revision

    @property
    def head_revision(self) -> str:
        """Return the source revision used to rescore the checkpoint."""
        return self._head_revision

    @property
    def rows(self) -> tuple[RescoringDelta, ...]:
        """Return metric differences in lexical metric order."""
        return self._rows

    @property
    def changed(self) -> int:
        """Return the number of metrics outside their committed tolerance."""
        return sum(not row.within_tolerance for row in self._rows)

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the checkpoint provenance and its JSON-safe delta table."""
        return {
            "checkpoint": self._checkpoint,
            "frozen_revision": self._frozen_revision,
            "head_revision": self._head_revision,
            "changed": self.changed,
            "rows": [row.to_spec() for row in self._rows],
        }


def _indexed_scores(
    scores: Sequence[CheckpointScore], origin: str, /
) -> dict[str, float]:
    """Return a complete named score mapping from one revision."""
    indexed: dict[str, float] = {}

    for score in scores:
        if not isinstance(score, CheckpointScore):
            msg = f"{origin} contains a non-checkpoint score: {score!r}"
            raise TypeError(msg)

        if score.metric in indexed:
            msg = f"{origin} repeats checkpoint metric {score.metric!r}"
            raise ValueError(msg)

        indexed[score.metric] = score.value

    if not indexed:
        msg = f"{origin} contains no checkpoint scores"
        raise ValueError(msg)

    return indexed


def rescore_frozen_checkpoint(
    checkpoint: Path,
    frozen: Sequence[CheckpointScore],
    /,
    *,
    scorer: CheckpointRescorer,
    tolerances: Mapping[str, float],
    frozen_revision: str,
    head_revision: str,
) -> RescoringDeltaTable:
    """Rescore a frozen checkpoint and compare it with its committed scores."""
    if not isinstance(checkpoint, Path):
        msg = f"a frozen checkpoint is not a path: {checkpoint!r}"
        raise TypeError(msg)

    baseline = _indexed_scores(frozen, "the committed delta baseline")
    current = _indexed_scores(scorer(checkpoint), "the head rescoring")

    if baseline.keys() != current.keys():
        msg = (
            "frozen and head checkpoint metric sets differ: "
            f"{tuple(sorted(baseline))!r} != {tuple(sorted(current))!r}"
        )
        raise ValueError(msg)

    if baseline.keys() != tolerances.keys():
        msg = (
            "checkpoint tolerance metrics differ from the committed baseline: "
            f"{tuple(sorted(tolerances))!r} != {tuple(sorted(baseline))!r}"
        )
        raise ValueError(msg)

    rows = tuple(
        RescoringDelta(
            metric,
            baseline[metric],
            current[metric],
            tolerance=tolerances[metric],
        )
        for metric in sorted(baseline)
    )
    return RescoringDeltaTable(str(checkpoint), frozen_revision, head_revision, rows)


class DeclaredBudget:
    """Prediction search depth, beam width, and mass limit.

    Attributes
    ----------
    depth : int
        The enumeration depth.
    beam : int
        The beam width.
    mass : float | None
        The optional maximum omitted mass.

    Parameters
    ----------
    depth : int
        The enumeration depth.
    beam : int
        The beam width.
    mass : float | None
        The optional maximum omitted mass.

    Raises
    ------
    TypeError
        If a bound has an invalid type.
    ValueError
        If a bound is negative or nonfinite.
    """

    __slots__ = ("_beam", "_depth", "_mass")

    def __init__(self, depth: int, beam: int, *, mass: float | None) -> None:
        self._depth = ensure_non_negative("a declared budget's depth", depth)
        self._beam = ensure_non_negative("a declared budget's beam", beam)
        self._mass = (
            None if mass is None else ensure_finite("a declared budget's mass", mass)
        )

    @property
    def depth(self) -> int:
        """Return the enumeration depth."""
        return self._depth

    @property
    def beam(self) -> int:
        """Return the beam width."""
        return self._beam

    @property
    def mass(self) -> float | None:
        """Return the maximum omitted mass, if set."""
        return self._mass

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a budget with the same three bounds."""
        if type(other) is not DeclaredBudget:
            return NotImplemented

        return (
            self._depth == other._depth
            and self._beam == other._beam
            and self._mass == other._mass
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the three bounds."""
        return hash((DeclaredBudget, self._depth, self._beam, self._mass))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this budget."""
        return f"DeclaredBudget({self._depth!r}, {self._beam!r}, mass={self._mass!r})"


class PredictionItem(NamedTuple):
    """A released prediction with its gold analysis and split.

    Parameters
    ----------
    item_id : ItemId
        The item identifier.
    split : Split
        The item's data split.
    pred : InducedBracketing | None
        The predicted bracketing, if available.
    gold : GoldBracketing
        The gold bracketing.
    """

    item_id: ItemId
    split: Split
    pred: InducedBracketing | None
    gold: GoldBracketing


class RescoringRow(NamedTuple):
    """A rescoring result for one convention vector and alignment family.

    Parameters
    ----------
    conventions : ConventionVector
        The scoring conventions.
    family : str
        The alignment family.
    source : str
        The prediction-set digest.
    row : ReportRow
        The report row.
    """

    conventions: ConventionVector
    family: str
    source: str
    row: ReportRow

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the sweep cell as output columns.

        Returns
        -------
        dict[str, JsonValue]
            The flattened rescoring row.
        """
        return {
            **dict(self.conventions.factors),
            "family": self.family,
            "source": self.source,
            **self.row.to_spec(),
        }


@final
class RescoredCell(FrozenValue):
    """One identity-free cell emitted by an experiment rescoring reducer."""

    _noun = "rescored cell"
    __slots__ = (
        "_approximation",
        "_conventions",
        "_family",
        "_method",
        "_metric",
        "_result",
        "_source",
    )

    def __init__(
        self,
        conventions: ConventionVector,
        family: str,
        source: str,
        method: str,
        metric: str,
        result: MetricResult,
        approximation: Approximation,
    ) -> None:
        object.__setattr__(self, "_conventions", conventions)
        object.__setattr__(self, "_family", ensure_name("an alignment family", family))
        object.__setattr__(self, "_source", ensure_name("a prediction source", source))
        object.__setattr__(self, "_method", ensure_name("a rescored method", method))
        object.__setattr__(self, "_metric", ensure_name("a rescoring metric", metric))
        object.__setattr__(self, "_result", result)
        object.__setattr__(self, "_approximation", approximation)

    @property
    def conventions(self) -> ConventionVector:
        """Return the convention vector used for this cell."""
        return self._conventions

    @property
    def family(self) -> str:
        """Return the fitted alignment family."""
        return self._family

    @property
    def source(self) -> str:
        """Return the digest of the released predictions."""
        return self._source

    @property
    def method(self) -> str:
        """Return the released system name."""
        return self._method

    @property
    def metric(self) -> str:
        """Return the declared metric name."""
        return self._metric

    @property
    def result(self) -> MetricResult:
        """Return the scored metric result."""
        return self._result

    @property
    def approximation(self) -> Approximation:
        """Return the released prediction budget and residual."""
        return self._approximation

    def to_spec(self) -> dict[str, JsonValue]:
        """Return this reducer cell as flat report columns."""
        interval = self._result.ci
        return {
            **dict(self._conventions.factors),
            "family": self._family,
            "source": self._source,
            "method": self._method,
            "metric": self._metric,
            "value": self._result.value,
            "n": self._result.n,
            "resampling_unit": self._result.resampling.unit,
            "resampling_seed": self._result.resampling.seed,
            "ci_low": None if interval is None else interval.low,
            "ci_high": None if interval is None else interval.high,
            "ci_level": None if interval is None else interval.level,
            **self._approximation.to_spec(),
        }


class ReleasedPredictions:
    """Released predictions, their budget, and digest.

    Attributes
    ----------
    system : str
        The system name.
    budget : DeclaredBudget
        The prediction budget.
    items : tuple[PredictionItem, ...]
        The released items in file order.
    digest : str
        The source-text digest.

    Parameters
    ----------
    system : str
        The system name.
    budget : DeclaredBudget
        The prediction budget.
    items : Sequence[PredictionItem]
        The released items.
    digest : str
        The source-text digest.

    Raises
    ------
    TypeError
        If the budget, system, digest, or an item has an invalid type.
    ValueError
        If a text field is empty or an item identifier is repeated.
    """

    __slots__ = ("_budget", "_digest", "_items", "_system")

    def __init__(
        self,
        *,
        system: str,
        budget: DeclaredBudget,
        items: Sequence[PredictionItem],
        digest: str,
    ) -> None:
        if not isinstance(budget, DeclaredBudget):
            msg = f"a released prediction set's budget is not one: {budget!r}"
            raise TypeError(msg)

        self._system = ensure_name("a released prediction set's system", system)
        self._budget = budget
        self._items = tuple(items)
        self._digest = ensure_name("a released prediction set's digest", digest)
        seen: set[ItemId] = set()

        for entry in self._items:
            if not isinstance(entry, PredictionItem):
                msg = f"a released prediction is not an item: {entry!r}"
                raise TypeError(msg)

            if entry.item_id in seen:
                msg = f"the item {entry.item_id!r} is released twice"
                raise ValueError(msg)

            seen.add(entry.item_id)

    @property
    def system(self) -> str:
        """Return the system name."""
        return self._system

    @property
    def budget(self) -> DeclaredBudget:
        """Return the prediction budget."""
        return self._budget

    @property
    def items(self) -> tuple[PredictionItem, ...]:
        """Return prediction items in file order."""
        return self._items

    @property
    def digest(self) -> str:
        """Return the source text digest."""
        return self._digest

    def on(self, split: Split) -> tuple[PredictionItem, ...]:
        """Return one split's prediction items in file order.

        Parameters
        ----------
        split : Split
            The split to select.

        Returns
        -------
        tuple[PredictionItem, ...]
            The matching items.
        """
        return tuple(entry for entry in self._items if entry.split == split)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same released set."""
        if type(other) is not ReleasedPredictions:
            return NotImplemented

        return (
            self._system == other._system
            and self._budget == other._budget
            and self._items == other._items
            and self._digest == other._digest
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the system, the budget, the items and the digest."""
        return hash(
            (ReleasedPredictions, self._system, self._budget, self._items, self._digest)
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this set."""
        return (
            f"ReleasedPredictions(system={self._system!r}, budget={self._budget!r}, "
            f"items={list(self._items)!r}, digest={self._digest!r})"
        )


def read_predictions(text: str) -> ReleasedPredictions:
    """Parse a released prediction set.

    Parameters
    ----------
    text : str
        The released prediction JSON.

    Returns
    -------
    ReleasedPredictions
        The parsed predictions and source digest.

    Raises
    ------
    PredictionError
        If the document or a required field is invalid.
    """
    try:
        decoded = decode_predictions(text)
    except json.JSONDecodeError as error:
        msg = f"a released prediction set is not JSON: {error}"
        raise PredictionError(msg) from error
    except (ValidationError, TypeError) as error:
        msg = f"a released prediction set does not validate: {error}"
        raise PredictionError(msg) from error

    try:
        return ReleasedPredictions(
            system=decoded.system,
            budget=DeclaredBudget(
                decoded.budget.depth,
                decoded.budget.beam,
                mass=decoded.budget.mass,
            ),
            items=[_prediction_item(entry) for entry in decoded.items],
            digest=sha256(text.encode("utf-8")).hexdigest(),
        )
    except (TypeError, ValueError) as refusal:
        msg = f"a released prediction set was refused: {refusal}"
        raise PredictionError(msg) from refusal


def read_declared_predictions(
    text: str,
    prediction_format: str,
    /,
    *,
    registry: Registry[PredictionReader] | None = None,
) -> ReleasedPredictions:
    """Read predictions through the declared format plugin."""
    readers = (
        Registry[PredictionReader]("prediction_format")
        if registry is None
        else registry
    )
    return readers.get(prediction_format)(text)


class RecallHomogeneityMetric:
    """Build recall-homogeneity contingencies for declared rescoring."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the registered metric name."""
        return "recall-homogeneity"

    def contingency(
        self,
        predictions: ReleasedPredictions,
        conventions: ConventionVector,
        split: str,
        /,
    ) -> tuple[Contingency, str]:
        """Return one split's matched-constituent contingency."""
        return _matched(predictions, conventions, split=split)


RECALL_HOMOGENEITY_METRIC: Final = RecallHomogeneityMetric()
"""The built-in recall-homogeneity rescoring metric plugin."""


def rescoring_metric(
    name: str,
    /,
    *,
    registry: Registry[RescoringMetric] | None = None,
) -> RescoringMetric:
    """Return the registered rescoring metric named by a declaration."""
    metrics = (
        Registry[RescoringMetric]("rescoring_metric") if registry is None else registry
    )
    member = metrics.get(name)

    if not isinstance(member, RescoringMetric):
        msg = f"the rescoring metric {name!r} names {member!r}, not a metric"
        raise TypeError(msg)

    return member


def declared_conventions(
    grid: ConventionGridSettings,
    register: Sequence[str],
    /,
    *,
    registry: Registry[ConventionProvider] | None = None,
) -> tuple[ConventionVector, ...]:
    """Return the declared convention grid followed by registered contributions."""
    crossed = tuple(
        ConventionVector(
            punctuation=cast("PunctuationPolicy", values[0]),
            unary_chains=cast("UnaryChainPolicy", values[1]),
            trivial_spans=cast("TrivialSpanPolicy", values[2]),
            minimum_length=values[3],
            aggregation=cast("Aggregation", values[4]),
            coverage=cast("CoveragePenalty", values[5]),
            self_arcs=cast("SelfArcPolicy", values[6]),
            gold_derivation=cast("GoldDerivation", values[7]),
        )
        for values in product(*grid)
    )
    providers = (
        Registry[ConventionProvider]("convention") if registry is None else registry
    )
    contributed = tuple(vector for tag in register for vector in providers.get(tag)())

    for vector in contributed:
        if not isinstance(vector, ConventionVector):
            msg = f"a convention provider returned a non-vector: {vector!r}"
            raise TypeError(msg)

    combined = (*crossed, *contributed)

    if len(set(combined)) != len(combined):
        msg = "a convention register repeats a vector already in the rescoring sweep"
        raise ValueError(msg)

    return combined


def rescore_predictions(
    predictions: ReleasedPredictions,
    *,
    alignments: Sequence[AlignmentSettings],
    metric: RescoringMetric,
    conventions: Sequence[ConventionVector],
    residual: Residual | None = None,
) -> tuple[RescoredCell, ...]:
    """Return the identity-free alignment-by-convention sweep a reducer names."""
    if not isinstance(predictions, ReleasedPredictions):
        msg = f"a re-scored prediction set is not one: {predictions!r}"
        raise TypeError(msg)

    if not isinstance(metric, RescoringMetric):
        msg = f"a rescoring metric is not one: {metric!r}"
        raise TypeError(msg)

    approximation = Approximation(predictions.budget, residual=residual)
    rows: list[RescoredCell] = []

    for convention in conventions:
        if not isinstance(convention, ConventionVector):
            msg = f"a rescoring convention is not a vector: {convention!r}"
            raise TypeError(msg)

        for settings in alignments:
            development, _ = metric.contingency(
                predictions, convention, settings.fit_on
            )
            held_out, unit = metric.contingency(
                predictions, convention, settings.score_on
            )
            fitted = fit_declared_alignment(development, settings)
            result = apply_alignment(fitted.alignment, held_out, unit=unit)
            rows.append(
                RescoredCell(
                    convention,
                    settings.kind,
                    predictions.digest,
                    predictions.system,
                    metric.name,
                    result,
                    approximation,
                )
            )

    return tuple(rows)


def rescore_declared(
    predictions: ReleasedPredictions,
    *,
    run_spec: ConfigSource,
    grammar: Grammar,
    freeze_id: str,
    alignments: Sequence[AlignmentSettings],
    metric: RescoringMetric,
    conventions: Sequence[ConventionVector],
    residual: Residual | None = None,
) -> tuple[RescoringRow, ...]:
    """Return the exact declared sweep as canonical report rows."""
    return tuple(
        RescoringRow(
            cell.conventions,
            cell.family,
            cell.source,
            ReportRow(
                identity=ReportIdentity(
                    run_spec, grammar, cell.conventions, freeze_id=freeze_id
                ),
                method=cell.method,
                metric=cell.metric,
                result=cell.result,
                approximation=cell.approximation,
            ),
        )
        for cell in rescore_predictions(
            predictions,
            alignments=alignments,
            metric=metric,
            conventions=conventions,
            residual=residual,
        )
    )


def _matched(
    predictions: ReleasedPredictions, conventions: ConventionVector, *, split: Split
) -> tuple[Contingency, str]:
    """Return matched-constituent counts and their resampling unit.

    Parameters
    ----------
    predictions : ReleasedPredictions
        The released prediction set.
    conventions : ConventionVector
        The scoring conventions.
    split : Split
        The split to accumulate.

    Returns
    -------
    tuple[Contingency, str]
        The contingency table and resampling unit.
    """
    metric = RecallHomogeneity(conventions, split=split)

    for entry in predictions.on(split):
        metric.update(entry.pred, entry.gold, key=entry.item_id)

    return metric.contingency(), metric.rows()["homogeneity"].resampling.unit


def _prediction_item(value: PredictionItemDocument) -> PredictionItem:
    """Return one released sentence.

    Parameters
    ----------
    value : PredictionItemDocument
        The decoded item value.

    Returns
    -------
    PredictionItem
        The parsed item.

    Raises
    ------
    ValueError
        If an item name or bracketing value is invalid.
    """
    return PredictionItem(
        ItemId(ensure_name("a released item's id", value.item_id)),
        ensure_name("a released item's split", value.split),
        None
        if value.spans is None
        else InducedBracketing(
            value.length,
            {
                (entry.start, entry.end): ensure_name("an induced label", entry.label)
                for entry in value.spans
            },
        ),
        GoldBracketing(
            value.length,
            [
                Constituent(
                    entry.start,
                    entry.end,
                    label=ensure_name("a gold constituent's label", entry.label),
                    depth=entry.depth,
                )
                for entry in value.gold
            ],
            punctuation=value.punctuation,
        ),
    )
