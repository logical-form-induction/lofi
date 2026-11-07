"""Evaluation protocols."""

from pathlib import PurePath, PureWindowsPath
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    NamedTuple,
    Protocol,
    override,
    runtime_checkable,
)

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.data import Categorical
from lofi.eval._held_out import HeldOutEvaluationContext, InferenceEvaluationContext

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from lofi._types import ItemId, ReadoutId
    from lofi.data import Derivation, EvalDataset, Example, GoldDerivation, Item
    from lofi.eval._alignment import Alignment, Contingency
    from lofi.eval._matched import InducedBracketing
    from lofi.eval._values import ConventionVector, MetricResult
    from lofi.lf import InferenceLabel
    from lofi.nn.loss._values import ObjectiveOutputs

__all__ = [
    "HYPOTHESIS_ONLY",
    "SZS_STATUSES",
    "TIE_BREAKS",
    "AlignmentPolicy",
    "AlignmentScore",
    "AlignmentSplits",
    "ArtifactEvaluator",
    "BaselineArm",
    "BaselineBuildContext",
    "BaselineMeasurement",
    "BaselineResult",
    "BaselineSystemOperation",
    "Budget",
    "CategorySideHeldOutEvaluator",
    "ControlRequired",
    "DerivationHeldOutEvaluator",
    "DevelopmentAlignedEvaluator",
    "EvaluationArtifact",
    "GrammaticalityHeldOutEvaluator",
    "GrammaticalityItems",
    "HeldOutData",
    "HeldOutEvaluationContext",
    "HeldOutEvaluator",
    "HeldOutItems",
    "InferenceEvaluationContext",
    "InferenceFeaturizer",
    "InferenceHeldOutEvaluator",
    "InferenceItems",
    "LabeledDerivationEvaluator",
    "Metric",
    "NormalFormMetric",
    "StratifiedMetric",
    "SzsStatus",
    "TieBreak",
]


type BaselineArm = Literal["re-scoring", "re-running"]


class BaselineMeasurement(NamedTuple):
    """One deterministic measurement emitted by a fitted baseline system."""

    family: str
    system: str
    metric: str
    arm: BaselineArm
    cites: str
    reason: str
    value: int | float | str | bool | None
    n: int
    unit: str


class BaselineBuildContext(NamedTuple):
    """Immutable resources supplied to one registered baseline system."""

    root: Path
    seed: int


class BaselineResult(FrozenValue):
    """Measurements and immutable content identities from one baseline fit."""

    __slots__ = ("_measurements", "_source_digests")

    _noun = "baseline result"

    def __init__(
        self,
        measurements: Sequence[BaselineMeasurement],
        source_digests: Mapping[str, str],
        /,
    ) -> None:
        held_measurements = tuple(measurements)
        if not held_measurements:
            msg = "a baseline result has no measurements"
            raise ValueError(msg)
        for position, measurement in enumerate(held_measurements):
            if not isinstance(measurement, BaselineMeasurement):
                msg = f"baseline measurement {position} is not one: {measurement!r}"
                raise TypeError(msg)
        held_digests: dict[str, str] = {}
        for source, digest in source_digests.items():
            name = ensure_name("a baseline source name", source)
            if (
                not isinstance(digest, str)
                or not digest.startswith("sha256:")
                or len(digest) != 71
                or any(character not in "0123456789abcdef" for character in digest[7:])
            ):
                msg = f"baseline source {name!r} has no SHA-256 digest: {digest!r}"
                raise ValueError(msg)
            held_digests[name] = digest
        if not held_digests:
            msg = "a baseline result has no source content identities"
            raise ValueError(msg)
        object.__setattr__(self, "_measurements", held_measurements)
        object.__setattr__(
            self,
            "_source_digests",
            MappingProxyType(dict(sorted(held_digests.items()))),
        )

    @property
    def measurements(self) -> tuple[BaselineMeasurement, ...]:
        """Return measurements in declaration order."""
        return self._measurements

    @property
    def source_digests(self) -> Mapping[str, str]:
        """Return content identities by declared source role."""
        return self._source_digests


class BaselineSystemOperation(Protocol):
    """Run one baseline system already bound to its declared settings."""

    def __call__(self, /) -> BaselineResult:
        """Return deterministic measurements and source identities."""
        ...


class EvaluationArtifact(FrozenValue):
    """One deterministic file emitted by a held-out evaluator."""

    __slots__ = ("_content", "_name")

    _noun = "evaluation artifact"

    def __init__(self, name: str, content: bytes, /) -> None:
        named = ensure_name("an evaluation-artifact name", name)

        if (
            PurePath(named).name != named
            or PureWindowsPath(named).name != named
            or named in {".", ".."}
        ):
            msg = f"an evaluation-artifact name is not one path segment: {named!r}"
            raise ValueError(msg)

        if not isinstance(content, bytes):
            msg = f"evaluation-artifact content is not bytes: {content!r}"
            raise TypeError(msg)

        if not content:
            msg = "evaluation-artifact content is empty"
            raise ValueError(msg)

        object.__setattr__(self, "_name", named)
        object.__setattr__(self, "_content", content)

    @property
    def name(self) -> str:
        """Return the artifact's filename."""
        return self._name

    @property
    def content(self) -> bytes:
        """Return the complete artifact bytes."""
        return self._content

    @override
    def __repr__(self) -> str:
        """Return the artifact name and byte count."""
        return f"EvaluationArtifact({self._name!r}, {len(self._content)} bytes)"


@runtime_checkable
class ArtifactEvaluator(Protocol):
    """A held-out evaluator that can emit deterministic run artifacts."""

    def artifacts(self) -> tuple[EvaluationArtifact, ...]:
        """Return artifacts after the evaluator has consumed its held-out pass."""
        ...


@runtime_checkable
class LabeledDerivationEvaluator[P](Protocol):
    """A held-out evaluator whose predictions can carry a labeled derivation."""

    def labeled_spans(self, prediction: P, /) -> InducedBracketing | None:
        """Return the spans of a prediction's derivation under their categories.

        Parameters
        ----------
        prediction : P
            One item's prediction, as the evaluator's ``predict`` returned it.

        Returns
        -------
        InducedBracketing | None
            Each span of the selected derivation under the category that
            concludes it, the highest node labeling a span a unary chain
            covers, or None when the prediction holds no derivation.
        """
        ...


HYPOTHESIS_ONLY: Final = "hypothesis-only"
"""The method name of the inference artifact control."""


type TieBreak = Literal["least-gold-label", "largest-gold-marginal"]
"""The fallback order for equally scoring assignments."""

TIE_BREAKS: Final[tuple[TieBreak, ...]] = ("least-gold-label", "largest-gold-marginal")
"""Tie-breaking orders in sweep order."""

type SzsStatus = Literal[
    "Theorem", "CounterSatisfiable", "Timeout", "ResourceOut", "GaveUp"
]
"""The SZS status of one equivalence check.

Theorem means the two forms were proved equivalent. CounterSatisfiable means a
countermodel separates them, which settles non-equivalence. The other three
are non-answers: Timeout means a decision ran out of wall clock, ResourceOut
that the prover exhausted another resource, and GaveUp that the prover
declined the question for another reason.
"""

SZS_STATUSES: Final[tuple[SzsStatus, ...]] = (
    "Theorem",
    "CounterSatisfiable",
    "Timeout",
    "ResourceOut",
    "GaveUp",
)
"""The statuses an equivalence check may return, answers first."""


@runtime_checkable
class Metric[Pred, Gold](Protocol):
    """An accumulator over predictions and gold analyses."""

    @property
    def name(self) -> str:
        """Return the metric's name, unique within the report it appears in."""

    @property
    def conventions(self) -> ConventionVector | None:
        """Return the scoring conventions this metric was constructed under, if any."""

    @property
    def order_invariant(self) -> bool:
        """Return whether the computed value is independent of the order of updates."""

    def reset(self) -> None:
        """Return the metric to the state it was constructed in."""

    def update(self, pred: Pred, gold: Gold, *, key: ItemId) -> None:
        """Accumulate one item's prediction against its gold analysis.

        Parameters
        ----------
        pred : Pred
            Predicted analysis.
        gold : Gold
            Gold analysis.
        key : ItemId
            Item identifier for the per-item result.
        """

    def compute(self) -> MetricResult:
        """Return the aggregate over everything accumulated so far.

        Returns
        -------
        MetricResult
            Aggregate metric result.
        """

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return results keyed by item identifier.

        Returns
        -------
        Mapping[ItemId, MetricResult]
            Per-item metric results.
        """


@runtime_checkable
class StratifiedMetric(Protocol):
    """A metric whose resamples must preserve observed stratum counts."""

    def strata(self) -> Mapping[ItemId, str]:
        """Return one stratum label per item eligible for resampling."""


@runtime_checkable
class NormalFormMetric(Protocol):
    """A metric whose report row identifies its own derivation filter."""

    @property
    def normal_form(self) -> str:
        """Return the normal form defining the metric's derivation set."""


@runtime_checkable
class ControlRequired(Protocol):
    """A metric that names the control required beside its reported value."""

    @property
    def required_control(self) -> str:
        """Return the method name a report must carry beside this metric."""


class HeldOutItems(Protocol):
    """A batch naming the identifier and the words of each of its items."""

    @property
    def words(self) -> tuple[tuple[str, ...], ...]:
        """Return the words of each item, in item order and then in word order."""

    @property
    def item_ids(self) -> Sequence[ItemId]:
        """Return the identifier of each item, in item order."""


class InferenceItems(HeldOutItems, Protocol):
    """A held-out batch of labeled inference pair occurrences."""

    @property
    def pair_ids(self) -> Sequence[ItemId]:
        """Return one stable occurrence identifier per inference pair."""

    @property
    def labels(self) -> Sequence[InferenceLabel]:
        """Return one decided gold label per inference pair."""

    @property
    def premises(self) -> Sequence[ReadoutId]:
        """Return the published premise-form address of each pair."""

    @property
    def hypotheses(self) -> Sequence[ReadoutId]:
        """Return the published hypothesis-form address of each pair."""


class GrammaticalityItems(HeldOutItems, Protocol):
    """A held-out batch carrying one binary judgment per item."""

    @property
    def lengths(self) -> Sequence[int]:
        """Return the number of words in each item."""

    @property
    def labels(self) -> Sequence[bool]:
        """Return whether each item is grammatical."""


class HeldOutData(Protocol):
    """The retained partitions of a trained run, in each shape an evaluator reads.

    Each method refuses, naming the evaluator, a run that holds out no data of
    the shape it returns.
    """

    def derivations(self, split: str, /) -> Sequence[Example[GoldDerivation[str]]]:
        """Return the retained corpus items of a partition with their derivations."""

    def side_derivations(self, split: str, /) -> Sequence[Example[GoldDerivation[str]]]:
        """Return the retained inference pair sides of a partition with theirs."""

    def pair_batches(self, split: str, /) -> Sequence[InferenceItems]:
        """Return the retained inference pair batches of a partition."""

    def judgment_batches(self, split: str, /) -> Sequence[GrammaticalityItems]:
        """Return the retained grammaticality-judgment batches of a partition."""


class HeldOutEvaluator[Pred, Gold](Protocol):
    """Scores what a trained parser publishes over held-out items.

    The evaluator reads predictions off a published forward pass and scores
    them with its ordered metrics under its declared conventions.
    """

    @property
    def conventions(self) -> ConventionVector:
        """Return the conventions every row is scored and reported under."""

    @property
    def metrics(self) -> tuple[Metric[Pred, Gold], ...]:
        """Return the metrics in report order."""

    def held_out(self, data: HeldOutData, split: str, /) -> EvalDataset[Gold]:
        """Return one retained partition annotated as this evaluator scores it.

        Parameters
        ----------
        data : HeldOutData
            The run's retained partitions.
        split : str
            The partition scored.
        """

    def predict[BatchT: HeldOutItems](
        self, outputs: ObjectiveOutputs[BatchT], /
    ) -> Mapping[ItemId, Pred]:
        """Return predictions keyed by their published identifiers."""


class AlignmentScore(Protocol):
    """Observed and random-floor results from one frozen test alignment."""

    @property
    def rows(self) -> Mapping[str, MetricResult]:
        """Return the alignment rows in report order."""


class AlignmentSplits(Protocol):
    """Exact retained partitions selected by an alignment declaration."""

    @property
    def fit_on(self) -> str:
        """Return the partition used only to fit correspondence."""

    @property
    def score_on(self) -> str:
        """Return the disjoint partition used only to score correspondence."""


@runtime_checkable
class DevelopmentAlignedEvaluator[Pred, Gold](HeldOutEvaluator[Pred, Gold], Protocol):
    """An evaluator that fits label correspondence before test scoring."""

    @property
    def alignment(self) -> AlignmentSplits:
        """Return settings carrying the exact fitting and scoring split names."""

    def fit_alignment(
        self,
        predictions: Mapping[ItemId, Pred],
        gold: Mapping[ItemId, Gold],
        /,
    ) -> object:
        """Fit and freeze the correspondence from development evidence."""

    def score_alignment(
        self,
        predictions: Mapping[ItemId, Pred],
        gold: Mapping[ItemId, Gold],
        /,
        *,
        seed: int,
    ) -> AlignmentScore:
        """Score the frozen correspondence and its floor on test evidence."""


class DerivationHeldOutEvaluator[Pred, Gold](HeldOutEvaluator[Pred, Gold], Protocol):
    """A held-out evaluator whose gold analyses are derivations."""

    def annotate(self, item: Item, derivation: Derivation[str], /) -> Gold:
        """Return the gold analysis a gold derivation gives an item.

        Parameters
        ----------
        item : Item
            The held-out item.
        derivation : Derivation[str]
            Its gold derivation, labeled with category text.
        """


class CategorySideHeldOutEvaluator[Pred, Gold](HeldOutEvaluator[Pred, Gold], Protocol):
    """An inference-run evaluator whose gold analyses are side derivations."""

    def annotate(self, item: Item, derivation: Derivation[str], /) -> Gold:
        """Return the category analyses one retained side derivation supplies."""


class InferenceHeldOutEvaluator[Pred](HeldOutEvaluator[Pred, Categorical], Protocol):
    """A held-out evaluator over labeled sentence-pair occurrences."""

    def dataset(self, batches: Sequence[InferenceItems], /) -> EvalDataset[Categorical]:
        """Return the evaluation dataset named by held-out pair batches."""


class GrammaticalityHeldOutEvaluator(HeldOutEvaluator[float, bool], Protocol):
    """A held-out evaluator over binary sentence judgments."""

    def dataset(self, batches: Sequence[GrammaticalityItems], /) -> EvalDataset[bool]:
        """Return the evaluation dataset named by labeled held-out batches."""


@runtime_checkable
class InferenceFeaturizer[Item](Protocol):
    """A featurization of one inference item as counted features.

    What the item type carries is what the head fitted over the featurization
    sees. A featurization withholds a field by taking an item that does not
    have it, so there is nothing downstream to mask and nothing for a mask to
    leak.
    """

    @property
    def name(self) -> str:
        """Return the featurization's name, carried by the arm fitted over it."""

    def __call__(self, item: Item, /) -> Mapping[str, int]:
        """Return the features the item presents, with their counts.

        Parameters
        ----------
        item : Item
            The item to featurize.

        Returns
        -------
        Mapping[str, int]
            Nonnegative counts keyed by feature name.
        """


@runtime_checkable
class AlignmentPolicy(Protocol):
    """A rule for matching two label inventories."""

    @property
    def name(self) -> str:
        """Return the policy's name, unique within the report it appears in."""

    @property
    def tie_break(self) -> TieBreak:
        """Return the total order used when two assignments score equally."""

    @property
    def injective(self) -> bool:
        """Return whether the fitted correspondence maps distinct labels distinctly."""

    def fit(self, contingency: Contingency, /) -> Alignment:
        """Return the correspondence these counts choose under this policy.

        Parameters
        ----------
        contingency : Contingency
            Label co-occurrence counts.

        Returns
        -------
        Alignment
            Fitted label correspondence.

        Raises
        ------
        SplitLeakError
            If the contingency uses a split on which fitting is prohibited.
        """


@runtime_checkable
class Budget(Protocol):
    """Limits for an otherwise unbounded computation."""

    @property
    def depth(self) -> int:
        """Return how many levels an enumeration was allowed to produce."""

    @property
    def beam(self) -> int:
        """Return how many candidates a search kept from one level to the next."""

    @property
    def mass(self) -> float | None:
        """Return the largest omitted mass an enumeration could stop on, if any."""
