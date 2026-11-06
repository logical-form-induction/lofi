"""The run declaration and its sink, probe and evaluator families."""

from enum import StrEnum
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    NamedTuple,
    Protocol,
    Self,
    cast,
    runtime_checkable,
)

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.deduce import (
    CKY,
    CONTIGUOUS_SPANS,
    DEFAULT_REFINEMENT_PAIRS,
    NATIVE_CHARTS,
    Boolean,
    CellPruner,
    ChartStore,
    Counting,
    Expectation,
    Forest,
    KBest,
    LogInside,
    ParsePolicy,
    ParseSettings,
    ParseStrategy,
    Reachability,
    Retain,
    Semiring,
    SpanIndexFactory,
    TopKPruner,
    Viterbi,
)
from lofi.registry import resolve_plugin
from lofi.spec._errors import DeclarationError, SemiringCapabilityError
from lofi.spec._plugins import component_groups
from lofi.spec._ports import ChartOutputConsumer, ChartOutputProvider
from lofi.spec._signature import SignatureAssignments
from lofi.spec.approximation import (
    ApproximationConsumer,
    ApproximationPlan,
    ApproximationSpec,
)
from lofi.spec.artifacts import ArtifactCacheSpec
from lofi.spec.components.base import ComponentSpec, RunRole, _in_role_order
from lofi.spec.components.counterfactuals import CounterfactualsSpec
from lofi.spec.components.data import (
    DataSection,
    TrainingAllocationSpec,
    TrainingBatchKind,
)
from lofi.spec.components.loss import LossSection
from lofi.spec.components.model import (
    ModelSection,
)
from lofi.spec.components.trainer import (
    OptimizerSpec,
    PairSamplerSpec,
    SchedulerSpec,
    TrainerSpec,
    TypeSamplerSpec,
)
from lofi.spec.policy import DevicePolicySpec

if TYPE_CHECKING:
    from collections import abc
    from collections.abc import Mapping

    from lofi.deduce import Derivations, Expected, Ranked

__all__ = [
    "COMPONENT_FAMILIES",
    "SEMIRING_CAPABILITIES",
    "BooleanSemiringSpec",
    "BootstrapIntervalSpec",
    "CellPrunerSpec",
    "ChartStoreSpec",
    "CkyStrategySpec",
    "ContiguousSpanIndexSpec",
    "CountingSemiringSpec",
    "EvaluationData",
    "EvaluatorSpec",
    "ExpectationSemiringSpec",
    "ForestSemiringSpec",
    "IntervalSettings",
    "KBestSemiringSpec",
    "LogInsideSemiringSpec",
    "NativeChartStoreSpec",
    "ParseConsumer",
    "ParsePlugin",
    "ParseSpec",
    "ParseStrategySpec",
    "ProbeSpec",
    "ReleaseCheckpoint",
    "ReleaseSpec",
    "RunSpec",
    "SemiringCapability",
    "SignatureAssignmentsSpec",
    "SinkSpec",
    "SourceComparisonPointSpec",
    "SpanIndexSpec",
    "SymbolicSemiringSpec",
    "TopKPrunerSpec",
    "ViterbiSemiringSpec",
    "parse_plugin",
]

type ReleaseCheckpoint = Literal["validation-best", "resume-latest"]
"""A checkpoint class a completed run may release."""


class ProbeSpec(ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares a probe."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry point group lists the plugins."""
        return "probe"


class SinkSpec(ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares a metric sink."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry point group lists the plugins."""
        return "sink"


class IntervalSettings(NamedTuple):
    """The bootstrap an evaluator's interval is drawn by."""

    #: How many resamples of the evaluator's resampling unit are drawn.
    replicates: int

    #: The coverage the interval aims at.
    level: float


class BootstrapIntervalSpec(dx.Model, extra="forbid"):
    """Declares the bootstrap interval an evaluator reports beside each value.

    Each value's interval resamples the units the evaluator's metric counts,
    such as sentences, and recomputes the metric over each resample.

    Parameters
    ----------
    replicates : int
        How many resamples are drawn.
    level : float
        The coverage the interval aims at.
    """

    replicates: int
    level: float

    @dx.validates("replicates")
    @classmethod
    def _replicates_are_positive(cls, replicates: int) -> int:
        """Return a positive replicate budget."""
        if replicates < 1:
            msg = f"a bootstrap draws at least one resample, not {replicates}"
            raise ValueError(msg)

        return replicates

    @dx.validates("level")
    @classmethod
    def _level_is_a_proportion(cls, level: float) -> float:
        """Return a coverage strictly inside the unit interval."""
        if not 0.0 < ensure_finite("an interval coverage", level) < 1.0:
            msg = f"an interval coverage is not a proportion: {level!r}"
            raise ValueError(msg)

        return level

    def settings(self) -> IntervalSettings:
        """Return the settings the evaluator's bootstrap is drawn by."""
        return IntervalSettings(self.replicates, self.level)


class EvaluationData(StrEnum):
    """Name the held-out data an evaluator scores."""

    DERIVATIONS = "derivations"
    """Retained items with their gold derivations."""

    PAIR_SIDE_DERIVATIONS = "pair-side-derivations"
    """The retained gold derivations of each side of the inference pairs."""

    INFERENCE_PAIRS = "inference-pairs"
    """Labeled sentence pairs scored against the data's label inventory."""


class EvaluatorSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declares an evaluator and the interval its values are reported with.

    Parameters
    ----------
    interval : BootstrapIntervalSpec
        The bootstrap interval every value the evaluator reports carries.
    """

    interval: BootstrapIntervalSpec

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry point group lists the plugins."""
        return "evaluator"

    def evaluation_data(self) -> EvaluationData:
        """Return the held-out data the evaluator scores.

        Returns
        -------
        EvaluationData
            Gold derivations unless a variant reads another shape of data.
        """
        return EvaluationData.DERIVATIONS

    def metrics(self, labels: tuple[str, ...] = (), /) -> tuple[str, ...]:
        """Return the names of the metrics the evaluator reports a row for.

        A variant that declares none reports no row an experiment can name.

        Parameters
        ----------
        labels : tuple[str, ...]
            The inference label inventory the run's data owns, read by an
            evaluator of inference pairs; empty for every other evaluator.

        Returns
        -------
        tuple[str, ...]
            The metric names, in report order.
        """
        del labels
        return ()

    def fit_split(self) -> str | None:
        """Return the retained split used during fitting, when one is declared."""
        return None

    def score_split(self) -> str | None:
        """Return the retained split used for held-out scoring, when declared."""
        return None


COMPONENT_FAMILIES: Final[tuple[str, ...]] = tuple(
    group.removeprefix("lofi.") for group in component_groups()
)
"""Installed families whose variants register through one plugin entry each."""


class ReleaseSpec(dx.Model, extra="forbid"):
    """Declares which fitted checkpoint represents a completed run.

    Parameters
    ----------
    checkpoint : ReleaseCheckpoint
        The validation-best or final resumable checkpoint restored by export,
        probing, and other post-training consumers.
    """

    checkpoint: ReleaseCheckpoint


class ParseStrategySpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """A symbolic chart traversal declaration."""


class CkyStrategySpec(ParseStrategySpec, extra="forbid"):
    """The exhaustive bottom-up CKY traversal."""

    kind: Literal["cky"] = "cky"


class SpanIndexSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """A symbolic region-index declaration."""


class ContiguousSpanIndexSpec(SpanIndexSpec, extra="forbid"):
    """The contiguous half-open span index."""

    kind: Literal["contiguous"] = "contiguous"


class ChartStoreSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """A symbolic chart-storage declaration."""


class NativeChartStoreSpec(ChartStoreSpec, extra="forbid"):
    """The native packed forest and inside table storage."""

    kind: Literal["native"] = "native"


type SemiringCapability = Literal["log-probability", "marginals"]
"""What a semiring's weights can serve beyond the semiring laws.

``log-probability`` weights are float log probabilities, greater being better,
which a beam ranks and A* bounds. ``marginals`` weights sum the probability of
every derivation, so posterior marginals and goal probability mass are defined.
"""

SEMIRING_CAPABILITIES: Final[Mapping[SemiringCapability, str]] = MappingProxyType(
    {
        "log-probability": "float log-probability weights",
        "marginals": "posterior marginals",
    }
)
"""Each semiring capability with the phrase a refusal names it by."""


@runtime_checkable
class _CapabilityDeclaration(Protocol):
    """A semiring declaration that states what its weights can serve."""

    @property
    def capabilities(self) -> frozenset[SemiringCapability]:
        """Return the capabilities charts under this semiring have."""
        ...


@runtime_checkable
class _RequirementDeclaration(Protocol):
    """A parse-component declaration that states what weights it reads."""

    @property
    def semiring_requirements(self) -> frozenset[SemiringCapability]:
        """Return the capabilities the component needs of the semiring."""
        ...


class CellPrunerSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """A per-cell chart-pruning declaration.

    A concrete pruner declares ``semiring_requirements``, the semiring
    capabilities the weights it ranks must have.
    """

    prune_lexical: bool = False


class TopKPrunerSpec(CellPrunerSpec, extra="forbid"):
    """Keep a declared number of the heaviest cell items."""

    kind: Literal["top-k"] = "top-k"
    width: int

    __axioms__ = (
        dx.axiom("width > 0", message="a cell-pruner width must be positive"),
    )

    @property
    def semiring_requirements(self) -> frozenset[SemiringCapability]:
        """Return that the beam ranks float log probability weights."""
        return frozenset({"log-probability"})


class SymbolicSemiringSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """A symbolic chart semiring declaration.

    A concrete semiring declares ``capabilities``, what its weights can serve
    beyond the semiring laws, which consumers of the parse are checked
    against.
    """


class LogInsideSemiringSpec(SymbolicSemiringSpec, extra="forbid"):
    """Log space sum-product inside weights."""

    kind: Literal["log-inside"] = "log-inside"

    @property
    def capabilities(self) -> frozenset[SemiringCapability]:
        """Return that summed log probabilities support marginals."""
        return frozenset({"log-probability", "marginals"})


class ViterbiSemiringSpec(SymbolicSemiringSpec, extra="forbid"):
    """Maximum-score derivation weights."""

    kind: Literal["viterbi"] = "viterbi"

    @property
    def capabilities(self) -> frozenset[SemiringCapability]:
        """Return that best-derivation log probabilities have no marginals."""
        return frozenset({"log-probability"})


class CountingSemiringSpec(SymbolicSemiringSpec, extra="forbid"):
    """Exact derivation counts."""

    kind: Literal["counting"] = "counting"

    @property
    def capabilities(self) -> frozenset[SemiringCapability]:
        """Return that integer counts are not probabilities."""
        return frozenset()


class BooleanSemiringSpec(SymbolicSemiringSpec, extra="forbid"):
    """Derivability of each item."""

    kind: Literal["boolean"] = "boolean"

    @property
    def capabilities(self) -> frozenset[SemiringCapability]:
        """Return that truth values are not probabilities."""
        return frozenset()


class KBestSemiringSpec(SymbolicSemiringSpec, extra="forbid"):
    """The declared number of highest derivation log probabilities."""

    kind: Literal["k-best"] = "k-best"
    k: int

    __axioms__ = (dx.axiom("k > 0", message="a k-best semiring keeps k > 0"),)

    @property
    def capabilities(self) -> frozenset[SemiringCapability]:
        """Return that a ranked tuple is not one float log probability."""
        return frozenset()


class ExpectationSemiringSpec(SymbolicSemiringSpec, extra="forbid"):
    """A probability paired with expectations over declared coordinates."""

    kind: Literal["expectation"] = "expectation"
    dimensions: int

    __axioms__ = (
        dx.axiom(
            "dimensions > 0",
            message="an expectation semiring accumulates dimensions > 0",
        ),
    )

    @property
    def capabilities(self) -> frozenset[SemiringCapability]:
        """Return that a probability-expectation pair is not a log weight."""
        return frozenset()


class ForestSemiringSpec(SymbolicSemiringSpec, extra="forbid"):
    """Every derivation as an edge sequence."""

    kind: Literal["forest"] = "forest"

    @property
    def capabilities(self) -> frozenset[SemiringCapability]:
        """Return that derivation sets are not probabilities."""
        return frozenset()


class ParsePlugin[Declaration, Runtime](FrozenValue):
    """Pair one open parse-component declaration with its runtime builder."""

    __slots__ = ("_build", "_spec")
    _noun = "parse plugin"

    def __init__(
        self,
        *,
        spec: type[Declaration],
        build: abc.Callable[[Declaration], Runtime],
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_build", build)

    @property
    def spec(self) -> type[Declaration]:
        """Return the declaration class this plugin builds."""
        return self._spec

    def construct(self, declaration: Declaration, /) -> Runtime:
        """Build one runtime component from a checked declaration."""
        return self._build(declaration)


class _Kinded(Protocol):
    """Expose the discriminator every concrete parse declaration carries."""

    kind: str


def parse_plugin[Declaration, Runtime](
    family: str,
    declaration: Declaration,
    /,
) -> ParsePlugin[Declaration, Runtime]:
    """Return the installed plugin selected within one parse family."""
    plugin = resolve_plugin(
        family, cast("_Kinded", declaration).kind, ParsePlugin, type(declaration)
    )

    return cast("ParsePlugin[Declaration, Runtime]", plugin)


def _cky(declaration: CkyStrategySpec, /) -> ParseStrategy:
    """Build the exhaustive bottom-up traversal."""
    del declaration
    return CKY


def _contiguous(declaration: ContiguousSpanIndexSpec, /) -> SpanIndexFactory:
    """Build the contiguous span-index factory."""
    del declaration
    return CONTIGUOUS_SPANS


def _native_charts(declaration: NativeChartStoreSpec, /) -> ChartStore:
    """Build the native chart-store factory."""
    del declaration
    return NATIVE_CHARTS


def _top_k(declaration: TopKPrunerSpec, /) -> CellPruner[object, float]:
    """Build the declared cell beam."""
    return TopKPruner[object](declaration.width)


def _log_inside(declaration: LogInsideSemiringSpec, /) -> Semiring[float]:
    """Build log space sum-product weights."""
    del declaration
    return LogInside()


def _viterbi(declaration: ViterbiSemiringSpec, /) -> Semiring[float]:
    """Build maximum-score derivation weights."""
    del declaration
    return Viterbi()


def _counting(declaration: CountingSemiringSpec, /) -> Semiring[int]:
    """Build exact derivation counts."""
    del declaration
    return Counting()


def _boolean(declaration: BooleanSemiringSpec, /) -> Semiring[bool]:
    """Build item derivability."""
    del declaration
    return Boolean()


def _k_best(declaration: KBestSemiringSpec, /) -> Semiring[Ranked]:
    """Build the declared number of best derivation log probabilities."""
    return KBest(declaration.k)


def _expectation(declaration: ExpectationSemiringSpec, /) -> Semiring[Expected]:
    """Build probabilities paired with the declared expectation coordinates."""
    return Expectation(declaration.dimensions)


def _forest(declaration: ForestSemiringSpec, /) -> Semiring[Derivations[object]]:
    """Build every derivation as an edge sequence."""
    del declaration
    return Forest[object]()


CKY_STRATEGY_PLUGIN = ParsePlugin(spec=CkyStrategySpec, build=_cky)
CONTIGUOUS_SPAN_INDEX_PLUGIN = ParsePlugin(
    spec=ContiguousSpanIndexSpec,
    build=_contiguous,
)
NATIVE_CHART_STORE_PLUGIN = ParsePlugin(
    spec=NativeChartStoreSpec,
    build=_native_charts,
)
TOP_K_PRUNER_PLUGIN = ParsePlugin(spec=TopKPrunerSpec, build=_top_k)
LOG_INSIDE_SEMIRING_PLUGIN = ParsePlugin(
    spec=LogInsideSemiringSpec,
    build=_log_inside,
)
VITERBI_SEMIRING_PLUGIN = ParsePlugin(
    spec=ViterbiSemiringSpec,
    build=_viterbi,
)
COUNTING_SEMIRING_PLUGIN = ParsePlugin(spec=CountingSemiringSpec, build=_counting)
BOOLEAN_SEMIRING_PLUGIN = ParsePlugin(spec=BooleanSemiringSpec, build=_boolean)
K_BEST_SEMIRING_PLUGIN = ParsePlugin(spec=KBestSemiringSpec, build=_k_best)
EXPECTATION_SEMIRING_PLUGIN = ParsePlugin(
    spec=ExpectationSemiringSpec,
    build=_expectation,
)
FOREST_SEMIRING_PLUGIN = ParsePlugin(spec=ForestSemiringSpec, build=_forest)

_DEFAULT_PARSE_STRATEGY: Final = CkyStrategySpec()
_DEFAULT_SPAN_INDEX: Final = ContiguousSpanIndexSpec()
_DEFAULT_CHART_STORE: Final = NativeChartStoreSpec()
_DEFAULT_SEMIRING: Final = LogInsideSemiringSpec()


@runtime_checkable
class ParseConsumer(Protocol):
    """A declaration whose runtime decodes with the run's symbolic parse."""

    def parse_refusal(self, parse: ParseSpec, /) -> DeclarationError | None:
        """Return why the declared parse cannot serve it, or None."""
        ...


class ParseSpec(dx.Model, extra="forbid"):
    """Declares the symbolic chart algorithm of one run's decoding parses.

    Every component declared here reaches the parse each held-out evaluator
    decodes with, the supertag-factored A* search of the decode-backed
    evaluators included. The retention and reachability policy also reaches
    the symbolic parses a dataset runs while it is lowered. The
    Bisk--Hockenmaier baseline, which runs outside any run, declares its own
    parse with this schema. Two parses keep a fixed semiring because the
    semiring is part of what they compute: synthetic corpus sampling draws
    derivations from log space inside weights, and the counterfactual probe
    reads reachability from Boolean charts.

    Any registered semiring may be declared. Each declares the capabilities
    its weights have, and a consumer that needs one the semiring lacks is
    refused with a ``SemiringCapabilityError``: the top-k pruner and A*
    need float log probability weights, and a minimum Bayes risk decode or a
    pruned-mass measurement needs posterior marginals.

    Parameters
    ----------
    retain : Literal["forest", "inside"]
        Whether symbolic parses retain packed steps or only inside weights.
    reachability : Literal["all", "roots"]
        Whether symbolic parses construct every item or only root-reaching
        items.
    strategy : ParseStrategySpec
        Traversal strategy that runs the chart kernel.
    span_index : SpanIndexSpec
        Region index the chart is addressed by.
    chart_store : ChartStoreSpec
        Storage for the retained forest or inside table.
    pruner : CellPrunerSpec | None
        Per-cell beam applied to a forest, or None for exhaustive cells.
        Every pruned forest records, per cell, the semiring sum of the
        weights its pruner removed.
    semiring : SymbolicSemiringSpec
        Weight algebra the chart's inside weights accumulate in.
    measure_pruned_mass : bool
        Whether each pruned parse is repeated without its pruner, so the
        pruned forest reports the exact goal probability mass the pruner
        removed. It needs a pruner and a semiring with posterior marginals.
    refinement_pairs : int
        The most binding-and-candidate pairs root reachability enumerates
        when it narrows one rule row's demands on its lexical children. A
        row past the bound keeps its wider demands, so a lower bound plans
        faster and prunes less without losing a root-reaching item; zero
        never refines. Only ``reachability: roots`` plans, so another value
        is refused under ``reachability: all``.
    """

    retain: Literal["forest", "inside"]
    reachability: Literal["all", "roots"]
    strategy: ParseStrategySpec = _DEFAULT_PARSE_STRATEGY
    span_index: SpanIndexSpec = _DEFAULT_SPAN_INDEX
    chart_store: ChartStoreSpec = _DEFAULT_CHART_STORE
    pruner: CellPrunerSpec | None = None
    semiring: SymbolicSemiringSpec = _DEFAULT_SEMIRING
    measure_pruned_mass: bool = False
    refinement_pairs: int = DEFAULT_REFINEMENT_PAIRS

    @dx.validates("refinement_pairs")
    @classmethod
    def _refinement_bound_is_non_negative(cls, value: int) -> int:
        """Return a nonnegative refinement bound."""
        return ensure_non_negative("refinement_pairs", value)

    @dx.model_validator()
    def _refinement_bound_has_a_plan(self) -> Self:
        """Refuse a changed refinement bound on a parse that plans nothing."""
        if (
            self.reachability == "all"
            and self.refinement_pairs != DEFAULT_REFINEMENT_PAIRS
        ):
            msg = (
                "refinement_pairs bounds the root-reachability plan, and the "
                "parse plans none; declare reachability: roots or remove it"
            )
            raise ValueError(msg)
        return self

    @dx.model_validator()
    def _pruner_has_items_to_keep(self) -> Self:
        """Refuse a cell pruner on a chart that retains only inside weights."""
        if self.retain == "inside" and self.pruner is not None:
            msg = (
                "an inside chart keeps no items, so it cannot be pruned; "
                "declare retain: forest or remove the pruner"
            )
            raise ValueError(msg)
        return self

    @dx.model_validator()
    def _semiring_serves_the_pruner(self) -> Self:
        """Refuse a pruner whose ranking the declared semiring cannot serve."""
        pruner = self.pruner

        if pruner is None:
            return self

        if not isinstance(pruner, _RequirementDeclaration):
            msg = (
                f"the {pruner.kind!r} pruner does not declare which semiring "
                "capabilities it needs"
            )
            raise TypeError(msg)

        for capability in sorted(pruner.semiring_requirements):
            refusal = self.semiring_refusal(
                capability, consumer=f"the {pruner.kind} pruner"
            )
            if refusal is not None:
                raise refusal

        return self

    @dx.model_validator()
    def _pruned_mass_has_a_pruner_and_marginals(self) -> Self:
        """Refuse a mass measurement with no pruner or no summed probabilities."""
        if not self.measure_pruned_mass:
            return self

        if self.pruner is None:
            msg = (
                "measure_pruned_mass measures the mass a pruner lost, and the "
                "parse declares no pruner"
            )
            raise ValueError(msg)

        refusal = self.semiring_refusal("marginals", consumer="measuring pruned mass")

        if refusal is not None:
            raise refusal

        return self

    def semiring_refusal(
        self,
        capability: SemiringCapability,
        /,
        *,
        consumer: str,
    ) -> SemiringCapabilityError | None:
        """Return why the declared semiring cannot serve one consumer, or None.

        Parameters
        ----------
        capability : SemiringCapability
            The capability the consumer needs.
        consumer : str
            The consumer, as the subject of the refusal.

        Returns
        -------
        SemiringCapabilityError | None
            The refusal, or None when the semiring declares the capability.

        Raises
        ------
        TypeError
            If the semiring declares no capabilities at all.
        """
        semiring = self.semiring

        if not isinstance(semiring, _CapabilityDeclaration):
            msg = (
                f"the semiring {cast('_Kinded', semiring).kind!r} does not "
                "declare what its weights can serve"
            )
            raise TypeError(msg)

        if capability in semiring.capabilities:
            return None

        return SemiringCapabilityError(
            consumer=consumer,
            capability=capability,
            description=SEMIRING_CAPABILITIES[capability],
            semiring=cast("_Kinded", semiring).kind,
        )

    def policy(self) -> ParsePolicy[Retain]:
        """Return the declared retention and reachability without plugins."""
        return ParsePolicy(
            Retain(self.retain),
            Reachability(self.reachability),
            refinement_pairs=self.refinement_pairs,
        )

    def build(self) -> ParseSettings[object]:
        """Lower every declared parse component into one immutable setup."""
        strategy = cast(
            "ParseStrategy",
            parse_plugin("parse_strategy", self.strategy).construct(self.strategy),
        )
        span_index = cast(
            "SpanIndexFactory",
            parse_plugin("span_index", self.span_index).construct(self.span_index),
        )
        chart_store = cast(
            "ChartStore",
            parse_plugin("chart_store", self.chart_store).construct(self.chart_store),
        )
        semiring = cast(
            "Semiring[object]",
            parse_plugin("semiring", self.semiring).construct(self.semiring),
        )
        pruner = (
            None
            if self.pruner is None
            else cast(
                "CellPruner[object, object]",
                parse_plugin("cell_pruner", self.pruner).construct(self.pruner),
            )
        )

        return ParseSettings(
            self.policy(),
            strategy,
            span_index,
            chart_store,
            pruner,
            semiring,
            prune_lexical=(False if self.pruner is None else self.pruner.prune_lexical),
            measure_pruned_mass=self.measure_pruned_mass,
        )


class SourceComparisonPointSpec(dx.Model, extra="forbid"):
    """Identify one fitted arm in a source-model comparison.

    The candidate and its data size remain owned by their ordinary run
    declarations. This record names only experimental roles that have no
    independent computational owner.

    Parameters
    ----------
    source : str
        Identity of the generating source whose recovery this point tests.
    condition : str
        Comparison condition, such as the primary or a validity-control arm.
    direction : str
        Direction class used to classify a winning grammar.
    twin : str
        Observed assignment or scrambled-signature twin identity.
    """

    source: str
    condition: str
    direction: str
    twin: str

    @dx.validates("source", "condition", "direction", "twin")
    @classmethod
    def _field_is_named(cls, value: str) -> str:
        """Return one nonempty comparison identity."""
        return ensure_name("a source-comparison point field", value)


class SignatureAssignmentsSpec(dx.Model, extra="forbid"):
    """Record the names assigned to the learner signature's fixed slots.

    The three sequences are observations of the compiled learner declaration.
    They do not declare model capacity; capacity is measured from the built
    modules and carried by the point manifest.

    Parameters
    ----------
    primitives : tuple[str, ...]
        Primitive names in primitive-slot order.
    constructors : tuple[str, ...]
        Constructor names in constructor-slot order.
    combinators : tuple[str, ...]
        Grammar-rule names in controller-column order.
    """

    primitives: tuple[str, ...]
    constructors: tuple[str, ...]
    combinators: tuple[str, ...]

    @dx.validates("primitives", "constructors", "combinators")
    @classmethod
    def _inventory_is_named_and_unique(
        cls, inventory: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Return one ordered inventory of distinct nonempty names."""
        checked = tuple(
            ensure_name("a signature-assignment name", name) for name in inventory
        )

        if len(set(checked)) != len(checked):
            msg = f"a signature-assignment inventory repeats a name: {checked!r}"
            raise ValueError(msg)

        return checked

    def build(self) -> SignatureAssignments:
        """Return the immutable assignment value carried below the boundary."""
        return SignatureAssignments(
            self.primitives, self.constructors, self.combinators
        )


def _approximation_plan(
    approximation: ApproximationSpec | None,
    roles: Mapping[str, ComponentSpec],
    /,
) -> ApproximationPlan:
    """Resolve one already enumerated component table against its mass budget."""
    requests = {
        path: request
        for path, component in roles.items()
        if isinstance(component, ApproximationConsumer)
        if (request := component.approximation_request()) is not None
    }
    return ApproximationPlan(approximation, requests)


def _declares(roles: Mapping[str, ComponentSpec], role: RunRole, /) -> bool:
    """Return whether one enumerated component table contains a role."""
    prefix = f"{role.value}."
    return any(path == role.value or path.startswith(prefix) for path in roles)


class RunSpec(dx.Model, extra="forbid"):
    """Declares a run configuration.

    Parameters
    ----------
    model : ModelSection
        Model components.
    loss : LossSection | None
        Optional training objective.
    data : DataSection
        Dataset and featurizer.
    trainer : TrainerSpec
        Trainer component.
    release : ReleaseSpec
        Checkpoint class restored by post-training consumers.
    parse : ParseSpec
        Symbolic chart retention and reachability policy.
    approximation : ApproximationSpec | None
        Run level residual mass budget inherited by truncating components.
    type_sampler : TypeSamplerSpec | None
        Optional finite type sampler.
    pair_sampler : PairSamplerSpec | None
        Optional labeled rule instance sampler.
    optimizer : OptimizerSpec
        Optimizer component.
    device_policy : DevicePolicySpec
        Device and precision settings.
    scheduler : SchedulerSpec | None
        Optional learning rate scheduler.
    sinks : dict[str, SinkSpec]
        Metric sinks keyed by role.
    probes : dict[str, ProbeSpec]
        Probes keyed by role.
    counterfactuals : CounterfactualsSpec | None
        The counterfactual population the held-out passes publish to the
        probes, generated under the run's grammar, or None for none.
    evaluators : dict[str, EvaluatorSpec]
        Evaluators keyed by role.
    comparison : SourceComparisonPointSpec | None
        Optional experimental role metadata for source-model reduction.
    signature_assignment : SignatureAssignmentsSpec | None
        Observed names assigned to the compiled learner's fixed signature slots.
    artifacts : ArtifactCacheSpec | None
        The cache every pinned pretrained release a model component loads is
        fetched into, required exactly when a component loads pretrained weights.
    """

    model: ModelSection
    loss: LossSection | None = None
    data: DataSection
    trainer: TrainerSpec
    release: ReleaseSpec
    parse: ParseSpec
    approximation: ApproximationSpec | None = None
    type_sampler: TypeSamplerSpec | None = None
    pair_sampler: PairSamplerSpec | None = None
    optimizer: OptimizerSpec
    device_policy: DevicePolicySpec
    scheduler: SchedulerSpec | None = None
    sinks: dict[str, SinkSpec] = dx.field(default_factory=dict)
    probes: dict[str, ProbeSpec] = dx.field(default_factory=dict)
    counterfactuals: CounterfactualsSpec | None = None
    evaluators: dict[str, EvaluatorSpec] = dx.field(default_factory=dict)
    comparison: SourceComparisonPointSpec | None = None
    signature_assignment: SignatureAssignmentsSpec | None = None
    artifacts: ArtifactCacheSpec | None = None

    def approximation_plan(self) -> ApproximationPlan:
        """Resolve truncating components against the run's residual budget."""
        return _approximation_plan(self.approximation, self.component_roles())

    @classmethod
    def observation_sections(cls) -> frozenset[str]:
        """Return the sections that say where the run's metrics go or inputs wait.

        They change what observes a run, or where its pinned releases are
        cached, and nothing it computes, so a point's identity is computed over
        every other section.
        """
        return frozenset({"sinks", "artifacts"})

    def component_roles(self) -> Mapping[str, ComponentSpec]:
        """Return every component declaration keyed by its structural path.

        A single component is keyed by its role, a keyed collection's members
        by the role and the member's key, and the loss's objectives by
        ``loss.terms.<position>``.

        Returns
        -------
        Mapping[str, ComponentSpec]
            The declarations, in lexical order of their paths.
        """
        model = self.model
        data = self.data
        single: dict[RunRole, ComponentSpec | None] = {
            RunRole.PAIR_SAMPLER: self.pair_sampler,
            RunRole.TYPE_SAMPLER: self.type_sampler,
            RunRole.CAPACITY: model.capacity,
            RunRole.PARSER: model.parser,
            RunRole.TOKENS_ENCODER: model.token_encoder,
            RunRole.SUBWORD_POOLER: model.subword_pooler,
            RunRole.CONTROLLER: model.controller,
            RunRole.GROUNDING: model.grounding,
            RunRole.TRAINER: self.trainer,
            RunRole.OPTIMIZER: self.optimizer,
            RunRole.SCHEDULER: self.scheduler,
        }
        keyed: dict[RunRole, Mapping[str, ComponentSpec]] = {
            RunRole.TERM_ENCODERS: model.encoders,
            RunRole.TERM_DECODERS: model.decoders,
            RunRole.READOUTS: model.readouts,
            RunRole.SPAN_DECODERS: model.span_decoders,
            RunRole.SINKS: self.sinks,
            RunRole.PROBES: self.probes,
            RunRole.EVALUATORS: self.evaluators,
        }
        roles: dict[str, ComponentSpec] = {
            role.value: declaration
            for role, declaration in single.items()
            if declaration is not None
        }

        for name, bundle in data.datasets.items():
            roles[f"data.datasets.{name}.dataset"] = bundle.dataset

            if bundle.featurizer is not None:
                roles[f"data.datasets.{name}.featurizer"] = bundle.featurizer

            if bundle.constraint_source is not None:
                roles[f"data.datasets.{name}.constraint_source"] = (
                    bundle.constraint_source
                )

        if model.grounding is not None:
            roles.update(
                {
                    f"{RunRole.GROUNDING.value}.{name}": declaration
                    for name, declaration in model.grounding.nested_components().items()
                }
            )

        if self.loss is not None:
            for position, term in enumerate(self.loss.terms):
                roles[f"{RunRole.LOSS.value}.terms.{position}"] = term.objective

        for name, space in model.spaces.items():
            relation = space.relation

            if relation is not None and relation.mapping is not None:
                roles[f"model.spaces.{name}.relation.mapping"] = relation.mapping

        for role, declarations in keyed.items():
            for name, declaration in declarations.items():
                roles[f"{role.value}.{name}"] = declaration

        return MappingProxyType(dict(sorted(roles.items())))

    def declares(self, role: RunRole, /) -> bool:
        """Return whether the run declares a component under a role.

        Parameters
        ----------
        role : RunRole
            The structural role.
        """
        return _declares(self.component_roles(), role)

    @dx.model_validator()
    def _pretrained_weights_have_a_cache(self) -> Self:
        """Refuse pretrained components without a cache, and a cache without any."""
        loaded = sorted(
            path
            for path, component in self.component_roles().items()
            if component.pretrained_weights() is not None
        )

        if loaded and self.artifacts is None:
            msg = (
                f"the components {loaded} load pretrained weights, so the run "
                "declares the artifacts cache they are fetched into"
            )
            raise ValueError(msg)

        if not loaded and self.artifacts is not None:
            msg = "the run declares an artifacts cache, but no component loads one"
            raise ValueError(msg)

        return self

    @dx.model_validator()
    def _trainer_composes(self) -> Self:
        """Refuse a run whose roles or dataset the trainer's declaration rejects.

        The trainer declares the roles it requires, the roles it does not
        consume and the kinds of training batch the dataset must yield, and
        refuses the loss terms it names that the run does not declare. Every
        readout a loss term depends on is a declared model readout, and every
        other model component a loss term names is one the model declares.
        """
        model = self.model
        loss = self.loss
        data = self.data
        roles = self.component_roles()
        _approximation_plan(self.approximation, roles)
        trainer = self.trainer
        kind = str(trainer.kind)
        missing = sorted(
            role.value
            for role in trainer.required_roles()
            if not _declares(roles, role)
        )

        if missing:
            msg = f"trainer: the {kind!r} trainer requires component roles {missing}"
            raise ValueError(msg)

        chart_consumers: tuple[tuple[str, ChartOutputConsumer], ...] = tuple(
            (f"model.readouts.{role}", readout)
            for role, readout in model.readouts.items()
            if isinstance(readout, ChartOutputConsumer)
        ) + tuple(
            (f"loss.terms.{term.key}", term.objective)
            for term in (() if loss is None else loss.terms)
            if isinstance(term.objective, ChartOutputConsumer)
        )
        parser = model.parser

        if chart_consumers:
            if not isinstance(parser, ChartOutputProvider):
                msg = (
                    "model.parser: the parser does not declare which chart outputs "
                    "it provides"
                )
                raise ValueError(msg)

            provided = parser.chart_outputs()

            for path, consumer in chart_consumers:
                required = consumer.required_chart_outputs()
                absent = sorted(required - provided)

                if absent:
                    msg = (
                        f"model.parser: the {str(parser.kind)!r} parser publishes "
                        f"chart outputs {sorted(provided)}, but {path} requires "
                        f"{absent}"
                    )
                    raise ValueError(msg)

        present = sorted(
            role.value for role in trainer.forbidden_roles() if _declares(roles, role)
        )

        if present:
            msg = (
                f"trainer: the {kind!r} trainer does not consume component roles "
                f"{present}"
            )
            raise ValueError(msg)

        requirements = trainer.dataset_requirements()
        missing_datasets = sorted(set(requirements) - set(data.datasets))

        if missing_datasets:
            msg = (
                f"trainer: the {kind!r} trainer names absent training datasets "
                f"{missing_datasets}; declared datasets are "
                f"{sorted(data.datasets)}"
            )
            raise ValueError(msg)

        for name, requirement in requirements.items():
            bundle = data.datasets[name]
            dataset = bundle.dataset
            yielded = dataset.training_batches()
            absent = sorted(model.fitted_batches(requirement) - yielded)

            if absent:
                msg = (
                    f"data.datasets.{name}.dataset: the {kind!r} trainer requires "
                    f"the training-batch capability {[str(one) for one in absent]}, "
                    f"which the {str(dataset.kind)!r} dataset does not declare; it "
                    f"yields {sorted(str(one) for one in yielded)}"
                )
                raise ValueError(msg)

            if name in trainer.featurized_datasets() and bundle.featurizer is None:
                msg = (
                    f"data.datasets.{name}.featurizer: the {kind!r} trainer "
                    "requires a featurizer for this training dataset"
                )
                raise ValueError(msg)

        yielded = frozenset(
            kind
            for name in requirements
            for kind in data.datasets[name].dataset.training_batches()
        )

        if loss is not None:
            for term in loss.terms:
                absent = sorted(term.objective.required_training_batches() - yielded)

                if absent:
                    msg = (
                        f"loss.terms.{term.key}: the objective requires the "
                        "training-batch capability "
                        f"{[str(one) for one in absent]}, which the trainer's "
                        "selected datasets do not declare"
                    )
                    raise ValueError(msg)

            for term in loss.terms:
                for dependency in term.objective.dependencies():
                    if dependency.role not in model.readouts:
                        msg = (
                            f"loss.terms.{term.key}: the objective names readout "
                            f"role {dependency.role!r}, which is not declared among "
                            f"{sorted(model.readouts)}"
                        )
                        raise ValueError(msg)

                term.objective.refuse_model(model, f"loss.terms.{term.key}")

        trainer.refuse_loss_terms(
            frozenset(() if loss is None else (term.key for term in loss.terms))
        )

        grounding = model.grounding
        signals = tuple(
            signal
            for term in (() if loss is None else loss.terms)
            if (signal := term.objective.grounding_signal()) is not None
        )

        if signals and grounding is None:
            msg = "loss: inference objectives require model.grounding"
            raise ValueError(msg)

        if (
            TrainingBatchKind.INFERENCE_PAIRS in yielded
            and TrainingBatchKind.INFERENCE_PAIRS
            not in frozenset(
                kind for required in requirements.values() for kind in required
            )
            and grounding is None
        ):
            msg = (
                "a selected dataset yields inference-pairs, but the trainer does not "
                "consume them and model.grounding is absent"
            )
            raise ValueError(msg)

        if grounding is not None:
            if not signals:
                msg = "model.grounding requires an inference objective"
                raise ValueError(msg)

            if TrainingBatchKind.INFERENCE_PAIRS not in yielded:
                msg = "model.grounding requires a selected inference-pair dataset"
                raise ValueError(msg)

            grounding.validate_readouts(model.readouts)
            unsupported = tuple(
                signal
                for signal in signals
                if signal not in grounding.supported_signals()
            )

            if unsupported:
                msg = (
                    "model.grounding supports "
                    f"{sorted(grounding.supported_signals())}, "
                    f"not inference signals {unsupported!r}"
                )
                raise ValueError(msg)

            rewarded = tuple(signal for signal in signals if signal == "reward")

            if bool(rewarded) != grounding.reward_declared():
                expected = (
                    "a declared reward spec"
                    if rewarded
                    else "no reward spec for likelihood training"
                )
                msg = f"model.grounding requires {expected}"
                raise ValueError(msg)

        return self

    @dx.model_validator()
    def _parse_serves_every_evaluator(self) -> Self:
        """Refuse a run parse that a declared evaluator cannot decode with."""
        for role, evaluator in self.evaluators.items():
            if not isinstance(evaluator, ParseConsumer):
                continue
            refusal = evaluator.parse_refusal(self.parse)
            if refusal is not None:
                path = f"evaluators.{role}"
                raise refusal.prefixed(path)

        return self

    @dx.model_validator()
    def _epochs_divide_into_steps(self) -> Self:
        """Refuse a dataset the trainer cannot divide into whole optimizer steps."""
        trainer = self.trainer
        data = self.data

        for name in trainer.dataset_requirements():
            dataset = data.datasets[name].dataset
            occurrences = dataset.observed_occurrences()

            if occurrences is not None:
                if isinstance(dataset, TrainingAllocationSpec):
                    occurrences = dataset.training_occurrences(occurrences)
                trainer.refuse_epoch(occurrences)

        return self

    @dx.validates("sinks", "probes", "evaluators")
    @classmethod
    def _order_by_name[S: ComponentSpec](cls, named: dict[str, S]) -> dict[str, S]:
        """Return components ordered by role name.

        Parameters
        ----------
        named : dict[str, S]
            Components keyed by role.
        """
        return _in_role_order(named)
