"""Experiment documents: a sweep with its compute, contraction and decisions."""

import enum
import itertools
import json
from importlib import import_module
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Protocol, Self, cast

import didactic.api as dx

from lofi._checks import (
    ensure_finite,
    ensure_name,
    ensure_non_negative,
    ensure_positive,
)
from lofi._types import JsonValue
from lofi.eval import SeedTarget, freeze_number
from lofi.report import report_columns, report_renderer, report_stem
from lofi.run._boundary.sweep import (
    RESOLUTION_IV,
    GeneratorSpec,
    Sweep,
    _as_point,
    _point_key,
    _StoredValue,
    validate_resolution,
)
from lofi.run._correction import correction
from lofi.run._seed import ExperimentSeed
from lofi.spec.reducers import ReducerSpec
from lofi.stats import Dependence

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import date

    from lofi.judgment import JudgmentFitSpec
    from lofi.report import ReportRenderer
    from lofi.run._boundary.sweep import AxisValue
    from lofi.stats import Correction

__all__ = [
    "UNPRICED_COMPUTE",
    "ActiveLearningExperimentConfig",
    "AmendmentFields",
    "AmendmentLedger",
    "AmendmentRecord",
    "ArrayLaunchSpec",
    "BaselineFitExperimentConfig",
    "BenjaminiHochbergSpec",
    "BenjaminiYekutieliSpec",
    "BonferroniSpec",
    "ComputeSpec",
    "CorpusScaleRung",
    "Direction",
    "EvaluationSpec",
    "ExperimentCompute",
    "ExperimentConfigSpec",
    "ExperimentContraction",
    "ExperimentDecision",
    "ExperimentDocument",
    "ExperimentSeedSpec",
    "FactorLevel",
    "FactorSpan",
    "FailurePolicySpec",
    "FallbackRung",
    "GeneratorSpec",
    "HolmSpec",
    "JudgmentFitExperimentConfig",
    "LanguageRung",
    "LaunchSpec",
    "LengthBatchPricing",
    "MetricPredicate",
    "PointCountPredicate",
    "PredicateSpec",
    "PricingSpec",
    "RemoteLaunchSpec",
    "ReportSpec",
    "ResolutionIVRung",
    "RowReference",
    "RunExperimentConfig",
    "RungSpec",
    "SeedBudgetSpec",
    "SeedRung",
    "SerialLaunchSpec",
    "SignatureTwinSettings",
    "SignatureTwinsSpec",
    "StackBootstrapPricing",
    "Statistic",
    "StopRuleSpec",
    "UnpreregisteredCellsRung",
    "UnpricedCompute",
    "decode_amendments",
    "decode_judgment_fit",
    "encode_amendments",
    "model_schema_uri",
]


def decode_judgment_fit(document: Mapping[str, JsonValue], /) -> JudgmentFitSpec:
    """Validate one resolved judgment-fit declaration at its run boundary."""
    schema = cast(
        "type[JudgmentFitSpec]",
        import_module("lofi.judgment").JudgmentFitSpec,
    )
    return schema.model_validate_json(json.dumps(dict(document)))


UNPRICED_COMPUTE: Final = "unpriced"
"""The ``compute`` an experiment document declares before it has been priced."""


class CorrectionSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare one installed multiple-comparison correction."""

    def procedure(self) -> Correction:
        """Resolve the correction selected by this declaration."""
        return correction(self)


class BonferroniSpec(CorrectionSpec):
    """Declare Bonferroni family-wise error control."""

    kind: Literal["bonferroni"] = "bonferroni"


class HolmSpec(CorrectionSpec):
    """Declare Holm step-down family-wise error control."""

    kind: Literal["holm"] = "holm"


class BenjaminiHochbergSpec(CorrectionSpec):
    """Declare Benjamini--Hochberg false-discovery-rate control."""

    kind: Literal["benjamini-hochberg"] = "benjamini-hochberg"


class BenjaminiYekutieliSpec(CorrectionSpec):
    """Declare Benjamini--Yekutieli false-discovery-rate control."""

    kind: Literal["benjamini-yekutieli"] = "benjamini-yekutieli"


class ExperimentConfigSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Name an installed experiment configuration and its document."""

    path: str

    @dx.validates("path")
    @classmethod
    def _path_is_named(cls, path: str) -> str:
        return ensure_name("a configuration document", path)


class RunExperimentConfig(ExperimentConfigSpec):
    """Declare a training-run configuration document."""

    kind: Literal["run"] = "run"


class JudgmentFitExperimentConfig(ExperimentConfigSpec):
    """Declare a judgment-model fit document."""

    kind: Literal["judgment-fit"] = "judgment-fit"


class BaselineFitExperimentConfig(ExperimentConfigSpec):
    """Declare a baseline-system fit document."""

    kind: Literal["baseline-fit"] = "baseline-fit"


class ActiveLearningExperimentConfig(ExperimentConfigSpec):
    """Declare an active-learning document, each sweep point one run of it."""

    kind: Literal["active-learning"] = "active-learning"


class LaunchSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare one installed sweep launcher."""


class SerialLaunchSpec(LaunchSpec):
    """Declare execution in the launching process."""

    kind: Literal["serial"] = "serial"


class ArrayLaunchSpec(LaunchSpec):
    """Declare scheduler-array execution."""

    kind: Literal["array"] = "array"
    chunk_size: int = 1
    throttle: int | None = None

    @dx.validates("chunk_size")
    @classmethod
    def _chunk_runs_a_point(cls, size: int) -> int:
        return ensure_positive("an array chunk size", size)

    @dx.validates("throttle")
    @classmethod
    def _throttle_runs_a_task(cls, throttle: int | None) -> int | None:
        return (
            None if throttle is None else ensure_positive("an array throttle", throttle)
        )


class RemoteLaunchSpec(LaunchSpec):
    """Declare one remote process per sweep point."""

    kind: Literal["remote"] = "remote"


def _named_texts(field: str, values: tuple[str, ...], /) -> tuple[str, ...]:
    """Return nonempty, nonblank, distinct texts."""
    checked = tuple(ensure_name(field, value) for value in values)

    if not checked:
        msg = f"{field} declares nothing"
        raise ValueError(msg)

    if any(not value.strip() for value in checked):
        msg = f"{field} has a blank entry"
        raise ValueError(msg)

    if len(checked) != len(set(checked)):
        msg = f"{field} repeats an entry: {checked}"
        raise ValueError(msg)

    return checked


class ComputeSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares what an experiment's points are expected to cost."""


class UnpricedCompute(ComputeSpec):
    """Declares an experiment that has not been priced yet.

    A freeze admits only a priced experiment.

    Attributes
    ----------
    kind : Literal["unpriced"]
        Variant tag.
    """

    kind: Literal["unpriced"] = "unpriced"


class ExperimentCompute(ComputeSpec):
    """Declares an experiment's accelerator-hour estimate and its cost model.

    Attributes
    ----------
    kind : Literal["estimate"]
        Variant tag.
    p50_hours : float
        The median accelerator-hour estimate.
    p90_hours : float
        The ninetieth-percentile estimate admission is checked against.
    cost_model : str
        The cost model that produced both estimates.
    """

    kind: Literal["estimate"] = "estimate"
    p50_hours: float
    p90_hours: float
    cost_model: str

    @dx.validates("p50_hours", "p90_hours")
    @classmethod
    def _hours_are_finite(cls, hours: float) -> float:
        """Return a finite, nonnegative hour count."""
        checked = ensure_finite("an accelerator-hour estimate", hours)

        if checked < 0:
            msg = f"an accelerator-hour estimate is negative: {checked}"
            raise ValueError(msg)

        return checked

    @dx.validates("cost_model")
    @classmethod
    def _cost_model_is_named(cls, cost_model: str) -> str:
        """Return a nonblank cost model name."""
        return _named_texts("a cost model", (cost_model,))[0]

    @dx.model_validator()
    def _p90_is_not_below_p50(self) -> Self:
        """Validate the order of the two estimates.

        Returns
        -------
        Self
            This estimate after validation.

        Raises
        ------
        ValueError
            If the P90 estimate is below the P50 estimate.
        """
        if self.p90_hours < self.p50_hours:
            msg = (
                f"the P90 estimate of {self.p90_hours} accelerator-hours is below "
                f"the P50 estimate of {self.p50_hours}"
            )
            raise ValueError(msg)

        return self


class PricingSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares the cost model an experiment's training steps are priced under.

    The cost model is the installed plugin of the ``lofi.pricing`` entry-point
    group the declaration's tag names.

    Attributes
    ----------
    resamples : int
        The bootstrap replicates the price's quantiles are read off.
    """

    resamples: int = 10_000

    @dx.validates("resamples")
    @classmethod
    def _resamples_are_positive(cls, resamples: int) -> int:
        """Return a positive count of bootstrap replicates."""
        return ensure_positive("a pricing bootstrap's replicate count", resamples)


class StackBootstrapPricing(PricingSpec):
    """Declares pricing every stack off a random sample of its own timed steps.

    Attributes
    ----------
    kind : Literal["stack-bootstrap"]
        Variant tag.
    resamples : int
        The bootstrap replicates the quantiles are read off.
    """

    kind: Literal["stack-bootstrap"] = "stack-bootstrap"


class LengthBatchPricing(PricingSpec):
    """Declares pricing a step by interpolation over sentence length and batch size.

    The reference run document's training step is timed at every sentence
    length and batch size of the grid, and each stack's step is priced at its
    own mean sentence length and batch size. The reference document declares
    the stack, its width, learning rate and grammar among them, and each cell
    composes it under the cell overrides with the cell's ``{length}`` and
    ``{batch}`` substituted.

    Attributes
    ----------
    kind : Literal["length-batch"]
        Variant tag.
    resamples : int
        The bootstrap replicates the quantiles are read off.
    lengths : tuple[int, ...]
        The sentence lengths the grid spans, in tokens, ascending.
    batches : tuple[int, ...]
        The batch sizes the grid spans, in sentences, ascending.
    reference : str
        The run document whose training step every cell times, a path or a
        ``lofi.resource:`` reference to a packaged document.
    cell_overrides : tuple[str, ...]
        The ``key=value`` overrides one cell composes the reference under,
        each naming ``{length}``, ``{batch}`` or neither.
    seed : int
        The experiment root seed the reference stack is planned under.
    """

    kind: Literal["length-batch"] = "length-batch"
    lengths: tuple[int, ...] = (10, 20, 30, 40)
    batches: tuple[int, ...] = (8, 16, 32)
    reference: str = "lofi.resource:run/pricing-reference.yaml"
    cell_overrides: tuple[str, ...] = (
        "data.datasets.train.dataset.length.min_length={length}",
        "data.datasets.train.dataset.length.max_length={length}",
        "trainer.batch_size={batch}",
    )
    seed: int = 0

    @dx.validates("lengths", "batches")
    @classmethod
    def _axis_ascends(cls, axis: tuple[int, ...]) -> tuple[int, ...]:
        """Return two or more ascending positive grid lines.

        Raises
        ------
        ValueError
            If the axis holds fewer than two lines, a line is not positive, or
            the lines do not strictly ascend.
        """
        if (
            len(axis) < 2
            or axis[0] <= 0
            or any(a >= b for a, b in itertools.pairwise(axis))
        ):
            msg = f"a pricing grid axis is two or more ascending positive lines: {axis}"
            raise ValueError(msg)

        return axis

    @dx.validates("reference")
    @classmethod
    def _reference_is_named(cls, reference: str) -> str:
        """Return a nonempty reference document."""
        return ensure_name("a pricing reference document", reference)

    @dx.validates("cell_overrides")
    @classmethod
    def _overrides_name_only_the_cell(
        cls, overrides: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Return overrides that substitute nothing but the cell's coordinates.

        Raises
        ------
        ValueError
            If an override names a placeholder other than ``{length}`` and
            ``{batch}``, or is not a ``key=value`` expression.
        """
        for override in overrides:
            try:
                _cell_override(override, length=1, batch=1)

            except (KeyError, IndexError, ValueError) as refused:
                msg = (
                    f"a pricing cell override substitutes only {{length}} and "
                    f"{{batch}}: {override!r}"
                )
                raise ValueError(msg) from refused

            if "=" not in override:
                msg = f"a pricing cell override is a key=value expression: {override!r}"
                raise ValueError(msg)

        return overrides

    def cells(self) -> tuple[tuple[int, int], ...]:
        """Return the grid's sentence length and batch size pairs, lengths outermost."""
        return tuple(
            (length, batch) for length in self.lengths for batch in self.batches
        )

    def overrides(self, length: int, batch: int, /) -> tuple[str, ...]:
        """Return the overrides one cell composes the reference document under."""
        return tuple(
            _cell_override(override, length=length, batch=batch)
            for override in self.cell_overrides
        )


def _cell_override(template: str, /, *, length: int, batch: int) -> str:
    """Return one pricing cell override with the cell's coordinates substituted.

    Raises
    ------
    KeyError
        If the template names a placeholder other than ``length`` and ``batch``.
    """
    return template.format_map({"length": length, "batch": batch})


class ExperimentSeedSpec(dx.Model, extra="forbid"):
    """Declares an experiment's root seed, from which every point seed is derived.

    Attributes
    ----------
    root : int
        The experiment's nonnegative root seed.
    """

    root: int

    @dx.validates("root")
    @classmethod
    def _root_counts_from_zero(cls, root: int) -> int:
        """Return a nonnegative root seed."""
        return ensure_non_negative("an experiment root seed", root)

    def seed(self) -> ExperimentSeed:
        """Return the experiment seed the declaration names."""
        return ExperimentSeed(self.root)


class RowReference(dx.Model, extra="forbid"):
    """Names one report row by the evaluator role that reports it and its metric.

    Attributes
    ----------
    evaluator : str
        The role the evaluator is declared under in the experiment's
        configuration document, which every row it reports names as its task.
    metric : str
        The name of a metric that evaluator reports.
    """

    evaluator: str
    metric: str

    @dx.validates("evaluator", "metric")
    @classmethod
    def _is_named(cls, value: str) -> str:
        """Return a nonblank role or metric name."""
        return _named_texts("a row reference", (value,))[0]

    @property
    def label(self) -> str:
        """Return the row as ``<evaluator>/<metric>``."""
        return f"{self.evaluator}/{self.metric}"


class FactorLevel(dx.Model, extra="forbid"):
    """Maps one preregistered level of a factor to the axis value that realizes it.

    Attributes
    ----------
    level : JsonValue
        The level as the preregistration's grid declares it.
    value : JsonValue
        The point of the sweep axis that realizes the level.
    """

    level: JsonValue
    value: JsonValue


class FactorSpan(dx.Model, extra="forbid"):
    """Binds one preregistered factor to the sweep axis that spans its levels.

    Attributes
    ----------
    axis : str
        The key of the grid or yoked axis that varies the factor.
    levels : tuple[FactorLevel, ...]
        The axis value of each preregistered level. Empty when every level is
        itself the axis value that realizes it.
    """

    axis: str
    levels: tuple[FactorLevel, ...] = ()

    @dx.validates("axis")
    @classmethod
    def _axis_is_named(cls, axis: str) -> str:
        """Return a nonblank axis key."""
        return _named_texts("a factor's axis", (axis,))[0]

    @dx.validates("levels")
    @classmethod
    def _levels_are_distinct(
        cls, levels: tuple[FactorLevel, ...]
    ) -> tuple[FactorLevel, ...]:
        """Return level mappings that map each level once."""
        keys = tuple(_point_key(_as_point(entry.level)) for entry in levels)

        if len(keys) != len(set(keys)):
            msg = f"a factor maps one level twice: {keys}"
            raise ValueError(msg)

        return levels


class Statistic(enum.StrEnum):
    """How the values of the rows a predicate reads are reduced to one number."""

    MEAN = "mean"
    MINIMUM = "min"
    MAXIMUM = "max"


class Direction(enum.StrEnum):
    """Which side of its threshold a reduced value must lie on."""

    BELOW = "below"
    ABOVE = "above"


class PredicateSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares a condition over an experiment's status and its ingested rows."""


class PointCountPredicate(PredicateSpec):
    """Holds once enough of the experiment's points are in the named statuses.

    Attributes
    ----------
    kind : Literal["points"]
        Variant tag.
    statuses : tuple[str, ...]
        The point statuses counted, as the experiment's status records them.
    at_least : int
        The fewest counted points at which the condition holds.
    """

    kind: Literal["points"] = "points"
    statuses: tuple[str, ...]
    at_least: int

    @dx.validates("statuses")
    @classmethod
    def _statuses_are_named(cls, statuses: tuple[str, ...]) -> tuple[str, ...]:
        """Return distinct status names."""
        return _named_texts("the counted statuses", statuses)

    @dx.validates("at_least")
    @classmethod
    def _count_is_positive(cls, at_least: int) -> int:
        """Return a positive point count."""
        return ensure_positive("a point count", at_least)


class MetricPredicate(PredicateSpec):
    """Holds once one row's reduced value lies on the declared side of a threshold.

    Attributes
    ----------
    kind : Literal["metric"]
        Variant tag.
    row : RowReference
        The row whose ingested values are reduced.
    statistic : Statistic
        How the values are reduced.
    direction : Direction
        The side of the threshold the reduced value lies on when the
        condition holds.
    threshold : float
        The threshold.
    minimum_rows : int
        The fewest ingested values the condition is read from; with fewer it
        does not hold.
    """

    kind: Literal["metric"] = "metric"
    row: RowReference
    statistic: Statistic
    direction: Direction
    threshold: float
    minimum_rows: int = 1

    @dx.validates("threshold")
    @classmethod
    def _threshold_is_finite(cls, threshold: float) -> float:
        """Return a finite threshold."""
        return ensure_finite("a predicate threshold", threshold)

    @dx.validates("minimum_rows")
    @classmethod
    def _rows_are_counted(cls, minimum_rows: int) -> int:
        """Return a positive row count."""
        return ensure_positive("a predicate's minimum row count", minimum_rows)


class StopRuleSpec(dx.Model, extra="forbid"):
    """Declares a condition under which an experiment halts early.

    Attributes
    ----------
    name : str
        The name the rule is reported under.
    when : PredicateSpec
        The condition that fires the rule.
    """

    name: str
    when: PredicateSpec

    @dx.validates("name")
    @classmethod
    def _name_is_named(cls, name: str) -> str:
        """Return a nonblank rule name."""
        return _named_texts("a stop rule name", (name,))[0]


class RungSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares one rung of an experiment's contraction ladder.

    The order of variants in :class:`ExperimentContraction.rungs` is the ladder.
    """


class UnpreregisteredCellsRung(RungSpec):
    """Declares dropping every axis value that realizes no preregistered level.

    Attributes
    ----------
    kind : Literal["unpreregistered-cells"]
        Variant tag.
    """

    kind: Literal["unpreregistered-cells"] = "unpreregistered-cells"


class ResolutionIVRung(RungSpec):
    """Declares reducing two-level grid axes to a resolution-IV fraction.

    Attributes
    ----------
    kind : Literal["resolution-iv"]
        Variant tag.
    generators : tuple[GeneratorSpec, ...]
        The generators the fraction is built from. Every word of the defining
        relation they span has four or more letters, so no main effect is
        aliased with a two-factor interaction.
    """

    kind: Literal["resolution-iv"] = "resolution-iv"

    generators: tuple[GeneratorSpec, ...]

    @dx.validates("generators")
    @classmethod
    def _generators_reach_resolution_iv(
        cls, generators: tuple[GeneratorSpec, ...]
    ) -> tuple[GeneratorSpec, ...]:
        """Return generators whose defining relation has resolution four or more.

        Raises
        ------
        ValueError
            If no generator is declared, two generators set one factor, a
            generated factor appears in a word, or a word of the defining
            relation has fewer than four letters.
        """
        return validate_resolution(generators, minimum=RESOLUTION_IV)


class LanguageRung(RungSpec):
    """Declares dropping languages from one axis in a declared order.

    Attributes
    ----------
    kind : Literal["languages"]
        Variant tag.
    axis : str
        The key of the axis whose values are languages.
    order : tuple[str, ...]
        The languages in the order they are dropped.
    """

    kind: Literal["languages"] = "languages"

    axis: str
    order: tuple[str, ...]

    @dx.validates("axis")
    @classmethod
    def _axis_is_named(cls, axis: str) -> str:
        """Return a nonblank axis key."""
        return _named_texts("the language axis", (axis,))[0]

    @dx.validates("order")
    @classmethod
    def _order_is_named(cls, order: tuple[str, ...]) -> tuple[str, ...]:
        """Return distinct languages."""
        return _named_texts("the language drop order", order)


class SeedRung(RungSpec):
    """Declares reducing every cell's seeds to a floor.

    Attributes
    ----------
    kind : Literal["seeds"]
        Variant tag.
    floor : int
        The fewest seeds a cell keeps.
    """

    kind: Literal["seeds"] = "seeds"

    floor: int

    @dx.validates("floor")
    @classmethod
    def _floor_is_positive(cls, floor: int) -> int:
        """Return a positive seed floor."""
        if floor < 1:
            msg = f"a seed floor of {floor} reports no seed"
            raise ValueError(msg)

        return floor


class CorpusScaleRung(RungSpec):
    """Declares dropping corpus scales from one axis in a declared order.

    Attributes
    ----------
    kind : Literal["corpus-scale"]
        Variant tag.
    axis : str
        The key of the axis whose values are corpus scales.
    order : tuple[JsonValue, ...]
        The axis values in the order they are dropped.
    """

    kind: Literal["corpus-scale"] = "corpus-scale"

    axis: str
    order: tuple[JsonValue, ...]

    @dx.validates("axis")
    @classmethod
    def _axis_is_named(cls, axis: str) -> str:
        """Return a nonblank axis key."""
        return _named_texts("the corpus-scale axis", (axis,))[0]

    @dx.validates("order")
    @classmethod
    def _order_is_distinct(
        cls, order: tuple[_StoredValue, ...]
    ) -> tuple[_StoredValue, ...]:
        """Return a nonempty order that names each value once."""
        keys = tuple(_point_key(_as_point(value)) for value in order)

        if not keys or len(keys) != len(set(keys)):
            msg = f"a corpus-scale drop order is empty or repeats a value: {keys}"
            raise ValueError(msg)

        return order


class FallbackRung(RungSpec):
    """Declares switching the experiment to its decision point's fallback claim.

    Attributes
    ----------
    kind : Literal["fallback"]
        Variant tag.
    """

    kind: Literal["fallback"] = "fallback"


class ExperimentContraction(dx.Model, extra="forbid"):
    """Declares an experiment's contraction ladder before launch.

    Attributes
    ----------
    rungs : tuple[RungSpec, ...]
        The rungs the experiment may contract through, in ladder order.
    factorial : str
        The factorial structure the ladder reduces.
    """

    rungs: tuple[RungSpec, ...]
    factorial: str

    @dx.validates("rungs")
    @classmethod
    def _rungs_follow_the_ladder(
        cls, rungs: tuple[RungSpec, ...]
    ) -> tuple[RungSpec, ...]:
        """Return a nonempty declared order of distinct rung kinds."""
        if not rungs:
            msg = "a contraction ladder declares no rung"
            raise ValueError(msg)

        kinds = tuple(cast("str", rung.kind) for rung in rungs)

        if len(kinds) != len(set(kinds)):
            msg = f"a contraction ladder repeats a rung kind: {kinds}"
            raise ValueError(msg)

        return rungs

    @dx.validates("factorial")
    @classmethod
    def _factorial_is_named(cls, factorial: str) -> str:
        """Return a nonblank factorial structure."""
        return _named_texts("a factorial structure", (factorial,))[0]


class SeedBudgetSpec(dx.Model, extra="forbid"):
    """Declares how the seed count of an experiment is priced from a pilot.

    Attributes
    ----------
    metric : str
        The row metric whose seed-to-seed spread the pilot measures.
    target : SeedTarget
        The quantity whose precision the seed count controls.
    precision : float
        The relative half-width the target is estimated to.
    level : float
        The confidence the half-width holds at.
    """

    metric: str
    target: SeedTarget
    precision: float
    level: float

    @dx.validates("metric")
    @classmethod
    def _metric_is_named(cls, metric: str) -> str:
        """Return a nonblank metric name."""
        return ensure_name("a budgeted metric", metric)

    @dx.validates("precision")
    @classmethod
    def _precision_is_positive(cls, precision: float) -> float:
        """Return a finite positive relative half-width."""
        if ensure_finite("a seed-budget precision", precision) <= 0.0:
            msg = f"a seed-budget precision is not positive: {precision!r}"
            raise ValueError(msg)

        return precision

    @dx.validates("level")
    @classmethod
    def _level_is_a_proportion(cls, level: float) -> float:
        """Return a confidence strictly inside the unit interval."""
        if not 0.0 < ensure_finite("a seed-budget level", level) < 1.0:
            msg = f"a seed-budget level is not a proportion: {level!r}"
            raise ValueError(msg)

        return level


class EvaluationSpec(dx.Model, extra="forbid"):
    """Declares the statistical settings an experiment's results are read under.

    Attributes
    ----------
    alpha : float
        The level each family of preregistered comparisons is controlled at.
    correction : CorrectionSpec
        The correction applied within each family.
    dependence : Dependence
        The dependence regime the tests of a family were run under.
    minimum_seeds : int
        The fewest seeds a reported cell may rest on, which bounds the
        contraction's seed floor from below.
    seed_budget : SeedBudgetSpec
        How the seed count is priced from a pilot.
    """

    alpha: float
    correction: CorrectionSpec
    dependence: Dependence
    minimum_seeds: int
    seed_budget: SeedBudgetSpec

    @dx.validates("alpha")
    @classmethod
    def _alpha_is_a_proportion(cls, alpha: float) -> float:
        """Return a level strictly inside the unit interval."""
        if not 0.0 < ensure_finite("a family-wise level", alpha) < 1.0:
            msg = f"a family-wise level is not a proportion: {alpha!r}"
            raise ValueError(msg)

        return alpha

    @dx.validates("minimum_seeds")
    @classmethod
    def _minimum_is_positive(cls, minimum: int) -> int:
        """Return a positive seed minimum."""
        if minimum < 1:
            msg = f"a reported cell rests on at least one seed, not {minimum}"
            raise ValueError(msg)

        return minimum

    @dx.model_validator()
    def _correction_holds_under_the_dependence(self) -> Self:
        """Validate the correction against the declared dependence regime."""
        procedure = self.correction.procedure()

        if self.dependence not in procedure.valid_under:
            msg = (
                f"the {procedure.name} correction is not valid under "
                f"{self.dependence.value} dependence"
            )
            raise ValueError(msg)

        return self


class ExperimentDecision(dx.Model, extra="forbid"):
    """Declares an experiment's decision point and the claim it falls back to.

    Attributes
    ----------
    date : str
        The ISO calendar date of the decision.
    signal : PredicateSpec
        The early signal read at that date.
    fallback : str
        The claim reported instead when the signal fails.
    """

    date: str
    signal: PredicateSpec
    fallback: str

    @dx.validates("date", "fallback")
    @classmethod
    def _is_named(cls, value: str) -> str:
        """Return nonblank decision text."""
        return _named_texts("a decision point field", (value,))[0]


class AmendmentRecord(dx.Model, extra="forbid"):
    """Records one contraction rung applied to an experiment's sweep.

    Attributes
    ----------
    rung : str
        The rung applied.
    dropped : tuple[JsonValue, ...]
        The axis values or seeds the rung removed, empty for a rung that
        removes no named value.
    previous_sweep_id : str
        The identifier of the sweep the rung was applied to.
    sweep_id : str
        The identifier of the sweep it produced.
    points : int
        The point count of the sweep it produced.
    date : str
        The ISO calendar date it was applied on.
    """

    rung: str
    dropped: tuple[JsonValue, ...]
    previous_sweep_id: str
    sweep_id: str
    points: int
    date: str

    @dx.validates("rung", "previous_sweep_id", "sweep_id", "date")
    @classmethod
    def _is_named(cls, value: str) -> str:
        """Return nonblank amendment text."""
        return _named_texts("an amendment field", (value,))[0]

    @dx.validates("points")
    @classmethod
    def _points_are_counted(cls, points: int) -> int:
        """Return a nonnegative point count."""
        return ensure_non_negative("an amended point count", points)


class AmendmentLedger(dx.Model, extra="forbid"):
    """Records every contraction applied to one experiment, in application order.

    Attributes
    ----------
    amendments : tuple[AmendmentRecord, ...]
        The applied rungs, each applied to the sweep the previous produced.
    """

    amendments: tuple[AmendmentRecord, ...]

    @dx.validates("amendments")
    @classmethod
    def _amendments_chain(
        cls, amendments: tuple[AmendmentRecord, ...]
    ) -> tuple[AmendmentRecord, ...]:
        """Return amendments that each start from the sweep the previous produced.

        Raises
        ------
        ValueError
            If an amendment starts from a sweep the one before it did not produce.
        """
        for earlier, later in itertools.pairwise(amendments):
            if later.previous_sweep_id != earlier.sweep_id:
                msg = (
                    f"an amendment starts from the sweep {later.previous_sweep_id!r}, "
                    f"not the sweep {earlier.sweep_id!r} the one before it produced"
                )
                raise ValueError(msg)

        return amendments


class AmendmentFields(Protocol):
    """What one applied amendment states, as the ledger records it."""

    @property
    def rung(self) -> str:
        """Return the rung applied."""
        ...

    @property
    def dropped(self) -> tuple[AxisValue, ...]:
        """Return the axis values or seeds the rung removed."""
        ...

    @property
    def previous_sweep_id(self) -> str:
        """Return the identifier of the sweep the rung was applied to."""
        ...

    @property
    def sweep_id(self) -> str:
        """Return the identifier of the sweep the rung produced."""
        ...

    @property
    def points(self) -> int:
        """Return the point count of the sweep the rung produced."""
        ...

    @property
    def amended_on(self) -> date:
        """Return the day the rung was applied."""
        ...


def encode_amendments(amendments: Sequence[AmendmentFields], /) -> bytes:
    """Return the ledger recording amendments, in application order, as JSON.

    Parameters
    ----------
    amendments : Sequence[AmendmentFields]
        The applied amendments.

    Raises
    ------
    ValueError
        If the amendments do not chain.
    """
    try:
        ledger = AmendmentLedger(
            amendments=tuple(
                AmendmentRecord(
                    rung=amendment.rung,
                    dropped=amendment.dropped,
                    previous_sweep_id=amendment.previous_sweep_id,
                    sweep_id=amendment.sweep_id,
                    points=amendment.points,
                    date=amendment.amended_on.isoformat(),
                )
                for amendment in amendments
            )
        )

    except dx.ValidationError as refused:
        msg = f"the amendments do not form a ledger: {refused}"
        raise ValueError(msg) from refused

    return f"{ledger.model_dump_json(indent=2)}\n".encode()


def decode_amendments(data: bytes, /) -> AmendmentLedger:
    """Return the amendment ledger a document's bytes declare.

    Parameters
    ----------
    data : bytes
        The ledger's JSON.

    Raises
    ------
    ValueError
        If the bytes are not a well-formed, chained ledger.
    """
    try:
        return AmendmentLedger.model_validate_json(data)

    except dx.ValidationError as refused:
        msg = f"an amendment ledger is refused: {refused}"
        raise ValueError(msg) from refused


class ReportSpec(dx.Model, extra="forbid"):
    """Declares one table rendered by ``lofi results``.

    Attributes
    ----------
    name : str
        Output filename stem.
    rows : tuple[str, ...]
        Labels of experiment rows included, in table order.
    columns : tuple[str, ...]
        Stored result columns included, in table order, or columns
        ``lofi.report.DERIVED_COLUMNS`` derives from them.
    labels : tuple[str, ...]
        Headings corresponding to ``columns``.
    precision : int
        Decimal places used for floating values.
    format : str
        The installed ``lofi.report_format`` renderer that writes the table.
    """

    name: str
    rows: tuple[str, ...]
    columns: tuple[str, ...]
    labels: tuple[str, ...]
    precision: int = 3
    format: str = "markdown"

    @dx.validates("name")
    @classmethod
    def _name_is_a_stem(cls, value: str) -> str:
        return report_stem(value)

    @dx.validates("rows")
    @classmethod
    def _rows_are_distinct(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _named_texts("a report layout", values)

    @dx.validates("columns")
    @classmethod
    def _columns_are_known(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return report_columns(_named_texts("a report layout", values))

    @dx.validates("labels")
    @classmethod
    def _labels_are_present(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        checked = tuple(ensure_name("a report column label", value) for value in values)
        if any(not value.strip() for value in checked):
            msg = f"a report column label has a blank entry: {checked!r}"
            raise ValueError(msg)
        return checked

    @dx.validates("precision")
    @classmethod
    def _precision_is_nonnegative(cls, value: int) -> int:
        return ensure_non_negative("a report precision", value)

    @dx.validates("format")
    @classmethod
    def _format_is_installed(cls, value: str) -> str:
        report_renderer(value)
        return value

    @dx.model_validator()
    def _columns_have_labels(self) -> Self:
        if len(self.columns) != len(self.labels):
            msg = (
                f"a report has {len(self.columns)} columns but "
                f"{len(self.labels)} labels"
            )
            raise ValueError(msg)
        return self

    def renderer(self) -> ReportRenderer:
        """Return the installed renderer the declared format names."""
        return report_renderer(self.format)


class FailurePolicySpec(dx.Model, extra="forbid"):
    """Declare how numerical failures and exhausted memory are retried.

    Parameters
    ----------
    numerical : Literal["final", "reseed"]
        Whether a numerical failure settles the point or retries it under the
        next deterministic attempt seed; it settles the point unless declared.
    out_of_memory : Literal["retry", "halve-batch"]
        Whether exhausted memory retries unchanged or halves the training
        batch before retrying; it retries unchanged unless declared.
    infrastructure_attempts : int
        The attempts each point receives across launches before an
        infrastructure failure settles it: one attempt and at most two retries
        unless declared.
    """

    numerical: Literal["final", "reseed"] = "final"
    out_of_memory: Literal["retry", "halve-batch"] = "retry"
    infrastructure_attempts: int = 3

    @dx.validates("infrastructure_attempts")
    @classmethod
    def _attempts_are_positive(cls, attempts: int) -> int:
        """Return a positive count of infrastructure attempts."""
        return ensure_positive("a point's infrastructure attempt count", attempts)


class SignatureTwinSettings(NamedTuple):
    """A local deterministic signature-twin draw."""

    count: int
    seed: int
    primary_condition: str
    observed_twin: str


class SignatureTwinsSpec(dx.Model, extra="forbid"):
    """Expand primary runs into observed assignments and fitted twins."""

    count: int
    seed: int
    primary_condition: str
    observed_twin: str

    @dx.validates("primary_condition", "observed_twin")
    @classmethod
    def _identity_is_named(cls, identity: str) -> str:
        """Return a nonempty condition or twin identity."""
        return ensure_name("a signature-twin identity", identity)

    @dx.validates("count")
    @classmethod
    def _count_supports_an_empirical_null(cls, count: int) -> int:
        """Return a twin count large enough to define the empirical null."""
        checked = ensure_non_negative("a signature-twin count", count)

        if checked < 3:
            msg = f"a signature-twin design declares fewer than three twins: {checked}"
            raise ValueError(msg)

        return checked

    @dx.validates("seed")
    @classmethod
    def _seed_is_nonnegative(cls, seed: int) -> int:
        """Return the local twin sampler's nonnegative seed."""
        return ensure_non_negative("a signature-twin seed", seed)

    @dx.model_validator()
    def _observed_is_distinct_from_draws(self) -> SignatureTwinsSpec:
        """Return a design whose observed identity cannot name a sampled twin."""
        twins = {f"twin-{position:03d}" for position in range(1, self.count + 1)}

        if self.observed_twin in twins:
            msg = (
                "a signature-twin design gives its observed arm a sampled-twin "
                f"identity: {self.observed_twin!r}"
            )
            raise ValueError(msg)

        return self

    def settings(self) -> SignatureTwinSettings:
        """Return immutable expansion settings."""
        return SignatureTwinSettings(
            self.count,
            self.seed,
            self.primary_condition,
            self.observed_twin,
        )


class ExperimentDocument(dx.Model, extra="forbid"):
    """Declares one experiment.

    Paths are relative to the directory holding the experiment document.

    Attributes
    ----------
    role : str
        The part of the empirical track the experiment serves, such as
        ``headline`` or ``analysis``.
    freeze_id : str
        The freeze the experiment runs against, ``freeze/<n>``.
    config : ExperimentConfigSpec
        The configuration document the sweep's points are composed over.
    sweep : Sweep
        The sweep the experiment expands, which declares the point count it
        must expand to.
    seed_policy : ExperimentSeedSpec
        The experiment's root seed, from which every point seed is derived.
    rows : tuple[RowReference, ...]
        The report rows the experiment produces, each a metric of an
        evaluator its configuration document declares.
    primary : RowReference
        The row the experiment's claim is read from, one of ``rows``.
    preregistration : str
        The preregistration document the factors and comparisons are
        declared in, which owns each comparison's family.
    factors : dict[str, FactorSpan]
        The axis that spans each preregistered factor, by factor name. A
        factor left out is spanned by the axis whose key is its name, each
        level being the axis value that realizes it.
    compute : ComputeSpec
        The accelerator-hour estimate, or the declaration that the experiment
        has none yet.
    pricing : PricingSpec
        The cost model the experiment's training steps are priced under; each
        stack off its own timed steps unless the document declares otherwise.
    contraction : ExperimentContraction
        The contraction ladder.
    stop_rules : tuple[StopRuleSpec, ...]
        The conditions under which the experiment halts early.
    decision_point : ExperimentDecision
        The decision point and its fallback claim.
    evaluation : EvaluationSpec
        The statistical settings the experiment's results are read under.
    """

    role: str
    freeze_id: str
    config: ExperimentConfigSpec
    launch: LaunchSpec
    failure: FailurePolicySpec
    sweep: Sweep
    signature_twins: SignatureTwinsSpec | None = None
    seed_policy: ExperimentSeedSpec
    rows: tuple[RowReference, ...]
    reports: tuple[ReportSpec, ...] = ()
    reducers: tuple[ReducerSpec, ...] = ()
    primary: RowReference
    preregistration: str
    factors: dict[str, FactorSpan] = dx.field(default_factory=dict)
    compute: ComputeSpec
    pricing: PricingSpec = StackBootstrapPricing()
    contraction: ExperimentContraction
    stop_rules: tuple[StopRuleSpec, ...]
    decision_point: ExperimentDecision
    evaluation: EvaluationSpec

    @dx.validates("role")
    @classmethod
    def _role_is_named(cls, role: str) -> str:
        """Return a nonblank experiment role."""
        return ensure_name("an experiment role", role)

    @dx.validates("freeze_id")
    @classmethod
    def _freeze_is_numbered(cls, freeze_id: str) -> str:
        """Return a freeze identifier of the form ``freeze/<n>``."""
        freeze_number(freeze_id)
        return freeze_id

    @dx.validates("preregistration")
    @classmethod
    def _text_is_named(cls, value: str) -> str:
        """Return nonblank experiment text."""
        return _named_texts("an experiment field", (value,))[0]

    @dx.validates("sweep")
    @classmethod
    def _sweep_declares_its_points(cls, sweep: Sweep) -> Sweep:
        """Return a sweep that declares its expected point count."""
        if sweep.expected_points is None:
            msg = "an experiment's sweep declares expected_points"
            raise ValueError(msg)

        return sweep

    @dx.validates("rows")
    @classmethod
    def _rows_are_distinct(
        cls, rows: tuple[RowReference, ...]
    ) -> tuple[RowReference, ...]:
        """Return a nonempty run of distinct row references."""
        _named_texts("the report rows", tuple(row.label for row in rows))

        return rows

    @dx.validates("reports")
    @classmethod
    def _reports_are_distinct(
        cls, reports: tuple[ReportSpec, ...]
    ) -> tuple[ReportSpec, ...]:
        if reports:
            _named_texts(
                "the experiment reports", tuple(report.name for report in reports)
            )
        return reports

    @dx.validates("reducers")
    @classmethod
    def _reducers_are_distinct(
        cls, reducers: tuple[ReducerSpec, ...]
    ) -> tuple[ReducerSpec, ...]:
        """Return at most one reducer declaration of each kind."""
        kinds = tuple(cast("str", reducer.kind) for reducer in reducers)

        if len(set(kinds)) != len(kinds):
            msg = f"experiment reducers repeat a kind: {kinds!r}"
            raise ValueError(msg)

        return reducers

    @dx.validates("stop_rules")
    @classmethod
    def _stop_rules_are_distinct(
        cls, rules: tuple[StopRuleSpec, ...]
    ) -> tuple[StopRuleSpec, ...]:
        """Return stop rules with distinct names."""
        names = tuple(rule.name for rule in rules)

        if len(names) != len(set(names)):
            msg = f"stop rules repeat a name: {names}"
            raise ValueError(msg)

        return rules

    @dx.validates("factors")
    @classmethod
    def _factors_are_named(
        cls, factors: dict[str, FactorSpan]
    ) -> dict[str, FactorSpan]:
        """Return factor bindings keyed by nonblank factor names."""
        for factor in factors:
            ensure_name("a preregistered factor", factor)

        return factors

    @dx.model_validator()
    def _rows_agree(self) -> Self:
        """Validate the rows reports and predicates read.

        Returns
        -------
        Self
            This experiment after validation.

        Raises
        ------
        ValueError
            If signature twins are declared over a configuration that is not
            a run, the primary row is not a declared row, or a row a report or
            predicate reads is not a declared row.
        """
        if self.signature_twins is not None and not isinstance(
            self.config, RunExperimentConfig
        ):
            msg = "signature twins require a run configuration"
            raise ValueError(msg)

        declared = frozenset(row.label for row in self.rows)

        for report in self.reports:
            unknown = tuple(label for label in report.rows if label not in declared)
            if unknown:
                msg = f"report {report.name!r} reads undeclared rows: {unknown!r}"
                raise ValueError(msg)

        if self.primary.label not in declared:
            msg = f"the primary row {self.primary.label!r} is not a declared row"
            raise ValueError(msg)

        predicates = (
            *(rule.when for rule in self.stop_rules),
            self.decision_point.signal,
        )

        for predicate in predicates:
            if isinstance(predicate, MetricPredicate) and (
                predicate.row.label not in declared
            ):
                msg = f"a predicate reads {predicate.row.label!r}, not a declared row"
                raise ValueError(msg)

        return self


def model_schema_uri(model: type[dx.Model], /) -> str:
    """Return the schema identifier of one boundary model a freeze records.

    Parameters
    ----------
    model : type[dx.Model]
        The boundary model, such as the checkpoint manifest.
    """
    return dx.schema_uri(model)
