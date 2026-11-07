"""Tests whether one root scalar suffices for cross-fitted acceptability fits.

The analysis reads one declared document holding the complete root-type
distribution and derivation count of every item, and for every run seed either
the reason the seed failed or its cross-fitted evidence: one held-out residual
per item under the one-scalar rung, and, for every fold, the held-out response
log densities of the one-scalar and two-scalar rungs, the participant and item
of each held-out response, the held-out response log densities of the
free-slope rung, and the posterior draws of the free signed slope.

Three falsifiers are computed for each completed seed. The residual-structure
test regresses the item residuals on the root entropy, the derivation count and
the identity of the highest-weight non-target type, under treatment encoding
with the first declared non-target type as reference. The two continuous
columns are standardized from the training rows of each fold alone, and the
statistic is the pooled out-of-fold R-squared against each fold's training
mean. Its null permutes the feature triples jointly across items, which keeps
their internal association while breaking their assignment to residuals. The
predictive-density comparison reports every rung's mean held-out log density
per fold and the two-scalar rung's improvement over the declared baseline rung,
the mean paired difference over every held-out response, with the interval of a
two-way cluster bootstrap that resamples participants and items together. The
slope summary reports the posterior median and equal-tail interval of the free
slope, together with the sign the interval establishes.

The permutation and improvement seeds of each run seed are derived from one
declared root. Statistics pooled over the folds are keyed by the folds and
rungs they pool. Every number is written once to a canonical Parquet report, and
the Markdown and LaTeX tables are rendered from that file by ordered selectors.
A JSON manifest records the digest of the document, each pooled key's folds and
rungs, the permutation and improvement provenance, each seed's outcome and each
established slope sign.
"""

import json
import math
import sys
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Self, final

import didactic.api as dx
import torch

from lofi.eval import (
    NO_ENUMERATION_BUDGET,
    Approximation,
    MetricResult,
    PooledJudgment,
    ReportIdentity,
    ReportRow,
    Resampling,
)
from lofi.judgment import Measurement, PosteriorDraws, predictive_density_row
from lofi.report import report_renderer
from lofi.run import judgment_report_row
from lofi.run.reporting import (
    ParquetReportSink,
    ReportSelector,
    render_table,
    select_required_rows,
)
from lofi.spec import derive_seed
from lofi.stats import CrossedBootstrap, LabelPermutation

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lofi._types import JsonValue
    from lofi.stats import BootstrapResult, PValue

PREDICTIVE_DENSITY: Final = "predictive_density"
"""Names the mean held-out log predictive density of one rung on one fold."""

FREE_SIGNED_SLOPE: Final = "free_signed_slope"
"""Names the posterior median and interval of the free signed slope."""

HELD_OUT_R_SQUARED: Final = "residual_structure_held_out_r_squared"
"""Names the pooled out-of-fold variance the root features explain."""

PERMUTATION_P_VALUE: Final = "residual_structure_permutation_p_value"
"""Names the joint item permutation p-value of the held-out R-squared."""

PREDICTIVE_DENSITY_IMPROVEMENT: Final = "predictive_density_improvement"
"""Names the two-scalar rung's mean paired log-density gain over the one-scalar."""

PERMUTATION_STREAM: Final = "residual-permutation"
"""Names the stream each run seed's joint item permutations are drawn from."""

PARTICIPANT: Final = "participant"
"""Names the row unit of the improvement's two-way cluster bootstrap."""

ITEM: Final = "item"
"""Names the column unit of the improvement's two-way cluster bootstrap."""

IMPROVEMENT_STREAM: Final = "improvement-bootstrap"
"""Names the stream each run seed's improvement resamples are drawn from."""

REPORT_NAME: Final = "report.parquet"
"""Names the canonical report file within the output directory."""

MARKDOWN_NAME: Final = "report.md"
"""Names the rendered Markdown table within the output directory."""

LATEX_NAME: Final = "report.tex"
"""Names the rendered LaTeX table within the output directory."""

MANIFEST_NAME: Final = "manifest.json"
"""Names the provenance manifest within the output directory."""

COLUMNS: Final = (
    "seed",
    "fold",
    "method",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "ci_level",
    "n",
    "resampling_unit",
    "resampling_seed",
)
"""Lists the report columns the rendered tables show, in order."""

LABELS: Final = (
    "Seed",
    "Fold",
    "Rung",
    "Metric",
    "Value",
    "Lower",
    "Upper",
    "Level",
    "Count",
    "Unit",
    "Permutation seed",
)
"""Lists the rendered column labels, one per entry of ``COLUMNS``."""

type SlopeSign = Literal["negative", "unresolved", "positive"]
"""Names the sign an equal-tail interval establishes for a free slope."""

type CountScale = Literal["log", "identity"]
"""Names the scale the derivation count enters the residual regression on."""

type BootstrapInterval = Literal["percentile", "bca"]
"""Names the bootstrap interval an improvement is reported with."""

type BaselineRole = Literal["one_scalar", "free_slope"]
"""Names the rung the two-scalar improvement is measured against."""


def _named(field: str, value: str, /) -> str:
    """Return a nonempty name or refuse it."""
    if not value:
        msg = f"{field} is empty"
        raise ValueError(msg)

    return value


def _distinct_names(field: str, values: Sequence[str], /) -> None:
    """Refuse an empty or repeated sequence of names."""
    for value in values:
        _named(f"a name in {field}", value)

    if not values:
        msg = f"{field} contains no names"
        raise ValueError(msg)

    if len(frozenset(values)) != len(values):
        repeated = sorted({value for value in values if values.count(value) > 1})
        msg = f"{field} repeats {repeated!r}"
        raise ValueError(msg)


def _finite(field: str, values: Sequence[float], /) -> None:
    """Refuse an empty or nonfinite sequence of numbers."""
    if not values:
        msg = f"{field} contains no values"
        raise ValueError(msg)

    for position, value in enumerate(values):
        if not math.isfinite(value):
            msg = f"{field} holds a nonfinite value at {position}: {value!r}"
            raise ValueError(msg)


class ImprovementDeclaration(dx.Model, extra="forbid"):
    """Declares how the two-scalar improvement's uncertainty is computed.

    The procedure is named explicitly. A two-way cluster bootstrap draws the
    participants and the items of every held-out response independently and
    with replacement, and holds each response once for every pairing of a draw
    of its participant with a draw of its item.

    Parameters
    ----------
    procedure : Literal["two-way-cluster-bootstrap"]
        Uncertainty procedure.
    replicates : int
        Resamples drawn for each run seed.
    level : float
        Coverage of the reported interval.
    interval : BootstrapInterval
        Bootstrap interval reported.
    """

    procedure: Literal["two-way-cluster-bootstrap"]
    replicates: int
    level: float
    interval: BootstrapInterval

    __axioms__ = (
        dx.axiom("replicates > 0", message="an improvement bootstrap resamples"),
    )

    @dx.model_validator()
    def _level_is_a_proportion(self) -> Self:
        """Refuse a coverage outside the open unit interval."""
        if not math.isfinite(self.level) or not 0.0 < self.level < 1.0:
            msg = f"an improvement's coverage is not a proportion: {self.level!r}"
            raise ValueError(msg)

        return self


class RungDeclaration(dx.Model, extra="forbid"):
    """Declares one fitted judgment-model rung.

    Parameters
    ----------
    name : str
        Rung name, reported as the row's method and rung.
    specification_hash : str
        Digest of the rung's judgment specification.
    """

    name: str
    specification_hash: str

    __axioms__ = (
        dx.axiom("len(name) > 0", message="a rung names itself"),
        dx.axiom(
            "len(specification_hash) > 0",
            message="a rung names its specification digest",
        ),
    )


class AnalysisDeclaration(dx.Model, extra="forbid"):
    """Declares the identity, rungs, inventory and budgets of one analysis.

    Parameters
    ----------
    freeze_id : str
        Freeze the report is quoted against.
    grammar_fingerprint : str
        Fingerprint of the grammar the root distributions were read under.
    suite : str
        Evaluation suite name.
    task : str
        Evaluation task name.
    dataset : str
        Judgment dataset name.
    split : str
        Judgment dataset split.
    normal_form : str | None
        Normal-form name, if one applies.
    held_out_convention : Literal["marginal", "plug_in"] | None
        Participant-effect treatment for held-out participants, if one applies.
    one_scalar : RungDeclaration
        Rung whose latent predictor is the target-mass scalar alone.
    two_scalar : RungDeclaration
        Rung that adds the second root scalar.
    free_slope : RungDeclaration
        Rung whose link slope is free to take either sign.
    improvement_baseline : str
        Name of the one-scalar or free-slope rung the two-scalar improvement
        is measured against.
    root_types : tuple[str, ...]
        Complete root-type inventory in its declared order.
    target_types : tuple[str, ...]
        Root types whose mass is the one scalar.
    derivation_count_scale : CountScale
        Scale the derivation count enters the residual regression on.
    level : float
        Coverage of the slope's equal-tail interval.
    replicates : int
        Joint item permutations drawn for each seed's residual test.
    improvement : ImprovementDeclaration
        Procedure giving the two-scalar improvement its uncertainty.
    seed_root : int
        Root each run seed's permutation and improvement seeds derive from.
    """

    freeze_id: str
    grammar_fingerprint: str
    suite: str
    task: str
    dataset: str
    split: str
    normal_form: str | None
    held_out_convention: Literal["marginal", "plug_in"] | None
    one_scalar: RungDeclaration
    two_scalar: RungDeclaration
    free_slope: RungDeclaration
    improvement_baseline: str
    root_types: tuple[str, ...]
    target_types: tuple[str, ...]
    derivation_count_scale: CountScale
    level: float
    replicates: int
    improvement: ImprovementDeclaration
    seed_root: int

    __axioms__ = (
        dx.axiom("len(freeze_id) > 0", message="an analysis names its freeze"),
        dx.axiom(
            "len(grammar_fingerprint) > 0",
            message="an analysis names its grammar fingerprint",
        ),
        dx.axiom("len(suite) > 0", message="an analysis names its suite"),
        dx.axiom("len(task) > 0", message="an analysis names its task"),
        dx.axiom("len(dataset) > 0", message="an analysis names its dataset"),
        dx.axiom("len(split) > 0", message="an analysis names its split"),
        dx.axiom("replicates > 0", message="an analysis draws a permutation"),
        dx.axiom("seed_root >= 0", message="a seed root is nonnegative"),
    )

    @dx.model_validator()
    def _inventory_rungs_and_level_are_declared(self) -> Self:
        """Validate the inventory, the targets, the rung names and the level.

        Returns
        -------
        Self
            This declaration after validation.

        Raises
        ------
        ValueError
            If the inventory or the targets are empty or repeated, a target is
            not in the inventory, every type is a target, two rungs share a
            name, the improvement baseline names neither the one-scalar nor
            the free-slope rung, or the level is not a proportion.
        """
        _distinct_names("the root-type inventory", self.root_types)
        _distinct_names("the target root types", self.target_types)
        unknown = sorted(frozenset(self.target_types) - frozenset(self.root_types))

        if unknown:
            msg = f"target root types are not in the inventory: {unknown!r}"
            raise ValueError(msg)

        if len(self.target_types) == len(self.root_types):
            msg = "every root type is a target, so no non-target type remains"
            raise ValueError(msg)

        _distinct_names(
            "the rung names",
            (self.one_scalar.name, self.two_scalar.name, self.free_slope.name),
        )
        baselines = (self.one_scalar.name, self.free_slope.name)

        if self.improvement_baseline not in baselines:
            msg = (
                f"the improvement baseline {self.improvement_baseline!r} names "
                f"neither of the rungs {baselines!r}"
            )
            raise ValueError(msg)

        if not math.isfinite(self.level) or not 0.0 < self.level < 1.0:
            msg = f"a slope interval's coverage is not a proportion: {self.level!r}"
            raise ValueError(msg)

        return self


class RootDistributionRecord(dx.Model, extra="forbid"):
    """Records one item's complete root-type distribution.

    Parameters
    ----------
    item : str
        Item identity.
    log_weights : tuple[float, ...]
        Finite log weight of each declared root type, in inventory order.
    derivation_count : int
        Number of derivations the parser admitted for the item.
    """

    item: str
    log_weights: tuple[float, ...]
    derivation_count: int

    __axioms__ = (
        dx.axiom("len(item) > 0", message="a root distribution names its item"),
        dx.axiom(
            "derivation_count > 0",
            message="an item admits at least one derivation",
        ),
    )

    @dx.model_validator()
    def _weights_are_finite(self) -> Self:
        """Refuse an empty or nonfinite weight vector."""
        _finite(f"the log weights of item {self.item!r}", self.log_weights)

        return self


class ResidualRecord(dx.Model, extra="forbid"):
    """Records one item's cross-fitted residual under the one-scalar rung.

    Parameters
    ----------
    item : str
        Item identity.
    fold : int
        Zero-based fold in which the item was held out.
    residual : float
        Held-out residual of the item.
    """

    item: str
    fold: int
    residual: float

    __axioms__ = (
        dx.axiom("len(item) > 0", message="a residual names its item"),
        dx.axiom("fold >= 0", message="a fold index is nonnegative"),
    )

    @dx.model_validator()
    def _residual_is_finite(self) -> Self:
        """Refuse a nonfinite residual."""
        _finite(f"the residual of item {self.item!r}", (self.residual,))

        return self


class FoldRecord(dx.Model, extra="forbid"):
    """Records one fold's held-out densities and free-slope posterior.

    Parameters
    ----------
    fold : int
        Zero-based fold index.
    one_scalar_log_densities : tuple[float, ...]
        Held-out response log densities under the one-scalar rung.
    two_scalar_log_densities : tuple[float, ...]
        Log densities of the same responses under the two-scalar rung.
    free_slope_log_densities : tuple[float, ...]
        Log densities of the same responses under the free-slope rung.
    slope_draws : tuple[float, ...]
        Posterior draws of the free signed slope fitted on the training rows.
    response_participants : tuple[str, ...]
        Participant of each held-out response, aligned with the densities.
    response_items : tuple[str, ...]
        Item of each held-out response, aligned with the densities.
    """

    fold: int
    one_scalar_log_densities: tuple[float, ...]
    two_scalar_log_densities: tuple[float, ...]
    free_slope_log_densities: tuple[float, ...]
    slope_draws: tuple[float, ...]
    response_participants: tuple[str, ...]
    response_items: tuple[str, ...]

    __axioms__ = (dx.axiom("fold >= 0", message="a fold index is nonnegative"),)

    @dx.model_validator()
    def _evidence_is_finite_and_paired(self) -> Self:
        """Refuse empty, nonfinite, unpaired or unlabeled fold evidence."""
        _finite(
            f"the one-scalar log densities of fold {self.fold}",
            self.one_scalar_log_densities,
        )
        _finite(
            f"the two-scalar log densities of fold {self.fold}",
            self.two_scalar_log_densities,
        )
        _finite(
            f"the free-slope log densities of fold {self.fold}",
            self.free_slope_log_densities,
        )
        _finite(f"the slope draws of fold {self.fold}", self.slope_draws)

        for rung, densities in (
            ("two-scalar", self.two_scalar_log_densities),
            ("free-slope", self.free_slope_log_densities),
        ):
            if len(densities) != len(self.one_scalar_log_densities):
                msg = (
                    f"fold {self.fold} scores {len(self.one_scalar_log_densities)} "
                    f"responses under the one-scalar rung and "
                    f"{len(densities)} under the {rung} rung"
                )
                raise ValueError(msg)

        for unit, labels in (
            ("participant", self.response_participants),
            ("item", self.response_items),
        ):
            if len(labels) != len(self.one_scalar_log_densities):
                msg = (
                    f"fold {self.fold} labels {len(labels)} of its "
                    f"{len(self.one_scalar_log_densities)} responses with a {unit}"
                )
                raise ValueError(msg)

            for label in labels:
                _named(f"a response {unit} of fold {self.fold}", label)

        return self


class SeedRecord(dx.TaggedUnion, discriminator="outcome", extra="forbid"):
    """Records the outcome of one run seed."""


class CompletedSeedRecord(SeedRecord, extra="forbid"):
    """Records the cross-fitted evidence of one completed run seed.

    Parameters
    ----------
    outcome : Literal["completed"]
        Outcome tag.
    seed : int
        Run seed.
    residuals : tuple[ResidualRecord, ...]
        One held-out residual per item.
    folds : tuple[FoldRecord, ...]
        Evidence of every fold.
    """

    outcome: Literal["completed"] = "completed"
    seed: int
    residuals: tuple[ResidualRecord, ...]
    folds: tuple[FoldRecord, ...]

    __axioms__ = (dx.axiom("seed >= 0", message="a run seed is nonnegative"),)

    @dx.model_validator()
    def _folds_partition_the_items(self) -> Self:
        """Validate that the residual folds are exactly the recorded folds.

        Returns
        -------
        Self
            This record after validation.

        Raises
        ------
        ValueError
            If an item or a fold repeats, fewer than two folds are recorded,
            or the residuals hold out items in a fold with no record.
        """
        _distinct_names(
            f"the residual items of seed {self.seed}",
            tuple(record.item for record in self.residuals),
        )
        recorded = tuple(record.fold for record in self.folds)

        if len(frozenset(recorded)) != len(recorded):
            msg = f"seed {self.seed} records a fold twice: {recorded!r}"
            raise ValueError(msg)

        if len(recorded) < 2:
            msg = f"seed {self.seed} records fewer than two folds: {recorded!r}"
            raise ValueError(msg)

        held_out = frozenset(record.fold for record in self.residuals)

        if held_out != frozenset(recorded):
            msg = (
                f"seed {self.seed} holds out items in folds {sorted(held_out)!r} "
                f"and records folds {sorted(recorded)!r}"
            )
            raise ValueError(msg)

        return self


class FailedSeedRecord(SeedRecord, extra="forbid"):
    """Records a run seed whose fits did not complete.

    Parameters
    ----------
    outcome : Literal["failed"]
        Outcome tag.
    seed : int
        Run seed.
    reason : str
        Why the seed's fits did not complete.
    """

    outcome: Literal["failed"] = "failed"
    seed: int
    reason: str

    __axioms__ = (
        dx.axiom("seed >= 0", message="a run seed is nonnegative"),
        dx.axiom("len(reason) > 0", message="a failed seed states its reason"),
    )


class AnalysisDocument(dx.Model, extra="forbid"):
    """Declares one monotone-scalar analysis and its caller-supplied evidence.

    Parameters
    ----------
    declaration : AnalysisDeclaration
        Identity, rungs, inventory and budgets.
    items : tuple[RootDistributionRecord, ...]
        Complete root distribution of every item.
    seeds : tuple[SeedRecord, ...]
        Outcome of every run seed.
    """

    declaration: AnalysisDeclaration
    items: tuple[RootDistributionRecord, ...]
    seeds: tuple[SeedRecord, ...]

    @dx.model_validator()
    def _evidence_covers_the_declared_items(self) -> Self:
        """Validate the items and seeds against the declaration.

        Returns
        -------
        Self
            This document after validation.

        Raises
        ------
        ValueError
            If an item repeats, a distribution does not weigh every declared
            type, no seed or a repeated seed is recorded, or a completed seed's
            residuals do not cover exactly the declared items.
        """
        items = tuple(record.item for record in self.items)
        _distinct_names("the root-distribution items", items)
        width = len(self.declaration.root_types)

        for record in self.items:
            if len(record.log_weights) != width:
                msg = (
                    f"item {record.item!r} weighs {len(record.log_weights)} root "
                    f"types and the inventory declares {width}"
                )
                raise ValueError(msg)

        seeds = tuple(record.seed for record in self.seeds)

        if not seeds:
            msg = "an analysis records no run seed"
            raise ValueError(msg)

        if len(frozenset(seeds)) != len(seeds):
            msg = f"an analysis records a run seed twice: {seeds!r}"
            raise ValueError(msg)

        declared = frozenset(items)

        for record in self.seeds:
            match record:
                case CompletedSeedRecord(seed=seed, residuals=residuals):
                    covered = frozenset(residual.item for residual in residuals)

                    if covered != declared:
                        msg = (
                            f"seed {seed} has residuals for "
                            f"{len(covered - declared)} undeclared items and "
                            f"lacks {len(declared - covered)} declared items"
                        )
                        raise ValueError(msg)

        return self


class Rung(NamedTuple):
    """One fitted rung's report identity.

    Attributes
    ----------
    name : str
        Rung name.
    specification_hash : str
        Digest of the rung's judgment specification.
    """

    name: str
    specification_hash: str


class RootFeatures(NamedTuple):
    """Features of a root distribution beyond its target-mass scalar.

    Attributes
    ----------
    entropy : float
        Shannon entropy of the complete root distribution, in nats.
    non_target : str
        Highest-weight type outside the targets, first in declared order on a tie.
    derivation_count : int
        Number of derivations admitted for the item.
    """

    entropy: float
    non_target: str
    derivation_count: int


class FoldEvidence(NamedTuple):
    """One fold's held-out densities and free-slope posterior.

    Attributes
    ----------
    fold : int
        Zero-based fold index.
    one_scalar : tuple[float, ...]
        Held-out response log densities under the one-scalar rung.
    two_scalar : tuple[float, ...]
        Log densities of the same responses under the two-scalar rung.
    free_slope : tuple[float, ...]
        Log densities of the same responses under the free-slope rung.
    slope : tuple[float, ...]
        Posterior draws of the free signed slope.
    participants : tuple[str, ...]
        Participant of each held-out response.
    items : tuple[str, ...]
        Item of each held-out response.
    """

    fold: int
    one_scalar: tuple[float, ...]
    two_scalar: tuple[float, ...]
    free_slope: tuple[float, ...]
    slope: tuple[float, ...]
    participants: tuple[str, ...]
    items: tuple[str, ...]


class CompletedSeed(NamedTuple):
    """One completed seed's item residuals, their features and its folds.

    Attributes
    ----------
    seed : int
        Run seed.
    held_out_folds : tuple[int, ...]
        Fold in which each item was held out.
    residuals : tuple[float, ...]
        Held-out residual of each item.
    features : tuple[RootFeatures, ...]
        Root features of each item, aligned with the residuals.
    folds : tuple[FoldEvidence, ...]
        Evidence of every fold, in fold order.
    """

    seed: int
    held_out_folds: tuple[int, ...]
    residuals: tuple[float, ...]
    features: tuple[RootFeatures, ...]
    folds: tuple[FoldEvidence, ...]


class FailedSeed(NamedTuple):
    """One run seed whose fits did not complete.

    Attributes
    ----------
    seed : int
        Run seed.
    reason : str
        Why the seed's fits did not complete.
    """

    seed: int
    reason: str


class BootstrapImprovement(NamedTuple):
    """A two-way cluster bootstrap of the improvement over participants and items.

    Attributes
    ----------
    replicates : int
        Resamples drawn for each run seed.
    level : float
        Coverage of the reported interval.
    interval : BootstrapInterval
        Bootstrap interval reported.
    """

    replicates: int
    level: float
    interval: BootstrapInterval


class Analysis(NamedTuple):
    """One validated monotone-scalar analysis.

    Attributes
    ----------
    identity : ReportIdentity
        Freeze, declaration, grammar and convention identity.
    suite : str
        Evaluation suite name.
    task : str
        Evaluation task name.
    dataset : str
        Judgment dataset name.
    split : str
        Judgment dataset split.
    normal_form : str | None
        Normal-form name, if one applies.
    held_out_convention : str | None
        Participant-effect treatment for held-out participants, if one applies.
    one_scalar : Rung
        One-scalar rung.
    two_scalar : Rung
        Two-scalar rung.
    free_slope : Rung
        Free signed-slope rung.
    baseline : BaselineRole
        Rung the two-scalar improvement is measured against.
    levels : tuple[str, ...]
        Non-target types in declared order; the first is the treatment reference.
    count_scale : CountScale
        Scale the derivation count enters the residual regression on.
    level : float
        Coverage of the slope's equal-tail interval.
    replicates : int
        Joint item permutations drawn for each seed.
    improvement : BootstrapImprovement
        Procedure giving the two-scalar improvement its uncertainty.
    seed_root : int
        Root each run seed's permutation and improvement seeds derive from.
    seeds : tuple[CompletedSeed | FailedSeed, ...]
        Every recorded seed, in document order.
    """

    identity: ReportIdentity
    suite: str
    task: str
    dataset: str
    split: str
    normal_form: str | None
    held_out_convention: str | None
    one_scalar: Rung
    two_scalar: Rung
    free_slope: Rung
    baseline: BaselineRole
    levels: tuple[str, ...]
    count_scale: CountScale
    level: float
    replicates: int
    improvement: BootstrapImprovement
    seed_root: int
    seeds: tuple[CompletedSeed | FailedSeed, ...]


@final
class DeclaredGrammar:
    """The grammar a report row is quoted under, named by its fingerprint.

    Parameters
    ----------
    fingerprint : str
        Stable grammar fingerprint.
    """

    __slots__ = ("_fingerprint",)

    def __init__(self, fingerprint: str) -> None:
        self._fingerprint = fingerprint

    @property
    def fingerprint(self) -> str:
        """Return the stable grammar fingerprint."""
        return self._fingerprint


def root_features(
    log_weights: Sequence[float],
    types: Sequence[str],
    targets: frozenset[str],
    derivation_count: int,
    /,
) -> RootFeatures:
    """Summarize one complete ordered distribution over root types.

    Parameters
    ----------
    log_weights : Sequence[float]
        Finite log weight of each type, unnormalized, in inventory order.
    types : Sequence[str]
        Complete root-type inventory in declared order.
    targets : frozenset[str]
        Target types, a proper subset of the inventory.
    derivation_count : int
        Number of derivations admitted for the item.

    Returns
    -------
    RootFeatures
        Entropy in nats, highest-weight non-target type and derivation count.
    """
    peak = max(log_weights)
    masses = tuple(math.exp(weight - peak) for weight in log_weights)
    total = math.fsum(masses)
    entropy = -math.fsum(
        mass / total * math.log(mass / total) for mass in masses if mass > 0.0
    )
    highest: tuple[str, float] | None = None

    for name, weight in zip(types, log_weights, strict=True):
        if name in targets:
            continue

        if highest is None or weight > highest[1]:
            highest = (name, weight)

    if highest is None:
        msg = "a root distribution has no non-target type"
        raise ValueError(msg)

    return RootFeatures(max(entropy, 0.0), highest[0], derivation_count)


def decode_document(document: AnalysisDocument, /) -> Analysis:
    """Lower one validated analysis document to its interior values.

    Parameters
    ----------
    document : AnalysisDocument
        Validated analysis document.

    Returns
    -------
    Analysis
        Identity, rungs and per-seed evidence with every item's root features.
    """
    declaration = document.declaration
    types = declaration.root_types
    targets = frozenset(declaration.target_types)
    features = {
        record.item: root_features(
            record.log_weights, types, targets, record.derivation_count
        )
        for record in document.items
    }
    seeds: list[CompletedSeed | FailedSeed] = []

    improvement = declaration.improvement
    for record in document.seeds:
        match record:
            case CompletedSeedRecord(seed=seed, residuals=residuals, folds=folds):
                seeds.append(
                    CompletedSeed(
                        seed,
                        tuple(residual.fold for residual in residuals),
                        tuple(residual.residual for residual in residuals),
                        tuple(features[residual.item] for residual in residuals),
                        tuple(
                            FoldEvidence(
                                fold.fold,
                                fold.one_scalar_log_densities,
                                fold.two_scalar_log_densities,
                                fold.free_slope_log_densities,
                                fold.slope_draws,
                                fold.response_participants,
                                fold.response_items,
                            )
                            for fold in sorted(folds, key=lambda fold: fold.fold)
                        ),
                    )
                )
            case FailedSeedRecord(seed=seed, reason=reason):
                seeds.append(FailedSeed(seed, reason))

    return Analysis(
        ReportIdentity(
            declaration,
            DeclaredGrammar(declaration.grammar_fingerprint),
            None,
            freeze_id=declaration.freeze_id,
        ),
        declaration.suite,
        declaration.task,
        declaration.dataset,
        declaration.split,
        declaration.normal_form,
        declaration.held_out_convention,
        Rung(declaration.one_scalar.name, declaration.one_scalar.specification_hash),
        Rung(declaration.two_scalar.name, declaration.two_scalar.specification_hash),
        Rung(declaration.free_slope.name, declaration.free_slope.specification_hash),
        "free_slope"
        if declaration.improvement_baseline == declaration.free_slope.name
        else "one_scalar",
        tuple(name for name in types if name not in targets),
        declaration.derivation_count_scale,
        declaration.level,
        declaration.replicates,
        BootstrapImprovement(
            improvement.replicates,
            improvement.level,
            improvement.interval,
        ),
        declaration.seed_root,
        tuple(seeds),
    )


def read_document(path: Path, /) -> Analysis:
    """Validate an analysis document file at its boundary and lower it.

    Parameters
    ----------
    path : Path
        JSON analysis document.

    Returns
    -------
    Analysis
        The validated analysis.
    """
    return decode_document(AnalysisDocument.model_validate_json(path.read_bytes()))


def _design(
    features: Sequence[RootFeatures], levels: tuple[str, ...], scale: CountScale, /
) -> torch.Tensor:
    """Return the unstandardized continuous columns, then the treatment columns."""
    return torch.tensor(
        [
            (
                feature.entropy,
                math.log(feature.derivation_count)
                if scale == "log"
                else float(feature.derivation_count),
                *(float(feature.non_target == level) for level in levels[1:]),
            )
            for feature in features
        ],
        dtype=torch.float64,
    )


def _with_intercept(
    rows: torch.Tensor, center: torch.Tensor, spread: torch.Tensor, /
) -> torch.Tensor:
    """Return an intercept, the standardized continuous columns and the rest."""
    return torch.cat(
        (
            rows.new_ones((rows.shape[0], 1)),
            (rows[:, :2] - center) / spread,
            rows[:, 2:],
        ),
        dim=1,
    )


def residual_statistic(
    seed: CompletedSeed, levels: tuple[str, ...], scale: CountScale, /
) -> Callable[[Sequence[RootFeatures]], float]:
    """Return the pooled out-of-fold R-squared of one seed's residuals.

    The returned statistic takes one feature triple per item, aligned with the
    seed's residuals. Within each fold, the continuous columns are centered and
    scaled by the training rows alone, a column constant over the training rows
    is only centered, and the least-squares fit to the training residuals
    predicts the held-out ones. The pooled held-out squared error is compared
    with the pooled squared deviation of the held-out residuals from their
    fold's training mean.

    Parameters
    ----------
    seed : CompletedSeed
        Item residuals and the folds holding them out.
    levels : tuple[str, ...]
        Declared non-target types; the first is the treatment reference.
    scale : CountScale
        Scale the derivation count enters on.

    Returns
    -------
    Callable[[Sequence[RootFeatures]], float]
        The statistic over one arrangement of the feature triples.

    Raises
    ------
    ValueError
        If the held-out residuals do not vary about their training means.
    """
    response = torch.tensor(seed.residuals, dtype=torch.float64)
    folds = sorted(frozenset(seed.held_out_folds))
    masks = tuple(
        torch.tensor([held == fold for held in seed.held_out_folds], dtype=torch.bool)
        for fold in folds
    )
    baseline = math.fsum(
        float(torch.sum(torch.square(response[mask] - response[~mask].mean())))
        for mask in masks
    )

    if baseline == 0.0:
        msg = (
            f"the held-out residuals of seed {seed.seed} do not vary about their "
            "training means"
        )
        raise ValueError(msg)

    def statistic(features: Sequence[RootFeatures]) -> float:
        matrix = _design(features, levels, scale)
        error = 0.0

        for mask in masks:
            train = matrix[~mask]
            center = train[:, :2].mean(dim=0)
            spread = torch.sqrt(torch.mean(torch.square(train[:, :2] - center), dim=0))
            spread = torch.where(spread > 0.0, spread, torch.ones_like(spread))
            coefficients = torch.linalg.lstsq(
                _with_intercept(train, center, spread),
                response[~mask].unsqueeze(1),
            ).solution
            predicted = _with_intercept(matrix[mask], center, spread) @ coefficients
            error += float(
                torch.sum(torch.square(response[mask] - predicted.squeeze(1)))
            )

        return 1.0 - error / baseline

    return statistic


def residual_structure(
    seed: CompletedSeed,
    levels: tuple[str, ...],
    scale: CountScale,
    null: LabelPermutation[RootFeatures],
    /,
) -> tuple[float, PValue]:
    """Return one seed's held-out R-squared and its joint item permutation p-value.

    Parameters
    ----------
    seed : CompletedSeed
        Item residuals, features and folds.
    levels : tuple[str, ...]
        Declared non-target types; the first is the treatment reference.
    scale : CountScale
        Scale the derivation count enters on.
    null : LabelPermutation[RootFeatures]
        Null permuting complete feature triples across items.

    Returns
    -------
    tuple[float, PValue]
        The observed statistic and its upper-tail p-value.
    """
    statistic = residual_statistic(seed, levels, scale)

    return statistic(seed.features), null.test(statistic, seed.features, tail="greater")


def slope_summary(
    draws: Sequence[float], level: float, /
) -> tuple[Measurement, SlopeSign]:
    """Summarize posterior draws of a free signed slope.

    Parameters
    ----------
    draws : Sequence[float]
        Finite posterior draws.
    level : float
        Equal-tail interval coverage.

    Returns
    -------
    tuple[Measurement, SlopeSign]
        The posterior median bracketed by its equal-tail interval, and the
        sign the interval establishes, which is unresolved when it covers zero.
    """
    values = torch.tensor(draws, dtype=torch.float64)
    intervals = PosteriorDraws(values.unsqueeze(1)).intervals(level)
    lower = float(intervals.lower[0])
    upper = float(intervals.upper[0])
    median = float(torch.quantile(values, 0.5))
    sign: SlopeSign = (
        "positive" if lower > 0.0 else "negative" if upper < 0.0 else "unresolved"
    )

    return (
        Measurement(
            value=median,
            n=len(draws),
            unit="posterior_draw",
            lower=lower,
            upper=upper,
            level=level,
        ),
        sign,
    )


def _mean(values: Sequence[float], /) -> float:
    """Return the exact mean of a nonempty sequence."""
    return math.fsum(values) / len(values)


def improvement_interval(
    seed: CompletedSeed,
    baseline: BaselineRole,
    procedure: BootstrapImprovement,
    bootstrap_seed: int,
    /,
) -> BootstrapResult:
    """Bootstrap the two-scalar rung's mean paired held-out log-density gain.

    Parameters
    ----------
    seed : CompletedSeed
        Held-out densities, participants and items of every fold.
    baseline : BaselineRole
        Rung whose densities the two-scalar densities are paired against.
    procedure : BootstrapImprovement
        Declared budget, coverage and interval.
    bootstrap_seed : int
        Seed the resamples are drawn under.

    Returns
    -------
    BootstrapResult
        The mean of the two-scalar minus baseline log density over every
        held-out response, with its intervals under participants and items
        resampled together.
    """
    differences = tuple(
        two - base
        for fold in seed.folds
        for base, two in zip(
            fold.free_slope if baseline == "free_slope" else fold.one_scalar,
            fold.two_scalar,
            strict=True,
        )
    )
    bootstrap = CrossedBootstrap(
        units=(PARTICIPANT, ITEM),
        rows=tuple(unit for fold in seed.folds for unit in fold.participants),
        columns=tuple(unit for fold in seed.folds for unit in fold.items),
        replicates=procedure.replicates,
        seed=bootstrap_seed,
        level=procedure.level,
    )

    return bootstrap.interval(differences, _mean)


class SeedReport(NamedTuple):
    """One completed seed's report rows, their selectors and its verdicts.

    Attributes
    ----------
    rows : tuple[ReportRow, ...]
        Canonical report rows in table order.
    selectors : tuple[ReportSelector, ...]
        Selector of each row, aligned with the rows.
    signs : tuple[tuple[int, SlopeSign], ...]
        Fold and established slope sign of each fold.
    pooled : tuple[PooledJudgment, ...]
        Key of each statistic pooled over the folds.
    null : PValue
        Joint item permutation p-value with its provenance.
    improvement : BootstrapResult
        Bootstrap of the two-scalar improvement with its provenance.
    """

    rows: tuple[ReportRow, ...]
    selectors: tuple[ReportSelector, ...]
    signs: tuple[tuple[int, SlopeSign], ...]
    pooled: tuple[PooledJudgment, ...]
    null: PValue
    improvement: BootstrapResult


def _selector(
    analysis: Analysis,
    *,
    method: str,
    metric: str,
    seed: int,
    specification_hash: str,
    fold: int | None,
    rung: str,
) -> ReportSelector:
    """Return the complete selector of one row this analysis emits."""
    identity = analysis.identity

    return ReportSelector(
        freeze_id=identity.freeze_id,
        config_hash=identity.config_hash,
        grammar_fingerprint=identity.grammar_fingerprint,
        convention_hash=identity.convention_hash,
        suite=analysis.suite,
        task=analysis.task,
        dataset=analysis.dataset,
        split=analysis.split,
        method=method,
        metric=metric,
        seed=seed,
        normal_form=analysis.normal_form,
        specification_hash=specification_hash,
        fold=fold,
        rung=rung,
        held_out_convention=analysis.held_out_convention,
    )


def _fold_row(
    analysis: Analysis,
    measurement: Measurement,
    *,
    rung: Rung,
    metric: str,
    seed: int,
    fold: int,
) -> tuple[ReportRow, ReportSelector]:
    """Lower one fold-level measurement of one rung into a report row."""
    row = judgment_report_row(
        measurement,
        analysis.identity,
        suite=analysis.suite,
        task=analysis.task,
        dataset=analysis.dataset,
        split=analysis.split,
        metric=metric,
        seed=seed,
        normal_form=analysis.normal_form,
        specification_hash=rung.specification_hash,
        fold=fold,
        rung=rung.name,
        held_out_convention=analysis.held_out_convention,
    )

    return row, _selector(
        analysis,
        method=rung.name,
        metric=metric,
        seed=seed,
        specification_hash=rung.specification_hash,
        fold=fold,
        rung=rung.name,
    )


def _pooled_row(
    analysis: Analysis,
    result: MetricResult,
    *,
    metric: str,
    seed: int,
    pooled: PooledJudgment,
) -> tuple[ReportRow, ReportSelector]:
    """Lower one statistic pooled over the folds into a report row."""
    row = ReportRow(
        identity=analysis.identity,
        method=pooled.rung,
        metric=metric,
        result=result,
        approximation=Approximation(NO_ENUMERATION_BUDGET, residual=None),
        suite=analysis.suite,
        task=analysis.task,
        dataset=analysis.dataset,
        split=analysis.split,
        seed=seed,
        normal_form=analysis.normal_form,
        pooled=pooled,
        held_out_convention=analysis.held_out_convention,
    )

    return row, _selector(
        analysis,
        method=pooled.rung,
        metric=metric,
        seed=seed,
        specification_hash=pooled.specification_hash,
        fold=None,
        rung=pooled.rung,
    )


def seed_report(analysis: Analysis, seed: CompletedSeed, /) -> SeedReport:
    """Compute and lower every falsifier of one completed seed.

    Each fold contributes the one-scalar, two-scalar and free-slope predictive
    densities and the free-slope summary. The residual-structure R-squared and
    its permutation p-value, keyed by the one-scalar rung over every fold, and
    the two-scalar improvement, keyed by the two-scalar and baseline rungs over
    every fold, follow them.
    The permutation and bootstrap seeds derive from the declared root and the
    run seed.

    Parameters
    ----------
    analysis : Analysis
        The validated analysis.
    seed : CompletedSeed
        One completed seed.

    Returns
    -------
    SeedReport
        Report rows, selectors, slope signs, pooled keys and the provenance of
        the permutation null and the improvement bootstrap.
    """
    lowered: list[tuple[ReportRow, ReportSelector]] = []
    signs: list[tuple[int, SlopeSign]] = []

    for fold in seed.folds:
        for rung, densities in (
            (analysis.one_scalar, fold.one_scalar),
            (analysis.two_scalar, fold.two_scalar),
            (analysis.free_slope, fold.free_slope),
        ):
            density = predictive_density_row(
                rung.name, torch.tensor(densities, dtype=torch.float64)
            ).predictive_density
            lowered.append(
                _fold_row(
                    analysis,
                    density,
                    rung=rung,
                    metric=PREDICTIVE_DENSITY,
                    seed=seed.seed,
                    fold=fold.fold,
                )
            )

        slope, sign = slope_summary(fold.slope, analysis.level)
        signs.append((fold.fold, sign))
        lowered.append(
            _fold_row(
                analysis,
                slope,
                rung=analysis.free_slope,
                metric=FREE_SIGNED_SLOPE,
                seed=seed.seed,
                fold=fold.fold,
            )
        )

    folds = tuple(fold.fold for fold in seed.folds)
    one, two = analysis.one_scalar, analysis.two_scalar
    base = analysis.free_slope if analysis.baseline == "free_slope" else one
    residual_key = PooledJudgment(
        rungs=(one.name,), specification_hashes=(one.specification_hash,), folds=folds
    )
    improvement_key = PooledJudgment(
        rungs=(two.name, base.name),
        specification_hashes=(two.specification_hash, base.specification_hash),
        folds=folds,
    )
    permutation_seed = derive_seed(
        analysis.seed_root, f"{PERMUTATION_STREAM}:{seed.seed}"
    )
    null: LabelPermutation[RootFeatures] = LabelPermutation(
        replicates=analysis.replicates, seed=permutation_seed
    )
    r_squared, p_value = residual_structure(
        seed, analysis.levels, analysis.count_scale, null
    )
    procedure = analysis.improvement
    improvement = improvement_interval(
        seed,
        analysis.baseline,
        procedure,
        derive_seed(analysis.seed_root, f"{IMPROVEMENT_STREAM}:{seed.seed}"),
    )
    items = len(seed.residuals)
    lowered.extend(
        (
            _pooled_row(
                analysis,
                MetricResult(r_squared, items, Resampling("item")),
                metric=HELD_OUT_R_SQUARED,
                seed=seed.seed,
                pooled=residual_key,
            ),
            _pooled_row(
                analysis,
                MetricResult(
                    p_value.value, items, Resampling("item", seed=permutation_seed)
                ),
                metric=PERMUTATION_P_VALUE,
                seed=seed.seed,
                pooled=residual_key,
            ),
            _pooled_row(
                analysis,
                MetricResult(
                    improvement.observed,
                    improvement.units,
                    Resampling(improvement.unit, seed=improvement.seed),
                    ci=improvement.percentile
                    if procedure.interval == "percentile"
                    else improvement.bca,
                ),
                metric=PREDICTIVE_DENSITY_IMPROVEMENT,
                seed=seed.seed,
                pooled=improvement_key,
            ),
        )
    )

    return SeedReport(
        tuple(row for row, _ in lowered),
        tuple(selector for _, selector in lowered),
        tuple(signs),
        (residual_key, improvement_key),
        p_value,
        improvement,
    )


def _completed_entry(
    seed: CompletedSeed, report: SeedReport, procedure: BootstrapImprovement, /
) -> JsonValue:
    """Return one completed seed's manifest entry."""
    provenance = report.null.provenance
    improvement = report.improvement

    return {
        "seed": seed.seed,
        "outcome": "completed",
        "slope_signs": [{"fold": fold, "sign": sign} for fold, sign in report.signs],
        "pooled": {key.specification_hash: key.to_spec() for key in report.pooled},
        "permutation": {
            "null": provenance.null,
            "family": provenance.family,
            "replicates": provenance.replicates,
            "tail": provenance.tail,
            "exact": provenance.exact,
            "seed": provenance.seed,
        },
        "improvement": {
            "procedure": "two-way-cluster-bootstrap",
            "units": [PARTICIPANT, ITEM],
            "participants": len(
                {unit for fold in seed.folds for unit in fold.participants}
            ),
            "items": len({unit for fold in seed.folds for unit in fold.items}),
            "cells": improvement.units,
            "replicates": improvement.replicates,
            "seed": improvement.seed,
            "level": improvement.level,
            "interval": procedure.interval,
        },
    }


def _failed_entry(seed: FailedSeed, /) -> JsonValue:
    """Return one failed seed's manifest entry."""
    return {"seed": seed.seed, "outcome": "failed", "reason": seed.reason}


class Written(NamedTuple):
    """What one analysis wrote.

    Attributes
    ----------
    rows : int
        Report rows emitted.
    completed : tuple[int, ...]
        Completed run seeds, in document order.
    failed : tuple[int, ...]
        Failed run seeds, in document order.
    manifest : dict[str, JsonValue]
        The manifest written beside the report.
    """

    rows: int
    completed: tuple[int, ...]
    failed: tuple[int, ...]
    manifest: dict[str, JsonValue]


def analyze(source: Path, out: Path, /, *, precision: int | None) -> Written:
    """Run every falsifier of one document and write the report and its tables.

    Parameters
    ----------
    source : Path
        JSON analysis document.
    out : Path
        Directory receiving the report, the tables and the manifest.
    precision : int | None
        Decimal places of rendered floating-point cells, or ``None`` for their
        shortest exact representation.

    Returns
    -------
    Written
        Row count, seed outcomes and the manifest.
    """
    analysis = read_document(source)
    report = out / REPORT_NAME
    rows: list[ReportRow] = []
    selectors: list[ReportSelector] = []
    entries: list[JsonValue] = []
    completed: list[int] = []
    failed: list[int] = []

    for seed in analysis.seeds:
        match seed:
            case CompletedSeed():
                lowered = seed_report(analysis, seed)
                rows.extend(lowered.rows)
                selectors.extend(lowered.selectors)
                entries.append(_completed_entry(seed, lowered, analysis.improvement))
                completed.append(seed.seed)
            case FailedSeed():
                entries.append(_failed_entry(seed))
                failed.append(seed.seed)

    out.mkdir(parents=True, exist_ok=True)
    ParquetReportSink(report).emit(rows)
    selected = select_required_rows(report, selectors)
    for name, renderer in (
        (MARKDOWN_NAME, report_renderer("markdown")),
        (LATEX_NAME, report_renderer("latex")),
    ):
        (out / name).write_bytes(
            render_table(
                renderer, selected, columns=COLUMNS, labels=LABELS, precision=precision
            )
        )
    manifest: dict[str, JsonValue] = {
        "source_path": source.name,
        "source_sha256": sha256(source.read_bytes()).hexdigest(),
        "config_hash": analysis.identity.config_hash,
        "report": REPORT_NAME,
        "tables": [MARKDOWN_NAME, LATEX_NAME],
        "rows": len(selectors),
        "seeds": entries,
    }
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    return Written(len(selectors), tuple(completed), tuple(failed), manifest)


def main(argv: list[str] | None = None) -> int:
    """Run the monotone-scalar falsifiers over one analysis document.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments without the program name, or ``None`` for ``sys.argv``.

    Returns
    -------
    int
        Zero after writing the report.
    """
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("document", type=Path, help="the JSON analysis document")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the directory receiving the report, its tables and the manifest",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=None,
        help="decimal places of rendered floating-point cells",
    )
    arguments = parser.parse_args(argv)
    source: Path = arguments.document
    out: Path = arguments.out
    written = analyze(source, out, precision=arguments.precision)

    sys.stdout.write(
        f"{source}\n"
        f"  rows      {written.rows}\n"
        f"  completed {', '.join(map(str, written.completed)) or 'none'}\n"
        f"  failed    {', '.join(map(str, written.failed)) or 'none'}\n"
        f"  report    {out / REPORT_NAME}\n"
        f"  tables    {out / MARKDOWN_NAME}, {out / LATEX_NAME}\n"
        f"  manifest  {out / MANIFEST_NAME}\n"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
