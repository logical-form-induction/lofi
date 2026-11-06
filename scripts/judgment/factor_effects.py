"""Fits factor-effect rungs and a shuffled-factor control by held-out NUTS.

One declaration names the rungs, the dataset each rung is fitted to, the
held-out split, the NUTS chain settings, the root seed, the factor whose
labels the control shuffles and the criterion the shuffled variance is read
against. Every rung is fitted to the retained rows of every fold, and its
draws predict that fold's held-out rows. Every rung carrying a random
intercept over the shuffled factor is also fitted to every row, and the
draws of that intercept's scale give the posterior variance component.

The control permutes the shuffled factor's level across items, so that each
item keeps one level and each level keeps its number of items. Every rung
reading that factor is then refitted on the same folds, under the same seeds
and chain settings, to the permuted rows; every other input is unchanged.

Every fit's draws and held-out log densities are written beneath the output
directory. The report reads them back and reduces them through the judgment
package's predictive-density and posterior-variance reducers, so each
reported scalar is computed from a persisted fit. The manifest records the
declaration, the provenance of each dataset, the folds, the rung digests, the
chain settings, the seeds, the permutation and the digest of every fit.
"""

import json
import random
import sys
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, cast

import didactic.api as dx
import torch
from safetensors.torch import load_file, save_file

from lofi.judgment import (
    Binary,
    Bind,
    CoverageGate,
    Cumulative,
    DesignCodec,
    Gather,
    HeldOutPredictor,
    JudgmentRow,
    Name,
    NutsInference,
    NutsPlan,
    Operator,
    QuiversNutsFitter,
    ReadDesign,
    ResponseCompiler,
    ResponseSpec,
    RowColumns,
    SplitterFactory,
    posterior_variance_row,
    predictive_density_row,
    predictive_log_density,
    spec_digest,
)
from lofi.registry import Registry
from lofi.run.fits import JudgmentDesigns, RegisteredDesigns
from lofi.spec import HeldOutParticipantPolicy, derive_seed

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.judgment import (
        DesignProvenance,
        Fold,
        JudgmentDesign,
        PosteriorVarianceRow,
        PredictiveDensityRow,
        ResponseProgram,
        Splitter,
    )

REAL: Final = "real"
"""Names the condition fitted to the rows as recorded."""

SHUFFLED: Final = "shuffled"
"""Names the condition fitted to the rows with the factor's labels permuted."""

EVERY_ROW: Final = "all"
"""Names the fit to every row, from which variance components are read."""

HELD_OUT: Final = "held_out_log_density"
"""Names the tensor holding each held-out row's predictive log density."""

DRAWS: Final = "draws."
"""Prefixes the tensor holding one drawn site's posterior draws."""

MANIFEST_NAME: Final = "manifest.json"
"""Names the manifest file beneath the output directory."""

REPORT_NAME: Final = "report.json"
"""Names the report file beneath the output directory."""

FITS_DIRECTORY: Final = "fits"
"""Names the directory holding one file per fit."""

SPLITTER_FAMILY: Final = "splitter"
"""Names the registry family of held-out splitters."""


class RungDeclaration(dx.Model, extra="forbid"):
    """Declares one rung and the dataset it is fitted to.

    Parameters
    ----------
    name : str
        Name the rung is reported under.
    dataset : str
        Registered judgment dataset the rung is fitted to.
    response : ResponseSpec
        Cumulative-link response model of the rung.
    """

    name: str
    dataset: str
    response: ResponseSpec

    __axioms__ = (
        dx.axiom("len(name) > 0", message="a rung is named"),
        dx.axiom("len(dataset) > 0", message="a rung names its dataset"),
    )

    @dx.validates("response")
    @classmethod
    def _response_is_cumulative(cls, value: ResponseSpec) -> ResponseSpec:
        """Return a response whose held-out levels the cumulative link predicts."""
        match value.family:
            case Cumulative():
                return value
            case family:
                msg = (
                    f"held-out prediction scores cumulative-link responses, not "
                    f"the {family.family!r} family"
                )
                raise ValueError(msg)


class HeldOutDeclaration(dx.Model, extra="forbid"):
    """Declares the held-out split every rung is fitted and scored over.

    Parameters
    ----------
    unit : str
        Row column or factor whose levels are held out together.
    splitter : str
        Registered splitter dealing those levels into folds.
    folds : int | None
        Fold count, or None for a splitter holding out one level per fold.
    seed : int
        Seed of the splitter's shuffle of the levels.
    coverage_threshold : float
        Largest fraction of a fold's held-out participants unseen in its
        retained rows admitted without the unseen-participant convention.
    unseen_participant : HeldOutParticipantPolicy
        Convention predicting a participant the retained rows never show.
    samples : int
        Draws of an unseen participant's effects paired with each posterior
        draw under the marginal convention.
    """

    unit: str
    splitter: str
    folds: int | None
    seed: int
    coverage_threshold: float
    unseen_participant: HeldOutParticipantPolicy
    samples: int

    __axioms__ = (
        dx.axiom("len(unit) > 0", message="a held-out split names its unit"),
        dx.axiom("seed >= 0", message="a held-out split's seed is not negative"),
        dx.axiom(
            "coverage_threshold >= 0.0 and coverage_threshold <= 1.0",
            message="a coverage threshold is a proportion",
        ),
        dx.axiom("samples >= 1", message="each posterior draw takes one sample"),
    )

    @dx.model_validator()
    def _splitter_accepts_the_folds(self) -> HeldOutDeclaration:
        """Validate that the registered splitter accepts the declared folds."""
        factory = Registry[SplitterFactory](SPLITTER_FAMILY).get(self.splitter)
        factory(self.unit, folds=self.folds, seed=self.seed)

        return self


class ShuffledControl(dx.Model, extra="forbid"):
    """Declares the factor whose labels the control permutes across items.

    Parameters
    ----------
    factor : str
        Item factor whose levels are permuted.
    seed : int
        Seed of the permutation.
    """

    factor: str
    seed: int

    __axioms__ = (
        dx.axiom("len(factor) > 0", message="a shuffled control names its factor"),
        dx.axiom("seed >= 0", message="a shuffled control's seed is not negative"),
    )


class VarianceCriterion(dx.Model, extra="forbid"):
    """Declares the criterion the posterior variance components are read against.

    The recorded variance is bounded away from zero when the lower end of
    its equal-tail interval lies above the floor. The shuffled variance has
    degenerated when the upper end of its interval lies at or below it.

    Parameters
    ----------
    level : float
        Coverage of every equal-tail posterior interval.
    floor : float
        Variance separating a bounded component from a degenerate one.
    """

    level: float
    floor: float

    __axioms__ = (
        dx.axiom(
            "level > 0.0 and level < 1.0",
            message="an interval's coverage is a proportion",
        ),
        dx.axiom("floor >= 0.0", message="a variance floor is not negative"),
    )


class FrameEffectsDeclaration(dx.Model, extra="forbid"):
    """Declares one factor-effect comparison and its shuffled-factor control.

    Parameters
    ----------
    rungs : tuple[RungDeclaration, ...]
        Rungs fitted, in report order.
    held_out : HeldOutDeclaration
        Held-out split every rung is fitted and scored over.
    inference : NutsInference
        Chain settings every fit runs under.
    seed : int
        Root seed every fit's NUTS seed and prediction seed derive from.
    control : ShuffledControl
        Factor whose labels the control permutes.
    criterion : VarianceCriterion
        Criterion the variance components are read against.
    """

    rungs: tuple[RungDeclaration, ...]
    held_out: HeldOutDeclaration
    inference: NutsInference
    seed: int
    control: ShuffledControl
    criterion: VarianceCriterion

    __axioms__ = (dx.axiom("seed >= 0", message="a root seed is not negative"),)

    @dx.model_validator()
    def _rungs_are_distinct_and_keep_the_folds(self) -> FrameEffectsDeclaration:
        """Validate rung names and the held-out unit against the shuffled factor."""
        names = tuple(rung.name for rung in self.rungs)

        if not names:
            msg = "a factor-effect comparison declares at least one rung"
            raise ValueError(msg)

        if len(frozenset(names)) != len(names):
            msg = f"a factor-effect comparison repeats a rung: {names!r}"
            raise ValueError(msg)

        factor = self.control.factor

        if self.held_out.unit == factor:
            msg = (
                f"the shuffled factor {factor!r} is also the held-out unit, so "
                f"the control would change the folds"
            )
            raise ValueError(msg)

        return self


class Rung(NamedTuple):
    """One compiled rung.

    Attributes
    ----------
    name : str
        Name the rung is reported under.
    dataset : str
        Registered judgment dataset the rung is fitted to.
    digest : str
        Content digest of the rung's response specification.
    program : ResponseProgram
        Compiled response program of the rung.
    reads_factor : bool
        Whether the program indexes a site by the shuffled factor.
    scale : str | None
        Drawn site scaling the random intercept over the shuffled factor, or
        None when the rung has no such intercept.
    """

    name: str
    dataset: str
    digest: str
    program: ResponseProgram
    reads_factor: bool
    scale: str | None


class Comparison(NamedTuple):
    """One validated factor-effect comparison.

    Attributes
    ----------
    rungs : tuple[Rung, ...]
        Compiled rungs, in report order.
    splitter : Splitter
        Splitter dealing the held-out unit's levels into folds.
    gate : CoverageGate
        Admission rule for a fold's participant coverage.
    policy : HeldOutParticipantPolicy
        Convention predicting a participant the retained rows never show.
    samples : int
        Unseen-participant draws paired with each posterior draw.
    warmup : int
        Discarded adaptation iterations per chain.
    retained : int
        Retained draws per chain.
    chains : int
        Independent chains per fit.
    seed : int
        Root seed every fit's seeds derive from.
    factor : str
        Item factor the control permutes.
    permutation_seed : int
        Seed of the permutation.
    level : float
        Coverage of every equal-tail posterior interval.
    floor : float
        Variance separating a bounded component from a degenerate one.
    """

    rungs: tuple[Rung, ...]
    splitter: Splitter
    gate: CoverageGate
    policy: HeldOutParticipantPolicy
    samples: int
    warmup: int
    retained: int
    chains: int
    seed: int
    factor: str
    permutation_seed: int
    level: float
    floor: float


class Dataset(NamedTuple):
    """One dataset read for the comparison.

    Attributes
    ----------
    codec : DesignCodec
        Codec encoding the dataset's rows as designs.
    rows : tuple[JudgmentRow, ...]
        Rows of the dataset, in design order.
    provenance : DesignProvenance
        What the design was read from.
    """

    codec: DesignCodec
    rows: tuple[JudgmentRow, ...]
    provenance: DesignProvenance


class Fit(NamedTuple):
    """One completed fit, as the manifest records it.

    Attributes
    ----------
    condition : str
        Condition the fit belongs to.
    rung : str
        Rung fitted.
    fold : str
        Fold index fitted, or ``EVERY_ROW`` for the fit to every row.
    path : str
        Path of the fit's file, relative to the output directory.
    record : dict[str, JsonValue]
        Seeds, chain settings, diagnostics and digest of the fit.
    """

    condition: str
    rung: str
    fold: str
    path: str
    record: dict[str, JsonValue]


def scale_site(program: ResponseProgram, factor: str, /) -> str | None:
    """Return the drawn scale of a program's random intercept over a factor.

    The intercept is the binding of a drawn scalar times standardized
    offsets gathered by a host column indexing the factor alone.

    Parameters
    ----------
    program : ResponseProgram
        Compiled response program.
    factor : str
        Grouping factor of the intercept.

    Returns
    -------
    str | None
        Site of the scale, or None when no intercept varies over the factor.

    Raises
    ------
    ValueError
        If more than one binding scales offsets gathered by the factor.
    """
    scalars = frozenset(
        draw.site for draw in program.draws if draw.index == (program.scalar_object,)
    )
    found: list[str] = []

    for step in program.body:
        match step:
            case Bind(
                expression=Binary(
                    operator=Operator.MUL,
                    left=Name(identifier=site),
                    right=Gather(index=column),
                )
            ) if site in scalars and program.indices.get(column) == (factor,):
                found.append(site)
            case _:
                pass

    if len(found) > 1:
        msg = f"the program scales {len(found)} intercepts over {factor!r}: {found!r}"
        raise ValueError(msg)

    return found[0] if found else None


def read_comparison(document: bytes, /) -> Comparison:
    """Validate a declaration and compile its rungs.

    Parameters
    ----------
    document : bytes
        JSON declaration of the comparison.

    Raises
    ------
    ValidationError
        If the declaration is malformed.
    ValueError
        If no rung's program, or a rung's program more than once, scales an
        intercept over the shuffled factor.
    """
    declared = FrameEffectsDeclaration.model_validate_json(document)
    compiler = ResponseCompiler.installed()
    factor = declared.control.factor
    rungs: list[Rung] = []

    for rung in declared.rungs:
        program = compiler.compile(rung.response)
        rungs.append(
            Rung(
                name=rung.name,
                dataset=rung.dataset,
                digest=spec_digest(rung.response),
                program=program,
                reads_factor=any(
                    factor in factors for factors in program.indices.values()
                ),
                scale=scale_site(program, factor),
            )
        )

    if all(rung.scale is None for rung in rungs):
        msg = f"no declared rung carries a random intercept over {factor!r}"
        raise ValueError(msg)

    held_out = declared.held_out
    factory = Registry[SplitterFactory](SPLITTER_FAMILY).get(held_out.splitter)
    inference = declared.inference

    return Comparison(
        rungs=tuple(rungs),
        splitter=factory(held_out.unit, folds=held_out.folds, seed=held_out.seed),
        gate=CoverageGate(
            held_out.coverage_threshold, policy=held_out.unseen_participant
        ),
        policy=held_out.unseen_participant,
        samples=held_out.samples,
        warmup=inference.warmup,
        retained=inference.retained,
        chains=inference.chains,
        seed=declared.seed,
        factor=factor,
        permutation_seed=declared.control.seed,
        level=declared.criterion.level,
        floor=declared.criterion.floor,
    )


def _identity(row: JudgmentRow, /) -> tuple[str, str, int, tuple[tuple[str, str], ...]]:
    """Return what a row shares with the same row of another dataset."""
    return (row.participant, row.item, row.response, tuple(sorted(row.factors.items())))


def read_datasets(
    comparison: Comparison, designs: JudgmentDesigns, /
) -> dict[str, Dataset]:
    """Read every dataset a rung names once and require them to share rows.

    Parameters
    ----------
    comparison : Comparison
        Validated comparison.
    designs : JudgmentDesigns
        Resolver of dataset identifiers to designs.

    Raises
    ------
    ValueError
        If a dataset is not an ordinal design, or two datasets differ in a
        row's participant, item, response or factor levels.
    """
    datasets: dict[str, Dataset] = {}

    for name in dict.fromkeys(rung.dataset for rung in comparison.rungs):
        read = designs.read(name)

        match read:
            case ReadDesign(design=design, provenance=provenance):
                codec = DesignCodec(
                    design.response_levels,
                    factors=tuple(design.factors),
                    readouts=tuple(design.readouts),
                    alternatives=design.alternatives_per_row,
                )
                datasets[name] = Dataset(codec, codec.decode(design), provenance)
            case _:
                msg = f"the dataset {name!r} is not an ordinal judgment design"
                raise ValueError(msg)

    first, *others = datasets
    expected = tuple(_identity(row) for row in datasets[first].rows)

    for name in others:
        if tuple(_identity(row) for row in datasets[name].rows) != expected:
            msg = (
                f"the datasets {first!r} and {name!r} do not record the same "
                f"responses, so their rungs cannot share folds"
            )
            raise ValueError(msg)

    return datasets


def permutation(
    rows: Sequence[JudgmentRow], factor: str, *, seed: int
) -> dict[str, tuple[str, str]]:
    """Return each item's recorded and permuted level of a factor.

    The items' levels are shuffled together, so each level keeps the number
    of items it labels.

    Parameters
    ----------
    rows : Sequence[JudgmentRow]
        Rows whose items carry the factor.
    factor : str
        Item factor permuted.
    seed : int
        Seed of the permutation.

    Raises
    ------
    ValueError
        If the rows lack the factor or give one item two of its levels.
    """
    recorded: dict[str, str] = {}

    for row in rows:
        if factor not in row.factors:
            msg = f"a row of item {row.item!r} has no level of {factor!r}"
            raise ValueError(msg)

        level = recorded.setdefault(row.item, row.factors[factor])

        if level != row.factors[factor]:
            msg = (
                f"item {row.item!r} is recorded under the levels {level!r} and "
                f"{row.factors[factor]!r} of {factor!r}, so it is not an item factor"
            )
            raise ValueError(msg)

    labels = list(recorded.values())
    random.Random(seed).shuffle(labels)

    return {
        item: (level, label)
        for (item, level), label in zip(recorded.items(), labels, strict=True)
    }


def shuffled(
    rows: Sequence[JudgmentRow],
    factor: str,
    levels: Mapping[str, tuple[str, str]],
    /,
) -> tuple[JudgmentRow, ...]:
    """Return rows with each item's factor level replaced by its permuted one.

    Parameters
    ----------
    rows : Sequence[JudgmentRow]
        Rows to relabel.
    factor : str
        Item factor permuted.
    levels : Mapping[str, tuple[str, str]]
        Each item's recorded and permuted level.
    """
    return tuple(
        JudgmentRow(
            row.participant,
            row.item,
            row.response,
            RowColumns(
                {**row.factors, factor: levels[row.item][1]},
                dict(row.readouts),
                dict(row.alternative_readouts),
            ),
            row.alternatives,
        )
        for row in rows
    )


def split(
    comparison: Comparison, datasets: Mapping[str, Dataset], /
) -> dict[str, tuple[Fold, ...]]:
    """Return the folds each dataset's rows are dealt into.

    Parameters
    ----------
    comparison : Comparison
        Validated comparison.
    datasets : Mapping[str, Dataset]
        Datasets sharing one row identity.

    Raises
    ------
    ValueError
        If two datasets are dealt into different folds.
    """
    splits = {
        name: comparison.splitter.split(dataset.rows, comparison.gate).folds
        for name, dataset in datasets.items()
    }
    first, *others = splits
    expected = tuple(
        (fold.units, tuple(_identity(row) for row in fold.held_out))
        for fold in splits[first]
    )

    for name in others:
        dealt = tuple(
            (fold.units, tuple(_identity(row) for row in fold.held_out))
            for fold in splits[name]
        )

        if dealt != expected:
            msg = f"the datasets {first!r} and {name!r} are dealt into different folds"
            raise ValueError(msg)

    return splits


def _flat_draws(
    program: ResponseProgram,
    design: JudgmentDesign,
    draws: Mapping[str, torch.Tensor],
    /,
) -> torch.Tensor:
    """Return posterior draws as one parameter vector per chain and draw."""
    layout = program.over(design).layout

    return torch.cat(
        [
            draws[slot.site].reshape(
                draws[slot.site].shape[0] * draws[slot.site].shape[1], slot.size
            )
            for slot in layout
        ],
        dim=-1,
    )


def _fit(
    comparison: Comparison,
    rung: Rung,
    codec: DesignCodec,
    retained: Sequence[JudgmentRow],
    held_out: Sequence[JudgmentRow] | None,
    *,
    condition: str,
    fold: str,
    out: Path,
) -> Fit:
    """Fit one rung by NUTS, score any held-out rows, and persist both."""
    plan = NutsPlan(
        warmup=comparison.warmup,
        retained=comparison.retained,
        chains=comparison.chains,
        seed=derive_seed(comparison.seed, f"nuts:{rung.name}:{fold}"),
    )
    fitted = codec.encode(retained)
    posterior = QuiversNutsFitter(plan).fit(rung.program.over(fitted))
    draws = posterior.draws
    tensors = {f"{DRAWS}{site}": value.contiguous() for site, value in draws.items()}
    record: dict[str, JsonValue] = {
        "warmup": plan.warmup,
        "retained": plan.retained,
        "chains": plan.chains,
        "nuts_seed": plan.seed,
        "rows": len(retained),
    }

    if held_out is not None:
        prediction_seed = derive_seed(comparison.seed, f"prediction:{rung.name}:{fold}")
        target = codec.encode(held_out)
        probabilities = HeldOutPredictor(
            rung.program,
            fitted,
            _flat_draws(rung.program, fitted, draws),
            samples=comparison.samples,
        ).probabilities(
            target,
            policy=comparison.policy,
            generator=torch.Generator().manual_seed(prediction_seed),
        )
        tensors[HELD_OUT] = predictive_log_density(
            probabilities, target.responses
        ).contiguous()
        record["prediction_seed"] = prediction_seed
        record["held_out_rows"] = len(held_out)

    diagnostics = posterior.diagnostics
    relative = f"{FITS_DIRECTORY}/{condition}/{rung.name}/{fold}.safetensors"
    path = out / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path)
    record["divergences"] = diagnostics.divergences
    record["max_rhat"] = diagnostics.max_rhat
    record["min_ess"] = diagnostics.min_ess
    record["sha256"] = sha256(path.read_bytes()).hexdigest()

    return Fit(condition, rung.name, fold, relative, record)


def _relabeled(
    rows: Sequence[JudgmentRow],
    factor: str,
    levels: Mapping[str, tuple[str, str]] | None,
    /,
) -> Sequence[JudgmentRow]:
    """Return rows as recorded, or permuted when the levels are given."""
    return rows if levels is None else shuffled(rows, factor, levels)


def fit_all(
    comparison: Comparison,
    datasets: Mapping[str, Dataset],
    folds: Mapping[str, tuple[Fold, ...]],
    levels: Mapping[str, tuple[str, str]],
    out: Path,
    /,
) -> tuple[Fit, ...]:
    """Fit every rung of both conditions and persist every fit.

    Each rung is fitted to every fold's retained rows and, when it carries
    the intercept over the shuffled factor, to every row. The shuffled
    condition refits each rung reading the factor, under the same folds and
    seeds, to rows whose items carry their permuted level.

    Parameters
    ----------
    comparison : Comparison
        Validated comparison.
    datasets : Mapping[str, Dataset]
        Datasets sharing one row identity.
    folds : Mapping[str, tuple[Fold, ...]]
        Folds of each dataset's rows.
    levels : Mapping[str, tuple[str, str]]
        Each item's recorded and permuted level of the shuffled factor.
    out : Path
        Output directory.
    """
    fits: list[Fit] = []
    factor = comparison.factor

    for condition in (REAL, SHUFFLED):
        for rung in comparison.rungs:
            if condition == SHUFFLED and not rung.reads_factor:
                continue

            dataset = datasets[rung.dataset]
            relabel = levels if condition == SHUFFLED else None

            fits.extend(
                _fit(
                    comparison,
                    rung,
                    dataset.codec,
                    _relabeled(fold.retained, factor, relabel),
                    _relabeled(fold.held_out, factor, relabel),
                    condition=condition,
                    fold=str(fold.index),
                    out=out,
                )
                for fold in folds[rung.dataset]
            )

            if rung.scale is not None:
                fits.append(
                    _fit(
                        comparison,
                        rung,
                        dataset.codec,
                        _relabeled(dataset.rows, factor, relabel),
                        None,
                        condition=condition,
                        fold=EVERY_ROW,
                        out=out,
                    )
                )

    return tuple(fits)


class Verdict(NamedTuple):
    """The criterion applied to one rung's two variance components.

    Attributes
    ----------
    rung : str
        Rung whose components are read.
    bounded : bool
        Whether the recorded component's interval lies above the floor.
    degenerate : bool
        Whether the shuffled component's interval lies at or below it.
    """

    rung: str
    bounded: bool
    degenerate: bool


class Reduction(NamedTuple):
    """Every reported row of one condition.

    Attributes
    ----------
    densities : tuple[PredictiveDensityRow, ...]
        Held-out predictive density of each rung fitted in the condition.
    variances : tuple[PosteriorVarianceRow, ...]
        Posterior variance of each intercept over the shuffled factor.
    """

    densities: tuple[PredictiveDensityRow, ...]
    variances: tuple[PosteriorVarianceRow, ...]


def reduce(
    comparison: Comparison, fits: Sequence[Fit], condition: str, out: Path, /
) -> Reduction:
    """Reduce one condition's persisted fits to its report rows.

    Each rung's held-out densities are read from every fold's file and
    reduced together. Each variance component is the square of the
    intercept scale drawn by the rung's fit to every row.

    Parameters
    ----------
    comparison : Comparison
        Validated comparison.
    fits : Sequence[Fit]
        Completed fits.
    condition : str
        Condition reduced.
    out : Path
        Output directory the fits were written beneath.
    """
    densities: list[PredictiveDensityRow] = []
    variances: list[PosteriorVarianceRow] = []

    for rung in comparison.rungs:
        held_out = tuple(
            fit
            for fit in fits
            if (fit.condition, fit.rung) == (condition, rung.name)
            and fit.fold != EVERY_ROW
        )

        if not held_out:
            continue

        densities.append(
            predictive_density_row(
                rung.name,
                torch.cat(
                    tuple(load_file(out / fit.path)[HELD_OUT] for fit in held_out)
                ),
            )
        )

        if rung.scale is None:
            continue

        every = next(
            fit
            for fit in fits
            if (fit.condition, fit.rung, fit.fold) == (condition, rung.name, EVERY_ROW)
        )
        scale = load_file(out / every.path)[f"{DRAWS}{rung.scale}"]
        variances.append(
            posterior_variance_row(
                rung.name,
                comparison.factor,
                scale.reshape(-1).square(),
                level=comparison.level,
            )
        )

    return Reduction(tuple(densities), tuple(variances))


def verdicts(
    real: Sequence[PosteriorVarianceRow],
    control: Sequence[PosteriorVarianceRow],
    *,
    floor: float,
) -> tuple[Verdict, ...]:
    """Apply the variance criterion to each rung's recorded and shuffled rows.

    Parameters
    ----------
    real : Sequence[PosteriorVarianceRow]
        Variance components of the recorded rows.
    control : Sequence[PosteriorVarianceRow]
        Variance components of the shuffled rows, for the same rungs.
    floor : float
        Variance separating a bounded component from a degenerate one.

    Raises
    ------
    ValueError
        If the two conditions report different rungs, or a component has no
        upper interval bound.
    """
    if tuple(row.fit for row in real) != tuple(row.fit for row in control):
        msg = "the recorded and shuffled conditions report different rungs"
        raise ValueError(msg)

    judged: list[Verdict] = []

    for recorded, permuted in zip(real, control, strict=True):
        lower = recorded.posterior_variance.lower
        upper = permuted.posterior_variance.upper

        if lower is None or upper is None:
            msg = f"the variance components of {recorded.fit!r} have no interval"
            raise ValueError(msg)

        judged.append(Verdict(recorded.fit, lower > floor, upper <= floor))

    return tuple(judged)


def _json(model: dx.Model, /) -> JsonValue:
    """Return a validated model as a JSON value."""
    return cast("JsonValue", json.loads(model.model_dump_json()))


def _rows(reduction: Reduction, /) -> dict[str, JsonValue]:
    """Return one condition's rows as JSON values."""
    return {
        "predictive_density": [_json(row) for row in reduction.densities],
        "posterior_variance": [_json(row) for row in reduction.variances],
    }


def report(
    comparison: Comparison, fits: Sequence[Fit], out: Path, /
) -> dict[str, JsonValue]:
    """Return the report of both conditions and the criterion applied to them.

    Parameters
    ----------
    comparison : Comparison
        Validated comparison.
    fits : Sequence[Fit]
        Completed fits.
    out : Path
        Output directory the fits were written beneath.
    """
    real = reduce(comparison, fits, REAL, out)
    control = reduce(comparison, fits, SHUFFLED, out)
    judged = verdicts(real.variances, control.variances, floor=comparison.floor)

    return {
        "factor": comparison.factor,
        "criterion": {
            "level": comparison.level,
            "floor": comparison.floor,
            "rule": (
                "the recorded component is bounded when its interval's lower end "
                "exceeds the floor; the shuffled component has degenerated when "
                "its interval's upper end is at most the floor"
            ),
        },
        REAL: _rows(real),
        SHUFFLED: _rows(control),
        "verdicts": [
            {
                "rung": verdict.rung,
                "bounded": verdict.bounded,
                "degenerate": verdict.degenerate,
            }
            for verdict in judged
        ],
        "criterion_met": all(
            verdict.bounded and verdict.degenerate for verdict in judged
        ),
    }


def manifest(
    document: bytes,
    comparison: Comparison,
    datasets: Mapping[str, Dataset],
    folds: Mapping[str, tuple[Fold, ...]],
    levels: Mapping[str, tuple[str, str]],
    fits: Sequence[Fit],
    /,
) -> dict[str, JsonValue]:
    """Return everything needed to reproduce the fits.

    Parameters
    ----------
    document : bytes
        Declaration the comparison was read from.
    comparison : Comparison
        Validated comparison.
    datasets : Mapping[str, Dataset]
        Datasets read.
    folds : Mapping[str, tuple[Fold, ...]]
        Folds of each dataset's rows.
    levels : Mapping[str, tuple[str, str]]
        Each item's recorded and permuted level of the shuffled factor.
    fits : Sequence[Fit]
        Completed fits.
    """
    first = next(iter(folds.values()))

    return {
        "declaration_sha256": sha256(document).hexdigest(),
        "declaration": cast("JsonValue", json.loads(document)),
        "datasets": {
            name: _json(dataset.provenance) for name, dataset in datasets.items()
        },
        "folds": [
            {
                "index": fold.index,
                "units": [*sorted(fold.units)],
                "retained": len(fold.retained),
                "held_out": len(fold.held_out),
                "policy": None if fold.policy is None else fold.policy.value,
            }
            for fold in first
        ],
        "rungs": [
            {
                "name": rung.name,
                "dataset": rung.dataset,
                "spec_digest": rung.digest,
                "scale": rung.scale,
            }
            for rung in comparison.rungs
        ],
        "permutation": {
            "factor": comparison.factor,
            "seed": comparison.permutation_seed,
            "items": {
                item: {"recorded": recorded, "permuted": permuted}
                for item, (recorded, permuted) in levels.items()
            },
        },
        "fits": [
            {
                "condition": fit.condition,
                "rung": fit.rung,
                "fold": fit.fold,
                "path": fit.path,
                **fit.record,
            }
            for fit in fits
        ],
    }


def run(document: bytes, out: Path, /, *, designs: JudgmentDesigns) -> Path:
    """Fit, persist and report one declared factor-effect comparison.

    Parameters
    ----------
    document : bytes
        JSON declaration of the comparison.
    out : Path
        Output directory, created if absent.
    designs : JudgmentDesigns
        Resolver of dataset identifiers to designs.

    Returns
    -------
    Path
        The written report.
    """
    comparison = read_comparison(document)
    datasets = read_datasets(comparison, designs)
    folds = split(comparison, datasets)
    levels = permutation(
        next(iter(datasets.values())).rows,
        comparison.factor,
        seed=comparison.permutation_seed,
    )
    out.mkdir(parents=True, exist_ok=True)
    fits = fit_all(comparison, datasets, folds, levels, out)
    written = report(comparison, fits, out)
    recorded = manifest(document, comparison, datasets, folds, levels, fits)
    (out / MANIFEST_NAME).write_text(
        json.dumps(recorded, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    destination = out / REPORT_NAME
    destination.write_text(
        json.dumps(written, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    return destination


def main(argv: list[str] | None = None) -> int:
    """Run one declared factor-effect comparison over the registered datasets.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments without the program name, or ``None`` for ``sys.argv``.

    Returns
    -------
    int
        Zero after writing the manifest and the report.
    """
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "declaration", type=Path, help="the JSON declaration of the comparison"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="the directory to write beneath"
    )
    arguments = parser.parse_args(argv)
    declaration: Path = arguments.declaration
    out: Path = arguments.out

    destination = run(declaration.read_bytes(), out, designs=RegisteredDesigns())
    sys.stdout.write(f"{declaration}\n  report   {destination}\n")
    sys.stdout.write(f"  manifest {out / MANIFEST_NAME}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
