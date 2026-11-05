"""Prior-predictive calibration of a compiled response program.

A replicate draws one parameter vector from the priors the program declares,
samples one response per row of the design at it, and fits the sampled design
three ways over the same compiled program: by genuine NUTS, and by Quivers'
structured and mean-field guides. A fit that meets a numerical condition is
recorded against its replicate and method, and that replicate is left out of
the reduction. Every replicate fitted by all three methods is then reduced
once by :func:`calibration_report`.

The truth of a replicate and every fitted draw are read through one parameter
layout, so a replicate's blocks are the program's drawn sites. Its cutpoints
are the participant-by-threshold table the program observes each row under,
read from the location and thresholds :func:`latent` interprets, so the
program must draw one row of thresholds per participant. A caller checks that
once with :func:`ensure_participant_thresholds` before calibrating.
"""

import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, final

import torch
from quivers.inference import ELBO, SVI
from quivers.inference.guides import AutoNormalGuide, AutoStructured

from lofi._checks import ensure_finite, ensure_non_negative, ensure_positive
from lofi._frozen import FrozenValue
from lofi.judgment._nuts import QuiversNutsFitter
from lofi.judgment._ports import PosteriorFittingError
from lofi.judgment._results import NutsPlan
from lofi.judgment._spec import GUIDE_METHODS, InferenceMethod
from lofi.judgment.backends import latent, sample_prior
from lofi.judgment.calibration import CalibrationReplicate, calibration_report
from lofi.judgment.compiler import PARTICIPANT
from lofi.judgment.runtime import QuiversRuntime
from lofi.judgment.simulate import sample_responses, simulated_design
from lofi.spec import derive_seed

if TYPE_CHECKING:
    from collections.abc import Mapping

    from quivers.continuous.programs import MonadicProgram
    from quivers.inference.guides import Guide
    from torch import Tensor

    from lofi.judgment._design import JudgmentDesign
    from lofi.judgment._results import PosteriorFit
    from lofi.judgment.calibration import CalibrationReport
    from lofi.judgment.compiler import BoundProgram, ResponseProgram

__all__ = [
    "CalibrationCampaign",
    "CalibrationSeeds",
    "GuideFit",
    "NutsDiagnostics",
    "QuiversGuideFitter",
    "ReplicateRecord",
    "ensure_participant_thresholds",
    "participant_cutpoints",
    "prior_predictive_calibration",
]

_GUIDES: Final = (InferenceMethod.STRUCTURED, InferenceMethod.MEAN_FIELD)
"""The guides every replicate is fitted by, in recording order."""


def _guide(
    method: InferenceMethod,
    model: MonadicProgram,
    program: BoundProgram,
    observed: set[str],
) -> Guide:
    """Return the Quivers guide a method names over a program's sites.

    The mean-field guide gives every site an independent Gaussian. The
    structured guide holds every site indexed by no grouping factor in one
    joint Gaussian and gives each grouped site a Gaussian whose location
    follows every ungrouped site drawn before it linearly, so the effects of
    each level are Gaussian given the shared parameters rather than
    independent of them.
    """
    if method is InferenceMethod.MEAN_FIELD:
        return AutoNormalGuide(model, observed)

    grouped = {
        draw.site
        for draw in program.program.draws
        if any(name in program.program.objects for name in draw.index)
    }
    shared: list[str] = []
    dependencies: dict[str, dict[str, str]] = {}

    for draw in program.program.draws:
        if draw.site in grouped:
            dependencies[draw.site] = dict.fromkeys(shared, "linear")
        else:
            shared.append(draw.site)

    return AutoStructured(
        model,
        observed,
        conditionals={
            draw.site: "normal" if draw.site in grouped else "mvn"
            for draw in program.program.draws
        },
        dependencies=dependencies,
    )


def _flat(program: BoundProgram, sites: Mapping[str, Tensor]) -> Tensor:
    """Return one draw of every site laid out as the program lays them out."""
    return torch.cat(
        [
            sites[slot.site].reshape(-1).to(dtype=torch.float64)
            for slot in program.layout
        ]
    )


class GuideFit(NamedTuple):
    """Draws from a fitted variational guide and the bound it was fitted to."""

    #: One chain of draws, over chains, draws and then the program's parameter
    #: layout, in double precision.
    draws: Tensor

    #: The evidence lower bound estimated at the last optimization step.
    evidence_bound: float


@final
class QuiversGuideFitter(FrozenValue):
    """Fit a bound response program with a Quivers variational guide.

    The guide's evidence lower bound is maximized by Adam at a declared
    learning rate for a declared number of steps, and draws are then taken
    from the fitted guide. Every
    draw descends from the fitter's seed through a private stream, so the
    process's global stream is left as it was.

    Parameters
    ----------
    method : InferenceMethod
        The structured or the mean-field guide.
    iterations : int
        Optimization steps the guide is fitted for.
    draws : int
        Draws taken from the fitted guide.
    learning_rate : float
        Adam step size every optimization step takes.
    seed : int
        Seed of the stream the fit and its draws descend from.

    Raises
    ------
    ValueError
        If the method is not a guide this fitter builds, a count is not
        positive, the learning rate is not positive and finite, or the seed
        is negative.
    """

    _noun = "Quivers guide fitter"

    __slots__ = ("_draws", "_iterations", "_learning_rate", "_method", "_seed")

    def __init__(
        self,
        method: InferenceMethod,
        *,
        iterations: int,
        draws: int,
        learning_rate: float,
        seed: int,
    ) -> None:
        if method not in GUIDE_METHODS:
            msg = (
                f"a Quivers guide fitter builds the structured or the mean-field "
                f"guide, not {method.value!r}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_method", method)
        object.__setattr__(
            self, "_iterations", ensure_positive("guide iterations", iterations)
        )
        rate = ensure_finite("a guide's learning rate", learning_rate)

        if rate <= 0.0:
            msg = f"a guide's learning rate is positive, not {learning_rate!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_draws", ensure_positive("guide draws", draws))
        object.__setattr__(self, "_learning_rate", rate)
        object.__setattr__(self, "_seed", ensure_non_negative("guide seed", seed))

    @property
    def method(self) -> InferenceMethod:
        """Return the inference a fit under this guide is recorded as."""
        return self._method

    @property
    def iterations(self) -> int:
        """Return the optimization steps the guide is fitted for."""
        return self._iterations

    @property
    def draws(self) -> int:
        """Return the draws taken from the fitted guide."""
        return self._draws

    @property
    def learning_rate(self) -> float:
        """Return the Adam step size every optimization step takes."""
        return self._learning_rate

    @property
    def seed(self) -> int:
        """Return the seed the fit and its draws descend from."""
        return self._seed

    def fit(self, program: BoundProgram, /) -> GuideFit:
        """Return draws from the guide fitted to a bound response program.

        A numerical condition met while the guide is fitted or drawn from is a
        fitting failure: a distribution refusing a non-finite parameter, which
        PyTorch reports as a ``ValueError``, or a covariance that is not
        positive definite, which it reports as a ``LinAlgError``. An exhausted
        accelerator is reported as a ``MemoryError``. Any other error,
        including one raised while the guide is built, is a defect and
        propagates.

        Parameters
        ----------
        program : BoundProgram
            Compiled response program bound to its observed design.

        Returns
        -------
        GuideFit
            One chain of draws over the program's parameter layout, and the
            evidence lower bound estimated at the last step.

        Raises
        ------
        PosteriorFittingError
            If fitting or drawing meets a numerical condition, the bound is
            not finite, or a draw is not finite.
        MemoryError
            If the accelerator runs out of memory.
        """
        model = QuiversRuntime().program(program)
        observations = dict(program.host_data)
        anchor = torch.zeros((1, 1), dtype=torch.float64)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._seed)
            guide = _guide(self._method, model, program, set(observations))
            optimizer = torch.optim.Adam(guide.parameters(), lr=self._learning_rate)
            inference = SVI(model, guide, optimizer, ELBO())

            try:
                losses = [
                    inference.step(anchor, observations)
                    for _ in range(self._iterations)
                ]

                with torch.no_grad():
                    draws = torch.stack(
                        [
                            _flat(program, guide.rsample(anchor))
                            for _ in range(self._draws)
                        ]
                    )

            except torch.OutOfMemoryError as error:
                raise MemoryError(str(error)) from error

            except (ValueError, torch.linalg.LinAlgError) as error:
                msg = f"{self._method.value} guide fitting failed: {error}"
                raise PosteriorFittingError(msg) from error

        if not all(math.isfinite(loss) for loss in losses):
            msg = (
                f"the {self._method.value} guide's bound is not finite: {losses[-1]!r}"
            )
            raise PosteriorFittingError(msg)

        if not bool(torch.isfinite(draws).all()):
            msg = f"the fitted {self._method.value} guide drew a nonfinite value"
            raise PosteriorFittingError(msg)

        return GuideFit(draws.unsqueeze(0), -losses[-1])


@final
class CalibrationSeeds(FrozenValue):
    """The streams every seed of a prior-predictive calibration descends from.

    A replicate's simulation seed fixes its parameter draw and its sampled
    responses, and each inference method takes its own seed for each
    replicate. The rank seed initializes the reducer's tie-breaking stream.

    Parameters
    ----------
    simulation : int
        Seed every replicate's simulation seed is derived from.
    inference : int
        Seed every replicate's seed for each inference method is derived from.
    rank : int
        Seed of the reducer's randomized-rank stream.

    Raises
    ------
    ValueError
        If a seed is negative.
    """

    _noun = "calibration seeds"

    __slots__ = ("_inference", "_rank", "_simulation")

    def __init__(self, *, simulation: int, inference: int, rank: int) -> None:
        object.__setattr__(
            self, "_simulation", ensure_non_negative("simulation seed", simulation)
        )
        object.__setattr__(
            self, "_inference", ensure_non_negative("inference seed", inference)
        )
        object.__setattr__(self, "_rank", ensure_non_negative("rank seed", rank))

    @property
    def rank(self) -> int:
        """Return the seed of the reducer's randomized-rank stream."""
        return self._rank

    def simulation(self, replicate: int, /) -> int:
        """Return the seed one replicate's parameters and responses are drawn by.

        Parameters
        ----------
        replicate : int
            Zero-based replicate position.
        """
        return derive_seed(self._simulation, f"replicate:{replicate}")

    def inference(self, method: InferenceMethod, replicate: int, /) -> int:
        """Return the seed one method fits one replicate under.

        Parameters
        ----------
        method : InferenceMethod
            The inference method.
        replicate : int
            Zero-based replicate position.
        """
        return derive_seed(self._inference, f"{method.value}:{replicate}")


def ensure_participant_thresholds(compiled: ResponseProgram, /) -> None:
    """Refuse a program whose thresholds do not vary over participants.

    Calibration recovers each participant's cutpoints, which only a program
    drawing one row of thresholds per participant has.

    Parameters
    ----------
    compiled : ResponseProgram
        Compiled response program.

    Raises
    ------
    ValueError
        If the program's thresholds do not vary over participants.
    """
    if compiled.threshold_group != PARTICIPANT:
        msg = (
            "calibration recovers each participant's cutpoints, and this "
            "program's thresholds do not vary over participants"
        )
        raise ValueError(msg)


def participant_cutpoints(
    program: BoundProgram, design: JudgmentDesign, params: Tensor, /
) -> Tensor:
    """Return the cutpoints a program gives each participant at a parameter vector.

    Every row of a participant is observed under that participant's row of
    cutpoints, so each participant's row is read from the rows they judged.

    Parameters
    ----------
    program : BoundProgram
        Program bound to the design.
    design : JudgmentDesign
        Design the program is bound to, whose participant column places
        each row.
    params : Tensor
        Value of every drawn site, laid out as the program lays them out.

    Returns
    -------
    Tensor
        One row per participant level and one column per threshold.
    """
    thresholds = latent(program, params).thresholds
    table = thresholds.new_zeros((len(design.participant_levels), thresholds.shape[1]))

    return table.index_copy(0, design.participants, thresholds)


def _blocks(program: BoundProgram, draws: Tensor) -> dict[str, Tensor]:
    """Return chain-and-draw draws of each site from flat laid-out draws."""
    pieces = torch.split(draws, [slot.size for slot in program.layout], dim=-1)

    return {
        slot.site: piece
        if not slot.shape
        else piece.reshape(*draws.shape[:2], *slot.shape)
        for slot, piece in zip(program.layout, pieces, strict=True)
    }


def _cutpoint_draws(
    program: BoundProgram, design: JudgmentDesign, draws: Tensor
) -> Tensor:
    """Return each draw's participant cutpoints, over chains and draws."""
    chains, count = draws.shape[0], draws.shape[1]
    flat = draws.reshape(chains * count, -1)
    tables = torch.stack(
        [participant_cutpoints(program, design, vector) for vector in flat]
    )

    return tables.reshape(chains, count, *tables.shape[1:])


def _nuts_draws(program: BoundProgram, posterior: PosteriorFit) -> Tensor:
    """Return NUTS draws over chains, draws and the program's parameter layout."""
    sites = posterior.draws
    chains, count = posterior.chains, posterior.retained

    return torch.cat(
        [
            sites[slot.site].reshape(chains, count, -1).to(dtype=torch.float64)
            for slot in program.layout
        ],
        dim=-1,
    )


@final
class NutsDiagnostics(FrozenValue):
    """The convergence diagnostics of one multi-chain NUTS fit.

    Parameters
    ----------
    posterior : PosteriorFit
        The NUTS fit the diagnostics are read from.
    """

    _noun = "NUTS diagnostics"

    __slots__ = (
        "_acceptance_rates",
        "_divergences",
        "_effective_sample_sizes",
        "_r_hat",
    )

    def __init__(self, posterior: PosteriorFit, /) -> None:
        object.__setattr__(
            self,
            "_divergences",
            tuple(int(count) for count in posterior.divergence_counts.tolist()),
        )
        object.__setattr__(
            self,
            "_acceptance_rates",
            tuple(float(rate) for rate in posterior.acceptance_rates.tolist()),
        )
        object.__setattr__(
            self,
            "_r_hat",
            MappingProxyType(
                {
                    site: float(values.max().item())
                    for site, values in posterior.r_hat.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "_effective_sample_sizes",
            MappingProxyType(
                {
                    site: float(values.min().item())
                    for site, values in posterior.effective_sample_sizes.items()
                }
            ),
        )

    @property
    def divergences(self) -> tuple[int, ...]:
        """Return the divergent transitions of each chain, in chain order."""
        return self._divergences

    @property
    def acceptance_rates(self) -> tuple[float, ...]:
        """Return the mean acceptance rate of each chain, in chain order."""
        return self._acceptance_rates

    @property
    def r_hat(self) -> Mapping[str, float]:
        """Return the largest R-hat over each drawn site's values, by site."""
        return self._r_hat

    @property
    def effective_sample_sizes(self) -> Mapping[str, float]:
        """Return the smallest bulk effective sample size of each site, by site."""
        return self._effective_sample_sizes

    @property
    def max_rhat(self) -> float:
        """Return the largest R-hat over every drawn value."""
        return max(self._r_hat.values())

    @property
    def min_ess(self) -> float:
        """Return the smallest bulk effective sample size over every drawn value."""
        return min(self._effective_sample_sizes.values())


@final
class ReplicateRecord(FrozenValue):
    """What fitting one prior-predictive replicate by every method produced.

    Each method either fitted the replicate, leaving NUTS's diagnostics or a
    guide's evidence lower bound, or met a numerical condition, leaving its
    message. Only a replicate every method fitted is reduced.

    Parameters
    ----------
    replicate : int
        Zero-based replicate position.
    seed : int
        Seed the replicate's parameters and responses were drawn by.
    nuts : NutsDiagnostics | None
        Diagnostics of the NUTS fit, or None if NUTS failed.
    evidence_bounds : Mapping[InferenceMethod, float]
        Final evidence lower bound of each guide that fitted the replicate.
    failures : Mapping[InferenceMethod, str]
        Message of each method that failed to fit the replicate.

    Raises
    ------
    ValueError
        If a position or seed is negative, a bound is not finite, or the
        methods are not each recorded exactly once as fitted or failed.
    """

    _noun = "replicate record"

    __slots__ = ("_evidence_bounds", "_failures", "_nuts", "_replicate", "_seed")

    def __init__(
        self,
        *,
        replicate: int,
        seed: int,
        nuts: NutsDiagnostics | None,
        evidence_bounds: Mapping[InferenceMethod, float],
        failures: Mapping[InferenceMethod, str],
    ) -> None:
        fitted = {
            *evidence_bounds,
            *(() if nuts is None else (InferenceMethod.NUTS,)),
        }

        if fitted & set(failures) or fitted | set(failures) != {
            InferenceMethod.NUTS,
            *_GUIDES,
        }:
            msg = (
                "a replicate record holds each method once, as fitted or failed, "
                f"and this one fitted {sorted(method.value for method in fitted)!r} "
                f"and failed {sorted(method.value for method in failures)!r}"
            )
            raise ValueError(msg)

        object.__setattr__(
            self, "_replicate", ensure_non_negative("replicate position", replicate)
        )
        object.__setattr__(self, "_seed", ensure_non_negative("replicate seed", seed))
        object.__setattr__(self, "_nuts", nuts)
        object.__setattr__(
            self,
            "_evidence_bounds",
            MappingProxyType(
                {
                    method: ensure_finite(
                        "an evidence lower bound", evidence_bounds[method]
                    )
                    for method in _GUIDES
                    if method in evidence_bounds
                }
            ),
        )
        object.__setattr__(
            self,
            "_failures",
            MappingProxyType(
                {
                    method: failures[method]
                    for method in (InferenceMethod.NUTS, *_GUIDES)
                    if method in failures
                }
            ),
        )

    @property
    def replicate(self) -> int:
        """Return the zero-based replicate position."""
        return self._replicate

    @property
    def seed(self) -> int:
        """Return the seed the replicate's parameters and responses were drawn by."""
        return self._seed

    @property
    def nuts(self) -> NutsDiagnostics | None:
        """Return the NUTS fit's diagnostics, or None if NUTS failed."""
        return self._nuts

    @property
    def evidence_bounds(self) -> Mapping[InferenceMethod, float]:
        """Return the final evidence lower bound of each guide that fitted."""
        return self._evidence_bounds

    @property
    def failures(self) -> Mapping[InferenceMethod, str]:
        """Return the message of each method that failed, in method order."""
        return self._failures

    @property
    def retained(self) -> bool:
        """Return whether every method fitted the replicate."""
        return not self._failures


@final
class CalibrationCampaign(FrozenValue):
    """Every replicate of a prior-predictive calibration and their reduction.

    Parameters
    ----------
    records : tuple[ReplicateRecord, ...]
        Each replicate's record, in replicate order.
    report : CalibrationReport | None
        The reduction of every retained replicate, or None if no replicate
        was retained.
    """

    _noun = "calibration campaign"

    __slots__ = ("_records", "_report")

    def __init__(
        self, records: tuple[ReplicateRecord, ...], report: CalibrationReport | None
    ) -> None:
        if not records:
            msg = "a calibration campaign records at least one replicate"
            raise ValueError(msg)

        if [record.replicate for record in records] != list(range(len(records))):
            msg = "a calibration campaign records its replicates in order from zero"
            raise ValueError(msg)

        if (report is None) != (not any(record.retained for record in records)):
            msg = "a calibration campaign reduces exactly when it retains a replicate"
            raise ValueError(msg)

        object.__setattr__(self, "_records", records)
        object.__setattr__(self, "_report", report)

    @property
    def records(self) -> tuple[ReplicateRecord, ...]:
        """Return each replicate's record, in replicate order."""
        return self._records

    @property
    def report(self) -> CalibrationReport | None:
        """Return the reduction of the retained replicates, or None if none was."""
        return self._report

    @property
    def failed(self) -> tuple[ReplicateRecord, ...]:
        """Return the record of each replicate some method failed to fit."""
        return tuple(record for record in self._records if not record.retained)


def _replicate(
    compiled: ResponseProgram,
    design: JudgmentDesign,
    replicate: int,
    *,
    seeds: CalibrationSeeds,
    plan: NutsPlan,
    guide_iterations: int,
    guide_learning_rate: float,
) -> tuple[ReplicateRecord, CalibrationReplicate | None]:
    """Simulate one replicate from the prior and fit it by every method."""
    seed = seeds.simulation(replicate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    unobserved = compiled.over(design)
    truth = sample_prior(unobserved, generator=generator)
    responses = sample_responses(unobserved, truth, generator=generator)
    sampled = simulated_design(design, responses)
    program = compiled.over(sampled)
    draws: dict[InferenceMethod, Tensor] = {}
    failures: dict[InferenceMethod, str] = {}
    bounds: dict[InferenceMethod, float] = {}
    diagnostics: NutsDiagnostics | None = None

    try:
        posterior = QuiversNutsFitter(
            NutsPlan(
                warmup=plan.warmup,
                retained=plan.retained,
                chains=plan.chains,
                seed=seeds.inference(InferenceMethod.NUTS, replicate),
            )
        ).fit(program)

    except PosteriorFittingError as error:
        failures[InferenceMethod.NUTS] = str(error)

    else:
        diagnostics = NutsDiagnostics(posterior)
        draws[InferenceMethod.NUTS] = _nuts_draws(program, posterior)

    for method in _GUIDES:
        fitter = QuiversGuideFitter(
            method,
            iterations=guide_iterations,
            draws=plan.chains * plan.retained,
            learning_rate=guide_learning_rate,
            seed=seeds.inference(method, replicate),
        )

        try:
            fitted = fitter.fit(program)

        except PosteriorFittingError as error:
            failures[method] = str(error)

        else:
            bounds[method] = fitted.evidence_bound
            draws[method] = fitted.draws

    record = ReplicateRecord(
        replicate=replicate,
        seed=seed,
        nuts=diagnostics,
        evidence_bounds=bounds,
        failures=failures,
    )

    if failures:
        return record, None

    return record, CalibrationReplicate(
        seed=seed,
        truths=program.sites(truth),
        posterior_draws={
            method: _blocks(program, values) for method, values in draws.items()
        },
        true_cutpoints=participant_cutpoints(program, sampled, truth),
        cutpoint_draws={
            method: _cutpoint_draws(program, sampled, values)
            for method, values in draws.items()
        },
    )


def prior_predictive_calibration(
    compiled: ResponseProgram,
    design: JudgmentDesign,
    /,
    *,
    replicates: int,
    guide_iterations: int,
    guide_learning_rate: float,
    plan: NutsPlan,
    seeds: CalibrationSeeds,
) -> CalibrationCampaign:
    """Return the calibration of a response program over prior-predictive replicates.

    Each replicate is simulated over the design's rows from the program's
    priors, then fitted by NUTS under the plan's warmup, retained draws and
    chains, and by the structured and mean-field guides, each drawing as many
    values as NUTS retains. Every seed descends from ``seeds``; the plan's
    own seed is not used. The program's thresholds must vary over
    participants, as :func:`ensure_participant_thresholds` checks.

    A fit meeting a numerical condition is recorded against its replicate
    and method, and the other methods still fit that replicate. The
    replicates every method fitted are reduced once; whether each NUTS
    cutpoint clears the recovery floor is part of that reduction.

    Parameters
    ----------
    compiled : ResponseProgram
        Compiled response program.
    design : JudgmentDesign
        Design whose rows every replicate is simulated over.
    replicates : int
        Prior-predictive replicates.
    guide_iterations : int
        Optimization steps each guide is fitted for.
    guide_learning_rate : float
        Adam step size each guide is fitted at.
    plan : NutsPlan
        Warmup, retained draws and chains of every NUTS fit.
    seeds : CalibrationSeeds
        Streams every simulation, inference and rank seed descends from.

    Returns
    -------
    CalibrationCampaign
        Every replicate's record and the reduction of those retained.

    Raises
    ------
    ValueError
        If fewer than one replicate is asked for.
    MemoryError
        If the accelerator runs out of memory during a fit.
    """
    count = ensure_positive("calibration replicates", replicates)
    fitted = [
        _replicate(
            compiled,
            design,
            replicate,
            seeds=seeds,
            plan=plan,
            guide_iterations=guide_iterations,
            guide_learning_rate=guide_learning_rate,
        )
        for replicate in range(count)
    ]
    retained = [replicate for _, replicate in fitted if replicate is not None]

    return CalibrationCampaign(
        tuple(record for record, _ in fitted),
        calibration_report(retained, rank_seed=seeds.rank) if retained else None,
    )
