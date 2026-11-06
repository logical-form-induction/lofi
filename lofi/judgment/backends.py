"""Torch and quivers backends for compiled response programs.

Both backends use the same compiler, bound host data, parameter layout and
expression evaluator. The Torch backend interprets the program's steps
directly and scores them with torch distributions. The quivers backend hands
the program to a runtime that builds it as a quivers program, whose priors,
plates, observation densities and trace machinery are quivers' own.
"""

import math
from typing import TYPE_CHECKING, NamedTuple, final, override

import torch
from torch.distributions import (
    Exponential,
    HalfNormal,
    LKJCholesky,
    LogNormal,
    Normal,
    Uniform,
    constraints,
)
from torch.nn import functional

from lofi.judgment._ports import MarginalLikelihoodError
from lofi.judgment._program import (
    Bind,
    Draw,
    ObserveBounded,
    ObserveChoice,
    ObserveOrdered,
    evaluate,
)
from lofi.judgment.bounded import closed_unit_log_prob
from lofi.judgment.compiler import TruncatedMarginalPlan
from lofi.judgment.ordinal import log_category_probabilities

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributions import Distribution
    from torch.distributions.constraints import Constraint

    from lofi.judgment._design import ResponseDesign
    from lofi.judgment._ports import ProgramRuntime
    from lofi.judgment._spec import ResponseDeclaration
    from lofi.judgment.compiler import BoundProgram, ResponseCompiler, ResponseProgram

__all__ = [
    "Latent",
    "QuiversBackend",
    "TorchBackend",
    "latent",
    "log_joint",
    "log_likelihood",
    "log_prior",
    "response_log_probabilities",
    "sample_prior",
]


class Latent(NamedTuple):
    """What a program observes each row under: its location and its thresholds."""

    #: One location on the latent scale per row.
    location: Tensor

    #: The cutpoints, one row shared by every response row or one row per row.
    thresholds: Tensor


def _traced(program: BoundProgram, params: Tensor, /) -> dict[str, Tensor]:
    """Return the value of every name an interpreted program binds or reads.

    The body is interpreted in order over an environment holding the host
    data and, as each draw or binding is reached, the value of its site, at
    the parameter vector's floating-point kind on its device.

    Parameters
    ----------
    program : BoundProgram
        Program bound to the design whose rows are observed.
    params : Tensor
        Value of every drawn site, laid out as the program lays them out.
    """
    environment: dict[str, Tensor] = dict(program.host_data)
    sites = program.sites(params)

    for step in program.program.body:
        match step:
            case Draw(site=site):
                environment[site] = sites[site]
            case Bind(site=site, expression=expression):
                environment[site] = evaluate(expression, environment, like=params)

    return environment


def latent(program: BoundProgram, params: Tensor, /) -> Latent:
    """Return the location and thresholds a program observes each row under.

    Parameters
    ----------
    program : BoundProgram
        Program bound to the design whose rows are observed.
    params : Tensor
        Value of every drawn site, laid out as the program lays them out.

    Raises
    ------
    TypeError
        If the program compares alternatives rather than placing each row
        on a scale of ordered levels, or the vector holds values of a
        nonfloating kind.
    ValueError
        If the vector is not one axis of the layout's length.
    """
    observation = program.observation

    if not isinstance(observation, ObserveOrdered):
        msg = (
            f"a location and its thresholds are what an ordered observation is "
            f"scored under, and this program ends with {observation!r}"
        )

        raise TypeError(msg)

    environment = _traced(program, params)

    return Latent(
        environment[observation.location], environment[observation.thresholds]
    )


def response_log_probabilities(program: BoundProgram, params: Tensor, /) -> Tensor:
    """Return each response row's conditional distribution in log space.

    Parameters
    ----------
    program : BoundProgram
        Program bound to the design whose response distribution is evaluated.
    params : Tensor
        Value of every drawn site, laid out as the program lays them out.

    Raises
    ------
    TypeError
        If the program observes a bounded response rather than a finite set of
        response levels, or the vector holds values of a nonfloating kind.
    ValueError
        If the vector is not one axis of the layout's length.
    """
    environment = _traced(program, params)
    observation = program.observation

    match observation:
        case ObserveOrdered(location=location, thresholds=thresholds):
            return log_category_probabilities(
                environment[location], environment[thresholds]
            )
        case ObserveChoice(logits=logits):
            return functional.log_softmax(environment[logits], dim=-1)
        case ObserveBounded():
            msg = (
                "a finite response distribution belongs to an ordered or choice "
                f"observation, and this program ends with {observation!r}"
            )

            raise TypeError(msg)


def _row_log_likelihood(program: BoundProgram, params: Tensor, /) -> Tensor:
    """Interpret a program and return each row's conditional log probability."""
    observation = program.observation

    match observation:
        case ObserveBounded(location=location, scale=scale):
            environment = _traced(program, params)
            recorded = program.host_data[observation.response]

            return closed_unit_log_prob(
                recorded,
                environment[location],
                environment[scale],
            )
        case ObserveOrdered() | ObserveChoice():
            log_probabilities = response_log_probabilities(program, params)

    recorded = program.host_data[observation.response]

    return log_probabilities.gather(1, recorded.unsqueeze(1)).squeeze(1)


def log_likelihood(program: BoundProgram, params: Tensor, /) -> Tensor:
    """Return the conditional log probability of each recorded level.

    A row placed on a scale of ordered levels is scored in log space under
    the location and thresholds the interpreted program observes it under.
    A row choosing between alternatives is scored under the multinomial
    logit of the scores the program gives them.

    Parameters
    ----------
    program : BoundProgram
        Program bound to the design whose rows are scored.
    params : Tensor
        Value of every drawn site, laid out as the program lays them out.

    Raises
    ------
    TypeError
        If the vector holds values of a nonfloating kind.
    ValueError
        If the vector is not one axis of the layout's length.
    MarginalLikelihoodError
        If the program declares the truncated marginal treatment, whose
        population layout and group reduction are evaluated by
        :class:`ApproximateMarginalLikelihood`.
    """
    if isinstance(program.program.objective, TruncatedMarginalPlan):
        msg = (
            "a truncated_marginal program is scored through "
            "ApproximateMarginalLikelihood over its population layout, not "
            "as per-row conditional likelihoods"
        )

        raise MarginalLikelihoodError(msg)

    return _row_log_likelihood(program, params)


class _Declared(NamedTuple):
    """What a site is drawn from: its distribution and the values it may take."""

    #: The distribution, with argument validation off so that a value
    #: outside the support is scored rather than refused. Off the support
    #: the number it returns is not the density, so a caller reads it
    #: through :func:`_log_density` rather than directly.
    distribution: Distribution

    #: The values the family gives a density, as a constraint on a tensor.
    support: Constraint


def _prior_distribution(draw: Draw, like: Tensor) -> _Declared:
    """Return the distribution a site is drawn from and the support it has.

    Parameters
    ----------
    draw : Draw
        Draw whose prior and event shape are interpreted.
    like : Tensor
        Tensor whose floating-point kind and device the prior's parameters
        are written at.

    Raises
    ------
    ValueError
        If the prior's family has no density this module computes.
    """
    prior = draw.prior
    arguments = tuple(like.new_tensor(value) for value in prior.arguments)

    match prior.family, arguments:
        case "Normal", (location, scale):
            drawn = Normal(location, scale, validate_args=False)

            return _Declared(drawn, constraints.real)
        case "HalfNormal", (scale,):
            drawn = HalfNormal(scale, validate_args=False)

            return _Declared(drawn, constraints.nonnegative)
        case "LogNormal", (location, scale):
            drawn = LogNormal(location, scale, validate_args=False)

            return _Declared(drawn, constraints.positive)
        case "Uniform", (low, high):
            drawn = Uniform(low, high, validate_args=False)

            return _Declared(drawn, constraints.interval(low, high))
        case "Exponential", (rate,):
            drawn = Exponential(rate, validate_args=False)

            return _Declared(drawn, constraints.nonnegative)
        case "LKJCholesky", (concentration,) if (
            len(draw.event_shape) == 2 and draw.event_shape[0] == draw.event_shape[1]
        ):
            drawn = LKJCholesky(draw.event_shape[0], concentration, validate_args=False)

            return _Declared(drawn, constraints.corr_cholesky)

    msg = (
        f"the site {draw.site!r} is drawn from {prior!r} with event shape "
        f"{draw.event_shape!r}, which has no density here"
    )

    raise ValueError(msg)


def _log_density(declared: _Declared, values: Tensor) -> Tensor:
    """Return the log density of a site's values, at negative infinity off support.

    With validation off, a family whose density is undefined outside its
    support returns something other than the density there: LogNormal
    returns a nan at a nonpositive value, and Exponential returns a finite
    number that rises without bound as the value falls, above the density
    of every value the family does give one. Either would pull an
    optimizer off the support. So the density is read only where the
    support holds the value, and elsewhere it is negative infinity. The
    substituted value the discarded density is computed at keeps a nan out
    of the graph, and so out of the gradient.

    Parameters
    ----------
    declared : _Declared
        What the site is drawn from.
    values : Tensor
        Value of every element of one drawn site.
    """
    inside = declared.support.check(values)
    mask = inside

    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)

    guarded = torch.where(mask, values, torch.ones_like(values))
    held = declared.distribution.log_prob(guarded)

    return torch.where(inside, held, held.new_full((), -math.inf))


def log_prior(program: BoundProgram, params: Tensor, /) -> Tensor:
    """Return the log density of every drawn site under the prior it declares.

    Parameters
    ----------
    program : BoundProgram
        Program bound to the design whose sites are laid out.
    params : Tensor
        Value of every drawn site, laid out as the program lays them out.

    Returns
    -------
    Tensor
        The summed log density over every element of every drawn site, as a
        scalar. An element outside the support of the prior it is drawn
        from contributes negative infinity, so the whole density is
        negative infinity there.

    Raises
    ------
    TypeError
        If the vector holds values of a nonfloating kind.
    ValueError
        If the vector is not one axis of the layout's length, or a site is
        drawn from a family this module has no density for.
    """
    sites = program.sites(params)
    total = params.new_zeros(())

    for draw in program.program.draws:
        declared = _prior_distribution(draw, params)
        total = total + _log_density(declared, sites[draw.site]).sum()

    return total


def sample_prior(program: BoundProgram, /, *, generator: torch.Generator) -> Tensor:
    """Return one parameter vector drawn from the priors a program declares.

    Each site is drawn from the prior its draw declares, one event per
    element of the site or per event its family draws whole, in draw order,
    so the vector is laid out as the program lays out its sites. The draws are
    taken from the generator's stream, which they advance, and the process's
    global stream is left as it was.

    Parameters
    ----------
    program : BoundProgram
        Program bound to the design whose sites are laid out.
    generator : torch.Generator
        CPU source of the draws, so that a seed fixes the vector.

    Returns
    -------
    Tensor
        Value of every drawn site, in double precision.

    Raises
    ------
    ValueError
        If a site is drawn from a family this module has no density for.
    """
    like = torch.zeros((), dtype=torch.float64)
    pieces: list[Tensor] = []

    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(generator.get_state())

        for draw, slot in zip(program.program.draws, program.layout, strict=True):
            declared = _prior_distribution(draw, like)
            distribution = declared.distribution
            count = slot.size // math.prod(distribution.event_shape)
            pieces.append(distribution.sample((count,)).reshape(-1))

        generator.set_state(torch.get_rng_state())

    return torch.cat(pieces).to(dtype=like.dtype)


def log_joint(program: BoundProgram, params: Tensor, /) -> Tensor:
    """Return the log joint density of the drawn sites and the recorded levels.

    The drawn sites contribute their prior density at the values the
    parameter vector holds them at, and the rows contribute the density of
    the level each one recorded. A host column a torch module computed
    enters through the rows, so the joint differentiates back into that
    module's parameters.

    Parameters
    ----------
    program : BoundProgram
        Program bound to the design whose rows are scored.
    params : Tensor
        Value of every drawn site, laid out as the program lays them out.

    Returns
    -------
    Tensor
        The log joint density over the whole design, as a scalar. A site
        held outside the support of its prior carries the whole joint to
        negative infinity, so no illegal parameter vector scores above a
        legal one. The rows are not scored there, since a link the prior
        rules out need not place them anywhere.

    Raises
    ------
    TypeError
        If the vector holds values of a nonfloating kind.
    ValueError
        If the vector is not one axis of the layout's length, or a site is
        drawn from a family this module has no density for.
    """
    prior = log_prior(program, params)

    if bool(torch.isneginf(prior)):
        return prior

    return prior + log_likelihood(program, params).sum()


@final
class TorchBackend:
    """The backend that interprets a compiled program in torch.

    Parameters
    ----------
    compiler : ResponseCompiler
        Compiler from specifications to programs.
    """

    __slots__ = ("_compiler",)

    def __init__(self, compiler: ResponseCompiler) -> None:
        self._compiler = compiler

    @property
    def compiler(self) -> ResponseCompiler:
        """Return the compiler from specifications to programs."""
        return self._compiler

    def compile(self, spec: ResponseDeclaration, /) -> ResponseProgram:
        """Compile a response specification.

        Parameters
        ----------
        spec : ResponseDeclaration
            Specification to compile.
        """
        return self._compiler.compile(spec)

    def log_likelihood(
        self,
        compiled: ResponseProgram,
        params: Tensor,
        design: ResponseDesign,
        /,
    ) -> Tensor:
        """Return one log likelihood per response.

        Parameters
        ----------
        compiled : ResponseProgram
            Compiled program.
        params : Tensor
            Value of every drawn site, laid out as the program bound to the
            design lays them out.
        design : ResponseDesign
            Design whose rows are scored.
        """
        if isinstance(compiled.objective, TruncatedMarginalPlan):
            msg = (
                "a truncated_marginal program is scored through "
                "ApproximateMarginalLikelihood over its population layout, not "
                "as per-row conditional likelihoods"
            )

            raise MarginalLikelihoodError(msg)

        return log_likelihood(compiled.over(design), params)

    @override
    def __repr__(self) -> str:
        """Return the linking families the backend compiles."""
        return f"TorchBackend(families={self._compiler.families!r})"


@final
class QuiversBackend:
    """The backend that scores a compiled program in a quivers runtime.

    A specification compiles only if the runtime can build its program, so
    a program with a step quivers has no form for is refused at compilation
    rather than at evaluation.

    Parameters
    ----------
    compiler : ResponseCompiler
        Compiler from specifications to programs.
    runtime : ProgramRuntime
        Runtime that builds a program and scores its rows.
    """

    __slots__ = ("_compiler", "_runtime")

    def __init__(self, compiler: ResponseCompiler, runtime: ProgramRuntime) -> None:
        self._compiler = compiler
        self._runtime = runtime

    @property
    def compiler(self) -> ResponseCompiler:
        """Return the compiler from specifications to programs."""
        return self._compiler

    @property
    def runtime(self) -> ProgramRuntime:
        """Return the runtime the program is scored in."""
        return self._runtime

    def compile(self, spec: ResponseDeclaration, /) -> ResponseProgram:
        """Compile a response specification and check the runtime can build it.

        Parameters
        ----------
        spec : ResponseDeclaration
            Specification to compile.

        Raises
        ------
        ValueError
            If the runtime cannot build the program.
        """
        program = self._compiler.compile(spec)
        self._runtime.load(program)

        return program

    def log_likelihood(
        self,
        compiled: ResponseProgram,
        params: Tensor,
        design: ResponseDesign,
        /,
    ) -> Tensor:
        """Return one log likelihood per response.

        Parameters
        ----------
        compiled : ResponseProgram
            Compiled program.
        params : Tensor
            Value of every drawn site, laid out as the program bound to the
            design lays them out.
        design : ResponseDesign
            Design whose rows are scored.
        """
        if isinstance(compiled.objective, TruncatedMarginalPlan):
            msg = (
                "a truncated_marginal program is scored through "
                "ApproximateMarginalLikelihood over its population layout, not "
                "as per-row conditional likelihoods"
            )

            raise MarginalLikelihoodError(msg)

        return self._runtime.observed_log_probabilities(compiled.over(design), params)

    @override
    def __repr__(self) -> str:
        """Return the linking families the backend compiles and its runtime."""
        return (
            f"QuiversBackend(families={self._compiler.families!r}, "
            f"runtime={self._runtime!r})"
        )
