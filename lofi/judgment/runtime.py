"""A quivers runtime for compiled response programs.

The runtime builds a bound program as a quivers program. A draw over the
one-element set becomes a fixed prior over its site, and a draw over any
other product of sets becomes a plate of that prior, one element per member
of the product. A binding becomes a step that reads the names its expression
refers to and evaluates the expression with :func:`evaluate`, the evaluator
the Torch backend uses. The observation becomes a vectorized observation
under an ordered logistic over a location and its cutpoints, or under a
categorical over a row of logits.

Scoring clamps every drawn site to the parameter vector, traces the run, and
scores each row under the observation's traced arguments, since a vectorized
observation records only the sum over its rows.
"""

import math
from typing import TYPE_CHECKING, final, override

import torch
from quivers import FinSet
from quivers.continuous import (
    Draw as QuiversDraw,
)
from quivers.continuous import (
    Euclidean,
    FixedDistribution,
    Let,
    LKJCorrelationFactor,
    MixedInlineDistribution,
    MonadicProgram,
    Observe,
    OrderedLogistic,
    PlateDraw,
    ProductSpace,
    Step,
    VectorisedObserve,
    make_fixed_exponential,
    make_fixed_halfnormal,
    make_fixed_lognormal,
    make_fixed_normal,
    make_fixed_uniform,
    reading,
)
from quivers.effects import TraceHandler, clamp, run_program
from torch.distributions import Categorical, constraints

from lofi.judgment._program import (
    Bind,
    ObserveBounded,
    ObserveChoice,
    ObserveOrdered,
    evaluate,
)
from lofi.judgment._program import (
    Draw as ProgramDraw,
)
from lofi.judgment.bounded import ClosedUnitTruncatedNormal
from lofi.judgment.compiler import PLATE

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from quivers.continuous import ContinuousMorphism
    from torch import Tensor
    from torch.distributions import Distribution

    from lofi.judgment._program import Expression, Observation, Prior
    from lofi.judgment.compiler import BoundProgram, ResponseProgram

__all__ = ["QuiversRuntime", "build"]

type _Closure = Callable[[Mapping[str, Tensor]], Tensor]
"""A binding's value as a function of the names it reads."""


def _prior(prior: Prior, site: str) -> FixedDistribution:
    """Return the fixed quivers distribution one element of a site is drawn from.

    Parameters
    ----------
    prior : Prior
        Prior the site declares.
    site : str
        Site drawn, which names the distribution's space.

    Raises
    ------
    ValueError
        If quivers has no fixed distribution for the prior's family at the
        number of parameters it gives.
    """
    space = Euclidean(name=site, dim=1)

    match prior.family, prior.arguments:
        case "Normal", (location, scale):
            return make_fixed_normal(location, scale, space)
        case "HalfNormal", (scale,):
            return make_fixed_halfnormal(scale, space)
        case "LogNormal", (location, scale):
            return make_fixed_lognormal(location, scale, space)
        case "Uniform", (low, high):
            return make_fixed_uniform(low, high, space)
        case "Exponential", (rate,):
            return make_fixed_exponential(rate, space)

    msg = (
        f"the site {site!r} is drawn from {prior!r}, which quivers builds no "
        f"fixed distribution for"
    )

    raise ValueError(msg)


def _lkj_dimension(draw: ProgramDraw) -> tuple[int, float]:
    """Return an LKJ draw's dimension and concentration after validation."""
    shape = draw.event_shape

    if (
        len(shape) != 2
        or shape[0] != shape[1]
        or shape[0] < 2
        or len(draw.prior.arguments) != 1
        or draw.prior.arguments[0] <= 0.0
    ):
        msg = (
            f"the site {draw.site!r} declares invalid LKJ arguments "
            f"{draw.prior.arguments!r} and event shape {shape!r}"
        )

        raise ValueError(msg)

    return shape[0], draw.prior.arguments[0]


def _drawn(
    draw: ProgramDraw, program: BoundProgram, domain: FinSet
) -> ContinuousMorphism:
    """Return the quivers morphism a site is drawn from.

    Parameters
    ----------
    draw : ProgramDraw
        Draw of the site.
    program : BoundProgram
        Program bound to the design that sizes the draw's sets.

    Raises
    ------
    ValueError
        If quivers has no fixed distribution for the site's prior.
    """
    if draw.prior.family == "LKJCholesky":
        if draw.index != (program.program.scalar_object,):
            msg = f"the LKJ factor {draw.site!r} must be drawn once"

            raise ValueError(msg)

        dimension, concentration = _lkj_dimension(draw)

        return LKJCorrelationFactor(dimension, concentration, domain)

    element = _prior(draw.prior, draw.site)
    event_size = math.prod(draw.event_shape)
    size = event_size

    for name in draw.index:
        size *= program.sizes[name]

    if size == 1:
        return element

    return PlateDraw(size, element)


def _binding(expression: Expression, drawn: Mapping[str, tuple[int, ...]]) -> _Closure:
    """Return the step value of a binding, reading the names it refers to.

    A drawn site may reach the step with a leading axis of its own, so it is
    read as one flat axis, which is how the parameter layout holds it. The
    expression's literals are written at the kind and on the device of the
    first floating-point value read, and in double precision on the host
    where none is read.

    Parameters
    ----------
    expression : Expression
        Expression the site is bound to.
    drawn : Mapping[str, tuple[int, ...]]
        Event shape of each site the program draws.
    """
    names = tuple(sorted(expression.names))

    def value(environment: Mapping[str, Tensor]) -> Tensor:
        read = {}

        for name in names:
            value = environment[name]
            shape = drawn.get(name)
            read[name] = (
                value
                if shape is None
                else value.reshape(-1)
                if not shape
                else value.reshape(shape)
            )
        floating = [tensor for tensor in read.values() if tensor.is_floating_point()]
        like = floating[0] if floating else torch.zeros((), dtype=torch.float64)

        return evaluate(expression, read, like=like)

    return reading(value, names)


def _ordered_logistic(parameters: list[Tensor]) -> Distribution:
    """Return the ordered logistic over a location and its cutpoints.

    Parameters
    ----------
    parameters : list[Tensor]
        Each row's location, then its cutpoints.
    """
    return OrderedLogistic(parameters[0], parameters[1])


def _categorical(parameters: list[Tensor]) -> Distribution:
    """Return the categorical over each row's logits.

    Parameters
    ----------
    parameters : list[Tensor]
        One row of logits per response row.
    """
    return Categorical(logits=parameters[0])


def _truncated_normal(parameters: list[Tensor]) -> Distribution:
    """Return the closed-unit truncated normal over each row.

    Parameters
    ----------
    parameters : list[Tensor]
        Each row's location, then the positive global scale.
    """
    return ClosedUnitTruncatedNormal(parameters[0], parameters[1])


def _observed(observation: Observation, program: BoundProgram) -> Observe:
    """Return the vectorized observation of the recorded response.

    The recorded levels reach the observation through the host data the
    program is run over, so the observation itself holds only a placeholder
    with one entry per row. The widths declared for the cutpoints and the
    logits describe the parameters; quivers gives every remaining column to
    the trailing vector parameter and does not check them.

    Parameters
    ----------
    observation : Observation
        Observation the program ends with.
    program : BoundProgram
        Program bound to the design whose responses are observed.

    Raises
    ------
    ValueError
        If a choice is observed in a program over no alternatives.
    """
    response = observation.response
    space = Euclidean(name=response, dim=1)
    continuous = False

    match observation:
        case ObserveBounded(location=location, scale=scale):
            space = Euclidean(name=response, dim=1, low=0.0, high=1.0)
            family = MixedInlineDistribution(
                ProductSpace(
                    components=(
                        Euclidean(name=location, dim=1),
                        Euclidean(name=scale, dim=1, low=0.0),
                    )
                ),
                space,
                [("var", 1), ("var", 1)],
                _truncated_normal,
                support=constraints.unit_interval,
                param_event_ranks=(0, 0),
            )
            arguments = (location, scale)
            continuous = True
        case ObserveOrdered(location=location, thresholds=thresholds):
            levels = program.levels

            if levels is None:
                msg = "an ordered observation needs an ordinal response level count"

                raise ValueError(msg)

            width = levels - 1
            family = MixedInlineDistribution(
                ProductSpace(
                    components=(
                        Euclidean(name=location, dim=1),
                        Euclidean(name=thresholds, dim=width),
                    )
                ),
                space,
                [("var", 1), ("var", width)],
                _ordered_logistic,
                discrete=True,
                support=constraints.nonnegative_integer,
                param_event_ranks=(0, 1),
            )
            arguments = (location, thresholds)
        case ObserveChoice(logits=logits):
            width = program.program.alternatives

            if width is None:
                msg = "a choice is observed between alternatives, and rows show none"

                raise ValueError(msg)

            family = MixedInlineDistribution(
                Euclidean(name=logits, dim=width),
                space,
                [("var", width)],
                _categorical,
                discrete=True,
                support=constraints.nonnegative_integer,
                param_event_ranks=(1,),
            )
            arguments = (logits,)

    dtype = torch.float64 if continuous else torch.int64
    placeholder = torch.zeros(program.rows, dtype=dtype)
    observed = VectorisedObserve(family, placeholder)

    return Observe(names=(response,), morphism=observed, args=arguments)


def build(program: BoundProgram, /) -> MonadicProgram:
    """Return a bound program built as a quivers program over its rows.

    Parameters
    ----------
    program : BoundProgram
        Program bound to the design whose rows it observes.

    Raises
    ------
    ValueError
        If quivers has no fixed distribution for a site's prior.
    """
    compiled = program.program
    drawn = {slot.site: slot.shape for slot in program.layout}
    steps: list[Step] = []
    plate = FinSet(name=PLATE, cardinality=program.rows)

    for step in compiled.body:
        match step:
            case ProgramDraw(site=site):
                steps.append(
                    QuiversDraw(names=(site,), morphism=_drawn(step, program, plate))
                )
            case Bind(site=site, expression=expression):
                steps.append(Let(name=site, value=_binding(expression, drawn)))

    observation = compiled.observation
    steps.append(_observed(observation, program))
    return MonadicProgram(plate, plate, steps, (observation.response,))


def _clamped_sites(program: BoundProgram, params: Tensor) -> dict[str, Tensor]:
    """Return site values in the carrier shapes quivers expects."""
    draws = {draw.site: draw for draw in program.program.draws}
    values = program.sites(params)

    return {
        name: (
            value.reshape(1, -1)
            if draws[name].prior.family == "LKJCholesky"
            else value.reshape(-1)
        )
        for name, value in values.items()
    }


@final
class QuiversRuntime:
    """The runtime that builds compiled programs in quivers and scores their rows.

    A program is built once for each set of sizes it is bound at and kept,
    since the recorded responses and every other host column reach the
    built program only when it is run.
    """

    __slots__ = ("_built",)

    def __init__(self) -> None:
        self._built: dict[
            tuple[ResponseProgram, tuple[tuple[str, int], ...]], MonadicProgram
        ] = {}

    @property
    def built(self) -> int:
        """Return the number of programs built and kept."""
        return len(self._built)

    def load(self, program: ResponseProgram, /) -> None:
        """Check that every site the program draws has a quivers prior.

        Parameters
        ----------
        program : ResponseProgram
            Compiled program.

        Raises
        ------
        ValueError
            If quivers has no fixed distribution for a site's prior.
        """
        for draw in program.draws:
            if draw.prior.family == "LKJCholesky":
                _lkj_dimension(draw)
            else:
                _prior(draw.prior, draw.site)

    def program(self, program: BoundProgram, /) -> MonadicProgram:
        """Return a bound program built in quivers, building it on first use.

        Parameters
        ----------
        program : BoundProgram
            Program bound to the design whose rows it observes.

        Raises
        ------
        ValueError
            If quivers has no fixed distribution for a site's prior.
        """
        key = (program.program, tuple(program.sizes.items()))
        built = self._built.get(key)

        if built is None:
            built = self._built[key] = build(program)

        return built

    def log_joint(self, program: BoundProgram, params: Tensor, /) -> Tensor:
        """Return the log joint density quivers gives the sites and the responses.

        A host column a torch module computed enters the observation through
        the host data, so the joint differentiates back into that module's
        parameters.

        Parameters
        ----------
        program : BoundProgram
            Program bound to the design whose rows are scored.
        params : Tensor
            Value of every drawn site, laid out as the program lays them out.

        Returns
        -------
        Tensor
            The log joint density over the whole design, as a scalar.

        Raises
        ------
        TypeError
            If the vector holds values of a nonfloating kind.
        ValueError
            If the vector is not one axis of the layout's length, or quivers
            has no fixed distribution for a site's prior.
        """
        sites = _clamped_sites(program, params)
        joint = self.program(program).log_joint(
            params.new_zeros((1, 1)), {**program.host_data, **sites}
        )

        return joint.reshape(())

    def observed_log_probabilities(
        self,
        program: BoundProgram,
        params: Tensor,
        /,
    ) -> Tensor:
        """Return the log probability of each row's recorded level.

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
            If the vector is not one axis of the layout's length, or quivers
            has no fixed distribution for a site's prior.
        """
        sites = _clamped_sites(program, params)
        built = self.program(program)
        handler = TraceHandler()

        with clamp(sites), handler:
            run_program(built, params.new_zeros((1, 1)), dict(program.host_data))

        observation = program.observation
        traced = handler.trace.sites
        recorded = program.host_data[observation.response]

        match observation:
            case ObserveBounded(location=location, scale=scale):
                scored = ClosedUnitTruncatedNormal(
                    traced[location].value,
                    traced[scale].value,
                )
            case ObserveOrdered(location=location, thresholds=thresholds):
                scored = OrderedLogistic(
                    traced[location].value, traced[thresholds].value
                )
            case ObserveChoice(logits=logits):
                scored = Categorical(logits=traced[logits].value)

        return scored.log_prob(recorded)

    @override
    def __repr__(self) -> str:
        """Return the number of programs built."""
        return f"QuiversRuntime(built={len(self._built)})"
