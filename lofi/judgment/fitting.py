"""Support-preserving maximum a posteriori fits of response programs."""

from typing import TYPE_CHECKING, Literal, cast, final

import torch
from quivers.inference import LatentRegistry
from torch import Tensor, nn
from torch.optim import Optimizer

from lofi._checks import ensure_finite, ensure_non_negative, ensure_positive
from lofi._frozen import FrozenValue
from lofi.judgment._ports import PointFittingError, PointObjective
from lofi.judgment._spec import InferenceMethod, MapInference
from lofi.judgment.backends import log_joint
from lofi.judgment.compiler import BoundProgram, ConditionalPlan, Slot
from lofi.judgment.marginal import ApproximateMarginalLikelihood
from lofi.judgment.runtime import QuiversRuntime
from lofi.registry import plugin_for

if TYPE_CHECKING:
    from collections.abc import Collection

    from lofi.spec import ConfiguredSpec

__all__ = [
    "ConditionalPointObjective",
    "MapFitter",
    "MarginalPointObjective",
    "PointFit",
]


@final
class PointFit(FrozenValue):
    """Record one fitted point in its objective owner's parameter layout."""

    _noun = "point fit"

    __slots__ = ("_log_density", "_parameters", "_steps")

    def __init__(self, *, parameters: Tensor, log_density: float, steps: int) -> None:
        if not isinstance(parameters, Tensor):
            msg = f"a point fit's parameters are not a tensor: {parameters!r}"
            raise TypeError(msg)
        if parameters.ndim != 1:
            msg = (
                f"a point fit's parameters have shape {tuple(parameters.shape)!r}, "
                "not one flat layout axis"
            )
            raise ValueError(msg)
        if not parameters.is_floating_point() or parameters.is_complex():
            msg = "a point fit's parameters are not real floating-point values"
            raise TypeError(msg)
        if not bool(torch.isfinite(parameters).all()):
            msg = "a point fit's parameters contain a nonfinite value"
            raise ValueError(msg)

        held_density = ensure_finite("a point fit's log density", log_density)
        held_steps = ensure_positive("a point fit's completed step count", steps)
        object.__setattr__(
            self, "_parameters", parameters.detach().to(device="cpu").clone()
        )
        object.__setattr__(self, "_log_density", held_density)
        object.__setattr__(self, "_steps", held_steps)

    @property
    def parameters(self) -> Tensor:
        """Return a clone of the fitted constrained parameter vector."""
        return self._parameters.clone()

    @property
    def log_density(self) -> float:
        """Return the final log posterior density."""
        return self._log_density

    @property
    def steps(self) -> int:
        """Return the completed optimizer-step count."""
        return self._steps


@final
class _SupportCoordinates(FrozenValue):
    """Map one owner layout through Quivers' public support bijectors."""

    _noun = "support coordinates"

    __slots__ = ("_layout", "_registry")

    def __init__(
        self,
        program: BoundProgram,
        layout: tuple[Slot, ...],
        observed: Collection[str],
    ) -> None:
        model = QuiversRuntime().program(program)
        registry = LatentRegistry.from_model(model, frozenset(observed))
        expected = tuple(slot.site for slot in layout)

        if registry.names != expected:
            msg = (
                f"the support registry lays out sites {registry.names!r}, not the "
                f"objective owner's sites {expected!r}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_layout", layout)
        object.__setattr__(self, "_registry", registry)

    @property
    def layout(self) -> tuple[Slot, ...]:
        """Return the constrained parameter layout in owner order."""
        return self._layout

    def initial_position(self, *, generator: torch.Generator, scale: float) -> Tensor:
        """Draw one flat unconstrained position from a caller-owned stream."""
        return scale * torch.randn(
            self._registry.total_unconstrained_dim,
            dtype=torch.float64,
            generator=generator,
        )

    def parameters(self, position: Tensor, /) -> Tensor:
        """Map a flat unconstrained position into the owner layout."""
        if not position.is_floating_point() or position.is_complex():
            msg = "a MAP position is not a real floating-point tensor"
            raise TypeError(msg)
        expected = self._registry.total_unconstrained_dim

        if position.ndim != 1 or position.numel() != expected:
            msg = (
                f"the unconstrained support layout has {expected} values, and the "
                f"position has shape {tuple(position.shape)!r}"
            )
            raise ValueError(msg)

        unconstrained = self._registry.unflatten_unconstrained(position)
        constrained, _ = self._registry.to_constrained(unconstrained)
        pieces: list[Tensor] = []

        for slot in self._layout:
            piece = constrained[slot.site].reshape(-1)

            if piece.numel() != slot.size:
                msg = (
                    f"the support transform produced {piece.numel()} values for "
                    f"site {slot.site!r}, not its layout size {slot.size}"
                )
                raise ValueError(msg)

            pieces.append(piece)

        if not pieces:
            return position.new_empty(0)

        return torch.cat(pieces)


@final
class ConditionalPointObjective(FrozenValue):
    """Own the full constrained layout and joint density of one program."""

    _noun = "conditional point objective"

    __slots__ = ("_coordinates", "_program")

    def __init__(self, program: BoundProgram, /) -> None:
        if not isinstance(program, BoundProgram):
            msg = (
                f"a conditional point objective received no bound program: {program!r}"
            )
            raise TypeError(msg)

        match program.program.objective:
            case ConditionalPlan():
                pass
            case _:
                msg = "a conditional point objective requires a conditional program"
                raise ValueError(msg)

        object.__setattr__(self, "_program", program)
        object.__setattr__(
            self,
            "_coordinates",
            _SupportCoordinates(program, program.layout, program.host_data),
        )

    @property
    def layout(self) -> tuple[Slot, ...]:
        """Return every drawn site in program order."""
        return self._coordinates.layout

    def initial_position(self, *, generator: torch.Generator, scale: float) -> Tensor:
        """Draw one support-coordinate initialization."""
        return self._coordinates.initial_position(generator=generator, scale=scale)

    def parameters(self, position: Tensor, /) -> Tensor:
        """Return constrained parameters in the complete program layout."""
        return self._coordinates.parameters(position)

    def log_density(self, parameters: Tensor, /) -> Tensor:
        """Return the conditional program's complete log joint density."""
        return log_joint(self._program, parameters)


@final
class MarginalPointObjective(FrozenValue):
    """Own the population layout and integrated posterior of one program."""

    _noun = "marginal point objective"

    __slots__ = ("_coordinates", "_marginal")

    def __init__(
        self,
        program: BoundProgram,
        /,
        *,
        active_groups: tuple[int, ...],
    ) -> None:
        if not isinstance(program, BoundProgram):
            msg = f"a marginal point objective received no bound program: {program!r}"
            raise TypeError(msg)

        marginal = ApproximateMarginalLikelihood(program, active_groups=active_groups)
        population = frozenset(slot.site for slot in marginal.population_layout)
        local = tuple(
            draw.site for draw in program.program.draws if draw.site not in population
        )
        observed = (*program.host_data, *local)
        object.__setattr__(self, "_marginal", marginal)
        object.__setattr__(
            self,
            "_coordinates",
            _SupportCoordinates(program, marginal.population_layout, observed),
        )

    @property
    def layout(self) -> tuple[Slot, ...]:
        """Return the nonlocal drawn sites in program order."""
        return self._coordinates.layout

    def initial_position(self, *, generator: torch.Generator, scale: float) -> Tensor:
        """Draw one population support-coordinate initialization."""
        return self._coordinates.initial_position(generator=generator, scale=scale)

    def parameters(self, position: Tensor, /) -> Tensor:
        """Return constrained parameters in the population layout."""
        return self._coordinates.parameters(position)

    def log_density(self, parameters: Tensor, /) -> Tensor:
        """Return the summed complete-group population log posterior."""
        return self._marginal.log_posterior(parameters)


def _optimizer(plan: MapInference, parameter: nn.Parameter, /) -> Optimizer:
    """Build the plan's installed optimizer over its one coordinate vector."""
    declaration = cast("ConfiguredSpec[object]", plan.optimizer)
    built = plugin_for(declaration).construct(declaration, (parameter,))

    if not isinstance(built, Optimizer):
        msg = (
            f"the optimizer plugin {plan.optimizer.kind!r} built {built!r}, not a "
            "Torch optimizer"
        )
        raise TypeError(msg)

    return built


def _finite_density(value: Tensor, *, step: int) -> Tensor:
    """Return one finite scalar log density or raise the named fit error."""
    if not isinstance(value, Tensor) or value.ndim != 0:
        shape = None if not isinstance(value, Tensor) else tuple(value.shape)
        msg = f"MAP fitting produced log density shape {shape!r} at step {step}"
        raise PointFittingError(msg)
    if not bool(torch.isfinite(value)):
        msg = f"MAP fitting produced a nonfinite log density at step {step}"
        raise PointFittingError(msg)

    return value


@final
class MapFitter(FrozenValue):
    """Optimize a point objective in support-preserving coordinates."""

    _noun = "MAP fitter"

    __slots__ = ("_plan", "_seed")

    def __init__(self, plan: MapInference, /, *, seed: int) -> None:
        if not isinstance(plan, MapInference):
            msg = f"a MAP fitter's plan is not a MapInference: {plan!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_plan", plan)
        object.__setattr__(self, "_seed", ensure_non_negative("a MAP fit's seed", seed))

    @property
    def method(self) -> Literal[InferenceMethod.MAP]:
        """Return maximum a posteriori inference."""
        return InferenceMethod.MAP

    @property
    def plan(self) -> MapInference:
        """Return the complete immutable optimization declaration."""
        return self._plan

    @property
    def seed(self) -> int:
        """Return the initialization seed."""
        return self._seed

    def fit(self, objective: PointObjective, /) -> PointFit:
        """Optimize and return one constrained point in the objective layout."""
        if not isinstance(objective, PointObjective):
            msg = f"a MAP fitter received no point objective: {objective!r}"
            raise TypeError(msg)

        generator = torch.Generator(device="cpu").manual_seed(self._seed)
        initial = objective.initial_position(
            generator=generator, scale=self._plan.initial_scale
        )

        if not bool(torch.isfinite(initial).all()):
            msg = "MAP fitting produced a nonfinite initial position"
            raise PointFittingError(msg)

        position = nn.Parameter(initial)
        optimizer = _optimizer(self._plan, position)

        try:
            for step in range(1, self._plan.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                parameters = objective.parameters(position)
                density = _finite_density(objective.log_density(parameters), step=step)
                (-density).backward()
                gradient = position.grad

                if gradient is None or not bool(torch.isfinite(gradient).all()):
                    msg = f"MAP fitting produced a nonfinite gradient at step {step}"
                    raise PointFittingError(msg)

                optimizer.step()

            parameters = objective.parameters(position)
            density = _finite_density(
                objective.log_density(parameters), step=self._plan.steps
            )

        except torch.OutOfMemoryError as error:
            raise MemoryError(str(error)) from error

        return PointFit(
            parameters=parameters,
            log_density=float(density.detach()),
            steps=self._plan.steps,
        )
