"""Truncated marginal likelihoods over complete group-local latent blocks."""

import math
from typing import TYPE_CHECKING, final

import torch
from torchdiffeq import odeint

from lofi.judgment._ports import MarginalLikelihoodError
from lofi.judgment._program import ObserveOrdered, Prior
from lofi.judgment.backends import _row_log_likelihood, log_prior
from lofi.judgment.compiler import PARTICIPANT, Slot, TruncatedMarginalPlan

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from lofi.judgment.compiler import BoundProgram

__all__ = ["ApproximateMarginalLikelihood"]


_STANDARD_NORMAL = Prior("Normal", 0.0, 1.0)
_LOG_TWO_PI = math.log(2.0 * math.pi)


@final
class ApproximateMarginalLikelihood:
    """Integrate one complete group-local standardized latent block.

    A compiled marginal objective determines the group, finite radius and
    numerical controls. The population vector follows :attr:`population_layout`
    and therefore excludes every draw whose declared index contains the local
    group. Rank-one blocks use the paper's adaptive ODE integral. Wider blocks
    use a positive in-ball quadrature rule, either generated from the declared
    count and local seed or supplied explicitly by the caller.

    Parameters
    ----------
    program : BoundProgram
        A compiled truncated-marginal response program bound to the complete
        response design.
    active_groups : Sequence[int]
        Distinct group indices whose complete response rows the program contains.
    nodes : Tensor | None
        Optional finite in-ball nodes, one row per quadrature point. They are
        accepted only for a block wider than one and must be supplied with
        ``weights``.
    weights : Tensor | None
        Optional finite positive quadrature weights, one per node.
    """

    __slots__ = (
        "_active_groups",
        "_group_count",
        "_group_index",
        "_local",
        "_local_width",
        "_nodes",
        "_plan",
        "_population_layout",
        "_program",
        "_weights",
    )

    def __init__(
        self,
        program: BoundProgram,
        /,
        *,
        active_groups: Sequence[int],
        nodes: Tensor | None = None,
        weights: Tensor | None = None,
    ) -> None:
        plan = program.program.objective

        if not isinstance(plan, TruncatedMarginalPlan):
            msg = "a marginal evaluator requires a truncated_marginal objective"

            raise MarginalLikelihoodError(msg)

        if not isinstance(program.observation, ObserveOrdered):
            msg = (
                "a truncated marginal likelihood requires the cumulative ordered "
                f"response family, not {type(program.observation).__name__}"
            )

            raise MarginalLikelihoodError(msg)

        if plan.group != PARTICIPANT:
            msg = (
                "the current truncated marginal treatment integrates the complete "
                f"{PARTICIPANT!r} block, not {plan.group!r}"
            )

            raise MarginalLikelihoodError(msg)

        groups = program.program.groups

        if plan.group not in groups:
            msg = (
                f"the marginal group {plan.group!r} has no declared random effect; "
                f"the program declares {groups!r}"
            )

            raise MarginalLikelihoodError(msg)

        if groups != (plan.group,):
            msg = (
                f"the marginal group {plan.group!r} must be the sole local grouping "
                f"factor; the crossed groups are {groups!r}"
            )

            raise MarginalLikelihoodError(msg)

        objects = program.program.objects
        group_objects = tuple(
            name for name, factors in objects.items() if factors == (plan.group,)
        )

        if len(group_objects) != 1:
            msg = (
                f"the group {plan.group!r} must have exactly one declared response "
                f"object; found {group_objects!r}"
            )

            raise MarginalLikelihoodError(msg)

        index_columns = tuple(
            name
            for name, factors in program.program.indices.items()
            if factors == (plan.group,)
        )

        if len(index_columns) != 1:
            msg = (
                f"the group {plan.group!r} must have exactly one response index "
                f"column; found {index_columns!r}"
            )

            raise MarginalLikelihoodError(msg)

        group_object = group_objects[0]
        group_count = program.sizes[group_object]
        group_index = program.host_data[index_columns[0]]

        if group_count < 1:
            msg = f"the marginal group {plan.group!r} is empty"

            raise MarginalLikelihoodError(msg)

        held_groups = tuple(active_groups)

        for group in held_groups:
            if isinstance(group, bool) or not isinstance(group, int):
                msg = f"an active marginal group is not an integer: {group!r}"

                raise MarginalLikelihoodError(msg)

            if not 0 <= group < group_count:
                msg = f"active marginal group {group} lies outside 0..{group_count - 1}"

                raise MarginalLikelihoodError(msg)

        active = tuple(sorted(held_groups))

        if not active:
            msg = "a marginal evaluator requires at least one active group"

            raise MarginalLikelihoodError(msg)

        if len(frozenset(active)) != len(active):
            msg = f"active marginal groups must be unique: {held_groups!r}"

            raise MarginalLikelihoodError(msg)

        observed = tuple(sorted(frozenset(int(value) for value in group_index)))

        if observed != active:
            msg = (
                f"the marginal rows belong to groups {observed!r}, not exactly "
                f"the active groups {active!r}"
            )

            raise MarginalLikelihoodError(msg)

        local: dict[str, int] = {}
        population: list[Slot] = []

        for draw, slot in zip(program.program.draws, program.layout, strict=True):
            local_objects = tuple(
                name for name in draw.index if plan.group in objects.get(name, ())
            )

            if not local_objects:
                population.append(slot)
                continue

            if len(local_objects) != 1:
                msg = (
                    f"the local site {draw.site!r} has ambiguous group objects "
                    f"{local_objects!r}"
                )

                raise MarginalLikelihoodError(msg)

            local_object = local_objects[0]
            factors = objects[local_object]

            if factors[0] != plan.group or draw.index[0] != local_object:
                msg = (
                    f"the local site {draw.site!r} does not place the group "
                    f"{plan.group!r} first in its declared structural index"
                )

                raise MarginalLikelihoodError(msg)

            if draw.prior != _STANDARD_NORMAL:
                msg = (
                    f"the local site {draw.site!r} must have the standard-normal "
                    f"prior, not {draw.prior!r}"
                )

                raise MarginalLikelihoodError(msg)

            if slot.size % group_count:
                msg = (
                    f"the local site {draw.site!r} has size {slot.size}, which "
                    f"cannot be partitioned over {group_count} group levels"
                )

                raise MarginalLikelihoodError(msg)

            local[draw.site] = slot.size // group_count

        if not local:
            msg = f"the marginal group {plan.group!r} has no local latent effect"

            raise MarginalLikelihoodError(msg)

        local_width = sum(local.values())
        supplied = nodes is not None or weights is not None

        if local_width == 1 and supplied:
            msg = "a scalar local block uses the adaptive ODE and accepts no quadrature"

            raise MarginalLikelihoodError(msg)

        if local_width > 1:
            if (nodes is None) != (weights is None):
                msg = "multivariate quadrature requires both nodes and weights"

                raise MarginalLikelihoodError(msg)

            if nodes is None or weights is None:
                nodes, weights = _ball_rule(
                    local_width,
                    plan.radius,
                    plan.quadrature_points,
                    plan.quadrature_seed,
                )

            nodes, weights = _checked_rule(
                nodes,
                weights,
                rank=local_width,
                radius=plan.radius,
                count=plan.quadrature_points,
            )

        self._program = program
        self._plan = plan
        self._population_layout = tuple(population)
        self._local = local
        self._local_width = local_width
        self._group_count = group_count
        self._group_index = group_index
        self._active_groups = active
        self._nodes = nodes
        self._weights = weights

    @property
    def population_layout(self) -> tuple[Slot, ...]:
        """Return the nonlocal drawn sites in their flat-vector order."""
        return self._population_layout

    @property
    def population_size(self) -> int:
        """Return the length of a population parameter vector."""
        return sum(slot.size for slot in self._population_layout)

    @property
    def group(self) -> str:
        """Return the grouping factor whose complete local block is integrated."""
        return self._plan.group

    @property
    def local_width(self) -> int:
        """Return the width of one group level's standardized local block."""
        return self._local_width

    @property
    def active_groups(self) -> tuple[int, ...]:
        """Return the integrated group indices in canonical order."""
        return self._active_groups

    def _population_sites(self, population: Tensor) -> dict[str, Tensor]:
        """Return flattened population pieces by site after validation."""
        if not population.is_floating_point():
            msg = (
                f"a marginal population vector holds {population.dtype}, not a "
                "floating-point kind"
            )

            raise MarginalLikelihoodError(msg)

        if population.ndim != 1 or population.numel() != self.population_size:
            msg = (
                f"the marginal population layout has {self.population_size} values, "
                f"and the vector has shape {tuple(population.shape)}"
            )

            raise MarginalLikelihoodError(msg)

        pieces = torch.split(
            population, [slot.size for slot in self._population_layout]
        )

        return {
            slot.site: piece
            for slot, piece in zip(self._population_layout, pieces, strict=True)
        }

    def _parameters(
        self,
        population_sites: dict[str, Tensor],
        node: Tensor,
        group: int,
    ) -> Tensor:
        """Insert one group's local node into the complete program layout."""
        pieces: list[Tensor] = []
        position = 0

        for draw in self._program.program.draws:
            width = self._local.get(draw.site)

            if width is None:
                pieces.append(population_sites[draw.site])
                continue

            values = node[position : position + width]
            position += width
            before = values.new_zeros(group * width)
            after = values.new_zeros((self._group_count - group - 1) * width)
            pieces.append(torch.cat((before, values, after)))

        return torch.cat(pieces)

    def _integrand(
        self,
        population_sites: dict[str, Tensor],
        node: Tensor,
        group: int,
    ) -> Tensor:
        """Return one complete group's log likelihood plus local log density."""
        params = self._parameters(population_sites, node, group)
        rows = self._group_index == group
        likelihood = _row_log_likelihood(self._program, params)[rows].sum()
        local_prior = -0.5 * (node.square().sum() + self._local_width * _LOG_TWO_PI)

        return likelihood + local_prior

    def _scalar(
        self,
        population_sites: dict[str, Tensor],
        group: int,
        like: Tensor,
    ) -> Tensor:
        """Integrate a scalar local block by the published adaptive ODE."""
        zero = like.new_zeros(1)
        shift = self._integrand(population_sites, zero, group).detach()

        def derivative(time: Tensor, state: Tensor) -> Tensor:
            del state
            value = self._integrand(population_sites, time.reshape(1), group)

            return torch.exp(value - shift).reshape(1)

        times = like.new_tensor((-self._plan.radius, self._plan.radius))
        solution = odeint(
            derivative,
            zero,
            times,
            rtol=self._plan.relative_tolerance,
            atol=self._plan.absolute_tolerance,
            method="dopri5",
        )
        integral = solution[-1, 0]

        if not bool(torch.isfinite(integral)) or bool(integral <= 0.0):
            msg = (
                f"the scalar marginal integral for group level {group} is not "
                f"finite and positive: {integral!r}"
            )

            raise MarginalLikelihoodError(msg)

        return torch.log(integral) + shift

    def _multivariate(
        self,
        population_sites: dict[str, Tensor],
        group: int,
        like: Tensor,
    ) -> Tensor:
        """Integrate a multivariate local block by positive in-ball quadrature."""
        nodes = self._nodes
        weights = self._weights

        if nodes is None or weights is None:
            msg = "a multivariate marginal has no quadrature rule"

            raise MarginalLikelihoodError(msg)

        moved_nodes = nodes.to(dtype=like.dtype, device=like.device)
        moved_weights = weights.to(dtype=like.dtype, device=like.device)
        values = torch.stack(
            tuple(
                self._integrand(population_sites, node, group) for node in moved_nodes
            )
        )

        return torch.logsumexp(torch.log(moved_weights) + values, dim=0)

    def group_log_likelihood(self, population: Tensor, group: int, /) -> Tensor:
        """Return one active group's complete integrated log likelihood."""
        if isinstance(group, bool) or not isinstance(group, int):
            msg = f"a marginal group index is not an integer: {group!r}"
            raise TypeError(msg)

        if group not in self._active_groups:
            msg = (
                f"group {group} is not among this marginal evaluator's active "
                f"groups {self._active_groups!r}"
            )
            raise MarginalLikelihoodError(msg)

        population_sites = self._population_sites(population)

        return self._group_log_likelihood(population_sites, group, population)

    def _group_log_likelihood(
        self,
        population_sites: dict[str, Tensor],
        group: int,
        like: Tensor,
    ) -> Tensor:
        """Integrate one validated active group from split population sites."""
        integrate = self._scalar if self._local_width == 1 else self._multivariate

        return integrate(population_sites, group, like)

    def log_likelihood(self, population: Tensor, /) -> Tensor:
        """Return the uniform group-mean integrated log likelihood."""
        population_sites = self._population_sites(population)
        values = torch.stack(
            tuple(
                self._group_log_likelihood(population_sites, group, population)
                for group in self._active_groups
            )
        )

        return values.mean()

    def log_posterior(self, population: Tensor, /) -> Tensor:
        """Return complete-group log marginals plus one population prior."""
        population_sites = self._population_sites(population)
        zero = population.new_zeros(self._local_width)
        complete = self._parameters(population_sites, zero, 0)
        local_zero_density = -0.5 * self._group_count * self._local_width * _LOG_TWO_PI
        population_prior = log_prior(self._program, complete) - local_zero_density
        likelihood = torch.stack(
            tuple(
                self._group_log_likelihood(population_sites, group, population)
                for group in self._active_groups
            )
        ).sum()

        return population_prior + likelihood

    def __call__(self, population: Tensor, /) -> Tensor:
        """Return the uniform mean log marginal over complete group levels."""
        return self.log_likelihood(population)


def _ball_rule(
    rank: int, radius: float, count: int, seed: int
) -> tuple[Tensor, Tensor]:
    """Construct a deterministic positive equal-weight rule inside a ball."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    uniforms = torch.rand((count, rank + 1), generator=generator, dtype=torch.float64)
    bounded = uniforms[:, 1:].clamp(torch.finfo(torch.float64).eps, 1.0 - 1e-15)
    directions = torch.special.ndtri(bounded)
    directions = directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    radii = radius * uniforms[:, :1].pow(1.0 / rank)
    nodes = radii * directions
    log_volume = (
        0.5 * rank * math.log(math.pi)
        + rank * math.log(radius)
        - math.lgamma(0.5 * rank + 1.0)
    )
    weights = torch.full((count,), math.exp(log_volume) / count, dtype=torch.float64)

    return nodes, weights


def _checked_rule(
    nodes: Tensor,
    weights: Tensor,
    *,
    rank: int,
    radius: float,
    count: int,
) -> tuple[Tensor, Tensor]:
    """Return an immutable copy of a finite positive in-ball quadrature rule."""
    if not nodes.is_floating_point() or not weights.is_floating_point():
        msg = "quadrature nodes and weights must have floating-point kinds"

        raise MarginalLikelihoodError(msg)

    if nodes.shape != (count, rank):
        msg = (
            f"quadrature nodes must have shape {(count, rank)!r}, not "
            f"{tuple(nodes.shape)!r}"
        )

        raise MarginalLikelihoodError(msg)

    if weights.shape != (count,):
        msg = (
            f"quadrature weights must have shape {(count,)!r}, not "
            f"{tuple(weights.shape)!r}"
        )

        raise MarginalLikelihoodError(msg)

    if not bool(torch.isfinite(nodes).all()):
        msg = "quadrature nodes must all be finite"

        raise MarginalLikelihoodError(msg)

    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0.0).any()):
        msg = "quadrature weights must all be finite and positive"

        raise MarginalLikelihoodError(msg)

    if bool((torch.linalg.vector_norm(nodes, dim=1) > radius).any()):
        msg = f"quadrature nodes must lie inside the radius-{radius!r} ball"

        raise MarginalLikelihoodError(msg)

    return nodes.detach().clone(), weights.detach().clone()
