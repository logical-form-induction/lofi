"""Distributed alignment search over a masked rotated subspace."""

from enum import Enum
from math import ceil, exp, log
from typing import TYPE_CHECKING, Protocol, Self, cast, override

import torch
from torch.nn.functional import cross_entropy

from lofi._checks import ensure_finite, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.nn.sites import Access, Site
from lofi.probe._causal import (
    CombinatorAt,
    IIAResult,
    NamedVariable,
    Side,
    SplitAt,
    TypeAt,
)
from lofi.probe._circuits import Unit
from lofi.probe._instrument import instrument
from lofi.probe._interventions import Interchange
from lofi.probe._pricing import ProbeUnit
from lofi.probe._rotation import OrthogonalOptimizer
from lofi.stats import LabelPermutation, p_value_against_replicates

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from torch import Tensor

    from lofi._types import JsonValue
    from lofi.nn.ports import Reading
    from lofi.probe._causal import (
        AlignmentPolicy,
        CausalPair,
        CounterfactualPair,
        HighLevelVariable,
    )
    from lofi.probe._ports import Subject
    from lofi.stats import PValue

__all__ = [
    "AlignmentObjective",
    "Annealing",
    "Assessment",
    "BoundlessSearch",
    "DistributedAlignment",
    "FitStep",
    "Fitted",
    "FixedRankSearch",
    "Schedule",
    "SearchStrategy",
]


class AlignmentObjective(Enum):
    """Names the declared loss optimized by a distributed alignment.

    Cross entropy fits the target decision as a class. Logit difference instead
    maximizes the target score's margin over the strongest competing decision,
    which is the causal-abstraction objective when only relative decision scores
    are meaningful.
    """

    CROSS_ENTROPY = "cross-entropy"
    LOGIT_DIFFERENCE = "logit-difference"

    def loss(self, scores: Tensor, targets: Tensor, /) -> Tensor:
        """Return the scalar loss for one batch of decision scores.

        Parameters
        ----------
        scores : Tensor
            Rank-two scores over examples and decisions.
        targets : Tensor
            One integer target decision per example.
        """
        if scores.ndim != 2:
            msg = f"alignment scores have rank {scores.ndim}, not two"
            raise ValueError(msg)
        if targets.ndim != 1 or int(targets.shape[0]) != int(scores.shape[0]):
            msg = (
                f"alignment targets have shape {tuple(targets.shape)}, not "
                f"{(int(scores.shape[0]),)}"
            )
            raise ValueError(msg)
        if scores.shape[1] < 2:
            msg = "an alignment objective compares at least two decisions"
            raise ValueError(msg)
        if targets.dtype is torch.bool or targets.is_floating_point():
            msg = (
                f"alignment targets hold integer decision indices, not {targets.dtype}"
            )
            raise TypeError(msg)
        if bool(((targets < 0) | (targets >= scores.shape[1])).any()):
            msg = "an alignment target lies outside the scored decisions"
            raise ValueError(msg)

        if self is AlignmentObjective.CROSS_ENTROPY:
            return cross_entropy(scores, targets)

        target = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
        competing = scores.scatter(1, targets.unsqueeze(1), -torch.inf).amax(dim=1)

        return -(target - competing).mean()


class Schedule(FrozenValue):
    """How long a search runs, at what rate, in what batches, and under what seed."""

    __slots__ = ("_batch_size", "_epochs", "_learning_rate", "_seed")

    _noun = "search schedule"

    def __init__(
        self, *, epochs: int, learning_rate: float, batch_size: int, seed: int
    ) -> None:
        held_epochs = ensure_non_negative("an epoch count", epochs)
        held_rate = ensure_finite("a learning rate", learning_rate)
        held_batch = ensure_non_negative("a batch size", batch_size)
        held_seed = ensure_non_negative("a search seed", seed)

        if held_epochs == 0:
            msg = "a search of zero epochs fits nothing"
            raise ValueError(msg)
        if held_rate <= 0.0:
            msg = f"a learning rate is not positive: {learning_rate!r}"
            raise ValueError(msg)
        if held_batch == 0:
            msg = "a batch of zero pairs feeds no step"
            raise ValueError(msg)
        object.__setattr__(self, "_epochs", held_epochs)
        object.__setattr__(self, "_learning_rate", held_rate)
        object.__setattr__(self, "_batch_size", held_batch)
        object.__setattr__(self, "_seed", held_seed)

    @property
    def epochs(self) -> int:
        """Return how many passes over the fitting pairs the search makes."""
        return self._epochs

    @property
    def learning_rate(self) -> float:
        """Return the optimizer's step size."""
        return self._learning_rate

    @property
    def batch_size(self) -> int:
        """Return how many pairs each step reads."""
        return self._batch_size

    @property
    def seed(self) -> int:
        """Return the seed used for the initial rotation and pair order."""
        return self._seed

    def _identity(self) -> tuple[int, float, int, int]:
        return (self._epochs, self._learning_rate, self._batch_size, self._seed)

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not Schedule:
            return NotImplemented
        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        return hash((Schedule, self._identity()))

    @override
    def __repr__(self) -> str:
        return (
            f"Schedule(epochs={self._epochs!r}, "
            f"learning_rate={self._learning_rate!r}, "
            f"batch_size={self._batch_size!r}, seed={self._seed!r})"
        )


class Annealing(FrozenValue):
    """A positive geometric temperature schedule for continuous masks."""

    __slots__ = ("_final", "_initial")

    _noun = "annealing schedule"

    def __init__(self, *, initial: float, final: float) -> None:
        held_initial = ensure_finite("an initial temperature", initial)
        held_final = ensure_finite("a final temperature", final)

        if held_initial <= 0.0:
            msg = f"an initial temperature is not positive: {initial!r}"
            raise ValueError(msg)
        if held_final <= 0.0:
            msg = f"a final temperature is not positive: {final!r}"
            raise ValueError(msg)
        if held_final >= held_initial:
            msg = (
                f"an annealing schedule ends below its initial temperature, not "
                f"at {final!r} from {initial!r}"
            )
            raise ValueError(msg)
        object.__setattr__(self, "_initial", held_initial)
        object.__setattr__(self, "_final", held_final)

    @property
    def initial(self) -> float:
        """Return the temperature at the first optimizer step."""
        return self._initial

    @property
    def final(self) -> float:
        """Return the temperature at the final optimizer step."""
        return self._final

    def temperature(self, step: int, total: int) -> float:
        """Return the geometric temperature at one zero-based optimizer step."""
        held_step = ensure_non_negative("an annealing step", step)
        held_total = ensure_non_negative("an annealing step count", total)
        if held_total == 0:
            msg = "an annealing schedule has at least one step"
            raise ValueError(msg)
        if held_step >= held_total:
            msg = f"annealing step {held_step} is outside {held_total} steps"
            raise ValueError(msg)
        if held_total == 1:
            return self._initial
        fraction = held_step / (held_total - 1)
        return exp(log(self._initial) + fraction * log(self._final / self._initial))

    def _identity(self) -> tuple[float, float]:
        return (self._initial, self._final)

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not Annealing:
            return NotImplemented
        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        return hash((Annealing, self._identity()))

    @override
    def __repr__(self) -> str:
        return f"Annealing(initial={self._initial!r}, final={self._final!r})"


def _search_parts(
    schedule: Schedule, null: LabelPermutation[int]
) -> tuple[Schedule, LabelPermutation[int]]:
    if type(schedule) is not Schedule:
        msg = f"a search runs under a Schedule, not {schedule!r}"
        raise TypeError(msg)
    if type(null) is not LabelPermutation:
        msg = f"a search is tested against a LabelPermutation, not {null!r}"
        raise TypeError(msg)
    return schedule, null


class _MaskState(Protocol):
    @property
    def parameter(self) -> Tensor | None: ...

    def mask(self, step: int, total: int, /) -> Tensor: ...

    def temperature(self, step: int, total: int, /) -> float | None: ...

    def step(self, gradient: Tensor | None) -> None: ...

    def finish(self) -> tuple[Tensor, Tensor, tuple[float, float], int]: ...


class FixedRankSearch(FrozenValue):
    """A search over a fixed binary prefix of a learned rotation."""

    __slots__ = ("_null", "_objective", "_rank", "_schedule")

    _noun = "fixed-rank search"

    def __init__(
        self,
        *,
        rank: int,
        schedule: Schedule,
        null: LabelPermutation[int],
        objective: AlignmentObjective,
    ) -> None:
        held_rank = ensure_non_negative("a fixed search's rank", rank)
        held_schedule, held_null = _search_parts(schedule, null)

        if held_rank == 0:
            msg = "a fixed search of rank zero swaps nothing"
            raise ValueError(msg)
        if not isinstance(objective, AlignmentObjective):
            msg = f"a fixed search objective is not one: {objective!r}"
            raise TypeError(msg)
        object.__setattr__(self, "_rank", held_rank)
        object.__setattr__(self, "_schedule", held_schedule)
        object.__setattr__(self, "_null", held_null)
        object.__setattr__(self, "_objective", objective)

    @property
    def rank(self) -> int:
        """Return the number of leading coordinates the search swaps."""
        return self._rank

    @property
    def schedule(self) -> Schedule:
        """Return the optimizer schedule."""
        return self._schedule

    @property
    def null(self) -> LabelPermutation[int]:
        """Return the label permutation used for refitting."""
        return self._null

    @property
    def objective(self) -> AlignmentObjective:
        """Return the declared loss this search optimizes."""
        return self._objective

    @property
    def fit_unit(self) -> ProbeUnit:
        """Return the measured pricing unit for one fit."""
        return ProbeUnit.FIXED_ALIGNMENT_FIT

    @property
    def refit_unit(self) -> ProbeUnit:
        """Return the measured pricing unit for one null refit."""
        return ProbeUnit.FIXED_PERMUTATION_REFIT

    @property
    def initial_width(self) -> int:
        """Return the fixed discrete width."""
        return self._rank

    def start(self, like: Tensor) -> _MaskState:
        """Return the fixed mask state for a reading's width and numeric policy."""
        return _FixedMask(like, self._rank)

    def to_json(self) -> dict[str, JsonValue]:
        """Return this strategy as deterministic JSON-ready data."""
        return {"kind": "fixed-rank", "rank": self._rank, **_shared_json(self)}

    @override
    def __repr__(self) -> str:
        return (
            f"FixedRankSearch(rank={self._rank!r}, schedule={self._schedule!r}, "
            f"null={self._null!r}, objective={self._objective!r})"
        )


class BoundlessSearch(FrozenValue):
    """A search learning one ordered continuous interval in a rotated basis."""

    __slots__ = (
        "_annealing",
        "_boundaries",
        "_margin",
        "_null",
        "_objective",
        "_schedule",
    )

    _noun = "boundless search"

    def __init__(
        self,
        *,
        boundaries: tuple[float, float],
        margin: float,
        annealing: Annealing,
        schedule: Schedule,
        null: LabelPermutation[int],
        objective: AlignmentObjective,
    ) -> None:
        held_margin = ensure_finite("a boundary share margin", margin)
        if not 0.0 < held_margin < 0.5:
            msg = f"a boundary share margin lies in (0, 0.5), not {margin!r}"
            raise ValueError(msg)
        if type(boundaries) is not tuple or len(boundaries) != 2:
            msg = f"a boundless search needs two boundary indices, not {boundaries!r}"
            raise TypeError(msg)
        lower = ensure_finite("a lower boundary", boundaries[0])
        upper = ensure_finite("an upper boundary", boundaries[1])

        if lower < 0.0:
            msg = f"a lower boundary is outside the coordinate range: {lower!r}"
            raise ValueError(msg)
        if lower >= upper:
            msg = f"boundaries are not ordered: {(lower, upper)!r}"
            raise ValueError(msg)
        if type(annealing) is not Annealing:
            msg = f"a boundless search anneals under an Annealing, not {annealing!r}"
            raise TypeError(msg)
        if not isinstance(objective, AlignmentObjective):
            msg = f"a boundless search objective is not one: {objective!r}"
            raise TypeError(msg)

        held_schedule, held_null = _search_parts(schedule, null)
        object.__setattr__(self, "_boundaries", (lower, upper))
        object.__setattr__(self, "_margin", held_margin)
        object.__setattr__(self, "_annealing", annealing)
        object.__setattr__(self, "_schedule", held_schedule)
        object.__setattr__(self, "_null", held_null)
        object.__setattr__(self, "_objective", objective)

    @property
    def boundaries(self) -> tuple[float, float]:
        """Return the initial ordered boundary indices."""
        return self._boundaries

    @property
    def margin(self) -> float:
        """Return how far an initial boundary share is held from zero and one."""
        return self._margin

    @property
    def annealing(self) -> Annealing:
        """Return the temperature schedule for the interval mask."""
        return self._annealing

    @property
    def schedule(self) -> Schedule:
        """Return the optimizer schedule."""
        return self._schedule

    @property
    def null(self) -> LabelPermutation[int]:
        """Return the label permutation used for refitting."""
        return self._null

    @property
    def objective(self) -> AlignmentObjective:
        """Return the declared loss this search optimizes."""
        return self._objective

    @property
    def fit_unit(self) -> ProbeUnit:
        """Return the measured pricing unit for one fit."""
        return ProbeUnit.BOUNDLESS_ALIGNMENT_FIT

    @property
    def refit_unit(self) -> ProbeUnit:
        """Return the measured pricing unit for one null refit."""
        return ProbeUnit.BOUNDLESS_PERMUTATION_REFIT

    @property
    def initial_width(self) -> int:
        """Return how many integer coordinates lie in the initial interval."""
        lower, upper = self._boundaries
        return max(0, ceil(upper) - ceil(lower))

    def start(self, like: Tensor) -> _MaskState:
        """Return trainable interval state for a reading's width and numeric policy."""
        return _BoundlessMask(
            like,
            self._boundaries,
            self._margin,
            self._annealing,
            self._schedule.learning_rate,
        )

    def to_json(self) -> dict[str, JsonValue]:
        """Return this strategy as deterministic JSON-ready data."""
        return {
            "kind": "boundless",
            "boundaries": list(self._boundaries),
            "margin": self._margin,
            "annealing": {
                "initial": self._annealing.initial,
                "final": self._annealing.final,
            },
            **_shared_json(self),
        }

    @override
    def __repr__(self) -> str:
        return (
            f"BoundlessSearch(boundaries={self._boundaries!r}, "
            f"margin={self._margin!r}, annealing={self._annealing!r}, "
            f"schedule={self._schedule!r}, "
            f"null={self._null!r}, objective={self._objective!r})"
        )


type SearchStrategy = FixedRankSearch | BoundlessSearch


def _shared_json(search: SearchStrategy) -> dict[str, JsonValue]:
    schedule = search.schedule
    null = search.null
    replicates = cast("int", null.replicates)
    return {
        "objective": search.objective.value,
        "schedule": {
            "epochs": schedule.epochs,
            "learning_rate": schedule.learning_rate,
            "batch_size": schedule.batch_size,
            "seed": schedule.seed,
        },
        "null": {"name": null.name, "replicates": replicates, "seed": null.seed},
    }


class _FixedMask:
    __slots__ = ("_boundaries", "_mask", "_width")

    def __init__(self, like: Tensor, rank: int) -> None:
        side = int(like.shape[-1])
        if rank > side:
            msg = f"a fixed rank {rank} exceeds the site's width {side}"
            raise ValueError(msg)
        coordinates = torch.arange(side, device=like.device)
        self._mask = (coordinates < rank).to(dtype=like.dtype)
        self._boundaries = (0.0, float(rank))
        self._width = rank

    @property
    def parameter(self) -> Tensor | None:
        return None

    def mask(self, _step: int, _total: int, /) -> Tensor:
        return self._mask

    def temperature(self, _step: int, _total: int, /) -> float | None:
        return None

    def step(self, gradient: Tensor | None) -> None:
        if gradient is not None:
            msg = "a fixed mask has no boundary gradient"
            raise ValueError(msg)

    def finish(self) -> tuple[Tensor, Tensor, tuple[float, float], int]:
        discrete = self._mask.to(dtype=torch.bool)
        return self._mask, discrete, self._boundaries, self._width


def _logit(value: float, margin: float) -> float:
    held = min(max(value, margin), 1.0 - margin)
    return log(held / (1.0 - held))


class _BoundlessMask:
    __slots__ = ("_annealing", "_optimizer", "_raw", "_side")

    def __init__(
        self,
        like: Tensor,
        boundaries: tuple[float, float],
        margin: float,
        annealing: Annealing,
        learning_rate: float,
    ) -> None:
        side = int(like.shape[-1])
        lower, upper = boundaries
        if upper > side:
            msg = f"an upper boundary {upper!r} is outside a site of width {side}"
            raise ValueError(msg)
        lower_share = lower / side
        width_share = (upper - lower) / (side - lower)
        self._raw = torch.tensor(
            [_logit(lower_share, margin), _logit(width_share, margin)],
            device=like.device,
            dtype=like.dtype,
            requires_grad=True,
        )
        self._side = side
        self._annealing = annealing
        self._optimizer = torch.optim.Adam([self._raw], lr=learning_rate)

    @property
    def parameter(self) -> Tensor | None:
        return self._raw

    def _boundaries(self) -> tuple[Tensor, Tensor]:
        shares = torch.sigmoid(self._raw)
        lower = self._side * shares[0]
        upper = lower + (self._side - lower) * shares[1]
        return lower, upper

    def mask(self, step: int, total: int, /) -> Tensor:
        temperature = self.temperature(step, total)
        lower, upper = self._boundaries()
        coordinates = torch.arange(
            self._side, device=self._raw.device, dtype=self._raw.dtype
        )
        return torch.sigmoid((coordinates - lower) / temperature) * torch.sigmoid(
            (upper - coordinates) / temperature
        )

    def temperature(self, step: int, total: int, /) -> float:
        return self._annealing.temperature(step, total)

    def step(self, gradient: Tensor | None) -> None:
        if gradient is None:
            msg = "no gradient reaches the continuous boundaries"
            raise ValueError(msg)
        self._optimizer.zero_grad(set_to_none=True)
        self._raw.grad = gradient
        self._optimizer.step()

    def finish(self) -> tuple[Tensor, Tensor, tuple[float, float], int]:
        lower, upper = self._boundaries()
        if not bool(torch.isfinite(torch.stack((lower, upper))).all()):
            msg = "a learned boundary is not finite"
            raise ValueError(msg)
        coordinates = torch.arange(
            self._side, device=self._raw.device, dtype=self._raw.dtype
        )
        temperature = self._annealing.final
        mask = (
            torch.sigmoid((coordinates - lower) / temperature)
            * torch.sigmoid((upper - coordinates) / temperature)
        ).detach()
        discrete = ((coordinates >= lower) & (coordinates < upper)).detach()
        boundaries = (float(lower.detach()), float(upper.detach()))
        return mask, discrete, boundaries, int(discrete.sum().item())


class FitStep(FrozenValue):
    """One optimizer step's temperature, loss, and two gradient norms."""

    __slots__ = (
        "_boundary_gradient_norm",
        "_loss",
        "_rotation_gradient_norm",
        "_step",
        "_temperature",
    )

    _noun = "fit step"

    def __init__(
        self,
        step: int,
        temperature: float | None,
        loss: float,
        rotation_gradient_norm: float,
        boundary_gradient_norm: float | None,
    ) -> None:
        object.__setattr__(self, "_step", ensure_non_negative("a trace step", step))
        held_temperature = (
            None
            if temperature is None
            else ensure_finite("a trace temperature", temperature)
        )
        if held_temperature is not None and held_temperature <= 0.0:
            msg = f"a trace temperature is not positive: {temperature!r}"
            raise ValueError(msg)
        object.__setattr__(
            self,
            "_temperature",
            held_temperature,
        )
        held_loss = ensure_finite("a trace loss", loss)
        rotation_norm = ensure_finite(
            "a rotation gradient norm", rotation_gradient_norm
        )
        boundary_norm = (
            None
            if boundary_gradient_norm is None
            else ensure_finite("a boundary gradient norm", boundary_gradient_norm)
        )
        if rotation_norm < 0.0:
            msg = f"a rotation gradient norm is negative: {rotation_gradient_norm!r}"
            raise ValueError(msg)
        if boundary_norm is not None and boundary_norm < 0.0:
            msg = f"a boundary gradient norm is negative: {boundary_gradient_norm!r}"
            raise ValueError(msg)
        object.__setattr__(self, "_loss", held_loss)
        object.__setattr__(
            self,
            "_rotation_gradient_norm",
            rotation_norm,
        )
        object.__setattr__(
            self,
            "_boundary_gradient_norm",
            boundary_norm,
        )

    @property
    def step(self) -> int:
        """Return the zero-based optimizer step."""
        return self._step

    @property
    def temperature(self) -> float | None:
        """Return the mask temperature, or None for a binary fixed mask."""
        return self._temperature

    @property
    def loss(self) -> float:
        """Return the declared objective loss before this step."""
        return self._loss

    @property
    def rotation_gradient_norm(self) -> float:
        """Return the rotation generator's gradient norm."""
        return self._rotation_gradient_norm

    @property
    def boundary_gradient_norm(self) -> float | None:
        """Return the boundary gradient norm, or None for fixed-rank search."""
        return self._boundary_gradient_norm

    def _identity(self) -> tuple[int, float | None, float, float, float | None]:
        return (
            self._step,
            self._temperature,
            self._loss,
            self._rotation_gradient_norm,
            self._boundary_gradient_norm,
        )

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not FitStep:
            return NotImplemented
        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        return hash((FitStep, self._identity()))

    @override
    def __repr__(self) -> str:
        return (
            f"FitStep({self._step!r}, {self._temperature!r}, {self._loss!r}, "
            f"{self._rotation_gradient_norm!r}, "
            f"{self._boundary_gradient_norm!r})"
        )


class Fitted[PairT: CausalPair = CounterfactualPair](FrozenValue):
    """A fitted rotation, interval masks, ordered boundaries, and fit trace."""

    __slots__ = (
        "_boundaries",
        "_discrete_mask",
        "_discrete_width",
        "_mask",
        "_pairs",
        "_rotation",
        "_trace",
    )

    _noun = "fitted alignment"

    def __init__(
        self,
        rotation: Tensor,
        mask: Tensor,
        discrete_mask: Tensor,
        boundaries: tuple[float, float],
        discrete_width: int,
        trace: Sequence[FitStep],
        pairs: Sequence[PairT],
    ) -> None:
        if not isinstance(rotation, torch.Tensor) or not rotation.is_floating_point():
            msg = f"a fitted rotation is a floating point tensor, not {rotation!r}"
            raise TypeError(msg)
        if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
            msg = (
                "a fitted rotation is a square matrix, not one of shape "
                f"{tuple(rotation.shape)}"
            )
            raise ValueError(msg)
        side = int(rotation.shape[0])
        if not isinstance(mask, torch.Tensor) or not mask.is_floating_point():
            msg = f"a fitted mask is a floating point tensor, not {mask!r}"
            raise TypeError(msg)
        if tuple(mask.shape) != (side,):
            msg = f"a fitted mask has shape {tuple(mask.shape)}, not {(side,)}"
            raise ValueError(msg)
        if mask.dtype != rotation.dtype or mask.device != rotation.device:
            msg = "a fitted mask does not share its rotation's dtype and placement"
            raise ValueError(msg)
        if not bool(torch.isfinite(mask).all()) or bool(
            ((mask < 0.0) | (mask > 1.0)).any()
        ):
            msg = "a fitted mask is not finite within zero to one"
            raise ValueError(msg)
        if (
            not isinstance(discrete_mask, torch.Tensor)
            or discrete_mask.dtype is not torch.bool
        ):
            msg = "a fitted discrete mask is a boolean tensor"
            raise TypeError(msg)
        if (
            tuple(discrete_mask.shape) != (side,)
            or discrete_mask.device != rotation.device
        ):
            msg = "a fitted discrete mask does not match its rotation"
            raise ValueError(msg)
        if type(boundaries) is not tuple or len(boundaries) != 2:
            msg = f"a fitted interval needs two boundaries, not {boundaries!r}"
            raise TypeError(msg)
        lower = ensure_finite("a fitted lower boundary", boundaries[0])
        upper = ensure_finite("a fitted upper boundary", boundaries[1])
        if not 0.0 <= lower < upper <= side:
            msg = f"fitted boundaries are unordered or out of range: {(lower, upper)!r}"
            raise ValueError(msg)
        width = ensure_non_negative("a fitted discrete width", discrete_width)
        counted = int(discrete_mask.sum().item())
        if width != counted:
            msg = (
                f"a fitted width {width} does not count its discrete mask's "
                f"{counted} coordinates"
            )
            raise ValueError(msg)
        held_trace = tuple(trace)
        if not held_trace or any(type(step) is not FitStep for step in held_trace):
            msg = "a fitted alignment retains at least one FitStep"
            raise ValueError(msg)
        if tuple(step.step for step in held_trace) != tuple(range(len(held_trace))):
            msg = "a fitted trace's step indices are not contiguous from zero"
            raise ValueError(msg)
        if not pairs:
            msg = "an alignment was fitted to no pair at all"
            raise ValueError(msg)

        object.__setattr__(self, "_rotation", rotation.detach().clone())
        object.__setattr__(self, "_mask", mask.detach().clone())
        object.__setattr__(self, "_discrete_mask", discrete_mask.detach().clone())
        object.__setattr__(self, "_boundaries", (lower, upper))
        object.__setattr__(self, "_discrete_width", width)
        object.__setattr__(self, "_trace", held_trace)
        object.__setattr__(self, "_pairs", tuple(pairs))

    @property
    def rotation(self) -> Tensor:
        """Return the fitted orthogonal rotation."""
        return self._rotation.clone()

    @property
    def mask(self) -> Tensor:
        """Return the final continuous interval mask."""
        return self._mask.clone()

    @property
    def discrete_mask(self) -> Tensor:
        """Return the final interval discretized over coordinates."""
        return self._discrete_mask.clone()

    @property
    def boundaries(self) -> tuple[float, float]:
        """Return the final ordered continuous boundary indices."""
        return self._boundaries

    @property
    def discrete_width(self) -> int:
        """Return how many coordinates the final discrete interval contains."""
        return self._discrete_width

    @property
    def aligned(self) -> bool:
        """Return whether the learned interval contains at least one coordinate."""
        return self._discrete_width > 0

    @property
    def trace(self) -> tuple[FitStep, ...]:
        """Return every optimizer step in order."""
        return self._trace

    @property
    def pairs(self) -> tuple[PairT, ...]:
        """Return the pairs the alignment was fitted to."""
        return self._pairs

    @override
    def __repr__(self) -> str:
        side = int(self._rotation.shape[0])
        return (
            f"Fitted(<{side}x{side}>, width={self._discrete_width}, "
            f"pairs={len(self._pairs)}, steps={len(self._trace)})"
        )


class Assessment:
    """An alignment's held-out accuracy and p-value under its declared null."""

    __slots__ = ("_p_value", "_result")

    def __init__(self, result: IIAResult, p_value: PValue) -> None:
        self._result = result
        self._p_value = p_value

    @property
    def result(self) -> IIAResult:
        """Return the accuracy, cell, discrete width, and null."""
        return self._result

    @property
    def p_value(self) -> PValue:
        """Return how often a null refit did as well."""
        return self._p_value

    @override
    def __repr__(self) -> str:
        return (
            f"Assessment(accuracy={self._result.accuracy!r}, "
            f"p_value={self._p_value.value!r})"
        )


class _Parametrized:
    __slots__ = ("_mask", "_rotation")

    def __init__(
        self,
        like: Tensor,
        learning_rate: float,
        draws: torch.Generator,
        mask: _MaskState,
    ) -> None:
        self._rotation = OrthogonalOptimizer(like, learning_rate, draws)
        self._mask = mask

    def rotation(self) -> Tensor:
        return self._rotation.matrix()

    def mask(self, step: int, total: int) -> Tensor:
        return self._mask.mask(step, total)

    def temperature(self, step: int, total: int) -> float | None:
        return self._mask.temperature(step, total)

    def gradients_of(self, loss: Tensor) -> tuple[Tensor, Tensor | None]:
        boundary = self._mask.parameter
        gradients = self._rotation.gradients_of(
            loss, () if boundary is None else (boundary,)
        )
        return gradients[0], None if boundary is None else gradients[1]

    def step(self, rotation: Tensor, boundary: Tensor | None) -> None:
        self._rotation.step(rotation)
        self._mask.step(boundary)

    def finish(
        self,
    ) -> tuple[Tensor, Tensor, Tensor, tuple[float, float], int]:
        mask, discrete, boundaries, width = self._mask.finish()
        return self._rotation.finish(), mask, discrete, boundaries, width


class DistributedAlignment[ReadingT: Reading, PairT: CausalPair = CounterfactualPair]:
    """One fixed-rank or boundless rotated alignment at a published model site."""

    __slots__ = ("_fitted", "_search", "_site", "_subject", "_variable")

    def __init__(
        self,
        subject: Subject[ReadingT, PairT],
        site: Site,
        variable: HighLevelVariable,
        *,
        search: SearchStrategy,
        fitted: Fitted[PairT] | None = None,
    ) -> None:
        if not isinstance(site, Site):
            msg = f"an alignment is read at a Site, not {site!r}"
            raise TypeError(msg)
        if type(variable) not in {TypeAt, SplitAt, CombinatorAt, NamedVariable}:
            msg = f"an alignment stands for a high-level variable, not {variable!r}"
            raise TypeError(msg)
        if not isinstance(search, FixedRankSearch | BoundlessSearch):
            msg = f"an alignment requires a search strategy, not {search!r}"
            raise TypeError(msg)
        if fitted is not None and type(fitted) is not Fitted:
            msg = f"an alignment holds a Fitted result or None, not {fitted!r}"
            raise TypeError(msg)
        self._subject = subject
        self._site = site
        self._variable = variable
        self._search = search
        self._fitted = fitted

    @property
    def policy(self) -> AlignmentPolicy:
        """Return the span-to-token policy of the subject."""
        return self._subject.policy

    @property
    def rank(self) -> int:
        """Return the fitted discrete width, or the strategy's initial width."""
        if self._fitted is not None:
            return self._fitted.discrete_width
        return self._search.initial_width

    @property
    def site(self) -> Site:
        """Return the published point holding the alignment."""
        return self._site

    @property
    def variable(self) -> HighLevelVariable:
        """Return the high-level decision the alignment stands for."""
        return self._variable

    @property
    def search(self) -> SearchStrategy:
        """Return the fixed-rank or boundless search strategy."""
        return self._search

    @property
    def fitted(self) -> Fitted[PairT] | None:
        """Return the fit result, if this alignment has been fitted."""
        return self._fitted

    def fit(self, pairs: Sequence[PairT]) -> Self:
        """Return a new alignment fitted to the supplied causal pairs."""
        held = tuple(pairs)
        labels = self._labels(held)
        self._ensure_access()
        fitted = self._fitted_to(held, labels)
        return type(self)(
            self._subject,
            self._site,
            self._variable,
            search=self._search,
            fitted=fitted,
        )

    def apply(self, variable: HighLevelVariable) -> frozenset[Unit]:
        """Return the aligned site for the fitted variable, if its width is nonzero."""
        if (
            self._fitted is None
            or not self._fitted.aligned
            or variable != self._variable
        ):
            return frozenset()
        return frozenset({Unit(self._site)})

    def matches(self, pairs: Sequence[PairT]) -> tuple[bool, ...]:
        """Return one held-out interchange outcome per supplied pair."""
        fitted = self._require_fitted("matches")
        held = tuple(pairs)
        labels = self._labels(held)
        return self._matches(fitted, held, labels)

    def moves_output(
        self, targets: Sequence[PairT], sources: Sequence[PairT], /
    ) -> tuple[bool, ...]:
        """Return whether each source patch changes its target's argmax output.

        Parameters
        ----------
        targets : Sequence[PairT]
            Pairs whose base-side outputs are patched and compared.
        sources : Sequence[PairT]
            Pairs whose variant-side readings supply the patches, position by
            position.

        Returns
        -------
        tuple[bool, ...]
            One movement outcome per target and source pair.

        Raises
        ------
        ValueError
            If the alignment is unfitted, either population is empty, the
            populations differ in size, or a pair varies another variable.
        """
        fitted = self._require_fitted("output movement")
        held_targets = tuple(targets)
        held_sources = tuple(sources)
        self._labels(held_targets)
        self._labels(held_sources)
        if len(held_targets) != len(held_sources):
            msg = (
                f"output movement pairs {len(held_targets)} targets with "
                f"{len(held_sources)} sources"
            )
            raise ValueError(msg)
        base, patched = self._output_predictions(
            fitted, held_targets, held_sources, compare_base=True
        )
        return tuple(
            before != after for before, after in zip(base, patched, strict=True)
        )

    def assess(self, pairs: Sequence[PairT]) -> Assessment:
        """Return held-out accuracy and its refitted label-permutation p-value."""
        fitted = self._require_fitted("accuracy")
        held = tuple(pairs)
        labels = self._labels(held)
        observed = self._accuracy(fitted, held, labels)
        fitting = fitted.pairs
        null = self._search.null

        def refitted(exchanged: Sequence[int]) -> float:
            found = self._fitted_to(fitting, torch.tensor(exchanged))
            return self._accuracy(found, held, labels)

        reference = null.reference(refitted, self._labels(fitting).tolist())
        p_value = p_value_against_replicates(
            observed,
            reference,
            tail="greater",
            null=null.name,
            family=null.family,
            exact=False,
            seed=null.seed,
        )
        result = IIAResult(
            observed, self._variable, fitted.discrete_width, p_value.provenance
        )
        return Assessment(result, p_value)

    def iia(self, pairs: Sequence[PairT]) -> IIAResult:
        """Return held-out interchange-intervention accuracy."""
        return self.assess(pairs).result

    def _ensure_access(self) -> None:
        published = self._subject.site_map()
        if self._site not in published:
            msg = f"{self._site!r} is not published by the alignment subject"
            raise ValueError(msg)
        specification = published[self._site]
        if Access.OBSERVE not in specification.access:
            msg = f"{self._site!r} does not permit observation"
            raise ValueError(msg)
        if Access.PATCH not in specification.access:
            msg = f"{self._site!r} does not permit patching"
            raise ValueError(msg)
        if not specification.differentiable:
            msg = f"{self._site!r} is declared non-differentiable"
            raise ValueError(msg)

    def _require_fitted(self, report: str) -> Fitted[PairT]:
        if self._fitted is None:
            msg = f"an alignment that has not been fitted has no {report} to report"
            raise ValueError(msg)
        return self._fitted

    def _labels(self, pairs: tuple[PairT, ...]) -> Tensor:
        if not pairs:
            msg = "an alignment is fitted or tested on at least one pair"
            raise ValueError(msg)
        decisions = self._subject.decisions(self._variable)
        rows: list[int] = []
        for pair in pairs:
            if pair.varied != self._variable:
                msg = (
                    f"an alignment for {self._variable!r} was given a pair "
                    f"varying {pair.varied!r}"
                )
                raise ValueError(msg)
            decided = pair.variant[self._variable]
            if decided not in decisions:
                msg = (
                    f"a pair decides {self._variable!r} as {decided!r}, which the "
                    f"subject does not score among {decisions!r}"
                )
                raise ValueError(msg)
            rows.append(decisions.index(decided))
        return torch.tensor(rows)

    def _sources(self, pairs: tuple[PairT, ...]) -> Tensor:
        with (
            torch.no_grad(),
            instrument(self._subject, observe=frozenset({self._site})) as run,
        ):
            run(lambda: self._subject.run(pairs, Side.VARIANT))
        return run.trace[self._site].tensor

    def _scores(
        self, pairs: tuple[PairT, ...], rotation: Tensor, mask: Tensor, sources: Tensor
    ) -> Tensor:
        interchange = Interchange(rotation, mask, sources)
        with instrument(self._subject, patch={self._site: interchange}) as run:
            scores = run(lambda: self._subject.run(pairs, Side.BASE))
        expected = (len(pairs), len(self._subject.decisions(self._variable)))
        if tuple(scores.shape) != expected:
            msg = (
                f"the subject scored {len(pairs)} pairs in the shape "
                f"{tuple(scores.shape)} rather than one row per pair over its "
                f"decisions, {expected}"
            )
            raise ValueError(msg)
        return scores

    def _batches(
        self, pairs: tuple[PairT, ...], order: Tensor
    ) -> tuple[tuple[tuple[PairT, ...], Tensor], ...]:
        size = self._search.schedule.batch_size
        batches: list[tuple[tuple[PairT, ...], Tensor]] = []
        for start in range(0, len(pairs), size):
            positions = order[start : start + size]
            batches.append((tuple(pairs[at] for at in positions.tolist()), positions))
        return tuple(batches)

    def _steps(
        self, pairs: tuple[PairT, ...], labels: Tensor, draws: torch.Generator
    ) -> Iterator[tuple[tuple[PairT, ...], Tensor, Tensor]]:
        for _ in range(self._search.schedule.epochs):
            order = torch.randperm(len(pairs), generator=draws)
            for batch, positions in self._batches(pairs, order):
                yield batch, labels[positions], self._sources(batch)

    def _one_step(
        self,
        fit: _Parametrized,
        batch: tuple[PairT, ...],
        targets: Tensor,
        sources: Tensor,
        step: int,
        total: int,
    ) -> FitStep:
        scores = self._scores(batch, fit.rotation(), fit.mask(step, total), sources)
        loss = self._search.objective.loss(scores, targets.to(scores.device))
        rotation_gradient, boundary_gradient = fit.gradients_of(loss)
        record = FitStep(
            step,
            fit.temperature(step, total),
            float(loss.detach()),
            float(rotation_gradient.detach().norm()),
            None
            if boundary_gradient is None
            else float(boundary_gradient.detach().norm()),
        )
        fit.step(rotation_gradient, boundary_gradient)
        return record

    def _fitted_to(self, pairs: tuple[PairT, ...], labels: Tensor) -> Fitted[PairT]:
        schedule = self._search.schedule
        draws = torch.Generator().manual_seed(schedule.seed)
        steps = self._steps(pairs, labels, draws)
        batch, targets, sources = next(steps)
        mask = self._search.start(sources)
        fit = _Parametrized(sources, schedule.learning_rate, draws, mask)
        total = schedule.epochs * ceil(len(pairs) / schedule.batch_size)
        trace = [self._one_step(fit, batch, targets, sources, 0, total)]
        for index, (batch, targets, sources) in enumerate(steps, start=1):
            trace.append(self._one_step(fit, batch, targets, sources, index, total))
        rotation, final_mask, discrete, boundaries, width = fit.finish()
        return Fitted(
            rotation,
            final_mask,
            discrete,
            boundaries,
            width,
            trace,
            pairs,
        )

    def _accuracy(
        self, fitted: Fitted[PairT], pairs: tuple[PairT, ...], labels: Tensor
    ) -> float:
        matched = self._matches(fitted, pairs, labels)
        return sum(matched) / len(matched)

    def _matches(
        self, fitted: Fitted[PairT], pairs: tuple[PairT, ...], labels: Tensor
    ) -> tuple[bool, ...]:
        _, predicted = self._output_predictions(
            fitted, pairs, pairs, compare_base=False
        )
        return tuple(
            found == wanted
            for found, wanted in zip(predicted, labels.tolist(), strict=True)
        )

    def _output_predictions(
        self,
        fitted: Fitted[PairT],
        targets: tuple[PairT, ...],
        sources: tuple[PairT, ...],
        *,
        compare_base: bool,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return unpatched and source-patched argmax decisions in target order."""
        order = torch.arange(len(targets))
        base_predictions = [0] * len(targets) if compare_base else []
        patched_predictions = [0] * len(targets)
        mask = fitted.discrete_mask.to(dtype=fitted.rotation.dtype)
        rotation = fitted.rotation
        with torch.no_grad():
            for targets_batch, positions in self._batches(targets, order):
                sources_batch = tuple(sources[at] for at in positions.tolist())
                patched_scores = self._scores(
                    targets_batch,
                    rotation,
                    mask,
                    self._sources(sources_batch),
                )
                patched = patched_scores.argmax(dim=-1).tolist()
                for position, after in zip(positions.tolist(), patched, strict=True):
                    patched_predictions[position] = after
                if compare_base:
                    base_scores = self._subject.run(targets_batch, Side.BASE)
                    expected = (
                        len(targets_batch),
                        len(self._subject.decisions(self._variable)),
                    )
                    if tuple(base_scores.shape) != expected:
                        msg = (
                            f"the subject scored {len(targets_batch)} pairs in the "
                            f"shape {tuple(base_scores.shape)} rather than one row per "
                            f"pair over its decisions, {expected}"
                        )
                        raise ValueError(msg)
                    bases = base_scores.argmax(dim=-1).tolist()
                    for position, before in zip(positions.tolist(), bases, strict=True):
                        base_predictions[position] = before
        return tuple(base_predictions), tuple(patched_predictions)

    @override
    def __repr__(self) -> str:
        return (
            f"DistributedAlignment({self._site!r}, {self._variable!r}, "
            f"width={self.rank!r}, fitted={self._fitted is not None!r})"
        )
