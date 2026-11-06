"""An optimization policy over a torch optimizer under a run's device policy."""

from contextlib import contextmanager
from copy import copy, deepcopy
from math import isfinite
from typing import TYPE_CHECKING, Never, cast, final, override

import torch
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_

from lofi.train._optimization import OptimizationResult
from lofi.train._resume import (
    _copy_state_tree,
    _validate_compatible_state,
    _validate_state_tree,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from torch import Tensor
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from lofi.nn.policy import DevicePolicy
    from lofi.train._memory import StepMemory
    from lofi.train._store import StateTree

__all__ = ["AutocastPolicy", "GradScaling", "OptimizationState", "StepOptions"]


@final
class GradScaling:
    """Dynamic gradient scaling parameters.

    Parameters
    ----------
    init_scale : float
        The scale a fresh run starts at.
    growth_factor : float
        What the scale is multiplied by after a run of finite steps.
    backoff_factor : float
        What the scale is multiplied by on a step with nonfinite gradients.
    growth_interval : int
        How many consecutive finite steps precede a growth.
    """

    __slots__ = ("_backoff_factor", "_growth_factor", "_growth_interval", "_init_scale")

    def __init__(
        self,
        *,
        init_scale: float,
        growth_factor: float,
        backoff_factor: float,
        growth_interval: int,
    ) -> None:
        object.__setattr__(self, "_init_scale", init_scale)
        object.__setattr__(self, "_growth_factor", growth_factor)
        object.__setattr__(self, "_backoff_factor", backoff_factor)
        object.__setattr__(self, "_growth_interval", growth_interval)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "gradient-scaling parameters are immutable"

        raise AttributeError(msg)

    @property
    def init_scale(self) -> float:
        """Return the scale a fresh run starts at."""
        return self._init_scale

    @property
    def growth_factor(self) -> float:
        """Return the growth factor."""
        return self._growth_factor

    @property
    def backoff_factor(self) -> float:
        """Return the backoff factor."""
        return self._backoff_factor

    @property
    def growth_interval(self) -> int:
        """Return the growth interval."""
        return self._growth_interval

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value scales identically."""
        if type(other) is not GradScaling:
            return NotImplemented

        return (
            self._init_scale == other._init_scale
            and self._growth_factor == other._growth_factor
            and self._backoff_factor == other._backoff_factor
            and self._growth_interval == other._growth_interval
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the four parameters."""
        return hash(
            (
                GradScaling,
                self._init_scale,
                self._growth_factor,
                self._backoff_factor,
                self._growth_interval,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds these parameters."""
        return (
            f"GradScaling(init_scale={self._init_scale!r}, "
            f"growth_factor={self._growth_factor!r}, "
            f"backoff_factor={self._backoff_factor!r}, "
            f"growth_interval={self._growth_interval!r})"
        )


@final
class StepOptions:
    """How an optimization policy takes its steps.

    Parameters
    ----------
    accumulation_steps : int
        How many accumulation units make one optimizer step.
    max_grad_norm : float | None
        The norm gradients are clipped to before each optimizer step, or None to
        leave them unclipped.
    scaling : GradScaling | None
        The dynamic gradient scaling to run under, or None to scale nothing.

    Raises
    ------
    ValueError
        If the accumulation count is below one, or the clipping norm is not a
        positive finite number.
    """

    __slots__ = ("_accumulation_steps", "_max_grad_norm", "_scaling")

    def __init__(
        self,
        *,
        accumulation_steps: int = 1,
        max_grad_norm: float | None = None,
        scaling: GradScaling | None = None,
    ) -> None:
        if accumulation_steps < 1:
            msg = f"an optimizer step takes at least one unit: {accumulation_steps}"

            raise ValueError(msg)

        if max_grad_norm is not None and not (
            isfinite(max_grad_norm) and max_grad_norm > 0.0
        ):
            msg = f"a clipping norm is positive and finite: {max_grad_norm!r}"

            raise ValueError(msg)

        object.__setattr__(self, "_accumulation_steps", accumulation_steps)
        object.__setattr__(self, "_max_grad_norm", max_grad_norm)
        object.__setattr__(self, "_scaling", scaling)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "step options are immutable"

        raise AttributeError(msg)

    @property
    def accumulation_steps(self) -> int:
        """Return how many accumulation units make one optimizer step."""
        return self._accumulation_steps

    @property
    def max_grad_norm(self) -> float | None:
        """Return the clipping norm, or None where gradients are unclipped."""
        return self._max_grad_norm

    @property
    def scaling(self) -> GradScaling | None:
        """Return the gradient scaling, or None where nothing is scaled."""
        return self._scaling

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value steps identically."""
        if type(other) is not StepOptions:
            return NotImplemented

        return (
            self._accumulation_steps == other._accumulation_steps
            and self._max_grad_norm == other._max_grad_norm
            and self._scaling == other._scaling
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the three options."""
        return hash(
            (StepOptions, self._accumulation_steps, self._max_grad_norm, self._scaling)
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds these options."""
        return (
            f"StepOptions(accumulation_steps={self._accumulation_steps!r}, "
            f"max_grad_norm={self._max_grad_norm!r}, scaling={self._scaling!r})"
        )


@final
class OptimizationState:
    """The saved state of an optimization policy, one block per stateful part.

    Parameters
    ----------
    optimizer : dict[str, StateTree]
        The optimizer's state.
    scaler : dict[str, StateTree] | None
        The gradient scaler's state, or None for a policy that scales nothing.
    scheduler : dict[str, StateTree] | None
        The learning rate scheduler's state, or None for a policy without one.
    """

    __slots__ = ("_optimizer", "_scaler", "_scheduler")

    def __init__(
        self,
        optimizer: dict[str, StateTree],
        *,
        scaler: dict[str, StateTree] | None = None,
        scheduler: dict[str, StateTree] | None = None,
    ) -> None:
        object.__setattr__(self, "_optimizer", optimizer)
        object.__setattr__(self, "_scaler", scaler)
        object.__setattr__(self, "_scheduler", scheduler)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "an optimization state is immutable"

        raise AttributeError(msg)

    @property
    def optimizer(self) -> dict[str, StateTree]:
        """Return the optimizer's state."""
        return self._optimizer

    @property
    def scaler(self) -> dict[str, StateTree] | None:
        """Return the gradient scaler's state, or None where nothing was scaled."""
        return self._scaler

    @property
    def scheduler(self) -> dict[str, StateTree] | None:
        """Return the scheduler's state, or None where there was no scheduler."""
        return self._scheduler

    @override
    def __repr__(self) -> str:
        """Return which parts the state holds a block for."""
        parts = ["optimizer"]

        if self._scaler is not None:
            parts.append("scaler")

        if self._scheduler is not None:
            parts.append("scheduler")

        return f"OptimizationState(parts=[{', '.join(parts)}])"


def _presence(part: str, *, saved: bool, held: bool) -> None:
    """Refuse a saved state whose parts are not the policy's.

    Parameters
    ----------
    part : str
        Which part is compared, for the message.
    saved : bool
        Whether the saved state holds a block for it.
    held : bool
        Whether the policy has that part.

    Raises
    ------
    ValueError
        If exactly one of the two has it.
    """
    if saved and not held:
        msg = f"the saved state holds {part} state and this policy has no {part}"

        raise ValueError(msg)

    if held and not saved:
        msg = f"this policy has a {part} and the saved state holds no {part} state"

        raise ValueError(msg)


def _state_mapping(name: str, state: StateTree, /) -> dict[str, StateTree]:
    """Return one string-keyed state-tree mapping."""
    _validate_state_tree(name, state)

    if not isinstance(state, dict) or any(type(key) is not str for key in state):
        msg = f"{name} state is not keyed by state names"
        raise TypeError(msg)

    return cast("dict[str, StateTree]", state)


def _state_sequence(name: str, state: StateTree, /) -> list[StateTree]:
    """Return one list-valued state-tree field."""
    _validate_state_tree(name, state)

    if not isinstance(state, list):
        msg = f"{name} state is not a list"
        raise TypeError(msg)

    return state


def _parameter_ids(name: str, state: StateTree, /) -> list[int]:
    """Return one optimizer group's ordered parameter identifiers."""
    values = _state_sequence(name, state)

    if any(type(value) is not int for value in values):
        msg = f"{name} state does not contain only integer parameter identifiers"
        raise TypeError(msg)

    return cast("list[int]", values)


def _validate_parameter_state(
    name: str,
    state: dict[str, StateTree],
    parameter: Tensor,
    live: dict[str, StateTree] | None,
    /,
) -> None:
    """Validate one optimizer parameter's state against its live parameter."""
    if live is not None:
        missing = sorted(set(live) - set(state))
        extra = sorted(set(state) - set(live))

        if missing or extra:
            msg = f"{name} has missing keys {missing} and extra keys {extra}"
            raise ValueError(msg)

    for key, value in state.items():
        path = f"{name}[{key!r}]"

        if live is not None:
            _validate_compatible_state(path, value, live[key])

        elif key in {"func_evals", "n_iter", "step"}:
            if isinstance(value, torch.Tensor):
                if value.ndim != 0:
                    msg = f"{path} is not a scalar optimizer control"
                    raise ValueError(msg)

                if value.dtype is torch.bool or value.is_complex():
                    msg = f"{path} is not a real numeric optimizer control"
                    raise ValueError(msg)

            elif type(value) not in {float, int}:
                msg = f"{path} is not a scalar optimizer control"
                raise TypeError(msg)

        elif isinstance(value, torch.Tensor):
            _validate_compatible_state(path, value, parameter)

        else:
            _validate_state_tree(path, value)


def _validate_optimizer_state(optimizer: Optimizer, state: StateTree, /) -> None:
    """Validate optimizer topology and tensor metadata without loading it."""
    saved = _state_mapping("optimizer", state)
    live = _state_mapping("live optimizer", cast("StateTree", optimizer.state_dict()))
    required = {"state", "param_groups"}

    if set(saved) != required:
        missing = sorted(required - set(saved))
        extra = sorted(set(saved) - required)
        msg = f"optimizer state has missing keys {missing} and extra keys {extra}"
        raise ValueError(msg)

    live_groups = _state_sequence("live optimizer param_groups", live["param_groups"])
    saved_groups = _state_sequence("optimizer param_groups", saved["param_groups"])

    if len(saved_groups) != len(live_groups):
        msg = (
            f"optimizer state has {len(saved_groups)} parameter groups; "
            f"the live optimizer has {len(live_groups)}"
        )
        raise ValueError(msg)

    saved_parameters: dict[int, Tensor] = {}
    saved_to_live: dict[int, int] = {}

    for index, (saved_group_state, live_group_state, runtime_group) in enumerate(
        zip(saved_groups, live_groups, optimizer.param_groups, strict=True)
    ):
        saved_group = _state_mapping(
            f"optimizer param_groups[{index}]", saved_group_state
        )
        live_group = _state_mapping(
            f"live optimizer param_groups[{index}]", live_group_state
        )
        missing = sorted(set(live_group) - set(saved_group))
        extra = sorted(set(saved_group) - set(live_group))

        if missing or extra:
            msg = (
                f"optimizer param_groups[{index}] has missing keys {missing} "
                f"and extra keys {extra}"
            )
            raise ValueError(msg)

        saved_ids = _parameter_ids(
            f"optimizer param_groups[{index}]['params']", saved_group["params"]
        )
        live_ids = _parameter_ids(
            f"live optimizer param_groups[{index}]['params']", live_group["params"]
        )
        parameters = cast("list[Tensor]", runtime_group["params"])

        if len(saved_ids) != len(live_ids) or len(saved_ids) != len(parameters):
            msg = (
                f"optimizer param_groups[{index}] has {len(saved_ids)} parameters; "
                f"the live optimizer has {len(parameters)}"
            )
            raise ValueError(msg)

        for saved_id, live_id, parameter in zip(
            saved_ids, live_ids, parameters, strict=True
        ):
            if saved_id in saved_parameters:
                msg = f"optimizer parameter identifier {saved_id} is repeated"
                raise ValueError(msg)

            saved_parameters[saved_id] = parameter
            saved_to_live[saved_id] = live_id

        for key in sorted(set(live_group) - {"params"}):
            _validate_compatible_state(
                f"optimizer param_groups[{index}][{key!r}]",
                saved_group[key],
                live_group[key],
            )

    saved_state = saved["state"]
    live_state = live["state"]
    _validate_state_tree("optimizer state", saved_state)
    _validate_state_tree("live optimizer state", live_state)

    if not isinstance(saved_state, dict) or any(
        type(key) is not int for key in saved_state
    ):
        msg = "optimizer parameter state is not keyed by integer identifiers"
        raise TypeError(msg)

    if not isinstance(live_state, dict) or any(
        type(key) is not int for key in live_state
    ):
        msg = "live optimizer parameter state is not keyed by integer identifiers"
        raise TypeError(msg)

    saved_state_by_id = cast("dict[int, StateTree]", saved_state)
    live_state_by_id = cast("dict[int, StateTree]", live_state)

    for parameter_id, parameter_state in saved_state_by_id.items():
        try:
            parameter = saved_parameters[parameter_id]

        except KeyError as error:
            msg = f"optimizer state names unknown parameter identifier {parameter_id}"
            raise ValueError(msg) from error

        held = _state_mapping(f"optimizer state[{parameter_id}]", parameter_state)
        live_id = saved_to_live[parameter_id]
        live_parameter_state = (
            None
            if live_id not in live_state_by_id
            else _state_mapping(
                f"live optimizer state[{live_id}]", live_state_by_id[live_id]
            )
        )
        _validate_parameter_state(
            f"optimizer state[{parameter_id}]",
            held,
            parameter,
            live_parameter_state,
        )


@final
class AutocastPolicy:
    """Takes optimizer steps under a run's autocast region, scaling when asked.

    Each call to ``step`` runs one accumulation unit: the forward under the device
    policy's autocast region, then a backward over the unit's share of the loss.
    On the last unit of a window the gradients are unscaled where a scaler is in
    use, clipped where a norm is set, stepped, and the scheduler is advanced. The
    scale a step reports is the one its gradients were scaled by, read before the
    scaler updates it.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer over the parameters the loss reaches.
    device_policy : DevicePolicy
        Where the run computes and in what precision.
    options : StepOptions
        Accumulation, clipping and scaling.
    scheduler : LRScheduler | None
        The learning rate scheduler advanced once per optimizer step, or None.

    Raises
    ------
    ValueError
        If scaling is asked of a device family that has no autocast implementation.
    """

    __slots__ = (
        "_device_policy",
        "_memory",
        "_optimizer",
        "_options",
        "_position",
        "_scaler",
        "_scheduler",
    )

    def __init__(
        self,
        optimizer: Optimizer,
        device_policy: DevicePolicy,
        options: StepOptions,
        *,
        scheduler: LRScheduler | None = None,
    ) -> None:
        scaler: GradScaler | None = None
        scaling = options.scaling

        if scaling is not None:
            device_type = device_policy.autocast_device_type

            if device_type is None:
                msg = (
                    f"device {device_policy.device.type!r} has no gradient "
                    f"scaler implementation"
                )

                raise ValueError(msg)

            scaler = GradScaler(
                device_type,
                init_scale=scaling.init_scale,
                growth_factor=scaling.growth_factor,
                backoff_factor=scaling.backoff_factor,
                growth_interval=scaling.growth_interval,
            )

        self._optimizer = optimizer
        self._device_policy = device_policy
        self._options = options
        self._scheduler = scheduler
        self._scaler = scaler
        self._position = 0
        self._memory: StepMemory | None = None

    @property
    def options(self) -> StepOptions:
        """Return how this policy takes its steps."""
        return self._options

    @property
    def scales_gradients(self) -> bool:
        """Return whether a gradient scaler is in use."""
        return self._scaler is not None

    def _clip(self) -> None:
        """Clip the gradients to the configured norm, where one is set."""
        max_norm = self._options.max_grad_norm

        if max_norm is None:
            return

        parameters = [
            parameter
            for group in self._optimizer.param_groups
            for parameter in group["params"]
        ]
        clip_grad_norm_(parameters, max_norm)

    def _finish(self) -> float | None:
        """Take the optimizer step that closes a window and return the scale used."""
        scaler = self._scaler

        if scaler is None:
            self._clip()
            self._optimizer.step()
            scale = None

        else:
            scaler.unscale_(self._optimizer)
            self._clip()
            scaler.step(self._optimizer)
            scale = scaler.get_scale()
            scaler.update()

        if self._scheduler is not None:
            self._scheduler.step()

        return scale

    @property
    def memory(self) -> StepMemory | None:
        """Return the meter steps are measured under, or None outside ``metered``."""
        return self._memory

    @contextmanager
    def metered(self, memory: StepMemory, /) -> Iterator[None]:
        """Measure every step taken inside the block against a step memory bound.

        Parameters
        ----------
        memory : StepMemory
            The meter each step runs under.
        """
        self._memory = memory

        try:
            yield

        finally:
            self._memory = None

    def step(self, forward: Callable[[], Tensor], /) -> OptimizationResult:
        """Run one accumulation unit and report its outcome.

        Parameters
        ----------
        forward : Callable[[], Tensor]
            The computation that produces the loss.

        Returns
        -------
        OptimizationResult
            Whether the loss was finite, whether the window closed with an optimizer
            step, and the scale that step's gradients were scaled by. A nonfinite
            loss ends the window without a step, and the next unit starts a fresh
            one.

        Raises
        ------
        ActivationMemoryExceededError
            If the step runs inside ``metered`` and holds more device memory
            than the meter's bound allows.
        """
        memory = self._memory

        if memory is None:
            return self._step(forward)

        with memory.step():
            return self._step(forward)

    def _step(self, forward: Callable[[], Tensor], /) -> OptimizationResult:
        """Run one accumulation unit unmeasured and report its outcome."""
        if self._position == 0:
            self._optimizer.zero_grad(set_to_none=True)

        with self._device_policy.autocast():
            loss = forward()

        if not bool(torch.isfinite(loss).all()):
            self._position = 0

            return OptimizationResult(loss_finite=False, stepped=False, scale=None)

        share = loss / self._options.accumulation_steps

        if self._scaler is None:
            share.backward()

        else:
            self._scaler.scale(share).backward()

        self._position += 1

        if self._position < self._options.accumulation_steps:
            return OptimizationResult(loss_finite=True, stepped=False, scale=None)

        scale = self._finish()
        self._position = 0

        return OptimizationResult(loss_finite=True, stepped=True, scale=scale)

    def state(self) -> OptimizationState:
        """Return the state a checkpoint saves for this policy."""
        scaler = None if self._scaler is None else self._scaler.state_dict()
        scheduler = None if self._scheduler is None else self._scheduler.state_dict()

        return OptimizationState(
            self._optimizer.state_dict(), scaler=scaler, scheduler=scheduler
        )

    def validate(self, state: OptimizationState, /) -> None:
        """Refuse state this policy cannot restore without changing live state.

        Parameters
        ----------
        state : OptimizationState
            What ``state`` returned on the run being resumed.

        Raises
        ------
        ValueError
            If the policy is mid-window or the saved topology, state-tree
            structure, or tensor metadata is incompatible with this policy.
        """
        _presence(
            "scaler", saved=state.scaler is not None, held=self._scaler is not None
        )
        _presence(
            "scheduler",
            saved=state.scheduler is not None,
            held=self._scheduler is not None,
        )

        if self._position != 0:
            msg = "optimization state cannot be restored during an accumulation window"
            raise ValueError(msg)

        _validate_optimizer_state(self._optimizer, state.optimizer)

        if self._scaler is not None and state.scaler is not None:
            _validate_compatible_state(
                "scaler",
                state.scaler,
                cast("StateTree", self._scaler.state_dict()),
            )

        if self._scheduler is not None and state.scheduler is not None:
            _validate_compatible_state(
                "scheduler",
                state.scheduler,
                cast("StateTree", self._scheduler.state_dict()),
            )

    def restore(self, state: OptimizationState, /) -> None:
        """Load saved state into the optimizer, the scaler and the scheduler.

        Parameters
        ----------
        state : OptimizationState
            What ``state`` returned on the run being resumed.

        Raises
        ------
        ValueError
            If the saved state holds a scaler or scheduler block this policy has no
            part for, or lacks one for a part it has.
        """
        self.prepare_restore(state)()

    def prepare_restore(self, state: OptimizationState, /) -> Callable[[], None]:
        """Validate state and return its non-validating restore operation.

        Parameters
        ----------
        state : OptimizationState
            What ``state`` returned on the run being resumed.
        """
        self.validate(state)
        prepared = OptimizationState(
            cast("dict[str, StateTree]", _copy_state_tree(state.optimizer)),
            scaler=(
                None
                if state.scaler is None
                else cast("dict[str, StateTree]", _copy_state_tree(state.scaler))
            ),
            scheduler=(
                None
                if state.scheduler is None
                else cast("dict[str, StateTree]", _copy_state_tree(state.scheduler))
            ),
        )

        try:
            isolated_optimizer = copy(self._optimizer)

        except Exception as error:
            msg = "the optimization policy cannot be isolated for restore preparation"
            raise ValueError(msg) from error

        if isolated_optimizer is self._optimizer:
            msg = "the optimizer did not produce an isolated restore copy"
            raise ValueError(msg)

        if any(
            kind.__dict__.get("__slots__") for kind in type(self._optimizer).__mro__
        ):
            msg = "an optimizer with slotted state cannot be isolated for restoration"
            raise ValueError(msg)

        try:
            isolated_optimizer.__dict__ = deepcopy(self._optimizer.__dict__)
            isolated_scaler = deepcopy(self._scaler)
            isolated_scheduler = deepcopy(self._scheduler)

        except Exception as error:
            msg = "the optimization policy cannot be isolated for restore preparation"
            raise ValueError(msg) from error

        if (self._scaler is not None and isolated_scaler is self._scaler) or (
            self._scheduler is not None and isolated_scheduler is self._scheduler
        ):
            msg = "the optimization policy did not produce an isolated restore copy"
            raise ValueError(msg)

        probe = OptimizationState(
            cast(
                "dict[str, StateTree]",
                _copy_state_tree(prepared.optimizer),
            ),
            scaler=(
                None
                if prepared.scaler is None
                else cast(
                    "dict[str, StateTree]",
                    _copy_state_tree(prepared.scaler),
                )
            ),
            scheduler=(
                None
                if prepared.scheduler is None
                else cast(
                    "dict[str, StateTree]",
                    _copy_state_tree(prepared.scheduler),
                )
            ),
        )

        try:
            isolated_optimizer.load_state_dict(probe.optimizer)

            if isolated_scaler is not None and probe.scaler is not None:
                isolated_scaler.load_state_dict(probe.scaler)

            if isolated_scheduler is not None and probe.scheduler is not None:
                isolated_scheduler.load_state_dict(probe.scheduler)

        except Exception as error:
            msg = "the optimization policy refused checkpoint state during preparation"
            raise ValueError(msg) from error

        def restore() -> None:
            self._restore_prepared(prepared)

        return restore

    def _restore_prepared(self, state: OptimizationState, /) -> None:
        """Apply state already accepted by :meth:`prepare_restore`."""
        self._optimizer.load_state_dict(state.optimizer)

        if self._scaler is not None and state.scaler is not None:
            self._scaler.load_state_dict(state.scaler)

        if self._scheduler is not None and state.scheduler is not None:
            self._scheduler.load_state_dict(state.scheduler)

        self._position = 0

    @override
    def __repr__(self) -> str:
        """Return the optimizer type, the device policy and the options."""
        return (
            f"AutocastPolicy({type(self._optimizer).__name__}, "
            f"{self._device_policy!r}, {self._options!r})"
        )
