"""Training-layer conformance laws."""

from typing import TYPE_CHECKING, Final, NamedTuple, Protocol

from hypothesis import given, settings

from lofi._checks import LawViolationError, fail_law
from lofi.train._boundary import disagreements
from lofi.train._checkpoint import FingerprintMismatchError, UnknownCheckpointError
from lofi.train._optimization import OptInSet
from lofi.train._steps import edges

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from hypothesis.strategies import SearchStrategy
    from torch import Tensor, dtype

    from lofi.train._boundary import CheckpointManifest
    from lofi.train._optimization import OptimizationResult
    from lofi.train._ports import (
        CheckpointStore,
        OptimizationPolicy,
        StepDeclaration,
        TrainStep,
    )

__all__ = [
    "AMP_STEP_ORDER",
    "CHECKPOINT_STORE_LAWS",
    "EVALUATION_REGIME_LAWS",
    "FULL_PRECISION_STEP_ORDER",
    "OPTIMIZATION_POLICY_LAWS",
    "TRAIN_STEP_LAWS",
    "CheckpointFixture",
    "EvaluationFixture",
    "OptimizationCase",
    "OptimizationFixture",
    "StepFixture",
    "a_fingerprint_mismatch_is_refused",
    "a_nonfinite_loss_is_reported_without_a_step",
    "a_round_trip_restores_the_step_count_and_the_random_state",
    "an_amp_step_matches_the_recorded_order",
    "association_is_by_declared_role",
    "autograd_status_is_invariant_across_opt_ins",
    "check_checkpoint_store_laws",
    "check_evaluation_regime_laws",
    "check_optimization_policy_laws",
    "check_train_step_laws",
    "every_declared_parameter_is_consumed",
    "every_saved_checkpoint_is_listed",
    "gradient_accumulation_delays_the_optimizer_step",
    "mixed_precision_off_skips_scaler_operations",
    "nothing_raises_to_report_success",
    "shape_and_dtype_are_invariant_across_opt_ins",
    "the_edges_are_exactly_the_data_dependencies",
    "the_output_schema_does_not_depend_on_a_setting",
    "the_policy_owns_gradient_zeroing",
    "the_reference_regime_is_declared",
    "the_step_scale_is_read_before_update",
    "two_components_of_one_kind_do_not_collide",
]

_CALLS: Final = 2
"""Calls per input and opt-in set."""

AMP_STEP_ORDER: Final = (
    "zero_grad",
    "autocast_enter",
    "forward",
    "autocast_exit",
    "scale",
    "backward",
    "unscale_",
    "clip",
    "step",
    "get_scale",
    "update",
    "scheduler_step",
)
"""Required calls for one AMP optimization step."""

FULL_PRECISION_STEP_ORDER: Final = (
    "zero_grad",
    "autocast_enter",
    "forward",
    "autocast_exit",
    "backward",
    "clip",
    "step",
    "scheduler_step",
)
"""Required calls for one full-precision optimization step."""


class OptimizationCase[LossT](NamedTuple):
    """One recorded optimization-policy run."""

    policy: OptimizationPolicy[LossT]
    finite: Callable[[], LossT]
    nonfinite: Callable[[], LossT]
    calls: Callable[[], Sequence[str]]


class _OptimizationBuilder[LossT](Protocol):
    def __call__(
        self,
        *,
        amp: bool,
        accumulation_steps: int,
    ) -> OptimizationCase[LossT]: ...


class OptimizationFixture[LossT](NamedTuple):
    """Builds recorded optimization policies for the laws."""

    build: _OptimizationBuilder[LossT]
    scale: float


class EvaluationFixture[In](NamedTuple):
    """An evaluation-regime law fixture.

    Parameters
    ----------
    evaluate : Callable[[In, OptInSet], Tensor]
        Run the component in evaluation mode under one set of value rewrites and return
        the tensor it publishes.
    inputs : SearchStrategy[In]
        A strategy over the inputs the component is evaluated on.
    opt_ins : Sequence[OptInSet]
        Every set of value rewrites the component declares it runs under.
    max_examples : int
        The example budget each law gives Hypothesis.
    """

    evaluate: Callable[[In, OptInSet], Tensor]
    inputs: SearchStrategy[In]
    opt_ins: Sequence[OptInSet]
    max_examples: int = 50


def _optimization_run[LossT](
    fixture: OptimizationFixture[LossT],
    *,
    amp: bool,
    accumulation_steps: int = 1,
    finite: bool = True,
) -> tuple[OptimizationCase[LossT], OptimizationResult, tuple[str, ...]]:
    """Run one recorded optimization case.

    Parameters
    ----------
    fixture : OptimizationFixture[LossT]
        The optimization-policy fixture.
    amp : bool
        Whether to enable automatic mixed precision.
    accumulation_steps : int
        The number of accumulation units in one optimizer step.
    finite : bool
        Whether the forward computation produces a finite loss.
    """
    case = fixture.build(amp=amp, accumulation_steps=accumulation_steps)
    forward = case.finite if finite else case.nonfinite
    result = case.policy.step(forward)

    return case, result, tuple(case.calls())


def an_amp_step_matches_the_recorded_order[LossT](
    fixture: OptimizationFixture[LossT],
) -> None:
    """Check the complete AMP step order.

    Parameters
    ----------
    fixture : OptimizationFixture[LossT]
        The optimization-policy fixture.
    """
    _, result, calls = _optimization_run(fixture, amp=True)

    if calls != AMP_STEP_ORDER:
        detail = f"the recorded calls were {list(calls)}"

        fail_law(detail, law="an_amp_step_matches_the_recorded_order")

    if not result.loss_finite or not result.stepped:
        detail = f"the complete finite step reported {result!r}"

        fail_law(detail, law="an_amp_step_matches_the_recorded_order")


def the_step_scale_is_read_before_update[LossT](
    fixture: OptimizationFixture[LossT],
) -> None:
    """Check that a step reports its pre-update scale.

    Parameters
    ----------
    fixture : OptimizationFixture[LossT]
        The optimization-policy fixture.
    """
    _, result, calls = _optimization_run(fixture, amp=True)
    law = "the_step_scale_is_read_before_update"

    try:
        read = calls.index("get_scale")
        update = calls.index("update")

    except ValueError:
        fail_law(f"the recorded calls were {list(calls)}", law=law)

    if read >= update:
        fail_law(f"get_scale was at {read} and update was at {update}", law=law)

    if result.scale != fixture.scale:
        detail = f"the used scale was {fixture.scale} and {result.scale} was reported"

        fail_law(detail, law=law)


def mixed_precision_off_skips_scaler_operations[LossT](
    fixture: OptimizationFixture[LossT],
) -> None:
    """Check a full-precision step.

    Parameters
    ----------
    fixture : OptimizationFixture[LossT]
        The optimization-policy fixture.
    """
    _, result, calls = _optimization_run(fixture, amp=False)
    law = "mixed_precision_off_skips_scaler_operations"

    if calls != FULL_PRECISION_STEP_ORDER:
        fail_law(f"the recorded calls were {list(calls)}", law=law)

    if result.scale is not None:
        fail_law(f"the full-precision step reported scale {result.scale}", law=law)


def a_nonfinite_loss_is_reported_without_a_step[LossT](
    fixture: OptimizationFixture[LossT],
) -> None:
    """Check nonfinite-loss reporting.

    Parameters
    ----------
    fixture : OptimizationFixture[LossT]
        The optimization-policy fixture.
    """
    _, result, calls = _optimization_run(fixture, amp=True, finite=False)
    law = "a_nonfinite_loss_is_reported_without_a_step"

    if result.loss_finite:
        fail_law(f"the nonfinite loss was reported as {result!r}", law=law)

    if result.stepped or "step" in calls or "scheduler_step" in calls:
        fail_law(f"the nonfinite run stepped in {list(calls)}", law=law)


def gradient_accumulation_delays_the_optimizer_step[LossT](
    fixture: OptimizationFixture[LossT],
) -> None:
    """Check optimizer work at an accumulation boundary.

    Parameters
    ----------
    fixture : OptimizationFixture[LossT]
        The optimization-policy fixture.
    """
    case = fixture.build(amp=True, accumulation_steps=_CALLS)
    first = case.policy.step(case.finite)
    after_first = tuple(case.calls())
    second = case.policy.step(case.finite)
    calls = tuple(case.calls())
    law = "gradient_accumulation_delays_the_optimizer_step"
    deferred = {"unscale_", "clip", "step", "get_scale", "update", "scheduler_step"}

    if first.stepped or deferred.intersection(after_first):
        detail = (
            f"the first accumulation unit reported {first!r} after {list(after_first)}"
        )

        fail_law(detail, law=law)

    if not second.stepped:
        fail_law(f"the accumulation boundary reported {second!r}", law=law)

    for call in ("forward", "scale", "backward"):
        if calls.count(call) != _CALLS:
            detail = f"{call} occurred {calls.count(call)} times in {list(calls)}"

            fail_law(detail, law=law)

    for call in deferred:
        if calls.count(call) != 1:
            detail = f"{call} occurred {calls.count(call)} times in {list(calls)}"

            fail_law(detail, law=law)


def the_policy_owns_gradient_zeroing[LossT](
    fixture: OptimizationFixture[LossT],
) -> None:
    """Check gradient zeroing at each accumulation window.

    Parameters
    ----------
    fixture : OptimizationFixture[LossT]
        The optimization-policy fixture.
    """
    case = fixture.build(amp=False, accumulation_steps=_CALLS)

    for _ in range(_CALLS):
        case.policy.step(case.finite)

    completed = tuple(case.calls())
    case.policy.step(case.finite)
    calls = tuple(case.calls())
    law = "the_policy_owns_gradient_zeroing"

    if completed.count("zero_grad") != 1:
        detail = f"one accumulation window zeroed {completed.count('zero_grad')} times"

        fail_law(detail, law=law)

    if calls.count("zero_grad") != _CALLS:
        detail = (
            f"the next window left the total at {calls.count('zero_grad')} zeroings"
        )

        fail_law(detail, law=law)

    last_zero = len(calls) - 1 - calls[::-1].index("zero_grad")
    last_forward = len(calls) - 1 - calls[::-1].index("forward")

    if last_zero >= last_forward:
        fail_law(
            f"the next zeroing followed its forward call in {list(calls)}", law=law
        )


OPTIMIZATION_POLICY_LAWS: Final = (
    an_amp_step_matches_the_recorded_order,
    the_step_scale_is_read_before_update,
    mixed_precision_off_skips_scaler_operations,
    a_nonfinite_loss_is_reported_without_a_step,
    gradient_accumulation_delays_the_optimizer_step,
    the_policy_owns_gradient_zeroing,
)
"""Optimization-policy laws in execution order."""


def check_optimization_policy_laws[LossT](
    fixture: OptimizationFixture[LossT],
) -> None:
    """Run every optimization-policy law.

    Parameters
    ----------
    fixture : OptimizationFixture[LossT]
        The optimization-policy fixture.
    """
    for law in OPTIMIZATION_POLICY_LAWS:
        law(fixture)


def _budget[In](fixture: EvaluationFixture[In]) -> settings:
    """Return the Hypothesis settings every evaluation regime law runs under.

    Parameters
    ----------
    fixture : EvaluationFixture[In]
        The component under test and its strategies.
    """
    # laws do not impose per-example deadlines
    return settings(max_examples=fixture.max_examples, deadline=None)


def the_reference_regime_is_declared[In](fixture: EvaluationFixture[In]) -> None:
    """Check that the component declares the regime with no value rewrite on.

    Parameters
    ----------
    fixture : EvaluationFixture[In]
        The component under test and its strategies.

    Raises
    ------
    LawViolationError
        When no declared regime is empty.
    """
    if OptInSet() not in fixture.opt_ins:
        declared = ", ".join(repr(opt_ins) for opt_ins in fixture.opt_ins)
        detail = f"the declared regimes are {declared} and none of them is empty"

        fail_law(detail, law="reference regime")


def autograd_status_is_invariant_across_opt_ins[In](
    fixture: EvaluationFixture[In],
) -> None:
    """Check autograd status across enabled rewrites.

    Parameters
    ----------
    fixture : EvaluationFixture[In]
        The component under test and its strategies.

    Raises
    ------
    LawViolationError
        When two calls on one input disagree about the autograd status of what they
        returned.
    """
    evaluate = fixture.evaluate

    @given(fixture.inputs)
    @_budget(fixture)
    def law(value: In) -> None:
        seen: dict[bool, tuple[OptInSet, int]] = {}

        for opt_ins in fixture.opt_ins:
            for call in range(_CALLS):
                status = evaluate(value, opt_ins).requires_grad
                seen.setdefault(status, (opt_ins, call))

        if len(seen) > 1:
            witnesses = ", ".join(
                f"requires_grad={status} on call {call + 1} under {opt_ins!r}"
                for status, (opt_ins, call) in seen.items()
            )
            detail = f"evaluating {value!r} gave {witnesses}"

            fail_law(detail, law="autograd status")

    law()


def shape_and_dtype_are_invariant_across_opt_ins[In](
    fixture: EvaluationFixture[In],
) -> None:
    """Check shape and dtype across enabled rewrites.

    Parameters
    ----------
    fixture : EvaluationFixture[In]
        The component under test and its strategies.

    Raises
    ------
    LawViolationError
        When two calls on one input return different shapes or dtypes.
    """
    evaluate = fixture.evaluate

    @given(fixture.inputs)
    @_budget(fixture)
    def law(value: In) -> None:
        seen: dict[tuple[tuple[int, ...], dtype], tuple[OptInSet, int]] = {}

        for opt_ins in fixture.opt_ins:
            for call in range(_CALLS):
                returned = evaluate(value, opt_ins)
                shape = (tuple(returned.shape), returned.dtype)
                seen.setdefault(shape, (opt_ins, call))

        if len(seen) > 1:
            witnesses = ", ".join(
                f"{shape} of {kind} on call {call + 1} under {opt_ins!r}"
                for (shape, kind), (opt_ins, call) in seen.items()
            )
            detail = f"evaluating {value!r} gave {witnesses}"

            fail_law(detail, law="shape and dtype")

    law()


EVALUATION_REGIME_LAWS: Final = (
    the_reference_regime_is_declared,
    autograd_status_is_invariant_across_opt_ins,
    shape_and_dtype_are_invariant_across_opt_ins,
)
"""Evaluation-regime laws in execution order."""


def check_evaluation_regime_laws[In](fixture: EvaluationFixture[In]) -> None:
    """Run every evaluation regime law against one component.

    Parameters
    ----------
    fixture : EvaluationFixture[In]
        The component under test and its strategies.

    Raises
    ------
    LawViolationError
        On the first law the component fails.
    """
    for law in EVALUATION_REGIME_LAWS:
        law(fixture)


class CheckpointFixture[StateT](NamedTuple):
    """A checkpoint-store law fixture."""

    store: CheckpointStore[StateT]
    manifest: CheckpointManifest
    states: Callable[[], Mapping[str, StateT]]
    mismatched: CheckpointManifest
    twins: tuple[str, str]


_ROLES: Final = 2
"""The minimum role count for a checkpoint fixture."""


def _saved[StateT](fixture: CheckpointFixture[StateT]) -> tuple[str, ...]:
    """Return validated fixture roles.

    Parameters
    ----------
    fixture : CheckpointFixture[StateT]
        The checkpoint-store fixture.
    """
    roles = tuple(sorted(fixture.manifest.roles))

    if len(roles) < _ROLES:
        detail = f"the manifest names the roles {list(roles)} and needs two"

        fail_law(detail, law="the fixture states two roles")

    return roles


def association_is_by_declared_role[StateT](
    fixture: CheckpointFixture[StateT],
) -> None:
    """Check checkpoint state association by role.

    Parameters
    ----------
    fixture : CheckpointFixture[StateT]
        The checkpoint-store fixture.
    """
    law = "association_is_by_declared_role"
    roles = _saved(fixture)

    for order in (roles, tuple(reversed(roles))):
        saving = fixture.states()
        handle = fixture.store.save(
            fixture.manifest, {role: saving[role] for role in order}
        )
        restored = fixture.store.load(handle, fixture.manifest)

        if frozenset(restored.states) != fixture.manifest.roles:
            detail = (
                f"the manifest names {list(roles)} and the store returned "
                f"{sorted(restored.states)}"
            )

            fail_law(detail, law=law)

        for role in roles:
            if restored.states[role] != saving[role]:
                detail = (
                    f"the role {role!r} was saved with {saving[role]!r} and "
                    f"read back as {restored.states[role]!r}, with the roles "
                    f"presented in the order {list(order)}"
                )

                fail_law(detail, law=law)


def a_fingerprint_mismatch_is_refused[StateT](
    fixture: CheckpointFixture[StateT],
) -> None:
    """Check that a fingerprint mismatch prevents checkpoint loading.

    Parameters
    ----------
    fixture : CheckpointFixture[StateT]
        The checkpoint-store fixture.
    """
    law = "a_fingerprint_mismatch_is_refused"
    reasons = disagreements(fixture.mismatched, fixture.manifest)

    if not reasons:
        detail = "the fixture's two manifests agree, so nothing is refused here"

        fail_law(detail, law=law)

    handle = fixture.store.save(fixture.manifest, fixture.states())

    try:
        fixture.store.load(handle, fixture.mismatched)

    except FingerprintMismatchError:
        return

    stated = "; ".join(reasons)
    detail = f"the store read {handle} back although {stated}"

    fail_law(detail, law=law)


def two_components_of_one_kind_do_not_collide[StateT](
    fixture: CheckpointFixture[StateT],
) -> None:
    """Check that same-kind components retain separate state.

    Parameters
    ----------
    fixture : CheckpointFixture[StateT]
        The checkpoint-store fixture.
    """
    law = "two_components_of_one_kind_do_not_collide"
    first, second = fixture.twins
    components = fixture.manifest.components

    if components[first] != components[second]:
        detail = (
            f"the roles {first!r} and {second!r} are named as twins and their "
            f"records differ, so no collision would show here"
        )

        fail_law(detail, law=law)

    saving = fixture.states()

    if saving[first] == saving[second]:
        detail = (
            f"the roles {first!r} and {second!r} are saved with equal state, "
            f"so a store overwriting one with the other would still pass"
        )

        fail_law(detail, law=law)

    handle = fixture.store.save(fixture.manifest, saving)
    restored = fixture.store.load(handle, fixture.manifest)
    missing = frozenset(fixture.twins) - frozenset(restored.states)

    if missing:
        detail = f"the store returned no state for the roles {sorted(missing)}"

        fail_law(detail, law=law)

    for role in (first, second):
        if restored.states[role] != saving[role]:
            detail = (
                f"the role {role!r} was saved with {saving[role]!r} and read "
                f"back as {restored.states[role]!r}"
            )

            fail_law(detail, law=law)


def a_round_trip_restores_the_step_count_and_the_random_state[StateT](
    fixture: CheckpointFixture[StateT],
) -> None:
    """Check checkpoint counter and random-state round trips.

    Parameters
    ----------
    fixture : CheckpointFixture[StateT]
        The checkpoint-store fixture.
    """
    law = "a_round_trip_restores_the_step_count_and_the_random_state"
    manifest = fixture.manifest

    if manifest.step == 0 or not manifest.rng.streams:
        detail = (
            f"the manifest sits at step {manifest.step} with the streams "
            f"{sorted(manifest.rng.streams)}, which a blank one would match"
        )

        fail_law(detail, law=law)

    handle = fixture.store.save(manifest, fixture.states())
    read = fixture.store.load(handle, manifest).manifest

    if read.step != manifest.step:
        detail = f"the run was at step {manifest.step} and read back at {read.step}"

        fail_law(detail, law=law)

    if read.rng != manifest.rng:
        detail = (
            f"the streams were saved as {manifest.rng.streams} and read back "
            f"as {read.rng.streams}"
        )

        fail_law(detail, law=law)


def every_saved_checkpoint_is_listed[StateT](
    fixture: CheckpointFixture[StateT],
) -> None:
    """Check that checkpoint listing preserves save order.

    Parameters
    ----------
    fixture : CheckpointFixture[StateT]
        The checkpoint-store fixture.
    """
    law = "every_saved_checkpoint_is_listed"
    before = fixture.store.list()
    first = fixture.store.save(fixture.manifest, fixture.states())
    second = fixture.store.save(
        fixture.manifest.with_(step=fixture.manifest.step + 1), fixture.states()
    )
    listed = fixture.store.list()
    added = listed[len(before) :]

    if added != (first, second):
        detail = f"two saves returned {(first, second)} and the listing grew by {added}"

        fail_law(detail, law=law)


def a_removed_checkpoint_is_neither_listed_nor_read[StateT](
    fixture: CheckpointFixture[StateT],
) -> None:
    """Check that removal takes one checkpoint out of the listing and the reads.

    Parameters
    ----------
    fixture : CheckpointFixture[StateT]
        The checkpoint-store fixture.
    """
    law = "a_removed_checkpoint_is_neither_listed_nor_read"
    first = fixture.store.save(fixture.manifest, fixture.states())
    second = fixture.store.save(
        fixture.manifest.with_(step=fixture.manifest.step + 1), fixture.states()
    )
    read = fixture.store.manifest(first)

    if read != fixture.manifest:
        detail = f"{first} was saved with {fixture.manifest!r} and read as {read!r}"

        fail_law(detail, law=law)

    fixture.store.remove(first)
    listed = fixture.store.list()

    if first in listed or second not in listed:
        detail = f"after removing {first} of {(first, second)} the listing is {listed}"

        fail_law(detail, law=law)

    for read_back in (
        lambda: fixture.store.load(first, fixture.manifest),
        lambda: fixture.store.manifest(first),
    ):
        try:
            read_back()

        except UnknownCheckpointError:
            continue

        detail = f"the removed checkpoint {first} was still read back"

        fail_law(detail, law=law)

    kept = fixture.store.load(second, fixture.manifest).manifest.step

    if kept != fixture.manifest.step + 1:
        detail = f"removing {first} changed the checkpoint saved after it"

        fail_law(detail, law=law)


CHECKPOINT_STORE_LAWS: Final = (
    association_is_by_declared_role,
    a_fingerprint_mismatch_is_refused,
    two_components_of_one_kind_do_not_collide,
    a_round_trip_restores_the_step_count_and_the_random_state,
    every_saved_checkpoint_is_listed,
    a_removed_checkpoint_is_neither_listed_nor_read,
)
"""Checkpoint-store laws in execution order."""


def check_checkpoint_store_laws[StateT](fixture: CheckpointFixture[StateT]) -> None:
    """Run every checkpoint store law against one store.

    Parameters
    ----------
    fixture : CheckpointFixture[StateT]
        The checkpoint-store fixture.
    """
    for law in CHECKPOINT_STORE_LAWS:
        law(fixture)


class StepFixture[InputsT, OutputsT](NamedTuple):
    """A training-step law fixture."""

    step: TrainStep[InputsT, OutputsT]
    inputs: Callable[[], InputsT]
    reads: Callable[[InputsT], frozenset[str]]
    idle: Callable[[], InputsT]
    variants: Sequence[TrainStep[InputsT, OutputsT]]
    graph: Sequence[StepDeclaration]
    available: frozenset[str] = frozenset()


_SETTINGS: Final = 2
"""The minimum variant count for a step fixture."""


def _read_names[InputsT, OutputsT](
    fixture: StepFixture[InputsT, OutputsT],
) -> frozenset[str]:
    """Return names read during one step call.

    Parameters
    ----------
    fixture : StepFixture[InputsT, OutputsT]
        The training-step fixture.
    """
    values = fixture.inputs()

    fixture.step.run(values)

    return fixture.reads(values)


def the_output_schema_does_not_depend_on_a_setting[InputsT, OutputsT](
    fixture: StepFixture[InputsT, OutputsT],
) -> None:
    """Check output schema across settings.

    Parameters
    ----------
    fixture : StepFixture[InputsT, OutputsT]
        The training-step fixture.
    """
    law = "the_output_schema_does_not_depend_on_a_setting"

    if len(fixture.variants) < _SETTINGS:
        detail = (
            f"the fixture states {len(fixture.variants)} variants and needs "
            f"two, since one variant agrees with itself"
        )

        fail_law(detail, law=law)

    schemas: dict[tuple[str, str, tuple[str, ...]], int] = {}

    for position, variant in enumerate(fixture.variants):
        outcome = variant.run(fixture.inputs())
        schema = (
            type(outcome.outputs).__name__,
            type(outcome.result).__name__,
            tuple(sorted(outcome.result.metrics)),
        )
        schemas.setdefault(schema, position)

    if len(schemas) > 1:
        stated = "; ".join(
            f"variant {position} publishes {published}, reports {reported} "
            f"and names {list(keys)}"
            for (published, reported, keys), position in schemas.items()
        )

        fail_law(stated, law=law)


def every_declared_parameter_is_consumed[InputsT, OutputsT](
    fixture: StepFixture[InputsT, OutputsT],
) -> None:
    """Check that a step reads every declared input.

    Parameters
    ----------
    fixture : StepFixture[InputsT, OutputsT]
        The training-step fixture.
    """
    law = "every_declared_parameter_is_consumed"
    declared = fixture.step.requires

    if not declared:
        detail = f"the step {fixture.step.name!r} declares no requirement"

        fail_law(detail, law=law)

    unread = declared - _read_names(fixture)

    if unread:
        detail = (
            f"{fixture.step.name!r} declares {sorted(declared)} and did not "
            f"read {sorted(unread)}"
        )

        fail_law(detail, law=law)


def nothing_raises_to_report_success[InputsT, OutputsT](
    fixture: StepFixture[InputsT, OutputsT],
) -> None:
    """Check that an idle step returns a result.

    Parameters
    ----------
    fixture : StepFixture[InputsT, OutputsT]
        The training-step fixture.

    Raises
    ------
    LawViolationError
        If a run raises or reports an invalid item count.
    """
    law = "nothing_raises_to_report_success"

    try:
        busy = fixture.step.run(fixture.inputs())
        quiet = fixture.step.run(fixture.idle())

    except Exception as error:
        message = f"{law}: {fixture.step.name!r} raised {error!r}"

        raise LawViolationError(message) from error

    if quiet.result.items != 0:
        detail = (
            f"the idle input was consumed as {quiet.result.items} items, so a "
            f"step raising on an empty one would still pass"
        )

        fail_law(detail, law=law)

    if busy.result.items == 0:
        detail = "the ordinary input was consumed as no items either"

        fail_law(detail, law=law)


def the_edges_are_exactly_the_data_dependencies[InputsT, OutputsT](
    fixture: StepFixture[InputsT, OutputsT],
) -> None:
    """Check that graph edges cover every read value.

    Parameters
    ----------
    fixture : StepFixture[InputsT, OutputsT]
        The training-step fixture.
    """
    law = "the_edges_are_exactly_the_data_dependencies"
    step = fixture.step
    incoming = frozenset(
        edge.value
        for edge in edges(fixture.graph, fixture.available)
        if edge.consumer == step.name
    )
    supplied = incoming | (step.requires & fixture.available)

    if not supplied:
        detail = f"nothing reaches {step.name!r}, by edge or from outside"

        fail_law(detail, law=law)

    undeclared = _read_names(fixture) - supplied

    if undeclared:
        detail = (
            f"{step.name!r} read {sorted(undeclared)}, which reaches it on no "
            f"edge and enters the graph from nowhere"
        )

        fail_law(detail, law=law)


TRAIN_STEP_LAWS: Final = (
    the_output_schema_does_not_depend_on_a_setting,
    every_declared_parameter_is_consumed,
    nothing_raises_to_report_success,
    the_edges_are_exactly_the_data_dependencies,
)
"""Training-step laws in execution order."""


def check_train_step_laws[InputsT, OutputsT](
    fixture: StepFixture[InputsT, OutputsT],
) -> None:
    """Run every training step law against one step.

    Parameters
    ----------
    fixture : StepFixture[InputsT, OutputsT]
        The training-step fixture.
    """
    for law in TRAIN_STEP_LAWS:
        law(fixture)
