from typing import TYPE_CHECKING, Final, Literal

import pytest
import torch

from lofi.train._boundary import (
    CheckpointManifest,
    ComponentRecord,
    ParserFittingRecord,
    PrecisionRecord,
    RngRecord,
    StateReference,
)
from lofi.train._checkpoint import (
    Checkpoint,
    CheckpointError,
    CheckpointId,
    CheckpointPolicy,
    CorruptCheckpointError,
    FingerprintMismatchError,
    UnknownCheckpointError,
)
from lofi.train._store import DirectoryStore

if TYPE_CHECKING:
    from pathlib import Path

FORWARD: Final = "forward_application"

BACKWARD: Final = "backward_application"

HANDLE: Final = CheckpointId("run-0001-0000")


def manifest() -> CheckpointManifest:
    record = ComponentRecord(
        tag="hrr",
        spec_fingerprint="spec-a1",
        signature_fingerprint="sig-b2",
        codec_identity="codec-c3",
        state=StateReference(key="decoder", digest="d4"),
    )

    return CheckpointManifest(
        run_id="run-0001",
        step=7,
        data_position=0,
        code_version="0.1.0",
        action_set="actions-a1",
        components={FORWARD: record, BACKWARD: record},
        rng=RngRecord(streams={"python": "9f3a"}),
        precision=PrecisionRecord(
            compute_dtype="bfloat16",
            autocast_device_type="cuda",
            gradient_scaling=True,
        ),
        optimizer=StateReference(key="optimizer", digest="a7"),
    )


def checkpoint() -> Checkpoint[str]:
    return Checkpoint(HANDLE, manifest(), {FORWARD: "one", BACKWARD: "two"})


def test_a_checkpoint_reads_back_the_state_of_each_role() -> None:
    held = checkpoint()

    assert held.states[FORWARD] == "one"
    assert held.states[BACKWARD] == "two"


def test_a_checkpoint_names_the_handle_and_the_manifest_it_was_read_under() -> None:
    held = checkpoint()

    assert held.checkpoint == HANDLE
    assert held.manifest == manifest()


def test_a_checkpoint_refuses_state_keyed_by_a_role_the_manifest_does_not_name() -> (
    None
):
    with pytest.raises(ValueError, match="controller"):
        Checkpoint(HANDLE, manifest(), {FORWARD: "one", "controller": "two"})


def test_a_checkpoint_refuses_state_missing_a_role_the_manifest_names() -> None:
    with pytest.raises(ValueError, match=BACKWARD):
        Checkpoint(HANDLE, manifest(), {FORWARD: "one"})


def test_a_checkpoint_read_without_run_state_holds_none() -> None:
    assert dict(checkpoint().run_state) == {}


def test_run_state_is_read_back_under_the_reference_key() -> None:
    held = Checkpoint(
        HANDLE,
        manifest(),
        {FORWARD: "one", BACKWARD: "two"},
        run_state={"optimizer": "moments"},
    )

    assert held.run_state["optimizer"] == "moments"
    assert held.states[FORWARD] == "one"


def test_run_state_under_a_name_the_manifest_holds_no_reference_for_is_refused() -> (
    None
):
    with pytest.raises(ValueError, match="scheduler"):
        Checkpoint(
            HANDLE,
            manifest(),
            {FORWARD: "one", BACKWARD: "two"},
            run_state={"scheduler": "steps"},
        )


def test_the_run_state_a_checkpoint_hands_out_is_no_container_a_caller_can_write() -> (
    None
):
    held = Checkpoint(
        HANDLE,
        manifest(),
        {FORWARD: "one", BACKWARD: "two"},
        run_state={"optimizer": "moments"},
    )

    assert not isinstance(held.run_state, dict | list | set)


def test_writing_into_the_run_state_a_checkpoint_was_built_from_changes_nothing() -> (
    None
):
    built = {"optimizer": "moments"}
    held = Checkpoint(
        HANDLE, manifest(), {FORWARD: "one", BACKWARD: "two"}, run_state=built
    )

    built["optimizer"] = "other"

    assert held.run_state["optimizer"] == "moments"


def test_a_checkpoint_refuses_an_empty_handle() -> None:
    with pytest.raises(ValueError, match="checkpoint is empty"):
        Checkpoint(CheckpointId(""), manifest(), {FORWARD: "one", BACKWARD: "two"})


def test_the_state_a_checkpoint_hands_out_is_no_container_a_caller_can_write() -> None:
    assert not isinstance(checkpoint().states, dict | list | set)


def test_writing_into_the_mapping_a_checkpoint_was_built_from_changes_nothing() -> None:
    built = {FORWARD: "one", BACKWARD: "two"}
    held = Checkpoint(HANDLE, manifest(), built)

    built[FORWARD] = "three"

    assert held.states[FORWARD] == "one"


@pytest.mark.parametrize(
    "field", ["checkpoint", "manifest", "states", "run_state", "extra"]
)
def test_a_checkpoint_takes_on_no_attribute_after_it_is_built(field: str) -> None:
    held = checkpoint()

    with pytest.raises(AttributeError):
        setattr(held, field, None)


def test_a_checkpoint_reports_its_handle_and_its_roles() -> None:
    assert repr(checkpoint()) == (
        f"Checkpoint({HANDLE!r}, roles=[{BACKWARD}, {FORWARD}])"
    )


def test_a_fingerprint_mismatch_names_every_reason_it_was_given() -> None:
    reasons = ("the tag moved", "the codec moved")
    error = FingerprintMismatchError(HANDLE, reasons)

    assert error.reasons == reasons
    assert "the codec moved" in str(error)
    assert HANDLE in str(error)


def test_an_unknown_checkpoint_names_the_handle_that_was_asked_for() -> None:
    assert HANDLE in str(UnknownCheckpointError(HANDLE))


def test_a_corrupt_checkpoint_names_the_handle_and_what_was_found() -> None:
    error = CorruptCheckpointError(HANDLE, "the block for 'decoder' is truncated")

    assert HANDLE in str(error)
    assert "truncated" in str(error)


def test_every_refusal_is_a_checkpoint_error_and_none_is_a_value_error() -> None:
    assert issubclass(FingerprintMismatchError, CheckpointError)
    assert issubclass(UnknownCheckpointError, CheckpointError)
    assert issubclass(CorruptCheckpointError, CheckpointError)
    assert not issubclass(CheckpointError, ValueError)


LATEST: Final = "resume-latest"

BEST: Final = "validation-best"


def handles(count: int) -> tuple[CheckpointId, ...]:
    return tuple(CheckpointId(f"{position:06d}-run") for position in range(count))


@pytest.mark.parametrize(
    ("interval", "horizon", "due"),
    [(1, 3, (1, 2, 3)), (2, 5, (2, 4, 5)), (3, 6, (3, 6)), (7, 4, (4,))],
)
def test_a_checkpoint_is_due_on_the_cadence_and_at_the_horizon(
    interval: int, horizon: int, due: tuple[int, ...]
) -> None:
    policy = CheckpointPolicy(interval, 1, 1)

    assert (
        tuple(
            completed
            for completed in range(1, horizon + 1)
            if policy.due(completed, horizon)
        )
        == due
    )


def test_the_newest_of_each_kind_are_kept_and_the_rest_are_removable() -> None:
    first, second, third, fourth, fifth, sixth = handles(6)
    stored = (
        (first, BEST),
        (second, LATEST),
        (third, BEST),
        (fourth, LATEST),
        (fifth, LATEST),
        (sixth, None),
    )

    assert CheckpointPolicy(1, 2, 1).removable(stored) == (first, second)
    assert CheckpointPolicy(1, 1, 2).removable(stored) == (second, fourth)
    assert CheckpointPolicy(1, 3, 2).removable(stored) == ()
    assert CheckpointPolicy(1, 1, 0).removable(stored) == (
        first,
        second,
        third,
        fourth,
    )


def test_a_checkpoint_written_outside_fitting_is_never_removable() -> None:
    (held,) = handles(1)

    assert CheckpointPolicy(1, 1, 0).removable(((held, None),)) == ()


def fitted(
    kind: Literal["validation-best", "resume-latest"], epochs: int
) -> CheckpointManifest:
    return manifest().with_(
        step=epochs,
        fitting=ParserFittingRecord(
            kind=kind,
            completed_epochs=epochs,
            metric="loss",
            best=1.0,
            failures=0,
            best_checkpoint=None if kind == BEST else "000000-run-0001-step1",
        ),
    )


def test_pruning_removes_what_the_retention_drops_from_a_store(
    tmp_path: Path,
) -> None:
    store = DirectoryStore(tmp_path / "checkpoints")
    blocks = {FORWARD: torch.zeros(1), BACKWARD: torch.ones(1), "optimizer": {}}
    written = [
        store.save(fitted(kind, epochs), blocks)
        for epochs, kind in ((1, BEST), (1, LATEST), (2, LATEST), (3, BEST))
    ]

    removed = CheckpointPolicy(1, 1, 1).prune(store)

    assert removed == (written[0], written[1])
    assert store.list() == (written[2], written[3])
    assert CheckpointPolicy(1, 1, 1).prune(store) == ()


@pytest.mark.parametrize(
    ("interval", "keep", "keep_best"), [(0, 1, 1), (1, 0, 1), (1, 1, -1)]
)
def test_a_policy_refuses_a_count_outside_its_range(
    interval: int, keep: int, keep_best: int
) -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        CheckpointPolicy(interval, keep, keep_best)


def test_a_policy_compares_by_its_three_counts() -> None:
    policy = CheckpointPolicy(2, 3, 1)

    assert policy == CheckpointPolicy(2, 3, 1)
    assert hash(policy) == hash(CheckpointPolicy(2, 3, 1))
    assert policy != CheckpointPolicy(2, 3, 2)
    assert policy != CheckpointPolicy(1, 3, 1)
    assert policy != CheckpointPolicy(2, 1, 1)
    assert (policy.interval, policy.keep, policy.keep_best) == (2, 3, 1)
    assert repr(policy) == "CheckpointPolicy(2, 3, 1)"

    with pytest.raises(AttributeError):
        policy.__setattr__("_keep", 4)
