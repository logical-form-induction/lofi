import json
from hashlib import sha256
from pathlib import PurePosixPath
from pickle import PicklingError
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch
from tests.conformance.checkpoint_store import (
    CheckpointStoreConformance,
)

from lofi.train._boundary import (
    CheckpointManifest,
    ComponentRecord,
    PrecisionRecord,
    RngRecord,
    StateReference,
)
from lofi.train._checkpoint import (
    CheckpointId,
    CorruptCheckpointError,
    FingerprintMismatchError,
    UnknownCheckpointError,
)
from lofi.train._ports import CheckpointStore
from lofi.train._store import DirectoryStore, StateTree
from lofi.train.laws import CheckpointFixture

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

FORWARD: Final = "forward_application"

BACKWARD: Final = "backward_application"

ENCODER: Final = "terms"

CONTROLLER: Final = "controller"

WIDTH: Final = 4


def decoder(**moved: str) -> ComponentRecord:
    return ComponentRecord(
        tag="hrr",
        spec_fingerprint="spec-a1",
        signature_fingerprint="sig-b2",
        codec_identity="codec-c3",
        state=StateReference(key="decoder", digest="d4"),
    ).with_(**moved)


def controller(labels: dict[str, int]) -> ComponentRecord:
    return ComponentRecord(
        tag="mlp",
        spec_fingerprint="spec-m1",
        signature_fingerprint="sig-b2",
        codec_identity="codec-c3",
        state=StateReference(key="controller", digest="e5"),
        labels=labels,
    )


def manifest(**components: ComponentRecord) -> CheckpointManifest:
    return CheckpointManifest(
        run_id="run-0001",
        step=7,
        data_position=0,
        code_version="0.1.0",
        action_set="actions-a1",
        components=components
        or {
            FORWARD: decoder(),
            BACKWARD: decoder(),
            ENCODER: decoder(tag="rnn", spec_fingerprint="spec-e5"),
            CONTROLLER: controller({FORWARD: 0, BACKWARD: 1}),
        },
        rng=RngRecord(streams={"python": "9f3a", "torch": "1c20"}),
        precision=PrecisionRecord(
            compute_dtype="bfloat16",
            autocast_device_type="cpu",
            gradient_scaling=False,
        ),
    )


def with_run_state(built: CheckpointManifest) -> CheckpointManifest:
    return built.with_(
        optimizer=StateReference(key="optimizer", digest="a7"),
        scaler=StateReference(key="scaler", digest="b8"),
        scheduler=StateReference(key="scheduler", digest="c9"),
    )


def scalars() -> Mapping[str, StateTree]:
    return {
        FORWARD: {"weight": torch.tensor(1.5), "bias": 0.25},
        BACKWARD: {"weight": torch.tensor(-1.5), "bias": 0.25},
        ENCODER: {"weight": torch.tensor(2.0)},
        CONTROLLER: {"weight": torch.tensor(0.5), "columns": [0, 1]},
    }


def modules() -> dict[str, torch.nn.Linear]:
    torch.manual_seed(0)

    return {
        FORWARD: torch.nn.Linear(WIDTH, WIDTH),
        BACKWARD: torch.nn.Linear(WIDTH, WIDTH),
        ENCODER: torch.nn.Linear(WIDTH, WIDTH),
        CONTROLLER: torch.nn.Linear(WIDTH, 2),
    }


def weights(held: Mapping[str, torch.nn.Linear]) -> dict[str, StateTree]:
    return {role: dict(module.state_dict()) for role, module in held.items()}


def tensors(state: StateTree) -> dict[str, torch.Tensor]:
    held = cast("dict[str, torch.Tensor]", state)

    return dict(held)


@pytest.fixture
def store(tmp_path: Path) -> DirectoryStore:
    return DirectoryStore(tmp_path / "checkpoints")


class TestADirectoryStore(CheckpointStoreConformance):
    @pytest.fixture
    def conformance(self, store: DirectoryStore) -> CheckpointFixture[StateTree]:
        mismatched = manifest().with_(
            components={
                **manifest().components,
                ENCODER: decoder(tag="rnn", spec_fingerprint="spec-e6"),
            }
        )

        return CheckpointFixture(
            store=store,
            manifest=manifest(),
            states=scalars,
            mismatched=mismatched,
            twins=(FORWARD, BACKWARD),
        )


def test_one_block_is_written_per_role_and_the_role_count_is_the_component_count(
    store: DirectoryStore,
) -> None:
    built = manifest()
    handle = store.save(built, weights(modules()))
    written = sorted(path.stem for path in (store.root / handle / "roles").iterdir())

    assert written == sorted(built.roles)
    assert len(written) == len(built.components) == len(built.roles)


def test_blocks_are_named_by_role_and_never_by_class_name(
    store: DirectoryStore,
) -> None:
    handle = store.save(manifest(), weights(modules()))
    names = [path.name for path in (store.root / handle).rglob("*")]

    assert f"{FORWARD}.pt" in names
    assert f"{BACKWARD}.pt" in names
    assert not any("Linear" in name for name in names)


def test_two_modules_of_one_class_read_back_their_own_weights(
    store: DirectoryStore,
) -> None:
    held = modules()
    handle = store.save(manifest(), weights(held))
    restored = store.load(handle, manifest())

    for role, module in held.items():
        assert torch.equal(
            tensors(restored.states[role])["weight"], module.state_dict()["weight"]
        )

    assert not torch.equal(
        tensors(restored.states[FORWARD])["weight"],
        tensors(restored.states[BACKWARD])["weight"],
    )


def test_a_loaded_block_restores_a_fresh_module_exactly(
    store: DirectoryStore,
) -> None:
    held = modules()
    handle = store.save(manifest(), weights(held))
    restored = store.load(handle, manifest())
    fresh = torch.nn.Linear(WIDTH, WIDTH)
    fresh.load_state_dict(tensors(restored.states[BACKWARD]))
    inputs = torch.arange(float(WIDTH)).unsqueeze(0)

    assert torch.equal(fresh(inputs), held[BACKWARD](inputs))


def test_reordering_two_distinct_decoders_is_refused_naming_both_roles(
    store: DirectoryStore,
) -> None:
    saved = manifest(
        **{
            FORWARD: decoder(spec_fingerprint="spec-f1"),
            BACKWARD: decoder(spec_fingerprint="spec-b1"),
        }
    )
    reordered = manifest(
        **{
            FORWARD: decoder(spec_fingerprint="spec-b1"),
            BACKWARD: decoder(spec_fingerprint="spec-f1"),
        }
    )
    handle = store.save(saved, {FORWARD: [1.0], BACKWARD: [2.0]})

    with pytest.raises(FingerprintMismatchError) as raised:
        store.load(handle, reordered)

    assert len(raised.value.reasons) == 2
    assert FORWARD in raised.value.reasons[1]
    assert BACKWARD in raised.value.reasons[0]
    assert "spec-f1" in str(raised.value)
    assert "spec-b1" in str(raised.value)


def test_reordering_two_identical_decoders_loads_each_role_its_own_block(
    store: DirectoryStore,
) -> None:
    saved = manifest(**{FORWARD: decoder(), BACKWARD: decoder()})
    reordered = manifest(**{BACKWARD: decoder(), FORWARD: decoder()})
    handle = store.save(saved, {FORWARD: [1.0], BACKWARD: [2.0]})
    restored = store.load(handle, reordered)

    assert restored.states[FORWARD] == [1.0]
    assert restored.states[BACKWARD] == [2.0]


def test_a_relabeled_controller_is_refused_naming_the_controller(
    store: DirectoryStore,
) -> None:
    handle = store.save(manifest(), scalars())
    swapped = manifest().with_(
        components={
            **manifest().components,
            CONTROLLER: controller({FORWARD: 1, BACKWARD: 0}),
        }
    )

    with pytest.raises(FingerprintMismatchError) as raised:
        store.load(handle, swapped)

    assert len(raised.value.reasons) == 1
    assert CONTROLLER in raised.value.reasons[0]
    assert "labels" in raised.value.reasons[0]


def test_the_manifest_is_read_back_equal_with_its_step_streams_and_labels(
    store: DirectoryStore,
) -> None:
    built = with_run_state(manifest())
    handle = store.save(
        built, {**scalars(), "optimizer": {}, "scaler": {}, "scheduler": {}}
    )
    restored = store.load(handle, built).manifest

    assert restored == built
    assert restored.components[CONTROLLER].labels == {FORWARD: 0, BACKWARD: 1}


def test_run_state_is_read_back_under_the_reference_keys(
    store: DirectoryStore,
) -> None:
    held = modules()
    optimizer = torch.optim.AdamW(held[FORWARD].parameters())
    held[FORWARD](torch.ones(1, WIDTH)).sum().backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    built = with_run_state(manifest())
    handle = store.save(
        built,
        {
            **weights(held),
            "optimizer": cast("StateTree", optimizer.state_dict()),
            "scaler": cast("StateTree", scaler.state_dict()),
            "scheduler": cast("StateTree", scheduler.state_dict()),
        },
    )
    restored = store.load(handle, built)
    fresh = torch.optim.AdamW(torch.nn.Linear(WIDTH, WIDTH).parameters())
    fresh.load_state_dict(cast("dict[str, StateTree]", restored.run_state["optimizer"]))

    assert frozenset(restored.run_state) == frozenset(built.references)
    assert restored.run_state["scheduler"] == scheduler.state_dict()
    assert restored.run_state["scaler"] == scaler.state_dict()
    assert torch.equal(
        fresh.state_dict()["state"][0]["exp_avg"],
        optimizer.state_dict()["state"][0]["exp_avg"],
    )


def test_a_save_refuses_blocks_that_are_not_the_roles_and_the_references(
    store: DirectoryStore,
) -> None:
    built = with_run_state(manifest())
    blocks = {**scalars(), "optimizer": {}, "scaler": {}, "momentum": {}}

    with pytest.raises(ValueError, match=r"scheduler.*momentum") as raised:
        store.save(built, blocks)

    assert "lack ['scheduler']" in str(raised.value)
    assert "add ['momentum']" in str(raised.value)
    assert store.list() == ()


def test_a_save_refuses_a_block_for_a_role_the_manifest_does_not_name(
    store: DirectoryStore,
) -> None:
    with pytest.raises(ValueError, match="parser"):
        store.save(manifest(), {**scalars(), "parser": {}})


def test_a_failed_save_leaves_nothing_listed_and_nothing_on_disk(
    store: DirectoryStore,
) -> None:
    unwritable = {**scalars(), CONTROLLER: cast("StateTree", lambda: None)}

    with pytest.raises(PicklingError):
        store.save(manifest(), unwritable)

    assert store.list() == ()
    assert list(store.root.iterdir()) == []


def test_a_stray_staging_directory_is_neither_listed_nor_loadable(
    store: DirectoryStore,
) -> None:
    handle = store.save(manifest(), scalars())
    stray = store.root / f".{handle}.partial"
    stray.mkdir()
    (stray / "manifest.json").write_text("{}", encoding="utf-8", newline="\n")

    assert store.list() == (handle,)

    with pytest.raises(UnknownCheckpointError):
        store.load(CheckpointId(stray.name), manifest())


def test_a_block_overwritten_with_another_role_is_refused_by_digest(
    store: DirectoryStore,
) -> None:
    handle = store.save(manifest(), weights(modules()))
    roles = store.root / handle / "roles"
    (roles / f"{BACKWARD}.pt").write_bytes((roles / f"{FORWARD}.pt").read_bytes())

    with pytest.raises(CorruptCheckpointError, match=f"'{BACKWARD}'.*digest"):
        store.load(handle, manifest())


def test_a_truncated_block_is_refused_by_digest(store: DirectoryStore) -> None:
    handle = store.save(manifest(), weights(modules()))
    block = store.root / handle / "roles" / f"{ENCODER}.pt"
    block.write_bytes(block.read_bytes()[:-16])

    with pytest.raises(CorruptCheckpointError, match=ENCODER):
        store.load(handle, manifest())


def test_deserialization_restricts_itself_to_weights(store: DirectoryStore) -> None:
    handle = store.save(manifest(), scalars())
    block = store.root / handle / "roles" / f"{ENCODER}.pt"
    index = store.root / handle / "blocks.json"
    torch.save({"path": PurePosixPath("elsewhere")}, block)
    record = json.loads(index.read_text(encoding="utf-8"))
    record["roles"][ENCODER] = sha256(block.read_bytes()).hexdigest()
    index.write_text(json.dumps(record), encoding="utf-8", newline="\n")

    with pytest.raises(CorruptCheckpointError, match="more than weights"):
        store.load(handle, manifest())


def test_a_mismatch_is_refused_before_any_block_is_read(
    store: DirectoryStore,
) -> None:
    handle = store.save(manifest(), scalars())
    (store.root / handle / "roles" / f"{ENCODER}.pt").write_bytes(b"")
    apart = manifest().with_(code_version="0.2.0")

    with pytest.raises(FingerprintMismatchError):
        store.load(handle, apart)


def test_a_manifest_that_cannot_be_read_is_reported_as_corrupt(
    store: DirectoryStore,
) -> None:
    handle = store.save(manifest(), scalars())
    (store.root / handle / "manifest.json").write_text(
        "{", encoding="utf-8", newline="\n"
    )

    with pytest.raises(CorruptCheckpointError, match=handle):
        store.load(handle, manifest())


def test_a_manifest_that_is_not_one_is_reported_as_corrupt(
    store: DirectoryStore,
) -> None:
    handle = store.save(manifest(), scalars())
    (store.root / handle / "manifest.json").write_text(
        '{"step": 1}', encoding="utf-8", newline="\n"
    )

    with pytest.raises(CorruptCheckpointError, match="manifest"):
        store.load(handle, manifest())


@pytest.mark.parametrize(
    "index",
    ["[]", '{"roles": {}}', '{"roles": [], "run_state": {}}', '{"roles": {"a": 1}}'],
)
def test_a_block_index_that_is_not_one_is_reported_as_corrupt(
    store: DirectoryStore,
    index: str,
) -> None:
    handle = store.save(manifest(), scalars())
    (store.root / handle / "blocks.json").write_text(
        index, encoding="utf-8", newline="\n"
    )

    with pytest.raises(CorruptCheckpointError, match="block index"):
        store.load(handle, manifest())


def test_a_block_index_that_drops_a_role_is_reported_as_corrupt(
    store: DirectoryStore,
) -> None:
    handle = store.save(manifest(), scalars())
    index = store.root / handle / "blocks.json"
    record = json.loads(index.read_text(encoding="utf-8"))
    del record["roles"][ENCODER]
    index.write_text(json.dumps(record), encoding="utf-8", newline="\n")

    with pytest.raises(CorruptCheckpointError, match=ENCODER):
        store.load(handle, manifest())


def test_a_missing_block_file_is_reported_as_corrupt(store: DirectoryStore) -> None:
    handle = store.save(manifest(), scalars())
    (store.root / handle / "roles" / f"{FORWARD}.pt").unlink()

    with pytest.raises(CorruptCheckpointError, match=FORWARD):
        store.load(handle, manifest())


@pytest.mark.parametrize("handle", ["", "../elsewhere", "run-0001", ".hidden"])
def test_a_handle_that_names_no_checkpoint_directory_is_unknown(
    store: DirectoryStore,
    handle: str,
) -> None:
    store.save(manifest(), scalars())

    with pytest.raises(UnknownCheckpointError):
        store.load(CheckpointId(handle), manifest())


def test_a_handle_naming_a_file_rather_than_a_directory_is_unknown(
    store: DirectoryStore,
) -> None:
    store.save(manifest(), scalars())
    (store.root / "000001-run-0001-step7").write_text(
        "", encoding="utf-8", newline="\n"
    )

    with pytest.raises(UnknownCheckpointError):
        store.load(CheckpointId("000001-run-0001-step7"), manifest())

    assert len(store.list()) == 1


def test_handles_list_oldest_first_across_store_instances(
    store: DirectoryStore,
) -> None:
    handles = tuple(
        store.save(manifest().with_(step=step), scalars()) for step in range(12)
    )

    assert store.list() == handles
    assert DirectoryStore(store.root).list() == handles
    assert DirectoryStore(store.root).save(manifest(), scalars()) not in handles


def test_a_handle_names_the_run_and_the_step(store: DirectoryStore) -> None:
    handle = store.save(manifest().with_(step=42), scalars())

    assert "run-0001" in handle
    assert "42" in handle


def test_an_absent_root_lists_nothing_and_is_not_created_by_listing(
    store: DirectoryStore,
) -> None:
    assert store.list() == ()
    assert not store.root.exists()


def test_the_root_is_created_on_first_save(store: DirectoryStore) -> None:
    store.save(manifest(), scalars())

    assert store.root.is_dir()


@pytest.mark.parametrize("role", ["a/b", "..", ".hidden", "a\\b"])
def test_a_role_that_cannot_name_a_file_is_refused(
    store: DirectoryStore,
    role: str,
) -> None:
    with pytest.raises(ValueError, match="file name"):
        store.save(manifest(**{role: decoder()}), {role: [1.0]})

    assert store.list() == ()


def test_a_run_id_that_cannot_name_a_file_is_refused(store: DirectoryStore) -> None:
    with pytest.raises(ValueError, match="run id"):
        store.save(manifest().with_(run_id="../run"), scalars())


def test_two_roles_differing_only_in_case_are_refused(store: DirectoryStore) -> None:
    built = manifest(Forward=decoder(), forward=decoder())

    with pytest.raises(ValueError, match="only in case"):
        store.save(built, {"Forward": [1.0], "forward": [2.0]})


def test_the_store_reports_its_root(store: DirectoryStore) -> None:
    assert repr(store) == f"DirectoryStore({store.root!r})"


def test_the_store_satisfies_the_checkpoint_store_protocol(
    store: DirectoryStore,
) -> None:
    assert isinstance(store, CheckpointStore)


def test_a_manifest_is_read_without_reading_a_block(store: DirectoryStore) -> None:
    handle = store.save(manifest(), scalars())
    (store.root / handle / "roles" / f"{ENCODER}.pt").write_bytes(b"")

    assert store.manifest(handle) == manifest()


def test_a_removed_checkpoint_leaves_nothing_on_disk_and_spares_the_others(
    store: DirectoryStore,
) -> None:
    first = store.save(manifest(), scalars())
    second = store.save(manifest().with_(step=8), scalars())

    store.remove(first)

    assert store.list() == (second,)
    assert sorted(entry.name for entry in store.root.iterdir()) == [second]
    assert store.load(second, manifest()).manifest.step == 8

    with pytest.raises(UnknownCheckpointError, match=first):
        store.manifest(first)


def test_a_save_after_a_removal_is_listed_after_the_survivors(
    store: DirectoryStore,
) -> None:
    first = store.save(manifest(), scalars())
    second = store.save(manifest().with_(step=8), scalars())
    store.remove(first)
    third = store.save(manifest().with_(step=9), scalars())

    assert store.list() == (second, third)


def test_removing_a_handle_the_store_never_wrote_is_refused(
    store: DirectoryStore,
) -> None:
    store.save(manifest(), scalars())

    with pytest.raises(UnknownCheckpointError, match="never-written"):
        store.remove(CheckpointId("never-written"))
