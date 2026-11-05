from typing import TYPE_CHECKING, ClassVar, Final, override

import pytest
from tests.conformance.checkpoint_store import (
    CHECKPOINT_STORE_CONFORMANCE_LAWS,
    CheckpointStoreConformance,
)

from lofi.train import (
    Checkpoint,
    CheckpointId,
    CheckpointManifest,
    ComponentRecord,
    FingerprintMismatchError,
    PrecisionRecord,
    RngRecord,
    StateReference,
    UnknownCheckpointError,
    disagreements,
)
from lofi.train.laws import CHECKPOINT_STORE_LAWS, CheckpointFixture

if TYPE_CHECKING:
    from collections.abc import Mapping

FORWARD: Final = "forward_application"

BACKWARD: Final = "backward_application"

ENCODER: Final = "terms"


def decoder() -> ComponentRecord:
    return ComponentRecord(
        tag="hrr",
        spec_fingerprint="spec-a1",
        signature_fingerprint="sig-b2",
        codec_identity="codec-c3",
        state=StateReference(key="decoder", digest="d4"),
    )


def manifest() -> CheckpointManifest:
    return CheckpointManifest(
        run_id="run-0001",
        step=7,
        data_position=0,
        code_version="0.1.0",
        action_set="actions-a1",
        components={
            FORWARD: decoder(),
            BACKWARD: decoder(),
            ENCODER: ComponentRecord(
                tag="rnn",
                spec_fingerprint="spec-e5",
                signature_fingerprint="sig-b2",
                codec_identity="codec-c3",
                state=StateReference(key="encoder", digest="f6"),
            ),
        },
        rng=RngRecord(streams={"python": "9f3a", "torch": "1c20"}),
        precision=PrecisionRecord(
            compute_dtype="bfloat16",
            autocast_device_type="cuda",
            gradient_scaling=True,
        ),
        optimizer=StateReference(key="optimizer", digest="a7"),
        scaler=StateReference(key="scaler", digest="b8"),
        scheduler=StateReference(key="scheduler", digest="c9"),
    )


def mismatched() -> CheckpointManifest:
    saved = manifest()
    moved = saved.components[ENCODER].with_(spec_fingerprint="spec-e6")

    return saved.with_(components={**saved.components, ENCODER: moved})


def states() -> Mapping[str, str]:
    return {
        FORWARD: "weights-of-the-forward-decoder",
        BACKWARD: "weights-of-the-backward-decoder",
        ENCODER: "weights-of-the-term-encoder",
    }


class MemoryStore:
    __slots__ = ("_held", "_order")

    def __init__(self) -> None:
        self._held: dict[CheckpointId, tuple[CheckpointManifest, dict[str, str]]] = {}
        self._order: list[CheckpointId] = []

    def save(
        self,
        manifest: CheckpointManifest,
        states: Mapping[str, str],
        /,
    ) -> CheckpointId:
        handle = CheckpointId(f"{manifest.run_id}-{len(self._order):04d}")

        self._held[handle] = (manifest, dict(states))
        self._order.append(handle)

        return handle

    def load(
        self,
        checkpoint: CheckpointId,
        expected: CheckpointManifest,
        /,
    ) -> Checkpoint[str]:
        if checkpoint not in self._held:
            raise UnknownCheckpointError(checkpoint)

        saved, held = self._held[checkpoint]
        reasons = disagreements(expected, saved)

        if reasons:
            raise FingerprintMismatchError(checkpoint, reasons)

        return Checkpoint(checkpoint, saved, held)

    def list(self) -> tuple[CheckpointId, ...]:
        return tuple(self._order)

    def manifest(self, checkpoint: CheckpointId, /) -> CheckpointManifest:
        if checkpoint not in self._held:
            raise UnknownCheckpointError(checkpoint)

        return self._held[checkpoint][0]

    def remove(self, checkpoint: CheckpointId, /) -> None:
        if checkpoint not in self._held:
            raise UnknownCheckpointError(checkpoint)

        del self._held[checkpoint]
        self._order.remove(checkpoint)


class RecordKeyedStore(MemoryStore):
    __slots__ = ()

    @override
    def save(
        self,
        manifest: CheckpointManifest,
        states: Mapping[str, str],
        /,
    ) -> CheckpointId:
        by_record: dict[ComponentRecord, str] = {
            manifest.components[role]: state for role, state in states.items()
        }
        collapsed = {
            role: by_record[record] for role, record in manifest.components.items()
        }

        return super().save(manifest, collapsed)


class TrustingStore(MemoryStore):
    __slots__ = ()

    @override
    def load(
        self,
        checkpoint: CheckpointId,
        expected: CheckpointManifest,
        /,
    ) -> Checkpoint[str]:
        del expected

        return super().load(checkpoint, self.manifest(checkpoint))


class ForgetfulStore(MemoryStore):
    __slots__ = ()

    @override
    def save(
        self,
        manifest: CheckpointManifest,
        states: Mapping[str, str],
        /,
    ) -> CheckpointId:
        blank = manifest.with_(step=0, rng=RngRecord(streams={}))

        return super().save(blank, states)


class UnforgettingStore(MemoryStore):
    __slots__ = ()

    @override
    def remove(self, checkpoint: CheckpointId, /) -> None:
        if checkpoint not in self._held:
            raise UnknownCheckpointError(checkpoint)


class NullStore:
    __slots__ = ()

    def save(
        self,
        manifest: CheckpointManifest,
        states: Mapping[str, str],
        /,
    ) -> CheckpointId:
        del states

        return CheckpointId(f"{manifest.run_id}-null")

    def load(
        self,
        checkpoint: CheckpointId,
        expected: CheckpointManifest,
        /,
    ) -> Checkpoint[str]:
        del expected

        return Checkpoint(
            checkpoint,
            CheckpointManifest(
                run_id="null",
                step=0,
                data_position=0,
                code_version="0.0.0",
                action_set="actions-a1",
                components={},
                rng=RngRecord(streams={}),
                precision=PrecisionRecord(
                    compute_dtype="float32",
                    autocast_device_type="cpu",
                    gradient_scaling=False,
                ),
            ),
            {},
        )

    def list(self) -> tuple[CheckpointId, ...]:
        return ()

    def manifest(self, checkpoint: CheckpointId, /) -> CheckpointManifest:
        return self.load(checkpoint, manifest()).manifest

    def remove(self, checkpoint: CheckpointId, /) -> None:
        del checkpoint


def fixture_for(store: NullStore | MemoryStore) -> CheckpointFixture[str]:
    return CheckpointFixture(
        store=store,
        manifest=manifest(),
        states=states,
        mismatched=mismatched(),
        twins=(FORWARD, BACKWARD),
    )


class TestAStoreKeyedOnTheDeclaredRole(CheckpointStoreConformance):
    @pytest.fixture
    def conformance(self) -> CheckpointFixture[str]:
        return fixture_for(MemoryStore())


class TestAStoreKeyedOnTheComponentRecord(CheckpointStoreConformance):
    @pytest.fixture
    def conformance(self) -> CheckpointFixture[str]:
        return fixture_for(RecordKeyedStore())

    @pytest.mark.xfail(strict=True, reason="the two decoders share one entry")
    @override
    def test_association_is_by_declared_role[StateT](
        self,
        conformance: CheckpointFixture[StateT],
    ) -> None:
        super().test_association_is_by_declared_role(conformance)

    @pytest.mark.xfail(strict=True, reason="the two decoders share one entry")
    @override
    def test_two_components_of_one_kind_do_not_collide[StateT](
        self,
        conformance: CheckpointFixture[StateT],
    ) -> None:
        super().test_two_components_of_one_kind_do_not_collide(conformance)


class TestAStoreThatComparesNothing(CheckpointStoreConformance):
    @pytest.fixture
    def conformance(self) -> CheckpointFixture[str]:
        return fixture_for(TrustingStore())

    @pytest.mark.xfail(strict=True, reason="the manifests are never compared")
    @override
    def test_a_fingerprint_mismatch_is_refused[StateT](
        self,
        conformance: CheckpointFixture[StateT],
    ) -> None:
        super().test_a_fingerprint_mismatch_is_refused(conformance)


class TestAStoreThatDropsTheRunState(CheckpointStoreConformance):
    @pytest.fixture
    def conformance(self) -> CheckpointFixture[str]:
        return fixture_for(ForgetfulStore())

    @pytest.mark.xfail(strict=True, reason="the counter and the streams are dropped")
    @override
    def test_a_round_trip_restores_the_step_count_and_the_random_state[StateT](
        self,
        conformance: CheckpointFixture[StateT],
    ) -> None:
        super().test_a_round_trip_restores_the_step_count_and_the_random_state(
            conformance
        )

    @pytest.mark.xfail(strict=True, reason="the manifest is read back without its step")
    @override
    def test_a_removed_checkpoint_is_neither_listed_nor_read[StateT](
        self,
        conformance: CheckpointFixture[StateT],
    ) -> None:
        super().test_a_removed_checkpoint_is_neither_listed_nor_read(conformance)


class TestAStoreThatRemovesNothing(CheckpointStoreConformance):
    @pytest.fixture
    def conformance(self) -> CheckpointFixture[str]:
        return fixture_for(UnforgettingStore())

    @pytest.mark.xfail(strict=True, reason="a removed checkpoint stays listed")
    @override
    def test_a_removed_checkpoint_is_neither_listed_nor_read[StateT](
        self,
        conformance: CheckpointFixture[StateT],
    ) -> None:
        super().test_a_removed_checkpoint_is_neither_listed_nor_read(conformance)


class TestANullStore(CheckpointStoreConformance):
    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [
        pytest.mark.conformance,
        pytest.mark.xfail(strict=True),
    ]

    @pytest.fixture
    def conformance(self) -> CheckpointFixture[str]:
        return fixture_for(NullStore())


def test_the_suite_binds_a_method_to_every_checkpoint_store_law() -> None:
    bound = {
        method.__name__.removeprefix("test_")
        for method in CHECKPOINT_STORE_CONFORMANCE_LAWS
    }

    assert bound == {law.__name__ for law in CHECKPOINT_STORE_LAWS}


def test_a_store_refuses_a_handle_it_never_wrote() -> None:
    with pytest.raises(UnknownCheckpointError, match="never-written"):
        MemoryStore().load(CheckpointId("never-written"), manifest())


def test_a_refusal_names_every_disagreement_rather_than_the_first() -> None:
    store = MemoryStore()
    handle = store.save(manifest(), states())
    apart = mismatched().with_(code_version="0.2.0")

    with pytest.raises(FingerprintMismatchError) as raised:
        store.load(handle, apart)

    assert len(raised.value.reasons) == 2
