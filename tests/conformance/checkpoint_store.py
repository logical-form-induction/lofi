from typing import TYPE_CHECKING, ClassVar, Final

import pytest

from lofi.train.laws import (
    a_fingerprint_mismatch_is_refused,
    a_removed_checkpoint_is_neither_listed_nor_read,
    a_round_trip_restores_the_step_count_and_the_random_state,
    association_is_by_declared_role,
    every_saved_checkpoint_is_listed,
    two_components_of_one_kind_do_not_collide,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.train.laws import CheckpointFixture

__all__ = [
    "CHECKPOINT_STORE_CONFORMANCE_LAWS",
    "EMPTY_SEED_POLICY",
    "CheckpointStoreConformance",
]

EMPTY_SEED_POLICY: Final[dict[str, JsonValue]] = {
    "root": 0,
    "accelerator": None,
    "samplers": [],
    "loaders": [],
    "workers": [],
}
"""Canonical no-named-stream policy metadata used by random-stream fixtures."""


class CheckpointStoreConformance:
    pytestmark: ClassVar[pytest.MarkDecorator] = pytest.mark.conformance

    def test_association_is_by_declared_role[StateT](
        self,
        conformance: CheckpointFixture[StateT],
    ) -> None:
        association_is_by_declared_role(conformance)

    def test_a_fingerprint_mismatch_is_refused[StateT](
        self,
        conformance: CheckpointFixture[StateT],
    ) -> None:
        a_fingerprint_mismatch_is_refused(conformance)

    def test_two_components_of_one_kind_do_not_collide[StateT](
        self,
        conformance: CheckpointFixture[StateT],
    ) -> None:
        two_components_of_one_kind_do_not_collide(conformance)

    def test_a_round_trip_restores_the_step_count_and_the_random_state[StateT](
        self,
        conformance: CheckpointFixture[StateT],
    ) -> None:
        a_round_trip_restores_the_step_count_and_the_random_state(conformance)

    def test_every_saved_checkpoint_is_listed[StateT](
        self,
        conformance: CheckpointFixture[StateT],
    ) -> None:
        every_saved_checkpoint_is_listed(conformance)

    def test_a_removed_checkpoint_is_neither_listed_nor_read[StateT](
        self,
        conformance: CheckpointFixture[StateT],
    ) -> None:
        a_removed_checkpoint_is_neither_listed_nor_read(conformance)


CHECKPOINT_STORE_CONFORMANCE_LAWS: Final = (
    CheckpointStoreConformance.test_association_is_by_declared_role,
    CheckpointStoreConformance.test_a_fingerprint_mismatch_is_refused,
    CheckpointStoreConformance.test_two_components_of_one_kind_do_not_collide,
    CheckpointStoreConformance.test_a_round_trip_restores_the_step_count_and_the_random_state,
    CheckpointStoreConformance.test_every_saved_checkpoint_is_listed,
    CheckpointStoreConformance.test_a_removed_checkpoint_is_neither_listed_nor_read,
)
