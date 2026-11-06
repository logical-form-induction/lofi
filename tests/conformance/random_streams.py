"""Reusable laws for restorable random-stream implementations."""

from typing import TYPE_CHECKING, NamedTuple

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.train import RestorableRandomStreams

__all__ = ["RandomStreamsConformance", "RandomStreamsFixture"]


class RandomStreamsFixture(NamedTuple):
    """Factories and observations for random-stream laws."""

    build: Callable[[int], RestorableRandomStreams]
    draw: Callable[[RestorableRandomStreams], tuple[int, ...]]


class RandomStreamsConformance:
    """Conformance tests for a complete restorable random-stream implementation."""

    def test_same_root_repeats_every_stream(
        self, random_streams_fixture: RandomStreamsFixture
    ) -> None:
        first = random_streams_fixture.draw(random_streams_fixture.build(41))
        repeated = random_streams_fixture.draw(random_streams_fixture.build(41))

        assert first == repeated

    def test_a_different_root_changes_every_stream(
        self, random_streams_fixture: RandomStreamsFixture
    ) -> None:
        first = random_streams_fixture.draw(random_streams_fixture.build(43))
        changed = random_streams_fixture.draw(random_streams_fixture.build(44))

        assert all(left != right for left, right in zip(first, changed, strict=True))

    def test_restoration_reproduces_the_next_values(
        self, random_streams_fixture: RandomStreamsFixture
    ) -> None:
        streams = random_streams_fixture.build(47)
        random_streams_fixture.draw(streams)
        snapshot = streams.snapshot()
        expected = random_streams_fixture.draw(streams)
        random_streams_fixture.draw(streams)

        streams.restore(snapshot)

        assert random_streams_fixture.draw(streams) == expected

    def test_successful_validation_does_not_advance_a_stream(
        self, random_streams_fixture: RandomStreamsFixture
    ) -> None:
        streams = random_streams_fixture.build(49)
        random_streams_fixture.draw(streams)
        snapshot = streams.snapshot()

        streams.validate(snapshot)
        observed = random_streams_fixture.draw(streams)
        control = random_streams_fixture.build(49)
        random_streams_fixture.draw(control)

        assert observed == random_streams_fixture.draw(control)

    def test_refused_validation_does_not_advance_a_stream(
        self, random_streams_fixture: RandomStreamsFixture
    ) -> None:
        streams = random_streams_fixture.build(53)
        random_streams_fixture.draw(streams)
        snapshot = streams.snapshot()
        malformed = dict(snapshot)
        del malformed[next(iter(malformed))]

        with pytest.raises(ValueError, match="missing keys"):
            streams.validate(malformed)
        observed = random_streams_fixture.draw(streams)
        control = random_streams_fixture.build(53)
        random_streams_fixture.draw(control)

        assert observed == random_streams_fixture.draw(control)

    def test_restore_preparation_does_not_advance_and_applies_later(
        self, random_streams_fixture: RandomStreamsFixture
    ) -> None:
        streams = random_streams_fixture.build(59)
        random_streams_fixture.draw(streams)
        snapshot = streams.snapshot()
        candidate = dict(snapshot)
        prepared = streams.prepare_restore(candidate)
        expected = random_streams_fixture.draw(streams)
        random_streams_fixture.draw(streams)
        candidate.clear()

        prepared()

        assert random_streams_fixture.draw(streams) == expected
