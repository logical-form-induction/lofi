"""Bind random-stream laws to the run implementation."""

import random
from typing import TYPE_CHECKING

import pytest
import torch
from tests.conformance.random_streams import (
    RandomStreamsConformance,
    RandomStreamsFixture,
)

from lofi.run import RandomStreams, SeedPolicy

if TYPE_CHECKING:
    from lofi.train import RestorableRandomStreams


def build(root: int) -> RestorableRandomStreams:
    streams = RandomStreams(
        SeedPolicy(root, samplers=("fit",), loaders=("fit",), workers=("fit",))
    )
    streams.apply()

    return streams


def draw(streams: RestorableRandomStreams) -> tuple[int, ...]:
    return (
        random.getrandbits(32),
        int(torch.randint(2**31, ()).item()),
        int(torch.randint(2**31, (), generator=streams.loader_generator("fit")).item()),
        streams.sampler_seed("fit"),
    )


class TestRunRandomStreams(RandomStreamsConformance):
    @pytest.fixture
    def random_streams_fixture(self) -> RandomStreamsFixture:
        return RandomStreamsFixture(build, draw)
