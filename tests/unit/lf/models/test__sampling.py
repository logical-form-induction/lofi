from random import Random

import pytest

from lofi.lf.models import (
    BERNOULLI_TRUTH,
    INDEPENDENT_STREAMS,
    TABLE_FUNCTIONS,
    UNIFORM_ATOMIC,
    AtomicSampling,
    DenotationStreams,
    Element,
    FunctionSampling,
    TruthSampling,
    sampling,
)
from lofi.registry import RegistryError


def test_uniform_atomic_sampling_replays_a_uniform_index_draw() -> None:
    values = tuple(Element("entity", position) for position in range(5))
    random = Random(17)
    reference = Random(17)

    assert isinstance(UNIFORM_ATOMIC, AtomicSampling)
    assert [UNIFORM_ATOMIC.sample(values, random) for _ in range(20)] == [
        values[reference.randrange(len(values))] for _ in range(20)
    ]


def test_independent_streams_do_not_depend_on_constant_visit_order() -> None:
    assert isinstance(INDEPENDENT_STREAMS, DenotationStreams)
    forward = {
        identifier: INDEPENDENT_STREAMS.stream(7, 2, 3, identifier).random()
        for identifier in (0, 1, 2)
    }
    backward = {
        identifier: INDEPENDENT_STREAMS.stream(7, 2, 3, identifier).random()
        for identifier in (2, 1, 0)
    }

    assert forward == backward
    assert len(set(forward.values())) == len(forward)
    assert INDEPENDENT_STREAMS.stream(7, 3, 3, 0).random() != forward[0]


def test_function_tables_draw_every_domain_members_result() -> None:
    domain = (Element("entity", 0), Element("entity", 1))
    outputs = iter((False, True))

    assert isinstance(TABLE_FUNCTIONS, FunctionSampling)
    found = TABLE_FUNCTIONS.sample(domain, lambda: next(outputs))

    assert found.domain == domain
    assert found.outputs == (False, True)
    assert found.known == (True, True)


def test_bernoulli_truth_uses_the_declared_density() -> None:
    assert isinstance(BERNOULLI_TRUTH, TruthSampling)
    random = Random(11)
    reference = Random(11)

    assert [BERNOULLI_TRUTH.sample(random, 0.3) for _ in range(20)] == [
        reference.random() < 0.3 for _ in range(20)
    ]
    assert not BERNOULLI_TRUTH.sample(random, 0.0)
    assert BERNOULLI_TRUTH.sample(random, 1.0)


def test_installed_sampling_choices_resolve_independently() -> None:
    choices = sampling("uniform", "independent", "table", "bernoulli")

    assert choices.atomic is UNIFORM_ATOMIC
    assert choices.streams is INDEPENDENT_STREAMS
    assert choices.functions is TABLE_FUNCTIONS
    assert choices.truth is BERNOULLI_TRUTH

    with pytest.raises(RegistryError, match="atomic_sampling"):
        sampling("unknown", "independent", "table", "bernoulli")
