"""Independent capabilities for drawing finite-model denotations."""

from __future__ import annotations

from hashlib import sha256
from random import Random
from typing import TYPE_CHECKING, NamedTuple, Protocol, final, runtime_checkable

from lofi._frozen import FrozenValue
from lofi.lf.models._frame import FunctionValue
from lofi.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.lf.models._frame import ModelValue

__all__ = [
    "BERNOULLI_TRUTH",
    "INDEPENDENT_STREAMS",
    "TABLE_FUNCTIONS",
    "UNIFORM_ATOMIC",
    "AtomicSampling",
    "DenotationStreams",
    "FunctionSampling",
    "Sampling",
    "TruthSampling",
    "sampling",
]


def _seed(*parts: object) -> int:
    """Derive one stable random seed without a shared stream."""
    digest = sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


@runtime_checkable
class AtomicSampling(Protocol):
    """Draw one member of an atomic carrier."""

    def sample(
        self, values: tuple[ModelValue, ...], random: Random, /
    ) -> ModelValue: ...


@runtime_checkable
class DenotationStreams(Protocol):
    """Derive a constant's random stream within one structure draw."""

    def stream(
        self, seed: int, step: int, position: int, identifier: int, /
    ) -> Random: ...


@runtime_checkable
class FunctionSampling(Protocol):
    """Draw a function from its domain and a sampler for its result type."""

    def sample(
        self, domain: tuple[ModelValue, ...], draw: Callable[[], ModelValue], /
    ) -> FunctionValue: ...


@runtime_checkable
class TruthSampling(Protocol):
    """Draw a truth value under a declared density."""

    def sample(self, random: Random, density: float, /) -> bool: ...


class Sampling(NamedTuple):
    """Resolved capabilities used by a sampled structure source."""

    atomic: AtomicSampling
    streams: DenotationStreams
    functions: FunctionSampling
    truth: TruthSampling


@final
class _UniformAtomic(FrozenValue):
    """Draw each atomic carrier value with equal probability."""

    _noun = "uniform atomic sampler"
    __slots__ = ()

    def sample(self, values: tuple[ModelValue, ...], random: Random, /) -> ModelValue:
        return values[random.randrange(len(values))]


@final
class _IndependentStreams(FrozenValue):
    """Derive independent streams from stable vocabulary identifiers."""

    _noun = "independent denotation streams"
    __slots__ = ()

    def stream(self, seed: int, step: int, position: int, identifier: int, /) -> Random:
        return Random(_seed(seed, step, position, identifier))


@final
class _TableFunctions(FrozenValue):
    """Draw one result per domain value to form a total function table."""

    _noun = "function-table sampler"
    __slots__ = ()

    def sample(
        self, domain: tuple[ModelValue, ...], draw: Callable[[], ModelValue], /
    ) -> FunctionValue:
        outputs = tuple(draw() for _ in domain)
        return FunctionValue(domain, outputs, (True,) * len(domain))


@final
class _BernoulliTruth(FrozenValue):
    """Draw true with probability equal to the declared density."""

    _noun = "Bernoulli truth sampler"
    __slots__ = ()

    def sample(self, random: Random, density: float, /) -> bool:
        return random.random() < density


UNIFORM_ATOMIC = _UniformAtomic()
INDEPENDENT_STREAMS = _IndependentStreams()
TABLE_FUNCTIONS = _TableFunctions()
BERNOULLI_TRUTH = _BernoulliTruth()


def sampling(atomic: str, streams: str, functions: str, truth: str, /) -> Sampling:
    """Resolve and validate the four declared sampling capabilities."""
    atomic_sampler = Registry[object]("atomic_sampling").get(atomic)
    stream_source = Registry[object]("denotation_streams").get(streams)
    function_sampler = Registry[object]("function_sampling").get(functions)
    truth_sampler = Registry[object]("truth_sampling").get(truth)

    if not isinstance(atomic_sampler, AtomicSampling):
        msg = f"atomic-sampling entry {atomic!r} supplies no atomic sampler"
        raise TypeError(msg)
    if not isinstance(stream_source, DenotationStreams):
        msg = f"denotation-stream entry {streams!r} supplies no random streams"
        raise TypeError(msg)
    if not isinstance(function_sampler, FunctionSampling):
        msg = f"function-sampling entry {functions!r} supplies no function sampler"
        raise TypeError(msg)
    if not isinstance(truth_sampler, TruthSampling):
        msg = f"truth-sampling entry {truth!r} supplies no truth sampler"
        raise TypeError(msg)

    return Sampling(atomic_sampler, stream_source, function_sampler, truth_sampler)
