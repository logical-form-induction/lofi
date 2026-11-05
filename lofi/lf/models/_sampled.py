"""Deterministic sampled finite structures for scored model checking."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, final, runtime_checkable

from lofi._frozen import FrozenValue
from lofi.lf.models._evaluate import Evaluator, ModelLanguage, truth_type
from lofi.lf.models._frame import Frame, ModelValue
from lofi.lf.models._sampling import Sampling, _seed, sampling
from lofi.lf.models._structure import Structure

if TYPE_CHECKING:
    from collections.abc import Mapping
    from random import Random

    from lofi.lf.models._algebra import TruthAlgebra
    from lofi.terms import Term

__all__ = [
    "SAMPLED",
    "SampledSettings",
    "SampledStructureSource",
    "StructureSource",
    "StructureSourceFactory",
]


def _semilattice_types(language: ModelLanguage, /) -> tuple[str, ...]:
    """Derive join-semilattice carrier types from semantic capabilities."""
    held: set[str] = set()

    for constant, semantics in language.profile.logical_semantics.items():
        resource = semantics.model_resource

        if resource is None:
            continue

        if resource == "choice-function":
            continue

        if resource != "join-semilattice":
            msg = f"sampled structures do not supply model resource {resource!r}"
            raise ValueError(msg)

        body = language.scheme(constant).body

        if body.op.name != language.function_type or len(body.args) != 2:
            msg = f"join-semilattice constant {constant!r} needs an argument type"
            raise ValueError(msg)

        argument = body.args[0]

        if not argument.op.is_constant or argument.payload is None:
            msg = (
                f"join-semilattice constant {constant!r} needs an atomic argument type"
            )
            raise ValueError(msg)

        held.add(language.store.payload_name(argument))

    return tuple(sorted(held))


@runtime_checkable
class StructureSource(Protocol):
    """Supply finite structures for one model-checking step."""

    @property
    def name(self) -> str: ...

    @property
    def trainable(self) -> bool: ...

    @property
    def evidence_label(self) -> str: ...

    def structures(self) -> tuple[Structure, ...]: ...

    def open_step(self) -> None: ...


@runtime_checkable
class StructureSourceFactory(Protocol):
    """Build a structure source from one variant's resolved settings."""

    def build(
        self,
        settings: object,
        language: ModelLanguage,
        algebra: TruthAlgebra,
        /,
        *,
        theory_floor: float,
    ) -> StructureSource: ...


@final
class SampledSettings(FrozenValue):
    """Resolved sampled-source settings produced by the boundary declaration."""

    _noun = "sampled-source settings"
    __slots__ = (
        "_atomic_sampling",
        "_carriers",
        "_denotation_streams",
        "_density",
        "_full_function_space_cap",
        "_function_sampling",
        "_henkin_pool",
        "_resample",
        "_seed",
        "_structures",
        "_truth_sampling",
    )

    def __init__(
        self,
        *,
        atomic_sampling: str,
        denotation_streams: str,
        function_sampling: str,
        truth_sampling: str,
        structures: int,
        carriers: Mapping[str, int],
        density: float,
        full_function_space_cap: int,
        henkin_pool: int,
        seed: int,
        resample: Literal["once", "per-step"],
    ) -> None:
        object.__setattr__(self, "_atomic_sampling", atomic_sampling)
        object.__setattr__(self, "_denotation_streams", denotation_streams)
        object.__setattr__(self, "_function_sampling", function_sampling)
        object.__setattr__(self, "_truth_sampling", truth_sampling)
        object.__setattr__(self, "_structures", structures)
        object.__setattr__(self, "_carriers", MappingProxyType(dict(carriers)))
        object.__setattr__(self, "_density", density)
        object.__setattr__(self, "_full_function_space_cap", full_function_space_cap)
        object.__setattr__(self, "_henkin_pool", henkin_pool)
        object.__setattr__(self, "_seed", seed)
        object.__setattr__(self, "_resample", resample)

    @property
    def structures(self) -> int:
        return self._structures

    @property
    def carriers(self) -> Mapping[str, int]:
        return self._carriers

    @property
    def density(self) -> float:
        return self._density

    @property
    def full_function_space_cap(self) -> int:
        return self._full_function_space_cap

    @property
    def henkin_pool(self) -> int:
        return self._henkin_pool

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def resample(self) -> Literal["once", "per-step"]:
        return self._resample

    @property
    def atomic_sampling(self) -> str:
        return self._atomic_sampling

    @property
    def denotation_streams(self) -> str:
        return self._denotation_streams

    @property
    def function_sampling(self) -> str:
        return self._function_sampling

    @property
    def truth_sampling(self) -> str:
        return self._truth_sampling


@final
class SampledStructureSource:
    """Draw crisp denotations from stable per-constant random streams."""

    __slots__ = (
        "_algebra",
        "_cache",
        "_carriers",
        "_count",
        "_density",
        "_function_cap",
        "_henkin_pool",
        "_language",
        "_resample",
        "_sampling",
        "_seed",
        "_semilattice_types",
        "_step",
        "_theory_floor",
        "_truth_type",
    )

    def __init__(
        self,
        language: ModelLanguage,
        algebra: TruthAlgebra,
        /,
        *,
        structures: int,
        carriers: Mapping[str, int],
        density: float,
        full_function_space_cap: int,
        henkin_pool: int,
        seed: int,
        resample: Literal["once", "per-step"],
        theory_floor: float,
        sampling: Sampling,
    ) -> None:
        if structures < 1:
            msg = f"a sampled source draws at least one structure, not {structures!r}"
            raise ValueError(msg)

        if not 0.0 <= density <= 1.0:
            msg = f"a denotation density lies in the unit interval, not {density!r}"
            raise ValueError(msg)

        if resample not in ("once", "per-step"):
            msg = f"unknown structure resampling schedule {resample!r}"
            raise ValueError(msg)

        if not 0.0 <= theory_floor <= 1.0:
            msg = f"a theory floor lies in the unit interval, not {theory_floor!r}"
            raise ValueError(msg)

        self._sampling = sampling
        self._language = language
        self._algebra = algebra
        self._count = structures
        self._carriers = dict(carriers)
        self._density = density
        self._function_cap = full_function_space_cap
        self._henkin_pool = henkin_pool
        self._seed = seed
        self._resample = resample
        self._theory_floor = theory_floor
        self._truth_type = truth_type(language)
        self._semilattice_types = _semilattice_types(language)
        self._step = 0
        self._cache: tuple[Structure, ...] | None = None

    @property
    def name(self) -> str:
        """Return the registered source name."""
        return "sampled"

    @property
    def trainable(self) -> bool:
        """Return false because sampled structures hold no parameters."""
        return False

    @property
    def evidence_label(self) -> str:
        """Return the evidence relation sampled structures estimate."""
        return "conditional satisfaction"

    def structures(self) -> tuple[Structure, ...]:
        """Return this step's deterministic, theory-satisfying structures."""
        if self._cache is None:
            self._cache = self._draw()

        return self._cache

    def open_step(self) -> None:
        """Advance the deterministic stream exactly under per-step resampling."""
        if self._resample == "per-step":
            self._step += 1
            self._cache = None

    def _draw(self) -> tuple[Structure, ...]:
        accepted: list[Structure] = []
        attempts = 0
        limit = self._count * 100

        while len(accepted) < self._count and attempts < limit:
            candidate = self._structure(attempts)
            attempts += 1
            evaluator = Evaluator(self._language, candidate, self._algebra)

            if evaluator.theory_holds(floor=self._theory_floor):
                accepted.append(candidate)

        if len(accepted) != self._count:
            msg = (
                f"sampled {attempts} structures but only {len(accepted)} satisfy "
                "the declared theory"
            )
            raise ValueError(msg)

        return tuple(accepted)

    def _structure(self, position: int, /) -> Structure:
        frame_seed = _seed(self._seed, self._step, position, "frame")
        truth_name = self._language.store.payload_name(self._truth_type)
        frame = Frame(
            self._language.store,
            function_type=self._language.function_type,
            truth_type=truth_name,
            carriers=self._carriers,
            full_function_space_cap=self._function_cap,
            henkin_pool=self._henkin_pool,
            seed=frame_seed,
        )
        vocabulary = self._language.store.signature.vocabulary(self._language.term_sort)
        logical = self._language.profile.logical_constants
        denotations: dict[int, ModelValue] = {}

        for identifier, name in enumerate(vocabulary.names):
            scheme = self._language.scheme(name)

            if name in logical or scheme.quantified:
                continue

            random = self._sampling.streams.stream(
                self._seed, self._step, position, identifier
            )
            denotations[identifier] = self._sample(scheme.body, frame, random)

        return Structure(
            f"sampled-{self._step}-{position}",
            frame,
            denotations,
            semilattices=self._semilattice_types,
        )

    def _sample(self, type_: Term, frame: Frame, random: Random, /) -> ModelValue:
        if type_ == self._truth_type:
            return self._sampling.truth.sample(random, self._density)

        if type_.op.name == self._language.function_type and len(type_.args) == 2:
            domain = frame.domain(type_.args[0]).values
            return self._sampling.functions.sample(
                domain, lambda: self._sample(type_.args[1], frame, random)
            )

        values = frame.domain(type_).values
        return self._sampling.atomic.sample(values, random)


@final
class _SampledFactory(FrozenValue):
    """Build the sampled source registered under its boundary tag."""

    _noun = "structure-source factory"
    __slots__ = ()

    def build(
        self,
        settings: object,
        language: ModelLanguage,
        algebra: TruthAlgebra,
        /,
        *,
        theory_floor: float,
    ) -> StructureSource:
        if not isinstance(settings, SampledSettings):
            msg = f"the sampled source reads SampledSettings, not {settings!r}"
            raise TypeError(msg)

        return SampledStructureSource(
            language,
            algebra,
            structures=settings.structures,
            carriers=settings.carriers,
            density=settings.density,
            full_function_space_cap=settings.full_function_space_cap,
            henkin_pool=settings.henkin_pool,
            seed=settings.seed,
            resample=settings.resample,
            theory_floor=theory_floor,
            sampling=sampling(
                settings.atomic_sampling,
                settings.denotation_streams,
                settings.function_sampling,
                settings.truth_sampling,
            ),
        )


SAMPLED = _SampledFactory()
