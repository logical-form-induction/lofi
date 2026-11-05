"""Run-level random-stream policy and restorable backend state."""

import base64
import json
import math
import random
from importlib import import_module
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Never, Protocol, TypeIs, cast, final, override

from lofi._checks import (
    ensure_name,
    ensure_non_negative,
    ensure_positive,
    refuse_reassignment,
)
from lofi._frozen import FrozenValue
from lofi.spec import data_seed, derive_seed, point_root

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from torch import Generator, Tensor

    from lofi._types import JsonValue

__all__ = [
    "AcceleratorRandom",
    "ExperimentSeed",
    "PointSeed",
    "RandomStreams",
    "SeedPolicy",
]

type _PythonState = tuple[int, tuple[int, ...], float | None]
type _PreparedRandomState = tuple[
    _PythonState,
    Tensor,
    dict[str, Tensor],
    AcceleratorRandom | None,
    bytes | None,
]
type _SeedPolicyReduction = tuple[
    Callable[
        [int, str | None, tuple[str, ...], tuple[str, ...], tuple[str, ...]],
        SeedPolicy,
    ],
    tuple[int, str | None, tuple[str, ...], tuple[str, ...], tuple[str, ...]],
]
type _WorkerReduction = tuple[
    Callable[[SeedPolicy, str], _WorkerInitializer],
    tuple[SeedPolicy, str],
]


class AcceleratorRandom(Protocol):
    """Random-state access for one declared accelerator."""

    @property
    def name(self) -> str:
        """Return the device declaration this backend controls."""
        ...

    def seed(self, seed: int, /) -> None:
        """Seed the accelerator's default stream.

        Parameters
        ----------
        seed : int
            The nonnegative seed assigned to the accelerator.
        """
        ...

    def snapshot(self) -> bytes:
        """Return the accelerator's current random state."""
        ...

    def validate(self, state: bytes, /) -> None:
        """Refuse bytes that cannot be restored without changing state.

        Parameters
        ----------
        state : bytes
            A candidate state returned by :meth:`snapshot`.
        """
        ...

    def restore(self, state: bytes, /) -> None:
        """Restore a previously validated state without validating it again.

        Parameters
        ----------
        state : bytes
            State returned by :meth:`snapshot`.
        """
        ...


@final
class SeedPolicy:
    """An immutable root seed and its declared named streams.

    Parameters
    ----------
    root : int
        The run seed from which every stream seed is derived.
    accelerator : str | None
        The accelerator whose default random stream participates, or None.
    samplers : Sequence[str]
        Sampler stream names.
    loaders : Sequence[str]
        Retained data-loader generator names.
    workers : Sequence[str]
        Data-loader worker-family names.
    """

    __slots__ = ("_accelerator", "_loaders", "_root", "_samplers", "_workers")

    def __init__(
        self,
        root: int,
        *,
        accelerator: str | None = None,
        samplers: Sequence[str] = (),
        loaders: Sequence[str] = (),
        workers: Sequence[str] = (),
    ) -> None:
        object.__setattr__(self, "_root", ensure_non_negative("root seed", root))
        object.__setattr__(
            self,
            "_accelerator",
            None
            if accelerator is None
            else ensure_name("accelerator device", accelerator),
        )
        object.__setattr__(self, "_samplers", _stream_names("sampler", samplers))
        object.__setattr__(self, "_loaders", _stream_names("loader", loaders))
        object.__setattr__(self, "_workers", _stream_names("worker", workers))

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("seed policy")

    @override
    def __delattr__(self, name: str) -> Never:
        """Raise AttributeError on field deletion."""
        del name

        refuse_reassignment("seed policy")

    @property
    def root(self) -> int:
        """Return the run's root seed."""
        return self._root

    @property
    def accelerator(self) -> str | None:
        """Return the declared accelerator, or None."""
        return self._accelerator

    @property
    def samplers(self) -> tuple[str, ...]:
        """Return the sampler stream names in canonical order."""
        return self._samplers

    @property
    def loaders(self) -> tuple[str, ...]:
        """Return the loader stream names in canonical order."""
        return self._loaders

    @property
    def workers(self) -> tuple[str, ...]:
        """Return the worker-family names in canonical order."""
        return self._workers

    @property
    def record(self) -> Mapping[str, JsonValue]:
        """Return canonical policy metadata for a manifest."""
        return MappingProxyType(
            {
                "root": self._root,
                "accelerator": self._accelerator,
                "samplers": list(self._samplers),
                "loaders": list(self._loaders),
                "workers": list(self._workers),
            }
        )

    def sampler_seed(self, name: str, /) -> int:
        """Return the seed assigned to one declared sampler.

        Parameters
        ----------
        name : str
            The sampler stream name.

        Raises
        ------
        KeyError
            If no sampler with that name was declared.
        """
        _require_declared("sampler", name, self._samplers)

        return self._derive("sampler", name)

    def retried(self, attempt: int, /) -> SeedPolicy:
        """Return this stream declaration beneath one numerical-retry seed."""
        position = ensure_positive("a numerical retry", attempt)
        return SeedPolicy(
            derive_seed(self._root, f"numerical-retry:{position}"),
            accelerator=self._accelerator,
            samplers=self._samplers,
            loaders=self._loaders,
            workers=self._workers,
        )

    def loader_seed(self, name: str, /) -> int:
        """Return the seed assigned to one declared loader.

        Parameters
        ----------
        name : str
            The loader stream name.

        Raises
        ------
        KeyError
            If no loader with that name was declared.
        """
        _require_declared("loader", name, self._loaders)

        return self._derive("loader", name)

    def worker_seed(self, name: str, worker: int, /) -> int:
        """Return the seed for a worker of a declared family.

        Parameters
        ----------
        name : str
            The worker-family name.
        worker : int
            The nonnegative worker position.

        Raises
        ------
        KeyError
            If no worker family with that name was declared.
        """
        _require_declared("worker", name, self._workers)
        position = ensure_non_negative("worker position", worker)

        return derive_seed(self._root, f"worker:{name}:{position}")

    def _derive(self, family: str, name: str) -> int:
        """Return one derived seed under a family and name."""
        return derive_seed(self._root, f"{family}:{name}")

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another policy declares the same streams."""
        if type(other) is not SeedPolicy:
            return NotImplemented

        return (
            self._root,
            self._accelerator,
            self._samplers,
            self._loaders,
            self._workers,
        ) == (
            other._root,
            other._accelerator,
            other._samplers,
            other._loaders,
            other._workers,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the complete policy."""
        return hash(
            (
                SeedPolicy,
                self._root,
                self._accelerator,
                self._samplers,
                self._loaders,
                self._workers,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this policy."""
        return (
            f"SeedPolicy({self._root!r}, accelerator={self._accelerator!r}, "
            f"samplers={self._samplers!r}, loaders={self._loaders!r}, "
            f"workers={self._workers!r})"
        )

    @override
    def __reduce__(self) -> _SeedPolicyReduction:
        """Return the reconstruction call."""
        return (
            _restore_seed_policy,
            (
                self._root,
                self._accelerator,
                self._samplers,
                self._loaders,
                self._workers,
            ),
        )


@final
class ExperimentSeed(FrozenValue):
    """An experiment's root seed, from which every point's run seed is derived.

    A point's run seed is ``derive_seed(root, "point:<point_id>")``, so it
    depends on the experiment's root and on the point's content-addressed
    identity and on nothing else: two experiments with different roots give one
    point different seeds, and one experiment gives a point the same seed in
    every process.

    Parameters
    ----------
    root : int
        The experiment's nonnegative root seed.
    """

    _noun = "experiment seed"

    __slots__ = ("_root",)

    def __init__(self, root: int) -> None:
        object.__setattr__(
            self, "_root", ensure_non_negative("experiment root seed", root)
        )

    @property
    def root(self) -> int:
        """Return the experiment's root seed."""
        return self._root

    def point(self, point_id: str, seed: int, /) -> PointSeed:
        """Return the run seed one point derives under this root.

        Parameters
        ----------
        point_id : str
            The point's content-addressed identifier.
        seed : int
            The sweep seed the point was expanded with, which its identifier
            already covers and which is recorded so the identifier can be
            recomputed from its configuration.

        Returns
        -------
        PointSeed
            The point's derivation.
        """
        return PointSeed(self, point_id, seed)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another experiment seed has the same root."""
        if type(other) is not ExperimentSeed:
            return NotImplemented

        return self._root == other._root

    @override
    def __hash__(self) -> int:
        """Return a hash over the root."""
        return hash((ExperimentSeed, self._root))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this seed."""
        return f"ExperimentSeed({self._root!r})"


@final
class PointSeed(FrozenValue):
    """The run seed one point derives from its experiment's root seed.

    Parameters
    ----------
    experiment : ExperimentSeed
        The experiment root the seed is derived from.
    point_id : str
        The point's content-addressed identifier.
    seed : int
        The sweep seed the point was expanded with.
    """

    _noun = "point seed"

    __slots__ = ("_experiment", "_point_id", "_root", "_seed")

    def __init__(self, experiment: ExperimentSeed, point_id: str, seed: int) -> None:
        checked = ensure_name("point identifier", point_id)
        object.__setattr__(self, "_experiment", experiment)
        object.__setattr__(self, "_point_id", checked)
        object.__setattr__(self, "_seed", ensure_non_negative("sweep seed", seed))
        object.__setattr__(self, "_root", point_root(experiment.root, checked))

    @property
    def experiment(self) -> ExperimentSeed:
        """Return the experiment root the seed is derived from."""
        return self._experiment

    @property
    def point_id(self) -> str:
        """Return the identifier of the point the seed belongs to."""
        return self._point_id

    @property
    def seed(self) -> int:
        """Return the sweep seed the point was expanded with."""
        return self._seed

    @property
    def root(self) -> int:
        """Return the derived run seed every stream of the point descends from."""
        return self._root

    def policy(self, *, accelerator: str | None, samplers: Sequence[str]) -> SeedPolicy:
        """Return the point's stream policy over its derived run seed.

        Parameters
        ----------
        accelerator : str | None
            The accelerator whose default stream participates, or None.
        samplers : Sequence[str]
            The sampler streams the run declares.

        Returns
        -------
        SeedPolicy
            The policy rooted at the derived seed.
        """
        return SeedPolicy(self._root, accelerator=accelerator, samplers=samplers)

    def data_seed(self, path: str, /) -> int:
        """Return the seed this point draws a derived data seed as.

        Each point of an experiment has its own run seed, so a corpus or a
        lexicon declared with a derived seed is drawn afresh at every point,
        including points that differ only in their sweep seed.

        Parameters
        ----------
        path : str
            The dotted path of the seed in the run document.
        """
        return data_seed(self._root, path)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another derivation has the same experiment and point."""
        if type(other) is not PointSeed:
            return NotImplemented

        return (
            self._experiment,
            self._point_id,
            self._seed,
        ) == (
            other._experiment,
            other._point_id,
            other._seed,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the experiment root and the point."""
        return hash((PointSeed, self._experiment, self._point_id, self._seed))

    @override
    def __repr__(self) -> str:
        """Return the call that rebuilds this derivation."""
        return f"{self._experiment!r}.point({self._point_id!r}, {self._seed!r})"


def _stream_names(family: str, names: Sequence[str], /) -> tuple[str, ...]:
    """Return distinct stream names in canonical order."""
    checked = tuple(ensure_name(f"{family} stream", name) for name in names)

    if len(checked) != len(set(checked)):
        msg = f"{family} streams repeat a name: {checked!r}"
        raise ValueError(msg)

    return tuple(sorted(checked))


def _require_declared(family: str, name: str, declared: tuple[str, ...], /) -> None:
    """Refuse a stream name outside one declared family."""
    checked = ensure_name(f"{family} stream", name)

    if checked not in declared:
        msg = f"undeclared {family} stream {checked!r}; declared: {declared!r}"
        raise KeyError(msg)


@final
class _WorkerInitializer:
    """Seeds Python and torch for one loader worker family."""

    __slots__ = ("_name", "_policy")

    def __init__(self, policy: SeedPolicy, name: str) -> None:
        object.__setattr__(self, "_policy", policy)
        object.__setattr__(self, "_name", name)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("worker initializer")

    def __call__(self, worker: int, /) -> None:
        """Seed one worker's host-library streams.

        Parameters
        ----------
        worker : int
            The worker position assigned by the loader.
        """
        torch = _torch_module()
        seed = self._policy.worker_seed(self._name, worker)
        random.seed(seed)
        torch.random.default_generator.manual_seed(seed)

    @override
    def __reduce__(self) -> _WorkerReduction:
        """Return the reconstruction call."""
        return (_restore_worker_initializer, (self._policy, self._name))


@final
class _TorchAcceleratorRandom:
    """Random-state access through torch for one accelerator device."""

    __slots__ = ("_family", "_name")

    def __init__(self, name: str) -> None:
        family = name.partition(":")[0]

        if family not in {"cuda", "mps", "xpu"}:
            msg = f"device {name!r} has no accelerator random-state backend"
            raise ValueError(msg)

        self._name = name
        self._family = family

    @property
    def name(self) -> str:
        """Return the device declaration this backend controls."""
        return self._name

    def seed(self, seed: int, /) -> None:
        """Seed the declared accelerator's default stream.

        Parameters
        ----------
        seed : int
            The nonnegative seed assigned to the accelerator.
        """
        torch = _torch_module()
        generator = torch.Generator(device=self._name)
        generator.manual_seed(seed)
        self.restore(_tensor_bytes(generator.get_state()))

    def snapshot(self) -> bytes:
        """Return the declared accelerator's current random state."""
        _torch_module()
        module = import_module(f"torch.{self._family}")
        state = (
            module.get_rng_state()
            if self._family == "mps"
            else module.get_rng_state(self._name)
        )

        return _tensor_bytes(state)

    def validate(self, state: bytes, /) -> None:
        """Refuse bytes that torch cannot install as generator state.

        Parameters
        ----------
        state : bytes
            A candidate accelerator state.
        """
        torch = _torch_module()
        _validate_torch_state(_state_tensor(state, torch), torch, device=self._name)

    def restore(self, state: bytes, /) -> None:
        """Restore the declared accelerator's random state.

        Parameters
        ----------
        state : bytes
            A state returned by :meth:`snapshot`.
        """
        torch = _torch_module()
        module = import_module(f"torch.{self._family}")
        restored = _state_tensor(state, torch)

        if self._family == "mps":
            module.set_rng_state(restored)
        else:
            module.set_rng_state(restored, self._name)


@final
class RandomStreams:
    """Python, torch and loader random streams for one seed policy.

    Parameters
    ----------
    policy : SeedPolicy
        The immutable policy governing these advancing streams.
    accelerator : AcceleratorRandom | None
        An injected backend for the declared accelerator, or None to resolve it
        through torch when the policy names one.

    Raises
    ------
    ValueError
        If the injected backend does not control the declared accelerator.
    """

    __slots__ = ("_accelerator", "_generators", "_policy", "_retry")

    def __init__(
        self,
        policy: SeedPolicy,
        *,
        accelerator: AcceleratorRandom | None = None,
    ) -> None:
        if accelerator is not None and accelerator.name != policy.accelerator:
            msg = (
                f"accelerator backend {accelerator.name!r} does not match "
                f"seed policy {policy.accelerator!r}"
            )
            raise ValueError(msg)

        self._policy = policy
        self._accelerator = accelerator
        self._generators: dict[str, Generator] = {}
        self._retry = 0

    @property
    def policy(self) -> SeedPolicy:
        """Return the immutable policy governing these streams."""
        return self._policy

    @property
    def seed_policy(self) -> Mapping[str, JsonValue]:
        """Return canonical policy metadata for a manifest."""
        return self._policy.record

    def apply(self) -> None:
        """Seed every global and retained stream from the policy.

        Raises
        ------
        RuntimeError
            If torch is not installed.
        """
        torch = _torch_module()
        policy = self._effective_policy()
        accelerator = self._accelerator_backend()
        generators = {
            name: torch.Generator().manual_seed(policy.loader_seed(name))
            for name in policy.loaders
        }
        random.seed(derive_seed(policy.root, "global:python"))
        torch.random.default_generator.manual_seed(
            derive_seed(policy.root, "global:torch")
        )

        if accelerator is not None:
            accelerator.seed(derive_seed(policy.root, "global:accelerator"))

        self._generators = generators

    def reseed(self, attempt: int, /) -> None:
        """Restart every stream at one deterministic numerical-retry seed.

        The retry seed is derived beneath the manifest's declared root; the
        point identity and recorded seed policy therefore remain unchanged.
        """
        if attempt < 1:
            msg = f"a numerical retry counts from one, not {attempt}"
            raise ValueError(msg)

        self._retry = attempt
        self.apply()

    def _effective_policy(self) -> SeedPolicy:
        """Return the declared policy or its current deterministic retry child."""
        if self._retry == 0:
            return self._policy

        return self._policy.retried(self._retry)

    def sampler_seed(self, name: str, /) -> int:
        """Return the seed assigned to one declared sampler.

        Parameters
        ----------
        name : str
            The sampler stream name.
        """
        return self._effective_policy().sampler_seed(name)

    def loader_generator(self, name: str, /) -> Generator:
        """Return the retained generator assigned to one declared loader.

        Parameters
        ----------
        name : str
            The loader stream name.

        Raises
        ------
        RuntimeError
            If :meth:`apply` has not initialized the streams.
        KeyError
            If no loader with that name was declared.
        """
        self._effective_policy().loader_seed(name)

        try:
            return self._generators[name]

        except KeyError as error:
            msg = "random streams must be applied before a loader generator is read"
            raise RuntimeError(msg) from error

    def worker_initializer(self, name: str, /) -> Callable[[int], None]:
        """Return the initializer assigned to one declared worker family.

        Parameters
        ----------
        name : str
            The worker-family name.

        Raises
        ------
        KeyError
            If no worker family with that name was declared.
        """
        policy = self._effective_policy()
        policy.worker_seed(name, 0)

        return _WorkerInitializer(policy, name)

    def snapshot(self) -> Mapping[str, str]:
        """Return every advancing stream in a safe textual wire form.

        Raises
        ------
        RuntimeError
            If :meth:`apply` has not initialized every retained loader stream.
        """
        torch = _torch_module()

        if set(self._generators) != set(self._policy.loaders):
            msg = "random streams must be applied before they are snapshotted"
            raise RuntimeError(msg)

        streams = {
            "python": _encode_json(_python_payload(random.getstate())),
            "torch": _encode_bytes(_tensor_bytes(torch.random.get_rng_state())),
            **{
                f"loader:{name}": _encode_bytes(
                    _tensor_bytes(self._generators[name].get_state())
                )
                for name in self._policy.loaders
            },
        }
        accelerator = self._accelerator_backend()

        if accelerator is not None:
            streams[f"accelerator:{accelerator.name}"] = _encode_bytes(
                accelerator.snapshot()
            )

        return MappingProxyType(streams)

    def validate(self, streams: Mapping[str, str], /) -> None:
        """Refuse a snapshot that cannot be restored without changing state.

        Parameters
        ----------
        streams : Mapping[str, str]
            The encoded state of every global and retained stream.

        Raises
        ------
        ValueError
            If the snapshot is incomplete, has an unknown stream or contains
            malformed state. No stream is changed in this case.
        RuntimeError
            If :meth:`apply` has not initialized every retained loader stream.
        """
        self._prepare_restore(streams)

    def prepare_restore(self, streams: Mapping[str, str], /) -> Callable[[], None]:
        """Validate a snapshot and return its non-validating restore operation.

        Parameters
        ----------
        streams : Mapping[str, str]
            The encoded state of every global and retained stream.
        """
        prepared = self._prepare_restore(streams)

        def restore() -> None:
            self._restore_prepared(prepared)

        return restore

    def restore(self, streams: Mapping[str, str], /) -> None:
        """Restore a complete snapshot after validating every stream.

        Parameters
        ----------
        streams : Mapping[str, str]
            The encoded state of every global and retained stream.

        Raises
        ------
        ValueError
            If the snapshot is incomplete, has an unknown stream or contains
            malformed state. No stream is changed in this case.
        RuntimeError
            If :meth:`apply` has not initialized every retained loader stream.
        """
        self.prepare_restore(streams)()

    def _restore_prepared(self, prepared: _PreparedRandomState, /) -> None:
        """Apply state returned by :meth:`_prepare_restore`."""
        torch = _torch_module()
        (
            python_state,
            torch_state,
            loader_states,
            accelerator,
            accelerator_state,
        ) = prepared

        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)

        for name, state in loader_states.items():
            self._generators[name].set_state(state)

        if accelerator is not None and accelerator_state is not None:
            accelerator.restore(accelerator_state)

    def _prepare_restore(self, streams: Mapping[str, str], /) -> _PreparedRandomState:
        """Return a fully decoded snapshot without changing a live stream."""
        torch = _torch_module()

        if set(self._generators) != set(self._policy.loaders):
            msg = "random streams must be applied before they are restored"
            raise RuntimeError(msg)

        accelerator = self._accelerator_backend()
        expected = {
            "python",
            "torch",
            *(f"loader:{name}" for name in self._policy.loaders),
        }

        if accelerator is not None:
            expected.add(f"accelerator:{accelerator.name}")

        if set(streams) != expected:
            missing = sorted(expected - set(streams))
            extra = sorted(set(streams) - expected)
            msg = (
                f"random-stream state has missing keys {missing} and extra keys {extra}"
            )
            raise ValueError(msg)

        if any(not isinstance(value, str) for value in streams.values()):
            msg = "random-stream state values must be encoded strings"
            raise ValueError(msg)

        python_state = _python_state(_decode_json(streams["python"], "python"))
        torch_state = _state_tensor(_decode_bytes(streams["torch"], "torch"), torch)
        loader_states = {
            name: _state_tensor(
                _decode_bytes(streams[f"loader:{name}"], f"loader:{name}"), torch
            )
            for name in self._policy.loaders
        }
        accelerator_state = (
            None
            if accelerator is None
            else _decode_bytes(
                streams[f"accelerator:{accelerator.name}"],
                f"accelerator:{accelerator.name}",
            )
        )

        _validate_python_state(python_state)
        _validate_torch_state(torch_state, torch)

        for state in loader_states.values():
            _validate_torch_state(state, torch)

        if accelerator is not None and accelerator_state is not None:
            accelerator.validate(accelerator_state)

        return (
            python_state,
            torch_state,
            loader_states,
            accelerator,
            accelerator_state,
        )

    def _accelerator_backend(self) -> AcceleratorRandom | None:
        """Return the injected or torch-backed declared accelerator."""
        declared = self._policy.accelerator

        if declared is None:
            return None

        if self._accelerator is None:
            self._accelerator = _TorchAcceleratorRandom(declared)

        return self._accelerator


def _torch_module() -> ModuleType:
    """Return torch at the policy application call site."""
    try:
        return import_module("torch")

    except ModuleNotFoundError as error:
        msg = "applying a seed policy requires the lofi[nn] capability"
        raise RuntimeError(msg) from error


def _encode_bytes(value: bytes, /) -> str:
    """Return bytes in a base64 wire form."""
    return base64.b64encode(value).decode("ascii")


def _decode_bytes(value: str, stream: str, /) -> bytes:
    """Return bytes decoded from one named base64 stream."""
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)

    except (UnicodeEncodeError, ValueError) as error:
        msg = f"random stream {stream!r} is not valid base64"
        raise ValueError(msg) from error

    if not decoded:
        msg = f"random stream {stream!r} has empty state"
        raise ValueError(msg)

    return decoded


def _encode_json(value: JsonValue, /) -> str:
    """Return JSON data as base64 text."""
    document = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()

    return _encode_bytes(document)


def _decode_json(value: str, stream: str, /) -> JsonValue:
    """Return JSON data decoded from one named base64 stream."""
    try:
        return cast("JsonValue", json.loads(_decode_bytes(value, stream)))

    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = f"random stream {stream!r} does not contain valid JSON"
        raise ValueError(msg) from error


def _python_payload(state: _PythonState, /) -> JsonValue:
    """Return Python random state as explicit JSON data."""
    version, inner, gaussian = state

    return {"version": version, "state": list(inner), "gaussian": gaussian}


def _python_state(value: JsonValue, /) -> _PythonState:
    """Return validated Python random state from JSON data."""
    if not isinstance(value, dict) or set(value) != {"version", "state", "gaussian"}:
        msg = "Python random state has the wrong fields"
        raise ValueError(msg)

    version = value["version"]
    state = value["state"]
    gaussian = value["gaussian"]

    if type(version) is not int:
        msg = "Python random-state version is not an integer"
        raise ValueError(msg)

    if not isinstance(state, list) or any(type(member) is not int for member in state):
        msg = "Python random state is not a list of integers"
        raise ValueError(msg)

    if gaussian is not None and not _is_finite_number(gaussian):
        msg = "Python Gaussian cache is not a finite number or null"
        raise ValueError(msg)

    return (
        version,
        tuple(cast("list[int]", state)),
        (None if gaussian is None else float(gaussian)),
    )


def _validate_python_state(state: _PythonState, /) -> None:
    """Refuse Python state without mutating the global generator."""
    try:
        random.Random().setstate(state)

    except (TypeError, ValueError) as error:
        msg = "Python random state is invalid"
        raise ValueError(msg) from error


def _is_finite_number(value: JsonValue, /) -> TypeIs[int | float]:
    """Return whether a JSON value is an exact finite integer or float."""
    if type(value) is int:
        try:
            float(value)

        except OverflowError:
            return False

        return True

    if type(value) is float:
        return math.isfinite(value)

    return False


def _tensor_bytes(state: Tensor, /) -> bytes:
    """Return a torch random-state tensor as bytes."""
    return bytes(state.cpu().tolist())


def _state_tensor(state: bytes, torch: ModuleType, /) -> Tensor:
    """Return torch random-state bytes as a host byte tensor."""
    return torch.tensor(list(state), dtype=torch.uint8)


def _validate_torch_state(
    state: Tensor,
    torch: ModuleType,
    /,
    *,
    device: str | None = None,
) -> None:
    """Refuse torch state without mutating a default generator."""
    try:
        generator = (
            torch.Generator() if device is None else torch.Generator(device=device)
        )
        generator.set_state(state)

    except (RuntimeError, TypeError, ValueError) as error:
        msg = "torch random state is invalid"
        raise ValueError(msg) from error


def _restore_seed_policy(
    root: int,
    accelerator: str | None,
    samplers: tuple[str, ...],
    loaders: tuple[str, ...],
    workers: tuple[str, ...],
) -> SeedPolicy:
    """Reconstruct an immutable seed policy."""
    return SeedPolicy(
        root,
        accelerator=accelerator,
        samplers=samplers,
        loaders=loaders,
        workers=workers,
    )


def _restore_worker_initializer(policy: SeedPolicy, name: str) -> _WorkerInitializer:
    """Reconstruct a loader worker initializer."""
    return _WorkerInitializer(policy, name)
