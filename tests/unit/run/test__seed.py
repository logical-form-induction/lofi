import base64
import json
import os
import pickle
import random
import subprocess
import sys
from typing import TYPE_CHECKING, cast

import pytest
import torch

from lofi.run import (
    ExperimentSeed,
    RandomStreams,
    SeedPolicy,
    decode_seed_derivation,
    encode_seed_derivation,
)
from lofi.run import _seed as seed_module
from lofi.spec import data_seed
from lofi.train import RestorableRandomStreams, RngRecord

if TYPE_CHECKING:
    from types import ModuleType

    from lofi._types import JsonValue


class FakeAccelerator:
    __slots__ = ("_name", "_state", "_validations")

    def __init__(self, name: str) -> None:
        self._name = name
        self._state = 0
        self._validations = 0

    @property
    def name(self) -> str:
        return self._name

    def seed(self, seed: int, /) -> None:
        self._state = seed

    def snapshot(self) -> bytes:
        return self._state.to_bytes(8, "big")

    def validate(self, state: bytes, /) -> None:
        self._validations += 1

        if len(state) != 8:
            msg = "fake accelerator state must hold eight bytes"
            raise ValueError(msg)

    def restore(self, state: bytes, /) -> None:
        self._state = int.from_bytes(state, "big")

    def draw(self) -> int:
        self._state = (1_664_525 * self._state + 1_013_904_223) % (2**32)
        return self._state

    @property
    def validations(self) -> int:
        return self._validations


def policy(root: int) -> SeedPolicy:
    return SeedPolicy(
        root,
        accelerator="fake:0",
        samplers=("training", "validation"),
        loaders=("training", "validation"),
        workers=("training", "validation"),
    )


def built(root: int) -> tuple[RandomStreams, FakeAccelerator]:
    accelerator = FakeAccelerator("fake:0")
    streams = RandomStreams(policy(root), accelerator=accelerator)
    streams.apply()
    return streams, accelerator


def draw(
    streams: RandomStreams, accelerator: FakeAccelerator
) -> tuple[float, float, int, float, float, int, int]:
    return (
        random.random(),
        float(torch.rand(())),
        accelerator.draw(),
        float(torch.rand((), generator=streams.loader_generator("training"))),
        float(torch.rand((), generator=streams.loader_generator("validation"))),
        streams.sampler_seed("training"),
        streams.sampler_seed("validation"),
    )


def worker_draw(
    streams: RandomStreams, family: str, worker: int
) -> tuple[float, float]:
    streams.worker_initializer(family)(worker)

    return (
        random.random(),
        float(torch.rand(())),
    )


def test_policy_metadata_is_canonical_and_immutable() -> None:
    declared = policy(17)

    assert dict(declared.record) == {
        "root": 17,
        "accelerator": "fake:0",
        "samplers": ["training", "validation"],
        "loaders": ["training", "validation"],
        "workers": ["training", "validation"],
    }

    with pytest.raises(AttributeError, match="immutable"):
        declared.__setattr__("root", 19)

    with pytest.raises(AttributeError, match="immutable"):
        declared.__delattr__("_root")


def test_same_root_repeats_and_a_different_root_changes_every_stream() -> None:
    first, first_accelerator = built(11)
    first_draw = draw(first, first_accelerator)
    first_workers = (
        worker_draw(first, "training", 0),
        worker_draw(first, "validation", 0),
    )
    repeated, repeated_accelerator = built(11)
    repeated_draw = draw(repeated, repeated_accelerator)
    repeated_workers = (
        worker_draw(repeated, "training", 0),
        worker_draw(repeated, "validation", 0),
    )
    changed, changed_accelerator = built(12)
    changed_draw = draw(changed, changed_accelerator)
    changed_workers = (
        worker_draw(changed, "training", 0),
        worker_draw(changed, "validation", 0),
    )

    assert repeated_draw == first_draw
    assert repeated_workers == first_workers
    assert all(
        left != right for left, right in zip(first_draw, changed_draw, strict=True)
    )
    assert all(
        left != right
        for first_family, changed_family in zip(
            first_workers, changed_workers, strict=True
        )
        for left, right in zip(first_family, changed_family, strict=True)
    )
    assert len({policy(11).sampler_seed(name) for name in policy(11).samplers}) == 2
    assert len({policy(11).loader_seed(name) for name in policy(11).loaders}) == 2


def test_a_numerical_retry_reseeds_every_stream_beneath_the_declared_root() -> None:
    first, first_accelerator = built(11)
    first.reseed(2)
    first_draw = draw(first, first_accelerator)
    repeated, repeated_accelerator = built(11)
    repeated.reseed(2)
    repeated_draw = draw(repeated, repeated_accelerator)
    changed, changed_accelerator = built(11)
    changed.reseed(3)

    assert repeated_draw == first_draw
    assert all(
        left != right
        for left, right in zip(
            first_draw, draw(changed, changed_accelerator), strict=True
        )
    )
    assert first.seed_policy["root"] == 11

    with pytest.raises(ValueError, match="counts from one"):
        first.reseed(0)


def test_snapshot_restore_reproduces_every_next_value_bitwise() -> None:
    streams, accelerator = built(23)
    draw(streams, accelerator)
    snapshot = streams.snapshot()
    assert set(snapshot) == {
        "python",
        "torch",
        "accelerator:fake:0",
        "loader:training",
        "loader:validation",
    }
    expected = draw(streams, accelerator)
    draw(streams, accelerator)

    streams.restore(snapshot)

    assert draw(streams, accelerator) == expected
    assert accelerator.validations == 1
    assert isinstance(streams, RestorableRandomStreams)
    record = RngRecord(streams=dict(snapshot))
    assert RngRecord.model_validate_json(record.model_dump_json()) == record


def test_a_numpy_global_state_record_is_refused_not_ignored() -> None:
    streams, _accelerator = built(23)
    snapshot = dict(streams.snapshot())
    snapshot["numpy"] = snapshot["python"]

    with pytest.raises(ValueError, match=r"extra keys \['numpy'\]"):
        streams.restore(snapshot)


@pytest.mark.parametrize("damage", ["malformed", "missing", "corrupt-torch"])
def test_malformed_state_is_refused_before_any_stream_changes(damage: str) -> None:
    streams, _accelerator = built(29)
    before = dict(streams.snapshot())
    malformed = dict(before)

    if damage == "malformed":
        malformed["python"] = "not base64"
    elif damage == "missing":
        del malformed["python"]
    else:
        malformed["torch"] = base64.b64encode(b"x").decode("ascii")

    with pytest.raises(ValueError, match=r"random.stream|torch random"):
        streams.validate(malformed)

    assert dict(streams.snapshot()) == before

    with pytest.raises(ValueError, match=r"random.stream|torch random"):
        streams.restore(malformed)

    assert dict(streams.snapshot()) == before


@pytest.mark.parametrize(
    ("stream", "field", "value"),
    [
        ("python", "gaussian", True),
        ("python", "gaussian", float("nan")),
        ("python", "gaussian", float("inf")),
    ],
)
def test_nonfinite_or_boolean_gaussian_state_is_refused_before_any_mutation(
    stream: str,
    field: str,
    *,
    value: bool | float,
) -> None:
    streams, _accelerator = built(29)
    before = dict(streams.snapshot())
    malformed = dict(before)
    payload = cast(
        "dict[str, JsonValue]",
        json.loads(base64.b64decode(malformed[stream])),
    )
    payload[field] = value
    malformed[stream] = base64.b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode("ascii")

    with pytest.raises(ValueError, match="Gaussian cache"):
        streams.restore(malformed)

    assert dict(streams.snapshot()) == before


def test_worker_families_and_positions_are_distinct() -> None:
    streams, _accelerator = built(31)
    draws = {
        worker_draw(streams, "training", 0),
        worker_draw(streams, "training", 1),
        worker_draw(streams, "validation", 0),
    }

    assert len(draws) == 3


def test_policy_and_worker_initializer_survive_worker_process_serialization() -> None:
    declared = policy(31)
    initializer = RandomStreams(declared, accelerator=FakeAccelerator("fake:0"))

    assert pickle.loads(pickle.dumps(declared)) == declared
    restored = pickle.loads(pickle.dumps(initializer.worker_initializer("training")))
    restored(2)
    observed = (random.random(), float(torch.rand(())))
    initializer.worker_initializer("training")(2)

    assert (random.random(), float(torch.rand(()))) == observed


def test_undeclared_named_streams_are_refused() -> None:
    streams, _accelerator = built(37)

    with pytest.raises(KeyError, match="undeclared sampler"):
        streams.sampler_seed("test")

    with pytest.raises(KeyError, match="undeclared loader"):
        streams.loader_generator("test")

    with pytest.raises(KeyError, match="undeclared worker"):
        streams.worker_initializer("test")


def test_import_is_lazy_and_missing_neural_dependencies_fail_at_application() -> None:
    source = """
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] in {'numpy', 'torch'}:
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, Blocker())
import lofi.run
print('imported')
try:
    lofi.run.RandomStreams(lofi.run.SeedPolicy(3)).apply()
except RuntimeError as error:
    print(error)
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert result.stdout.splitlines() == [
        "imported",
        "applying a seed policy requires the lofi[nn] capability",
    ]


def test_applying_and_restoring_streams_never_resolves_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    resolve = seed_module.import_module

    def recording(name: str) -> ModuleType:
        imported.append(name)

        return resolve(name)

    monkeypatch.setattr(seed_module, "import_module", recording)
    streams = RandomStreams(SeedPolicy(3, loaders=("fit",), workers=("fit",)))
    streams.apply()
    snapshot = streams.snapshot()
    streams.worker_initializer("fit")(0)
    streams.restore(snapshot)

    assert imported
    assert all(not name.startswith("numpy") for name in imported)


def test_experiments_with_different_roots_derive_different_point_seeds() -> None:
    points = [(f"{index:016x}", index % 3) for index in range(32)]
    first = ExperimentSeed(7)
    second = ExperimentSeed(8)

    derived = [first.point(point_id, seed).root for point_id, seed in points]
    other = [second.point(point_id, seed).root for point_id, seed in points]

    assert all(left != right for left, right in zip(derived, other, strict=True))
    assert len(set(derived)) == len(points)


def test_an_identical_point_reproduces_its_seed_under_the_same_root() -> None:
    derived = ExperimentSeed(7).point("0123456789abcdef", 2)
    again = ExperimentSeed(7).point("0123456789abcdef", 2)

    assert derived == again
    assert derived.root == again.root


def test_a_point_seed_records_its_experiment_root_and_derivation() -> None:
    derived = ExperimentSeed(7).point("0123456789abcdef", 2)

    record = encode_seed_derivation(derived)

    assert record.model_dump() == {
        "experiment_root": 7,
        "point_id": "0123456789abcdef",
        "seed": 2,
        "root": derived.root,
    }
    assert decode_seed_derivation(record) == derived
    assert not hasattr(derived, "record")
    assert not hasattr(derived, "derivation_record")
    assert not hasattr(type(derived), "from_derivation")


def test_a_point_policy_takes_its_root_from_the_derivation() -> None:
    derived = ExperimentSeed(7).point("0123456789abcdef", 2)

    policy = derived.policy(accelerator=None, samplers=("fit",))

    assert policy == SeedPolicy(derived.root, samplers=("fit",))


def test_a_point_seed_is_stable_across_hash_seeds() -> None:
    script = (
        "from lofi.run import ExperimentSeed;"
        "print(ExperimentSeed(11).point('0123456789abcdef', 4).root)"
    )
    seen = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            encoding="utf-8",
        ).stdout
        for hash_seed in ("0", "1")
    }

    assert seen == {f"{ExperimentSeed(11).point('0123456789abcdef', 4).root}\n"}


def test_an_experiment_seed_records_its_root() -> None:
    root = ExperimentSeed(7)

    assert root.root == 7
    assert not hasattr(root, "derivation")


def test_a_point_derives_its_own_data_seed_for_each_path() -> None:
    experiment = ExperimentSeed(5)
    first = experiment.point("0123456789abcdef", 0)
    second = experiment.point("fedcba9876543210", 1)

    assert first.data_seed("data.datasets.train.dataset.seed") == data_seed(
        first.root, "data.datasets.train.dataset.seed"
    )
    assert first.data_seed("data.datasets.train.dataset.seed") == experiment.point(
        "0123456789abcdef", 0
    ).data_seed("data.datasets.train.dataset.seed")
    assert first.data_seed("data.datasets.train.dataset.seed") != second.data_seed(
        "data.datasets.train.dataset.seed"
    )
    assert first.data_seed("data.datasets.train.dataset.seed") != first.data_seed(
        "data.datasets.train.dataset.grammar.lexicon.seed"
    )
    assert first.data_seed("data.datasets.train.dataset.seed") != ExperimentSeed(
        6
    ).point("0123456789abcdef", 0).data_seed("data.datasets.train.dataset.seed")
