from random import Random
from typing import TYPE_CHECKING, Literal

from tests.language import language

from lofi.lf.models import (
    CRISP,
    EXTREME,
    PRODUCT,
    RESIDUUM,
    SAMPLED,
    Element,
    FunctionValue,
    ModelValue,
    SampledSettings,
    SampledStructureSource,
    sampling,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest


def source(*, resample: Literal["once", "per-step"]) -> SampledStructureSource:
    declared = language(("p", "q"))
    return SampledStructureSource(
        declared,
        PRODUCT.build(
            implication=RESIDUUM,
            quantifier=EXTREME,
            equality=CRISP,
            margin=0.0,
        ),
        sampling=sampling("uniform", "independent", "table", "bernoulli"),
        structures=3,
        carriers={"e": 2, "v": 1},
        density=0.5,
        full_function_space_cap=8,
        henkin_pool=2,
        seed=11,
        resample=resample,
        theory_floor=1.0,
    )


def test_sampled_structures_are_reproducible_without_a_shared_random_stream() -> None:
    sampled = source(resample="once")
    first = sampled.structures()
    second = source(resample="once").structures()

    assert sampled.evidence_label == "conditional satisfaction"
    assert tuple(item.denotations for item in first) == tuple(
        item.denotations for item in second
    )
    assert tuple(structure.name for structure in first) == (
        "sampled-0-0",
        "sampled-0-1",
        "sampled-0-2",
    )


def test_per_step_resampling_advances_only_when_the_step_opens() -> None:
    sampled = source(resample="per-step")
    first = sampled.structures()

    assert sampled.structures() is first

    sampled.open_step()
    second = sampled.structures()

    assert second is not first
    assert tuple(structure.name for structure in second) == (
        "sampled-1-0",
        "sampled-1-1",
        "sampled-1-2",
    )


def test_per_step_resampling_replays_the_same_seeded_sequence() -> None:
    first = source(resample="per-step")
    second = source(resample="per-step")

    for _ in range(3):
        assert tuple(item.denotations for item in first.structures()) == tuple(
            item.denotations for item in second.structures()
        )
        first.open_step()
        second.open_step()


class LastAtomic:
    def sample(self, values: tuple[ModelValue, ...], random: Random, /) -> ModelValue:
        del random
        return values[-1]


class FixedStreams:
    def stream(self, seed: int, step: int, position: int, identifier: int, /) -> Random:
        del seed, step, position, identifier
        return Random(0)


class FalseFunctions:
    def sample(
        self, domain: tuple[ModelValue, ...], draw: Callable[[], ModelValue], /
    ) -> FunctionValue:
        del draw
        return FunctionValue(domain, (False,) * len(domain), (True,) * len(domain))


class ComplementTruth:
    def sample(self, random: Random, density: float, /) -> bool:
        return random.random() > density


LAST_ATOMIC = LastAtomic()
FIXED_STREAMS = FixedStreams()
FALSE_FUNCTIONS = FalseFunctions()
COMPLEMENT_TRUTH = ComplementTruth()


def test_sampled_structures_consume_every_declared_sampling_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    distribution = tmp_path / "sampling_choices-1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text("Name: sampling-choices\nVersion: 1.0\n")
    (distribution / "entry_points.txt").write_text(
        "\n".join(
            f"[lofi.{family}]\ncustom = tests.unit.lf.models.test__sampled:{name}\n"
            for family, name in (
                ("atomic_sampling", "LAST_ATOMIC"),
                ("denotation_streams", "FIXED_STREAMS"),
                ("function_sampling", "FALSE_FUNCTIONS"),
                ("truth_sampling", "COMPLEMENT_TRUTH"),
            )
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    declared = language(
        ("p", "individual", "property"), types={"individual": "e", "property": "e -> t"}
    )
    settings = SampledSettings(
        atomic_sampling="custom",
        denotation_streams="custom",
        function_sampling="custom",
        truth_sampling="custom",
        structures=1,
        carriers={"e": 2, "v": 1},
        density=0.5,
        full_function_space_cap=8,
        henkin_pool=2,
        seed=11,
        resample="once",
    )
    source = SAMPLED.build(
        settings,
        declared,
        PRODUCT.build(
            implication=RESIDUUM, quantifier=EXTREME, equality=CRISP, margin=0.0
        ),
        theory_floor=1.0,
    )
    (structure,) = source.structures()
    vocabulary = declared.store.signature.vocabulary(declared.term_sort)
    values = dict(zip(vocabulary.names, range(len(vocabulary.names)), strict=True))

    assert structure.denotation(values["p"]) is (Random(0).random() > 0.5)
    assert structure.denotation(values["individual"]) == Element("e", 1)
    assert structure.denotation(values["property"]) == FunctionValue(
        (Element("e", 0), Element("e", 1)), (False, False), (True, True)
    )
