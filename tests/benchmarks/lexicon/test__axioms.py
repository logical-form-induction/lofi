from typing import TYPE_CHECKING, Final

import didactic.api as dx
import pytest

from lofi.lexicon import BeamSource, Entry

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_benchmark.fixture import BenchmarkFixture

POSITIONS: Final = 40
WIDTH: Final = 8
TOKENS: Final = tuple(f"token{position}" for position in range(POSITIONS))


def _beam(position: int) -> tuple[Entry[str], ...]:
    """Return one position's candidates, worst first so the source must rank them.

    Every candidate bears the token at ``position``, so every one of them is
    admitted, and no two share a weight anywhere in the fixture.

    Parameters
    ----------
    position : int
        Zero-based token position.

    Returns
    -------
    tuple[Entry[str], ...]
        The eight candidates, in ascending weight order.
    """
    return tuple(
        Entry(TOKENS[position], f"C{rank}", None, -(position * WIDTH + rank + 1) / 100)
        for rank in reversed(range(WIDTH))
    )


SOURCE: Final = BeamSource[str](_beam(position) for position in range(POSITIONS))


def _constructions(call: Callable[[], object]) -> int:
    """Return how many lexical values and boundary objects a call builds.

    Parameters
    ----------
    call : Callable[[], object]
        The work to run under the counter.

    Returns
    -------
    int
        The number of ``Entry`` and ``didactic.api.Model`` constructions.
    """
    counted = 0
    lexical = Entry.__init__
    boundary = dx.Model.__init__

    def counting_entry(
        entry: Entry[str],
        form: str,
        category: str,
        meaning: str | None = None,
        weight: float = 0.0,
    ) -> None:
        nonlocal counted
        counted += 1

        lexical(entry, form, category, meaning, weight)

    def counting_model(model: dx.Model, /, **fields: int) -> None:
        nonlocal counted
        counted += 1

        boundary(model, **fields)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Entry, "__init__", counting_entry)
        patch.setattr(dx.Model, "__init__", counting_model)
        call()

    return counted


class Coordinate(dx.Model):
    span: int = dx.field(default=0)


def test_the_construction_counter_sees_one_construction_of_each_kind() -> None:
    def lexical() -> Entry[str]:
        return Entry("token0", "C0", None, -0.5)

    def boundary() -> Coordinate:
        return Coordinate(span=1)

    assert _constructions(lexical) == 1
    assert _constructions(boundary) == 1
    assert _constructions(lambda: (lexical(), boundary())) == 2


def test_the_construction_counter_leaves_its_call_and_the_constructors_alone() -> None:
    lexical = Entry.__init__
    boundary = dx.Model.__init__
    entries: list[Entry[str]] = []
    coordinates: list[Coordinate] = []

    def keep_one_entry() -> None:
        entries.append(Entry("token3", "C5", None, -0.25))

    def keep_one_coordinate() -> None:
        coordinates.append(Coordinate(span=7))

    assert _constructions(keep_one_entry) == 1
    assert _constructions(keep_one_coordinate) == 1
    assert _constructions(lambda: None) == 0

    assert Entry.__init__ is lexical
    assert dx.Model.__init__ is boundary

    assert [(entry.form, entry.category, entry.meaning) for entry in entries] == [
        ("token3", "C5", None)
    ]
    assert [entry.weight for entry in entries] == [-0.25]
    assert [coordinate.span for coordinate in coordinates] == [7]


def test_the_fixture_offers_eight_matching_candidates_at_every_position() -> None:
    assert SOURCE.positions == POSITIONS
    assert SOURCE.widths == (WIDTH,) * POSITIONS

    weights = [entry.weight for beam in SOURCE.beams for entry in beam]

    assert len(set(weights)) == POSITIONS * WIDTH
    assert all(weight < 0 for weight in weights)

    for position, beam in enumerate(SOURCE.beams):
        assert [entry.form for entry in beam] == [TOKENS[position]] * WIDTH
        assert list(beam) == sorted(beam, key=lambda entry: -entry.weight)


@pytest.mark.parametrize(("length", "count"), [(20, 160), (40, 320)])
def test_enumerating_the_beam_source_emits_the_fixture_entries(
    benchmark: BenchmarkFixture, length: int, count: int
) -> None:
    target = SOURCE.axioms
    arguments = (TOKENS[:length],)

    benchmark.pedantic(target, args=arguments, rounds=50, warmup_rounds=3)
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the enumeration ran but pytest-benchmark timed nothing")

    median_us = measured.stats.median * 1e6
    benchmark.extra_info["length"] = length
    benchmark.extra_info["beam_width"] = WIDTH
    benchmark.extra_info["count"] = count
    benchmark.extra_info["median_us"] = median_us
    benchmark.extra_info["ns_per_axiom"] = median_us * 1e3 / count
    benchmark.extra_info["baseline"] = "initial baseline, 2026-09-30"


@pytest.mark.parametrize(("length", "count"), [(20, 160), (40, 320)])
def test_enumerating_the_beam_source_reuses_its_fixture_entries(
    length: int, count: int
) -> None:
    target = SOURCE.axioms
    arguments = (TOKENS[:length],)
    built = _constructions(lambda: target(*arguments))
    emitted = target(*arguments)
    seen: dict[int, list[Entry[str]]] = {}

    assert built == 0
    assert len(emitted) == count

    for position, entry, weight in emitted:
        assert weight == entry.weight
        seen.setdefault(position, []).append(entry)

    assert list(seen) == list(range(length))

    for position, entries in seen.items():
        beam = SOURCE.beams[position]

        assert len(entries) == WIDTH
        assert all(
            entry is original for entry, original in zip(entries, beam, strict=True)
        )
        assert all(entry.form == TOKENS[position] for entry in entries)
