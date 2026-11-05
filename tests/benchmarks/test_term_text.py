from itertools import product
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
from tests.support.term_reference import infix_syntax

from lofi.terms import ConstructorSyntax, Operator, Signature, Term, TermStore

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


class CategoryText(NamedTuple):
    """A syntax, signature, hot store, and canonical inventory."""

    signature: Signature
    syntax: ConstructorSyntax
    hot_store: TermStore
    texts: tuple[str, ...]


class ParsedInventory(NamedTuple):
    """A store paired with the terms parsed into it."""

    store: TermStore
    terms: tuple[Term, ...]


CATEGORY_COUNT: Final = 2_485
PRIMITIVES: Final = (
    "s",
    "np",
    "n",
    "pp",
    "conj",
    ".",
    ",",
    ":",
    ";",
    "lrb",
    "rrb",
)
FEATURES: Final = {
    "s": (
        "dcl",
        "wq",
        "q",
        "qem",
        "em",
        "bem",
        "b",
        "frg",
        "for",
        "intj",
        "inv",
        "to",
        "pss",
        "pt",
        "ng",
        "adj",
        "as",
        "poss",
        "asup",
    ),
    "n": ("num",),
    "np": ("expl", "thr", "nb", "conj"),
}


def _signature() -> Signature:
    """Build the open category declaration."""
    primitives = tuple(Operator(name, "Cat") for name in PRIMITIVES)

    return Signature(
        "open-category-benchmark",
        ("Cat",),
        (
            *primitives,
            Operator("fwd", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
            Operator("bwd", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
        ),
        features=FEATURES,
    )


def _parse_hot(fixture: CategoryText) -> tuple[Term, ...]:
    """Parse the inventory into a preinterned store."""
    return tuple(
        fixture.syntax.parse(fixture.hot_store, text) for text in fixture.texts
    )


def _parse_cold(fixture: CategoryText) -> ParsedInventory:
    """Parse the inventory into a fresh store."""
    store = TermStore(fixture.signature)
    terms = tuple(fixture.syntax.parse(store, text) for text in fixture.texts)

    return ParsedInventory(store, terms)


def _record(
    benchmark: BenchmarkFixture,
    fixture: CategoryText,
    *,
    mode: str,
) -> None:
    """Record inventory size and throughput."""
    measured = benchmark.stats

    if measured is None:
        pytest.fail("category parsing ran but pytest-benchmark timed nothing")

    elapsed = measured.stats.median
    benchmark.extra_info["mode"] = mode
    benchmark.extra_info["categories"] = len(fixture.texts)
    benchmark.extra_info["characters"] = sum(map(len, fixture.texts))
    benchmark.extra_info["median_seconds"] = elapsed
    benchmark.extra_info["median_categories_per_second"] = len(fixture.texts) / elapsed


@pytest.fixture(scope="module")
def category_text() -> CategoryText:
    """Build the open 2,485-category inventory."""
    signature = _signature()
    store = TermStore(signature)
    syntax = infix_syntax(signature, ("/", "fwd"), ("\\", "bwd"))
    atoms = tuple(store.op(op) for op in signature.by_arity(0))
    inventory = (
        *atoms,
        *(
            store.op(op, left, right)
            for op in signature.by_arity(2)
            for left, right in product(atoms, repeat=2)
        ),
    )
    texts = tuple(syntax.render(store, term) for term in inventory)

    if len(texts) != CATEGORY_COUNT:
        pytest.fail(f"the open inventory has {len(texts)} categories")

    for text in texts:
        syntax.parse(store, text)

    return CategoryText(signature, syntax, store, texts)


def test_cold_ccg_inventory_parse_round_trips_every_category(
    category_text: CategoryText,
) -> None:
    parsed = _parse_cold(category_text)

    assert len(parsed.terms) == CATEGORY_COUNT
    assert all(
        category_text.syntax.render(parsed.store, term) == text
        for term, text in zip(parsed.terms, category_text.texts, strict=True)
    )


def test_cold_ccg_inventory_parse_throughput(
    benchmark: BenchmarkFixture,
    category_text: CategoryText,
) -> None:
    benchmark.pedantic(
        _parse_cold,
        args=(category_text,),
        rounds=5,
        warmup_rounds=1,
        iterations=1,
    )
    _record(benchmark, category_text, mode="cold")


def test_hot_ccg_inventory_parse_round_trips_every_category(
    category_text: CategoryText,
) -> None:
    parsed = _parse_hot(category_text)

    assert len(parsed) == CATEGORY_COUNT
    assert all(
        category_text.syntax.render(category_text.hot_store, term) == text
        for term, text in zip(parsed, category_text.texts, strict=True)
    )


def test_hot_ccg_inventory_parse_throughput(
    benchmark: BenchmarkFixture,
    category_text: CategoryText,
) -> None:
    benchmark(_parse_hot, category_text)
    _record(benchmark, category_text, mode="hot")
