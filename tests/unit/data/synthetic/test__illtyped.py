"""Tests for the ill-typed logical-form values."""

from typing import TYPE_CHECKING, Final, cast

import pytest

from lofi._types import ItemId
from lofi.data.synthetic import (
    FormTypingExample,
    IllTyped,
    TypingConflict,
    TypingFailure,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.data.synthetic import FormTypingLabel

DIGEST: Final = "c" * 64
ITEM: Final = ItemId("grammar:0:3")
SOURCE: Final = "sleep(mary)"
REFUSED: Final = "mary(sleep)"
CONFLICT: Final = TypingConflict("unification", "e", '(-> "e" "t")', (0,))
FAILURE: Final = TypingFailure((CONFLICT,), language_digest=DIGEST)


def test_a_form_typing_example_carries_the_form_and_its_observed_label() -> None:
    example = FormTypingExample(ITEM, SOURCE, "ill-typed", "well-typed")

    assert example.item_id == ITEM
    assert example.text == SOURCE
    assert example.true_label == "ill-typed"
    assert example.observed_label == "well-typed"
    assert example.flipped
    assert example == FormTypingExample(ITEM, SOURCE, "ill-typed", "well-typed")
    assert hash(example) == hash(
        FormTypingExample(ITEM, SOURCE, "ill-typed", "well-typed")
    )
    assert repr(example) == (
        f"FormTypingExample({ITEM!r}, {SOURCE!r}, 'ill-typed', 'well-typed')"
    )


@pytest.mark.parametrize(
    ("build", "fault"),
    [
        (
            lambda: FormTypingExample(ITEM, "", "well-typed", "well-typed"),
            "logical form",
        ),
        (
            lambda: FormTypingExample(
                ITEM,
                SOURCE,
                cast("FormTypingLabel", "accepted"),
                "well-typed",
            ),
            "true label",
        ),
        (
            lambda: FormTypingExample(
                ITEM,
                SOURCE,
                "well-typed",
                cast("FormTypingLabel", "accepted"),
            ),
            "observed label",
        ),
    ],
)
def test_a_form_typing_example_refuses_an_empty_form_or_unknown_label(
    build: Callable[[], FormTypingExample], fault: str
) -> None:
    with pytest.raises(ValueError, match=fault):
        build()


def _record(
    kind: str = "argument-swap",
    source: str = SOURCE,
    text: str = REFUSED,
    certificate: TypingFailure = FAILURE,
    observed_label: FormTypingLabel = "ill-typed",
) -> IllTyped:
    return IllTyped(
        ITEM, kind, source, text, certificate, observed_label=observed_label
    )


def test_a_conflict_holds_the_kind_the_sides_and_the_node() -> None:
    assert CONFLICT.kind == "unification"
    assert CONFLICT.left == "e"
    assert CONFLICT.right == '(-> "e" "t")'
    assert CONFLICT.path == (0,)


def test_a_conflict_is_equal_on_its_kind_sides_and_node() -> None:
    assert TypingConflict("unification", "e", '(-> "e" "t")', (0,)) == CONFLICT
    assert hash(CONFLICT) == hash(
        TypingConflict("unification", "e", '(-> "e" "t")', (0,))
    )
    assert TypingConflict("unification", "e", '(-> "e" "t")', (1,)) != CONFLICT
    assert TypingConflict("infinite_type", "e", '(-> "e" "t")', (0,)) != CONFLICT
    assert TypingConflict("unification", '(-> "e" "t")', "e", (0,)) != CONFLICT
    assert CONFLICT.__eq__(CONFLICT.path) is NotImplemented
    assert repr(CONFLICT) == (
        "TypingConflict('unification', 'e', '(-> \"e\" \"t\")', (0,))"
    )


@pytest.mark.parametrize(
    ("build", "fault"),
    [
        (lambda: TypingConflict("unification", "", "t", ()), "left type is empty"),
        (lambda: TypingConflict("unification", "e", "", ()), "right type is empty"),
        (lambda: TypingConflict("unification", "e", "t", (-1,)), "path\\[0\\]"),
    ],
)
def test_a_conflict_refuses_an_empty_side_or_a_negative_node(
    build: Callable[[], TypingConflict], fault: str
) -> None:
    with pytest.raises(ValueError, match=fault):
        build()


def test_a_failure_holds_its_conflicts_their_kinds_and_the_language() -> None:
    other = TypingConflict("infinite_type", "e", '(-> "e" "t")', ())
    failure = TypingFailure((CONFLICT, other), language_digest=DIGEST)

    assert failure.conflicts == (CONFLICT, other)
    assert failure.kinds == ("unification", "infinite_type")
    assert failure.language_digest == DIGEST


def test_a_failure_is_equal_on_its_conflicts_and_language() -> None:
    assert TypingFailure((CONFLICT,), language_digest=DIGEST) == FAILURE
    assert hash(FAILURE) == hash(TypingFailure((CONFLICT,), language_digest=DIGEST))
    assert TypingFailure((CONFLICT,), language_digest="d" * 64) != FAILURE
    assert FAILURE.__eq__(FAILURE.conflicts) is NotImplemented
    assert repr(FAILURE) == (
        f"TypingFailure({(CONFLICT,)!r}, language_digest={DIGEST!r})"
    )


@pytest.mark.parametrize(
    ("build", "fault"),
    [
        (
            lambda: TypingFailure((), language_digest=DIGEST),
            "names no unsatisfied constraint",
        ),
        (
            lambda: TypingFailure((CONFLICT,), language_digest=""),
            "language digest is empty",
        ),
    ],
)
def test_a_failure_refuses_no_conflict_or_no_language(
    build: Callable[[], TypingFailure], fault: str
) -> None:
    with pytest.raises(ValueError, match=fault):
        build()


def test_a_record_holds_the_item_the_kind_both_forms_and_the_certificate() -> None:
    record = _record()

    assert record.item_id == ITEM
    assert record.kind == "argument-swap"
    assert record.source == SOURCE
    assert record.text == REFUSED
    assert record.certificate == FAILURE
    assert record.true_label == "ill-typed"
    assert record.observed_label == "ill-typed"
    assert not record.flipped

    contaminated = _record(observed_label="well-typed")

    assert contaminated.true_label == "ill-typed"
    assert contaminated.observed_label == "well-typed"
    assert contaminated.flipped


def test_a_record_is_equal_on_the_item_the_kind_the_forms_and_the_warrant() -> None:
    assert _record() == _record()
    assert hash(_record()) == hash(_record())
    assert _record() != _record(kind="sort-mismatch")
    assert _record() != _record(text="mary(mary)")
    assert _record() != _record(
        certificate=TypingFailure((CONFLICT,), language_digest="d" * 64)
    )
    assert _record().__eq__(FAILURE) is NotImplemented
    assert repr(_record()) == (
        f"IllTyped({ITEM!r}, 'argument-swap', {SOURCE!r}, {REFUSED!r}, "
        f"{FAILURE!r}, observed_label='ill-typed')"
    )


@pytest.mark.parametrize(
    ("build", "fault"),
    [
        (lambda: _record(kind=""), "ill-typing operator"),
        (lambda: _record(source=""), "source form"),
        (lambda: _record(text=""), "ill-typed form"),
        (lambda: _record(text=SOURCE), "is its own source"),
        (
            lambda: _record(observed_label=cast("FormTypingLabel", "unknown")),
            "unknown observed label",
        ),
    ],
)
def test_a_record_refuses_an_empty_kind_an_empty_form_or_no_corruption(
    build: Callable[[], IllTyped], fault: str
) -> None:
    with pytest.raises(ValueError, match=fault):
        build()
