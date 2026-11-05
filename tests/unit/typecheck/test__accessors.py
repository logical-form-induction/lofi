import copy
from typing import TYPE_CHECKING, Final, cast

import pytest

from lofi.typecheck import (
    FixedSpan,
    FormPattern,
    SpanAccessor,
    TokenPosition,
    WholeInput,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

TOKENS: Final = ("jo", "saw", "which", "thing", "and", "which", "thing")
"""The committed input every accessor below is read against."""

VERB: Final = (1, 2)
"""The span a caller writes today to clamp the matrix verb."""

PREDICATE: Final = (1, 7)
"""The span a caller writes today to clamp the matrix predicate."""

ROOT: Final = (0, 7)
"""The span a caller writes today to clamp the whole committed input."""


def test_the_token_accessor_covers_the_position_a_caller_writes_today() -> None:
    accessor = TokenPosition(VERB[0])

    assert accessor.position == VERB[0]
    assert accessor.spans(TOKENS) == (VERB,)


def test_the_fixed_accessor_covers_the_span_a_caller_writes_today() -> None:
    accessor = FixedSpan(*PREDICATE)

    assert (accessor.start, accessor.end) == PREDICATE
    assert accessor.spans(TOKENS) == (PREDICATE,)


def test_the_whole_input_accessor_covers_every_token() -> None:
    assert WholeInput().spans(TOKENS) == (ROOT,)
    assert WholeInput().spans(TOKENS[:3]) == ((0, 3),)


def test_each_accessor_reads_back_what_it_was_declared_over() -> None:
    for position in (VERB[0], 3, len(TOKENS) - 1):
        assert TokenPosition(position).position == position
        assert TokenPosition(position).spans(TOKENS) == ((position, position + 1),)

    for span in (VERB, PREDICATE, (3, 5)):
        accessor = FixedSpan(*span)

        assert (accessor.start, accessor.end) == span
        assert accessor.spans(TOKENS) == (span,)

    assert FormPattern(("which", None)).pattern == ("which", None)
    assert FormPattern(["which", "thing"]).pattern == ("which", "thing")
    assert FormPattern((None,)).pattern == (None,)


def test_a_token_position_past_the_input_is_refused() -> None:
    with pytest.raises(ValueError, match="reaches past the 7-token input"):
        TokenPosition(len(TOKENS)).spans(TOKENS)

    assert TokenPosition(len(TOKENS) - 1).spans(TOKENS) == (
        (len(TOKENS) - 1, len(TOKENS)),
    )


def test_a_fixed_span_past_the_input_is_refused() -> None:
    with pytest.raises(ValueError, match=r"the span \[5, 8\) reaches past"):
        FixedSpan(5, 8).spans(TOKENS)

    assert FixedSpan(5, 7).spans(TOKENS) == ((5, 7),)


def test_the_whole_of_an_empty_input_is_refused() -> None:
    with pytest.raises(ValueError, match="empty input covers no position"):
        WholeInput().spans(())


def test_a_fixed_span_covering_no_position_is_refused() -> None:
    with pytest.raises(ValueError, match=r"covers no position over \[2, 2\)"):
        FixedSpan(2, 2)

    with pytest.raises(ValueError, match=r"covers no position over \[3, 1\)"):
        FixedSpan(3, 1)


def test_negative_endpoints_are_refused() -> None:
    with pytest.raises(ValueError, match="a token position"):
        TokenPosition(-1)

    with pytest.raises(ValueError, match="a fixed span's start"):
        FixedSpan(-1, 2)

    with pytest.raises(ValueError, match="a fixed span's end"):
        FixedSpan(0, -2)


def test_a_pattern_matches_two_non_contiguous_spans() -> None:
    matched = FormPattern(("which", None)).spans(TOKENS)

    assert matched == ((2, 4), (5, 7))
    assert matched[0][1] < matched[1][0]


def fixed_shapes(width: int) -> tuple[SpanAccessor, ...]:
    """Return every fixed accessor writable over an input of a given width."""
    return (
        *(TokenPosition(at) for at in range(width)),
        *(
            FixedSpan(start, end)
            for start in range(width)
            for end in range(start + 1, width + 1)
        ),
        WholeInput(),
    )


def test_no_fixed_shape_covers_the_spans_one_pattern_does() -> None:
    matched = FormPattern(("which", "thing")).spans(TOKENS)
    shapes = fixed_shapes(len(TOKENS))
    agreeing = tuple(shape for shape in shapes if shape.spans(TOKENS)[0] in matched)

    assert matched == ((2, 4), (5, 7))
    assert matched[0][1] < matched[1][0]
    assert all(len(shape.spans(TOKENS)) == 1 for shape in shapes)
    assert agreeing == (FixedSpan(*matched[0]), FixedSpan(*matched[1]))


def test_the_pattern_reaches_spans_no_declared_fixed_shape_covers() -> None:
    matched = frozenset(FormPattern(("which", None)).spans(TOKENS))
    declared = frozenset(
        span
        for accessor in (TokenPosition(VERB[0]), FixedSpan(*PREDICATE), WholeInput())
        for span in accessor.spans(TOKENS)
    )

    assert matched == {(2, 4), (5, 7)}
    assert declared == {VERB, PREDICATE, ROOT}
    assert not matched & declared


def test_a_pattern_of_literals_matches_only_where_every_word_agrees() -> None:
    assert FormPattern(("jo", "saw")).spans(TOKENS) == ((0, 2),)
    assert FormPattern(("saw", "jo")).spans(TOKENS) == ()
    assert FormPattern(("saw", "which", "thing")).spans(TOKENS) == ((1, 4),)


def test_a_matcher_is_held_and_compared_exactly_as_it_was_written() -> None:
    padded = FormPattern((" which ",))
    capitalized = FormPattern(("WHICH", None))

    assert padded.pattern == (" which ",)
    assert padded.spans(TOKENS) == ()
    assert FormPattern(("which",)).spans(TOKENS) == ((2, 3), (5, 6))
    assert capitalized.pattern == ("WHICH", None)
    assert capitalized.spans(TOKENS) == ()
    assert FormPattern(("Which", "Thing")).spans(("Which", "Thing")) == ((0, 2),)
    assert FormPattern(("which", "thing")).spans(("Which", "Thing")) == ()


def test_a_pattern_wider_than_the_input_matches_nothing() -> None:
    assert FormPattern((None, None, None)).spans(TOKENS[:2]) == ()


def test_a_wildcard_pattern_matches_every_span_of_its_width() -> None:
    assert FormPattern((None, None)).spans(TOKENS[:3]) == ((0, 2), (1, 3))


def test_overlapping_matches_are_all_returned() -> None:
    assert FormPattern(("a", "a")).spans(("a", "a", "a")) == ((0, 2), (1, 3))


def test_a_pattern_needs_a_sequence_of_matchers() -> None:
    with pytest.raises(TypeError, match="sequence of matchers"):
        FormPattern("which")

    with pytest.raises(ValueError, match="no matcher covers no span"):
        FormPattern(())

    with pytest.raises(TypeError, match="matcher at 1 is not a word"):
        FormPattern(cast("Sequence[str | None]", ("which", 3)))

    with pytest.raises(ValueError, match="matcher at 0 is blank"):
        FormPattern((" ",))


def test_every_accessor_presents_the_accessor_surface() -> None:
    accessors: tuple[SpanAccessor, ...] = (
        TokenPosition(0),
        FixedSpan(0, 2),
        WholeInput(),
        FormPattern(("which",)),
    )

    for accessor in accessors:
        assert callable(accessor.spans)

        covered = accessor.spans(TOKENS)

        assert isinstance(covered, tuple)
        assert covered
        assert all(0 <= start < end <= len(TOKENS) for start, end in covered)
        assert list(covered) == sorted(covered)
        assert accessor.spans(TOKENS) == covered


def test_accessors_compare_on_what_they_cover() -> None:
    assert TokenPosition(2) == TokenPosition(2)
    assert TokenPosition(2) != TokenPosition(3)
    assert FixedSpan(1, 3) == FixedSpan(1, 3)
    assert FixedSpan(1, 3) != FixedSpan(1, 4)
    assert WholeInput() == WholeInput()
    assert FormPattern(("which", None)) == FormPattern(["which", None])
    assert FormPattern(("which", None)) != FormPattern((None, "which"))
    assert TokenPosition(2) != FixedSpan(2, 3)
    assert WholeInput() != FormPattern(("which",))


def test_accessors_hash_with_what_they_cover() -> None:
    held = {
        TokenPosition(2),
        TokenPosition(2),
        FixedSpan(1, 3),
        WholeInput(),
        WholeInput(),
        FormPattern(("which", None)),
    }

    differing = (
        (TokenPosition(2), TokenPosition(3)),
        (FixedSpan(1, 3), FixedSpan(1, 4)),
        (FormPattern(("which", None)), FormPattern((None, "which"))),
    )

    assert len(held) == 4
    assert hash(FormPattern(("which", None))) == hash(FormPattern(("which", None)))
    assert all(hash(left) != hash(right) for left, right in differing)


def test_accessors_render_as_the_call_that_rebuilds_them() -> None:
    assert repr(TokenPosition(2)) == "TokenPosition(2)"
    assert repr(FixedSpan(1, 3)) == "FixedSpan(1, 3)"
    assert repr(WholeInput()) == "WholeInput()"
    assert repr(FormPattern(("which", None))) == "FormPattern(['which', None])"


def test_accessors_are_immutable_and_copy_to_themselves() -> None:
    accessor = FixedSpan(1, 3)

    with pytest.raises(AttributeError, match="span accessor"):
        accessor.__setattr__("_start", 0)

    with pytest.raises(AttributeError, match="span accessor"):
        accessor.__delattr__("_start")

    assert copy.copy(accessor) is accessor
    assert copy.deepcopy(accessor) is accessor
