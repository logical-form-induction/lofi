from lofi.lexicon._errors import LexiconError, UnsatisfiableClampError
from lofi.lexicon._ports import Clamp


def refusal(
    clamp: Clamp[str],
    offered: tuple[str, ...] = ("N", "NP"),
) -> UnsatisfiableClampError[str]:
    """Return the error raised when a source cannot satisfy a clamp.

    Parameters
    ----------
    clamp
        The unsatisfied clamp.
    offered
        The categories available at the position.

    Returns
    -------
    UnsatisfiableClampError
        The error with the clamp, position, form, and offered categories recorded.
    """
    return UnsatisfiableClampError(
        clamp=clamp,
        position=1,
        form="cat",
        offered=offered,
    )


def test_the_refusal_reads_back_what_it_was_built_from() -> None:
    clamp: Clamp[str] = Clamp((1, 2), category={"S\\NP"})

    error = refusal(clamp)

    assert error.clamp == clamp
    assert error.position == 1
    assert error.form == "cat"
    assert error.offered == ("N", "NP")


def test_the_message_names_the_span_the_category_and_what_was_offered() -> None:
    message = str(refusal(Clamp((1, 2), category={"S\\NP"})))

    assert "[1, 2)" in message
    assert "requires 'S\\\\NP'" in message
    assert "position 1, 'cat'" in message
    assert "offers 'N', 'NP'" in message


def test_the_message_names_a_meaning_beside_a_category() -> None:
    message = str(refusal(Clamp((1, 2), category={"N"}, meaning="dog")))

    assert "requires 'N', 'dog'" in message


def test_the_message_names_every_category_a_set_holds_in_one_order() -> None:
    forward = str(refusal(Clamp((1, 2), category=("S\\NP", "N"))))
    backward = str(refusal(Clamp((1, 2), category=("N", "S\\NP"))))

    assert "requires 'N', 'S\\\\NP'" in forward
    assert backward == forward


def test_the_message_names_a_wide_set_in_the_order_its_renderings_sort_in() -> None:
    wide: Clamp[str] = Clamp(
        (1, 2), category=("S\\NP", "N", "NP", "NP/N", "S", "S/NP", "N/N", "PP")
    )

    message = str(refusal(wide))
    named = "'N', 'N/N', 'NP', 'NP/N', 'PP', 'S', 'S/NP', 'S\\\\NP'"

    assert f"requires {named}, and" in message


def test_the_message_names_the_meaning_after_the_categories() -> None:
    message = str(refusal(Clamp((1, 2), category=("NP", "N"), meaning="dog")))

    assert "requires 'N', 'NP', 'dog'" in message


def test_the_message_says_what_a_negated_clamp_forbids() -> None:
    message = str(refusal(Clamp((1, 2), category={"N"}, negate=True)))

    assert "forbids 'N'" in message


def test_a_position_offering_nothing_is_said_to_offer_nothing() -> None:
    message = str(refusal(Clamp((1, 2), category={"N"}), offered=()))

    assert "offers nothing" in message


def test_a_clamp_naming_neither_slot_is_still_rendered() -> None:
    message = str(refusal(Clamp((1, 2))))

    assert "requires an entry" in message


def test_a_negative_bracket_is_said_to_exclude_a_span_every_derivation_builds() -> None:
    bracket: Clamp[str] = Clamp((0, 3), require_constituent=False)

    message = str(refusal(bracket, offered=()))

    assert "negative bracket on the span [0, 3)" in message
    assert "every derivation of the input brackets" in message
    assert "entry" not in message
    assert "offers" not in message


def test_the_refusal_is_a_lexicon_error_and_no_argument_error() -> None:
    error = refusal(Clamp((1, 2), category={"N"}))

    assert isinstance(error, LexiconError)
    assert not isinstance(error, ValueError)
    assert not isinstance(error, TypeError)
