from lofi.interpret._errors import CyclicForestError


def test_the_error_names_the_item_that_was_reached_from_itself() -> None:
    error = CyclicForestError(item="ItemKey(span=(0, 1), reading='N', nf_tag='')")

    assert error.item == "ItemKey(span=(0, 1), reading='N', nf_tag='')"
    assert "reached from itself" in str(error)
    assert "(0, 1)" in str(error)


def test_the_error_is_not_swallowed_by_a_handler_around_argument_checking() -> None:
    assert not issubclass(CyclicForestError, ValueError)
    assert not issubclass(CyclicForestError, TypeError)
