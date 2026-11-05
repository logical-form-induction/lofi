"""Tests for shared synthetic declaration axioms."""

import didactic.api as dx
import pytest

from lofi.data.synthetic.spec._validation import all_named, nonempty


class _Declaration(dx.Model, extra="forbid"):
    """A text and tuple checked by the shared declaration axioms."""

    name: str
    members: tuple[str, ...]

    __axioms__ = (
        nonempty("name", "a name is required"),
        all_named("members", "every member needs a name"),
    )


def test_shared_axioms_accept_named_values() -> None:
    assert _Declaration(name="held", members=("one", "two")).name == "held"


@pytest.mark.parametrize(
    ("name", "members", "message"),
    [("", ("one",), "name is required"), ("held", ("",), "member needs")],
)
def test_shared_axioms_refuse_empty_values(
    name: str, members: tuple[str, ...], message: str
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        _Declaration(name=name, members=members)
