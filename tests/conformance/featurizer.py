"""Reusable conformance laws for data featurizers."""

from typing import TYPE_CHECKING, ClassVar, Final, NamedTuple, NoReturn

import pytest

from lofi._checks import LawViolationError
from lofi.data.featurizers import FeaturizedBatch, MixedSupervisionError

if TYPE_CHECKING:
    from lofi._types import ItemId
    from lofi.data import Example, Featurizer, Supervision

__all__ = [
    "FEATURIZER_LAWS",
    "FeaturizerConformance",
    "FeaturizerSuite",
]


class FeaturizerConformance(NamedTuple):
    """Inputs and expected outputs for the featurizer laws.

    Attributes
    ----------
    subject : Featurizer[Supervision[str], FeaturizedBatch[Supervision[str]]]
        Featurizer under test.
    examples : tuple[Example[Supervision[str]], ...]
        Compatible examples forming a nonempty minibatch.
    supervision : tuple[Supervision[str], ...]
        Expected output supervision in example order.
    members : tuple[tuple[ItemId, ...], ...]
        Item identifiers each example contributes, in semantic order.
    incompatible : Example[Supervision[str]]
        Example outside the selected supervision arm.
    """

    subject: Featurizer[Supervision[str], FeaturizedBatch[Supervision[str]]]
    examples: tuple[Example[Supervision[str]], ...]
    supervision: tuple[Supervision[str], ...]
    members: tuple[tuple[ItemId, ...], ...]
    incompatible: Example[Supervision[str]]


def _fail(detail: str, *, law: str) -> NoReturn:
    """Raise a law violation with its counterexample.

    Parameters
    ----------
    detail : str
        Counterexample description.
    law : str
        Law name.

    Raises
    ------
    LawViolationError
        Always.
    """
    message = f"{law}: {detail}"
    raise LawViolationError(message)


def the_selected_arm_is_accepted(fixture: FeaturizerConformance) -> None:
    """Check that the selected supervision arm is accepted.

    Parameters
    ----------
    fixture : FeaturizerConformance
        Featurizer and its examples.

    Raises
    ------
    LawViolationError
        If selection is absent or is not accepted.
    """
    selection = fixture.subject.selection

    if selection is None:
        _fail("the featurizer selects no arm", law="the_selected_arm_is_accepted")

    if selection not in fixture.subject.accepts:
        _fail(
            f"{selection.__name__} is selected but not accepted",
            law="the_selected_arm_is_accepted",
        )


def a_minibatch_matches_declared_outputs(
    fixture: FeaturizerConformance,
) -> None:
    """Check minibatch order, anchors, supervision, and member rows.

    Parameters
    ----------
    fixture : FeaturizerConformance
        Featurizer, examples, expected supervision, and expected members.

    Raises
    ------
    LawViolationError
        If the fixture is vacuous or the result does not match the declared
        supervision, anchors, or members.
    """
    if not fixture.examples:
        _fail(
            "the fixture supplies no example",
            law="a_minibatch_matches_declared_outputs",
        )

    result = fixture.subject(fixture.examples)
    if result.supervision != fixture.supervision:
        _fail(
            f"the result holds {result.supervision!r} instead of "
            f"{fixture.supervision!r}",
            law="a_minibatch_matches_declared_outputs",
        )

    anchors = tuple(example.item.item_id for example in fixture.examples)
    hashed = tuple(item.item_id for item in result.hashed)

    if hashed != anchors:
        _fail(
            f"the hashed anchors are {hashed!r} instead of {anchors!r}",
            law="a_minibatch_matches_declared_outputs",
        )

    members = tuple(
        tuple(result.batch.item_ids[row] for row in rows) for rows in result.member_rows
    )

    if members != fixture.members:
        _fail(
            f"the member groups are {members!r} instead of {fixture.members!r}",
            law="a_minibatch_matches_declared_outputs",
        )


def an_unselected_arm_is_refused(fixture: FeaturizerConformance) -> None:
    """Check that a minibatch from another supervision arm is refused.

    Parameters
    ----------
    fixture : FeaturizerConformance
        Featurizer and an incompatible example.

    Raises
    ------
    LawViolationError
        If the featurizer accepts the incompatible example or raises another error.
    """
    try:
        result = fixture.subject((fixture.incompatible,))
    except MixedSupervisionError:
        return
    except Exception as refusal:
        message = (
            "an_unselected_arm_is_refused: the incompatible example raised "
            f"{type(refusal).__name__} instead of MixedSupervisionError"
        )
        raise LawViolationError(message) from refusal

    _fail(
        f"the incompatible example produced {result!r}",
        law="an_unselected_arm_is_refused",
    )


FEATURIZER_LAWS: Final = (
    the_selected_arm_is_accepted,
    a_minibatch_matches_declared_outputs,
    an_unselected_arm_is_refused,
)


class FeaturizerSuite:
    """Shared pytest binding for the featurizer laws."""

    pytestmark: ClassVar = pytest.mark.conformance

    def test_the_selected_arm_is_accepted(
        self, conformance: FeaturizerConformance
    ) -> None:
        """Run the selected-arm law."""
        the_selected_arm_is_accepted(conformance)

    def test_a_minibatch_matches_declared_outputs(
        self, conformance: FeaturizerConformance
    ) -> None:
        """Run the minibatch-preservation law."""
        a_minibatch_matches_declared_outputs(conformance)

    def test_an_unselected_arm_is_refused(
        self, conformance: FeaturizerConformance
    ) -> None:
        """Run the incompatible-arm law."""
        an_unselected_arm_is_refused(conformance)
