from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tests.conformance._sites import SiteMapSuite, SiteObservation

from lofi._checks import LawViolationError
from lofi.judge import Action, ActionSet

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.nn._values import Encoded
    from lofi.nn.ports import Controller

__all__ = [
    "CONTROLLER_LAWS",
    "ControllerConformance",
    "ControllerSuite",
    "action_sets",
]

_ACTION_AXIS: Final = "action"


def _fail(detail: str, *, law: str) -> NoReturn:
    """Raise a law violation, naming the law and what went wrong.

    Parameters
    ----------
    detail : str
        The counterexample, in that law's own terms.
    law : str
        The law's short name.

    Raises
    ------
    LawViolationError
        Always.
    """
    message = f"{law}: {detail}"

    raise LawViolationError(message)


def action_sets(smallest: int = 2, largest: int = 12) -> st.SearchStrategy[ActionSet]:
    """Return a strategy over action sets of a bounded size.

    Parameters
    ----------
    smallest : int
        The fewest actions a generated set holds.
    largest : int
        The most it holds.

    Returns
    -------
    hypothesis.strategies.SearchStrategy
        The strategy.
    """
    names = st.lists(
        st.text(alphabet="abcdefgh", min_size=1, max_size=3),
        min_size=smallest,
        max_size=largest,
        unique=True,
    )

    return names.flatmap(
        lambda drawn: st.integers(min_value=0, max_value=len(drawn) - 1).map(
            lambda at: ActionSet(
                tuple(
                    Action(name, None if position == at else f"rule-{name}")
                    for position, name in enumerate(drawn)
                ),
                Action(drawn[at], None),
            )
        )
    )


class ControllerConformance[T: Controller](NamedTuple):
    """Inputs for the controller conformance laws.

    Parameters
    ----------
    build : Callable[[ActionSet], T]
        Return a controller weighing the actions of a given set.
    actions : ActionSet
        The set the subject under test is built for.
    cells : Callable[[int], Encoded]
        Return the vectors of a given number of chart cells, over the axes ``("cell",
        "width")``.
    load : Callable[[T, str], None]
        Read a checkpoint into a controller, given the action-set digest that checkpoint
        recorded.
    sites : SiteObservation[T]
        How the site map law reaches this controller's interior.
    """

    build: Callable[[ActionSet], T]
    actions: ActionSet
    cells: Callable[[int], Encoded]
    load: Callable[[T, str], None]
    sites: SiteObservation[T]


class ControllerSuite(SiteMapSuite):
    pytestmark = pytest.mark.conformance

    def test_the_column_count_is_the_size_of_the_action_set[T: Controller](
        self,
        conformance: ControllerConformance[T],
    ) -> None:
        """Check that the output has one column per action and no more.

        Parameters
        ----------
        conformance : ControllerConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the action axis is not the size of the set the controller publishes,
            which is what a column count kept beside the set allows.
        """
        subject = conformance.build(conformance.actions)
        declared = subject.action_set
        weights = subject.weigh(conformance.cells(3), right=conformance.cells(3))
        columns = weights.size(_ACTION_AXIS)

        if columns != len(declared.actions):
            _fail(
                f"{len(declared.actions)} actions are weighed in {columns} columns",
                law="the_column_count_is_the_size_of_the_action_set",
            )

    def test_the_column_order_is_the_action_order[T: Controller](
        self,
        conformance: ControllerConformance[T],
    ) -> None:
        """Check that the columns are the set's actions in the set's order.

        Parameters
        ----------
        conformance : ControllerConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the controller reports the actions in an order other than the one it was
            built with, which repoints every column of every checkpoint trained against
            the old order.
        """
        subject = conformance.build(conformance.actions)
        reported = subject.action_set.actions

        if reported != conformance.actions.actions:
            _fail(
                f"the columns are {[action.name for action in reported]} where the "
                f"set orders them "
                f"{[action.name for action in conformance.actions.actions]}",
                law="the_column_order_is_the_action_order",
            )

        for column, action in enumerate(conformance.actions.actions):
            if subject.action_set.index(action.name) != column:
                _fail(
                    f"{action.name!r} is column {column} of the set and column "
                    f"{subject.action_set.index(action.name)} of the controller",
                    law="the_column_order_is_the_action_order",
                )

    def test_the_identity_is_a_column_rather_than_an_increment[T: Controller](
        self,
        conformance: ControllerConformance[T],
    ) -> None:
        """Check that the identity is the column the set names, not one more.

        Parameters
        ----------
        conformance : ControllerConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the identity is not among the actions, or is one the controller chose
            rather than the one the set names, which is how a column reserved for the
            null action appears from outside.
        """
        subject = conformance.build(conformance.actions)
        declared = subject.action_set

        if declared.identity not in declared.actions:
            _fail(
                f"the identity {declared.identity!r} is not one of "
                f"{[action.name for action in declared.actions]}",
                law="the_identity_is_a_column_rather_than_an_increment",
            )

        if declared.identity != conformance.actions.identity:
            _fail(
                f"the set names {conformance.actions.identity!r} as its identity "
                f"and the controller takes {declared.identity!r}",
                law="the_identity_is_a_column_rather_than_an_increment",
            )

    def test_a_checkpoint_from_another_action_set_is_refused[T: Controller](
        self,
        conformance: ControllerConformance[T],
    ) -> None:
        """Check that a stored digest other than this set's is not loaded.

        Parameters
        ----------
        conformance : ControllerConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the controller loads a checkpoint recorded under a digest that is not its
            own, or rejects one recorded under its own.
        """
        subject = conformance.build(conformance.actions)
        reversed_set = ActionSet(
            tuple(reversed(conformance.actions.actions)),
            conformance.actions.identity,
        )
        foreign = reversed_set.digest

        if foreign == subject.action_set.digest:
            _fail(
                f"reversing {conformance.actions!r} leaves the digest alone, so "
                "the law has nothing to offer the loader",
                law="a_checkpoint_from_another_action_set_is_refused",
            )

        try:
            conformance.load(subject, foreign)

        except ValueError:
            pass

        else:
            _fail(
                f"a checkpoint recorded under {foreign} loaded into a controller "
                f"whose set digests to {subject.action_set.digest}",
                law="a_checkpoint_from_another_action_set_is_refused",
            )

        conformance.load(subject, subject.action_set.digest)

    @settings(
        max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(actions=action_sets())
    def test_the_columns_are_any_generated_set_in_order[T: Controller](
        self,
        conformance: ControllerConformance[T],
        actions: ActionSet,
    ) -> None:
        """Check the ordering against sets of two to twelve members.

        Parameters
        ----------
        conformance : ControllerConformance[T]
            What the laws run against.
        actions : ActionSet
            A generated action set the controller is rebuilt for.

        Raises
        ------
        LawViolationError
            If a controller built for a generated set reports that set in another order,
            or weighs a number of columns other than its size.
        """
        subject = conformance.build(actions)
        weights = subject.weigh(conformance.cells(2), right=conformance.cells(2))

        if subject.action_set.actions != actions.actions:
            _fail(
                f"built for {[action.name for action in actions.actions]}, it "
                f"reports {[action.name for action in subject.action_set.actions]}",
                law="the_columns_are_any_generated_set_in_order",
            )

        if weights.size(_ACTION_AXIS) != len(actions.actions):
            _fail(
                f"{len(actions.actions)} actions are weighed in "
                f"{weights.size(_ACTION_AXIS)} columns",
                law="the_columns_are_any_generated_set_in_order",
            )


CONTROLLER_LAWS: Final = (
    ControllerSuite.test_the_column_count_is_the_size_of_the_action_set,
    ControllerSuite.test_the_column_order_is_the_action_order,
    ControllerSuite.test_the_identity_is_a_column_rather_than_an_increment,
    ControllerSuite.test_a_checkpoint_from_another_action_set_is_refused,
    ControllerSuite.test_the_site_map_matches_what_instrumentation_yields,
)
