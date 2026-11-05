from typing import TYPE_CHECKING, Final

import pytest

from lofi.nn.loss.laws import (
    a_masked_position_does_not_move_the_value,
    an_inapplicable_term_is_a_connected_zero,
    no_didactic_model_is_built,
    only_the_declared_outputs_are_read,
    permuting_the_items_leaves_the_value_alone,
    the_gradient_reaches_the_declared_inputs,
    the_objective_applies_no_weight_of_its_own,
    the_same_seed_gives_the_same_value,
    the_value_is_a_scalar,
)

if TYPE_CHECKING:
    from lofi.nn.loss.laws import ObjectiveFixture

__all__ = [
    "OBJECTIVE_CONFORMANCE_LAWS",
    "ObjectiveLawSuite",
]


class ObjectiveLawSuite:
    pytestmark = pytest.mark.conformance

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        hook = ObjectiveLawSuite.test_the_value_is_monotone_where_the_fixture_says

        if cls.test_the_value_is_monotone_where_the_fixture_says is hook:
            msg = (
                f"{cls.__name__} inherits the monotonicity hook without saying "
                "which way its objective orders two batches. Override "
                "test_the_value_is_monotone_where_the_fixture_says with the "
                "ordering this objective imposes."
            )
            raise TypeError(msg)

    def test_the_value_is_a_scalar[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that a term reports one number rather than a batch of them.

        Parameters
        ----------
        conformance
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the value, the residual or a diagnostic has any dimension, or if the
            value is not of floating dtype.
        """
        the_value_is_a_scalar(conformance)

    def test_an_inapplicable_term_is_a_connected_zero[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that a term that does not apply is a zero still on the graph.

        Parameters
        ----------
        conformance
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the term the fixture calls inapplicable reports items, is not zero, is
            detached from the graph, or leaves a gradient anywhere.
        """
        an_inapplicable_term_is_a_connected_zero(conformance)

    def test_only_the_declared_outputs_are_read[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that the objective reads the bundle, and only what it named.

        Parameters
        ----------
        conformance
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the objective declares a name outside the published vocabulary, reaches
            for a field it did not declare, or reads nothing at all.
        """
        only_the_declared_outputs_are_read(conformance)

    def test_a_masked_position_does_not_move_the_value[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that what the masks hide contributes nothing.

        Parameters
        ----------
        conformance
            What the laws run against.

        Raises
        ------
        LawViolationError
            If refilling the positions the masks exclude moves the value.
        """
        a_masked_position_does_not_move_the_value(conformance)

    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """State the ordering this objective imposes, which no subclass inherits.

        Parameters
        ----------
        conformance
            What the laws run against.

        Raises
        ------
        NotImplementedError
            Always.
        """
        del conformance
        msg = (
            "the monotonicity law is per-objective and this suite states none. "
            "Override test_the_value_is_monotone_where_the_fixture_says."
        )
        raise NotImplementedError(msg)

    def test_the_gradient_reaches_the_declared_inputs[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that a backward pass reaches every input the fixture names.

        Parameters
        ----------
        conformance
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the objective does not apply to the bundle the fixture names, if an input
            takes no gradient, or if the gradient it takes is zero or not finite
            everywhere.
        """
        the_gradient_reaches_the_declared_inputs(conformance)

    def test_the_objective_applies_no_weight_of_its_own[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that the term is the raw quantity and that the suite scales it.

        Parameters
        ----------
        conformance
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the term is not the quantity the fixture names, if it differs between two
            compositions, or if the total is not the weight times the term.
        """
        the_objective_applies_no_weight_of_its_own(conformance)

    def test_permuting_the_items_leaves_the_value_alone[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that the term does not depend on how the batch was ordered.

        Parameters
        ----------
        conformance
            What the laws run against.

        Raises
        ------
        LawViolationError
            If reordering the items moves the value.
        """
        permuting_the_items_leaves_the_value_alone(conformance)

    def test_the_same_seed_gives_the_same_value[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that two runs from one random state agree exactly.

        Parameters
        ----------
        conformance
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the two values differ at all.
        """
        the_same_seed_gives_the_same_value(conformance)

    def test_no_didactic_model_is_built[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that scoring a step constructs no boundary object.

        Parameters
        ----------
        conformance
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the fixture's counter reports any construction.
        """
        no_didactic_model_is_built(conformance)


OBJECTIVE_CONFORMANCE_LAWS: Final = (
    ObjectiveLawSuite.test_the_value_is_a_scalar,
    ObjectiveLawSuite.test_an_inapplicable_term_is_a_connected_zero,
    ObjectiveLawSuite.test_only_the_declared_outputs_are_read,
    ObjectiveLawSuite.test_a_masked_position_does_not_move_the_value,
    ObjectiveLawSuite.test_the_value_is_monotone_where_the_fixture_says,
    ObjectiveLawSuite.test_the_gradient_reaches_the_declared_inputs,
    ObjectiveLawSuite.test_the_objective_applies_no_weight_of_its_own,
    ObjectiveLawSuite.test_permuting_the_items_leaves_the_value_alone,
    ObjectiveLawSuite.test_the_same_seed_gives_the_same_value,
    ObjectiveLawSuite.test_no_didactic_model_is_built,
)
