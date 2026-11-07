from typing import TYPE_CHECKING, Final

import pytest

from lofi.train.laws import (
    every_declared_parameter_is_consumed,
    nothing_raises_to_report_success,
    the_edges_are_exactly_the_data_dependencies,
    the_output_schema_does_not_depend_on_a_setting,
)

if TYPE_CHECKING:
    from lofi.train.laws import StepFixture

__all__ = [
    "TRAIN_STEP_CONFORMANCE_LAWS",
    "TrainStepConformance",
]


class TrainStepConformance:
    pytestmark = pytest.mark.conformance

    def test_the_output_schema_does_not_depend_on_a_setting[InputsT, OutputsT](
        self,
        conformance: StepFixture[InputsT, OutputsT],
    ) -> None:
        the_output_schema_does_not_depend_on_a_setting(conformance)

    def test_every_declared_parameter_is_consumed[InputsT, OutputsT](
        self,
        conformance: StepFixture[InputsT, OutputsT],
    ) -> None:
        every_declared_parameter_is_consumed(conformance)

    def test_nothing_raises_to_report_success[InputsT, OutputsT](
        self,
        conformance: StepFixture[InputsT, OutputsT],
    ) -> None:
        nothing_raises_to_report_success(conformance)

    def test_the_edges_are_exactly_the_data_dependencies[InputsT, OutputsT](
        self,
        conformance: StepFixture[InputsT, OutputsT],
    ) -> None:
        the_edges_are_exactly_the_data_dependencies(conformance)


TRAIN_STEP_CONFORMANCE_LAWS: Final = (
    TrainStepConformance.test_the_output_schema_does_not_depend_on_a_setting,
    TrainStepConformance.test_every_declared_parameter_is_consumed,
    TrainStepConformance.test_nothing_raises_to_report_success,
    TrainStepConformance.test_the_edges_are_exactly_the_data_dependencies,
)
