import pytest

from lofi.nn.loss._schedule import TeacherForcingSchedule
from lofi.nn.ports import ForwardMode


def test_the_default_schedule_draws_twice_the_elapsed_fraction_capped_at_one() -> None:
    schedule = TeacherForcingSchedule()

    assert schedule.saturation == 0.5
    assert schedule.rate(0, horizon=100) == 0.0
    assert schedule.rate(10, horizon=100) == pytest.approx(0.2)
    assert schedule.rate(25, horizon=100) == pytest.approx(0.5)
    assert schedule.rate(50, horizon=100) == 1.0
    assert schedule.rate(99, horizon=100) == 1.0
    assert schedule.rate(300, horizon=100) == 1.0


def test_the_saturation_point_sets_where_the_whole_batch_is_first_drawn() -> None:
    schedule = TeacherForcingSchedule(0.25)

    assert schedule.rate(10, horizon=100) == pytest.approx(0.4)
    assert schedule.rate(25, horizon=100) == 1.0

    assert TeacherForcingSchedule(1.0).rate(50, horizon=100) == pytest.approx(0.5)


def test_the_first_step_observes_and_every_later_step_samples_at_its_rate() -> None:
    schedule = TeacherForcingSchedule()

    first = schedule.at(0, horizon=10)
    later = schedule.at(2, horizon=10)
    saturated = schedule.at(7, horizon=10)

    assert first.mode is ForwardMode.OBSERVE
    assert later.mode is ForwardMode.SAMPLE
    assert later.sampling_rate == pytest.approx(0.4)
    assert saturated.mode is ForwardMode.SAMPLE
    assert saturated.sampling_rate == 1.0
    assert schedule.modes == frozenset({ForwardMode.OBSERVE, ForwardMode.SAMPLE})


def test_a_schedule_without_a_saturation_point_observes_throughout() -> None:
    schedule = TeacherForcingSchedule(None)

    assert schedule.saturation is None
    assert schedule.rate(0, horizon=4) == 0.0
    assert schedule.rate(4, horizon=4) == 0.0
    assert schedule.at(3, horizon=4).mode is ForwardMode.OBSERVE
    assert schedule.modes == frozenset({ForwardMode.OBSERVE})


@pytest.mark.parametrize("saturation", [0.0, -0.5, 1.5, float("inf"), float("nan")])
def test_a_saturation_point_outside_the_unit_interval_is_refused(
    saturation: float,
) -> None:
    with pytest.raises(ValueError, match="saturation"):
        TeacherForcingSchedule(saturation)


def test_a_saturation_point_that_is_not_a_number_is_refused() -> None:
    flag = bool(1)

    with pytest.raises(TypeError, match="saturation"):
        TeacherForcingSchedule(flag)


def test_a_position_outside_a_run_is_refused() -> None:
    schedule = TeacherForcingSchedule()

    with pytest.raises(ValueError, match="step"):
        schedule.rate(-1, horizon=10)

    with pytest.raises(ValueError, match="no steps"):
        schedule.rate(0, horizon=0)

    flag = bool(1)

    with pytest.raises(TypeError, match="step"):
        schedule.at(flag, horizon=10)


def test_schedules_are_equal_when_they_saturate_at_the_same_point() -> None:
    assert TeacherForcingSchedule(0.5) == TeacherForcingSchedule()
    assert TeacherForcingSchedule(0.5) != TeacherForcingSchedule(0.25)
    assert TeacherForcingSchedule(None) != TeacherForcingSchedule(1.0)
    assert hash(TeacherForcingSchedule(0.5)) == hash(TeacherForcingSchedule())
    assert TeacherForcingSchedule(0.5) != "0.5"
    assert repr(TeacherForcingSchedule(0.25)) == "TeacherForcingSchedule(0.25)"
