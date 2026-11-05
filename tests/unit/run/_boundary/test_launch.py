from pathlib import Path

import didactic.api as dx
import pytest
from tests.unit.run.conftest import a_profile

from lofi.run import (
    ArrayLaunchSpec,
    LaunchReport,
    LaunchRequest,
    LaunchSpec,
    PointOutcome,
    RunStatus,
    SerialLaunchSpec,
    SiteProfile,
)

FREEZE_ID = "freeze/1"


def a_request(
    *,
    experiment: str = "headline",
    freeze_id: str = FREEZE_ID,
    experiment_seed: int = 7,
    launch: LaunchSpec | None = None,
    root: str = "/runs",
    points: tuple[str, ...] = ("a", "b"),
    program: tuple[str, ...] = (),
    profile: SiteProfile | None = None,
) -> LaunchRequest:
    return LaunchRequest(
        experiment=experiment,
        freeze_id=freeze_id,
        experiment_seed=experiment_seed,
        launch=SerialLaunchSpec() if launch is None else launch,
        root=root,
        points=points,
        program=program,
        profile=profile,
    )


def variations() -> dict[str, LaunchRequest]:
    """Return one request per declared field, each varied in that field alone."""
    return {
        "experiment": a_request(experiment="analysis"),
        "freeze_id": a_request(freeze_id="freeze/2"),
        "experiment_seed": a_request(experiment_seed=8),
        "launch": a_request(launch=ArrayLaunchSpec(chunk_size=8, throttle=4)),
        "root": a_request(root="/elsewhere"),
        "points": a_request(points=("a", "b", "c")),
        "program": a_request(program=("-m", "lofi.run")),
        "profile": a_request(profile=a_profile(Path("sites").resolve(), tag="varied")),
    }


def an_outcome(point_id: str, *, cached: bool) -> PointOutcome:
    return PointOutcome(
        point_id=point_id,
        seed=0,
        status=RunStatus.COMPLETED,
        directory=f"/runs/{point_id}",
        cached=cached,
    )


@pytest.mark.parametrize("field", tuple(variations()))
def test_a_declared_parameter_varies_the_request_digest(field: str) -> None:
    assert variations()[field].digest() != a_request().digest()


def test_every_declared_parameter_is_varied() -> None:
    assert set(LaunchRequest.__field_specs__) == set(variations())


def test_the_site_profile_varies_the_request_digest(tmp_path: Path) -> None:
    one = a_request(profile=a_profile(tmp_path, tag="one"))
    other = a_request(profile=a_profile(tmp_path, tag="two"))
    accelerators = a_request(
        profile=a_profile(tmp_path, tag="one", accelerators_per_node=4)
    )

    assert one.digest() != other.digest()
    assert one.digest() != accelerators.digest()
    assert one.digest() == a_request(profile=a_profile(tmp_path, tag="one")).digest()


def test_a_request_digest_is_thirty_two_hexadecimal_digits() -> None:
    digest = a_request().digest()

    assert len(digest) == 32
    assert set(digest) <= set("0123456789abcdef")


def test_a_request_declaring_no_root_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="names the root it writes under"):
        a_request(root="")


def test_a_report_whose_hit_count_is_not_its_cached_outcomes_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="counts 1 cache hits"):
        LaunchReport(
            launch_id="0" * 32,
            launcher="serial",
            offered=2,
            cache_hits=1,
            launched=1,
            outcomes=(an_outcome("a", cached=False), an_outcome("b", cached=False)),
        )


def test_a_report_whose_outcomes_are_not_its_offered_points_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="offers 3 points"):
        LaunchReport(
            launch_id="0" * 32,
            launcher="serial",
            offered=3,
            cache_hits=0,
            launched=3,
            outcomes=(an_outcome("a", cached=False), an_outcome("b", cached=False)),
        )


def test_a_report_whose_halves_do_not_sum_to_its_offered_points_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="are not the 2 offered"):
        LaunchReport(
            launch_id="0" * 32,
            launcher="serial",
            offered=2,
            cache_hits=1,
            launched=2,
            outcomes=(an_outcome("a", cached=True), an_outcome("b", cached=False)),
        )


def test_a_report_naming_one_point_twice_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="two outcomes name one point"):
        LaunchReport(
            launch_id="0" * 32,
            launcher="serial",
            offered=2,
            cache_hits=0,
            launched=2,
            outcomes=(an_outcome("a", cached=False), an_outcome("a", cached=False)),
        )


def test_a_report_halting_points_while_no_stop_rule_fired_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="no stop rule fired"):
        LaunchReport(
            launch_id="0" * 32,
            launcher="serial",
            offered=2,
            cache_hits=0,
            launched=1,
            outcomes=(an_outcome("a", cached=False),),
            halted=("b",),
        )


def test_a_report_giving_a_halted_point_an_outcome_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="a halted point has an outcome"):
        LaunchReport(
            launch_id="0" * 32,
            launcher="serial",
            offered=2,
            cache_hits=1,
            launched=0,
            outcomes=(an_outcome("a", cached=True),),
            halted=("a",),
            stop_rules=("two completed",),
        )


@pytest.mark.parametrize(
    ("offered", "cache_hits", "launched"), [(-1, 1, 1), (2, -1, 3), (2, 3, -1)]
)
def test_a_report_counting_a_negative_number_of_points_is_refused(
    offered: int, cache_hits: int, launched: int
) -> None:
    with pytest.raises(dx.ValidationError, match="a count is not negative"):
        LaunchReport(
            launch_id="0" * 32,
            launcher="serial",
            offered=offered,
            cache_hits=cache_hits,
            launched=launched,
            outcomes=(an_outcome("a", cached=True), an_outcome("b", cached=False)),
        )


@pytest.mark.parametrize("field", ["point_id", "directory"])
@pytest.mark.parametrize("blank", ["", "   "])
def test_an_outcome_whose_identity_is_blank_is_refused(field: str, blank: str) -> None:
    identity = {"point_id": "a", "directory": "/runs/a", field: blank}

    with pytest.raises(dx.ValidationError, match="identity is empty"):
        PointOutcome(seed=0, status=RunStatus.COMPLETED, cached=False, **identity)


def test_an_outcome_whose_seed_is_negative_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="a seed counts from zero"):
        PointOutcome(
            point_id="a",
            seed=-1,
            status=RunStatus.COMPLETED,
            directory="/runs/a",
            cached=False,
        )
