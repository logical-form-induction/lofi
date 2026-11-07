import math
import shlex
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from tests.conformance.launcher import LauncherConformance, LauncherFixture
from tests.unit.run.conftest import (
    ScratchStore,
    a_document,
    a_point,
    a_price,
    a_scratch_store,
    record_manifest,
)
from tests.unit.run.test__experiment import an_experiment_document, write_experiment

from lofi.run import (
    POINTS_NAME,
    SCRIPT_NAME,
    SWEEPS_NAME,
    ArrayJob,
    ArrayLauncher,
    CancellationError,
    ExperimentIdentity,
    ExperimentStore,
    ListedPoints,
    ManifestStatus,
    RunHandle,
    RunStatus,
    SiteAllocation,
    SiteEnvironment,
    SiteInterpreter,
    SitePlacement,
    SiteProfile,
    SiteQuota,
    SiteStorage,
    SiteSubmission,
    SiteTransport,
    SubmissionError,
    SweepPoint,
    UnpricedPointError,
    read_experiment,
    read_points,
    resource_request,
    sweep_id,
    task_points,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lofi.run import ArraySubmission, ExperimentPrice

PROGRAM = ("-m", "lofi", "sweep-task")


def a_slurm_submission(tag: str) -> SiteSubmission:
    """Return a submission declaration spelling directives the way Slurm does."""
    return SiteSubmission(
        command=f"submit-{tag}",
        directive_prefix=f"#DIRECTIVE-{tag}",
        array_syntax="--array={first}-{last}",
        throttle_syntax="%{throttle}",
        account_syntax="--account={account}",
        partition_syntax="--partition={partition}",
        accelerator_syntax="--gres={accelerator}:{accelerators_per_node}",
        task_index_variable=f"TASK_INDEX_{tag}",
        walltime_syntax="--time={minutes}",
        memory_syntax="--mem={megabytes}",
        cpus_syntax="--cpus-per-task={cpus}",
        log_syntax="--output={directory}/%A_%a.log",
        name_syntax="--job-name={name}",
        job_prefix=f"{tag}-",
        cancel_command=(f"cancel-{tag}", "--name={name}"),
    )


def a_pbs_submission(tag: str) -> SiteSubmission:
    """Return a submission declaration spelling directives the way PBS does."""
    return SiteSubmission(
        command=f"qsub-{tag}",
        directive_prefix="#PBS",
        array_syntax="-J {first}-{last}",
        throttle_syntax="%{throttle}",
        account_syntax="-A {account}",
        partition_syntax="-q {partition}",
        accelerator_syntax="-l select=1:{accelerator}={accelerators_per_node}",
        task_index_variable=f"PBS_ARRAY_INDEX_{tag}",
        walltime_syntax="-l walltime=00:{minutes}:00",
        memory_syntax="-l mem={megabytes}mb",
        cpus_syntax="-l ncpus={cpus}",
        log_syntax="-o {directory}",
        name_syntax="-N {name}",
        job_prefix=f"{tag}-",
        cancel_command=(f"qdel-{tag}", "{name}"),
    )


def a_profile(
    scratch: Path,
    *,
    tag: str,
    accelerators_per_node: int = 1,
    submission: SiteSubmission | None = None,
    accelerator: str | None = None,
) -> SiteProfile:
    """Return a profile whose every string value is marked with a tag."""
    return SiteProfile(
        submission=submission if submission is not None else a_slurm_submission(tag),
        placement=SitePlacement(
            account=f"account-{tag}",
            partition=f"partition-{tag}",
            accelerator=accelerator
            if accelerator is not None
            else f"accelerator-{tag}",
            accelerators_per_node=accelerators_per_node,
            cpus_per_task=2,
        ),
        storage=SiteStorage(
            scratch_root=str(scratch / f"scratch-{tag}"),
            cache_root=str(scratch / f"cache-{tag}"),
            quota=SiteQuota(
                command=("echo", "used 0 bytes"),
                used_pattern=r"used (?P<used>\d+)",
                unit_bytes=1,
                limit_bytes=10**12,
            ),
            artifacts=(),
        ),
        allocation=SiteAllocation(accelerator_hours=1000.0, share=0.6),
        environment=SiteEnvironment(
            preamble=(f"module load toolkit-{tag}", f"export SITE={tag}")
        ),
        interpreter=SiteInterpreter(
            command=(f"container-{tag}", "exec", f"/images/{tag}.sif", "python")
        ),
        transport=SiteTransport(
            host=f"host-{tag}", command=(f"shell-{tag}", "-o", f"Option={tag}")
        ),
    )


@pytest.fixture
def make_profile(tmp_path: Path) -> Callable[[str], SiteProfile]:
    """Return a factory for profiles whose values differ by tag."""
    return lambda tag: a_profile(tmp_path, tag=tag)


class RecordingRunner:
    """Records every command it is given and exits with a fixed status."""

    def __init__(self, exit_status: int = 0) -> None:
        self._exit_status = exit_status
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str]) -> int:
        self.commands.append(tuple(argv))

        return self._exit_status


def three_points() -> tuple[SweepPoint, SweepPoint, SweepPoint]:
    return (a_point(1), a_point(2), a_point(3))


def a_launcher(
    profile: SiteProfile,
    *,
    runner: RecordingRunner,
    chunk_size: int = 2,
    throttle: int | None = 4,
    price: ExperimentPrice | None = None,
) -> ArrayLauncher:
    return ArrayLauncher(
        profile,
        store=a_scratch_store(profile),
        program=PROGRAM,
        chunk_size=chunk_size,
        price=a_price(three_points()) if price is None else price,
        throttle=throttle,
        run=runner,
    )


def script_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


class TestArrayLauncher(LauncherConformance):
    @pytest.fixture
    def launcher_fixture(
        self, make_profile: Callable[[str], SiteProfile]
    ) -> LauncherFixture:
        launcher = a_launcher(make_profile("conformance"), runner=RecordingRunner())

        def record(point: SweepPoint, status: ManifestStatus) -> None:
            record_manifest(launcher.store, point, status)

        return LauncherFixture(launcher, three_points(), record)


def test_a_dry_run_writes_one_script_and_one_points_file_and_runs_nothing(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    profile = make_profile("a")
    runner = RecordingRunner()
    launcher = a_launcher(profile, runner=runner, chunk_size=2)
    points = three_points()

    rendered = launcher.render(points)

    assert rendered is not None
    assert runner.commands == []
    assert rendered.directory == launcher.store.scope / SWEEPS_NAME / sweep_id(points)
    assert sorted(path.name for path in rendered.directory.iterdir()) == sorted(
        (POINTS_NAME, SCRIPT_NAME)
    )
    assert rendered.script == rendered.directory / SCRIPT_NAME
    assert rendered.points == rendered.directory / POINTS_NAME
    assert rendered.tasks == 2
    assert rendered.command == (profile.submission.command, str(rendered.script))
    assert read_points(rendered.points).points == points
    assert launcher.store.root == Path(profile.storage.scratch_root)


def test_the_script_reads_every_site_value_from_the_profile(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    profile = make_profile("a")
    launcher = a_launcher(profile, runner=RecordingRunner(), chunk_size=2, throttle=4)
    rendered = launcher.render(three_points())

    assert rendered is not None
    lines = script_lines(rendered.script)
    prefix = profile.submission.directive_prefix
    request = rendered.request
    assert lines[0] == "#!/bin/sh"
    assert lines[1:10] == [
        f"{prefix} --account={profile.placement.account}",
        f"{prefix} --partition={profile.placement.partition}",
        f"{prefix} --gres={profile.placement.accelerator}:1",
        f"{prefix} --time={request.minutes}",
        f"{prefix} --mem={request.megabytes}",
        f"{prefix} --cpus-per-task={profile.placement.cpus_per_task}",
        f"{prefix} --output={rendered.directory}/%A_%a.log",
        f"{prefix} --job-name={launcher.job.name}",
        f"{prefix} {profile.submission.array_directive(first=0, last=1, throttle=4)}",
    ]
    assert lines[10] == "set -eu"
    assert tuple(lines[11:13]) == profile.environment.preamble
    assert shlex.split(lines[13]) == [
        "exec",
        *profile.interpreter.command,
        *PROGRAM,
        str(rendered.points),
        f"${profile.submission.task_index_variable}",
        "2",
    ]
    assert lines[13].endswith(f' "${profile.submission.task_index_variable}" 2')
    assert len(lines) == 14


def test_two_profiles_differ_only_in_profile_values(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    points = three_points()
    first = make_profile("alpha")
    second = make_profile("beta")
    rendered_first = a_launcher(first, runner=RecordingRunner()).render(points)
    rendered_second = a_launcher(second, runner=RecordingRunner()).render(points)

    assert rendered_first is not None
    assert rendered_second is not None
    first_lines = script_lines(rendered_first.script)
    second_lines = script_lines(rendered_second.script)
    differing = [line for line in first_lines if line not in second_lines]

    assert differing
    assert all("alpha" in line for line in differing)
    assert [line.replace("alpha", "beta") for line in first_lines] == second_lines


def test_a_profile_without_accelerators_omits_the_accelerator_directive(
    tmp_path: Path,
) -> None:
    profile = a_profile(tmp_path, tag="cpu", accelerators_per_node=0)
    rendered = a_launcher(profile, runner=RecordingRunner()).render(three_points())

    assert rendered is not None
    lines = script_lines(rendered.script)
    assert not any("--gres" in line for line in lines)
    assert (
        sum(line.startswith(profile.submission.directive_prefix) for line in lines) == 8
    )


def test_a_scheduler_with_other_directive_spellings_needs_no_code_change(
    tmp_path: Path,
) -> None:
    profile = a_profile(
        tmp_path,
        tag="pbs",
        accelerators_per_node=2,
        submission=a_pbs_submission("pbs"),
        accelerator="ngpus",
    )
    rendered = a_launcher(profile, runner=RecordingRunner(), throttle=4).render(
        three_points()
    )

    assert rendered is not None
    request = rendered.request
    job = ArrayJob(profile, a_scratch_store(profile).identity, run=RecordingRunner())
    assert script_lines(rendered.script)[1:10] == [
        "#PBS -A account-pbs",
        "#PBS -q partition-pbs",
        "#PBS -l select=1:ngpus=2",
        f"#PBS -l walltime=00:{request.minutes}:00",
        f"#PBS -l mem={request.megabytes}mb",
        "#PBS -l ncpus=2",
        f"#PBS -o {rendered.directory}",
        f"#PBS -N {job.name}",
        "#PBS -J 0-1%4",
    ]


def test_the_slurm_spellings_render_the_directives_they_always_did(
    tmp_path: Path,
) -> None:
    profile = a_profile(tmp_path, tag="slurm", accelerators_per_node=3)
    rendered = a_launcher(profile, runner=RecordingRunner(), throttle=4).render(
        three_points()
    )

    assert rendered is not None
    lines = script_lines(rendered.script)
    assert lines[1:4] == [
        "#DIRECTIVE-slurm --account=account-slurm",
        "#DIRECTIVE-slurm --partition=partition-slurm",
        "#DIRECTIVE-slurm --gres=accelerator-slurm:3",
    ]
    assert lines[9] == "#DIRECTIVE-slurm --array=0-1%4"


def test_submission_runs_the_submit_command_once_and_hands_back_every_point(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    profile = make_profile("a")
    runner = RecordingRunner()
    launcher = a_launcher(profile, runner=runner, chunk_size=2)
    points = three_points()

    handles = launcher.submit(points)

    script = launcher.store.scope / SWEEPS_NAME / sweep_id(points) / SCRIPT_NAME
    assert runner.commands == [(profile.submission.command, str(script))]
    submission = sweep_id(points)
    assert handles == [
        RunHandle(points[0].point_id, f"{submission}:0"),
        RunHandle(points[1].point_id, f"{submission}:0"),
        RunHandle(points[2].point_id, f"{submission}:1"),
    ]
    assert [launcher.status(handle) for handle in handles] == [RunStatus.SUBMITTED] * 3

    record_manifest(launcher.store, points[1], ManifestStatus.COMPLETED)
    record_manifest(launcher.store, points[2], ManifestStatus.RUNNING)

    assert [launcher.status(handle) for handle in handles] == [
        RunStatus.SUBMITTED,
        RunStatus.COMPLETED,
        RunStatus.RUNNING,
    ]


def test_resubmission_lists_exactly_the_points_without_a_completed_manifest(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    runner = RecordingRunner()
    launcher = a_launcher(make_profile("a"), runner=runner, chunk_size=2)
    first, second, third = three_points()
    record_manifest(launcher.store, second, ManifestStatus.COMPLETED)
    record_manifest(launcher.store, third, ManifestStatus.KILLED)

    rendered = launcher.render((first, second, third))

    assert rendered is not None
    assert read_points(rendered.points).points == (first, third)
    assert rendered.tasks == 1
    assert rendered.directory.name == sweep_id((first, third))

    handles = launcher.submit((first, second, third))

    assert [handle.point_id for handle in handles] == [
        first.point_id,
        third.point_id,
    ]
    assert len(runner.commands) == 1

    record_manifest(launcher.store, first, ManifestStatus.COMPLETED)
    record_manifest(launcher.store, third, ManifestStatus.COMPLETED)

    assert launcher.render((first, second, third)) is None
    assert launcher.submit((first, second, third)) == []
    assert len(runner.commands) == 1


def test_a_failed_submit_command_raises_and_hands_back_no_handle(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    profile = make_profile("a")
    runner = RecordingRunner(exit_status=3)
    launcher = a_launcher(profile, runner=runner)
    points = three_points()

    with pytest.raises(SubmissionError, match=profile.submission.command) as caught:
        launcher.submit(points)

    assert caught.value.exit_status == 3
    assert caught.value.command == runner.commands[0]

    with pytest.raises(LookupError):
        launcher.status(RunHandle(points[0].point_id, "anything"))


def test_a_publication_sized_sweep_renders_in_chunks(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    profile = make_profile("a")
    points = tuple(SweepPoint(a_document(n)) for n in range(4300))
    launcher = a_launcher(
        profile,
        runner=RecordingRunner(),
        chunk_size=8,
        throttle=64,
        price=a_price(points),
    )

    rendered = launcher.render(points)

    assert rendered is not None
    assert rendered.tasks == 538
    assert (
        f"{profile.submission.directive_prefix} "
        f"{profile.submission.array_directive(first=0, last=537, throttle=64)}"
    ) in script_lines(rendered.script)
    listed = read_points(rendered.points).points
    assert len(listed) == 4300
    assert task_points(listed, 537, 8) == points[4296:]


@pytest.mark.parametrize(
    ("program", "chunk_size", "throttle", "detail"),
    [
        ((), 2, 4, "program is empty"),
        (("", "lofi"), 2, 4, "program word"),
        (PROGRAM, 0, 4, "chunk size"),
        (PROGRAM, 2, 0, "throttle"),
    ],
)
def test_a_launcher_refuses_an_empty_program_or_a_nonpositive_size(
    make_profile: Callable[[str], SiteProfile],
    program: tuple[str, ...],
    chunk_size: int,
    throttle: int,
    detail: str,
) -> None:
    profile = make_profile("a")

    with pytest.raises(ValueError, match=detail):
        ArrayLauncher(
            profile,
            store=a_scratch_store(profile),
            program=program,
            chunk_size=chunk_size,
            price=a_price(three_points()),
            throttle=throttle,
            run=RecordingRunner(),
        )


def an_experiment_store(profile: SiteProfile, repository: Path) -> ExperimentStore:
    experiment = read_experiment(write_experiment(repository, an_experiment_document()))

    return ExperimentStore(Path(profile.storage.scratch_root), experiment)


def test_an_experiment_launcher_names_its_experiment_and_freeze_in_the_points_file(
    make_profile: Callable[[str], SiteProfile], tmp_path: Path
) -> None:
    profile = make_profile("a")
    store = an_experiment_store(profile, tmp_path / "repository")
    launcher = ArrayLauncher(
        profile,
        store=store,
        program=PROGRAM,
        chunk_size=2,
        price=a_price(three_points()),
        throttle=4,
        run=RecordingRunner(),
    )

    rendered = launcher.render(three_points())

    assert rendered is not None
    assert read_points(rendered.points) == ListedPoints(
        ExperimentIdentity("headline", store.experiment.freeze_id),
        three_points(),
        sha256(rendered.script.read_bytes()).hexdigest()[:16],
    )
    assert rendered.directory.is_relative_to(store.scope)
    assert store.scope == store.root / store.experiment.freeze_id


def test_two_freezes_queue_their_submissions_apart(
    make_profile: Callable[[str], SiteProfile], tmp_path: Path
) -> None:
    profile = make_profile("a")
    first = an_experiment_store(profile, tmp_path / "first")
    document = an_experiment_document()
    document["freeze_id"] = "freeze/2"
    experiment = read_experiment(write_experiment(tmp_path / "second", document))
    second = ExperimentStore(first.root, experiment)
    rendered = [
        ArrayLauncher(
            profile,
            store=store,
            program=PROGRAM,
            chunk_size=2,
            price=a_price(three_points()),
            throttle=4,
            run=RecordingRunner(),
        ).render(three_points())
        for store in (first, second)
    ]

    queued, requeued = (submission for submission in rendered if submission)
    assert queued.points != requeued.points
    assert read_points(queued.points).identity == first.identity
    assert read_points(requeued.points).identity == second.identity


def test_a_launcher_refuses_a_store_off_the_scratch_root(
    make_profile: Callable[[str], SiteProfile], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="scratch root"):
        ArrayLauncher(
            make_profile("a"),
            store=ScratchStore(tmp_path / "elsewhere"),
            program=PROGRAM,
            chunk_size=2,
            price=a_price(three_points()),
            throttle=4,
            run=RecordingRunner(),
        )


def test_an_experiment_launcher_reads_status_from_the_experiment_store(
    make_profile: Callable[[str], SiteProfile], tmp_path: Path
) -> None:
    profile = make_profile("a")
    store = an_experiment_store(profile, tmp_path / "repository")
    point = a_point(1)
    launcher = ArrayLauncher(
        profile,
        store=store,
        program=PROGRAM,
        chunk_size=2,
        price=a_price((point,)),
        throttle=4,
        run=RecordingRunner(),
    )
    (handle,) = launcher.submit((point,))
    store.claim(point)
    record_manifest(store, point, ManifestStatus.COMPLETED)

    assert store.status(point) is RunStatus.FAILED
    assert launcher.status(handle) is RunStatus.FAILED


def test_the_resource_directives_change_with_the_price(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    profile = make_profile("a")
    points = three_points()
    quick_price = a_price(points)
    slow_price = a_price(points, seconds=(10.0, 20.0, 30.0), peak=6 * 1024**3)
    quick = a_launcher(profile, runner=RecordingRunner(), price=quick_price)
    slow = a_launcher(profile, runner=RecordingRunner(), price=slow_price)

    quick_lines = script_lines(cast("ArraySubmission", quick.render(points)).script)
    slow_lines = script_lines(cast("ArraySubmission", slow.render(points)).script)
    prefix = profile.submission.directive_prefix
    fast_minutes = resource_request(quick_price, points, 2).minutes
    slow_minutes = resource_request(slow_price, points, 2).minutes

    assert slow_minutes > 5 * fast_minutes
    assert f"{prefix} --time={fast_minutes}" in quick_lines
    assert f"{prefix} --mem=3072" in quick_lines
    assert f"{prefix} --time={slow_minutes}" in slow_lines
    assert f"{prefix} --mem=6144" in slow_lines


def test_a_task_is_granted_its_chunk_at_the_slowest_points_p90_seconds() -> None:
    points = three_points()
    price = a_price(points, seconds=(10.0, 20.0, 30.0))
    p90 = max(price.point_seconds(0.9).values())

    assert resource_request(price, points, 1).minutes == math.ceil(p90 / 60)
    assert resource_request(price, points, 3).minutes == math.ceil(3 * p90 / 60)
    assert resource_request(price, points, 3).megabytes == 3 * 1024
    assert resource_request(a_price(points, peak=None), points, 1).megabytes == 8 * 1024


def test_an_over_budget_stack_is_absent_from_the_points_file(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    runner = RecordingRunner()
    admitted = three_points()
    refused = (a_point(4), a_point(5))
    launcher = a_launcher(
        make_profile("a"), runner=runner, price=a_price(admitted, refused=refused)
    )

    rendered = launcher.render((*admitted, *refused))

    assert rendered is not None
    assert read_points(rendered.points).points == admitted
    assert rendered.refused == tuple(point.point_id for point in refused)
    assert launcher.admitted((*refused, admitted[0])) == (admitted[0],)

    handles = launcher.submit((*admitted, *refused))

    assert [handle.point_id for handle in handles] == [
        point.point_id for point in admitted
    ]
    assert launcher.render(refused) is None


def test_the_points_file_records_the_version_of_its_script(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    rendered = a_launcher(make_profile("a"), runner=RecordingRunner()).render(
        three_points()
    )
    other = a_launcher(
        make_profile("b"),
        runner=RecordingRunner(),
        price=a_price(three_points(), peak=1024**3),
    ).render(three_points())

    assert rendered is not None
    assert other is not None
    assert (
        read_points(rendered.points).script
        == (sha256(rendered.script.read_bytes()).hexdigest()[:16])
    )
    assert read_points(other.points).script != read_points(rendered.points).script


def test_the_throttle_defaults_to_the_task_count(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    profile = make_profile("a")
    rendered = a_launcher(
        profile, runner=RecordingRunner(), chunk_size=1, throttle=None
    ).render(three_points())

    assert rendered is not None
    assert rendered.tasks == 3
    assert (
        f"{profile.submission.directive_prefix} "
        f"{profile.submission.array_directive(first=0, last=2, throttle=3)}"
    ) in script_lines(rendered.script)


def test_a_point_the_price_does_not_price_is_refused(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    launcher = a_launcher(
        make_profile("a"), runner=RecordingRunner(), price=a_price((a_point(1),))
    )

    with pytest.raises(UnpricedPointError, match="prices no stack") as refused:
        launcher.render((a_point(1), a_point(2)))

    assert refused.value.point_ids == (a_point(2).point_id,)


def test_an_experiment_names_its_array_job_by_its_experiment_and_freeze(
    make_profile: Callable[[str], SiteProfile], tmp_path: Path
) -> None:
    profile = make_profile("a")
    runner = RecordingRunner()
    store = an_experiment_store(profile, tmp_path / "repository")
    job = ArrayJob(profile, store.identity, run=runner)
    names = {
        ArrayJob(profile, identity, run=runner).name
        for identity in (
            store.identity,
            ExperimentIdentity("headline", "freeze/2"),
            ExperimentIdentity("analysis", store.experiment.freeze_id),
        )
    }

    assert job.name == ArrayJob(profile, store.identity, run=runner).name
    assert job.name.startswith(profile.submission.job_prefix)
    assert len(names) == 3
    assert job.name.isascii()
    assert all(character.isalnum() or character == "-" for character in job.name)


def test_canceling_an_array_job_runs_the_declared_cancel_command(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    profile = make_profile("a")
    runner = RecordingRunner()
    launcher = a_launcher(profile, runner=runner)
    launcher.submit(three_points())

    launcher.cancel()

    assert runner.commands[-1] == ("cancel-a", f"--name={launcher.job.name}")


def test_a_failed_cancel_command_is_a_named_error(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    profile = make_profile("a")
    runner = RecordingRunner(exit_status=5)
    job = ArrayJob(profile, a_scratch_store(profile).identity, run=runner)

    with pytest.raises(CancellationError, match="cancel-a") as caught:
        job.cancel()

    assert caught.value.exit_status == 5
    assert caught.value.command == runner.commands[0]
