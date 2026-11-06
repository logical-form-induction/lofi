import json
import re
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Final, cast

import didactic.api as dx
import pytest

from lofi._types import JsonValue
from lofi.run import SiteProfile
from lofi.run._boundary.site import (
    SITE_PROFILE_VARIABLE,
    ArtifactClass,
    Launcher,
    RunHandle,
    RunStatus,
    SiteAllocation,
    SitePlacement,
    SiteQuota,
    SweepExpander,
    resolve_site_profile,
)
from lofi.run._boundary.sweep import Sweep
from lofi.spec import (
    ConfigDocumentError,
    ConfigValueError,
    MissingSettingError,
    UnknownSettingError,
)


def test_run_handles_are_value_objects() -> None:
    first = RunHandle("abcd", "job-1")
    second = RunHandle("abcd", "job-1")

    assert first == second
    assert hash(first) == hash(second)
    assert first != RunHandle("abcd", "job-2")
    assert first.point_id == "abcd"
    assert first.backend_id == "job-1"


def test_the_protocols_require_their_complete_surfaces() -> None:
    assert not isinstance(cast("SweepExpander", Sweep()), SweepExpander)
    assert not isinstance(cast("Launcher", RunHandle("abcd", "job-1")), Launcher)
    assert tuple(RunStatus) == (
        RunStatus.SUBMITTED,
        RunStatus.RUNNING,
        RunStatus.COMPLETED,
        RunStatus.FAILED,
    )


EXAMPLE_SITE_PROFILE: Final = (
    Path(__file__).resolve().parents[4] / "docs" / "examples" / "site-profile.toml"
)
WORKED_SITE_PROFILES: Final = (
    Path(__file__).resolve().parents[4] / "config" / "sites" / "array.toml",
    Path(__file__).resolve().parents[4] / "config" / "sites" / "local-accelerator.toml",
)

type Document = dict[str, JsonValue]


def _example_document() -> Document:
    return tomllib.loads(EXAMPLE_SITE_PROFILE.read_text(encoding="utf-8"))


def _paths(document: Document, prefix: str = "") -> tuple[str, ...]:
    """Return every dotted path in a document, tables and leaves alike."""
    found: list[str] = []

    for key, value in document.items():
        path = f"{prefix}{key}"
        found.append(path)

        if isinstance(value, dict):
            found.extend(_paths(value, f"{path}."))

    return tuple(found)


def _leaves(document: Document, prefix: str = "") -> dict[str, JsonValue]:
    """Return every non-table value in a document keyed by dotted path."""
    found: dict[str, JsonValue] = {}

    for key, value in document.items():
        path = f"{prefix}{key}"

        if isinstance(value, dict):
            found.update(_leaves(value, f"{path}."))

        else:
            found[path] = value

    return found


def _strings(value: JsonValue) -> tuple[str, ...]:
    """Return every string inside a document value."""
    if isinstance(value, str):
        return (value,)

    if isinstance(value, dict):
        return tuple(text for inner in value.values() for text in _strings(inner))

    if isinstance(value, list):
        return tuple(text for inner in value for text in _strings(inner))

    return ()


def _edited(document: Document, path: str, value: JsonValue | None) -> Document:
    """Return a copy with the value at a path replaced, or removed when None."""
    edited = deepcopy(document)
    *tables, key = path.split(".")
    table = edited

    for name in tables:
        table = cast("Document", table[name])

    if value is None:
        del table[key]

    else:
        table[key] = value

    return edited


def _resolve_document(tmp_path: Path, document: Document) -> SiteProfile:
    """Resolve a profile from a document written where the environment points."""
    written = tmp_path / "site-profile.json"
    written.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    return resolve_site_profile({SITE_PROFILE_VARIABLE: str(written)})


def _assert_every_site_field_is_required(model: dx.Model) -> None:
    """Assert that this site model and every nested model declare no default."""
    fields = type(model).__field_specs__

    assert set(model.model_dump()) == set(fields)
    assert all(field.is_required for field in fields.values())

    for name in fields:
        value = getattr(model, name)
        if isinstance(value, dx.Model):
            _assert_every_site_field_is_required(value)


def test_the_example_profile_resolves_from_the_file_the_environment_names() -> None:
    profile = resolve_site_profile({SITE_PROFILE_VARIABLE: str(EXAMPLE_SITE_PROFILE)})

    assert json.loads(profile.model_dump_json()) == _example_document()


@pytest.mark.parametrize("path", WORKED_SITE_PROFILES, ids=lambda path: path.stem)
def test_a_worked_profile_resolves_without_defaulting_a_field(path: Path) -> None:
    profile = resolve_site_profile({SITE_PROFILE_VARIABLE: str(path)})

    assert isinstance(profile, SiteProfile)
    _assert_every_site_field_is_required(profile)


@pytest.mark.parametrize("path", WORKED_SITE_PROFILES, ids=lambda path: path.stem)
def test_every_worked_profile_value_has_an_explanatory_comment(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()

    for index, line in enumerate(lines):
        if "=" not in line:
            continue

        assert index > 0
        assert lines[index - 1].lstrip().startswith("#"), line


def test_a_resolved_profile_round_trips_through_json() -> None:
    profile = resolve_site_profile({SITE_PROFILE_VARIABLE: str(EXAMPLE_SITE_PROFILE)})

    assert SiteProfile.model_validate_json(profile.model_dump_json()) == profile


def test_resolution_without_the_variable_names_the_variable() -> None:
    with pytest.raises(MissingSettingError) as refused:
        resolve_site_profile({"PATH": "/placeholder/bin"})

    assert refused.value.path == SITE_PROFILE_VARIABLE


def test_resolution_against_an_absent_file_names_the_document(tmp_path: Path) -> None:
    absent = tmp_path / "absent.toml"

    with pytest.raises(ConfigDocumentError) as refused:
        resolve_site_profile({SITE_PROFILE_VARIABLE: str(absent)})

    assert refused.value.document == absent


def test_resolution_against_an_unparsable_file_names_the_document(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "broken.toml"
    broken.write_text("[submission]\ncommand = \n", encoding="utf-8", newline="\n")

    with pytest.raises(ConfigDocumentError) as refused:
        resolve_site_profile({SITE_PROFILE_VARIABLE: str(broken)})

    assert refused.value.document == broken
    assert "parse" in refused.value.reason


@pytest.mark.parametrize("path", _paths(_example_document()))
def test_removing_any_site_value_is_an_error_naming_it(
    tmp_path: Path, path: str
) -> None:
    document = _edited(_example_document(), path, None)

    with pytest.raises(MissingSettingError) as refused:
        _resolve_document(tmp_path, document)

    assert refused.value.path == path


@pytest.mark.parametrize("path", sorted(_leaves(_example_document())))
def test_a_malformed_site_value_is_an_error_naming_it(
    tmp_path: Path, path: str
) -> None:
    document = _edited(_example_document(), path, {"malformed": True})

    with pytest.raises(ConfigValueError) as refused:
        _resolve_document(tmp_path, document)

    assert refused.value.path == path


def test_an_undeclared_site_value_is_refused_by_name(tmp_path: Path) -> None:
    document = _edited(_example_document(), "placement.cluster", "placeholder")

    with pytest.raises(UnknownSettingError) as refused:
        _resolve_document(tmp_path, document)

    assert refused.value.path == "placement.cluster"


def test_the_example_profile_names_no_real_site() -> None:
    document = _example_document()
    leaves = _leaves(document)
    identities = (
        "placement.account",
        "placement.partition",
        "storage.scratch_root",
        "storage.cache_root",
        "transport.host",
    )

    for path in identities:
        assert "placeholder" in cast("str", leaves[path]), path

    domain = re.compile(r"[\w-]+\.(edu|org|com|net|gov|io|ac)\b")

    for text in _strings(document):
        assert "@" not in text, text
        assert domain.search(text) is None, text

        for word in text.split():
            if word.startswith("/"):
                assert word.startswith("/placeholder/"), text


@pytest.mark.parametrize(
    ("path", "value", "detail"),
    [
        ("submission.array_syntax", "--array={first}", "last"),
        ("submission.array_syntax", "--array={first}-{last}-{node}", "node"),
        ("submission.throttle_syntax", "%", "throttle"),
        ("submission.throttle_syntax", "{first}", "first"),
        ("submission.account_syntax", "-A", "account"),
        ("submission.account_syntax", "-A {account} {partition}", "partition"),
        ("submission.partition_syntax", "-q", "partition"),
        (
            "submission.accelerator_syntax",
            "-l ngpus={accelerator}",
            "accelerators_per_node",
        ),
        (
            "submission.accelerator_syntax",
            "-l ngpus={accelerators_per_node}",
            "accelerator",
        ),
        ("submission.command", "", "submission command"),
        ("submission.task_index_variable", "", "task index variable"),
        ("submission.walltime_syntax", "--time", "minutes"),
        ("submission.walltime_syntax", "--time={hours}", "hours"),
        ("submission.memory_syntax", "--mem={minutes}", "megabytes"),
        ("submission.cpus_syntax", "-c", "cpus"),
        ("submission.log_syntax", "--output=log", "directory"),
        ("submission.name_syntax", "--job-name", "name"),
        ("submission.name_syntax", "--job-name={name}-{first}", "first"),
        ("submission.cancel_command", [], "cancel command"),
        ("submission.cancel_command", ["cancel", ""], "cancel command"),
        ("submission.cancel_command", ["cancel", "--all"], "name"),
        ("submission.cancel_command", ["cancel", "--job={job}"], "job"),
        ("placement.account", "", "account"),
        ("placement.accelerators_per_node", -1, "negative"),
        ("placement.cpus_per_task", 0, "positive"),
        ("storage.scratch_root", "scratch", "absolute"),
        ("storage.cache_root", "", "cache root"),
        ("environment.preamble", [""], "preamble line"),
        ("interpreter.command", [], "interpreter command"),
        ("interpreter.command", ["python", ""], "interpreter command"),
        ("transport.host", "", "host"),
        ("transport.command", [], "transport command"),
    ],
)
def test_a_site_value_that_cannot_work_is_refused_at_resolution(
    tmp_path: Path, path: str, value: JsonValue, detail: str
) -> None:
    document = _edited(_example_document(), path, value)

    with pytest.raises(ConfigValueError) as refused:
        _resolve_document(tmp_path, document)

    assert refused.value.path == path
    assert detail in refused.value.reason


def test_the_example_profile_spells_the_placement_directives() -> None:
    profile = resolve_site_profile({SITE_PROFILE_VARIABLE: str(EXAMPLE_SITE_PROFILE)})

    assert profile.submission.placement_directives(profile.placement) == (
        "--account=placeholder-account",
        "--partition=placeholder-partition",
        "--gres=placeholder-accelerator:1",
    )


def test_a_placement_without_accelerators_spells_no_accelerator_directive() -> None:
    profile = resolve_site_profile({SITE_PROFILE_VARIABLE: str(EXAMPLE_SITE_PROFILE)})
    placement = SitePlacement(
        account="second-account",
        partition="second-partition",
        accelerator="second-accelerator",
        accelerators_per_node=0,
        cpus_per_task=1,
    )

    assert profile.submission.placement_directives(placement) == (
        "--account=second-account",
        "--partition=second-partition",
    )


def test_the_example_profile_spells_the_resource_directives() -> None:
    profile = resolve_site_profile({SITE_PROFILE_VARIABLE: str(EXAMPLE_SITE_PROFILE)})

    assert profile.submission.resource_directives(
        profile.placement,
        minutes=95,
        megabytes=2048,
        directory="/placeholder/submission",
    ) == (
        "--time=95",
        "--mem=2048",
        "--cpus-per-task=4",
        "--output=/placeholder/submission/%A_%a.log",
    )


def test_the_array_directive_places_the_range_and_the_throttle() -> None:
    profile = resolve_site_profile({SITE_PROFILE_VARIABLE: str(EXAMPLE_SITE_PROFILE)})
    submission = profile.submission
    directive = submission.array_directive(first=0, last=41, throttle=8)

    assert directive.startswith(submission.array_syntax.split("{", 1)[0])
    assert re.search(r"\b0\b", directive)
    assert re.search(r"\b41\b", directive)
    assert re.search(r"\b8\b", directive)
    assert directive != submission.array_directive(first=0, last=41, throttle=9)
    assert directive != submission.array_directive(first=1, last=41, throttle=8)


def test_the_example_profile_spells_the_job_name_and_its_cancellation() -> None:
    profile = resolve_site_profile({SITE_PROFILE_VARIABLE: str(EXAMPLE_SITE_PROFILE)})
    submission = profile.submission

    assert submission.name_directive(name="lofi-0123") == "--job-name=lofi-0123"
    assert submission.cancellation(name="lofi-0123") == (
        "placeholder-cancel",
        "--name=lofi-0123",
    )


def test_a_quota_pattern_must_capture_the_amount_it_reads() -> None:
    quota = SiteQuota(
        command=("site-quota",),
        used_pattern=r"used (?P<used>\d+)",
        unit_bytes=1024,
        limit_bytes=4096,
    )

    assert quota.used_pattern == r"used (?P<used>\d+)"

    with pytest.raises(dx.ValidationError, match="no group named used"):
        SiteQuota(
            command=("site-quota",),
            used_pattern=r"held (\d+)",
            unit_bytes=1,
            limit_bytes=1,
        )

    with pytest.raises(dx.ValidationError, match="does not compile"):
        SiteQuota(
            command=("site-quota",), used_pattern="(", unit_bytes=1, limit_bytes=1
        )


def test_an_artifact_class_costs_a_finite_nonnegative_number_of_hours() -> None:
    with pytest.raises(dx.ValidationError, match="negative"):
        ArtifactClass(name="fits", pattern="fits", regeneration_hours=-1.0)


def test_an_allocation_admits_its_share_of_its_hours() -> None:
    allocation = SiteAllocation(accelerator_hours=1000.0, share=0.6)

    assert allocation.admitted_hours == 600.0

    for share in (0.0, 1.5):
        with pytest.raises(dx.ValidationError, match="outside"):
            SiteAllocation(accelerator_hours=1000.0, share=share)


@pytest.mark.parametrize("field", ["scratch_root", "cache_root"])
@pytest.mark.parametrize(
    "root", ["/scratch/run", r"C:\scratch\run", r"\\server\share\run"]
)
def test_storage_accepts_absolute_paths_for_each_host_platform(
    field: str, root: str
) -> None:
    document = _edited(_example_document(), f"storage.{field}", root)
    profile = SiteProfile.model_validate(document)

    assert getattr(profile.storage, field) == root


@pytest.mark.parametrize("field", ["scratch_root", "cache_root"])
@pytest.mark.parametrize("root", ["scratch/run", r"C:scratch\run", r"\scratch\run"])
def test_storage_refuses_paths_that_depend_on_a_working_directory(
    tmp_path: Path, field: str, root: str
) -> None:
    document = _edited(_example_document(), f"storage.{field}", root)

    with pytest.raises(ConfigValueError, match="not absolute"):
        _resolve_document(tmp_path, document)


@pytest.mark.parametrize("root", ["C:/scratch/run", "//server/share/run"])
def test_a_site_file_resolves_windows_storage_paths(tmp_path: Path, root: str) -> None:
    document = _edited(_example_document(), "storage.scratch_root", root)
    profile = _resolve_document(tmp_path, document)

    assert profile.storage.scratch_root == root
