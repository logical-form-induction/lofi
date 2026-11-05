"""Site profiles and the launcher protocol that submits points under one."""

import enum
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from string import Formatter
from typing import TYPE_CHECKING, Final, Protocol, override, runtime_checkable

import didactic.api as dx

from lofi._checks import (
    ensure_finite,
    ensure_name,
    ensure_non_negative,
    ensure_positive,
)
from lofi.spec import MissingSettingError, SpecLoader

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.run._boundary.sweep import Sweep, SweepPoint

__all__ = [
    "SITE_PROFILE_VARIABLE",
    "ArtifactClass",
    "Launcher",
    "RunHandle",
    "RunStatus",
    "SiteAllocation",
    "SiteEnvironment",
    "SiteInterpreter",
    "SitePlacement",
    "SiteProfile",
    "SiteQuota",
    "SiteStorage",
    "SiteSubmission",
    "SiteTransport",
    "SweepExpander",
    "resolve_site_profile",
]

SITE_PROFILE_VARIABLE: Final = "LOFI_SITE_PROFILE"
"""The environment variable naming the site profile file."""


class RunStatus(enum.StrEnum):
    """Reports the current launcher state of a run."""

    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunHandle:
    """Identifies one submitted sweep point at a launcher backend."""

    __slots__ = ("_backend_id", "_point_id")

    def __init__(self, point_id: str, backend_id: str) -> None:
        self._point_id = ensure_name("point id", point_id)
        self._backend_id = ensure_name("backend id", backend_id)

    @property
    def point_id(self) -> str:
        """Return the submitted point identifier."""
        return self._point_id

    @property
    def backend_id(self) -> str:
        """Return the backend job identifier."""
        return self._backend_id

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another handle names the same backend job."""
        if type(other) is not RunHandle:
            return NotImplemented

        return (self._point_id, self._backend_id) == (
            other._point_id,
            other._backend_id,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over both identifiers."""
        return hash((RunHandle, self._point_id, self._backend_id))

    @override
    def __repr__(self) -> str:
        """Return both identifiers."""
        return f"RunHandle({self._point_id!r}, {self._backend_id!r})"


@runtime_checkable
class SweepExpander(Protocol):
    """Expands a sweep declaration into unique points."""

    def expand(self, sweep: Sweep, /) -> tuple[SweepPoint, ...]:
        """Return the unique points in deterministic order."""
        ...


@runtime_checkable
class Launcher(Protocol):
    """Sweep-point admission, rehearsal, submission and status reporting."""

    def admitted(self, points: Sequence[SweepPoint], /) -> tuple[SweepPoint, ...]:
        """Return the points the launcher admits for submission, in order."""
        ...

    def rehearse(self, points: Sequence[SweepPoint], /) -> str:
        """Describe what submitting the points would do, and submit nothing."""
        ...

    def submit(self, points: Sequence[SweepPoint], /) -> list[RunHandle]:
        """Submit points in order."""
        ...

    def status(self, handle: RunHandle, /) -> RunStatus:
        """Return a submitted run's status."""
        ...


def _placeholders(template: str) -> frozenset[str]:
    """Return the replacement fields named by a format template."""
    return frozenset(
        field for _, field, _, _ in Formatter().parse(template) if field is not None
    )


def _ensure_template(field: str, template: str, *, fields: frozenset[str]) -> str:
    """Return a template naming exactly the given replacement fields."""
    named = _placeholders(ensure_name(field, template))
    missing = sorted(fields - named)
    unknown = sorted(named - fields)

    if missing or unknown:
        details = []

        if missing:
            details.append(f"lacks {', '.join(missing)}")

        if unknown:
            details.append(f"names {', '.join(unknown)}")

        msg = f"the {field} {' and '.join(details)}"
        raise ValueError(msg)

    return template


def _ensure_absolute(field: str, root: str) -> str:
    """Return an absolute POSIX or Windows path."""
    named = ensure_name(field, root)
    if not (PurePosixPath(named).is_absolute() or PureWindowsPath(named).is_absolute()):
        msg = f"the {field} is not absolute: {root!r}"
        raise ValueError(msg)

    return root


def _ensure_words(field: str, words: tuple[str, ...]) -> tuple[str, ...]:
    """Return a nonempty argument vector of nonempty words."""
    if not words:
        msg = f"the {field} is empty"
        raise ValueError(msg)

    return tuple(ensure_name(field, word) for word in words)


class SitePlacement(dx.Model, extra="forbid"):
    """Declares where a site runs a job and what accelerators a node has.

    Attributes
    ----------
    account : str
        The account charged for the job.
    partition : str
        The partition or queue the job is placed in.
    accelerator : str
        The accelerator kind a node offers, as the scheduler names it.
    accelerators_per_node : int
        The number of accelerators a job takes on each node.
    cpus_per_task : int
        The number of processor cores one array task takes.
    """

    account: str
    partition: str
    accelerator: str
    accelerators_per_node: int
    cpus_per_task: int

    @dx.validates("account")
    @classmethod
    def _account_is_named(cls, account: str) -> str:
        """Validate the account."""
        return ensure_name("account", account)

    @dx.validates("partition")
    @classmethod
    def _partition_is_named(cls, partition: str) -> str:
        """Validate the partition."""
        return ensure_name("partition", partition)

    @dx.validates("accelerator")
    @classmethod
    def _accelerator_is_named(cls, accelerator: str) -> str:
        """Validate the accelerator kind."""
        return ensure_name("accelerator", accelerator)

    @dx.validates("accelerators_per_node")
    @classmethod
    def _accelerator_count_is_countable(cls, count: int) -> int:
        """Validate the per-node accelerator count."""
        return ensure_non_negative("accelerators per node", count)

    @dx.validates("cpus_per_task")
    @classmethod
    def _cpu_count_is_positive(cls, count: int) -> int:
        """Validate the per-task core count."""
        return ensure_positive("cpus per task", count)


class SiteSubmission(dx.Model, extra="forbid"):
    """Declares how a site's scheduler takes an array submission.

    Attributes
    ----------
    command : str
        The command that submits a script.
    directive_prefix : str
        The text a scheduler directive line starts with.
    array_syntax : str
        The array range spelling, a template over ``first`` and ``last``.
    throttle_syntax : str
        The concurrent-task limit spelling, a template over ``throttle``, which
        follows the formatted array syntax in one directive.
    account_syntax : str
        The charged-account spelling, a template over ``account``.
    partition_syntax : str
        The partition or queue spelling, a template over ``partition``.
    accelerator_syntax : str
        The per-node accelerator spelling, a template over ``accelerator`` and
        ``accelerators_per_node``.
    task_index_variable : str
        The environment variable the scheduler sets to a task's array index.
    walltime_syntax : str
        The per-task wall-clock limit spelling, a template over ``minutes``.
    memory_syntax : str
        The per-task memory request spelling, a template over ``megabytes``.
    cpus_syntax : str
        The per-task core request spelling, a template over ``cpus``.
    log_syntax : str
        The task log path spelling, a template over ``directory``, the
        submission's directory, which exists when the script is submitted.
    name_syntax : str
        The job name spelling, a template over ``name``, the name every
        submission of one experiment under one freeze shares.
    job_prefix : str
        The text every job name starts with, before the digest of the
        experiment and its freeze; letters, digits and hyphens only.
    cancel_command : tuple[str, ...]
        The argument vector that cancels every job of a name, whose words
        are templates naming ``name`` and nothing else, at least one of them
        naming it.
    """

    command: str
    directive_prefix: str
    array_syntax: str
    throttle_syntax: str
    account_syntax: str
    partition_syntax: str
    accelerator_syntax: str
    task_index_variable: str
    walltime_syntax: str
    memory_syntax: str
    cpus_syntax: str
    log_syntax: str
    name_syntax: str
    job_prefix: str
    cancel_command: tuple[str, ...]

    @dx.validates("job_prefix")
    @classmethod
    def _prefix_names_a_job(cls, prefix: str) -> str:
        """Return a job-name prefix of letters, digits and hyphens.

        Raises
        ------
        ValueError
            If the prefix is empty or holds any other character.
        """
        named = ensure_name("job prefix", prefix)

        if not all(character.isalnum() or character == "-" for character in named):
            msg = f"a job prefix holds letters, digits and hyphens only: {named!r}"
            raise ValueError(msg)

        return named

    @dx.validates("command")
    @classmethod
    def _command_is_named(cls, command: str) -> str:
        """Validate the submission command."""
        return ensure_name("submission command", command)

    @dx.validates("directive_prefix")
    @classmethod
    def _prefix_is_named(cls, prefix: str) -> str:
        """Validate the directive prefix."""
        return ensure_name("directive prefix", prefix)

    @dx.validates("array_syntax")
    @classmethod
    def _array_syntax_spans_a_range(cls, syntax: str) -> str:
        """Validate the array range template."""
        return _ensure_template(
            "array syntax", syntax, fields=frozenset({"first", "last"})
        )

    @dx.validates("throttle_syntax")
    @classmethod
    def _throttle_syntax_names_a_limit(cls, syntax: str) -> str:
        """Validate the throttle template."""
        return _ensure_template(
            "throttle syntax", syntax, fields=frozenset({"throttle"})
        )

    @dx.validates("account_syntax")
    @classmethod
    def _account_syntax_names_an_account(cls, syntax: str) -> str:
        """Validate the account template."""
        return _ensure_template("account syntax", syntax, fields=frozenset({"account"}))

    @dx.validates("partition_syntax")
    @classmethod
    def _partition_syntax_names_a_partition(cls, syntax: str) -> str:
        """Validate the partition template."""
        return _ensure_template(
            "partition syntax", syntax, fields=frozenset({"partition"})
        )

    @dx.validates("accelerator_syntax")
    @classmethod
    def _accelerator_syntax_names_a_kind_and_a_count(cls, syntax: str) -> str:
        """Validate the accelerator template."""
        return _ensure_template(
            "accelerator syntax",
            syntax,
            fields=frozenset({"accelerator", "accelerators_per_node"}),
        )

    @dx.validates("task_index_variable")
    @classmethod
    def _task_index_variable_is_named(cls, variable: str) -> str:
        """Validate the task index variable name."""
        return ensure_name("task index variable", variable)

    @dx.validates("walltime_syntax")
    @classmethod
    def _walltime_syntax_names_minutes(cls, syntax: str) -> str:
        """Validate the wall-clock template."""
        return _ensure_template(
            "walltime syntax", syntax, fields=frozenset({"minutes"})
        )

    @dx.validates("memory_syntax")
    @classmethod
    def _memory_syntax_names_megabytes(cls, syntax: str) -> str:
        """Validate the memory template."""
        return _ensure_template(
            "memory syntax", syntax, fields=frozenset({"megabytes"})
        )

    @dx.validates("cpus_syntax")
    @classmethod
    def _cpus_syntax_names_cpus(cls, syntax: str) -> str:
        """Validate the core-count template."""
        return _ensure_template("cpus syntax", syntax, fields=frozenset({"cpus"}))

    @dx.validates("log_syntax")
    @classmethod
    def _log_syntax_names_a_directory(cls, syntax: str) -> str:
        """Validate the log path template."""
        return _ensure_template("log syntax", syntax, fields=frozenset({"directory"}))

    @dx.validates("name_syntax")
    @classmethod
    def _name_syntax_names_a_name(cls, syntax: str) -> str:
        """Validate the job name template."""
        return _ensure_template("name syntax", syntax, fields=frozenset({"name"}))

    @dx.validates("cancel_command")
    @classmethod
    def _cancel_command_names_a_job(cls, command: tuple[str, ...]) -> tuple[str, ...]:
        """Validate the cancel command's words and the job name they spell."""
        words = _ensure_words("cancel command", command)
        named = frozenset(field for word in words for field in _placeholders(word))
        unknown = sorted(named - {"name"})

        if unknown:
            msg = f"the cancel command names {', '.join(unknown)}"
            raise ValueError(msg)

        if not named:
            msg = "the cancel command lacks name"
            raise ValueError(msg)

        return words

    def name_directive(self, *, name: str) -> str:
        """Return the directive text naming a job.

        Parameters
        ----------
        name : str
            The job name.

        Returns
        -------
        str
            The formatted name syntax, without the directive prefix.
        """
        return self.name_syntax.format(name=name)

    def cancellation(self, *, name: str) -> tuple[str, ...]:
        """Return the argument vector that cancels every job of a name.

        Parameters
        ----------
        name : str
            The job name.

        Returns
        -------
        tuple[str, ...]
            The cancel command with the name formatted into each word.
        """
        return tuple(word.format(name=name) for word in self.cancel_command)

    def resource_directives(
        self,
        placement: SitePlacement,
        /,
        *,
        minutes: int,
        megabytes: int,
        directory: str,
    ) -> tuple[str, ...]:
        """Return the directive texts requesting one task's resources.

        Parameters
        ----------
        placement : SitePlacement
            The placement the same site profile declares, whose core count
            each task takes.
        minutes : int
            The wall-clock minutes one task is granted.
        megabytes : int
            The memory one task is granted, in mebibytes.
        directory : str
            The submission's directory, which task logs are written under.

        Returns
        -------
        tuple[str, ...]
            The wall-clock, memory, core and log directives, in that order,
            without the directive prefix.
        """
        return (
            self.walltime_syntax.format(minutes=minutes),
            self.memory_syntax.format(megabytes=megabytes),
            self.cpus_syntax.format(cpus=placement.cpus_per_task),
            self.log_syntax.format(directory=directory),
        )

    def array_directive(self, *, first: int, last: int, throttle: int) -> str:
        """Return the directive text for an array of tasks.

        Parameters
        ----------
        first : int
            The first task index.
        last : int
            The last task index, inclusive.
        throttle : int
            The number of tasks the scheduler may run at once.

        Returns
        -------
        str
            The formatted array syntax followed by the formatted throttle
            syntax, without the directive prefix.
        """
        span = self.array_syntax.format(first=first, last=last)
        limit = self.throttle_syntax.format(throttle=throttle)

        return span + limit

    def placement_directives(self, placement: SitePlacement, /) -> tuple[str, ...]:
        """Return the directive texts for a placement, without the prefix.

        Parameters
        ----------
        placement : SitePlacement
            The placement the same site profile declares.

        Returns
        -------
        tuple[str, ...]
            The account and partition directives, followed by the accelerator
            directive when a node takes accelerators.
        """
        directives = [
            self.account_syntax.format(account=placement.account),
            self.partition_syntax.format(partition=placement.partition),
        ]

        if placement.accelerators_per_node:
            directives.append(
                self.accelerator_syntax.format(
                    accelerator=placement.accelerator,
                    accelerators_per_node=placement.accelerators_per_node,
                )
            )

        return tuple(directives)


class SiteQuota(dx.Model, extra="forbid"):
    """Declares how a site reports the storage its scratch root holds, and the cap.

    Attributes
    ----------
    command : tuple[str, ...]
        The argument vector that prints the site's quota report.
    used_pattern : str
        A regular expression with one group, ``used``, matching the amount
        the report states is held.
    unit_bytes : int
        The bytes one unit of the reported amount stands for.
    limit_bytes : int
        The bytes reclamation keeps the scratch root under.
    """

    command: tuple[str, ...]
    used_pattern: str
    unit_bytes: int
    limit_bytes: int

    @dx.validates("command")
    @classmethod
    def _command_has_words(cls, command: tuple[str, ...]) -> tuple[str, ...]:
        """Validate the quota command."""
        return _ensure_words("quota command", command)

    @dx.validates("used_pattern")
    @classmethod
    def _pattern_names_its_amount(cls, pattern: str) -> str:
        """Validate that the pattern compiles and captures the held amount."""
        try:
            compiled = re.compile(pattern)

        except re.error as refused:
            msg = f"the quota pattern does not compile: {refused}"
            raise ValueError(msg) from refused

        if "used" not in compiled.groupindex:
            msg = f"the quota pattern captures no group named used: {pattern!r}"
            raise ValueError(msg)

        return pattern

    @dx.validates("unit_bytes", "limit_bytes")
    @classmethod
    def _bytes_are_positive(cls, value: int) -> int:
        """Validate a positive byte count."""
        return ensure_positive("a quota byte count", value)


class ArtifactClass(dx.Model, extra="forbid"):
    """Declares one kind of file a point's directory holds and what remaking it costs.

    Attributes
    ----------
    name : str
        The name reclamation reports the class under.
    pattern : str
        The glob, relative to a point's directory, its files match.
    regeneration_hours : float
        The accelerator-hours remaking one point's files of the class costs;
        reclamation removes the cheapest classes first.
    """

    name: str
    pattern: str
    regeneration_hours: float

    @dx.validates("name", "pattern")
    @classmethod
    def _is_named(cls, value: str) -> str:
        """Validate a nonblank name or glob."""
        return ensure_name("an artifact class field", value)

    @dx.validates("regeneration_hours")
    @classmethod
    def _cost_is_finite(cls, hours: float) -> float:
        """Validate a finite, nonnegative cost."""
        checked = ensure_finite("a regeneration cost", hours)

        if checked < 0:
            msg = f"a regeneration cost is negative: {checked}"
            raise ValueError(msg)

        return checked


class SiteStorage(dx.Model, extra="forbid"):
    """Declares the filesystem roots a site provides and how they are kept in quota.

    Attributes
    ----------
    scratch_root : str
        The absolute path run outputs are written under.
    cache_root : str
        The absolute path downloaded and derived data is kept under.
    quota : SiteQuota
        How the scratch root's holdings are reported, and the cap
        reclamation keeps them under.
    artifacts : tuple[ArtifactClass, ...]
        The reclaimable files of a completed point, by class; files no class
        matches are never reclaimed.
    """

    scratch_root: str
    cache_root: str
    quota: SiteQuota
    artifacts: tuple[ArtifactClass, ...]

    @dx.validates("scratch_root")
    @classmethod
    def _scratch_root_is_absolute(cls, root: str) -> str:
        """Validate the scratch root."""
        return _ensure_absolute("scratch root", root)

    @dx.validates("cache_root")
    @classmethod
    def _cache_root_is_absolute(cls, root: str) -> str:
        """Validate the cache root."""
        return _ensure_absolute("cache root", root)

    @dx.validates("artifacts")
    @classmethod
    def _classes_are_distinct(
        cls, artifacts: tuple[ArtifactClass, ...]
    ) -> tuple[ArtifactClass, ...]:
        """Validate that no two artifact classes share a name."""
        names = tuple(artifact.name for artifact in artifacts)

        if len(names) != len(set(names)):
            msg = f"artifact classes repeat a name: {names}"
            raise ValueError(msg)

        return artifacts


class SiteAllocation(dx.Model, extra="forbid"):
    """Declares the accelerator-hours a site grants and the share one paper may hold.

    Attributes
    ----------
    accelerator_hours : float
        The accelerator-hours the allocation grants.
    share : float
        The largest fraction of the allocation one paper's admitted
        experiments may hold at once, in ``(0, 1]``.
    """

    accelerator_hours: float
    share: float

    @dx.validates("accelerator_hours")
    @classmethod
    def _hours_are_positive(cls, hours: float) -> float:
        """Validate a positive, finite allocation."""
        checked = ensure_finite("an allocation", hours)

        if checked <= 0:
            msg = f"an allocation grants no hours: {checked}"
            raise ValueError(msg)

        return checked

    @dx.validates("share")
    @classmethod
    def _share_is_a_fraction(cls, share: float) -> float:
        """Validate a share in the half-open unit interval."""
        checked = ensure_finite("an admission share", share)

        if not 0 < checked <= 1:
            msg = f"an admission share lies outside (0, 1]: {checked}"
            raise ValueError(msg)

        return checked

    @property
    def admitted_hours(self) -> float:
        """Return the accelerator-hours one paper's experiments may hold."""
        return self.accelerator_hours * self.share


class SiteEnvironment(dx.Model, extra="forbid"):
    """Declares the shell lines a site runs before the interpreter.

    Attributes
    ----------
    preamble : tuple[str, ...]
        Lines such as module loads and exports, in the order they run; empty
        when the site needs none.
    """

    preamble: tuple[str, ...]

    @dx.validates("preamble")
    @classmethod
    def _lines_are_present(cls, lines: tuple[str, ...]) -> tuple[str, ...]:
        """Validate the preamble lines."""
        return tuple(ensure_name("preamble line", line) for line in lines)


class SiteInterpreter(dx.Model, extra="forbid"):
    """Declares how a site runs Python.

    Attributes
    ----------
    command : tuple[str, ...]
        The argument vector that runs the interpreter, with any container
        wrapper spelled out in full.
    """

    command: tuple[str, ...]

    @dx.validates("command")
    @classmethod
    def _command_has_words(cls, command: tuple[str, ...]) -> tuple[str, ...]:
        """Validate the interpreter command."""
        return _ensure_words("interpreter command", command)


class SiteTransport(dx.Model, extra="forbid"):
    """Declares how a remote host is reached.

    Attributes
    ----------
    host : str
        The name the transport is given to reach the host.
    command : tuple[str, ...]
        The argument vector that opens a session, to which the host and then
        the remote command are appended.
    """

    host: str
    command: tuple[str, ...]

    @dx.validates("host")
    @classmethod
    def _host_is_named(cls, host: str) -> str:
        """Validate the host."""
        return ensure_name("host", host)

    @dx.validates("command")
    @classmethod
    def _command_has_words(cls, command: tuple[str, ...]) -> tuple[str, ...]:
        """Validate the transport command."""
        return _ensure_words("transport command", command)


class SiteProfile(dx.Model, extra="forbid"):
    """Declares everything a launcher takes from a compute site.

    Every field is required, so a profile resolved from a file states each
    site value itself and the launcher reads none from anywhere else.

    Attributes
    ----------
    submission : SiteSubmission
        How the scheduler takes an array submission.
    placement : SitePlacement
        Where a job runs and what accelerators it takes.
    storage : SiteStorage
        The filesystem roots outputs and caches live under.
    environment : SiteEnvironment
        The shell lines run before the interpreter.
    interpreter : SiteInterpreter
        How Python is run.
    transport : SiteTransport
        How the host is reached from elsewhere.
    allocation : SiteAllocation
        The accelerator-hours the site grants and the share a paper may hold.
    """

    submission: SiteSubmission
    placement: SitePlacement
    storage: SiteStorage
    environment: SiteEnvironment
    interpreter: SiteInterpreter
    transport: SiteTransport
    allocation: SiteAllocation


def resolve_site_profile(environ: Mapping[str, str], /) -> SiteProfile:
    """Resolve the site profile from the file the environment names.

    Parameters
    ----------
    environ : Mapping[str, str]
        The process environment, read at ``SITE_PROFILE_VARIABLE``.

    Returns
    -------
    SiteProfile
        The validated profile.

    Raises
    ------
    MissingSettingError
        If the variable is unset, or a field the profile requires is absent
        from the file; the error names the variable or the field.
    ConfigValueError
        If a field's value is malformed or cannot work; the error names the
        field.
    UnknownSettingError
        If the file declares a field the profile does not have.
    ConfigDocumentError
        If the file is absent, does not parse or violates a whole-profile
        invariant.
    """
    named = environ.get(SITE_PROFILE_VARIABLE)

    if named is None:
        raise MissingSettingError(path=SITE_PROFILE_VARIABLE)

    return SpecLoader().compose_traced(Path(named), schema=SiteProfile).spec
