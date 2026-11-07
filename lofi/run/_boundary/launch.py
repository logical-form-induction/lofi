"""The declared request, outcomes and record of one experiment launch.

A launch request holds everything that varies what one launch produces, and
its digest names the launch's artifacts. A launch report records what the
launch did point by point and is the file ``lofi launch --out`` writes.
"""

from __future__ import annotations

import json
from hashlib import blake2b
from typing import TYPE_CHECKING, Self

import didactic.api as dx

from lofi.run._boundary.experiment import LaunchSpec
from lofi.run._boundary.site import RunStatus, SiteProfile

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["LaunchReport", "LaunchRequest", "PointOutcome"]


class LaunchRequest(dx.Model, extra="forbid"):
    """Everything that varies what one launch produces.

    Attributes
    ----------
    experiment : str
        The experiment launched, by name.
    freeze_id : str
        The freeze the experiment is launched from.
    experiment_seed : int
        The root seed every point's run seed is derived from.
    launch : LaunchSpec
        The launcher declaration the pending points are driven through, with
        every setting it declares.
    root : str
        The store root the launch's freeze addresses its run directories under.
    points : tuple[str, ...]
        The pending point identifiers, in submission order.
    program : tuple[str, ...]
        The argument vector, after the site interpreter, that runs one task.
    profile : SiteProfile | None
        The site the launch is submitted to, or None for a launch that runs
        in the calling process.
    """

    experiment: str
    freeze_id: str
    experiment_seed: int
    launch: LaunchSpec
    root: str
    points: tuple[str, ...]
    program: tuple[str, ...] = ()
    profile: SiteProfile | None = None

    __axioms__ = (
        dx.axiom("len(root) > 0", message="a launch names the root it writes under"),
        dx.axiom("len(experiment) > 0", message="a launch names its experiment"),
        dx.axiom("len(freeze_id) > 0", message="a launch names its freeze"),
        dx.axiom("experiment_seed >= 0", message="an experiment root counts from zero"),
    )

    def digest(self) -> str:
        """Return the digest the launch's artifacts are named by.

        The digest is ``blake2b`` over the request's canonical JSON: every
        declared field, keys sorted, no whitespace. The document comes from
        the declaration rather than from a list written out here, so a field
        added to the request varies the digest on its own.

        Returns
        -------
        str
            Thirty-two hexadecimal digits.
        """
        document = json.loads(self.model_dump_json())
        canonical = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

        return blake2b(canonical.encode(), digest_size=16).hexdigest()


class PointOutcome(dx.Model, extra="forbid"):
    """What one point's run directory held when the launch reached it.

    Attributes
    ----------
    point_id : str
        The identifier of the point's resolved configuration and seed.
    seed : int
        The seed the point was expanded with.
    status : RunStatus
        The run status the store records, read through the backend for a
        point it was given.
    directory : str
        The run directory the point is addressed by.
    cached : bool
        Whether the store settled the point without the backend running it.
    """

    point_id: str
    seed: int
    status: RunStatus
    directory: str
    cached: bool

    __axioms__ = (dx.axiom("seed >= 0", message="a seed counts from zero"),)

    @dx.validates("point_id", "directory")
    @classmethod
    def _is_present(cls, value: str) -> str:
        """Return a nonempty identity string.

        Parameters
        ----------
        value : str
            The string to check.

        Returns
        -------
        str
            The value.

        Raises
        ------
        ValueError
            If it is empty or blank.
        """
        if not value.strip():
            msg = f"a point outcome identity is empty: {value!r}"

            raise ValueError(msg)

        return value


class LaunchReport(dx.Model, extra="forbid"):
    """What one launch of a sweep did, point by point.

    Attributes
    ----------
    launch_id : str
        The digest of the request the launch was made under.
    launcher : str
        The launcher the pending points were driven through.
    offered : int
        How many points the sweep expanded to.
    cache_hits : int
        How many of them the store settled without running anything.
    launched : int
        How many of them the backend was given.
    outcomes : tuple[PointOutcome, ...]
        One entry per settled or launched point, settled points first, each
        half in expansion order.
    refused : tuple[str, ...]
        The points never offered, because the price record predicts their
        stack above its activation budget, in expansion order.
    halted : tuple[str, ...]
        The offered points left unsubmitted because a stop rule fired, in
        expansion order.
    stop_rules : tuple[str, ...]
        The experiment's stop rules that fired, in declared order.
    """

    launch_id: str
    launcher: str
    offered: int
    cache_hits: int
    launched: int
    outcomes: tuple[PointOutcome, ...]
    refused: tuple[str, ...] = ()
    halted: tuple[str, ...] = ()
    stop_rules: tuple[str, ...] = ()

    __axioms__ = (
        dx.axiom("offered >= 0", message="a count is not negative"),
        dx.axiom("cache_hits >= 0", message="a count is not negative"),
        dx.axiom("launched >= 0", message="a count is not negative"),
        dx.axiom("len(launch_id) > 0", message="a report names its launch"),
    )

    @dx.model_validator()
    def _counts_are_the_outcomes(self) -> Self:
        """Validate the counts against the outcomes they summarize.

        Returns
        -------
        Self
            This report after validation.

        Raises
        ------
        ValueError
            If a count disagrees with the outcomes, two outcomes name one
            point, a halted point has an outcome, or points are halted while
            no stop rule fired.
        """
        if len(self.outcomes) + len(self.halted) != self.offered:
            msg = (
                f"the report offers {self.offered} points and holds "
                f"{len(self.outcomes)} outcomes and {len(self.halted)} halted "
                f"points"
            )

            raise ValueError(msg)

        if self.halted and not self.stop_rules:
            msg = "the report halts points while no stop rule fired"

            raise ValueError(msg)

        hits = sum(outcome.cached for outcome in self.outcomes)

        if hits != self.cache_hits:
            msg = (
                f"the report counts {self.cache_hits} cache hits and holds "
                f"{hits} cached outcomes"
            )

            raise ValueError(msg)

        if self.cache_hits + self.launched + len(self.halted) != self.offered:
            msg = (
                f"{self.cache_hits} cache hits, {self.launched} launched and "
                f"{len(self.halted)} halted points are not the {self.offered} "
                f"offered"
            )

            raise ValueError(msg)

        identifiers = {outcome.point_id for outcome in self.outcomes}

        if len(identifiers) != len(self.outcomes):
            msg = "two outcomes name one point"

            raise ValueError(msg)

        if identifiers & set(self.halted):
            msg = "a halted point has an outcome"

            raise ValueError(msg)

        return self

    def write(self, path: Path) -> None:
        """Write the report as JSON.

        Parameters
        ----------
        path : Path
            The file to write, whose parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{self.model_dump_json()}\n", encoding="utf-8", newline="\n")

    @classmethod
    def read(cls, path: Path) -> Self:
        """Read and validate a report written by ``write``.

        Parameters
        ----------
        path : Path
            The file to read.

        Returns
        -------
        Self
            The validated report.

        Raises
        ------
        didactic.ValidationError
            If the file does not hold exactly a valid report.
        """
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
