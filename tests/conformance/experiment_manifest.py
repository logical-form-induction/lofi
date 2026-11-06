from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn

import pytest

from lofi._checks import LawViolationError
from lofi.run import ManifestStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.run import ExperimentManifest

__all__ = [
    "EXPERIMENT_MANIFEST_CONFORMANCE_LAWS",
    "EXPERIMENT_MANIFEST_LAWS",
    "ExperimentManifestConformance",
    "ManifestFixture",
    "an_unclosed_manifest_is_distinguishable_from_a_closed_one",
    "manifest_is_opened_at_run_start_and_closed_at_run_end",
    "nothing_in_the_manifest_path_raises_to_report_success",
]


class ManifestFixture(NamedTuple):
    """Configures experiment-manifest lifecycle laws."""

    open: Callable[[], ExperimentManifest]
    close: Callable[[ExperimentManifest, ManifestStatus], ExperimentManifest]


def _fail_manifest_law(detail: str, *, law: str) -> NoReturn:
    """Raise a named manifest-law violation."""
    message = f"{law}: {detail}"

    raise LawViolationError(message)


def manifest_is_opened_at_run_start_and_closed_at_run_end(
    fixture: ManifestFixture,
) -> None:
    """Check the running-to-completed manifest transition."""
    law = "manifest_is_opened_at_run_start_and_closed_at_run_end"
    opened = fixture.open()

    if opened.status is not ManifestStatus.RUNNING:
        _fail_manifest_law(f"the manifest opened as {opened.status!r}", law=law)

    closed = fixture.close(opened, ManifestStatus.COMPLETED)

    if closed.status is not ManifestStatus.COMPLETED:
        _fail_manifest_law(f"the manifest closed as {closed.status!r}", law=law)


def an_unclosed_manifest_is_distinguishable_from_a_closed_one(
    fixture: ManifestFixture,
) -> None:
    """Check that open and closed manifests remain distinct."""
    law = "an_unclosed_manifest_is_distinguishable_from_a_closed_one"
    opened = fixture.open()
    completed = fixture.close(opened, ManifestStatus.COMPLETED)
    killed = fixture.close(opened, ManifestStatus.KILLED)
    manifests = (opened, completed, killed)

    if len(set(manifests)) != len(manifests) or len(
        {manifest.status for manifest in manifests}
    ) != len(manifests):
        _fail_manifest_law(
            "running, completed and killed manifests are indistinguishable", law=law
        )


def nothing_in_the_manifest_path_raises_to_report_success(
    fixture: ManifestFixture,
) -> None:
    """Check that completion is returned as a manifest value."""
    law = "nothing_in_the_manifest_path_raises_to_report_success"

    try:
        closed = fixture.close(fixture.open(), ManifestStatus.COMPLETED)

    except Exception as error:
        _fail_manifest_law(
            f"completion raised {type(error).__name__}: {error}", law=law
        )

    if closed.status is not ManifestStatus.COMPLETED:
        _fail_manifest_law("completion returned no completed manifest", law=law)


EXPERIMENT_MANIFEST_LAWS: Final = (
    manifest_is_opened_at_run_start_and_closed_at_run_end,
    an_unclosed_manifest_is_distinguishable_from_a_closed_one,
    nothing_in_the_manifest_path_raises_to_report_success,
)
"""Experiment-manifest laws in conformance order."""


class ExperimentManifestConformance:
    pytestmark = pytest.mark.conformance

    def test_manifest_is_opened_at_run_start_and_closed_at_run_end(
        self, manifest_fixture: ManifestFixture
    ) -> None:
        manifest_is_opened_at_run_start_and_closed_at_run_end(manifest_fixture)

    def test_an_unclosed_manifest_is_distinguishable_from_a_closed_one(
        self, manifest_fixture: ManifestFixture
    ) -> None:
        an_unclosed_manifest_is_distinguishable_from_a_closed_one(manifest_fixture)

    def test_nothing_in_the_manifest_path_raises_to_report_success(
        self, manifest_fixture: ManifestFixture
    ) -> None:
        nothing_in_the_manifest_path_raises_to_report_success(manifest_fixture)


EXPERIMENT_MANIFEST_CONFORMANCE_LAWS: Final = (
    ExperimentManifestConformance.test_manifest_is_opened_at_run_start_and_closed_at_run_end,
    ExperimentManifestConformance.test_an_unclosed_manifest_is_distinguishable_from_a_closed_one,
    ExperimentManifestConformance.test_nothing_in_the_manifest_path_raises_to_report_success,
)
