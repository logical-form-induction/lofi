"""Admission of a paper's experiments against the share of an allocation.

An experiment is admitted to a scheduler with the experiments it is admitted
together with, all of one paper's freeze. The group's P90 accelerator-hour
estimates are summed and must not exceed the hours the site lets one paper
hold. A headline experiment and an analysis experiment are only admitted
together, so a group holding one of them holds the other.
"""

from typing import TYPE_CHECKING, NamedTuple

from lofi.run._experiment import ComputeEstimate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.run._experiment import Experiment

__all__ = ["Admission", "AdmissionError", "admit"]

_PAIRED = frozenset({"headline", "analysis"})
"""The kinds of experiment admitted only together."""


class AdmissionError(ValueError):
    """Signals a group of experiments that cannot be admitted."""


class Admission(NamedTuple):
    """An admitted group of experiments and the hours it holds."""

    experiments: tuple[str, ...]
    p90_hours: float
    admitted_hours: float


def admit(
    experiment: Experiment,
    companions: Sequence[Experiment],
    /,
    *,
    admitted_hours: float,
) -> Admission:
    """Admit an experiment with its companions, or refuse the group.

    Parameters
    ----------
    experiment : Experiment
        The experiment being launched.
    companions : Sequence[Experiment]
        The experiments of the same freeze admitted together with it.
    admitted_hours : float
        The accelerator-hours the site lets one paper's experiments hold.

    Returns
    -------
    Admission
        The group's names, its summed P90 hours and the hours it was admitted
        against.

    Raises
    ------
    AdmissionError
        If a companion belongs to another freeze or repeats an experiment, an
        experiment of the group is unpriced, a headline or analysis
        experiment lacks its partner, or the summed P90 exceeds the admitted
        hours.
    """
    group = (experiment, *companions)
    names = tuple(member.name for member in group)

    if len(names) != len(set(names)):
        msg = f"an admitted group names one experiment twice: {names}"
        raise AdmissionError(msg)

    foreign = tuple(
        member.name for member in group if member.freeze_id != experiment.freeze_id
    )

    if foreign:
        msg = (
            f"the experiments {foreign} do not run against {experiment.freeze_id!r} "
            f"and cannot be admitted with {experiment.name!r}"
        )
        raise AdmissionError(msg)

    unpriced = tuple(
        member.name
        for member in group
        if not isinstance(member.compute, ComputeEstimate)
    )

    if unpriced:
        msg = f"the experiments {unpriced} are unpriced and cannot be admitted"
        raise AdmissionError(msg)

    kinds = {member.role for member in group}

    if kinds & _PAIRED and not kinds >= _PAIRED:
        missing = sorted(_PAIRED - kinds)
        msg = (
            f"a headline experiment and an analysis experiment are admitted only "
            f"together, and the group {names} has no {missing[0]} experiment"
        )
        raise AdmissionError(msg)

    p90 = sum(
        member.compute.p90_hours
        for member in group
        if isinstance(member.compute, ComputeEstimate)
    )

    if p90 > admitted_hours:
        msg = (
            f"the group {names} needs {p90} accelerator-hours at P90, above the "
            f"{admitted_hours} one paper may hold"
        )
        raise AdmissionError(msg)

    return Admission(names, p90, admitted_hours)
