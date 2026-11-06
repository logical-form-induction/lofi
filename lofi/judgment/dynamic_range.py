"""The dynamic range gate over the published root log odds.

A recorder reads the mass pair published under the declared id at every
item's sentence-spanning cell, step by step, and keeps the log odds of each
item. A gate reads the log odds of a whole corpus against a criterion that
was registered in advance and refuses a distribution that fails it.
"""

import json
from typing import TYPE_CHECKING, final, override

import torch

from lofi.judgment._spec import (
    DynamicRangeCriterion,
    DynamicRangeReport,
    decode_dynamic_range_criterion,
    make_dynamic_range_report,
)

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from torch import Tensor

__all__ = [
    "DynamicRangeError",
    "DynamicRangeGate",
    "read_criterion",
]

_QUARTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
"""The points of the distribution a report states, as fractions of the items."""


class DynamicRangeError(ValueError):
    """Reports a root log odds distribution that fails the registered criterion.

    Parameters
    ----------
    report : DynamicRangeReport
        The distribution and the criterion it fails.
    """

    def __init__(self, report: DynamicRangeReport) -> None:
        clauses = "; ".join(report.failures)
        super().__init__(
            f"the declared readout's scores over {report.items} items under the "
            f"inventory {report.inventory!r} fail the criterion registered at "
            f"{report.criterion.registered_at}: {clauses}"
        )
        self.report = report


def read_criterion(path: Path, /) -> DynamicRangeCriterion:
    """Return the criterion registered in a JSON file, validated as one.

    Parameters
    ----------
    path : Path
        The file the criterion was registered in.

    Raises
    ------
    OSError
        If the file cannot be read.
    ValueError
        If the file is not JSON.
    """
    return decode_dynamic_range_criterion(json.loads(path.read_text(encoding="utf-8")))


@final
class DynamicRangeGate:
    """Reads a root log odds distribution against a criterion registered in advance.

    Parameters
    ----------
    criterion : DynamicRangeCriterion
        The registered criterion.
    """

    __slots__ = ("_criterion",)

    def __init__(self, criterion: DynamicRangeCriterion) -> None:
        self._criterion = criterion

    @property
    def criterion(self) -> DynamicRangeCriterion:
        """Return the registered criterion."""
        return self._criterion

    def report(
        self, odds: Tensor, /, *, inventory: str, observed_at: datetime
    ) -> DynamicRangeReport:
        """Return the distribution of the log odds read against the criterion.

        The report is validated as it is built, so an unnamed inventory, an
        observation with no UTC offset and an observation earlier than the
        criterion's registration are refused.

        Parameters
        ----------
        odds : Tensor
            The log odds of every item, in any order.
        inventory : str
            Name of the declared inventory the scores were computed under.
        observed_at : datetime
            When the distribution was observed, with a UTC offset.

        Raises
        ------
        ValueError
            If there is no item, or the log odds are not a vector.
        """
        if odds.ndim != 1:
            msg = (
                "the log-odds of a corpus are a vector, and these have rank "
                f"{odds.ndim}"
            )
            raise ValueError(msg)

        if odds.shape[0] == 0:
            msg = "there is no item, so there is no distribution to report"
            raise ValueError(msg)

        held = odds.detach().to(torch.float64)
        points = torch.quantile(held, held.new_tensor(_QUARTILES))
        minimum, lower, median, upper, maximum = (float(point) for point in points)
        at_ceiling = (held >= self._criterion.ceiling).to(torch.float64).mean()

        return make_dynamic_range_report(
            self._criterion,
            inventory=inventory,
            observed_at=observed_at.isoformat(),
            items=int(held.shape[0]),
            minimum=minimum,
            lower_quartile=lower,
            median=median,
            upper_quartile=upper,
            maximum=maximum,
            ceiling_fraction=float(at_ceiling),
        )

    def admit(
        self, odds: Tensor, /, *, inventory: str, observed_at: datetime
    ) -> DynamicRangeReport:
        """Return the report of a distribution that passes, refusing one that fails.

        The report is validated as it is built, so an unnamed inventory, an
        observation with no UTC offset and an observation earlier than the
        criterion's registration are refused.

        Parameters
        ----------
        odds : Tensor
            The log odds of every item, in any order.
        inventory : str
            Name of the declared inventory the scores were computed under.
        observed_at : datetime
            When the distribution was observed, with a UTC offset.

        Raises
        ------
        DynamicRangeError
            If the distribution fails the criterion.
        ValueError
            If there is no item, or the log odds are not a vector.
        """
        report = self.report(odds, inventory=inventory, observed_at=observed_at)

        if not report.passed:
            raise DynamicRangeError(report)

        return report

    @override
    def __repr__(self) -> str:
        """Return the criterion."""
        return f"DynamicRangeGate({self._criterion!r})"
