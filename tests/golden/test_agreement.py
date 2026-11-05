import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
import torch

import lofi.nn._values as nn_values
import lofi.nn.loss._ports as loss_ports
import lofi.nn.loss._suite as loss_suite
import lofi.nn.loss._values as loss_values
from lofi._types import ReadoutId
from lofi.deduce import SpanIndex

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from types import ModuleType
    from typing import SupportsFloat

    from tests.conftest import ChecksReload

    from lofi._types import JsonValue

_GOLDEN: Final = Path(__file__).parent / "values.json"

_PLACES: Final = 6


class Entry(NamedTuple):
    """A deterministic entry point recorded by the golden test.

    Parameters
    ----------
    name
        The key used in the recording.
    modules
        Modules to reload, in order, when the debug-check setting changes.
    run
        The deterministic computation to record.
    """

    name: str
    modules: tuple[ModuleType, ...]
    run: Callable[[], JsonValue]


def _strings(names: Iterable[str]) -> list[JsonValue]:
    """Return names in recording order.

    Parameters
    ----------
    names
        The names to record.

    Returns
    -------
    list
        The names as a list.
    """
    return list(names)


def _numbers(values: Iterable[SupportsFloat]) -> list[JsonValue]:
    """Return numbers rounded to the recording precision.

    Parameters
    ----------
    values
        The numbers to record.

    Returns
    -------
    list
        Rounded floating-point values in input order.
    """
    return [round(float(value), _PLACES) for value in values]


def _scalars(named: Mapping[str, SupportsFloat]) -> dict[str, JsonValue]:
    """Return named scalars rounded in key order.

    Parameters
    ----------
    named
        Scalars keyed by their recorded names.

    Returns
    -------
    dict
        Rounded floating-point values in sorted key order.
    """
    return {name: round(float(value), _PLACES) for name, value in sorted(named.items())}


def _log_distribution() -> dict[str, JsonValue]:
    """Return the recorded properties of a masked normalized distribution.

    Returns
    -------
    dict
        Its axes, mask axes, normalization axis, split count, and live peak weights.
    """
    half = -0.6931471805599453
    tensor = torch.tensor(
        [[half, half], [0.0, -torch.inf], [-torch.inf, -torch.inf]],
    )
    mask = torch.tensor([True, True, False])
    value = nn_values.LogDistribution(tensor, ("item", "split"), mask, "split")
    live = value.tensor[mask].amax(dim=value.axis("split"))

    return {
        "axes": _strings(value.axes),
        "mask_axes": _strings(value.mask_axes),
        "normalized_over": value.normalized_over,
        "splits": value.size("split"),
        "peak": _numbers(live),
    }


def _loss_term() -> dict[str, JsonValue]:
    """Return the recorded properties of applicable and inapplicable loss terms.

    Returns
    -------
    dict
        Values, counts, residual mass, and diagnostic names for both terms.
    """
    term = loss_ports.LossTerm(
        torch.tensor(1.5),
        4,
        residual=torch.tensor(0.25),
        diagnostics={"mass": torch.tensor(0.75)},
    )
    residual = term.residual
    inapplicable = loss_ports.LossTerm.inapplicable(torch.tensor([1.0, 2.0]))

    return {
        "value": round(float(term.value), _PLACES),
        "count": term.count,
        "residual": None if residual is None else round(float(residual), _PLACES),
        "diagnostics": _strings(sorted(term.diagnostics)),
        "inapplicable_value": round(float(inapplicable.value), _PLACES),
        "inapplicable_count": inapplicable.count,
    }


def _published_bundle() -> dict[str, JsonValue]:
    """Return the recorded properties of a bundle with both chart passes.

    Returns
    -------
    dict
        Chart axes, outside-pass state, readout names, and span shape.
    """
    axes = ("item", "span")
    inside = nn_values.LogDistribution(torch.full((2, 3), -1.0), axes)
    outside = nn_values.LogDistribution(torch.full((2, 3), -2.0), axes)
    readout = nn_values.LogDistribution(torch.full((2, 3), -3.0), axes)
    outputs = loss_values.ObjectiveOutputs(
        spans=SpanIndex(2),
        chart=loss_values.CellChart(inside, outside),
        batch=("first", "second"),
        readouts={
            ReadoutId("span-head"): loss_values.SpanReadout(SpanIndex(2), readout)
        },
    )
    published = outputs.chart.outside
    mean = 0.0 if published is None else float(published.tensor.mean())

    return {
        "axes": _strings(outputs.chart.inside.axes),
        "outside": None if published is None else round(mean, _PLACES),
        "readouts": _strings(sorted(outputs.readouts)),
        "spans": outputs.spans.num_spans,
        "batch": _strings(outputs.batch),
    }


class _Fixed:
    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self.requires = frozenset({"batch"})
        self._value = value

    def evaluate(
        self,
        outputs: loss_ports.PublishedOutputs[tuple[str, ...]],
        /,
    ) -> loss_ports.LossTerm:
        """Return a term scaled by the number of batch items.

        Parameters
        ----------
        outputs
            The published outputs for the step.

        Returns
        -------
        lofi.nn.loss._ports.LossTerm
            The scaled term with its scale diagnostic.
        """
        items = len(outputs.batch)

        return loss_ports.LossTerm(
            torch.tensor(self._value * items),
            items,
            diagnostics={"scale": torch.tensor(self._value)},
        )


def _objective_suite() -> dict[str, JsonValue]:
    """Return the recorded result of two weighted objectives.

    Returns
    -------
    dict
        Objective order, weights, total, applicability, and reported scalars.
    """
    batch: tuple[str, ...] = ("first", "second")
    inside = nn_values.LogDistribution(torch.full((2, 3), -1.0), ("item", "span"))
    outputs = loss_values.ObjectiveOutputs(
        spans=SpanIndex(2),
        chart=loss_values.CellChart(inside),
        batch=batch,
    )
    suite = loss_suite.ObjectiveSuite(
        [(1.0, _Fixed("coverage", 0.5)), (0.25, _Fixed("agreement", 2.0))],
    )
    result = suite.evaluate(outputs)

    return {
        "names": _strings(suite.names),
        "weights": _numbers(suite.weights),
        "total": round(float(result.total), _PLACES),
        "applied": _strings(sorted(result.applied)),
        "diagnostics": _scalars(result.diagnostics),
    }


_ENTRIES: Final = (
    Entry("log_distribution", (nn_values,), _log_distribution),
    Entry("loss_term", (loss_ports,), _loss_term),
    Entry("published_bundle", (loss_values,), _published_bundle),
    Entry("objective_suite", (loss_suite,), _objective_suite),
)


def _both(entry: Entry, reload: ChecksReload) -> tuple[JsonValue, JsonValue]:
    """Return one entry's checked and unchecked values, in that order.

    Parameters
    ----------
    entry
        The entry point to evaluate twice.
    reload
        The function that changes the debug-check setting.

    Returns
    -------
    tuple
        The checked value followed by the unchecked value.
    """
    reload(*entry.modules, on=True)
    checked = entry.run()

    reload(*entry.modules, on=False)
    unchecked = entry.run()

    return checked, unchecked


@pytest.fixture(scope="session")
def recorded() -> dict[str, JsonValue]:
    """Return the stored golden values keyed by entry-point name.

    Returns
    -------
    dict
        The decoded golden recording.
    """
    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


@pytest.mark.parametrize("entry", _ENTRIES, ids=[entry.name for entry in _ENTRIES])
def test_the_checked_and_unchecked_runs_agree(
    entry: Entry,
    reload_with_checks: ChecksReload,
    recorded: dict[str, JsonValue],
) -> None:
    checked, unchecked = _both(entry, reload_with_checks)

    assert checked == unchecked
    assert checked == recorded[entry.name]


def test_the_recording_covers_the_entry_points(
    recorded: dict[str, JsonValue],
) -> None:
    assert sorted(recorded) == sorted(entry.name for entry in _ENTRIES)


if __name__ == "__main__":
    _GOLDEN.write_text(
        json.dumps(
            {entry.name: entry.run() for entry in _ENTRIES},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
