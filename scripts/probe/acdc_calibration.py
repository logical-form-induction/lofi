"""Calibrates EAP-IG against budgeted ACDC over the receiver edges of a chart scorer.

Every declared cell is one clean and counterfactual pair of pooled leaves. For
each cell the script draws the chart's receiver-edge circuit, in which every
receiver edge is one graph edge targeted at its own coordinate of the scorer's
receiver site and the pruning order runs the widest receivers first and then by
ascending edge coordinate. It attributes every edge with EAP-IG, prunes the same
edges with budgeted ACDC in that order, and compares the two: the Jaccard
overlap of ACDC's retained edges with the same number of top EAP-IG edges, the
downstream faithfulness of each circuit, as a normalized share and as the raw
divergences of its circuit-only and fully corrupted outputs from the full
model's, the wall time of each estimator and a measured unit price for each.

The declaration names every choice the comparison depends on, and none has a
default: the component role whose chart scorer is calibrated, the grid's cells
with the clean and counterfactual sentence of each, the EAP-IG step count, the
ACDC threshold, either an absolute bound on a divergence increment or a fraction
of the cell's full-corruption divergence, and its trial budget, the divergence
and metric by name, the compute budget in wall seconds, and the warmup and
repeat counts each unit price is read off. ``main`` restores the trained point a
run directory holds, pools each cell's two sentences through the point's own
token encoder and subword pooler, and calibrates the declared scorer over them.
``run`` writes two files into a fresh output directory. ``manifest.json``
records the declaration, the trained point, the environment, the wall time and
the outcome. ``rows.jsonl`` holds one row per calibrated cell, with the edge
order, the retained circuits and every measured quantity. A campaign whose wall
time reaches its budget stops before its next cell and records the outcome
``over-budget``.

Run it as::

    python -m scripts.probe.acdc_calibration --run RUN --declaration DOC --out DIR
"""

import json
from argparse import ArgumentParser
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, cast

import didactic.api as dx
import torch

from lofi._checks import (
    ensure_finite,
    ensure_name,
    ensure_non_negative,
    ensure_positive,
)
from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded
from lofi.nn.policy import DevicePolicy
from lofi.nn.ports import (
    Instrumentable,
    Observable,
    Reading,
    ReceiverLocalChartScorer,
)
from lofi.probe import (
    ACDCThresholdDecl,
    PatchFrom,
    Probed,
    ProbeUnit,
    Provenance,
    PruningBudget,
    Within,
    acdc,
    attribute_edges,
    instrument,
    matched_cardinality_jaccard,
    price_unit,
    receiver_circuit,
)
from lofi.run import execution_environment, read_manifest
from lofi.run.hub import restore_trained_point
from lofi.run.probing import pooled_leaves

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from torch import Tensor

    from lofi._types import JsonValue
    from lofi.nn._values import ChartReadouts
    from lofi.probe import (
        ACDCResult,
        EdgeAttribution,
        GraphEdge,
        PruningThreshold,
        ReceiverCircuit,
        Unit,
    )
    from lofi.train import PreparedTraining

MANIFEST_NAME: Final = "manifest.json"
"""The file recording the declaration, environment, wall time and outcome."""

ROWS_NAME: Final = "rows.jsonl"
"""The file holding one row per calibrated cell."""

COMPLETED: Final = "completed"
"""The outcome of a campaign that calibrated every declared cell."""

OVER_BUDGET: Final = "over-budget"
"""The outcome of a campaign stopped by its declared compute budget."""

DECLARED_FIELDS: Final = (
    "scorer",
    "grid",
    "steps",
    "threshold",
    "maximum_trials",
    "divergence",
    "metric",
    "budget_seconds",
    "price_warmup",
    "price_repeats",
)
"""Every field a declaration names, each required."""


def _mean_absolute(reference: Tensor, trial: Tensor) -> Tensor:
    """Return the mean absolute difference between two outputs."""
    return (reference - trial).abs().mean()


def _mean_squared(reference: Tensor, trial: Tensor) -> Tensor:
    """Return the mean squared difference between two outputs."""
    return (reference - trial).square().mean()


def _log_partition(readouts: ChartReadouts) -> Tensor:
    """Return each item's log partition.

    Raises
    ------
    ValueError
        If the chart pass filled no log partition.
    """
    log_z = readouts.log_z

    if log_z is None:
        msg = "the chart pass filled no log partition to calibrate over"
        raise ValueError(msg)

    return log_z.tensor


DIVERGENCES: Final[Mapping[str, Callable[[Tensor, Tensor], Tensor]]] = {
    "mean-absolute": _mean_absolute,
    "mean-squared": _mean_squared,
}
"""The divergences a declaration may name, from clean to candidate output."""

METRICS: Final[Mapping[str, Callable[[ChartReadouts], Tensor]]] = {
    "log-partition": _log_partition,
}
"""The per-item chart outputs a declaration may name."""


class CalibratedScorer(ReceiverLocalChartScorer, Probed[Reading], Protocol):
    """A chart scorer that publishes receiver edges and admits a region."""


class CellDeclaration(NamedTuple):
    """One declared cell's name and its two sentences."""

    #: The cell's name.
    name: str

    #: The words whose metric is explained.
    clean: tuple[str, ...]

    #: The words whose receiver activations stand in for the clean ones.
    counterfactual: tuple[str, ...]


class CalibrationDeclaration(NamedTuple):
    """The declared scorer, grid and policy of one calibration campaign."""

    #: The component role of the trained point whose chart scorer is calibrated.
    scorer: str

    #: The cells, in calibration order.
    grid: tuple[CellDeclaration, ...]

    #: The EAP-IG right-endpoint step count.
    steps: int

    #: The rule giving ACDC's strict upper bound on an accepted increment.
    threshold: PruningThreshold

    #: ACDC's maximum number of candidate trials per cell.
    maximum_trials: int

    #: The name of the divergence ACDC and faithfulness compare outputs by.
    divergence: str

    #: The name of the per-item chart output both estimators read.
    metric: str

    #: The wall seconds the campaign may spend.
    budget_seconds: float

    #: Untimed runs before each unit price is timed.
    price_warmup: int

    #: Timed runs each unit price is read off.
    price_repeats: int


class CalibrationCell(NamedTuple):
    """One declared cell's clean and counterfactual pooled leaves."""

    #: The cell's name in the declared grid.
    name: str

    #: The leaves whose metric is explained.
    clean: Encoded

    #: The leaves whose receiver activations stand in for the clean ones.
    counterfactual: Encoded


def _integer(document: Mapping[str, JsonValue], field: str) -> int:
    """Return one declared integer field."""
    value = document[field]

    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"the declared {field} is not an integer: {value!r}"
        raise TypeError(msg)

    return value


def _number(document: Mapping[str, JsonValue], field: str) -> float:
    """Return one declared real field."""
    value = document[field]

    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"the declared {field} is not a number: {value!r}"
        raise TypeError(msg)

    return ensure_finite(f"the declared {field}", float(value))


def _named(
    document: Mapping[str, JsonValue], field: str, known: Mapping[str, object]
) -> str:
    """Return one declared name drawn from a closed set."""
    value = document[field]

    if not isinstance(value, str) or value not in known:
        msg = f"the declared {field} {value!r} is not one of {sorted(known)}"
        raise ValueError(msg)

    return value


class _ThresholdDeclaration(Protocol):
    """A declaration that supplies an ACDC pruning threshold."""

    def threshold(self) -> PruningThreshold:
        """Return the rule bounding an accepted divergence increment."""
        ...


def _threshold(document: Mapping[str, JsonValue]) -> PruningThreshold:
    """Return the declared ACDC threshold, absolute or relative."""
    value = document["threshold"]

    if not isinstance(value, dict):
        msg = (
            "the declared threshold is not an object naming its kind, "
            f"absolute or relative: {value!r}"
        )
        raise TypeError(msg)

    try:
        declared = ACDCThresholdDecl.model_validate(value)
    except dx.ValidationError as refused:
        msg = f"the declared threshold is not a pruning threshold: {refused}"
        raise ValueError(msg) from refused

    return cast("_ThresholdDeclaration", declared).threshold()


CELL_FIELDS: Final = ("name", "clean", "counterfactual")
"""Every field a declared cell names, each required."""


def _words(cell: str, field: str, value: JsonValue) -> tuple[str, ...]:
    """Return one declared sentence of a cell."""
    if not isinstance(value, list) or not value:
        msg = f"the {field} sentence of cell {cell!r} is not a nonempty word list"
        raise ValueError(msg)

    words: list[str] = []

    for word in value:
        if not isinstance(word, str):
            msg = f"a word of the {field} sentence of cell {cell!r} is {word!r}"
            raise TypeError(msg)

        words.append(ensure_name(f"a word of cell {cell!r}", word))

    return tuple(words)


def _cell(value: JsonValue) -> CellDeclaration:
    """Return one validated declared cell."""
    if not isinstance(value, dict):
        msg = f"a declared cell is not an object of {list(CELL_FIELDS)}: {value!r}"
        raise TypeError(msg)

    missing = sorted(set(CELL_FIELDS).difference(value))
    unknown = sorted(set(value).difference(CELL_FIELDS))

    if missing or unknown:
        msg = f"a declared cell is missing {missing} and names unknown {unknown}"
        raise ValueError(msg)

    name = value["name"]

    if not isinstance(name, str):
        msg = f"a declared cell name is not a string: {name!r}"
        raise TypeError(msg)

    held = ensure_name("a declared cell name", name)
    clean = _words(held, "clean", value["clean"])
    counterfactual = _words(held, "counterfactual", value["counterfactual"])

    if len(clean) != len(counterfactual):
        msg = (
            f"the two sentences of cell {held!r} hold {len(clean)} and "
            f"{len(counterfactual)} words, and their leaves must align word by word"
        )
        raise ValueError(msg)

    return CellDeclaration(held, clean, counterfactual)


def declaration_of(document: Mapping[str, JsonValue]) -> CalibrationDeclaration:
    """Return a validated declaration from a JSON document.

    Parameters
    ----------
    document : Mapping[str, JsonValue]
        One object naming exactly the declared fields.

    Returns
    -------
    CalibrationDeclaration
        The declaration.

    Raises
    ------
    TypeError
        If a field has the wrong type.
    ValueError
        If a field is missing or unknown, or a value is out of range.
    """
    missing = sorted(set(DECLARED_FIELDS).difference(document))
    unknown = sorted(set(document).difference(DECLARED_FIELDS))

    if missing or unknown:
        msg = (
            f"a calibration declaration is missing {missing} and names unknown "
            f"{unknown}"
        )
        raise ValueError(msg)

    grid = document["grid"]

    if not isinstance(grid, list) or not grid:
        msg = f"the declared grid is not a nonempty list of cells: {grid!r}"
        raise TypeError(msg)

    cells = tuple(_cell(cell) for cell in grid)
    names = [cell.name for cell in cells]

    if len(set(names)) != len(names):
        msg = f"the declared grid repeats a cell: {names!r}"
        raise ValueError(msg)

    scorer = document["scorer"]

    if not isinstance(scorer, str):
        msg = f"the declared scorer is not a component role: {scorer!r}"
        raise TypeError(msg)

    budget_seconds = _number(document, "budget_seconds")

    if budget_seconds <= 0.0:
        msg = f"the declared compute budget is not positive: {budget_seconds!r}"
        raise ValueError(msg)

    return CalibrationDeclaration(
        scorer=ensure_name("the declared scorer role", scorer),
        grid=cells,
        steps=ensure_positive("the declared steps", _integer(document, "steps")),
        threshold=_threshold(document),
        maximum_trials=ensure_non_negative(
            "the declared maximum trials", _integer(document, "maximum_trials")
        ),
        divergence=_named(document, "divergence", DIVERGENCES),
        metric=_named(document, "metric", METRICS),
        budget_seconds=budget_seconds,
        price_warmup=ensure_non_negative(
            "the declared price warmup", _integer(document, "price_warmup")
        ),
        price_repeats=ensure_positive(
            "the declared price repeats", _integer(document, "price_repeats")
        ),
    )


class Faithfulness(NamedTuple):
    """How far one circuit's output stays from the full model's."""

    #: One minus the share of the full corruption the circuit lets through, or
    #: None where patching every edge leaves the output unchanged.
    share: float | None

    #: The divergence of the circuit-only output from the full model's output.
    circuit_divergence: float

    #: The divergence of the fully corrupted output from the full model's output.
    corrupted_divergence: float


def _faithfulness[T](
    scorer: CalibratedScorer,
    circuit: ReceiverCircuit[T],
    kept: frozenset[GraphEdge[T]],
    *,
    cell: CalibrationCell,
    forward: Callable[[Tensor], Tensor],
    divergence: Callable[[Tensor, Tensor], Tensor],
) -> Faithfulness:
    """Return a circuit's normalized faithfulness and both raw divergences.

    Every receiver edge outside the circuit is patched from the counterfactual
    pass to give the circuit-only output.
    """
    site = scorer.receiver_site

    with torch.no_grad(), instrument(scorer, observe=frozenset({site})) as corrupted:
        corrupt_output = corrupted(lambda: forward(cell.counterfactual.tensor))

    with torch.no_grad():
        clean_output = forward(cell.clean.tensor)

    removed: tuple[Unit, ...] = tuple(
        circuit.targets[edge] for edge in circuit.order if edge not in kept
    )
    patch = Within(removed, PatchFrom(corrupted.trace, site))

    with torch.no_grad(), instrument(scorer, patch={site: patch}) as patched:
        circuit_output = patched(lambda: forward(cell.clean.tensor))

    full = float(divergence(clean_output, corrupt_output))
    kept_divergence = float(divergence(clean_output, circuit_output))

    return Faithfulness(
        None if full == 0.0 else 1.0 - kept_divergence / full,
        kept_divergence,
        full,
    )


def _positions[T](
    circuit: ReceiverCircuit[T], edges: Sequence[GraphEdge[T]]
) -> list[JsonValue]:
    """Return the receiver-edge coordinates of some edges, in their order."""
    return [circuit.targets[edge].index["edge"] for edge in edges]


def _price(
    unit: ProbeUnit,
    body: Callable[[], object],
    policy: DevicePolicy,
    declaration: CalibrationDeclaration,
) -> dict[str, JsonValue]:
    """Return one measured unit price as JSON."""
    price = price_unit(
        unit,
        body,
        policy=policy,
        warmup=declaration.price_warmup,
        repeats=declaration.price_repeats,
    )

    return {
        "unit": price.unit.value,
        "seconds_per_run": price.seconds_per_run,
        "seconds_fastest": price.seconds_fastest,
        "warmup": price.warmup,
        "repeats": price.repeats,
        "device": price.device,
    }


def calibrate_cell(
    scorer: CalibratedScorer,
    cell: CalibrationCell,
    declaration: CalibrationDeclaration,
) -> dict[str, JsonValue]:
    """Return one cell's comparison of EAP-IG with budgeted ACDC.

    Parameters
    ----------
    scorer : CalibratedScorer
        The chart scorer whose receiver edges are attributed and pruned.
    cell : CalibrationCell
        The cell's clean and counterfactual leaves.
    declaration : CalibrationDeclaration
        The declared policy.

    Returns
    -------
    dict[str, JsonValue]
        The cell's row.

    Raises
    ------
    ValueError
        If the two leaves differ in shape, axes or mask.
    """
    clean, counterfactual = cell.clean, cell.counterfactual

    if (
        clean.axes != counterfactual.axes
        or clean.tensor.shape != counterfactual.tensor.shape
    ):
        msg = f"the leaves of cell {cell.name!r} differ in axes or shape"
        raise ValueError(msg)

    if (clean.mask is None) != (counterfactual.mask is None) or (
        clean.mask is not None
        and counterfactual.mask is not None
        and not torch.equal(clean.mask, counterfactual.mask)
    ):
        msg = f"the leaves of cell {cell.name!r} hold words at different positions"
        raise ValueError(msg)

    circuit = receiver_circuit(scorer, SpanIndex(clean.size("word")), reading=cell.name)
    metric = METRICS[declaration.metric]
    divergence = DIVERGENCES[declaration.divergence]

    def forward(given: Tensor) -> Tensor:
        return metric(scorer.readouts(Encoded(given, clean.axes, clean.mask)))

    def attribute() -> EdgeAttribution[str]:
        return attribute_edges(
            scorer,
            circuit.graph,
            targets=circuit.targets,
            clean=clean.tensor,
            counterfactual=counterfactual.tensor,
            forward=lambda given: forward(given).sum(),
            steps=declaration.steps,
            dataset=cell.name,
        )

    def prune() -> ACDCResult[str]:
        return acdc(
            scorer,
            circuit.graph,
            targets=circuit.targets,
            order=circuit.order,
            clean=clean.tensor,
            counterfactual=counterfactual.tensor,
            forward=forward,
            divergence=divergence,
            threshold=declaration.threshold,
            budget=PruningBudget(declaration.maximum_trials),
            data_provenance=Provenance("calibration cell", cell.name, None),
            policy_provenance=Provenance(
                "receiver-order-acdc", declaration.divergence, None
            ),
        )

    started = perf_counter()
    attribution = attribute()
    attribution_seconds = perf_counter() - started
    result = prune()
    retained = frozenset(result.retained)
    ranked = attribution.ranked()[: len(retained)]
    policy = DevicePolicy(clean.tensor.device, clean.tensor.dtype)
    acdc_faithfulness = _faithfulness(
        scorer, circuit, retained, cell=cell, forward=forward, divergence=divergence
    )
    eap_faithfulness = _faithfulness(
        scorer,
        circuit,
        frozenset(ranked),
        cell=cell,
        forward=forward,
        divergence=divergence,
    )
    effects: list[JsonValue] = [attribution[edge].effect for edge in circuit.order]

    if not all(isinstance(effect, float) and isfinite(effect) for effect in effects):
        msg = f"cell {cell.name!r} has a nonfinite attributed effect"
        raise ValueError(msg)

    return {
        "cell": cell.name,
        "edges": len(circuit.order),
        "edge_order": _positions(circuit, circuit.order),
        "eap_ig": {
            "effects": effects,
            "ranked": _positions(circuit, attribution.ranked()),
            "matched_circuit": _positions(circuit, ranked),
            "faithfulness": eap_faithfulness.share,
            "circuit_divergence": eap_faithfulness.circuit_divergence,
            "corrupted_divergence": eap_faithfulness.corrupted_divergence,
            "wall_seconds": attribution_seconds,
            "price": _price(ProbeUnit.ATTRIBUTION, attribute, policy, declaration),
        },
        "acdc": {
            "threshold": result.threshold.to_spec(),
            "cutoff": result.cutoff,
            "tested": _positions(circuit, result.tested),
            "retained": _positions(circuit, result.retained),
            "pruned": _positions(circuit, result.pruned),
            "divergence": result.divergence,
            "scheduling_residual": result.scheduling_residual,
            "faithfulness": acdc_faithfulness.share,
            "circuit_divergence": acdc_faithfulness.circuit_divergence,
            "corrupted_divergence": acdc_faithfulness.corrupted_divergence,
            "wall_seconds": result.wall_seconds,
            "price": _price(ProbeUnit.PRUNING, prune, policy, declaration),
        },
        "matched_cardinality_jaccard": matched_cardinality_jaccard(result, attribution),
    }


def run(
    scorer: CalibratedScorer,
    cells: Sequence[CalibrationCell],
    declaration: CalibrationDeclaration,
    out: Path,
    *,
    point: str,
) -> str:
    """Calibrate every declared cell and write the manifest and rows.

    Parameters
    ----------
    scorer : CalibratedScorer
        The chart scorer the campaign runs over.
    cells : Sequence[CalibrationCell]
        One cell per declared grid name, in declared order.
    declaration : CalibrationDeclaration
        The declared grid and policy.
    out : Path
        A directory that does not yet exist.
    point : str
        The identity of the trained point the scorer belongs to.

    Returns
    -------
    str
        The outcome, ``completed`` or ``over-budget``.

    Raises
    ------
    FileExistsError
        If the output directory already exists.
    ValueError
        If the cells do not match the declared grid.
    """
    supplied = tuple(cell.name for cell in cells)
    declared = tuple(cell.name for cell in declaration.grid)

    if supplied != declared:
        msg = f"the supplied cells {supplied!r} are not the declared grid {declared!r}"
        raise ValueError(msg)

    out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    rows: list[dict[str, JsonValue]] = []
    outcome = COMPLETED

    for cell in cells:
        if perf_counter() - started >= declaration.budget_seconds:
            outcome = OVER_BUDGET
            break

        rows.append(calibrate_cell(scorer, cell, declaration))

    wall_seconds = perf_counter() - started

    if wall_seconds > declaration.budget_seconds:
        outcome = OVER_BUDGET

    environment = execution_environment()
    manifest: dict[str, JsonValue] = {
        "declaration": {
            "scorer": declaration.scorer,
            "grid": [
                {
                    "name": cell.name,
                    "clean": list(cell.clean),
                    "counterfactual": list(cell.counterfactual),
                }
                for cell in declaration.grid
            ],
            "steps": declaration.steps,
            "threshold": declaration.threshold.to_spec(),
            "maximum_trials": declaration.maximum_trials,
            "divergence": declaration.divergence,
            "metric": declaration.metric,
            "budget_seconds": declaration.budget_seconds,
            "price_warmup": declaration.price_warmup,
            "price_repeats": declaration.price_repeats,
        },
        "point": point,
        "calibrated": [row["cell"] for row in rows],
        "packages": dict(environment.packages),
        "git_revision": environment.git_revision,
        "accelerator": environment.accelerator,
        "wall_seconds": wall_seconds,
        "outcome": outcome,
    }
    lines = (json.dumps(row, sort_keys=True, allow_nan=False) for row in rows)
    (out / ROWS_NAME).write_text(
        "".join(f"{line}\n" for line in lines), "utf-8", newline="\n"
    )
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return outcome


def scorer_of(prepared: PreparedTraining, role: str, /) -> CalibratedScorer:
    """Return the declared component of a trained point as a calibrated scorer.

    Raises
    ------
    ValueError
        If the point holds no component under the role.
    TypeError
        If the component publishes no receiver edges or admits no region.
    """
    modules = prepared.model.modules

    if role not in modules:
        msg = f"the trained point holds no component role {role!r}: {sorted(modules)}"
        raise ValueError(msg)

    module = modules[role]

    if not (
        isinstance(module, ReceiverLocalChartScorer)
        and isinstance(module, Observable)
        and isinstance(module, Instrumentable)
    ):
        msg = (
            f"component role {role!r} is not an instrumentable chart scorer that "
            "publishes receiver edges"
        )
        raise TypeError(msg)

    return cast("CalibratedScorer", module)


def cells_of(
    prepared: PreparedTraining, declaration: CalibrationDeclaration, /
) -> tuple[CalibrationCell, ...]:
    """Return each declared cell's pooled leaves under a trained point."""
    return tuple(
        CalibrationCell(
            cell.name,
            pooled_leaves(prepared, (cell.clean,)),
            pooled_leaves(prepared, (cell.counterfactual,)),
        )
        for cell in declaration.grid
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Calibrate the declared cells of the trained point a run directory holds.

    Parameters
    ----------
    argv : Sequence[str] | None
        The arguments without the program name, or None to read them from the
        process.

    Returns
    -------
    int
        Zero once the campaign has written its files, whatever its outcome.

    Raises
    ------
    ValueError
        If the run directory records no completed run, or the declaration is
        refused.
    TypeError
        If the declared scorer role is not a calibrated chart scorer.
    FileExistsError
        If the output directory exists.
    """
    parser = ArgumentParser(
        prog="python -m scripts.probe.acdc_calibration",
        description=(
            "Calibrate EAP-IG against budgeted ACDC over a trained point's "
            "receiver edges."
        ),
    )
    parser.add_argument("--run", type=Path, required=True, metavar="RUN")
    parser.add_argument("--declaration", type=Path, required=True, metavar="DOC")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    namespace = parser.parse_args(argv)
    run_directory: Path = namespace.run
    document = json.loads(namespace.declaration.read_text(encoding="utf-8"))

    if not isinstance(document, dict):
        msg = f"the calibration declaration is not an object: {document!r}"
        raise TypeError(msg)

    declaration = declaration_of(document)
    manifest = read_manifest(run_directory)

    if manifest is None:
        msg = f"the run directory {run_directory} records no run"
        raise ValueError(msg)

    prepared = restore_trained_point(run_directory)
    scorer = scorer_of(prepared, declaration.scorer)
    run(
        scorer,
        cells_of(prepared, declaration),
        declaration,
        namespace.out,
        point=manifest.seed_derivation.point_id,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
