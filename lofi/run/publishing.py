"""The publish surface: results projected onto the forms external targets validate.

A publish lens renders one in-memory result as the artifact its target takes,
and ``LayersPredictionLens``, installed in the ``lofi.publisher`` entry-point
group as ``layers-records``, renders a ``PredictedAnalysis`` as a Layers
annotation-layer record. The record models come from ``lairs``, which the
``publish`` extra installs, so importing ``lofi.run`` does not import this
module, and this module is the only importer of the evaluation package's
publish implementation.

:func:`publish_trained_point` publishes a completed point's predictions. It
restores the point from its declared release checkpoint, reruns each
evaluator role a selection document names over the selection's retained split,
and labels each item's spans with the categories of the derivation the role's
evaluator selected. Each role's items are rendered through the Layers lens, one
record per line of a file named for the role, under the tokenization a mapping
document joins the item to and the terms the item's corpus declared. A pilot's
point, run outside any freeze, is refused, and so is a role whose evaluator
labels no derivation.
"""

from datetime import UTC, datetime
from importlib.metadata import version
from typing import TYPE_CHECKING, Final, cast

from lofi._types import ItemId
from lofi.eval import UNFROZEN, LabeledDerivationEvaluator
from lofi.eval._publish import (
    LAYERS_TARGET,
    LayersPredictionLens,
    PredictedAnalysis,
    PublishLens,
    PublishTarget,
)
from lofi.run._boundary.manifest import ManifestStatus
from lofi.run._boundary.publishing import PublishSelection, TokenizationMapping
from lofi.run._manifest import read_binding, read_manifest
from lofi.run._ports import Published
from lofi.run.held_out import held_out_predictions
from lofi.run.hub import restore_trained_point
from lofi.run.results import PilotResultsError
from lofi.spec import SpecLoader

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from lofi.data import Item, Provenance
    from lofi.run.held_out import HeldOutPredictions

__all__ = [
    "LAYERS_TARGET",
    "RECORDS_SUFFIX",
    "LayersPredictionLens",
    "PredictedAnalysis",
    "PublishLens",
    "PublishTarget",
    "UnlabeledDerivationError",
    "publish_trained_point",
]

RECORDS_SUFFIX: Final = ".jsonl"
"""The suffix of each role's records file, which holds one record per line."""

_DISTRIBUTION: Final = "lofi"
"""The distribution whose name and version a published record names as its tool."""


class UnlabeledDerivationError(ValueError):
    """Signals an evaluator role whose predictions carry no labeled derivation."""


def _refuse_pilot(run: Path, /) -> None:
    """Refuse a point a pilot bound outside any freeze.

    Raises
    ------
    PilotResultsError
        If the point's binding records the freeze as ``UNFROZEN``.
    """
    binding = read_binding(run)

    if binding is not None and binding.freeze_id == UNFROZEN:
        msg = (
            f"the point at {run} is a pilot's, run outside any freeze, and a "
            "pilot's predictions are not published"
        )
        raise PilotResultsError(msg)


def _provenance(item: Item, /) -> Provenance:
    """Return the terms an item's corpus declared.

    Raises
    ------
    ValueError
        If the item records no provenance.
    """
    provenance = item.provenance

    if provenance is None:
        msg = f"the item {item.item_id!r} records no provenance to publish under"
        raise ValueError(msg)

    return provenance


def _analyses(
    role: str,
    rerun: HeldOutPredictions,
    joined: Mapping[ItemId, str],
    produced_at: datetime,
    /,
) -> tuple[PredictedAnalysis | None, ...]:
    """Return each item's labeled analysis, or None where none was selected.

    Raises
    ------
    UnlabeledDerivationError
        If the role's evaluator cannot label derivations.
    ValueError
        If an item of the split has no tokenization or no provenance.
    """
    evaluator = rerun.evaluator

    if not isinstance(evaluator, LabeledDerivationEvaluator):
        msg = (
            f"the evaluator role {role!r} predicts no labeled derivation, so its "
            "predictions label no span"
        )
        raise UnlabeledDerivationError(msg)

    unjoined = tuple(item.item_id for item in rerun.items if item.item_id not in joined)

    if unjoined:
        msg = (
            f"the tokenization mapping joins no tokenization to the items {unjoined!r}"
        )
        raise ValueError(msg)

    labeler = cast("LabeledDerivationEvaluator[object]", evaluator)
    analyses: list[PredictedAnalysis | None] = []

    for item in rerun.items:
        prediction = rerun.predictions.get(item.item_id)
        spans = None if prediction is None else labeler.labeled_spans(prediction)
        analyses.append(
            None
            if spans is None
            else PredictedAnalysis(
                item.item_id,
                joined[item.item_id],
                spans,
                _provenance(item),
                produced_at,
            )
        )

    return tuple(analyses)


def publish_trained_point(
    run: Path, tokenizations: Path, selection: Path, destination: Path, /
) -> tuple[Published, ...]:
    """Publish a completed point's labeled derivations as Layers records.

    Parameters
    ----------
    run : Path
        The run directory of a completed point.
    tokenizations : Path
        The tokenization mapping document.
    selection : Path
        The publish selection document.
    destination : Path
        The records directory to create, which must not exist. It is written
        only once every selected role has been rendered.

    Returns
    -------
    tuple[Published, ...]
        One entry per selected role, in selection order.

    Raises
    ------
    PilotResultsError
        If the point is a pilot's, bound outside any freeze.
    FileExistsError
        If the destination exists.
    ConfigError
        If either document is refused.
    UnlabeledDerivationError
        If a selected role's evaluator cannot label derivations, or labels none
        of the split's items.
    ValueError
        If the directory records no completed run, the run declares no
        evaluator under a selected role or retains no gold item under the
        selection's split, or an item of the split has no tokenization or no
        provenance.
    """
    _refuse_pilot(run)

    if destination.exists():
        msg = f"the records directory {destination} exists"
        raise FileExistsError(msg)

    declared = SpecLoader().compose_traced(selection, schema=PublishSelection).spec
    mapping = (
        SpecLoader().compose_traced(tokenizations, schema=TokenizationMapping).spec
    )
    joined = {
        ItemId(record.item): record.tokenization for record in mapping.tokenizations
    }
    manifest = read_manifest(run)

    if manifest is None or manifest.status is not ManifestStatus.COMPLETED:
        msg = f"the run directory {run} records no completed run"
        raise ValueError(msg)

    prepared = restore_trained_point(run)
    lens = LayersPredictionLens(
        f"{_DISTRIBUTION} {version(_DISTRIBUTION)}", declared.formalism
    )
    produced_at = datetime.now(UTC)
    rendered: list[tuple[str, tuple[str, ...], int]] = []

    for role in declared.evaluators:
        rerun = held_out_predictions(prepared, manifest.run_spec, role, declared.split)
        analyses = _analyses(role, rerun, joined, produced_at)
        records = tuple(
            lens.render(analysis) for analysis in analyses if analysis is not None
        )

        if not records:
            msg = (
                f"the evaluator role {role!r} labels no derivation of the "
                f"{len(analyses)} items of the split {declared.split!r}"
            )
            raise UnlabeledDerivationError(msg)

        rendered.append((role, records, len(analyses) - len(records)))

    destination.mkdir(parents=True)
    published: list[Published] = []

    for role, records, unlabeled in rendered:
        path = destination / f"{role}{RECORDS_SUFFIX}"
        path.write_text(
            "".join(f"{record}\n" for record in records), encoding="utf-8", newline="\n"
        )
        published.append(Published(role, path, len(records), unlabeled))

    return tuple(published)
