"""Summaries of node vocabulary sweeps over one evaluated pair population."""

from enum import Enum
from hashlib import sha256
from json import dumps, loads
from statistics import fmean
from typing import TYPE_CHECKING, cast, final, override

from lofi._checks import ensure_finite, ensure_name
from lofi._frozen import FrozenValue
from lofi.probe._sweep import SweepReport

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lofi._types import JsonValue

__all__ = [
    "NodeVocabularyArm",
    "NodeVocabularyCell",
    "NodeVocabularyComparison",
    "NodeVocabularyRow",
    "node_vocabulary_row",
]

_PAIR_SET_DOMAIN = "lofi.node-vocabulary-pairs"
"""The content-address domain for evaluated node vocabulary pair sets."""

_PAIR_SET_VERSION = 1
"""The canonical pair-set record version."""


class NodeVocabularyArm(Enum):
    """The four fixed representations in a node vocabulary comparison."""

    #: High-level variables supplied by the derivations.
    DERIVATION_VARIABLES = "derivation-variables"

    #: Sparse-autoencoder latents from a trained transformer.
    TRAINED_TRANSFORMER_SAE_LATENTS = "trained-transformer-sae-latents"

    #: A capacity-declared dense linear readout.
    DENSE_LINEAR_PROBE = "dense-linear-probe"

    #: Sparse-autoencoder latents from a randomly initialized transformer.
    RANDOM_TRANSFORMER_SAE_CONTROL = "random-transformer-sae-control"


def _pair_population(pair_ids: Iterable[str]) -> tuple[tuple[str, ...], str]:
    """Return canonical pair identities and their versioned content address.

    Parameters
    ----------
    pair_ids : Iterable[str]
        Stable identities of the evaluated pairs, in any order.

    Raises
    ------
    TypeError
        If the population is text or an identity is not text.
    ValueError
        If the population is empty or repeats an identity.
    """
    if isinstance(pair_ids, str):
        msg = "a pair population is an iterable of identities, not text"
        raise TypeError(msg)

    held = tuple(
        ensure_name("an evaluated pair identity", identity) for identity in pair_ids
    )

    if not held:
        msg = "a node-vocabulary row evaluates at least one pair"
        raise ValueError(msg)

    if len(set(held)) != len(held):
        msg = f"a node-vocabulary row repeats a pair identity: {held!r}"
        raise ValueError(msg)

    ordered = tuple(sorted(held))
    payload: dict[str, JsonValue] = {
        "domain": _PAIR_SET_DOMAIN,
        "version": _PAIR_SET_VERSION,
        "pair_ids": list(ordered),
    }
    encoded = dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    return ordered, sha256(encoded).hexdigest()


def _canonical(value: JsonValue, /) -> str:
    """Return one JSON value in canonical, comparable spelling."""
    return dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


@final
class NodeVocabularyCell(FrozenValue):
    """One logical cell's interchange accuracy and the null it was tested under.

    Parameters
    ----------
    cell : str
        The canonical JSON spelling of the logical cell, without its site.
    accuracy : float
        The cell's held-out interchange intervention accuracy.
    null : str
        The canonical JSON spelling of the cell's null provenance.

    Raises
    ------
    ValueError
        If the cell or null is empty, or the accuracy lies outside the unit
        interval.
    """

    _noun = "node-vocabulary cell"

    __slots__ = ("_accuracy", "_cell", "_null")

    def __init__(self, cell: str, accuracy: float, null: str) -> None:
        held = ensure_finite("a node-vocabulary cell accuracy", accuracy)

        if not 0.0 <= held <= 1.0:
            msg = f"a node-vocabulary cell accuracy lies outside [0, 1]: {held!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_cell", ensure_name("a logical cell", cell))
        object.__setattr__(self, "_accuracy", held)
        object.__setattr__(self, "_null", ensure_name("a cell's null", null))

    @property
    def cell(self) -> str:
        """Return the canonical spelling of the logical cell."""
        return self._cell

    @property
    def accuracy(self) -> float:
        """Return the cell's interchange intervention accuracy."""
        return self._accuracy

    @property
    def null(self) -> str:
        """Return the canonical spelling of the cell's null provenance."""
        return self._null

    @override
    def __repr__(self) -> str:
        """Return the cell and its accuracy."""
        return f"NodeVocabularyCell({self._cell!r}, {self._accuracy!r})"


@final
class NodeVocabularyRow(FrozenValue):
    """One arm's complete logical-cell results over its evaluated pair population.

    Parameters
    ----------
    arm : NodeVocabularyArm
        Representation assessed by the sweep.
    cells : Iterable[NodeVocabularyCell]
        One result per preregistered logical cell, in the sweep's cell order.
    pair_ids : Iterable[str]
        Stable identities of the evaluated pairs, in any order.

    Raises
    ------
    TypeError
        If the arm or a cell has the wrong type, the pair population is text, or
        an identity is not text.
    ValueError
        If no cell is given, a logical cell repeats, or the pair population is
        empty or repeats an identity.
    """

    _noun = "node-vocabulary row"

    __slots__ = ("_arm", "_cells", "_pair_digest", "_pair_ids")

    def __init__(
        self,
        arm: NodeVocabularyArm,
        cells: Iterable[NodeVocabularyCell],
        pair_ids: Iterable[str],
    ) -> None:
        if type(arm) is not NodeVocabularyArm:
            msg = f"a node-vocabulary row names a NodeVocabularyArm, not {arm!r}"
            raise TypeError(msg)

        held = tuple(cells)

        for cell in held:
            if type(cell) is not NodeVocabularyCell:
                msg = f"a node-vocabulary row holds NodeVocabularyCell values: {cell!r}"
                raise TypeError(msg)

        if not held:
            msg = "a node-vocabulary row reports at least one logical cell"
            raise ValueError(msg)

        names = tuple(cell.cell for cell in held)

        if len(set(names)) != len(names):
            msg = "a node-vocabulary row reports one logical cell twice"
            raise ValueError(msg)

        identities, digest = _pair_population(pair_ids)
        object.__setattr__(self, "_arm", arm)
        object.__setattr__(self, "_cells", held)
        object.__setattr__(self, "_pair_ids", identities)
        object.__setattr__(self, "_pair_digest", digest)

    @classmethod
    def from_record(cls, record: JsonValue, /) -> NodeVocabularyRow:
        """Decode a persisted row record at its reading boundary.

        Parameters
        ----------
        record : JsonValue
            A record :meth:`to_record` wrote.

        Raises
        ------
        TypeError
            If the record is not an object of the recorded shape.
        ValueError
            If the arm is undeclared, the recorded digest is not the digest of
            the recorded pairs, or a field is invalid.
        """
        if not isinstance(record, dict):
            msg = f"a node-vocabulary record is an object, not {record!r}"
            raise TypeError(msg)

        if set(record) != {"arm", "cells", "pair_ids", "pair_digest"}:
            msg = f"a node-vocabulary record has fields {sorted(record)!r}"
            raise ValueError(msg)

        cells = record["cells"]
        pairs = record["pair_ids"]

        if not isinstance(cells, list) or not isinstance(pairs, list):
            msg = "a node-vocabulary record lists its cells and pairs"
            raise TypeError(msg)

        decoded: list[NodeVocabularyCell] = []

        for cell in cells:
            if not isinstance(cell, dict) or set(cell) != {"cell", "accuracy", "null"}:
                msg = f"a node-vocabulary record cell is malformed: {cell!r}"
                raise ValueError(msg)

            accuracy = cell["accuracy"]

            if isinstance(accuracy, bool) or not isinstance(accuracy, int | float):
                msg = f"a node-vocabulary cell accuracy is not a number: {accuracy!r}"
                raise TypeError(msg)

            decoded.append(
                NodeVocabularyCell(
                    _canonical(cell["cell"]), float(accuracy), _canonical(cell["null"])
                )
            )

        row = cls(
            NodeVocabularyArm(record["arm"]),
            decoded,
            (cast("str", pair) for pair in pairs),
        )

        if record["pair_digest"] != row.pair_digest:
            msg = (
                f"a node-vocabulary record names pair digest "
                f"{record['pair_digest']!r}, not that of its pairs {row.pair_digest!r}"
            )
            raise ValueError(msg)

        return row

    @property
    def arm(self) -> NodeVocabularyArm:
        """Return the representation assessed by this row."""
        return self._arm

    @property
    def cells(self) -> tuple[NodeVocabularyCell, ...]:
        """Return one result per logical cell in the sweep's cell order."""
        return self._cells

    @property
    def pair_ids(self) -> tuple[str, ...]:
        """Return evaluated pair identities in canonical order."""
        return self._pair_ids

    @property
    def pair_digest(self) -> str:
        """Return the versioned content address of the evaluated pair set."""
        return self._pair_digest

    def to_record(self) -> dict[str, JsonValue]:
        """Return the row as the JSON record :meth:`from_record` decodes."""
        return {
            "arm": self._arm.value,
            "cells": [
                {
                    "cell": cast("JsonValue", loads(cell.cell)),
                    "accuracy": cell.accuracy,
                    "null": cast("JsonValue", loads(cell.null)),
                }
                for cell in self._cells
            ],
            "pair_ids": list(self._pair_ids),
            "pair_digest": self._pair_digest,
        }

    @override
    def __repr__(self) -> str:
        """Return the arm, pair count, and cell count."""
        return (
            f"NodeVocabularyRow({self._arm!r}, pairs={len(self._pair_ids)!r}, "
            f"cells={len(self._cells)!r})"
        )


def node_vocabulary_row(
    arm: NodeVocabularyArm, report: SweepReport, pair_ids: Iterable[str], /
) -> NodeVocabularyRow:
    """Return one arm's row from its complete sweep report.

    Parameters
    ----------
    arm : NodeVocabularyArm
        Representation assessed by the sweep.
    report : SweepReport
        Complete logical-cell report for the arm.
    pair_ids : Iterable[str]
        Stable identities of the evaluated pairs, in any order.

    Raises
    ------
    TypeError
        If the report is not a sweep report.
    ValueError
        If the report is incomplete or a result's null differs from its p-value
        provenance.
    """
    if type(report) is not SweepReport:
        msg = f"a node-vocabulary row holds a SweepReport, not {report!r}"
        raise TypeError(msg)

    if report.unassessed:
        msg = (
            "a node-vocabulary row requires every preregistered cell; "
            f"{len(report.unassessed)} are unassessed"
        )
        raise ValueError(msg)

    for result in report.results:
        if result.result.null != result.p_value.provenance:
            msg = (
                f"the IIA result for {result.cell!r} names null "
                f"{result.result.null!r}, but its p-value names "
                f"{result.p_value.provenance!r}"
            )
            raise ValueError(msg)

    return NodeVocabularyRow(
        arm,
        (
            NodeVocabularyCell(
                _canonical(result.cell.to_json()),
                result.result.accuracy,
                _canonical(result.result.null.to_json()),
            )
            for result in report.results
        ),
        pair_ids,
    )


@final
class NodeVocabularyComparison(FrozenValue):
    """Four complete node vocabulary sweeps over identical evaluated evidence.

    Parameters
    ----------
    rows : Iterable[NodeVocabularyRow]
        Exactly one row for each fixed arm, in any order.

    Raises
    ------
    TypeError
        If a member is not a node vocabulary row.
    ValueError
        If an arm is absent or repeated, or the rows differ in pair population,
        logical cells, or per-cell null provenance.
    """

    _noun = "node-vocabulary comparison"

    __slots__ = ("_rows",)

    def __init__(self, rows: Iterable[NodeVocabularyRow]) -> None:
        held = tuple(rows)

        for row in held:
            if type(row) is not NodeVocabularyRow:
                msg = f"a node-vocabulary comparison contains a non-row: {row!r}"
                raise TypeError(msg)

        arms = tuple(row.arm for row in held)

        if len(set(arms)) != len(arms):
            repeated = next(arm for arm in arms if arms.count(arm) > 1)
            msg = f"a node-vocabulary comparison repeats arm {repeated.value!r}"
            raise ValueError(msg)

        missing = tuple(arm for arm in NodeVocabularyArm if arm not in arms)

        if missing:
            named = tuple(arm.value for arm in missing)
            msg = f"a node-vocabulary comparison is missing arms {named!r}"
            raise ValueError(msg)

        ordered = tuple(
            next(row for row in held if row.arm is arm) for arm in NodeVocabularyArm
        )
        first = ordered[0]

        for row in ordered[1:]:
            if row.pair_digest != first.pair_digest:
                msg = (
                    f"arm {row.arm.value!r} evaluated pair digest "
                    f"{row.pair_digest!r}, not {first.pair_digest!r}"
                )
                raise ValueError(msg)

            cells = tuple(cell.cell for cell in row.cells)
            first_cells = tuple(cell.cell for cell in first.cells)

            if cells != first_cells:
                msg = (
                    f"arm {row.arm.value!r} reports logical cells {cells!r}, "
                    f"not {first_cells!r}"
                )
                raise ValueError(msg)

            provenance = tuple(cell.null for cell in row.cells)
            first_provenance = tuple(cell.null for cell in first.cells)

            if provenance != first_provenance:
                msg = f"arm {row.arm.value!r} uses different per-cell null provenance"
                raise ValueError(msg)

        object.__setattr__(self, "_rows", ordered)

    @property
    def rows(self) -> tuple[NodeVocabularyRow, ...]:
        """Return one typed row per arm in canonical arm order."""
        return self._rows

    def to_json(self) -> list[dict[str, JsonValue]]:
        """Return one JSON-safe scalar summary per arm in canonical order."""
        return [
            {
                "arm": row.arm.value,
                "pair_digest": row.pair_digest,
                "cell_count": len(row.cells),
                "mean_iia": fmean(cell.accuracy for cell in row.cells),
            }
            for row in self._rows
        ]

    @override
    def __repr__(self) -> str:
        """Return the arm, pair, and cell counts."""
        first = self._rows[0]

        return (
            f"NodeVocabularyComparison(arms={len(self._rows)!r}, "
            f"pairs={len(first.pair_ids)!r}, cells={len(first.cells)!r})"
        )
