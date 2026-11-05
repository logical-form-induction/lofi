"""Validated comparison metadata from completed experiment points."""

from types import MappingProxyType
from typing import TYPE_CHECKING

from lofi.eval import config_digest
from lofi.eval._comparison import PointMetadata
from lofi.spec import RunSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.run._experiment import ManifestPoint

__all__ = ["comparison_points"]


class ResultsLayoutError(ValueError):
    """Signals result records that cannot identify completed points."""


def comparison_points(
    manifests: Sequence[ManifestPoint],
    measured: Sequence[Mapping[str, JsonValue]],
    /,
) -> Mapping[str, tuple[PointMetadata, ...]]:
    """Join validated manifests to measured capacities by point identifier.

    Each point retains both the sweep-level replicate seed used for comparison
    pairing and the derived run root written on its report rows.
    """
    measured_by_point: dict[str, Mapping[str, JsonValue]] = {}
    for row in measured:
        point = row.get("point_id")
        if not isinstance(point, str) or not point:
            msg = f"a measured point has no point identifier: {point!r}"
            raise ResultsLayoutError(msg)
        if point in measured_by_point:
            msg = f"the measured table repeats point {point!r}"
            raise ResultsLayoutError(msg)
        measured_by_point[point] = row

    grouped: dict[str, list[PointMetadata]] = {}
    for manifest in manifests:
        identity = manifest.seed.point_id
        signature = manifest.signature_fingerprint
        if identity not in measured_by_point or signature is None:
            continue
        count = measured_by_point[identity].get("point/parameter_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            msg = (
                f"measured point {identity!r} has no nonnegative parameter count: "
                f"{count!r}"
            )
            raise ResultsLayoutError(msg)
        config = manifest.config.document
        try:
            declared = RunSpec.model_validate(config).signature_assignment
        except (TypeError, ValueError) as refused:
            msg = f"point {identity!r} records a malformed run specification: {refused}"
            raise ResultsLayoutError(msg) from refused
        assignments = None if declared is None else declared.build()
        digest = config_digest(config)
        grouped.setdefault(digest, []).append(
            PointMetadata(
                config,
                point_id=identity,
                seed=manifest.seed.seed,
                result_seed=manifest.seed.root,
                parameter_count=count,
                signature_fingerprint=signature,
                assignments=assignments,
            )
        )

    return MappingProxyType(
        {
            digest: tuple(sorted(members, key=lambda member: member.seed))
            for digest, members in grouped.items()
        }
    )
