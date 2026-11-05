"""Projections of run results onto the forms external targets validate."""

from typing import TYPE_CHECKING, Final, Protocol, final, override

from lairs.records import annotation as annotation_records
from lairs.records import defs as defs_records

from lofi._checks import ensure_name
from lofi._types import ItemId

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from lofi.data import Licensing, Provenance, Reproducibility
    from lofi.eval._matched import InducedBracketing

__all__ = [
    "LAYERS_TARGET",
    "LayersPredictionLens",
    "PredictedAnalysis",
    "PublishLens",
    "PublishTarget",
]

type _Span = tuple[int, int]
"""A half-open span over token positions."""

_INDUCED_SOURCE_METHOD: Final = "automatic"
"""The Layers source method for predictions made by a fitted system."""


@final
class PublishTarget:
    """A publication target and the media type of the artifact it takes.

    Parameters
    ----------
    tag : str
        The target's name, compared for equality and never by substring.
    media_type : str
        The media type of the text a lens for this target renders.

    Raises
    ------
    ValueError
        If either field is empty.
    """

    __slots__ = ("_media_type", "_tag")

    def __init__(self, tag: str, media_type: str) -> None:
        self._tag = ensure_name("a publish target's tag", tag)
        self._media_type = ensure_name("a publish target's media type", media_type)

    @property
    def tag(self) -> str:
        """Return the target's name."""
        return self._tag

    @property
    def media_type(self) -> str:
        """Return the media type of the artifact the target takes."""
        return self._media_type

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value names the same target."""
        if type(other) is not PublishTarget:
            return NotImplemented

        return self._tag == other._tag and self._media_type == other._media_type

    @override
    def __hash__(self) -> int:
        """Return a hash over the tag and the media type."""
        return hash((PublishTarget, self._tag, self._media_type))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this target."""
        return f"PublishTarget({self._tag!r}, {self._media_type!r})"


LAYERS_TARGET: Final = PublishTarget("layers-records", "application/json")
"""The Layers repository, which takes annotation-layer records."""


@final
class PredictedAnalysis:
    """One item's predicted constituents and the terms its corpus declared.

    Parameters
    ----------
    item_id : ItemId
        The stable identifier of the item the constituents were predicted for.
    tokenization_id : str
        The identifier of the tokenization the spans index into.
    bracketing : InducedBracketing
        The predicted labeled spans, half-open over token positions.
    provenance : Provenance
        The terms and source method the item's corpus declared.
    produced_at : datetime
        When the prediction was made.

    Raises
    ------
    ValueError
        If the item identifier or the tokenization identifier is empty.
    """

    __slots__ = (
        "_bracketing",
        "_item_id",
        "_produced_at",
        "_provenance",
        "_tokenization_id",
    )

    def __init__(
        self,
        item_id: ItemId,
        tokenization_id: str,
        bracketing: InducedBracketing,
        provenance: Provenance,
        produced_at: datetime,
    ) -> None:
        self._item_id = ItemId(ensure_name("a predicted analysis's item id", item_id))
        self._tokenization_id = ensure_name(
            "a predicted analysis's tokenization", tokenization_id
        )
        self._bracketing = bracketing
        self._provenance = provenance
        self._produced_at = produced_at

    @property
    def item_id(self) -> ItemId:
        """Return the stable identifier the prediction joins to its source by."""
        return self._item_id

    @property
    def expression(self) -> str:
        """Return the Layers expression reference for the source item."""
        return self._item_id

    @property
    def tokenization_id(self) -> str:
        """Return the identifier of the tokenization the spans index into."""
        return self._tokenization_id

    @property
    def bracketing(self) -> InducedBracketing:
        """Return the predicted labeled spans."""
        return self._bracketing

    @property
    def provenance(self) -> Provenance:
        """Return the terms and source method the item's corpus declared."""
        return self._provenance

    @property
    def produced_at(self) -> datetime:
        """Return when the prediction was made."""
        return self._produced_at

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same predicted analysis."""
        if type(other) is not PredictedAnalysis:
            return NotImplemented

        return (
            self._item_id == other._item_id
            and self._tokenization_id == other._tokenization_id
            and self._bracketing == other._bracketing
            and self._provenance == other._provenance
            and self._produced_at == other._produced_at
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the expression, spans, provenance and time."""
        return hash(
            (
                PredictedAnalysis,
                self._item_id,
                self._tokenization_id,
                self._bracketing,
                self._provenance,
                self._produced_at,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this analysis."""
        return (
            f"PredictedAnalysis({self._item_id!r}, {self._tokenization_id!r}, "
            f"{self._bracketing!r}, {self._provenance!r}, {self._produced_at!r})"
        )


class PublishLens[S](Protocol):
    """Projects one in-memory result onto the form its target validates.

    The projection runs off line and has no inverse: an artifact is rendered
    and written, and no lens reads one back into the result it came from.
    """

    @property
    def target(self) -> PublishTarget:
        """Return the target this lens renders for."""
        ...

    def render(self, source: S, /) -> str:
        """Return the artifact text the target's own models accept.

        Parameters
        ----------
        source : S
            The result to publish.
        """
        ...


def _span_id(span: _Span) -> str:
    """Return the identifier a predicted span is referred to by.

    Parameters
    ----------
    span : _Span
        The half-open span over token positions.
    """
    start, end = span

    return f"{start}-{end}"


def _enclosing(span: _Span, spans: Sequence[_Span]) -> _Span | None:
    """Return the shortest predicted span containing another one.

    Parameters
    ----------
    span : _Span
        The span whose parent is wanted.
    spans : Sequence[_Span]
        Every predicted span, including the one being placed.

    Returns
    -------
    _Span | None
        The parent, or None where no other predicted span contains this one.
    """
    shortest: _Span | None = None

    for other in spans:
        if other == span or other[0] > span[0] or span[1] > other[1]:
            continue

        if shortest is None or other[1] - other[0] < shortest[1] - shortest[0]:
            shortest = other

    return shortest


def _uuids(spans: Sequence[_Span]) -> tuple[defs_records.Uuid, ...]:
    """Return the record identifiers of a run of predicted spans.

    Parameters
    ----------
    spans : Sequence[_Span]
        The spans to identify.
    """
    return tuple(defs_records.Uuid(value=_span_id(span)) for span in spans)


def _annotation(
    span: _Span,
    label: str,
    parent: _Span | None,
    children: Sequence[_Span],
    tokenization: defs_records.Uuid,
) -> annotation_records.Annotation:
    """Return the tree annotation for one predicted span.

    Parameters
    ----------
    span : _Span
        The half-open span over token positions.
    label : str
        The label predicted for the span.
    parent : _Span | None
        The shortest predicted span containing this one, where one does.
    children : Sequence[_Span]
        The predicted spans this one immediately contains.
    tokenization : defs_records.Uuid
        The tokenization the span indexes into.
    """
    start, end = span

    return annotation_records.Annotation(
        uuid=defs_records.Uuid(value=_span_id(span)),
        label=label,
        anchor=defs_records.Anchor(
            tokenRefSequence=defs_records.TokenRefSequence(
                tokenIndexes=tuple(range(start, end)),
                tokenizationId=tokenization,
            )
        ),
        parentId=(
            None if parent is None else defs_records.Uuid(value=_span_id(parent))
        ),
        childIds=_uuids(children),
    )


def _annotations(
    labels: Mapping[_Span, str], tokenization: defs_records.Uuid
) -> tuple[annotation_records.Annotation, ...]:
    """Return one tree annotation per predicted span, in span order.

    Parameters
    ----------
    labels : Mapping[_Span, str]
        The label predicted for each half-open span.
    tokenization : defs_records.Uuid
        The tokenization the spans index into.

    Returns
    -------
    tuple[annotation_records.Annotation, ...]
        The annotations, each linked to the shortest predicted span containing
        it and to the spans it immediately contains.
    """
    spans = tuple(labels)
    parents = {span: _enclosing(span, spans) for span in spans}
    children: dict[_Span, list[_Span]] = {span: [] for span in spans}

    for span, parent in parents.items():
        if parent is not None:
            children[parent].append(span)

    return tuple(
        _annotation(
            span, labels[span], parents[span], sorted(children[span]), tokenization
        )
        for span in spans
    )


def _licensing(licensing: Licensing) -> defs_records.Licensing:
    """Return the Layers form of the terms a corpus declared.

    Parameters
    ----------
    licensing : Licensing
        The terms the corpus declared.
    """
    return defs_records.Licensing(
        expression=licensing.expression,
        licenses=tuple(
            defs_records.LicenseRef(
                spdx=reference.identifier, appliesTo=reference.applies_to
            )
            for reference in licensing.licenses
        ),
    )


def _reproducibility(
    reproducibility: Reproducibility | None,
) -> defs_records.ReproducibilityInfo | None:
    """Return the Layers form of how a corpus was computed.

    Parameters
    ----------
    reproducibility : Reproducibility | None
        How the corpus was computed, where it was computed rather than
        annotated.
    """
    if reproducibility is None:
        return None

    return defs_records.ReproducibilityInfo(
        codeUri=reproducibility.code_uri,
        command=reproducibility.command,
        commitHash=reproducibility.commit_hash,
        environment=reproducibility.environment,
        randomSeed=reproducibility.random_seed,
    )


@final
class LayersPredictionLens:
    """Renders a predicted analysis as the record a Layers repository validates.

    Parameters
    ----------
    tool : str
        The name of the software that produced the prediction.
    formalism : str
        The slug naming the formalism the constituent labels are drawn from.

    Raises
    ------
    ValueError
        If either field is empty.
    """

    __slots__ = ("_formalism", "_tool")

    def __init__(self, tool: str, formalism: str) -> None:
        self._tool = ensure_name("a publishing tool's name", tool)
        self._formalism = ensure_name("a published layer's formalism", formalism)

    @property
    def tool(self) -> str:
        """Return the name of the software that produced the prediction."""
        return self._tool

    @property
    def formalism(self) -> str:
        """Return the formalism the constituent labels are drawn from."""
        return self._formalism

    @property
    def target(self) -> PublishTarget:
        """Return the Layers repository."""
        return LAYERS_TARGET

    def render(self, source: PredictedAnalysis, /) -> str:
        """Return the annotation-layer record for one predicted analysis.

        Parameters
        ----------
        source : PredictedAnalysis
            The analysis to publish, with the terms its corpus declared.

        Returns
        -------
        str
            The record as JSON, validated on the way out by the Layers record
            models themselves.
        """
        provenance = source.provenance
        tokenization = defs_records.Uuid(value=source.tokenization_id)
        layer = annotation_records.AnnotationLayer(
            annotations=_annotations(source.bracketing.spans, tokenization),
            createdAt=source.produced_at,
            eprintRefs=provenance.eprint_refs,
            expression=source.item_id,
            formalism=self._formalism,
            kind="tree",
            licensing=_licensing(provenance.licensing),
            metadata=defs_records.AnnotationMetadata(
                tool=self._tool, timestamp=source.produced_at
            ),
            reproducibility=_reproducibility(provenance.reproducibility),
            sourceMethod=_INDUCED_SOURCE_METHOD,
            subkind="constituency",
            tokenizationId=tokenization,
        )

        return layer.model_dump_json()
