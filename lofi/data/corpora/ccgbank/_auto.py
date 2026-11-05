"""CCGbank AUTO derivation encoding and decoding."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, NamedTuple

from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from lairs.records import annotation as annotation_records
from lairs.records import defs as defs_records
from lairs.records import expression as expression_records
from lairs.records import segmentation as segmentation_records

from lofi.data.corpora.ccgbank._auto_parse import (
    AutoDerivation,
    AutoFile,
    AutoLeaf,
    AutoNode,
    AutoTree,
    MalformedAutoError,
    parse_auto,
    write_auto,
)
from lofi.data.registration import registers_as

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi.data import Iso

__all__ = [
    "NAME",
    "CcgbankAutoCodec",
    "CcgbankAutoIso",
]

NAME: Final = "ccgbank-auto"
"""The registered codec name."""

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
"""The fixed epoch used when AUTO supplies no timestamp."""

_SCHEME: Final = "at://local"
"""The local record URI prefix."""

_EXPRESSION_NSID: Final = "pub.layers.expression.expression"
"""The expression-record collection."""

_SEGMENTATION_NSID: Final = "pub.layers.segmentation.segmentation"
"""The segmentation-record collection."""

_ANNOTATION_NSID: Final = "pub.layers.annotation.annotationLayer"
"""The annotation-layer collection."""

_ONTOLOGY: Final = f"{_SCHEME}/pub.layers.ontology.typeDef/ccgbank"
"""The type definition for category fragments."""

_HEAD_KEY: Final = "head"
"""The feature key for a node's head daughter."""

_POS_KEY: Final = "pos"
"""The feature key for a CCGbank tag."""

_ORIGINAL_POS_KEY: Final = "originalPos"
"""The feature key for a source-treebank tag."""

_SLOTTED_KEY: Final = "predicateArgumentCategory"
"""The feature key for a category with numbered argument slots."""


def _leaves(tree: AutoNode) -> tuple[AutoLeaf, ...]:
    """Return the words of an analysis, in surface order."""
    return tuple(node for node, _ in _placed(tree) if isinstance(node, AutoLeaf))


def _placed(tree: AutoNode) -> tuple[tuple[AutoNode, int | None], ...]:
    """Return every node of an analysis beside the node it falls under."""
    order: list[tuple[AutoNode, int | None]] = []
    stack: list[tuple[AutoNode, int | None]] = [(tree, None)]

    while stack:
        node, parent = stack.pop()
        place = len(order)
        order.append((node, parent))

        if isinstance(node, AutoTree):
            stack.extend((child, place) for child in reversed(node.children))

    return tuple(order)


def _key(index: int) -> str:
    """Return the record key one derivation's three records share."""
    return f"derivation-{index}"


def _uri(nsid: str, key: str) -> str:
    """Return the AT-URI one record is addressed under."""
    return f"{_SCHEME}/{nsid}/{key}"


def _node_id(place: int) -> str:
    """Return the identifier for a node position."""
    return f"node-{place}"


def _type_ref(category: str) -> str:
    """Return the ontology type reference one category is written as."""
    return f"{_ONTOLOGY}#{category}"


def _category_of(reference: str | None) -> str:
    """Return the category one ontology type reference names."""
    named, marked, category = (reference or "").partition("#")

    if not marked or named != _ONTOLOGY:
        msg = f"an annotation names {reference!r}, which is no category of this corpus"
        raise MalformedAutoError(msg)

    return category


def _features(entries: tuple[tuple[str, str], ...]) -> defs_records.FeatureMap | None:
    """Return the feature map a batch of entries is written as."""
    if not entries:
        return None

    return defs_records.FeatureMap(
        entries=tuple(
            defs_records.Feature(key=key, value=value) for key, value in entries
        )
    )


def _entries(features: defs_records.FeatureMap | None) -> tuple[tuple[str, str], ...]:
    """Return the entries a feature map states, in the order it states them."""
    if features is None:
        return ()

    return tuple((entry.key, entry.value) for entry in features.entries)


def _stated(features: defs_records.FeatureMap | None, key: str) -> str:
    """Return the value one feature states."""
    for stated_key, value in _entries(features):
        if stated_key == key:
            return value

    msg = f"an annotation states no {key!r}, which a node of this corpus states"
    raise MalformedAutoError(msg)


def _text_of(leaves: tuple[AutoLeaf, ...]) -> str:
    """Return the text an analysis's words spell, one space between each two."""
    return " ".join(leaf.word for leaf in leaves)


def _tokens(leaves: tuple[AutoLeaf, ...]) -> tuple[segmentation_records.Token, ...]:
    """Return the tokens an analysis's words become."""
    written: list[segmentation_records.Token] = []
    cursor = 0

    for position, leaf in enumerate(leaves):
        if position:
            cursor += 1

        width = len(leaf.word.encode("utf-8"))
        written.append(
            segmentation_records.Token(
                tokenIndex=position,
                text=leaf.word,
                textSpan=defs_records.Span(byteStart=cursor, byteEnd=cursor + width),
            )
        )
        cursor += width

    return tuple(written)


def _annotation(
    node: AutoNode,
    place: int,
    parent: int | None,
    children: Sequence[int],
    anchor: defs_records.Anchor | None,
) -> annotation_records.Annotation:
    """Return the annotation one node of an analysis is recorded as."""
    entries = (
        (
            (_POS_KEY, node.pos),
            (_ORIGINAL_POS_KEY, node.original_pos),
            (_SLOTTED_KEY, node.slotted),
        )
        if isinstance(node, AutoLeaf)
        else ((_HEAD_KEY, str(node.head)),)
    )

    return annotation_records.Annotation(
        uuid=defs_records.Uuid(value=_node_id(place)),
        ontologyTypeRef=_type_ref(node.category),
        parentId=None if parent is None else defs_records.Uuid(value=_node_id(parent)),
        childIds=tuple(defs_records.Uuid(value=_node_id(child)) for child in children),
        anchor=anchor,
        features=_features(entries),
    )


def _annotations(
    tree: AutoNode,
    tokenization: str,
) -> tuple[annotation_records.Annotation, ...]:
    """Return the annotations one analysis is recorded as."""
    placed = _placed(tree)
    daughters: dict[int, list[int]] = {}

    for place, (_, parent) in enumerate(placed):
        if parent is not None:
            daughters.setdefault(parent, []).append(place)

    written: list[annotation_records.Annotation] = []
    position = 0

    for place, (node, parent) in enumerate(placed):
        anchor = None

        if isinstance(node, AutoLeaf):
            anchor = defs_records.Anchor(
                tokenRef=defs_records.TokenRef(
                    tokenIndex=position,
                    tokenizationId=defs_records.Uuid(value=tokenization),
                )
            )
            position += 1

        written.append(
            _annotation(node, place, parent, daughters.get(place, ()), anchor)
        )

    return tuple(written)


def _records_of(derivation: AutoDerivation, index: int) -> tuple[FragmentRecord, ...]:
    """Return the three records one analysis is written as."""
    key = _key(index)
    leaves = _leaves(derivation.tree)
    expression = _uri(_EXPRESSION_NSID, key)
    tokenization = f"tokenization-{index}"
    written = (
        expression_records.Expression(
            id=key,
            kind="sentence",
            createdAt=_EPOCH,
            text=_text_of(leaves),
            features=_features(derivation.header),
        ),
        segmentation_records.Segmentation(
            createdAt=_EPOCH,
            expression=expression,
            tokenizations=(
                segmentation_records.Tokenization(
                    uuid=defs_records.Uuid(value=tokenization),
                    kind="penn-treebank",
                    tokens=_tokens(leaves),
                ),
            ),
        ),
        annotation_records.AnnotationLayer(
            createdAt=_EPOCH,
            expression=expression,
            kind="tree",
            subkind="ccg",
            formalism="ccgbank",
            tokenizationId=defs_records.Uuid(value=tokenization),
            annotations=_annotations(derivation.tree, tokenization),
        ),
    )
    nsids = (_EXPRESSION_NSID, _SEGMENTATION_NSID, _ANNOTATION_NSID)

    return tuple(
        FragmentRecord(
            local_id=_uri(nsid, key),
            nsid=nsid,
            value_json=record.model_dump_json(),
        )
        for nsid, record in zip(nsids, written, strict=True)
    )


class _Group(NamedTuple):
    """One derivation's fragment records by collection."""

    key: str
    held: Mapping[str, str]


def _grouped(records: Iterable[FragmentRecord]) -> tuple[_Group, ...]:
    """Return the records of a fragment grouped by the analysis they came from."""
    grouped: dict[str, dict[str, str]] = {}

    for record in records:
        named, marked, rest = record.local_id.removeprefix(f"{_SCHEME}/").rpartition(
            "/"
        )

        if not marked or named != record.nsid:
            msg = (
                f"a record is addressed at {record.local_id!r}, which is no "
                f"record of the {record.nsid} collection of this codec"
            )
            raise MalformedAutoError(msg)

        held = grouped.setdefault(rest, {})

        if record.nsid in held:
            msg = f"two {record.nsid} records are keyed {rest!r}"
            raise MalformedAutoError(msg)

        held[record.nsid] = record.value_json

    return tuple(_Group(key, held) for key, held in grouped.items())


def _held(group: _Group, nsid: str) -> str:
    """Return the text of one record of a group."""
    held = group.held.get(nsid)

    if held is None:
        msg = f"the records keyed {group.key!r} hold no {nsid} record"
        raise MalformedAutoError(msg)

    return held


def _node_of(
    annotation: annotation_records.Annotation,
    children: tuple[AutoNode, ...],
    words: Sequence[str],
) -> AutoNode:
    """Return the node one annotation stands for."""
    category = _category_of(annotation.ontologyTypeRef)

    if children:
        return AutoTree(category, _count_feature(annotation), children)

    anchor = annotation.anchor

    if anchor is None or anchor.tokenRef is None:
        msg = f"the annotation {annotation.uuid.value!r} has no daughter and no word"
        raise MalformedAutoError(msg)

    position = anchor.tokenRef.tokenIndex

    if not 0 <= position < len(words):
        msg = (
            f"the annotation {annotation.uuid.value!r} anchors position "
            f"{position}, which the analysis does not hold"
        )
        raise MalformedAutoError(msg)

    return AutoLeaf(
        category,
        _stated(annotation.features, _POS_KEY),
        _stated(annotation.features, _ORIGINAL_POS_KEY),
        words[position],
        _stated(annotation.features, _SLOTTED_KEY),
    )


def _count_feature(annotation: annotation_records.Annotation) -> int:
    """Return which daughter an annotation states its node takes its head from."""
    stated = _stated(annotation.features, _HEAD_KEY)

    try:
        return int(stated)
    except ValueError as error:
        msg = f"an annotation states the head {stated!r}, which is no daughter"
        raise MalformedAutoError(msg) from error


def _tree_from(
    layer: annotation_records.AnnotationLayer,
    words: Sequence[str],
) -> AutoNode:
    """Return the analysis one annotation layer records."""
    held = {annotation.uuid.value: annotation for annotation in layer.annotations}
    daughters = {
        annotation.uuid.value: tuple(child.value for child in annotation.childIds or ())
        for annotation in layer.annotations
    }
    roots = [
        annotation.uuid.value
        for annotation in layer.annotations
        if annotation.parentId is None
    ]

    if len(roots) != 1:
        msg = f"an annotation layer records {len(roots)} analyses rather than one"
        raise MalformedAutoError(msg)

    for named in daughters.values():
        for child in named:
            if child not in held:
                msg = f"an annotation layer names {child!r}, which it does not hold"
                raise MalformedAutoError(msg)

    built: dict[str, AutoNode] = {}
    stack: list[tuple[str, bool]] = [(roots[0], False)]

    while stack:
        identifier, walked = stack.pop()

        if walked:
            built[identifier] = _node_of(
                held[identifier],
                tuple(built[child] for child in daughters[identifier]),
                words,
            )

            continue

        stack.append((identifier, True))
        stack.extend((child, False) for child in reversed(daughters[identifier]))

    return built[roots[0]]


def _derivation_of(group: _Group) -> AutoDerivation:
    """Return the analysis one group of records was written from."""
    expression = expression_records.Expression.model_validate_json(
        _held(group, _EXPRESSION_NSID)
    )
    segmentation = segmentation_records.Segmentation.model_validate_json(
        _held(group, _SEGMENTATION_NSID)
    )
    layer = annotation_records.AnnotationLayer.model_validate_json(
        _held(group, _ANNOTATION_NSID)
    )
    tokenizations = segmentation.tokenizations

    if len(tokenizations) != 1:
        msg = (
            f"the records keyed {group.key!r} hold {len(tokenizations)} "
            "tokenizations rather than one"
        )
        raise MalformedAutoError(msg)

    words = [token.text or "" for token in tokenizations[0].tokens]

    return AutoDerivation(_entries(expression.features), _tree_from(layer, words))


class CcgbankAutoIso:
    """A translation from parsed AUTO derivations to corpus fragments."""

    __slots__ = ()

    def forward(self, source: AutoFile, /) -> CorpusFragment:
        """Return the fragment a batch of parsed analyses stands for."""
        return CorpusFragment(
            records=tuple(
                record
                for index, derivation in enumerate(source)
                for record in _records_of(derivation, index)
            ),
            source=NAME,
        )

    def backward(self, target: CorpusFragment, /) -> AutoFile:
        """Return the analyses a fragment was written from."""
        return tuple(_derivation_of(group) for group in _grouped(target.records))


@registers_as(NAME)
class CcgbankAutoCodec:
    """A CCGbank AUTO encoder and decoder."""

    __slots__ = ("_iso", "name")

    #: The name this codec is registered and reported under.
    name: str

    def __init__(self) -> None:
        self.name = NAME
        self._iso = CcgbankAutoIso()

    @property
    def iso(self) -> Iso[AutoFile, CorpusFragment]:
        """Return the translation between the parsed analyses and a fragment."""
        return self._iso

    def parse(self, src: str | bytes, /) -> AutoFile:
        """Parse AUTO text directly, without constructing corpus records.

        This is the lossless hot path for consumers that need the CCGbank trees
        themselves.  :meth:`decode` retains its corpus-fragment contract for
        registry clients and continues to translate the same parsed value.
        """
        return parse_auto(src)

    def decode(
        self,
        src: str | bytes,
        *,
        into: CorpusFragment | None = None,
    ) -> CorpusFragment:
        """Return the records a derivation file holds."""
        parsed = self.parse(src)

        if into is None:
            return self._iso.forward(parsed)

        offset = sum(1 for record in into.records if record.nsid == _EXPRESSION_NSID)

        return CorpusFragment(
            records=into.records
            + tuple(
                record
                for index, derivation in enumerate(parsed, start=offset)
                for record in _records_of(derivation, index)
            ),
            source=NAME,
        )

    def encode(self, records: Iterable[FragmentRecord]) -> str:
        """Return the derivation file a batch of records is written as."""
        held = CorpusFragment(records=tuple(records), source=NAME)

        return write_auto(self._iso.backward(held))
