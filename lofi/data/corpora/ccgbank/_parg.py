"""CCGbank PARG dependency encoding and decoding."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, NamedTuple

from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from lairs.records import annotation as annotation_records
from lairs.records import defs as defs_records
from lairs.records import expression as expression_records

from lofi.data.corpora.ccgbank._parg_parse import (
    MalformedPargError,
    PargDependency,
    PargFile,
    PargSentence,
    parse_parg,
    validate_parg_sentence,
    write_parg,
)
from lofi.data.registration import registers_as

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lofi.data import Iso

__all__ = [
    "PARG_NAME",
    "PargCodec",
    "PargIso",
]

PARG_NAME: Final = "ccgbank-parg"

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_SCHEME: Final = "at://local"
_EXPRESSION_NSID: Final = "pub.layers.expression.expression"
_ANNOTATION_NSID: Final = "pub.layers.annotation.annotationLayer"
_HEADER_LAST_KEY: Final = "headerLastIndex"
_HEAD_CATEGORY_KEY: Final = "headLexicalCategory"
_ARGUMENT_SLOT_KEY: Final = "argumentSlot"
_TARGET_WORD_KEY: Final = "targetWord"
_HEAD_WORD_KEY: Final = "headWord"
_MARKER_KEY: Final = "marker"
_DEPENDENCY_KEYS: Final = frozenset(
    (
        _HEAD_CATEGORY_KEY,
        _ARGUMENT_SLOT_KEY,
        _TARGET_WORD_KEY,
        _HEAD_WORD_KEY,
        _MARKER_KEY,
    )
)


def _key(index: int) -> str:
    """Return the key for one sentence's records."""
    return f"sentence-{index}"


def _uri(nsid: str, key: str) -> str:
    """Return the local URI for a record."""
    return f"{_SCHEME}/{nsid}/{key}"


def _features(entries: tuple[tuple[str, str], ...]) -> defs_records.FeatureMap:
    """Build a feature map."""
    return defs_records.FeatureMap(
        entries=tuple(
            defs_records.Feature(key=key, value=value) for key, value in entries
        )
    )


def _entries(features: defs_records.FeatureMap | None) -> tuple[tuple[str, str], ...]:
    """Read a feature map."""
    if features is None:
        return ()

    return tuple((entry.key, entry.value) for entry in features.entries)


def _feature_values(
    features: defs_records.FeatureMap | None,
    allowed: frozenset[str],
    source: str,
) -> dict[str, str]:
    """Return unique allowed features."""
    values: dict[str, str] = {}

    for key, value in _entries(features):
        if key not in allowed:
            msg = f"{source} states the unknown feature {key!r}"
            raise MalformedPargError(msg)

        if key in values:
            msg = f"{source} states feature {key!r} twice"
            raise MalformedPargError(msg)

        values[key] = value

    return values


def _required(values: Mapping[str, str], key: str, source: str) -> str:
    """Return a required feature."""
    try:
        return values[key]
    except KeyError as error:
        msg = f"{source} states no {key!r} feature"
        raise MalformedPargError(msg) from error


def _annotation(
    dependency: PargDependency, index: int
) -> annotation_records.Annotation:
    """Build a dependency annotation."""
    entries = (
        (_HEAD_CATEGORY_KEY, dependency.head_category),
        (_ARGUMENT_SLOT_KEY, str(dependency.argument_slot)),
        (_TARGET_WORD_KEY, dependency.target_word),
        (_HEAD_WORD_KEY, dependency.head_word),
    )

    if dependency.marker is not None:
        entries += ((_MARKER_KEY, dependency.marker),)

    return annotation_records.Annotation(
        uuid=defs_records.Uuid(value=f"dependency-{index}"),
        headIndex=dependency.head_index,
        targetIndex=dependency.target_index,
        features=_features(entries),
    )


def _records_of(sentence: PargSentence, index: int) -> tuple[FragmentRecord, ...]:
    """Build the records for one sentence."""
    validate_parg_sentence(sentence, index)
    key = _key(index)
    expression = _uri(_EXPRESSION_NSID, key)
    records = (
        expression_records.Expression(
            id=sentence.identifier,
            kind="sentence",
            createdAt=_EPOCH,
            features=_features(((_HEADER_LAST_KEY, str(sentence.last_index)),)),
        ),
        annotation_records.AnnotationLayer(
            createdAt=_EPOCH,
            expression=expression,
            kind="relation",
            subkind="dependency",
            formalism="ccgbank",
            annotations=tuple(
                _annotation(dependency, position)
                for position, dependency in enumerate(sentence.dependencies)
            ),
        ),
    )
    nsids = (_EXPRESSION_NSID, _ANNOTATION_NSID)

    return tuple(
        FragmentRecord(
            local_id=_uri(nsid, key),
            nsid=nsid,
            value_json=record.model_dump_json(),
        )
        for nsid, record in zip(nsids, records, strict=True)
    )


class _Group(NamedTuple):
    """One sentence's fragment records by collection."""

    key: str
    held: Mapping[str, str]


def _grouped(records: Iterable[FragmentRecord]) -> tuple[_Group, ...]:
    """Group records by sentence."""
    grouped: dict[str, dict[str, str]] = {}

    for record in records:
        if record.nsid not in {_EXPRESSION_NSID, _ANNOTATION_NSID}:
            msg = f"the codec does not accept records from {record.nsid!r}"
            raise MalformedPargError(msg)

        named, marked, key = record.local_id.removeprefix(f"{_SCHEME}/").rpartition("/")

        if not marked or named != record.nsid:
            msg = f"record {record.local_id!r} is not in its stated collection"
            raise MalformedPargError(msg)

        held = grouped.setdefault(key, {})

        if record.nsid in held:
            msg = f"sentence {key!r} has two {record.nsid} records"
            raise MalformedPargError(msg)

        held[record.nsid] = record.value_json

    return tuple(_Group(key, held) for key, held in grouped.items())


def _held(group: _Group, nsid: str) -> str:
    """Return a required record."""
    try:
        return group.held[nsid]
    except KeyError as error:
        msg = f"sentence {group.key!r} has no {nsid} record"
        raise MalformedPargError(msg) from error


def _number(value: str, description: str) -> int:
    """Parse an integer stored in a feature."""
    if not value.isascii() or not value.isdecimal():
        msg = f"{description} is {value!r} rather than a nonnegative integer"
        raise MalformedPargError(msg)

    return int(value)


def _dependency_of(annotation: annotation_records.Annotation) -> PargDependency:
    """Read one dependency annotation."""
    source = f"annotation {annotation.uuid.value!r}"
    values = _feature_values(annotation.features, _DEPENDENCY_KEYS, source)

    if annotation.targetIndex is None or annotation.headIndex is None:
        msg = f"{source} has no target and head indices"
        raise MalformedPargError(msg)

    marker = values.get(_MARKER_KEY)
    return PargDependency(
        target_index=annotation.targetIndex,
        head_index=annotation.headIndex,
        head_category=_required(values, _HEAD_CATEGORY_KEY, source),
        argument_slot=_number(
            _required(values, _ARGUMENT_SLOT_KEY, source),
            f"{source}'s argument slot",
        ),
        target_word=_required(values, _TARGET_WORD_KEY, source),
        head_word=_required(values, _HEAD_WORD_KEY, source),
        marker=marker,
    )


def _sentence_of(group: _Group) -> PargSentence:
    """Read one sentence from fragment records."""
    expression = expression_records.Expression.model_validate_json(
        _held(group, _EXPRESSION_NSID)
    )
    layer = annotation_records.AnnotationLayer.model_validate_json(
        _held(group, _ANNOTATION_NSID)
    )
    source = f"sentence {group.key!r}"
    values = _feature_values(
        expression.features,
        frozenset((_HEADER_LAST_KEY,)),
        source,
    )
    last_index = _number(
        _required(values, _HEADER_LAST_KEY, source),
        f"{source}'s header last index",
    )

    if layer.expression != _uri(_EXPRESSION_NSID, group.key):
        msg = f"{source}'s dependency layer refers to another expression"
        raise MalformedPargError(msg)

    if (layer.kind, layer.subkind, layer.formalism) != (
        "relation",
        "dependency",
        "ccgbank",
    ):
        msg = f"{source} is not a CCGbank dependency relation layer"
        raise MalformedPargError(msg)

    identifiers = [annotation.uuid.value for annotation in layer.annotations]

    if len(set(identifiers)) != len(identifiers):
        msg = f"{source} repeats a dependency identifier"
        raise MalformedPargError(msg)

    sentence = PargSentence(
        identifier=expression.id,
        last_index=last_index,
        dependencies=tuple(
            _dependency_of(annotation) for annotation in layer.annotations
        ),
    )
    validate_parg_sentence(sentence, 0)

    return sentence


class PargIso:
    """A translation from PARG sentences to corpus fragments."""

    __slots__ = ()

    def forward(self, source: PargFile, /) -> CorpusFragment:
        """Translate parsed sentences to fragment records."""
        return CorpusFragment(
            records=tuple(
                record
                for index, sentence in enumerate(source)
                for record in _records_of(sentence, index)
            ),
            source=PARG_NAME,
        )

    def backward(self, target: CorpusFragment, /) -> PargFile:
        """Translate fragment records to parsed sentences."""
        return tuple(_sentence_of(group) for group in _grouped(target.records))


@registers_as(PARG_NAME)
class PargCodec:
    """A CCGbank PARG encoder and decoder."""

    __slots__ = ("_iso", "name")

    name: str

    def __init__(self) -> None:
        self.name = PARG_NAME
        self._iso = PargIso()

    @property
    def iso(self) -> Iso[PargFile, CorpusFragment]:
        """Return the PARG-to-fragment translation."""
        return self._iso

    def parse(self, src: str | bytes, /) -> PargFile:
        """Parse PARG text into its sentences, without constructing corpus records."""
        return parse_parg(src)

    def decode(
        self,
        src: str | bytes,
        *,
        into: CorpusFragment | None = None,
    ) -> CorpusFragment:
        """Decode PARG text into fragment records."""
        parsed = self.parse(src)

        if into is None:
            return self._iso.forward(parsed)

        offset = sum(1 for record in into.records if record.nsid == _EXPRESSION_NSID)

        return CorpusFragment(
            records=into.records
            + tuple(
                record
                for index, sentence in enumerate(parsed, start=offset)
                for record in _records_of(sentence, index)
            ),
            source=PARG_NAME,
        )

    def encode(self, records: Iterable[FragmentRecord]) -> str:
        """Encode fragment records as PARG text."""
        held = CorpusFragment(records=tuple(records), source=PARG_NAME)

        return write_parg(self._iso.backward(held))
