"""Layers record export encoding and decoding."""

import json
from typing import TYPE_CHECKING, Final

from lairs.integrations.codecs import CorpusFragment, FragmentRecord

from lofi.data.registration import registers_as

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lofi._types import JsonValue
    from lofi.data import Iso

__all__ = [
    "NAME",
    "ExportLine",
    "LayersExportCodec",
    "LayersExportIso",
    "MalformedExportError",
]

type ExportLine = dict[str, JsonValue]
"""A parsed export line containing an AT-URI and value."""

NAME: Final = "layers-export"
"""The registered codec name."""

_SCHEME: Final = "at://"
"""The AT-URI prefix."""

_URI_KEY: Final = "uri"
"""The export key for a record's AT-URI."""

_VALUE_KEY: Final = "value"
"""The export key for a record's value."""

_LINE_KEYS: Final = frozenset({_URI_KEY, _VALUE_KEY})
"""The required export-line keys."""

_URI_SEGMENTS: Final = 3
"""The number of segments after an AT-URI scheme."""


class MalformedExportError(ValueError):
    """An error for a malformed Layers export line."""


def _canonical(value: dict[str, JsonValue]) -> str:
    """Return canonical JSON text for a record value.

    Parameters
    ----------
    value : dict[str, JsonValue]
        The record value.

    Returns
    -------
    str
        The value as JSON, keys sorted and separators tightened.

    Raises
    ------
    ValueError
        If the value holds a float JSON cannot write, which is any infinity
        and any not-a-number.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _collection(uri: str) -> str:
    """Return the record collection an AT-URI names.

    Parameters
    ----------
    uri : str
        The record's AT-URI.

    Returns
    -------
    str
        The collection, which is the segment between the authority and the
        record key.

    Raises
    ------
    MalformedExportError
        If the URI does not begin with the AT scheme, or if it does not name
        an authority, a collection and a record key.
    """
    if not uri.startswith(_SCHEME):
        msg = f"a record is identified by {uri!r}, which is not an AT-URI"
        raise MalformedExportError(msg)

    segments = uri.removeprefix(_SCHEME).split("/")

    if len(segments) != _URI_SEGMENTS or not all(segments):
        msg = (
            f"the AT-URI {uri!r} does not name an authority, a collection "
            "and a record key"
        )
        raise MalformedExportError(msg)

    # the AT-URI collection lies between the authority and key
    return segments[1]


def _record_of(line: ExportLine) -> FragmentRecord:
    """Return the fragment record one parsed line stands for.

    Parameters
    ----------
    line : ExportLine
        The parsed line.

    Returns
    -------
    FragmentRecord
        The record, holding the URI as its local id, the collection the URI
        names, and the value as canonical JSON text.

    Raises
    ------
    MalformedExportError
        If the line names other than the two expected keys, if the URI is not
        a string, or if the value is not a JSON object.
    """
    if set(line) != _LINE_KEYS:
        msg = (
            f"an exported record names {sorted(line)} rather than {sorted(_LINE_KEYS)}"
        )
        raise MalformedExportError(msg)

    uri = line[_URI_KEY]

    if not isinstance(uri, str):
        msg = f"a record is identified by {uri!r}, which is not a string"
        raise MalformedExportError(msg)

    value = line[_VALUE_KEY]

    if not isinstance(value, dict):
        msg = f"the record at {uri} holds {value!r}, which is not a JSON object"
        raise MalformedExportError(msg)

    return FragmentRecord(
        local_id=uri,
        nsid=_collection(uri),
        value_json=_canonical(value),
    )


def _line_of(record: FragmentRecord) -> ExportLine:
    """Return the line a fragment record is written as.

    Parameters
    ----------
    record : FragmentRecord
        The record to write.

    Returns
    -------
    ExportLine
        The line, naming the record's local id as its URI and its parsed value.

    Raises
    ------
    MalformedExportError
        If the record's value text is not JSON.
    """
    try:
        value = json.loads(record.value_json)
    except json.JSONDecodeError as error:
        msg = f"the record at {record.local_id} holds a value that is not JSON"
        raise MalformedExportError(msg) from error

    return {_URI_KEY: record.local_id, _VALUE_KEY: value}


def _parsed(line: str, number: int) -> ExportLine:
    """Return one line of an export, parsed.

    Parameters
    ----------
    line : str
        The text of the line.
    number : int
        Which line of the export it is, counting from one.

    Returns
    -------
    ExportLine
        The parsed line.

    Raises
    ------
    MalformedExportError
        If the line is not JSON, or if it is not a JSON object.
    """
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as error:
        msg = f"line {number} of the export is not JSON"
        raise MalformedExportError(msg) from error

    if not isinstance(parsed, dict):
        msg = f"line {number} of the export holds {parsed!r} rather than a record"
        raise MalformedExportError(msg)

    return parsed


def _lines(src: str | bytes) -> tuple[ExportLine, ...]:
    """Return every record an export's text holds, parsed and in order.

    Parameters
    ----------
    src : str | bytes
        The export, as text or as UTF-8 bytes.

    Returns
    -------
    tuple[ExportLine, ...]
        The parsed lines.

    Raises
    ------
    MalformedExportError
        If some line is not a JSON object.
    UnicodeDecodeError
        If the bytes are not UTF-8.
    """
    text = src.decode("utf-8") if isinstance(src, bytes) else src

    return tuple(
        _parsed(line, number)
        for number, line in enumerate(text.split("\n"), start=1)
        if line.strip()
    )


class LayersExportIso:
    """A translation from parsed export lines to corpus fragments."""

    __slots__ = ()

    def forward(self, source: tuple[ExportLine, ...], /) -> CorpusFragment:
        """Return the fragment a batch of parsed lines stands for.

        Parameters
        ----------
        source : tuple[ExportLine, ...]
            The parsed lines, in the order they were written.

        Returns
        -------
        CorpusFragment
            The fragment holding one record per line.

        Raises
        ------
        MalformedExportError
            If some line is not a record.
        """
        return CorpusFragment(
            records=tuple(_record_of(line) for line in source),
            source=NAME,
        )

    def backward(self, target: CorpusFragment, /) -> tuple[ExportLine, ...]:
        """Return the lines a fragment is written as.

        Parameters
        ----------
        target : CorpusFragment
            The fragment to write.

        Returns
        -------
        tuple[ExportLine, ...]
            One line per record, in the order the fragment holds them.

        Raises
        ------
        MalformedExportError
            If some record's value text is not JSON.
        """
        return tuple(_line_of(record) for record in target.records)


@registers_as(NAME)
class LayersExportCodec:
    """A Layers record export encoder and decoder."""

    __slots__ = ("_iso", "name")

    name: str

    def __init__(self) -> None:
        self.name = NAME
        self._iso = LayersExportIso()

    @property
    def iso(self) -> Iso[tuple[ExportLine, ...], CorpusFragment]:
        """Return the translation between the parsed lines and a fragment."""
        return self._iso

    def decode(
        self,
        src: str | bytes,
        *,
        into: CorpusFragment | None = None,
    ) -> CorpusFragment:
        """Return the records an export holds.

        Parameters
        ----------
        src : str | bytes
            The export, as text or as UTF-8 bytes.
        into : CorpusFragment | None
            A fragment whose records the decoded ones follow, where a caller is
            reading a corpus one file at a time.

        Returns
        -------
        CorpusFragment
            The fragment holding the decoded records.

        Raises
        ------
        MalformedExportError
            If some line is not a record.
        """
        decoded = self._iso.forward(_lines(src))

        if into is None:
            return decoded

        return CorpusFragment(
            records=into.records + decoded.records,
            source=NAME,
        )

    def encode(self, records: Iterable[FragmentRecord]) -> str:
        """Return the export text a batch of records is written as.

        Parameters
        ----------
        records : Iterable[FragmentRecord]
            The records to write out, in the order they are to be read back.

        Returns
        -------
        str
            One JSON object per line, each line ending in a newline. Every
            character outside ASCII is escaped, so no value can hold something
            a reader would take for the end of a line.

        Raises
        ------
        MalformedExportError
            If some record's value text is not JSON.
        """
        held = CorpusFragment(records=tuple(records), source=NAME)

        return "".join(
            f"{json.dumps(line, sort_keys=True)}\n" for line in self._iso.backward(held)
        )
