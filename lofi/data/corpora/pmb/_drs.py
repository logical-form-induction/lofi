"""PMB clausal-form DRS encoding and decoding."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, Self

from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from lairs.records import defs as defs_records
from lairs.records import expression as expression_records
from lairs.records import graph as graph_records
from lairs.records import segmentation as segmentation_records

from lofi.data.registration import registers_as

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi.data import Iso

__all__ = [
    "DRS_NAME",
    "MalformedPmbDrsError",
    "PmbClause",
    "PmbDrs",
    "PmbDrsCodec",
    "PmbDrsFile",
    "PmbDrsIso",
    "PmbToken",
]

DRS_NAME: Final = "pmb-drs"
"""The PMB DRS codec name."""

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
"""The timestamp used for locally decoded records."""

_SCHEME: Final = "at://local"
"""The local record URI prefix."""

_EXPRESSION_NSID: Final = "pub.layers.expression.expression"
"""The expression-record collection."""

_SEGMENTATION_NSID: Final = "pub.layers.segmentation.segmentation"
"""The segmentation-record collection."""

_GRAPH_NODE_NSID: Final = "pub.layers.graph.graphNode"
"""The graph-node collection."""

_GRAPH_EDGE_NSID: Final = "pub.layers.graph.graphEdge"
"""The graph-edge collection."""

_TOKENIZATION_KEY: Final = "pmb.tokenizationId"
"""The feature naming a graph record's tokenization."""

_GROUNDING_INDEX_PREFIX: Final = "pmb.tokenIndex."
"""The feature prefix for graph-record token positions."""

_KIND_KEY: Final = "pmb.kind"
"""The feature distinguishing PMB graph record roles."""

_VARIABLE_KEY: Final = "pmb.variable"
"""The node feature containing a PMB variable."""

_BOX_KEY: Final = "pmb.box"
"""The semantic-edge feature containing its DRS box."""

_CLAUSE_ORDINAL_KEY: Final = "pmb.clauseOrdinal"
"""The semantic-edge feature containing its source position."""

_ARGUMENT_COUNT_KEY: Final = "pmb.argumentCount"
"""The semantic-edge feature containing its argument count."""

_ARGUMENT_PREFIX: Final = "pmb.argument."
"""The semantic-edge feature prefix for ordered arguments."""

_INDEX_WIDTH: Final = 8
"""The width of indices embedded in feature keys and record ids."""

_ANCHOR_PATTERN: Final = re.compile(
    r"(?:^|\s)(?P<surface>.+?) \[(?P<start>\d+)\.\.\.(?P<end>\d+)\](?=\s|$)"
)
"""The PMB token-alignment syntax."""

_DOCUMENT_KEY_PATTERN: Final = re.compile(r"drs-(?P<index>\d+)")
"""The local key for one DRS document."""

_NODE_KEY_PATTERN: Final = re.compile(r"(?P<document>drs-\d+)-node-(?P<index>\d+)")
"""The local key for one DRS graph node."""

_CLAUSE_KEY_PATTERN: Final = re.compile(r"(?P<document>drs-\d+)-clause-(?P<index>\d+)")
"""The local key for one DRS clause edge."""

_ATTACHMENT_KEY_PATTERN: Final = re.compile(r"(?P<document>drs-\d+)-attachment")
"""The local key for one graph attachment edge."""

_CONNECTIVITY_KEY_PATTERN: Final = re.compile(
    r"(?P<document>drs-\d+)-connectivity-(?P<index>\d+)"
)
"""The local key for one graph-connectivity edge."""


class MalformedPmbDrsError(ValueError):
    """Error raised for malformed PMB clausal form or graph records."""


class PmbToken(NamedTuple):
    """A PMB token with half-open character offsets."""

    surface: str
    start: int
    end: int


class PmbClause(NamedTuple):
    """A PMB DRS clause and its zero-based token positions."""

    box: str
    operator: str
    arguments: tuple[str, ...]
    token_indices: tuple[int, ...]


class PmbDrs(NamedTuple):
    """One parsed PMB clausal-form DRS."""

    text: str
    tokens: tuple[PmbToken, ...]
    clauses: tuple[PmbClause, ...]


type PmbDrsFile = tuple[PmbDrs, ...]
"""The DRS documents in one clausal-form source."""


class _RawClause(NamedTuple):
    """A parsed clause before token positions are assigned."""

    box: str
    operator: str
    arguments: tuple[str, ...]
    anchors: tuple[tuple[int, int], ...]
    line: int


class _DocumentRecords(NamedTuple):
    """One DRS document's fragment records."""

    index: int
    records: tuple[FragmentRecord, ...]


class _JsonModel(Protocol):
    """A Layers record model that validates JSON."""

    @classmethod
    def model_validate_json(cls, raw: str | bytes | bytearray) -> Self:
        """Validate one JSON-encoded record."""
        ...


def _model_from_json[M: _JsonModel](
    model: type[M],
    record: FragmentRecord,
) -> M:
    """Return a validated Layers model for one fragment record.

    Parameters
    ----------
    model : type[M]
        Layers model to validate.
    record : FragmentRecord
        Fragment record containing the JSON value.

    Raises
    ------
    MalformedPmbDrsError
        If the record is not valid JSON for the requested model.
    """
    try:
        return model.model_validate_json(record.value_json)
    except Exception as error:
        msg = f"record {record.local_id!r} has invalid Layers record JSON"
        raise MalformedPmbDrsError(msg) from error


def _source_text(src: str | bytes) -> str:
    """Return source text decoded as UTF-8.

    Parameters
    ----------
    src : str | bytes
        PMB clausal-form text or bytes.

    Raises
    ------
    MalformedPmbDrsError
        If byte input is not UTF-8.
    """
    if isinstance(src, str):
        return src

    try:
        return src.decode("utf-8")
    except UnicodeDecodeError as error:
        msg = "PMB DRS input is not UTF-8"
        raise MalformedPmbDrsError(msg) from error


def _anchors(comment: str, line: int) -> tuple[tuple[PmbToken, tuple[int, int]], ...]:
    """Return token alignments from a PMB comment.

    Parameters
    ----------
    comment : str
        Text following the comment marker.
    line : int
        Source line number.

    Raises
    ------
    MalformedPmbDrsError
        If the comment has no alignment or has an invalid or repeated span.
    """
    found: list[tuple[PmbToken, tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()

    for match in _ANCHOR_PATTERN.finditer(comment.strip()):
        surface = match.group("surface")
        start = int(match.group("start"))
        end = int(match.group("end"))
        span = (start, end)

        if end <= start or end - start != len(surface):
            msg = (
                f"line {line} has an invalid token alignment "
                f"{surface!r} [{start}...{end}]"
            )
            raise MalformedPmbDrsError(msg)

        if span in seen:
            msg = f"line {line} repeats token alignment [{start}...{end}]"
            raise MalformedPmbDrsError(msg)

        seen.add(span)
        found.append((PmbToken(surface, start, end), span))

    if not found:
        msg = f"line {line} has no token alignment"
        raise MalformedPmbDrsError(msg)

    return tuple(found)


def _clause(line: str, number: int) -> tuple[_RawClause, tuple[PmbToken, ...]]:
    """Return one parsed PMB clause and its aligned tokens.

    Parameters
    ----------
    line : str
        Source line.
    number : int
        Source line number.

    Raises
    ------
    MalformedPmbDrsError
        If the clause has invalid fields, quotes, token alignment, or
        arity.
    """
    body, marked, comment = line.partition("%")

    if not marked:
        msg = f"line {number} has no token-alignment comment"
        raise MalformedPmbDrsError(msg)

    fields = body.split()

    if len(fields) not in {3, 4}:
        msg = f"line {number} has {len(fields)} clause fields rather than three or four"
        raise MalformedPmbDrsError(msg)

    if any('"' in field for field in fields[:2]):
        msg = f"line {number} has a quoted box or operator"
        raise MalformedPmbDrsError(msg)

    for argument in fields[2:]:
        quoted = argument.startswith('"') or argument.endswith('"')

        if quoted and re.fullmatch(r'"[^"]+"', argument) is None:
            msg = f"line {number} has malformed quoted argument {argument!r}"
            raise MalformedPmbDrsError(msg)

        if not quoted and '"' in argument:
            msg = f"line {number} has malformed argument {argument!r}"
            raise MalformedPmbDrsError(msg)

    aligned = _anchors(comment, number)
    return (
        _RawClause(
            fields[0],
            fields[1],
            tuple(fields[2:]),
            tuple(span for _, span in aligned),
            number,
        ),
        tuple(token for token, _ in aligned),
    )


def _tokens(
    candidates: Sequence[PmbToken],
    line: int,
) -> tuple[PmbToken, ...]:
    """Return unique source tokens in character order.

    Parameters
    ----------
    candidates : Sequence[PmbToken]
        Tokens named by clause and token-only comments.
    line : int
        Last source line in the document.

    Raises
    ------
    MalformedPmbDrsError
        If two alignments disagree or overlap.
    """
    by_span: dict[tuple[int, int], str] = {}

    for token in candidates:
        span = (token.start, token.end)
        prior = by_span.get(span)

        if prior is not None and prior != token.surface:
            msg = f"line {line} assigns two tokens to [{token.start}...{token.end}]"
            raise MalformedPmbDrsError(msg)

        by_span[span] = token.surface

    ordered = tuple(
        PmbToken(surface, start, end)
        for (start, end), surface in sorted(by_span.items())
    )
    previous = 0

    for token in ordered:
        if token.start < previous:
            msg = f"line {line} has overlapping token alignments at {token.start}"
            raise MalformedPmbDrsError(msg)

        previous = token.end

    return ordered


def _surface(tokens: Sequence[PmbToken]) -> str:
    """Return the expression reconstructed from aligned tokens."""
    parts: list[str] = []
    cursor = 0

    for token in tokens:
        parts.append(" " * (token.start - cursor))
        parts.append(token.surface)
        cursor = token.end

    return "".join(parts)


def _document(
    raw_clauses: Sequence[_RawClause],
    candidates: Sequence[PmbToken],
    line: int,
) -> PmbDrs:
    """Return one parsed DRS document.

    Parameters
    ----------
    raw_clauses : Sequence[_RawClause]
        Clauses before token positions are assigned.
    candidates : Sequence[PmbToken]
        Tokens named by the document's comments.
    line : int
        Last source line in the document.

    Raises
    ------
    MalformedPmbDrsError
        If token alignments disagree, overlap, or name no document token.
    """
    tokens = _tokens(candidates, line)
    positions = {(token.start, token.end): index for index, token in enumerate(tokens)}
    clauses: list[PmbClause] = []

    for raw in raw_clauses:
        try:
            indices = tuple(positions[span] for span in raw.anchors)
        except KeyError as error:
            msg = f"line {raw.line} names a token outside its document"
            raise MalformedPmbDrsError(msg) from error

        clauses.append(PmbClause(raw.box, raw.operator, raw.arguments, indices))

    return PmbDrs(_surface(tokens), tokens, tuple(clauses))


def _parse(src: str | bytes) -> PmbDrsFile:
    """Return DRS documents parsed from PMB clausal form.

    Parameters
    ----------
    src : str | bytes
        PMB clausal-form text or bytes.

    Raises
    ------
    MalformedPmbDrsError
        If the source has malformed clauses, token alignment, or document boundaries.
    """
    documents: list[PmbDrs] = []
    clauses: list[_RawClause] = []
    candidates: list[PmbToken] = []
    last_line = 0

    def finish() -> None:
        if clauses:
            documents.append(_document(clauses, candidates, last_line))
            clauses.clear()
            candidates.clear()
        elif candidates:
            msg = f"document ending at line {last_line} has tokens but no clauses"
            raise MalformedPmbDrsError(msg)

    for number, line in enumerate(_source_text(src).splitlines(), start=1):
        last_line = number
        stripped = line.strip()

        if not stripped:
            finish()
            continue

        if stripped.startswith("%%%"):
            continue

        if stripped.startswith("%"):
            candidates.extend(
                token for token, _ in _anchors(stripped.removeprefix("%"), number)
            )
            continue

        raw, aligned = _clause(line, number)
        clauses.append(raw)
        candidates.extend(aligned)

    finish()

    if not documents:
        msg = "PMB DRS input has no clauses"
        raise MalformedPmbDrsError(msg)

    return tuple(documents)


def _alignment_text(tokens: Sequence[PmbToken], indices: Sequence[int]) -> str:
    """Return canonical PMB alignment text."""
    return " ".join(
        f"{tokens[index].surface} [{tokens[index].start}...{tokens[index].end}]"
        for index in indices
    )


def _write(source: PmbDrsFile) -> str:
    """Return canonical PMB clausal form."""
    blocks: list[str] = []

    for document in source:
        lines = [f"%%% {document.text}"]
        referenced: set[int] = set()

        for clause in document.clauses:
            referenced.update(clause.token_indices)
            fields = (clause.box, clause.operator, *clause.arguments)
            lines.append(
                f"{' '.join(fields)} % "
                f"{_alignment_text(document.tokens, clause.token_indices)}"
            )

        lines.extend(
            f"% {_alignment_text(document.tokens, (index,))}"
            for index in sorted(set(range(len(document.tokens))) - referenced)
        )

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + "\n"


def _features(entries: Sequence[tuple[str, str]]) -> defs_records.FeatureMap:
    """Return a Layers feature map with unique keys.

    Parameters
    ----------
    entries : Sequence[tuple[str, str]]
        Ordered feature entries.

    Raises
    ------
    MalformedPmbDrsError
        If a feature key repeats.
    """
    keys = tuple(key for key, _ in entries)

    if len(set(keys)) != len(keys):
        msg = "a PMB DRS feature map repeats a key"
        raise MalformedPmbDrsError(msg)

    return defs_records.FeatureMap(
        entries=tuple(
            defs_records.Feature(key=key, value=value) for key, value in entries
        )
    )


def _feature_entries(
    features: defs_records.FeatureMap | None,
    subject: str,
) -> tuple[tuple[str, str], ...]:
    """Return unique feature entries.

    Parameters
    ----------
    features : defs_records.FeatureMap | None
        Feature map to read.
    subject : str
        Record named in error messages.

    Raises
    ------
    MalformedPmbDrsError
        If the feature map is absent or repeats a key.
    """
    if features is None:
        msg = f"{subject} has no PMB properties"
        raise MalformedPmbDrsError(msg)

    entries = tuple((entry.key, entry.value) for entry in features.entries)
    keys = tuple(key for key, _ in entries)

    if len(set(keys)) != len(keys):
        msg = f"{subject} repeats a PMB property"
        raise MalformedPmbDrsError(msg)

    return entries


def _mapping(
    features: defs_records.FeatureMap | None,
    subject: str,
) -> Mapping[str, str]:
    """Return a record's unique PMB properties by key.

    Parameters
    ----------
    features : defs_records.FeatureMap | None
        Feature map to read.
    subject : str
        Record named in error messages.

    Raises
    ------
    MalformedPmbDrsError
        If the feature map is absent or repeats a key.
    """
    return dict(_feature_entries(features, subject))


def _index_key(prefix: str, index: int) -> str:
    """Return a zero-padded indexed feature key."""
    return f"{prefix}{index:0{_INDEX_WIDTH}d}"


def _anchor_entries(
    tokenization: str,
    indices: Sequence[int],
) -> tuple[tuple[str, str], ...]:
    """Return canonical graph grounding properties.

    Parameters
    ----------
    tokenization : str
        Tokenization identifier.
    indices : Sequence[int]
        Anchored token positions.

    Raises
    ------
    MalformedPmbDrsError
        If no position is supplied or a position repeats.
    """
    if not indices:
        msg = "a PMB graph record has no token position"
        raise MalformedPmbDrsError(msg)

    if len(set(indices)) != len(indices):
        msg = "a PMB graph record repeats a token position"
        raise MalformedPmbDrsError(msg)

    return (
        (_TOKENIZATION_KEY, tokenization),
        *(
            (_index_key(_GROUNDING_INDEX_PREFIX, index), str(index))
            for index in sorted(indices)
        ),
    )


def _document_key(index: int) -> str:
    """Return one document's local record key."""
    return f"drs-{index}"


def _uri(nsid: str, key: str) -> str:
    """Return a local Layers record URI."""
    return f"{_SCHEME}/{nsid}/{key}"


def _variables(document: PmbDrs) -> tuple[tuple[str, str], ...]:
    """Return PMB variables and their graph kinds in source order."""
    boxes = {clause.box for clause in document.clauses}
    found: dict[str, str] = {}

    for clause in document.clauses:
        found.setdefault(clause.box, "box")

        for argument in clause.arguments:
            if argument.startswith('"') and argument.endswith('"'):
                continue

            found.setdefault(argument, "box" if argument in boxes else "referent")

    return tuple(found.items())


def _node_anchors(document: PmbDrs) -> Mapping[str, tuple[int, ...]]:
    """Return the union of clause positions mentioning each variable."""
    variables = dict(_variables(document))
    anchors: dict[str, set[int]] = {variable: set() for variable in variables}

    for clause in document.clauses:
        mentioned = (clause.box, *clause.arguments)

        for variable in mentioned:
            if variable in anchors:
                anchors[variable].update(clause.token_indices)

    return {variable: tuple(sorted(indices)) for variable, indices in anchors.items()}


def _tokens_of(
    document: PmbDrs,
) -> tuple[segmentation_records.Token, ...]:
    """Return Layers tokens with UTF-8 byte spans."""
    return tuple(
        segmentation_records.Token(
            tokenIndex=index,
            text=token.surface,
            textSpan=defs_records.Span(
                byteStart=len(document.text[: token.start].encode("utf-8")),
                byteEnd=len(document.text[: token.end].encode("utf-8")),
            ),
        )
        for index, token in enumerate(document.tokens)
    )


def _node_records(
    document: PmbDrs,
    key: str,
    tokenization: str,
) -> tuple[FragmentRecord, ...]:
    """Return graph-node records for one DRS."""
    anchors = _node_anchors(document)
    return tuple(
        FragmentRecord(
            local_id=_uri(
                _GRAPH_NODE_NSID,
                f"{key}-node-{index:0{_INDEX_WIDTH}d}",
            ),
            nsid=_GRAPH_NODE_NSID,
            value_json=graph_records.GraphNode(
                createdAt=_EPOCH,
                nodeType="custom",
                label=variable,
                properties=_features(
                    (
                        (_KIND_KEY, kind),
                        (_VARIABLE_KEY, variable),
                        *_anchor_entries(tokenization, anchors[variable]),
                    )
                ),
            ).model_dump_json(),
        )
        for index, (variable, kind) in enumerate(_variables(document))
    )


def _node_uris(document: PmbDrs, key: str) -> Mapping[str, str]:
    """Return graph-node URIs by PMB variable."""
    return {
        variable: _uri(
            _GRAPH_NODE_NSID,
            f"{key}-node-{index:0{_INDEX_WIDTH}d}",
        )
        for index, (variable, _) in enumerate(_variables(document))
    }


def _clause_target(
    clause: PmbClause,
    nodes: Mapping[str, str],
    ordinal: int,
) -> str:
    """Return the last graph-node argument of one clause.

    Parameters
    ----------
    clause : PmbClause
        Clause to project.
    nodes : Mapping[str, str]
        Graph-node URIs by PMB variable.
    ordinal : int
        Clause position used in error messages.

    Raises
    ------
    MalformedPmbDrsError
        If the clause has no graph-node argument.
    """
    targets = tuple(argument for argument in clause.arguments if argument in nodes)

    if not targets:
        msg = f"clause {ordinal} has no graph-node argument"
        raise MalformedPmbDrsError(msg)

    return targets[-1]


def _semantic_edges(
    document: PmbDrs,
    key: str,
    tokenization: str,
) -> tuple[FragmentRecord, ...]:
    """Return one custom graph edge per DRS clause.

    Parameters
    ----------
    document : PmbDrs
        Parsed DRS document.
    key : str
        Document record key.
    tokenization : str
        Tokenization identifier.

    Raises
    ------
    MalformedPmbDrsError
        If a clause has no graph-node argument.
    """
    nodes = _node_uris(document, key)
    records: list[FragmentRecord] = []

    for ordinal, clause in enumerate(document.clauses):
        properties = (
            (_KIND_KEY, "clause"),
            (_BOX_KEY, clause.box),
            (_CLAUSE_ORDINAL_KEY, str(ordinal)),
            (_ARGUMENT_COUNT_KEY, str(len(clause.arguments))),
            *(
                (_index_key(_ARGUMENT_PREFIX, index), argument)
                for index, argument in enumerate(clause.arguments)
            ),
            *_anchor_entries(tokenization, clause.token_indices),
        )
        record_key = f"{key}-clause-{ordinal:0{_INDEX_WIDTH}d}"
        records.append(
            FragmentRecord(
                local_id=_uri(_GRAPH_EDGE_NSID, record_key),
                nsid=_GRAPH_EDGE_NSID,
                value_json=graph_records.GraphEdge(
                    createdAt=_EPOCH,
                    edgeType="custom",
                    label=clause.operator,
                    ordinal=ordinal,
                    source=defs_records.ObjectRef(recordRef=nodes[clause.box]),
                    target=defs_records.ObjectRef(
                        recordRef=nodes[_clause_target(clause, nodes, ordinal)]
                    ),
                    properties=_features(properties),
                ).model_dump_json(),
            )
        )

    return tuple(records)


def _reachable(start: str, neighbors: Mapping[str, set[str]]) -> set[str]:
    """Return the nodes reachable from one variable."""
    reached = {start}
    pending = [start]

    while pending:
        here = pending.pop()

        for neighbor in neighbors[here] - reached:
            reached.add(neighbor)
            pending.append(neighbor)

    return reached


def _connectivity_edges(document: PmbDrs, key: str) -> tuple[FragmentRecord, ...]:
    """Return structural edges that make every DRS node graph-reachable.

    Parameters
    ----------
    document : PmbDrs
        Parsed DRS document.
    key : str
        Document record key.

    Raises
    ------
    MalformedPmbDrsError
        If a clause has no graph-node argument.
    """
    nodes = _node_uris(document, key)
    root = document.clauses[0].box
    neighbors: dict[str, set[str]] = {variable: set() for variable in nodes}

    for ordinal, clause in enumerate(document.clauses):
        target = _clause_target(clause, nodes, ordinal)
        neighbors[clause.box].add(target)
        neighbors[target].add(clause.box)

    reached = _reachable(root, neighbors)
    linked: list[str] = []

    for variable in nodes:
        if variable in reached:
            continue

        linked.append(variable)
        reached.update(_reachable(variable, neighbors))

    return tuple(
        FragmentRecord(
            local_id=_uri(
                _GRAPH_EDGE_NSID,
                f"{key}-connectivity-{index:0{_INDEX_WIDTH}d}",
            ),
            nsid=_GRAPH_EDGE_NSID,
            value_json=graph_records.GraphEdge(
                createdAt=_EPOCH,
                edgeType="related-to",
                source=defs_records.ObjectRef(recordRef=nodes[root]),
                target=defs_records.ObjectRef(recordRef=nodes[variable]),
                properties=_features(((_KIND_KEY, "connectivity"),)),
            ).model_dump_json(),
        )
        for index, variable in enumerate(linked)
    )


def _attachment(document: PmbDrs, key: str) -> FragmentRecord:
    """Return the edge attaching an expression to its root box."""
    root = document.clauses[0].box
    nodes = _node_uris(document, key)
    record_key = f"{key}-attachment"
    return FragmentRecord(
        local_id=_uri(_GRAPH_EDGE_NSID, record_key),
        nsid=_GRAPH_EDGE_NSID,
        value_json=graph_records.GraphEdge(
            createdAt=_EPOCH,
            edgeType="denotes",
            source=defs_records.ObjectRef(recordRef=_uri(_EXPRESSION_NSID, key)),
            target=defs_records.ObjectRef(recordRef=nodes[root]),
            properties=_features(((_KIND_KEY, "attachment"),)),
        ).model_dump_json(),
    )


def _validate_tokens(document: PmbDrs, index: int) -> None:
    """Validate one parsed DRS's tokens and expression text.

    Parameters
    ----------
    document : PmbDrs
        Parsed DRS document.
    index : int
        Document position in the fragment.

    Raises
    ------
    MalformedPmbDrsError
        If a token is malformed or the reconstructed text disagrees.
    """
    previous_end = 0

    for position, token in enumerate(document.tokens):
        if (
            token.start < previous_end
            or token.end <= token.start
            or token.end - token.start != len(token.surface)
        ):
            msg = f"DRS document {index} has invalid token {position}"
            raise MalformedPmbDrsError(msg)

        previous_end = token.end

    if document.text != _surface(document.tokens):
        msg = f"DRS document {index} text disagrees with its tokens"
        raise MalformedPmbDrsError(msg)


def _validate_clauses(document: PmbDrs, index: int) -> None:
    """Validate one parsed DRS's clauses and token positions.

    Parameters
    ----------
    document : PmbDrs
        Parsed DRS document.
    index : int
        Document position in the fragment.

    Raises
    ------
    MalformedPmbDrsError
        If clauses are absent or have invalid fields, arity, or grounding.
    """
    if not document.clauses:
        msg = f"DRS document {index} has no clauses"
        raise MalformedPmbDrsError(msg)

    for ordinal, clause in enumerate(document.clauses):
        if not clause.box or not clause.operator or len(clause.arguments) not in {1, 2}:
            msg = f"clause {ordinal} has invalid fields or arity"
            raise MalformedPmbDrsError(msg)

        if not clause.token_indices:
            msg = f"clause {ordinal} has no token position"
            raise MalformedPmbDrsError(msg)

        if len(set(clause.token_indices)) != len(clause.token_indices):
            msg = f"clause {ordinal} repeats a token position"
            raise MalformedPmbDrsError(msg)

        if tuple(sorted(clause.token_indices)) != clause.token_indices:
            msg = f"clause {ordinal} token positions are not in canonical order"
            raise MalformedPmbDrsError(msg)

        if any(
            position < 0 or position >= len(document.tokens)
            for position in clause.token_indices
        ):
            msg = f"clause {ordinal} names a token outside its document"
            raise MalformedPmbDrsError(msg)


def _records_of(document: PmbDrs, index: int) -> tuple[FragmentRecord, ...]:
    """Return canonical Layers records for one validated DRS.

    Parameters
    ----------
    document : PmbDrs
        Parsed DRS document.
    index : int
        Document position in the fragment.

    Raises
    ------
    MalformedPmbDrsError
        If text, token, clause, or graph grounding data is inconsistent.
    """
    _validate_tokens(document, index)
    _validate_clauses(document, index)
    key = _document_key(index)
    expression_uri = _uri(_EXPRESSION_NSID, key)
    tokenization = f"drs-tokenization-{index}"
    expression = expression_records.Expression(
        id=key,
        kind="sentence",
        createdAt=_EPOCH,
        text=document.text,
    )
    segmentation = segmentation_records.Segmentation(
        createdAt=_EPOCH,
        expression=expression_uri,
        tokenizations=(
            segmentation_records.Tokenization(
                uuid=defs_records.Uuid(value=tokenization),
                kind="pmb",
                tokens=_tokens_of(document),
            ),
        ),
    )
    return (
        FragmentRecord(
            local_id=expression_uri,
            nsid=_EXPRESSION_NSID,
            value_json=expression.model_dump_json(),
        ),
        FragmentRecord(
            local_id=_uri(_SEGMENTATION_NSID, key),
            nsid=_SEGMENTATION_NSID,
            value_json=segmentation.model_dump_json(),
        ),
        *_node_records(document, key, tokenization),
        *_semantic_edges(document, key, tokenization),
        *_connectivity_edges(document, key),
        _attachment(document, key),
    )


def _local_key(record: FragmentRecord) -> str:
    """Return a record's local key after validating its collection.

    Parameters
    ----------
    record : FragmentRecord
        Fragment record.

    Raises
    ------
    MalformedPmbDrsError
        If the URI does not agree with the record collection.
    """
    prefix = f"{_SCHEME}/{record.nsid}/"

    if not record.local_id.startswith(prefix):
        msg = f"record URI {record.local_id!r} disagrees with {record.nsid!r}"
        raise MalformedPmbDrsError(msg)

    key = record.local_id.removeprefix(prefix)

    if not key or "/" in key:
        msg = f"record URI {record.local_id!r} has no local PMB DRS key"
        raise MalformedPmbDrsError(msg)

    return key


def _record_document(record: FragmentRecord) -> tuple[int, str]:
    """Return the document index and record role for one fragment record.

    Parameters
    ----------
    record : FragmentRecord
        Fragment record.

    Raises
    ------
    MalformedPmbDrsError
        If the record collection or local key is not part of this codec.
    """
    key = _local_key(record)
    match: re.Match[str] | None

    if record.nsid in {_EXPRESSION_NSID, _SEGMENTATION_NSID}:
        match = _DOCUMENT_KEY_PATTERN.fullmatch(key)
        role = record.nsid
    elif record.nsid == _GRAPH_NODE_NSID:
        match = _NODE_KEY_PATTERN.fullmatch(key)
        role = "node"
    elif record.nsid == _GRAPH_EDGE_NSID:
        match = _CLAUSE_KEY_PATTERN.fullmatch(key)
        role = "clause"

        if match is None:
            match = _CONNECTIVITY_KEY_PATTERN.fullmatch(key)
            role = "connectivity"

        if match is None:
            match = _ATTACHMENT_KEY_PATTERN.fullmatch(key)
            role = "attachment"
    else:
        match = None
        role = ""

    if match is None:
        msg = f"record {record.local_id!r} is not a PMB DRS record"
        raise MalformedPmbDrsError(msg)

    document_key = match.groupdict().get("document", match.group(0))
    document_match = _DOCUMENT_KEY_PATTERN.fullmatch(document_key)

    if document_match is None:
        msg = f"record {record.local_id!r} has no PMB DRS document index"
        raise MalformedPmbDrsError(msg)

    return int(document_match.group("index")), role


def _groups(records: Iterable[FragmentRecord]) -> tuple[_DocumentRecords, ...]:
    """Return fragment records grouped by DRS document.

    Parameters
    ----------
    records : Iterable[FragmentRecord]
        PMB DRS fragment records.

    Raises
    ------
    MalformedPmbDrsError
        If records are absent, foreign, malformed, or use noncontiguous
        document indices.
    """
    grouped: dict[int, list[FragmentRecord]] = {}

    for record in records:
        index, _ = _record_document(record)
        grouped.setdefault(index, []).append(record)

    indices = tuple(sorted(grouped))

    if not indices:
        msg = "a PMB DRS fragment has no records"
        raise MalformedPmbDrsError(msg)

    if indices != tuple(range(len(indices))):
        msg = "PMB DRS document indices are not contiguous from zero"
        raise MalformedPmbDrsError(msg)

    return tuple(_DocumentRecords(index, tuple(grouped[index])) for index in indices)


def _one(
    records: Sequence[FragmentRecord],
    role: str,
) -> FragmentRecord:
    """Return one required record by its document role.

    Parameters
    ----------
    records : Sequence[FragmentRecord]
        One document's records.
    role : str
        Required record role.

    Raises
    ------
    MalformedPmbDrsError
        If the role is absent or repeated.
    """
    selected = tuple(
        record for record in records if _record_document(record)[1] == role
    )

    if len(selected) != 1:
        msg = f"a PMB DRS document has {len(selected)} {role} records"
        raise MalformedPmbDrsError(msg)

    return selected[0]


def _tokens_from(
    expression: expression_records.Expression,
    segmentation: segmentation_records.Segmentation,
) -> tuple[PmbToken, ...]:
    """Return character-aligned tokens from Layers records.

    Parameters
    ----------
    expression : expression_records.Expression
        Expression holding the source text.
    segmentation : segmentation_records.Segmentation
        Attached segmentation record.

    Raises
    ------
    MalformedPmbDrsError
        If the expression or segmentation has invalid text, spans, or positions.
    """
    if expression.text is None:
        msg = "a PMB DRS expression has no text"
        raise MalformedPmbDrsError(msg)

    if len(segmentation.tokenizations) != 1:
        msg = "a PMB DRS expression does not have one tokenization"
        raise MalformedPmbDrsError(msg)

    encoded = expression.text.encode("utf-8")
    tokens: list[PmbToken] = []

    for position, token in enumerate(segmentation.tokenizations[0].tokens):
        if token.tokenIndex != position or token.textSpan is None or token.text is None:
            msg = f"PMB DRS token {position} has invalid position, text, or span"
            raise MalformedPmbDrsError(msg)

        span = token.textSpan

        if span.byteEnd > len(encoded):
            msg = f"PMB DRS token {position} ends beyond its expression"
            raise MalformedPmbDrsError(msg)

        try:
            prefix = encoded[: span.byteStart].decode("utf-8")
            surface = encoded[span.byteStart : span.byteEnd].decode("utf-8")
        except UnicodeDecodeError as error:
            msg = f"PMB DRS token {position} splits a UTF-8 character"
            raise MalformedPmbDrsError(msg) from error

        if surface != token.text:
            msg = f"PMB DRS token {position} text disagrees with its span"
            raise MalformedPmbDrsError(msg)

        tokens.append(PmbToken(surface, len(prefix), len(prefix) + len(surface)))

    return tuple(tokens)


def _integer(value: str | None, subject: str) -> int:
    """Return a nonnegative integer property.

    Parameters
    ----------
    value : str | None
        Property value.
    subject : str
        Property named in error messages.

    Raises
    ------
    MalformedPmbDrsError
        If the property is absent or not a canonical nonnegative integer.
    """
    try:
        parsed = int(value or "")
    except ValueError as error:
        msg = f"{subject} is not an integer"
        raise MalformedPmbDrsError(msg) from error

    if parsed < 0 or str(parsed) != value:
        msg = f"{subject} is not a canonical nonnegative integer"
        raise MalformedPmbDrsError(msg)

    return parsed


def _grounding(
    properties: Mapping[str, str],
    tokenization: str,
    token_count: int,
    subject: str,
) -> tuple[int, ...]:
    """Return validated token positions from PMB properties.

    Parameters
    ----------
    properties : Mapping[str, str]
        Graph record properties.
    tokenization : str
        Expected tokenization identifier.
    token_count : int
        Number of tokens in that tokenization.
    subject : str
        Record named in error messages.

    Raises
    ------
    MalformedPmbDrsError
        If grounding is absent, mixed, duplicated, noncanonical, or out of range.
    """
    if properties.get(_TOKENIZATION_KEY) != tokenization:
        msg = f"{subject} is grounded to another tokenization"
        raise MalformedPmbDrsError(msg)

    indices: list[int] = []

    for key, value in properties.items():
        if not key.startswith(_GROUNDING_INDEX_PREFIX):
            continue

        suffix = key.removeprefix(_GROUNDING_INDEX_PREFIX)
        index = _integer(value, f"{subject} token position")

        if suffix != f"{index:0{_INDEX_WIDTH}d}":
            msg = f"{subject} has a noncanonical token-position key"
            raise MalformedPmbDrsError(msg)

        if index >= token_count:
            msg = f"{subject} names token {index} outside its segmentation"
            raise MalformedPmbDrsError(msg)

        indices.append(index)

    if not indices:
        msg = f"{subject} has no token position"
        raise MalformedPmbDrsError(msg)

    if len(indices) != len(set(indices)):
        msg = f"{subject} repeats a token position"
        raise MalformedPmbDrsError(msg)

    return tuple(sorted(indices))


def _clauses_from(
    records: Sequence[FragmentRecord],
    tokenization: str,
    token_count: int,
) -> tuple[PmbClause, ...]:
    """Return validated clauses from semantic graph edges.

    Parameters
    ----------
    records : Sequence[FragmentRecord]
        One document's records.
    tokenization : str
        Expected tokenization identifier.
    token_count : int
        Number of tokens in that tokenization.

    Raises
    ------
    MalformedPmbDrsError
        If an edge loses its clause label, ordinal, arguments, or grounding.
    """
    indexed: list[tuple[int, PmbClause]] = []

    for record in records:
        _, role = _record_document(record)

        if role != "clause":
            continue

        edge = _model_from_json(graph_records.GraphEdge, record)
        properties = _mapping(edge.properties, f"clause edge {record.local_id!r}")
        ordinal = _integer(
            properties.get(_CLAUSE_ORDINAL_KEY),
            f"clause edge {record.local_id!r} ordinal",
        )

        if edge.edgeType != "custom" or edge.label is None or edge.ordinal != ordinal:
            msg = f"clause edge {record.local_id!r} loses its type, label, or ordinal"
            raise MalformedPmbDrsError(msg)

        count = _integer(
            properties.get(_ARGUMENT_COUNT_KEY),
            f"clause edge {record.local_id!r} argument count",
        )
        arguments: list[str] = []

        for index in range(count):
            value = properties.get(_index_key(_ARGUMENT_PREFIX, index))

            if value is None:
                msg = f"clause edge {record.local_id!r} has no argument {index}"
                raise MalformedPmbDrsError(msg)

            arguments.append(value)

        supplied_arguments = {
            key for key in properties if key.startswith(_ARGUMENT_PREFIX)
        }
        expected_arguments = {
            _index_key(_ARGUMENT_PREFIX, index) for index in range(count)
        }

        if supplied_arguments != expected_arguments:
            msg = f"clause edge {record.local_id!r} has noncanonical arguments"
            raise MalformedPmbDrsError(msg)

        box = properties.get(_BOX_KEY)

        if not box:
            msg = f"clause edge {record.local_id!r} has no DRS box"
            raise MalformedPmbDrsError(msg)

        positions = _grounding(
            properties,
            tokenization,
            token_count,
            f"clause edge {record.local_id!r}",
        )
        indexed.append(
            (ordinal, PmbClause(box, edge.label, tuple(arguments), positions))
        )

    indexed.sort(key=lambda item: item[0])

    if tuple(ordinal for ordinal, _ in indexed) != tuple(range(len(indexed))):
        msg = "PMB DRS clause ordinals are not contiguous from zero"
        raise MalformedPmbDrsError(msg)

    return tuple(clause for _, clause in indexed)


def _document_from(group: _DocumentRecords) -> PmbDrs:
    """Return one DRS from canonical Layers records.

    Parameters
    ----------
    group : _DocumentRecords
        One document's records.

    Raises
    ------
    MalformedPmbDrsError
        If records are incomplete, inconsistent, or noncanonical.
    """
    expression_record = _one(group.records, _EXPRESSION_NSID)
    segmentation_record = _one(group.records, _SEGMENTATION_NSID)
    _one(group.records, "attachment")
    expression = _model_from_json(expression_records.Expression, expression_record)
    segmentation = _model_from_json(
        segmentation_records.Segmentation,
        segmentation_record,
    )
    expected_expression = _uri(_EXPRESSION_NSID, _document_key(group.index))

    if segmentation.expression != expected_expression:
        msg = f"DRS document {group.index} segmentation names another expression"
        raise MalformedPmbDrsError(msg)

    tokens = _tokens_from(expression, segmentation)
    tokenization = segmentation.tokenizations[0].uuid.value
    clauses = _clauses_from(group.records, tokenization, len(tokens))

    if expression.text is None:
        msg = f"DRS document {group.index} has no expression text"
        raise MalformedPmbDrsError(msg)

    document = PmbDrs(expression.text, tokens, clauses)
    expected = _records_of(document, group.index)

    if group.records != expected:
        msg = f"DRS document {group.index} records are not in canonical form"
        raise MalformedPmbDrsError(msg)

    return document


class PmbDrsIso:
    """A translation from parsed PMB DRSs to Layers fragments."""

    __slots__ = ()

    def forward(self, source: PmbDrsFile, /) -> CorpusFragment:
        """Return canonical Layers records for parsed DRSs.

        Parameters
        ----------
        source : PmbDrsFile
            Parsed PMB DRS documents.

        Raises
        ------
        MalformedPmbDrsError
            If the source is empty or a document has inconsistent text,
            tokens, clauses, or grounding.
        """
        if not source:
            msg = "a PMB DRS file has no documents"
            raise MalformedPmbDrsError(msg)

        return CorpusFragment(
            records=tuple(
                record
                for index, document in enumerate(source)
                for record in _records_of(document, index)
            ),
            source=DRS_NAME,
        )

    def backward(self, target: CorpusFragment, /) -> PmbDrsFile:
        """Return parsed DRSs from canonical Layers records.

        Parameters
        ----------
        target : CorpusFragment
            Layers fragment produced by this translation.

        Raises
        ------
        MalformedPmbDrsError
            If record JSON is invalid or the records are incomplete,
            inconsistent, or noncanonical.
        """
        return tuple(_document_from(group) for group in _groups(target.records))


@registers_as(DRS_NAME)
class PmbDrsCodec:
    """A codec for PMB clausal-form DRS text."""

    __slots__ = ("_iso", "name")

    name: str

    def __init__(self) -> None:
        self.name = DRS_NAME
        self._iso = PmbDrsIso()

    @property
    def iso(self) -> Iso[PmbDrsFile, CorpusFragment]:
        """Return the DRS-to-fragment translation."""
        return self._iso

    def decode(
        self,
        src: str | bytes,
        *,
        into: CorpusFragment | None = None,
    ) -> CorpusFragment:
        """Return Layers records decoded from PMB clausal form.

        Parameters
        ----------
        src : str | bytes
            PMB clausal-form text or bytes.
        into : CorpusFragment | None
            Existing fragment whose records remain first.

        Raises
        ------
        MalformedPmbDrsError
            If UTF-8 decoding fails or the source is empty or has malformed
            clauses, token alignment, or grounding.
        """
        parsed = _parse(src)

        if into is None:
            return self._iso.forward(parsed)

        offset = sum(record.nsid == _EXPRESSION_NSID for record in into.records)
        added = tuple(
            record
            for index, document in enumerate(parsed, start=offset)
            for record in _records_of(document, index)
        )
        return CorpusFragment(records=into.records + added, source=DRS_NAME)

    def encode(self, records: Iterable[FragmentRecord]) -> str:
        """Return PMB clausal form for canonical Layers records.

        Parameters
        ----------
        records : Iterable[FragmentRecord]
            Records produced by this codec.

        Raises
        ------
        MalformedPmbDrsError
            If the records are incomplete, inconsistent, or noncanonical.
        """
        fragment = CorpusFragment(records=tuple(records), source=DRS_NAME)
        return _write(self._iso.backward(fragment))
