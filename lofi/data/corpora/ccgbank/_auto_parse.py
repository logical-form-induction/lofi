"""Pure CCGbank AUTO values, parsing, and serialization."""

from typing import TYPE_CHECKING, Final, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "AutoDerivation",
    "AutoFile",
    "AutoLeaf",
    "AutoNode",
    "AutoTree",
    "MalformedAutoError",
    "parse_auto",
    "write_auto",
]

_TREE_OPEN: Final = "(<T"
_LEAF_OPEN: Final = "(<L"
_OPEN: Final = "("
_CLOSE: Final = ")"
_TREE_FIELDS: Final = 3
_LEAF_FIELDS: Final = 5
_ASSIGNMENT: Final = "="


class MalformedAutoError(ValueError):
    """An error for malformed CCGbank AUTO text."""


class AutoLeaf(NamedTuple):
    """A lexical AUTO node."""

    category: str
    pos: str
    original_pos: str
    word: str
    slotted: str


class AutoTree(NamedTuple):
    """An internal AUTO node."""

    category: str
    head: int
    children: tuple[AutoNode, ...]


type AutoNode = AutoTree | AutoLeaf
"""An internal or lexical AUTO node."""


class AutoDerivation(NamedTuple):
    """A parsed AUTO derivation."""

    header: tuple[tuple[str, str], ...]
    tree: AutoNode


type AutoFile = tuple[AutoDerivation, ...]
"""The analyses in one derivation file, in source order."""


def _header_of(line: str, number: int) -> tuple[tuple[str, str], ...]:
    """Return the fields one header line states."""
    fields: list[tuple[str, str]] = []
    for field in line.split():
        key, assigned, value = field.partition(_ASSIGNMENT)
        if not assigned or not key:
            msg = f"line {number} states {field!r}, which assigns no value"
            raise MalformedAutoError(msg)
        fields.append((key, value))
    if not fields:
        msg = f"line {number} heads a derivation with no field at all"
        raise MalformedAutoError(msg)
    return tuple(fields)


def _field(tokens: Sequence[str], at: int, number: int) -> str:
    """Return one field of a node, reporting a line that ends inside one."""
    if at >= len(tokens):
        msg = f"line {number} ends in the middle of a node"
        raise MalformedAutoError(msg)
    return tokens[at]


def _count(field: str, number: int) -> int:
    """Return a node's count field as the number it states."""
    try:
        return int(field)
    except ValueError as error:
        msg = f"line {number} states {field!r} where a node states a number"
        raise MalformedAutoError(msg) from error


def _opened(tokens: Sequence[str], at: int, number: int) -> tuple[str, int, int]:
    """Return the category, head and daughter count a node opens with."""
    fields = [_field(tokens, at + step, number) for step in range(1, _TREE_FIELDS + 1)]
    last = fields[-1]
    if not last.endswith(">"):
        msg = f"line {number} opens a node that states no end"
        raise MalformedAutoError(msg)
    head = _count(fields[1], number)
    daughters = _count(last.removesuffix(">"), number)
    if not 0 <= head < daughters:
        msg = (
            f"line {number} opens a node of {daughters} daughters whose head "
            f"is daughter {head}"
        )
        raise MalformedAutoError(msg)
    return fields[0], head, daughters


def _leaf_at(tokens: Sequence[str], at: int, number: int) -> AutoLeaf:
    """Return the leaf opening at one position."""
    fields = [_field(tokens, at + step, number) for step in range(1, _LEAF_FIELDS + 1)]
    last = fields[-1]
    if not last.endswith(f">{_CLOSE}"):
        msg = f"line {number} opens a leaf that states no end"
        raise MalformedAutoError(msg)
    return AutoLeaf(
        fields[0], fields[1], fields[2], fields[3], last.removesuffix(f">{_CLOSE}")
    )


def _closed(
    pending: list[tuple[str, int, int, list[AutoNode]]],
    number: int,
) -> AutoNode:
    """Return the node one closing bracket completes."""
    if not pending:
        msg = f"line {number} closes a node that was never opened"
        raise MalformedAutoError(msg)
    category, head, daughters, children = pending.pop()
    if len(children) != daughters:
        msg = (
            f"line {number} closes a node of {daughters} daughters over "
            f"{len(children)} of them"
        )
        raise MalformedAutoError(msg)
    return AutoTree(category, head, tuple(children))


def _attach(
    node: AutoNode,
    pending: list[tuple[str, int, int, list[AutoNode]]],
    roots: list[AutoNode],
) -> None:
    """Attach a completed node to its parent or the roots."""
    if pending:
        pending[-1][3].append(node)
    else:
        roots.append(node)


def _tree_of(line: str, number: int) -> AutoNode:
    """Return the analysis one derivation line states."""
    tokens = line.split()
    pending: list[tuple[str, int, int, list[AutoNode]]] = []
    roots: list[AutoNode] = []
    cursor = 0
    while cursor < len(tokens):
        token = tokens[cursor]
        if token == _TREE_OPEN:
            category, head, daughters = _opened(tokens, cursor, number)
            pending.append((category, head, daughters, []))
            cursor += _TREE_FIELDS + 1
            continue
        if token == _LEAF_OPEN:
            _attach(_leaf_at(tokens, cursor, number), pending, roots)
            cursor += _LEAF_FIELDS + 1
            continue
        if token == _CLOSE:
            _attach(_closed(pending, number), pending, roots)
            cursor += 1
            continue
        msg = f"line {number} holds {token!r}, which opens and closes nothing"
        raise MalformedAutoError(msg)
    if pending:
        msg = f"line {number} leaves {len(pending)} nodes unclosed"
        raise MalformedAutoError(msg)
    if len(roots) != 1:
        msg = f"line {number} states {len(roots)} analyses rather than one"
        raise MalformedAutoError(msg)
    return roots[0]


def parse_auto(src: str | bytes, /) -> AutoFile:
    """Parse every analysis in an AUTO document."""
    text = src.decode("utf-8") if isinstance(src, bytes) else src
    read: list[AutoDerivation] = []
    header: tuple[tuple[str, str], ...] | None = None
    headed = 0
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        if not line.lstrip().startswith(_OPEN):
            if header is not None:
                msg = f"line {headed} heads no derivation"
                raise MalformedAutoError(msg)
            header = _header_of(line, number)
            headed = number
            continue
        if header is None:
            msg = f"line {number} states a derivation under no header"
            raise MalformedAutoError(msg)
        read.append(AutoDerivation(header, _tree_of(line, number)))
        header = None
    if header is not None:
        msg = f"line {headed} heads no derivation"
        raise MalformedAutoError(msg)
    return tuple(read)


def _written(node: AutoNode) -> str:
    """Return the text one analysis is written as."""
    parts: list[str] = []
    stack: list[AutoNode | str] = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, AutoLeaf):
            parts.append(
                f"{_LEAF_OPEN} {item.category} {item.pos} {item.original_pos} "
                f"{item.word} {item.slotted}>{_CLOSE}"
            )
        else:
            parts.append(
                f"{_TREE_OPEN} {item.category} {item.head} {len(item.children)}>"
            )
            stack.append(_CLOSE)
            stack.extend(reversed(item.children))
    return " ".join(parts)


def write_auto(source: AutoFile, /) -> str:
    """Serialize parsed AUTO derivations."""
    return "".join(
        " ".join(f"{key}={value}" for key, value in derivation.header)
        + "\n"
        + _written(derivation.tree)
        + "\n"
        for derivation in source
    )
