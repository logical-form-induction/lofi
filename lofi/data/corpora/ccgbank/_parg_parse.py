"""Pure CCGbank PARG values, parsing, and serialization."""

import re
from typing import Final, NamedTuple

__all__ = [
    "MalformedPargError",
    "PargDependency",
    "PargFile",
    "PargSentence",
    "parse_parg",
    "write_parg",
]

_HEADER: Final = re.compile(r'^<s id="([^"]+)"> ([0-9]+)$')
_CLOSE: Final = r"<\s>"
_MARKERS: Final = frozenset(("<XB>", "<XU>"))
_DEPENDENCY_FIELDS: Final = 6
_MARKED_DEPENDENCY_FIELDS: Final = 7


class MalformedPargError(ValueError):
    """An error for malformed CCGbank PARG data."""


class PargDependency(NamedTuple):
    """A PARG dependency."""

    target_index: int
    head_index: int
    head_category: str
    argument_slot: int
    target_word: str
    head_word: str
    marker: str | None = None


class PargSentence(NamedTuple):
    """A sentence in a PARG file."""

    identifier: str
    last_index: int
    dependencies: tuple[PargDependency, ...]


type PargFile = tuple[PargSentence, ...]
"""The sentences in one PARG file."""


def _natural(field: str, description: str, number: int) -> int:
    """Parse a nonnegative integer field."""
    if not field.isascii() or not field.isdecimal():
        msg = f"line {number} states {field!r} where {description} is an integer"
        raise MalformedPargError(msg)
    return int(field)


def _header(line: str, number: int) -> tuple[str, int]:
    """Parse a sentence header."""
    match = _HEADER.fullmatch(line)
    if match is None:
        msg = f"line {number} is not a PARG sentence header"
        raise MalformedPargError(msg)
    return match[1], _natural(match[2], "the header's last index", number)


def _validate_dependency(
    dependency: PargDependency,
    last_index: int,
    source: str,
) -> None:
    """Validate one dependency against its sentence header."""
    if dependency.argument_slot < 1:
        msg = (
            f"{source} states argument slot {dependency.argument_slot}; "
            "slots start at one"
        )
        raise MalformedPargError(msg)
    for description, index in (
        ("target", dependency.target_index),
        ("head", dependency.head_index),
    ):
        if not 0 <= index <= last_index:
            msg = (
                f"{source} states {description} index {index}, outside the header's "
                f"0..{last_index} range"
            )
            raise MalformedPargError(msg)
    for description, value in (
        ("head category", dependency.head_category),
        ("target word", dependency.target_word),
        ("head word", dependency.head_word),
    ):
        if not value or any(character.isspace() for character in value):
            msg = f"{source} states an invalid {description} {value!r}"
            raise MalformedPargError(msg)
    if dependency.marker is not None and dependency.marker not in _MARKERS:
        msg = f"{source} states the unknown marker {dependency.marker!r}"
        raise MalformedPargError(msg)


def _dependency(line: str, last_index: int, number: int) -> PargDependency:
    """Parse a dependency line."""
    fields = line.split()
    if len(fields) not in (_DEPENDENCY_FIELDS, _MARKED_DEPENDENCY_FIELDS):
        msg = f"line {number} has {len(fields)} fields rather than six or seven"
        raise MalformedPargError(msg)
    marker = fields[6] if len(fields) == _MARKED_DEPENDENCY_FIELDS else None
    if marker is not None and marker not in _MARKERS:
        msg = f"line {number} states the unknown marker {marker!r}"
        raise MalformedPargError(msg)
    parsed = PargDependency(
        target_index=_natural(fields[0], "the target index", number),
        head_index=_natural(fields[1], "the head index", number),
        head_category=fields[2],
        argument_slot=_natural(fields[3], "the argument slot", number),
        target_word=fields[4],
        head_word=fields[5],
        marker=marker,
    )
    _validate_dependency(parsed, last_index, f"line {number}")
    return parsed


def validate_parg_sentence(sentence: PargSentence, position: int, /) -> None:
    """Validate one parsed PARG sentence."""
    source = f"sentence {position}"
    if (
        not sentence.identifier
        or '"' in sentence.identifier
        or "\n" in sentence.identifier
        or "\r" in sentence.identifier
    ):
        msg = f"{source} has an invalid identifier {sentence.identifier!r}"
        raise MalformedPargError(msg)
    if sentence.last_index < 0:
        msg = f"{source} has the negative last index {sentence.last_index}"
        raise MalformedPargError(msg)
    for dependency in sentence.dependencies:
        _validate_dependency(dependency, sentence.last_index, source)


def parse_parg(src: str | bytes, /) -> PargFile:
    """Parse every sentence in a PARG document."""
    text = src.decode("utf-8") if isinstance(src, bytes) else src
    read: list[PargSentence] = []
    identifier: str | None = None
    last_index = 0
    dependencies: list[PargDependency] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("<s"):
            if identifier is not None:
                msg = f"line {number} opens a sentence before the prior one closes"
                raise MalformedPargError(msg)
            identifier, last_index = _header(line, number)
            dependencies = []
            continue
        if line == _CLOSE:
            if identifier is None:
                msg = f"line {number} closes a sentence that was never opened"
                raise MalformedPargError(msg)
            sentence = PargSentence(identifier, last_index, tuple(dependencies))
            validate_parg_sentence(sentence, len(read))
            read.append(sentence)
            identifier = None
            continue
        if identifier is None:
            msg = f"line {number} states a dependency outside a sentence"
            raise MalformedPargError(msg)
        dependencies.append(_dependency(line, last_index, number))
    if identifier is not None:
        msg = f"sentence {identifier!r} is not closed"
        raise MalformedPargError(msg)
    return tuple(read)


def write_parg(source: PargFile, /) -> str:
    """Serialize parsed PARG sentences."""
    lines: list[str] = []
    for position, sentence in enumerate(source):
        validate_parg_sentence(sentence, position)
        lines.append(f'<s id="{sentence.identifier}"> {sentence.last_index}')
        for dependency in sentence.dependencies:
            fields = [
                str(dependency.target_index),
                str(dependency.head_index),
                dependency.head_category,
                str(dependency.argument_slot),
                dependency.target_word,
                dependency.head_word,
            ]
            if dependency.marker is not None:
                fields.append(dependency.marker)
            lines.append("\t".join(fields))
        lines.append(_CLOSE)
    return "" if not lines else "\n".join(lines) + "\n"
