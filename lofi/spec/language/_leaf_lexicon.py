"""Declared inline and file-backed sources of non-logical constants."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import didactic.api as dx
from didactic.settings import ConfigValue, load_document

from lofi.typecheck import LeafConstant, LeafLexicon

if TYPE_CHECKING:
    from lofi._types import JsonValue

__all__ = [
    "InlineLeafLexiconSpec",
    "LeafConstantSpec",
    "LeafLexiconSpec",
    "PathLeafLexiconSpec",
]

_PACKAGED_LEXICON_ROOT = Path(__file__).parents[2] / "configs" / "leaf_lexicons"


def _json_value(value: ConfigValue, /) -> JsonValue:
    """Copy one settings value into the JSON shape accepted by didactic."""
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}

    if isinstance(value, str | int | float | bool) or value is None:
        return value

    return [_json_value(item) for item in value]


class LeafConstantSpec(dx.Model, extra="forbid"):
    """One typed or inferred non-logical constant in a leaf source."""

    name: str
    type: str = "infer"
    classes: tuple[str, ...] = ()

    __axioms__ = (
        dx.axiom("len(name) > 0", message="a leaf constant needs a name"),
        dx.axiom("len(type) > 0", message="a leaf constant needs a type or 'infer'"),
        dx.axiom("all(classes)", message="a leaf constant class needs a name"),
    )

    @dx.validates("classes")
    @classmethod
    def _classes_are_distinct(cls, classes: tuple[str, ...]) -> tuple[str, ...]:
        """Reject a class declared twice on one leaf constant."""
        if len(set(classes)) != len(classes):
            msg = "a leaf constant declares a class twice"
            raise ValueError(msg)

        return classes

    def lower(self) -> LeafConstant:
        """Return the immutable interior declaration."""
        return LeafConstant(self.name, self.type, self.classes)


def _entries_are_distinct(
    entries: tuple[LeafConstantSpec, ...], /
) -> tuple[LeafConstantSpec, ...]:
    """Return entries after rejecting duplicate constant names."""
    names = tuple(entry.name for entry in entries)

    if len(set(names)) != len(names):
        msg = "a leaf constant is declared twice"
        raise ValueError(msg)

    return entries


class LeafLexiconDocument(dx.Model, extra="forbid"):
    """The complete didactic document stored at a leaf-lexicon path."""

    entries: tuple[LeafConstantSpec, ...]

    __axioms__ = (
        dx.axiom("len(entries) > 0", message="a leaf lexicon needs a constant"),
    )

    @dx.validates("entries")
    @classmethod
    def _constant_names_are_distinct(
        cls, entries: tuple[LeafConstantSpec, ...]
    ) -> tuple[LeafConstantSpec, ...]:
        """Reject a constant declared twice."""
        return _entries_are_distinct(entries)

    def lower(self) -> LeafLexicon:
        """Return immutable interior declarations in source order."""
        return LeafLexicon(tuple(entry.lower() for entry in self.entries))


class LeafLexiconSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """A source of non-logical constant declarations."""


class InlineLeafLexiconSpec(LeafLexiconSpec, extra="forbid"):
    """Leaf constants listed directly in a language declaration."""

    kind: Literal["inline"] = "inline"
    entries: tuple[LeafConstantSpec, ...]

    __axioms__ = (
        dx.axiom("len(entries) > 0", message="a leaf lexicon needs a constant"),
    )

    @dx.validates("entries")
    @classmethod
    def _constant_names_are_distinct(
        cls, entries: tuple[LeafConstantSpec, ...]
    ) -> tuple[LeafConstantSpec, ...]:
        """Reject a constant declared twice."""
        return _entries_are_distinct(entries)

    def load(self) -> LeafLexicon:
        """Lower the inline entries."""
        return LeafLexicon(tuple(entry.lower() for entry in self.entries))


class PathLeafLexiconSpec(LeafLexiconSpec, extra="forbid"):
    """Leaf constants read from one JSON, TOML, or YAML document."""

    kind: Literal["path"] = "path"
    path: str

    __axioms__ = (
        dx.axiom("len(path) > 0", message="a leaf-lexicon path needs a file"),
    )

    def load(self) -> LeafLexicon:
        """Validate and lower the document at the declared path."""
        declared = Path(self.path)
        path = declared if declared.is_absolute() else _PACKAGED_LEXICON_ROOT / declared
        document = {
            key: _json_value(value) for key, value in load_document(path).items()
        }
        return LeafLexiconDocument.model_validate(document).lower()
