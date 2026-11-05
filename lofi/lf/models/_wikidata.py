"""A local Wikidata snapshot as a finite intended interpretation."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast, final, runtime_checkable

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.lf.models._evaluate import ModelLanguage, truth_type
from lofi.lf.models._frame import Element, Frame, FunctionValue, ModelValue
from lofi.lf.models._structure import Structure
from lofi.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue
    from lofi.terms import Term

__all__ = [
    "UNDECIDED_CONSTANT",
    "WIKIDATA",
    "UnmappedConstantPolicy",
    "WikidataSettings",
]

type _FactKey = tuple[str, str, str]
type _World = Literal["open", "closed"]


@final
class WikidataSettings(FrozenValue):
    """Resolved local files and world policy for a Wikidata interpretation."""

    _noun = "Wikidata settings"
    __slots__ = ("_mapping", "_snapshot", "_unmapped", "_world")

    def __init__(
        self,
        snapshot: str,
        mapping: str,
        world: _World,
        unmapped: str,
        /,
    ) -> None:
        object.__setattr__(self, "_snapshot", snapshot)
        object.__setattr__(self, "_mapping", mapping)
        object.__setattr__(self, "_world", world)
        object.__setattr__(self, "_unmapped", unmapped)

    @property
    def snapshot(self) -> str:
        return self._snapshot

    @property
    def mapping(self) -> str:
        return self._mapping

    @property
    def world(self) -> _World:
        return self._world

    @property
    def unmapped(self) -> str:
        return self._unmapped


@runtime_checkable
class UnmappedConstantPolicy(Protocol):
    """Supply a denotation for an unmapped constant, or leave it unknown."""

    def resolve(
        self, identifier: int, language: ModelLanguage, frame: Frame, /
    ) -> ModelValue | None: ...


@final
class _UndecidedConstant(FrozenValue):
    """Leave an unmapped constant without an intended denotation."""

    _noun = "undecided constant policy"
    __slots__ = ()

    def resolve(
        self, identifier: int, language: ModelLanguage, frame: Frame, /
    ) -> ModelValue | None:
        del identifier, language, frame
        return None


UNDECIDED_CONSTANT = _UndecidedConstant()


class WikidataFormatError(ValueError):
    """Report a malformed snapshot or vocabulary mapping."""


def _json(path: Path, /) -> JsonValue:
    try:
        return cast("JsonValue", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        msg = f"cannot read JSON knowledge-base file {path}"
        raise WikidataFormatError(msg) from error


def _object(value: JsonValue, field: str, /) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        msg = f"{field} is a JSON object"
        raise WikidataFormatError(msg)

    return value


def _text(fields: Mapping[str, JsonValue], field: str, /) -> str:
    value = fields.get(field)

    if not isinstance(value, str) or not value:
        msg = f"{field} is a nonempty string"
        raise WikidataFormatError(msg)

    return value


def _truth(fields: Mapping[str, JsonValue], field: str, /) -> bool:
    value = fields.get(field)

    if not isinstance(value, bool):
        msg = f"{field} is a Boolean truth value"
        raise WikidataFormatError(msg)

    return value


def _facts(path: Path, /) -> dict[_FactKey, bool]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        msg = f"cannot read JSONL knowledge-base snapshot {path}"
        raise WikidataFormatError(msg) from error

    facts: dict[_FactKey, bool] = {}

    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        try:
            row = _object(cast("JsonValue", json.loads(line)), f"line {number}")
        except json.JSONDecodeError as error:
            msg = f"snapshot line {number} is not JSON"
            raise WikidataFormatError(msg) from error

        key = (
            _text(row, "subject"),
            _text(row, "property"),
            _text(row, "object"),
        )
        value = _truth(row, "value")
        prior = facts.get(key)

        if prior is not None and prior is not value:
            msg = f"snapshot line {number} contradicts an earlier fact {key!r}"
            raise WikidataFormatError(msg)

        facts[key] = value

    if not facts:
        msg = "a Wikidata snapshot contains at least one fact"
        raise WikidataFormatError(msg)

    return facts


def _carriers(mapping: Mapping[str, JsonValue], /) -> dict[str, tuple[str, ...]]:
    declared = _object(mapping.get("carriers"), "mapping carriers")
    carriers: dict[str, tuple[str, ...]] = {}

    for type_name, value in declared.items():
        if not isinstance(value, list) or not value:
            msg = f"carrier {type_name!r} is a nonempty array of item ids"
            raise WikidataFormatError(msg)

        items: list[str] = []

        for item in value:
            if not isinstance(item, str) or not item:
                msg = f"carrier {type_name!r} contains a nonempty item id"
                raise WikidataFormatError(msg)
            items.append(item)

        if len(set(items)) != len(items):
            msg = f"carrier {type_name!r} contains distinct item ids"
            raise WikidataFormatError(msg)

        carriers[ensure_name("an atomic carrier", type_name)] = tuple(items)

    return carriers


def _known(world: _World, /, *, value: bool | None) -> tuple[bool, bool]:
    if value is None:
        return False, world == "closed"

    return value, True


def _spine(language: ModelLanguage, type_: Term, /) -> tuple[tuple[Term, ...], Term]:
    arguments: list[Term] = []
    result = type_

    while result.op.name == language.function_type and len(result.args) == 2:
        arguments.append(result.args[0])
        result = result.args[1]

    return tuple(arguments), result


def _elements(
    frame: Frame,
    type_: Term,
    carriers: Mapping[str, tuple[str, ...]],
    language: ModelLanguage,
    /,
) -> tuple[tuple[Element, str], ...]:
    values = frame.domain(type_).values

    if any(not isinstance(value, Element) for value in values):
        msg = "a Wikidata property argument has an atomic carrier"
        raise WikidataFormatError(msg)

    held = cast("tuple[Element, ...]", values)
    type_name = language.store.payload_name(type_)
    items = carriers.get(type_name)

    if items is None or len(items) != len(held):
        msg = f"mapping declares no complete carrier for type {type_name!r}"
        raise WikidataFormatError(msg)

    return tuple(zip(held, items, strict=True))


def _entity(
    declaration: Mapping[str, JsonValue],
    type_: Term,
    frame: Frame,
    carriers: Mapping[str, tuple[str, ...]],
    language: ModelLanguage,
    /,
) -> Element:
    item = _text(declaration, "item")
    arguments, _ = _spine(language, type_)

    if arguments:
        msg = "an item mapping denotes an atomic constant"
        raise WikidataFormatError(msg)

    candidates = _elements(frame, type_, carriers, language)

    for element, identifier in candidates:
        if identifier == item:
            return element

    msg = f"item {item!r} is absent from its declared carrier"
    raise WikidataFormatError(msg)


def _predicate(
    declaration: Mapping[str, JsonValue],
    type_: Term,
    frame: Frame,
    carriers: Mapping[str, tuple[str, ...]],
    facts: Mapping[_FactKey, bool],
    world: _World,
    language: ModelLanguage,
    truth: Term,
    /,
) -> FunctionValue:
    property_ = _text(declaration, "property")
    object_ = _text(declaration, "object")
    arguments, result = _spine(language, type_)

    if len(arguments) != 1 or result != truth:
        msg = "a predicate mapping denotes a unary truth-valued constant"
        raise WikidataFormatError(msg)

    pairs = _elements(frame, arguments[0], carriers, language)
    values_known = tuple(
        _known(world, value=facts.get((item, property_, object_))) for _, item in pairs
    )
    return FunctionValue(
        tuple(element for element, _ in pairs),
        tuple(value for value, _ in values_known),
        tuple(known for _, known in values_known),
    )


def _relation(
    declaration: Mapping[str, JsonValue],
    type_: Term,
    frame: Frame,
    carriers: Mapping[str, tuple[str, ...]],
    facts: Mapping[_FactKey, bool],
    world: _World,
    language: ModelLanguage,
    truth: Term,
    /,
) -> FunctionValue:
    property_ = _text(declaration, "property")
    arguments, result = _spine(language, type_)

    if len(arguments) != 2 or result != truth:
        msg = "a relation mapping denotes a binary truth-valued constant"
        raise WikidataFormatError(msg)

    subjects = _elements(frame, arguments[0], carriers, language)
    objects = _elements(frame, arguments[1], carriers, language)
    inner_domain = tuple(element for element, _ in objects)
    functions: list[ModelValue] = []

    for _, subject in subjects:
        values_known = tuple(
            _known(world, value=facts.get((subject, property_, object_)))
            for _, object_ in objects
        )
        functions.append(
            FunctionValue(
                inner_domain,
                tuple(value for value, _ in values_known),
                tuple(known for _, known in values_known),
            )
        )

    return FunctionValue(
        tuple(element for element, _ in subjects),
        tuple(functions),
        (True,) * len(subjects),
    )


def _denotation(
    declaration: Mapping[str, JsonValue],
    type_: Term,
    frame: Frame,
    carriers: Mapping[str, tuple[str, ...]],
    facts: Mapping[_FactKey, bool],
    settings: WikidataSettings,
    language: ModelLanguage,
    truth: Term,
    /,
) -> ModelValue:
    match _text(declaration, "kind"):
        case "item":
            return _entity(declaration, type_, frame, carriers, language)
        case "predicate":
            return _predicate(
                declaration,
                type_,
                frame,
                carriers,
                facts,
                settings.world,
                language,
                truth,
            )
        case "relation":
            return _relation(
                declaration,
                type_,
                frame,
                carriers,
                facts,
                settings.world,
                language,
                truth,
            )
        case kind:
            msg = f"unknown Wikidata constant mapping kind {kind!r}"
            raise WikidataFormatError(msg)


@final
class _WikidataReader(FrozenValue):
    """Read a local fact snapshot under an explicit vocabulary-id mapping."""

    _noun = "Wikidata reader"
    __slots__ = ()

    def read(
        self,
        settings: object,
        language: ModelLanguage,
        /,
        *,
        full_function_space_cap: int,
        henkin_pool: int,
    ) -> Structure:
        if not isinstance(settings, WikidataSettings):
            msg = f"the Wikidata reader reads WikidataSettings, not {settings!r}"
            raise TypeError(msg)

        unmapped = Registry[object]("unmapped_constant").get(settings.unmapped)
        if not isinstance(unmapped, UnmappedConstantPolicy):
            msg = f"unmapped-constant entry {settings.unmapped!r} supplies no policy"
            raise TypeError(msg)

        snapshot_path = Path(settings.snapshot)
        mapping_path = Path(settings.mapping)
        mapping = _object(_json(mapping_path), "vocabulary mapping")
        carriers = _carriers(mapping)
        facts = _facts(snapshot_path)
        truth = truth_type(language)
        truth_name = language.store.payload_name(truth)
        digest = sha256(snapshot_path.read_bytes() + mapping_path.read_bytes()).digest()
        frame = Frame(
            language.store,
            function_type=language.function_type,
            truth_type=truth_name,
            carriers={name: len(items) for name, items in carriers.items()},
            full_function_space_cap=full_function_space_cap,
            henkin_pool=henkin_pool,
            seed=int.from_bytes(digest[:8], byteorder="big", signed=False),
        )
        declared = _object(mapping.get("constants"), "mapping constants")
        vocabulary = language.store.signature.vocabulary(language.term_sort)
        denotations: dict[int, ModelValue] = {}

        for written_identifier, value in declared.items():
            if not written_identifier.isdecimal():
                msg = f"vocabulary id {written_identifier!r} is a nonnegative integer"
                raise WikidataFormatError(msg)

            identifier = int(written_identifier)

            if identifier >= len(vocabulary.names):
                msg = f"vocabulary id {identifier} lies outside the term vocabulary"
                raise WikidataFormatError(msg)

            name = vocabulary.names[identifier]

            if name in language.profile.logical_constants:
                msg = f"logical constant {name!r} is interpreted by the language"
                raise WikidataFormatError(msg)

            denotations[identifier] = _denotation(
                _object(value, f"constant {identifier}"),
                language.scheme(name).body,
                frame,
                carriers,
                facts,
                settings,
                language,
                truth,
            )

        for identifier, name in enumerate(vocabulary.names):
            if identifier in denotations or name in language.profile.logical_constants:
                continue
            if language.scheme(name).quantified:
                continue

            value = unmapped.resolve(identifier, language, frame)
            if value is not None:
                denotations[identifier] = value

        return Structure("wikidata", frame, denotations)


WIKIDATA = _WikidataReader()
