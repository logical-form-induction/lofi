"""Parsing of untyped configuration input against a schema.

The parsers convert environment variables, namespace attributes, and override
expressions to JSON-shaped values of the declared leaf type. Unsupported
annotations are rejected, while ``${...}`` expressions remain text for the
composition engine to resolve and validate.
"""

import enum
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePath
from types import GenericAlias, MappingProxyType, NoneType, UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Final,
    ForwardRef,
    Literal,
    NamedTuple,
    TypeAliasType,
    TypeIs,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

import didactic.api as dx
from didactic.settings import CoercionError, decode_text, parse_override

from lofi.spec._errors import (
    AmbiguousSettingError,
    SettingParseError,
    UndeclaredLeafError,
    UnknownSettingError,
)

if TYPE_CHECKING:
    import argparse

    from lofi._types import JsonValue

__all__ = [
    "PARSERS",
    "Annotation",
    "LeafKind",
    "Parser",
    "TypedSetting",
    "annotation_at",
    "declared_paths",
    "leaf_kind",
    "parse_text",
    "settings_from_environment",
    "settings_from_namespace",
    "settings_from_overrides",
    "typed_setting",
]

type Annotation = type | UnionType | GenericAlias | TypeAliasType | TypeVar | ForwardRef
"""A field annotation as a schema declares it."""

type Parser = Callable[[str, Annotation, str], JsonValue]
"""Reads text for an annotation at a dotted path, named for the error."""

_NULL_TEXTS: Final = frozenset({"", "null", "~"})
"""Text an optional setting reads as ``None``, compared case-insensitively."""


class TypedSetting(NamedTuple):
    """One setting read from text: a dotted path and the value it now holds."""

    path: str
    value: JsonValue


class LeafKind(enum.Enum):
    """The classes of annotation a parser is declared for."""

    #: Text, kept as it arrived.
    TEXT = "text"

    #: A boolean, written as ``true``, ``yes``, ``on`` or ``1`` and their negations.
    BOOLEAN = "boolean"

    #: An integer literal.
    INTEGER = "integer"

    #: A float literal, including ``inf`` and ``nan``.
    FLOAT = "float"

    #: A filesystem path, kept as text.
    PATH = "path"

    #: One member of a ``Literal``.
    LITERAL = "literal"

    #: A member of an enumeration, by name or by value.
    ENUM = "enum"

    #: ``T | None``: null text, or the text read as ``T``.
    OPTIONAL = "optional"

    #: A union of scalars such as ``int | str``: the text read as the first
    #: member, in declaration order, that reads it.
    CHOICE = "choice"

    #: ``tuple[T, ...]`` or ``frozenset[T]``: a JSON list, or comma-separated items.
    SEQUENCE = "sequence"

    #: ``dict[str, T]``, set whole from JSON object text.
    MAPPING = "mapping"

    #: A model or a tagged-union slot, set whole from JSON object text.
    MODEL = "model"


def leaf_kind(annotation: Annotation, *, path: str) -> LeafKind:
    """Return the kind whose parser reads text for an annotation.

    Parameters
    ----------
    annotation : Annotation
        The annotation a schema declares at ``path``.
    path : str
        Dotted path of the setting, named in the error.

    Returns
    -------
    LeafKind
        The kind the annotation falls under.

    Raises
    ------
    UndeclaredLeafError
        If the annotation falls under no declared kind.
    """
    base = _unwrapped(annotation)
    origin = get_origin(base)

    if origin is UnionType:
        union = _union_kind(base, path=path)

        if union is not None:
            return union

    elif origin is Literal:
        return LeafKind.LITERAL

    elif origin is tuple:
        arguments = get_args(base)

        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return _containing(LeafKind.SEQUENCE, arguments[0], path=path)

    elif origin is frozenset:
        return _containing(LeafKind.SEQUENCE, get_args(base)[0], path=path)

    elif origin is dict:
        key, value = get_args(base)

        if key is str:
            return _containing(LeafKind.MAPPING, value, path=path)

    elif isinstance(base, type):
        return _class_kind(base, annotation, path=path)

    raise UndeclaredLeafError(path=path, annotation=_rendered(annotation))


def _union_kind(base: Annotation, *, path: str) -> LeafKind | None:
    """Return the kind of a union annotation, or None where no kind reads it.

    Parameters
    ----------
    base : Annotation
        The unwrapped union.
    path : str
        The setting path used in errors.
    """
    arguments = get_args(base)
    members = [member for member in arguments if member is not NoneType]

    if len(members) == 1 and len(arguments) == 2:
        return _containing(LeafKind.OPTIONAL, members[0], path=path)

    if len(members) == len(arguments) and all(
        member in _SCALAR_KINDS for member in members
    ):
        return LeafKind.CHOICE

    return None


def _containing(kind: LeafKind, inner: Annotation, *, path: str) -> LeafKind:
    """Return a container's kind once its contained annotation has one.

    Parameters
    ----------
    kind : LeafKind
        The container kind.
    inner : Annotation
        The contained annotation.
    path : str
        The setting path used in errors.
    """
    leaf_kind(inner, path=path)

    return kind


_SCALAR_KINDS: Final[Mapping[type, LeafKind]] = MappingProxyType(
    {
        str: LeafKind.TEXT,
        bool: LeafKind.BOOLEAN,
        int: LeafKind.INTEGER,
        float: LeafKind.FLOAT,
    }
)
"""The kind of each built-in scalar class, matched exactly rather than by subclass."""


def _class_kind(base: type, annotation: Annotation, *, path: str) -> LeafKind:
    """Return the kind of a bare class annotation.

    Parameters
    ----------
    base : type
        The class to classify.
    annotation : Annotation
        The original annotation.
    path : str
        The setting path used in errors.
    """
    scalar = _SCALAR_KINDS.get(base)

    if scalar is not None:
        return scalar

    if issubclass(base, PurePath):
        return LeafKind.PATH

    if issubclass(base, enum.Enum):
        return LeafKind.ENUM

    if issubclass(base, dx.Model):
        return LeafKind.MODEL

    raise UndeclaredLeafError(path=path, annotation=_rendered(annotation))


def _unwrapped(annotation: Annotation) -> Annotation:
    """Return an annotation with ``Annotated`` metadata and aliases stripped.

    A ``type`` alias is replaced by its value, with the alias's arguments
    substituted when it is generic, so ``Embed[Corner]`` reads as ``Corner``.

    Parameters
    ----------
    annotation : Annotation
        The annotation to unwrap.
    """
    base = annotation

    while True:
        origin = get_origin(base)

        if origin is Annotated:
            base = cast("Annotation", get_args(base)[0])

        elif isinstance(origin, TypeAliasType):
            arguments = get_args(base)
            base = cast("Annotation", origin.__value__[arguments])

        elif isinstance(base, TypeAliasType):
            base = cast("Annotation", base.__value__)

        else:
            return base


def _rendered(annotation: Annotation) -> str:
    """Return an annotation as an error message names it.

    Parameters
    ----------
    annotation : Annotation
        The annotation to render.
    """
    if isinstance(annotation, type):
        return annotation.__name__

    return repr(annotation).replace("typing.", "")


def _read_text(text: str, annotation: Annotation, path: str) -> JsonValue:
    """Return text as it arrived.

    Parameters
    ----------
    text : str
        The text to return.
    annotation : Annotation
        The target annotation.
    path : str
        The setting path.
    """
    del annotation, path

    return text


def _read_scalar(text: str, annotation: Annotation, path: str) -> JsonValue:
    """Return text read by the engine's grammar for a scalar annotation.

    Parameters
    ----------
    text : str
        The text to parse.
    annotation : Annotation
        The scalar annotation.
    path : str
        The setting path used in errors.
    """
    try:
        # a scalar annotation decodes to a scalar, so the engine's wider
        # value type narrows to the JSON one
        return cast("JsonValue", decode_text(text, _unwrapped(annotation), path=path))

    except CoercionError as refused:
        raise SettingParseError(
            path=path, expected=refused.expected, text=text
        ) from refused


def _read_optional(text: str, annotation: Annotation, path: str) -> JsonValue:
    """Return ``None`` for null text, else the text read as the other member.

    Parameters
    ----------
    text : str
        The text to parse.
    annotation : Annotation
        The optional annotation.
    path : str
        The setting path used in errors.
    """
    if text.strip().lower() in _NULL_TEXTS:
        return None

    base = _unwrapped(annotation)
    inner = next(member for member in get_args(base) if member is not NoneType)

    return parse_text(text, cast("Annotation", inner), path=path)


def _read_choice(text: str, annotation: Annotation, path: str) -> JsonValue:
    """Return the text read as the first scalar member of a union that reads it.

    Parameters
    ----------
    text : str
        The text to parse.
    annotation : Annotation
        The union annotation.
    path : str
        The setting path used in errors.

    Raises
    ------
    SettingParseError
        If no member reads the text.
    """
    members = get_args(_unwrapped(annotation))
    refusals: list[SettingParseError] = []

    for member in members:
        try:
            return parse_text(text, cast("Annotation", member), path=path)

        except SettingParseError as refused:
            refusals.append(refused)

    expected = " or ".join(refused.expected for refused in refusals)

    raise SettingParseError(path=path, expected=expected, text=text)


def _read_sequence(text: str, annotation: Annotation, path: str) -> JsonValue:
    """Return a JSON list, or comma-separated items each read as the item type.

    Text that is one whole expression is kept as it arrived, so the list the
    expression resolves to fills the slot.

    Parameters
    ----------
    text : str
        The text to parse.
    annotation : Annotation
        The sequence annotation.
    path : str
        The setting path used in errors.
    """
    stripped = text.strip()

    if stripped.startswith("["):
        return _read_json(stripped, annotation, path=path)

    if not stripped:
        return []

    # the engine's item grammar keeps a whole expression as text and otherwise
    # splits on the commas outside brackets and quotes; reading the text as
    # items of text keeps every piece verbatim
    pieces = decode_text(stripped, tuple[str, ...], path=path)

    if isinstance(pieces, str):
        return pieces

    item = cast("Annotation", get_args(_unwrapped(annotation))[0])

    return [parse_text(piece, item, path=path) for piece in cast("list[str]", pieces)]


def _read_object(text: str, annotation: Annotation, path: str) -> JsonValue:
    """Return the JSON object a whole mapping or model slot is set from.

    Text holding an expression is kept as it arrived.

    Parameters
    ----------
    text : str
        The text to parse.
    annotation : Annotation
        The mapping or model annotation.
    path : str
        The setting path used in errors.
    """
    stripped = text.strip()

    if stripped.startswith("{"):
        return _read_json(stripped, annotation, path=path)

    # the engine keeps text holding an expression and refuses any other text
    # that is not JSON object text
    return _read_scalar(text, annotation, path)


def _read_json(text: str, annotation: Annotation, *, path: str) -> JsonValue:
    """Return the value JSON text holds.

    Parameters
    ----------
    text : str
        The JSON text to parse.
    annotation : Annotation
        The expected annotation.
    path : str
        The setting path used in errors.
    """
    try:
        return cast("JsonValue", json.loads(text))

    except json.JSONDecodeError as malformed:
        expected = f"valid JSON for {_rendered(annotation)}: {malformed.msg}"

        raise SettingParseError(path=path, expected=expected, text=text) from None


PARSERS: Final[Mapping[LeafKind, Parser]] = MappingProxyType(
    {
        LeafKind.TEXT: _read_text,
        LeafKind.BOOLEAN: _read_scalar,
        LeafKind.INTEGER: _read_scalar,
        LeafKind.FLOAT: _read_scalar,
        LeafKind.PATH: _read_text,
        LeafKind.LITERAL: _read_scalar,
        LeafKind.ENUM: _read_scalar,
        LeafKind.OPTIONAL: _read_optional,
        LeafKind.CHOICE: _read_choice,
        LeafKind.SEQUENCE: _read_sequence,
        LeafKind.MAPPING: _read_object,
        LeafKind.MODEL: _read_object,
    }
)
"""The parser declared for each leaf kind."""


def parse_text(text: str, annotation: Annotation, *, path: str) -> JsonValue:
    """Return text read as a value of the type an annotation declares.

    Parameters
    ----------
    text : str
        The text as it arrived.
    annotation : Annotation
        The annotation the schema declares at ``path``.
    path : str
        Dotted path of the setting, named in the error.

    Returns
    -------
    JsonValue
        The value, in the JSON shape the engine passes to the field.

    Raises
    ------
    SettingParseError
        If the text does not read as the annotation.
    UndeclaredLeafError
        If the annotation falls under no declared kind.
    """
    return PARSERS[leaf_kind(annotation, path=path)](text, annotation, path)


def annotation_at(schema: type[dx.Model], path: str, /) -> Annotation:
    """Return the annotation a schema declares at a dotted path.

    A segment names a field of the model at that depth, a variant field of a
    tagged-union slot, or an entry of a mapping. The discriminator of a
    tagged-union slot reads as text whichever variant is later selected. A
    field several variants declare must be declared alike by all of them.

    Parameters
    ----------
    schema : type[dx.Model]
        The model the path starts from.
    path : str
        Dotted path of the setting.

    Returns
    -------
    Annotation
        The annotation at the path, as the field declares it.

    Raises
    ------
    UnknownSettingError
        If a segment names no field, or continues below a setting that is set
        whole.
    AmbiguousSettingError
        If the variants of a slot declare the field with different types.
    """
    segments = path.split(".")

    if not all(segments):
        declared = tuple(_below((), sorted(schema.__field_specs__)))

        raise UnknownSettingError(path=path, declared=declared)

    annotation: Annotation = schema
    prefix: tuple[str, ...] = ()

    for segment in segments:
        annotation = _member(annotation, segment, prefix=prefix)
        prefix = (*prefix, segment)

    return annotation


def _member(
    annotation: Annotation, segment: str, *, prefix: tuple[str, ...]
) -> Annotation:
    """Return the annotation one segment below a node.

    Parameters
    ----------
    annotation : Annotation
        The current node annotation.
    segment : str
        The child field name.
    prefix : tuple[str, ...]
        The path above the child.
    """
    base = _node(annotation)
    here = ".".join((*prefix, segment))

    if isinstance(base, type) and issubclass(base, dx.Model):
        if _is_union_root(base):
            return _variant_member(base, segment, here=here)

        spec = base.__field_specs__.get(segment)

        if spec is None:
            declared = tuple(_below(prefix, sorted(base.__field_specs__)))

            raise UnknownSettingError(path=here, declared=declared)

        return spec.annotation

    if get_origin(base) is dict:
        return cast("Annotation", get_args(base)[1])

    raise UnknownSettingError(path=here, declared=(".".join(prefix),))


def _variant_member(
    root: type[dx.TaggedUnion], segment: str, *, here: str
) -> Annotation:
    """Return the annotation the variants of a union root declare for a field.

    Parameters
    ----------
    root : type[dx.TaggedUnion]
        The tagged-union root.
    segment : str
        The field name to inspect.
    here : str
        The full setting path used in errors.
    """
    if segment == _discriminator(root):
        return str

    found: dict[str, Annotation] = {}

    for tag, variant in _variants(root).items():
        spec = variant.__field_specs__.get(segment)

        if spec is not None:
            found[tag] = spec.annotation

    if not found:
        prefix = tuple(here.split(".")[:-1])
        declared = tuple(_below(prefix, _variant_field_names(root)))

        raise UnknownSettingError(path=here, declared=declared)

    first = next(iter(found.values()))

    if any(annotation != first for annotation in found.values()):
        raise AmbiguousSettingError(path=here, variants=tuple(found))

    return first


def _node(annotation: Annotation) -> Annotation:
    """Return the annotation a path descends through, its optional wrapper peeled.

    Parameters
    ----------
    annotation : Annotation
        The annotation to inspect.
    """
    base = _unwrapped(annotation)

    if get_origin(base) is UnionType:
        members = [member for member in get_args(base) if member is not NoneType]

        if len(members) == 1:
            return _unwrapped(cast("Annotation", members[0]))

    return base


def _is_union_root(model: type[dx.Model]) -> TypeIs[type[dx.TaggedUnion]]:
    """Return whether a model is a tagged-union root rather than a variant.

    Parameters
    ----------
    model : type[dx.Model]
        The model to inspect.
    """
    return (
        issubclass(model, dx.TaggedUnion)
        and model.__dict__.get("__discriminator__") is not None
    )


def _discriminator(root: type[dx.TaggedUnion]) -> str:
    """Return the field a root selects its variant by.

    Parameters
    ----------
    root : type[dx.TaggedUnion]
        The tagged-union root.
    """
    field = root.__discriminator__

    if field is None:
        msg = f"{root.__name__} is not a tagged-union root"

        raise TypeError(msg)

    return field


def _variants(root: type[dx.TaggedUnion]) -> dict[str, type[dx.TaggedUnion]]:
    """Return a root's registered variants by tag, in registration order.

    Parameters
    ----------
    root : type[dx.TaggedUnion]
        The tagged-union root.
    """
    return {str(tag): variant for tag, variant in root.__variants__.items()}


def _variant_field_names(root: type[dx.TaggedUnion]) -> tuple[str, ...]:
    """Return the discriminator and every field name a variant declares, sorted.

    Parameters
    ----------
    root : type[dx.TaggedUnion]
        The tagged-union root.
    """
    names: set[str] = {_discriminator(root), *root.__field_specs__}

    for variant in _variants(root).values():
        names.update(variant.__field_specs__)

    return tuple(sorted(names))


def _below(prefix: tuple[str, ...], names: Sequence[str]) -> list[str]:
    """Return names as dotted paths under a prefix.

    Parameters
    ----------
    prefix : tuple[str, ...]
        The path prefix.
    names : Sequence[str]
        The child names.
    """
    return [".".join((*prefix, name)) for name in names]


def declared_paths(schema: type[dx.Model], /) -> tuple[tuple[str, Annotation], ...]:
    """Return every dotted path text may set, with the annotation it reads as.

    Model slots include the slot itself and its fields. Tagged unions include
    every variant, mappings use ``*`` for arbitrary keys, and recursive models
    stop at the repeated slot. Sequence elements are not listed separately.

    Parameters
    ----------
    schema : type[dx.Model]
        The model the paths start from.

    Returns
    -------
    tuple[tuple[str, Annotation], ...]
        Paths in declaration order, each with its annotation.
    """
    found: dict[tuple[str, str], Annotation] = {}
    _collect(schema, prefix=(), ancestors=(), found=found)

    return tuple((path, annotation) for (path, _), annotation in found.items())


def _collect(
    annotation: Annotation,
    *,
    prefix: tuple[str, ...],
    ancestors: tuple[type, ...],
    found: dict[tuple[str, str], Annotation],
) -> None:
    """Record the paths below a node, keyed by path and rendered annotation.

    Parameters
    ----------
    annotation : Annotation
        The current node annotation.
    prefix : tuple[str, ...]
        The path to the current node.
    ancestors : tuple[type, ...]
        The model types above the current node.
    found : dict[tuple[str, str], Annotation]
        The path table to update.
    """
    if prefix:
        found.setdefault((".".join(prefix), _rendered(annotation)), annotation)

    base = _node(annotation)

    if get_origin(base) is dict:
        value = cast("Annotation", get_args(base)[1])
        _collect(value, prefix=(*prefix, "*"), ancestors=ancestors, found=found)

        return

    if not (isinstance(base, type) and issubclass(base, dx.Model)):
        return

    if base in ancestors:
        return

    if _is_union_root(base):
        discriminator = _discriminator(base)
        found.setdefault((".".join((*prefix, discriminator)), "str"), str)
        fields = [
            (name, spec)
            for variant in _variants(base).values()
            for name, spec in variant.__field_specs__.items()
            if name != discriminator
        ]

    else:
        fields = list(base.__field_specs__.items())

    for name, spec in fields:
        _collect(
            spec.annotation,
            prefix=(*prefix, name),
            ancestors=(*ancestors, base),
            found=found,
        )


def typed_setting(schema: type[dx.Model], path: str, text: str) -> TypedSetting:
    """Return the setting text at a dotted path reads as under a schema.

    Parameters
    ----------
    schema : type[dx.Model]
        The model the path starts from.
    path : str
        Dotted path of the setting.
    text : str
        The text as it arrived.

    Returns
    -------
    TypedSetting
        The path with the value the text reads as.

    Raises
    ------
    UnknownSettingError
        If the path names no setting.
    AmbiguousSettingError
        If the variants of a slot declare the setting with different types.
    SettingParseError
        If the text does not read as the setting's type.
    UndeclaredLeafError
        If the setting's type falls under no declared kind.
    """
    annotation = annotation_at(schema, path)

    return TypedSetting(path, parse_text(text, annotation, path=path))


def settings_from_environment(
    schema: type[dx.Model],
    environ: Mapping[str, str],
    *,
    prefix: str = "LOFI_",
    separator: str = "__",
    switches: frozenset[str] = frozenset({"LOFI_CHECK"}),
) -> tuple[TypedSetting, ...]:
    """Return the settings the environment sets under a schema.

    A variable is a setting when its name starts with ``prefix`` and is not
    one of the ``switches``; the rest of the name, split on ``separator`` and
    lower-cased, is the dotted path. ``LOFI_TRAINER__EPOCHS`` sets
    ``trainer.epochs`` and ``LOFI_SINKS`` sets ``sinks`` whole. Every such
    variable is read, so one naming no setting is refused rather than passed
    over.

    Parameters
    ----------
    schema : type[dx.Model]
        The model the paths start from.
    environ : Mapping[str, str]
        The environment to read.
    prefix : str
        The text a setting's variable name starts with.
    separator : str
        The text between the segments of the path.
    switches : frozenset[str]
        Full names of the prefixed variables the process reads for itself,
        which are not settings.

    Returns
    -------
    tuple[TypedSetting, ...]
        One setting per variable read, in variable-name order.

    Raises
    ------
    UnknownSettingError
        If a variable names no setting.
    AmbiguousSettingError
        If the variants of a slot declare a setting with different types.
    SettingParseError
        If a variable's text does not read as its setting's type.
    UndeclaredLeafError
        If a setting's type falls under no declared kind.
    """
    settings: list[TypedSetting] = []

    for name in sorted(environ):
        if not name.startswith(prefix) or name in switches:
            continue

        rest = name.removeprefix(prefix)
        path = ".".join(segment.lower() for segment in rest.split(separator))
        settings.append(typed_setting(schema, path, environ[name]))

    return tuple(settings)


def settings_from_namespace(
    schema: type[dx.Model], namespace: argparse.Namespace, /
) -> tuple[TypedSetting, ...]:
    """Read schema settings from a parsed namespace.

    Attribute names are dotted paths, with ``__`` also accepted. ``None`` values
    are ignored, and provided values must be strings.

    Parameters
    ----------
    schema : type[dx.Model]
        The model the paths start from.
    namespace : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    tuple[TypedSetting, ...]
        One setting per given argument, in attribute order.

    Raises
    ------
    TypeError
        If a given argument holds something other than text.
    UnknownSettingError
        If an argument names no setting.
    AmbiguousSettingError
        If the variants of a slot declare a setting with different types.
    SettingParseError
        If an argument's text does not read as its setting's type.
    UndeclaredLeafError
        If a setting's type falls under no declared kind.
    """
    settings: list[TypedSetting] = []

    for name, value in vars(namespace).items():
        if value is None:
            continue

        if not isinstance(value, str):
            msg = (
                f"argument {name!r} holds {type(value).__name__} rather than text; "
                "the command line leaves typing to the schema"
            )

            raise TypeError(msg)

        settings.append(typed_setting(schema, name.replace("__", "."), value))

    return tuple(settings)


def settings_from_overrides(
    schema: type[dx.Model], expressions: Sequence[str], /
) -> tuple[str | TypedSetting, ...]:
    """Return the settings a sequence of ``key=value`` expressions sets.

    A key containing ``/`` selects a configuration group rather than a field;
    such an expression is returned as it arrived, for the engine to resolve.
    Every other expression is read against the schema.

    Parameters
    ----------
    schema : type[dx.Model]
        The model the paths start from.
    expressions : Sequence[str]
        The ``key=value`` expressions, in the order given.

    Returns
    -------
    tuple[str | TypedSetting, ...]
        One entry per expression, in order: the expression itself for a group
        selection, the typed setting otherwise.

    Raises
    ------
    didactic.settings.OverrideSyntaxError
        If an expression lacks ``=`` or its key is malformed.
    UnknownSettingError
        If a key names no setting.
    AmbiguousSettingError
        If the variants of a slot declare a setting with different types.
    SettingParseError
        If a value does not read as its setting's type.
    UndeclaredLeafError
        If a setting's type falls under no declared kind.
    """
    settings: list[str | TypedSetting] = []

    for expression in expressions:
        key, text = parse_override(expression)

        if "/" in key:
            settings.append(expression)

        else:
            settings.append(typed_setting(schema, key, text))

    return tuple(settings)
