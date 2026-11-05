"""Configuration composition with per-leaf provenance.

:class:`SpecLoader` applies defaults, group fragments, a profile, in-process
settings, and command-line overrides in ascending precedence, then validates
the result. It records the source of each leaf and translates engine errors
into LoFI configuration errors.

An override whose key contains ``/`` selects a configuration group rather
than setting a leaf: ``model/token_encoder=static`` fills the
``model.token_encoder`` slot with the ``static`` fragment. A slot at the root is written
with a leading ``/``, so ``/device_policy=cpu`` selects a top-level group.
On the final dotted segment, ``field+`` appends to a list and ``+field``
prepends to it; an unmarked field replaces the list whole.
"""

import json
import tomllib
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple

import didactic.api as dx
from didactic.settings import (
    CoercionError,
    ConfigError,
    ConfigValue,
    InterpolationError,
    Layer,
    MissingFragmentError,
    Origin,
    OriginKind,
    OverrideSyntaxError,
    UnknownKeyError,
    compose_traced,
    nest_override,
    parse_override,
)

from lofi.spec._coerce import TypedSetting, settings_from_overrides
from lofi.spec._errors import (
    AmbiguousSettingError,
    ConfigDocumentError,
    ConfigValueError,
    MalformedOverrideError,
    MissingSettingError,
    UnknownSettingError,
    UnresolvedFragmentError,
)
from lofi.spec._interpolation import register_resolvers
from lofi.spec._plugins import load_plugins
from lofi.spec._ports import ConfigSource, LeafOrigin, Provenance
from lofi.spec.components.run import RunSpec

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from lofi._types import JsonValue

__all__ = [
    "SOURCE_OF_KIND",
    "Composition",
    "RunSpecCodec",
    "SpecLoader",
]

SOURCE_OF_KIND: Final[Mapping[OriginKind, ConfigSource]] = MappingProxyType(
    {
        "default": ConfigSource.DEFAULT,
        "defaults": ConfigSource.GROUP,
        "group": ConfigSource.GROUP,
        "file": ConfigSource.FILE,
        "profile": ConfigSource.PROFILE,
        "base": ConfigSource.EXTRA,
        "overlay": ConfigSource.EXTRA,
        "source": ConfigSource.EXTRA,
        "override": ConfigSource.OVERRIDE,
    }
)
"""The reported source for each configuration-engine origin kind."""


class Composition[M: dx.Model](NamedTuple):
    """A composed specification with the record of where each leaf came from.

    Attributes
    ----------
    spec : M
        The validated specification.
    provenance : Provenance
        The source of the value each leaf holds.
    sources : Mapping[str, tuple[LeafOrigin, ...]]
        Every source that set each dotted path, lowest precedence first. A
        path is a leaf of one layer's document: a scalar, a list, an empty
        mapping, or the root of a mapping a layer set whole. Where
        :attr:`provenance` names the source that won, this names each source
        that was overruled as well.
    """

    spec: M
    provenance: Provenance
    sources: Mapping[str, tuple[LeafOrigin, ...]]


class SpecLoader:
    """Composes specifications through the configuration engine.

    Parameters
    ----------
    search_path : tuple[Path, ...]
        The roots group fragments and profiles are looked for under, after
        the directory of the document being composed.
    """

    __slots__ = ("_search_path",)

    def __init__(self, *, search_path: tuple[Path, ...] = ()) -> None:
        self._search_path = search_path

    @property
    def search_path(self) -> tuple[Path, ...]:
        """Return the roots searched after the document's own directory."""
        return self._search_path

    def compose_traced[M: dx.Model](
        self,
        path: Path,
        /,
        *,
        schema: type[M],
        profile: str | None = None,
        overrides: tuple[str, ...] = (),
        extra: Mapping[str, JsonValue] | None = None,
    ) -> Composition[M]:
        """Compose a specification and the record of where each leaf came from.

        Parameters
        ----------
        path : Path
            Configuration document to load.
        schema : type[M]
            Model used to validate the result.
        profile : str | None
            Named profile to apply.
        overrides : tuple[str, ...]
            Command-line override expressions.
        extra : Mapping[str, JsonValue] | None
            In-process settings applied before overrides, keyed by dotted
            path.

        Returns
        -------
        Composition[M]
            The validated specification with its provenance.

        Raises
        ------
        ConfigError
            If composition or schema validation fails at a leaf.
        ConfigDocumentError
            If a document is refused as a whole.
        MalformedOverrideError
            If an override does not read as ``key=value``.
        Exception
            If a declaration plugin fails to load.
        """
        load_plugins()
        register_resolvers(replace=True)
        typed = _typed_overrides(schema, overrides)
        prepared = tuple(zip(overrides, typed, strict=True))
        groups = tuple(
            (expression, setting)
            for expression, setting in prepared
            if isinstance(setting, str) and "/" in parse_override(setting)[0]
        )
        group_expressions = frozenset(expression for expression, _ in groups)
        fields = tuple(
            (expression, setting)
            for expression, setting in prepared
            if expression not in group_expressions
        )
        settings = tuple(
            (expression, setting)
            for expression, setting in fields
            if isinstance(setting, TypedSetting)
        )
        expressions = {setting.path: expression for expression, setting in settings}
        overlays = [nest_override(key, value) for key, value in (extra or {}).items()]

        try:
            composed = compose_traced(
                path,
                schema=schema,
                groups=dict(_selection(setting) for _, setting in groups),
                profile=profile,
                overlays=overlays,
                overrides=[setting for _, setting in fields],
                search_path=self._search_path,
            )

        except OSError as refused:
            raise ConfigDocumentError(document=path, reason=str(refused)) from refused

        except (json.JSONDecodeError, tomllib.TOMLDecodeError) as refused:
            raise ConfigDocumentError(
                document=path, reason=f"a file does not parse: {refused}"
            ) from refused

        except UnknownKeyError as refused:
            raise _unknown_setting(refused, document=path) from refused

        except CoercionError as refused:
            raise _at_leaf_or_document(
                refused.path,
                document=path,
                reason=f"expects {refused.expected}; got {refused.text}",
            ) from refused

        except MissingFragmentError as refused:
            raise UnresolvedFragmentError(
                slot=refused.group or "defaults",
                name=refused.name,
                tried=tuple(str(tried) for tried in refused.tried),
                available=refused.available,
            ) from refused

        except ConfigError as refused:
            raise _at_leaf_or_document(
                refused.path, document=path, reason=str(refused)
            ) from refused

        except InterpolationError as refused:
            leaf = (
                None
                if refused.path is None
                else ".".join(str(segment) for segment in refused.path)
            )

            raise _at_leaf_or_document(
                leaf, document=path, reason=str(refused)
            ) from refused

        except dx.ValidationError as refused:
            raise _schema_refusal(refused, document=path) from refused

        origins = {
            leaf: _leaf_origin(origin, expressions)
            for leaf, origin in composed.provenance.items()
        }

        return Composition(
            composed.value,
            Provenance(origins=origins),
            MappingProxyType(
                _layer_sources(
                    composed.layers, tuple(expression for expression, _ in fields)
                )
            ),
        )


class RunSpecCodec:
    """Composes run specifications: a loader bound to the run schema.

    Parameters
    ----------
    search_path : tuple[Path, ...]
        The roots searched after the configuration document's directory.
    """

    __slots__ = ("_loader",)

    def __init__(self, *, search_path: tuple[Path, ...] = ()) -> None:
        self._loader = SpecLoader(search_path=search_path)

    @property
    def search_path(self) -> tuple[Path, ...]:
        """Return the roots searched after the document's directory."""
        return self._loader.search_path

    def __call__(
        self,
        path: Path,
        /,
        *,
        profile: str | None = None,
        overrides: tuple[str, ...] = (),
        extra: Mapping[str, JsonValue] | None = None,
    ) -> RunSpec:
        """Compose and validate a run specification.

        Parameters
        ----------
        path : Path
            Configuration document to load.
        profile : str | None
            Named profile to apply.
        overrides : tuple[str, ...]
            Command-line override expressions.
        extra : Mapping[str, JsonValue] | None
            In-process settings applied before overrides, keyed by dotted
            path.

        Returns
        -------
        RunSpec
            Validated run specification.

        Raises
        ------
        ConfigError
            If composition or schema validation fails at a leaf.
        ConfigDocumentError
            If a document is refused as a whole.
        MalformedOverrideError
            If an override does not read as ``key=value``.
        Exception
            If a declaration plugin fails to load.
        """
        return self._loader.compose_traced(
            path,
            schema=RunSpec,
            profile=profile,
            overrides=overrides,
            extra=extra,
        ).spec


def _typed_overrides(
    schema: type[dx.Model], overrides: tuple[str, ...]
) -> list[str | TypedSetting]:
    """Return override expressions read against the schema.

    Overrides whose path is not an ordinary declared leaf remain textual for
    the composition engine. This includes configuration-group selections and
    the engine's native append and prepend operations. It also includes a
    field the variants of a slot declare with different types: the engine
    reads that text against the variant the composed document has selected
    by the time the override applies, and refuses it while no variant is
    selected.

    Parameters
    ----------
    schema : type[dx.Model]
        Model that declares the setting types.
    overrides : tuple[str, ...]
        Override expressions in input order.

    Returns
    -------
    list[str | TypedSetting]
        Parsed overrides in input order.

    Raises
    ------
    MalformedOverrideError
        If an expression does not use ``key=value`` syntax.
    """
    typed: list[str | TypedSetting] = []

    for expression in overrides:
        try:
            try:
                typed.extend(settings_from_overrides(schema, (expression,)))

            except UnknownSettingError, AmbiguousSettingError:
                typed.append(expression)

        except OverrideSyntaxError as refused:
            raise MalformedOverrideError(
                expression=expression, reason=str(refused)
            ) from refused

    return typed


def _selection(expression: str, /) -> tuple[str, str | None]:
    """Return the dotted slot and fragment name a group selection names.

    Parameters
    ----------
    expression : str
        A ``key=value`` override whose key contains ``/``. A leading or
        trailing ``/`` is part of the spelling rather than of the slot, so
        ``/device_policy`` names the top-level ``device_policy`` slot.

    Returns
    -------
    tuple[str, str | None]
        The slot in dotted form, and the fragment name, or None when the value
        is empty or ``null``, which leaves the slot unfilled.
    """
    key, text = parse_override(expression)
    slot = key.strip("/").replace("/", ".")
    name = text.strip()

    return slot, None if name.lower() in {"", "null", "~"} else name


def _unknown_setting(
    refused: UnknownKeyError, *, document: Path
) -> UnknownSettingError | ConfigDocumentError:
    """Return an unknown-key refusal in the loader's error vocabulary.

    Parameters
    ----------
    refused : UnknownKeyError
        Configuration-engine error.
    document : Path
        Document being composed.

    Returns
    -------
    UnknownSettingError | ConfigDocumentError
        Leaf error when a path is known, otherwise a document error.
    """
    if refused.path is None:
        return ConfigDocumentError(document=document, reason=str(refused))

    prefix, _, _ = refused.path.rpartition(".")
    declared = tuple(f"{prefix}.{name}" if prefix else name for name in refused.allowed)

    return UnknownSettingError(
        path=refused.path,
        declared=declared,
        owners=tuple(str(owner) for owner in refused.declared_by),
    )


def _at_leaf_or_document(
    leaf: str | None, *, document: Path, reason: str
) -> ConfigValueError | ConfigDocumentError:
    """Return a leaf error or a document error.

    Parameters
    ----------
    leaf : str | None
        Dotted leaf path, if known.
    document : Path
        Document being composed.
    reason : str
        Refusal message.

    Returns
    -------
    ConfigValueError | ConfigDocumentError
        Error at the most specific known location.
    """
    if leaf:
        return ConfigValueError(path=leaf, reason=reason)

    return ConfigDocumentError(document=document, reason=reason)


def _schema_refusal(
    refused: dx.ValidationError, *, document: Path
) -> ConfigValueError | MissingSettingError | ConfigDocumentError:
    """Return the first schema refusal as a LoFI configuration error.

    Parameters
    ----------
    refused : dx.ValidationError
        Schema validation error.
    document : Path
        Document being composed.

    Returns
    -------
    ConfigValueError | MissingSettingError | ConfigDocumentError
        Error at the first refused leaf or at the document.
    """
    first = refused.entries[0]
    leaf = ".".join(str(segment) for segment in first.loc)
    reasons = [first.msg]
    reasons.extend(
        f"{'.'.join(str(segment) for segment in entry.loc)}: {entry.msg}"
        for entry in refused.entries[1:]
    )
    reason = "; also ".join(reasons)

    if first.type == "missing_required" and leaf:
        return MissingSettingError(path=leaf)

    return _at_leaf_or_document(leaf, document=document, reason=reason)


def _document_paths(document: ConfigValue, prefix: str, /) -> tuple[str, ...]:
    """Return the dotted paths a layer document sets.

    Parameters
    ----------
    document : ConfigValue
        The layer's document, or a value inside it.
    prefix : str
        The dotted path the value sits at, empty at the root.
    """
    if isinstance(document, dict) and document:
        return tuple(
            path
            for key, value in document.items()
            for path in _document_paths(value, f"{prefix}.{key}" if prefix else key)
        )

    return (prefix,)


def _layer_sources(
    layers: tuple[Layer, ...], expressions: tuple[str, ...]
) -> dict[str, tuple[LeafOrigin, ...]]:
    """Return every source that set each dotted path, lowest precedence first.

    Parameters
    ----------
    layers : tuple[Layer, ...]
        The engine's ladder, lowest precedence first.
    expressions : tuple[str, ...]
        Field overrides in the order their layers were applied.
    """
    overrides = iter(expressions)
    sources: dict[str, list[LeafOrigin]] = {}

    for layer in layers:
        origin = layer.origin
        named = (
            LeafOrigin(source=ConfigSource.OVERRIDE, layer=next(overrides))
            if origin.kind == "override"
            else LeafOrigin(
                source=SOURCE_OF_KIND[origin.kind], layer=_layer(origin, {})
            )
        )

        for path in _document_paths(dict(layer.document), ""):
            sources.setdefault(path, []).append(named)

    return {path: tuple(named) for path, named in sources.items()}


def _leaf_origin(origin: Origin, expressions: Mapping[str, str]) -> LeafOrigin:
    """Return the engine's origin record in the loader's vocabulary.

    Parameters
    ----------
    origin : Origin
        The engine's record for one leaf.
    expressions : Mapping[str, str]
        The override expression as the user typed it, keyed by the setting it
        sets.
    """
    return LeafOrigin(
        source=SOURCE_OF_KIND[origin.kind],
        layer=_layer(origin, expressions),
        expression=origin.expression,
    )


def _layer(origin: Origin, expressions: Mapping[str, str]) -> str:
    """Return the source member named by an origin.

    Parameters
    ----------
    origin : Origin
        Configuration-engine origin.
    expressions : Mapping[str, str]
        Original override expressions keyed by setting.

    Returns
    -------
    str
        Profile, override, or file name; empty for an anonymous overlay.
    """
    if origin.kind == "profile":
        return origin.name

    if origin.kind == "override":
        key, _, _ = origin.name.partition("=")

        return expressions.get(key, origin.name)

    if origin.path is not None:
        return origin.path

    if origin.kind == "overlay":
        return ""

    return origin.name
