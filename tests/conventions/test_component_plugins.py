"""Conventions of the component families registered through plugin objects.

Each variant of a component family has exactly one entry point, in its family's
group, naming the plugin object its runtime module defines. Every default lives on
the variant's declaration: the declaration's settings value, the builder and the
constructor of the built LoFI class declare none.
"""

import annotationlib
from collections import Counter
from importlib import import_module
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast, get_args, get_origin

import pytest

from lofi.data.datasets import ArtifactDatasetPlugin, DatasetPlugin
from lofi.registry import ComponentPlugin
from lofi.spec import component_groups
from lofi.training_plugins import TrainerPlugin

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

    import didactic.api as dx


class Declaring(Protocol):
    """A declaration that resolves to a settings value."""

    def settings(self) -> object:
        """Return the settings value."""
        ...


class Entry(NamedTuple):
    """One entry point of a component group and the plugin it names."""

    group: str
    point: EntryPoint
    plugin: ComponentPlugin[object, object, object]


def installed() -> tuple[Entry, ...]:
    """Return every entry point of every component group, loaded."""
    return tuple(
        Entry(group, point, cast("ComponentPlugin[object, object, object]", loaded))
        for group in component_groups()
        for point in sorted(entry_points(group=group), key=lambda point: point.name)
        if isinstance(loaded := point.load(), ComponentPlugin)
    )


def _annotations(function: object) -> dict[str, object]:
    """Return a callable's annotations, unresolved names left as references."""
    return annotationlib.get_annotations(
        function, format=annotationlib.Format.FORWARDREF
    )


def _built(annotation: object) -> object:
    """Return a built-class annotation, a module-qualified reference resolved.

    A builder that loads its implementation module only when it builds names the
    built class through that module, so the reference is resolved by importing
    the module that owns the class.
    """
    if isinstance(annotation, annotationlib.ForwardRef):
        module, _, name = annotation.__forward_arg__.rpartition(".")

        if module:
            return getattr(import_module(module), name)

    return annotation


def _declared_defaults(function: object) -> tuple[str, ...]:
    """Return the names of the parameters a callable gives a default."""
    code = getattr(function, "__code__", None)

    if code is None:
        return ()

    positional = getattr(function, "__defaults__", None) or ()
    keyword = getattr(function, "__kwdefaults__", None) or {}
    names = code.co_varnames[: code.co_argcount]

    return (*names[len(names) - len(positional) :], *keyword)


def _settings_classes(annotation: object) -> tuple[type, ...]:
    """Return the settings classes an annotation names, nested ones included."""
    found: list[type] = []
    pending = [annotation]

    while pending:
        current = pending.pop()
        pending.extend(get_args(current))

        if (
            isinstance(current, type)
            and isinstance(getattr(current, "_field_defaults", None), dict)
            and current not in found
        ):
            found.append(current)
            pending.extend(_annotations(current).values())

    return tuple(found)


def _field_defaults(annotation: object) -> tuple[str, ...]:
    """Return every field an annotation's settings classes give a default."""
    return tuple(
        f"{settings.__name__}.{field}"
        for settings in _settings_classes(annotation)
        for field in cast("dict[str, object]", getattr(settings, "_field_defaults", {}))
    )


def defined_in_lofi(built: type) -> bool:
    """Return whether a class is defined in the LoFI package."""
    return built.__module__.startswith("lofi.")


def restated_defaults(
    spec: type[Declaring],
    build: Callable[..., object],
    *,
    owned: Callable[[type], bool] = defined_in_lofi,
) -> tuple[str, ...]:
    """Return every default a plugin states outside its declaration's fields.

    Parameters
    ----------
    spec : type[Declaring]
        The declaration class.
    build : Callable[..., object]
        The builder the plugin pairs with it.
    owned : Callable[[type], bool]
        Whether the built class is one whose constructor the convention governs.

    Returns
    -------
    tuple[str, ...]
        One description per default found on the settings value, the builder,
        the constructor of the built class or a settings value that constructor
        takes. A class defined outside LoFI is built with every argument the
        builder passes, and its own constructor is not inspected.
    """
    found = [f"builder default {name}" for name in _declared_defaults(build)]
    annotations = _annotations(spec.settings)
    settings = annotations.get("return")

    if "return" not in annotations or (
        settings is not None and not isinstance(settings, type)
    ):
        found.append(f"unresolved settings value {settings!r}")

    found.extend(f"settings default {name}" for name in _field_defaults(settings))
    returned = _built(_annotations(build).get("return"))
    built = get_origin(returned) or returned

    if not isinstance(built, type):
        found.append(f"unresolved built module {returned!r}")

    elif owned(built):
        constructor = built.__init__
        found.extend(
            f"constructor default {name}" for name in _declared_defaults(constructor)
        )

        for name, annotation in _annotations(constructor).items():
            found.extend(
                f"constructor settings {name} default {field}"
                for field in _field_defaults(annotation)
            )

    return tuple(found)


ENTRIES = installed()
ENTRY_ROWS = [
    pytest.param(entry, id=f"{entry.group}:{entry.point.name}") for entry in ENTRIES
]


def test_every_component_group_lists_only_plugins() -> None:
    listed = [
        point.load()
        for group in component_groups()
        for point in entry_points(group=group)
    ]

    assert listed
    assert all(
        isinstance(
            plugin,
            ComponentPlugin | DatasetPlugin | ArtifactDatasetPlugin | TrainerPlugin,
        )
        for plugin in listed
    )
    assert sum(isinstance(plugin, ComponentPlugin) for plugin in listed) == len(ENTRIES)


def test_each_variant_has_exactly_one_entry_point() -> None:
    counted = Counter(entry.plugin.spec for entry in ENTRIES)

    assert [spec for spec, count in counted.items() if count > 1] == []


@pytest.mark.parametrize("entry", ENTRY_ROWS)
def test_an_entry_names_the_plugin_its_variants_runtime_module_defines(
    entry: Entry,
) -> None:
    plugin = entry.plugin
    family = entry.group.removeprefix("lofi.")

    assert plugin.spec.family() == family
    kind = cast("type[dx.Model]", plugin.spec).__field_specs__["kind"]

    assert kind.default == entry.point.name
    assert entry.point.module == plugin.build.__module__
    assert entry.point.module.startswith("lofi.")
    assert not entry.point.module.startswith("lofi.spec.")


@pytest.mark.parametrize("entry", ENTRY_ROWS)
def test_no_builder_or_constructor_restates_a_default(entry: Entry) -> None:
    assert restated_defaults(entry.plugin.spec, entry.plugin.build) == ()


class _Window(NamedTuple):
    rows: int = 8


class _Width(NamedTuple):
    width: int = 4
    window: _Window | None = None


class _Module:
    def __init__(self, *, width: int, hidden: int = 64) -> None:
        self.width = width
        self.hidden = hidden


class _Spec:
    def settings(self) -> _Width:
        return _Width()


def _restating(settings: _Width, context: object, /, scale: int = 2) -> _Module:
    del context

    return _Module(width=settings.width * scale)


def _every_class(built: type) -> bool:
    del built

    return True


def test_a_restated_default_is_found_on_each_side_of_the_plugin() -> None:
    assert restated_defaults(_Spec, _restating, owned=_every_class) == (
        "builder default scale",
        "settings default _Width.width",
        "settings default _Width.window",
        "settings default _Window.rows",
        "constructor default hidden",
    )


class _Held[T]:
    def __init__(self, *, value: T, label: str = "held") -> None:
        self.value = value
        self.label = label


def _holding(settings: _Width, context: object, /) -> _Held[int]:
    del context

    return _Held(value=settings.width, label="held")


def test_a_generic_built_class_is_inspected_through_its_origin() -> None:
    found = restated_defaults(_Spec, _holding, owned=_every_class)

    assert found[-1] == "constructor default label"


def test_a_class_defined_outside_lofi_is_built_but_not_inspected() -> None:
    assert not defined_in_lofi(_Held)
    assert "constructor default label" not in restated_defaults(_Spec, _holding)
    assert "constructor default hidden" not in restated_defaults(_Spec, _restating)


def _parsing(settings: _Width, context: object, /) -> argparse.ArgumentParser:
    del settings, context

    return cast("argparse.ArgumentParser", object())


def test_a_qualified_forward_reference_is_resolved_through_its_module() -> None:
    assert "constructor default prog" in restated_defaults(
        _Spec, _parsing, owned=_every_class
    )


def test_an_unresolvable_annotation_is_reported_rather_than_skipped() -> None:
    assert restated_defaults(_Spec, lambda settings, context: (settings, context)) == (
        "settings default _Width.width",
        "settings default _Width.window",
        "settings default _Window.rows",
        "unresolved built module None",
    )


class _NoSettings:
    def settings(self) -> None:
        """Return no settings for a component without configuration."""


def _fixed(settings: None, context: object, /) -> _Held[int]:
    del settings, context
    return _Held(value=4)


def test_an_explicit_none_settings_value_has_no_defaults() -> None:
    assert restated_defaults(_NoSettings, _fixed) == ()
