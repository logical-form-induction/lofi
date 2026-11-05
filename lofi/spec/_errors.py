"""Specification and wiring errors."""

from typing import TYPE_CHECKING, Final, override

from lofi._checks import ensure_name

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "AmbiguousSettingError",
    "ConfigDirectoryError",
    "ConfigDocumentError",
    "ConfigError",
    "ConfigValueError",
    "DeclarationError",
    "DependencyCycleError",
    "DerivationIndexError",
    "IdentifierError",
    "InventoryExhaustedError",
    "LinearizationDisagreementError",
    "MalformedOverrideError",
    "MissingSettingError",
    "SemiringCapabilityError",
    "SettingParseError",
    "UndeclaredLeafError",
    "UnknownSettingError",
    "UnresolvedDependencyError",
    "UnresolvedFragmentError",
    "WiringError",
]

_SHORTEST_CYCLE: Final = 1
"""The minimum role count in a cycle."""


class ConfigError(Exception):
    """An invalid process-entry configuration.

    Parameters
    ----------
    message : str
        Human-readable error message.
    path : str
        Dotted configuration path.

    Raises
    ------
    TypeError
        If ``path`` is not a string.
    ValueError
        If ``path`` is empty.
    """

    __slots__ = ("path",)

    def __init__(self, message: str, *, path: str) -> None:
        self.path = ensure_name("path", path)

        super().__init__(message)


class UnknownSettingError(ConfigError):
    """Reports an unknown configuration path.

    Parameters
    ----------
    path : str
        Unknown dotted path.
    declared : tuple[str, ...]
        Declared paths to include in the message.
    owners : tuple[str, ...]
        The variants of the enclosing slot that declare the setting, by kind,
        when the slot selects a variant that does not; empty when no variant
        declares it.
    """

    __slots__ = ("declared", "owners")

    def __init__(
        self, *, path: str, declared: tuple[str, ...], owners: tuple[str, ...] = ()
    ) -> None:
        self.declared: tuple[str, ...] = declared
        self.owners: tuple[str, ...] = owners

        offered = ", ".join(declared) or "nothing"
        msg = f"{path} is not a declared setting; the schema declares {offered}"

        if owners:
            msg = (
                f"{path} is not a setting of the selected variant; it belongs to "
                f"{', '.join(owners)}, and the selected variant declares {offered}"
            )

        super().__init__(msg, path=path)


class MissingSettingError(ConfigError):
    """Reports a missing configuration path.

    Parameters
    ----------
    path : str
        Required dotted path.
    """

    __slots__ = ()

    def __init__(self, *, path: str) -> None:
        msg = f"{path} is required and no source set it"

        super().__init__(msg, path=path)


class ConfigValueError(ConfigError):
    """Reports a value rejected by the configuration schema.

    Parameters
    ----------
    path : str
        Dotted path of the rejected value.
    reason : str
        Schema rejection message.
    """

    __slots__ = ("reason",)

    def __init__(self, *, path: str, reason: str) -> None:
        self.reason = reason

        msg = f"{path} was refused: {reason}"

        super().__init__(msg, path=path)


class SettingParseError(ConfigError):
    """Reports text that does not read as the value a setting declares.

    Parameters
    ----------
    path : str
        Dotted path of the setting.
    expected : str
        The type the text had to read as.
    text : str
        The text that was refused.
    """

    __slots__ = ("expected", "text")

    def __init__(self, *, path: str, expected: str, text: str) -> None:
        self.expected = expected
        self.text = text

        msg = f"{path} expects {expected}; got {text!r}"

        super().__init__(msg, path=path)


class AmbiguousSettingError(ConfigError):
    """Reports a setting the variants of one slot declare with different types.

    Parameters
    ----------
    path : str
        Dotted path of the setting.
    variants : tuple[str, ...]
        Tags of the variants that declare it.
    """

    __slots__ = ("variants",)

    def __init__(self, *, path: str, variants: tuple[str, ...]) -> None:
        self.variants = variants

        named = ", ".join(variants)
        msg = (
            f"{path} is declared with a different type by each of {named}, "
            "so text for it cannot be read until the variant is selected"
        )

        super().__init__(msg, path=path)


class UnresolvedFragmentError(ConfigError):
    """Reports a fragment selection that no search root holds.

    Parameters
    ----------
    slot : str
        Dotted path of the configuration group the fragment was selected
        for; ``defaults`` for a fragment mounted at the root, ``profiles``
        for a profile.
    name : str
        The option that was selected.
    tried : tuple[str, ...]
        Every file looked for, in the order looked.
    available : tuple[str, ...]
        The options that do exist for the slot, across every root.
    """

    __slots__ = ("available", "name", "slot", "tried")

    def __init__(
        self,
        *,
        slot: str,
        name: str,
        tried: tuple[str, ...],
        available: tuple[str, ...],
    ) -> None:
        self.slot = ensure_name("slot", slot)
        self.name = name
        self.tried = tried
        self.available = available

        offered = ", ".join(available) or "nothing"
        looked = ", ".join(tried) or "nowhere"
        msg = (
            f"{slot} selects {name!r}, which is not an option; "
            f"the search path offers {offered}, having looked in {looked}"
        )

        super().__init__(msg, path=slot)


class MalformedOverrideError(ValueError):
    """An override expression that does not read as ``key=value``.

    Parameters
    ----------
    expression : str
        The expression as it arrived.
    reason : str
        What the expression gets wrong.
    """

    __slots__ = ("expression", "reason")

    def __init__(self, *, expression: str, reason: str) -> None:
        self.expression = expression
        self.reason = reason

        super().__init__(f"override {expression!r} is malformed: {reason}")


class ConfigDocumentError(Exception):
    """A configuration refused as a whole rather than at one leaf.

    Parameters
    ----------
    document : Path
        The document the loader was pointed at.
    reason : str
        Why the composition was refused.
    """

    __slots__ = ("document", "reason")

    def __init__(self, *, document: Path, reason: str) -> None:
        self.document = document
        self.reason = reason

        super().__init__(f"{document} cannot be composed: {reason}")


class UndeclaredLeafError(TypeError):
    """A setting whose declared type no text parser is declared for.

    Parameters
    ----------
    path : str
        Dotted path of the setting.
    annotation : str
        The type the schema declares at the path.
    """

    __slots__ = ("annotation", "path")

    def __init__(self, *, path: str, annotation: str) -> None:
        self.path = ensure_name("path", path)
        self.annotation = annotation

        super().__init__(f"{path} is declared as {annotation}, which no parser reads")


class ConfigDirectoryError(Exception):
    """A configuration-directory entry point that names no package directory.

    Parameters
    ----------
    entry : str
        Name of the entry point.
    value : str
        What the entry point names.
    reason : str
        Why that is not a package directory.
    """

    __slots__ = ("entry", "reason", "value")

    def __init__(self, *, entry: str, value: str, reason: str) -> None:
        self.entry = ensure_name("entry", entry)
        self.value = value
        self.reason = reason

        super().__init__(f"config_dirs entry point {entry!r} ({value}) {reason}")


class WiringError(Exception):
    """An invalid component graph.

    Parameters
    ----------
    message : str
        Human-readable error message.
    left : str
        First component role involved in the error.
    right : str
        Second component role involved in the error.

    Raises
    ------
    TypeError
        If either role is not a string.
    ValueError
        If either role is empty.
    """

    __slots__ = ("left", "right")

    def __init__(self, message: str, *, left: str, right: str) -> None:
        self.left = ensure_name("left", left)
        self.right = ensure_name("right", right)

        super().__init__(message)


class DependencyCycleError(WiringError):
    """Reports a dependency cycle.

    Parameters
    ----------
    cycle : tuple[str, ...]
        Distinct component roles in cycle order.

    Raises
    ------
    ValueError
        If no role is provided or a role is repeated.
    """

    __slots__ = ("cycle",)

    def __init__(self, *, cycle: tuple[str, ...]) -> None:
        if len(cycle) < _SHORTEST_CYCLE:
            msg = "a dependency cycle must name at least one role"

            raise ValueError(msg)

        if len(set(cycle)) != len(cycle):
            msg = f"{cycle} names a role twice, so it is not one cycle"

            raise ValueError(msg)

        self.cycle: tuple[str, ...] = cycle

        route = " -> ".join((*cycle, cycle[0]))
        message = f"the dependencies close a cycle: {route}"

        super().__init__(message, left=cycle[0], right=cycle[-1])


class UnresolvedDependencyError(WiringError):
    """Reports an unresolved component dependency.

    Parameters
    ----------
    consumer : str
        Dependent component role.
    producer : str
        Referenced component role.
    port : str
        Required input port.
    reason : str
        Reason the dependency cannot be resolved.
    """

    __slots__ = ("port", "reason")

    def __init__(self, *, consumer: str, producer: str, port: str, reason: str) -> None:
        self.port = port
        self.reason = reason

        message = f"{consumer} takes {port} from {producer}, which {reason}"

        super().__init__(message, left=consumer, right=producer)


class LinearizationDisagreementError(WiringError):
    """Reports component codec views that do not share one identity.

    Parameters
    ----------
    left : str
        First component role.
    left_codec : str
        Codec-view identity used by the first component.
    right : str
        Second component role.
    right_codec : str
        Codec-view identity used by the second component.
    """

    __slots__ = ("left_codec", "right_codec")

    def __init__(
        self, *, left: str, left_codec: str, right: str, right_codec: str
    ) -> None:
        self.left_codec = left_codec
        self.right_codec = right_codec

        message = (
            f"{left} reads codec identity {left_codec!r} and {right} reads codec "
            f"identity {right_codec!r}, so they do not share one linearization"
        )

        super().__init__(message, left=left, right=right)


class DeclarationError(ValueError):
    """An invalid declaration, optionally located at its configuration path."""

    __slots__ = ("path", "reason")

    def __init__(self, reason: str, *, path: str | None = None) -> None:
        self.reason = reason
        self.path = None if path is None else ensure_name("path", path)
        message = reason if self.path is None else f"{self.path} was refused: {reason}"

        super().__init__(message)

    def prefixed(self, prefix: str, /) -> DeclarationError:
        """Return this refusal under one enclosing configuration path."""
        held = ensure_name("path prefix", prefix)
        path = held if self.path is None else f"{held}.{self.path}"

        return DeclarationError(self.reason, path=path)


class SemiringCapabilityError(DeclarationError):
    """A parse consumer refusing a declared semiring that lacks a capability.

    Parameters
    ----------
    consumer : str
        The component that needs the capability, as a sentence subject.
    capability : str
        The capability the declared semiring does not supply.
    description : str
        What the capability provides, as the consumer's object.
    semiring : str
        The declared semiring's kind.
    path : str | None
        Dotted configuration path of the refusing component.
    """

    __slots__ = ("capability", "consumer", "description", "semiring")

    def __init__(
        self,
        *,
        consumer: str,
        capability: str,
        description: str,
        semiring: str,
        path: str | None = None,
    ) -> None:
        self.consumer = consumer
        self.capability = capability
        self.description = description
        self.semiring = semiring

        super().__init__(
            f"{consumer} needs {description}, which parse.semiring "
            f"{semiring!r} does not supply",
            path=path,
        )

    @override
    def prefixed(self, prefix: str, /) -> SemiringCapabilityError:
        """Return this refusal under one enclosing configuration path."""
        held = ensure_name("path prefix", prefix)

        return SemiringCapabilityError(
            consumer=self.consumer,
            capability=self.capability,
            description=self.description,
            semiring=self.semiring,
            path=held if self.path is None else f"{held}.{self.path}",
        )


class IdentifierError(DeclarationError):
    """An identifier that no declared name encodes to."""


class DerivationIndexError(DeclarationError):
    """A rule schema whose result index the derivation family cannot form.

    Parameters
    ----------
    rule : str
        Name of the rule whose schema is malformed.
    reason : str
        What the schema gets wrong.
    """

    __slots__ = ("reason", "rule")

    def __init__(self, *, rule: str, reason: str) -> None:
        self.rule = ensure_name("rule", rule)
        raw_reason = ensure_name("reason", reason)

        super().__init__(
            f"rule {self.rule!r} has a malformed derivation index: {raw_reason}"
        )
        self.reason = raw_reason


class InventoryExhaustedError(DeclarationError):
    """A generated category inventory the rules stopped growing before it was full.

    Parameters
    ----------
    requested : int
        How many categories the inventory was to hold.
    reached : int
        How many it held when generation stopped.
    patience : int
        How many consecutive derivations were refused before it stopped.
    """

    __slots__ = ("patience", "reached", "requested")

    def __init__(self, *, requested: int, reached: int, patience: int) -> None:
        self.requested = requested
        self.reached = reached
        self.patience = patience

        super().__init__(
            f"the rules reached {reached} of {requested} lexical categories, and "
            f"the {patience} derivations drawn after the last admitted one added "
            "none that fit the inventory and its bounds"
        )
