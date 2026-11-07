"""Content identities for published evaluation rows.

A published number is reported against a freeze, never against a moving
branch, so the freeze identifier ``freeze/<n>`` is an identity column
beside the configuration, grammar and convention digests. A row scored during
development, outside any freeze, names ``UNFROZEN`` in that column instead, so
it can never be mistaken for a row of a freeze.
"""

import re
from collections.abc import Mapping
from datetime import date, datetime, time
from decimal import Decimal
from hashlib import sha256
from json import dumps
from typing import (
    TYPE_CHECKING,
    Final,
    Protocol,
    Self,
    final,
    override,
    runtime_checkable,
)
from uuid import UUID

from lofi._checks import ensure_name
from lofi.eval._values import ConventionVector

if TYPE_CHECKING:
    from collections.abc import Sequence
    from collections.abc import Set as AbstractSet

    from lofi._types import JsonValue

__all__ = [
    "UNFROZEN",
    "ConfigSource",
    "ReportIdentity",
    "config_digest",
    "freeze_number",
]

_FREEZE_ID: Final = re.compile(r"freeze/([1-9][0-9]*)")
"""The form of a freeze identifier: a positive freeze number."""

UNFROZEN: Final = "unfrozen"
"""The freeze column of a row scored outside any freeze."""

type _ConfigValue = (
    str
    | int
    | float
    | bool
    | bytes
    | Decimal
    | datetime
    | date
    | time
    | UUID
    | Sequence[_ConfigValue]
    | AbstractSet[_ConfigValue]
    | Mapping[str, _ConfigValue]
    | ConfigSource
    | _TermSource
    | None
)
"""Decoded configuration values accepted from a model dump."""


@runtime_checkable
class ConfigSource(Protocol):
    """A configuration with a JSON-shaped model dump."""

    def model_dump(self) -> Mapping[str, _ConfigValue]:
        """Return the configuration as JSON-shaped data."""
        ...


class _TermSource(Protocol):
    """A theory term with a JSON-shaped specification."""

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the term as JSON-shaped data."""
        ...


@runtime_checkable
class _GrammarSource(Protocol):
    """A compiled grammar with a stable fingerprint."""

    @property
    def fingerprint(self) -> str:
        """Return the stable grammar fingerprint."""
        ...


def _digest(value: JsonValue) -> str:
    """Return the SHA-256 digest of canonical JSON."""
    encoded = dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    return sha256(encoded.encode("utf-8")).hexdigest()


def _json_value(value: _ConfigValue) -> JsonValue:
    """Return a recursively JSON-shaped configuration value."""
    if isinstance(value, ConfigSource):
        return _json_value(value.model_dump())

    if isinstance(value, Mapping):
        converted: dict[str, JsonValue] = {}

        for key, member in value.items():
            if not isinstance(key, str):
                msg = f"a configuration key is not text: {key!r}"
                raise TypeError(msg)

            converted[key] = _json_value(member)

        return converted

    if isinstance(value, tuple | list):
        return [_json_value(member) for member in value]

    if value is None or isinstance(value, str | int | float | bool):
        return value

    msg = f"a configuration value is not JSON-shaped: {value!r}"
    raise TypeError(msg)


def config_digest(config: Mapping[str, _ConfigValue]) -> str:
    """Return the digest of a run configuration's model dump.

    Parameters
    ----------
    config : Mapping[str, _ConfigValue]
        The configuration's JSON-shaped model dump, in any key order.

    Returns
    -------
    str
        The SHA-256 digest of the canonical JSON, in hexadecimal.

    Raises
    ------
    TypeError
        If a key is not text or a value is not JSON-shaped.
    ValueError
        If a number is not JSON compliant.
    """
    if not isinstance(config, Mapping):
        msg = f"a run specification's model dump is not a mapping: {config!r}"
        raise TypeError(msg)

    return _digest(_json_value(config))


def freeze_number(freeze_id: str) -> int:
    """Return the number a freeze identifier names.

    Parameters
    ----------
    freeze_id : str
        An identifier of the form ``freeze/<n>``, where ``n`` is a positive
        integer written without leading zeros.

    Returns
    -------
    int
        The freeze number.

    Raises
    ------
    TypeError
        If the identifier is not text.
    ValueError
        If the identifier is not of the form ``freeze/<n>``.
    """
    matched = _FREEZE_ID.fullmatch(ensure_name("a freeze identifier", freeze_id))

    if matched is None:
        msg = (
            f"a freeze identifier is not of the form 'freeze/<n>' with n a "
            f"positive integer: {freeze_id!r}"
        )
        raise ValueError(msg)

    return int(matched.group(1))


@final
class ReportIdentity:
    """A freeze, run configuration, grammar, and parsing-convention identity.

    Parameters
    ----------
    run_spec : ConfigSource
        The run specification.
    grammar : _GrammarSource
        The grammar the predictions were produced under.
    conventions : ConventionVector | None
        The parsing conventions the scores depend on, or ``None`` for rows,
        such as judgment model scores, that no parsing convention affects.
    freeze_id : str
        The freeze the row is reported against, ``freeze/<n>``, or
        ``UNFROZEN`` for a row scored outside any freeze.

    Raises
    ------
    TypeError
        If an input has an invalid type.
    ValueError
        If the grammar fingerprint is empty, the freeze identifier is neither
        a freeze nor ``UNFROZEN``, or the configuration holds a non-compliant
        number.
    """

    __slots__ = (
        "_config_hash",
        "_convention_hash",
        "_freeze_id",
        "_grammar_fingerprint",
    )

    def __init__(
        self,
        run_spec: ConfigSource,
        grammar: _GrammarSource,
        conventions: ConventionVector | None,
        *,
        freeze_id: str,
    ) -> None:
        if not isinstance(run_spec, ConfigSource):
            msg = f"a report identity's run specification is not one: {run_spec!r}"
            raise TypeError(msg)

        if not isinstance(grammar, _GrammarSource):
            msg = f"a report identity's grammar is not one: {grammar!r}"
            raise TypeError(msg)

        if conventions is not None and not isinstance(conventions, ConventionVector):
            msg = f"a report identity's conventions are not a vector: {conventions!r}"
            raise TypeError(msg)

        if freeze_id != UNFROZEN:
            freeze_number(freeze_id)

        object.__setattr__(self, "_freeze_id", freeze_id)
        object.__setattr__(self, "_config_hash", config_digest(run_spec.model_dump()))
        object.__setattr__(
            self,
            "_grammar_fingerprint",
            ensure_name("a grammar fingerprint", grammar.fingerprint),
        )
        object.__setattr__(
            self,
            "_convention_hash",
            None if conventions is None else _digest(dict(conventions.factors)),
        )

    @property
    def freeze_id(self) -> str:
        """Return the freeze the row is reported against, or ``UNFROZEN``."""
        return self._freeze_id

    @property
    def config_hash(self) -> str:
        """Return the run-configuration digest."""
        return self._config_hash

    @property
    def grammar_fingerprint(self) -> str:
        """Return the grammar fingerprint."""
        return self._grammar_fingerprint

    @property
    def convention_hash(self) -> str | None:
        """Return the parsing-convention digest, absent for judgment analyses."""
        return self._convention_hash

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the four report identity columns."""
        return {
            "freeze_id": self._freeze_id,
            "config_hash": self._config_hash,
            "grammar_fingerprint": self._grammar_fingerprint,
            "convention_hash": self._convention_hash,
        }

    def __copy__(self) -> Self:
        """Return this immutable value."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable value."""
        del memo

        return self

    @override
    def __setattr__[V](self, name: str, value: V) -> None:
        """Refuse field reassignment."""
        del name, value
        msg = "a report identity is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> None:
        """Refuse field deletion."""
        del name
        msg = "a report identity is immutable"
        raise AttributeError(msg)

    def _identity(self) -> tuple[str, str, str, str | None]:
        """Return the four identity columns."""
        return (
            self._freeze_id,
            self._config_hash,
            self._grammar_fingerprint,
            self._convention_hash,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Compare the four identity columns."""
        if type(other) is not ReportIdentity:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Hash the four identity columns."""
        return hash((ReportIdentity, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return a representation of the identity columns."""
        return (
            f"ReportIdentity(freeze_id={self._freeze_id!r}, "
            f"config_hash={self._config_hash!r}, "
            f"grammar_fingerprint={self._grammar_fingerprint!r}, "
            f"convention_hash={self._convention_hash!r})"
        )
