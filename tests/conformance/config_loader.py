import json
from typing import TYPE_CHECKING, ClassVar, Final, NamedTuple, NoReturn

import didactic.api as dx
import pytest

from lofi._checks import LawViolationError
from lofi.spec import ConfigError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.spec import ConfigSource, Provenance, SpecLoader

__all__ = [
    "CONFIG_LOADER_LAWS",
    "ConfigLoaderConformance",
    "ConfigLoaderFixture",
    "Layer",
    "Request",
    "an_unknown_setting_is_refused_naming_its_path",
    "composition_is_deterministic",
    "leaves",
    "precedence_follows_the_declared_order",
    "provenance_covers_exactly_the_leaves",
    "recomposing_a_resolved_document_changes_nothing",
]


class Request(NamedTuple):
    path: Path
    profile: str | None = None
    overrides: tuple[str, ...] = ()
    extra: Mapping[str, JsonValue] | None = None


class Layer(NamedTuple):
    source: ConfigSource
    request: Request
    value: JsonValue


class ConfigLoaderFixture[M: dx.Model](NamedTuple):
    loader: SpecLoader
    schema: type[M]
    leaf: str
    read: Callable[[M], JsonValue]
    ladder: tuple[Layer, ...]
    resolved: Request
    reseat: Callable[[M], Request]
    unknown: Request
    unknown_leaf: str


_RUNG: Final = 2


def _fail(detail: str, *, law: str) -> NoReturn:
    message = f"{law}: {detail}"

    raise LawViolationError(message)


def _compose[M: dx.Model](
    fixture: ConfigLoaderFixture[M], request: Request, *, law: str
) -> M:
    try:
        return fixture.loader.compose_traced(
            request.path,
            schema=fixture.schema,
            profile=request.profile,
            overrides=request.overrides,
            extra=request.extra,
        ).spec

    except Exception as refusal:
        message = f"{law}: composing {request!r} raised {refusal!r}"

        raise LawViolationError(message) from refusal


def _trace[M: dx.Model](
    fixture: ConfigLoaderFixture[M], request: Request, *, law: str
) -> Provenance:
    try:
        return fixture.loader.compose_traced(
            request.path,
            schema=fixture.schema,
            profile=request.profile,
            overrides=request.overrides,
            extra=request.extra,
        ).provenance

    except Exception as refusal:
        message = f"{law}: tracing {request!r} raised {refusal!r}"

        raise LawViolationError(message) from refusal


def leaves(value: JsonValue, prefix: str = "") -> frozenset[str]:
    if isinstance(value, list) and value:
        return frozenset(f"{prefix}[{index}]" for index, _ in enumerate(value))

    if not isinstance(value, dict) or not value:
        return frozenset({prefix} if prefix else set())

    return frozenset(
        path
        for key, held in value.items()
        for path in leaves(held, f"{prefix}.{key}" if prefix else key)
    )


def precedence_follows_the_declared_order[M: dx.Model](
    fixture: ConfigLoaderFixture[M],
) -> None:
    law = "precedence_follows_the_declared_order"

    if len(fixture.ladder) < _RUNG:
        _fail(
            f"the ladder has {len(fixture.ladder)} rungs, which orders nothing",
            law=law,
        )

    for lower, higher in zip(fixture.ladder, fixture.ladder[1:], strict=False):
        if not higher.source.outranks(lower.source):
            _fail(
                f"{higher.source} is placed above {lower.source} and does not "
                "outrank it",
                law=law,
            )

    for rung in fixture.ladder:
        document = _compose(fixture, rung.request, law=law)
        reading = fixture.read(document)

        if reading != rung.value:
            _fail(
                f"with {rung.source} on top, {fixture.leaf} reads as "
                f"{reading!r} rather than {rung.value!r}",
                law=law,
            )

        provenance = _trace(fixture, rung.request, law=law)
        named = _source_of(provenance, fixture.leaf, law=law)

        if named != rung.source:
            _fail(
                f"with {rung.source} on top, the provenance names {named} as "
                f"the source of {fixture.leaf}",
                law=law,
            )


def _source_of(provenance: Provenance, leaf: str, *, law: str) -> ConfigSource:
    try:
        return provenance.source_of(leaf)

    except ConfigError as refusal:
        message = f"{law}: the provenance names no source for {leaf}"

        raise LawViolationError(message) from refusal


def an_unknown_setting_is_refused_naming_its_path[M: dx.Model](
    fixture: ConfigLoaderFixture[M],
) -> None:
    law = "an_unknown_setting_is_refused_naming_its_path"

    try:
        document = fixture.loader.compose_traced(
            fixture.unknown.path,
            schema=fixture.schema,
            profile=fixture.unknown.profile,
            overrides=fixture.unknown.overrides,
            extra=fixture.unknown.extra,
        ).spec

    except ConfigError as refusal:
        if refusal.path != fixture.unknown_leaf:
            _fail(
                f"the request named {fixture.unknown_leaf} and the refusal "
                f"records {refusal.path}",
                law=law,
            )

        if fixture.unknown_leaf not in str(refusal):
            _fail(f"the message leaves out {fixture.unknown_leaf}", law=law)

        return

    except Exception as refusal:
        message = (
            f"{law}: setting {fixture.unknown_leaf} raised "
            f"{type(refusal).__name__}, which records no dotted path"
        )

        raise LawViolationError(message) from refusal

    _fail(
        f"setting {fixture.unknown_leaf} composed to {document!r} rather than "
        "being refused",
        law=law,
    )


def composition_is_deterministic[M: dx.Model](fixture: ConfigLoaderFixture[M]) -> None:
    law = "composition_is_deterministic"

    first = _compose(fixture, fixture.resolved, law=law)
    second = _compose(fixture, fixture.resolved, law=law)

    if first != second:
        _fail(
            f"the same request composed to {first!r} and then to {second!r}",
            law=law,
        )

    if _trace(fixture, fixture.resolved, law=law) != _trace(
        fixture, fixture.resolved, law=law
    ):
        _fail("the same request traced to two different provenances", law=law)


def recomposing_a_resolved_document_changes_nothing[M: dx.Model](
    fixture: ConfigLoaderFixture[M],
) -> None:
    law = "recomposing_a_resolved_document_changes_nothing"

    once = _compose(fixture, fixture.resolved, law=law)
    twice = _compose(fixture, fixture.reseat(once), law=law)

    if once != twice:
        _fail(
            f"composing {once!r} a second time gave {twice!r}",
            law=law,
        )


def provenance_covers_exactly_the_leaves[M: dx.Model](
    fixture: ConfigLoaderFixture[M],
) -> None:
    law = "provenance_covers_exactly_the_leaves"

    document = _compose(fixture, fixture.resolved, law=law)
    provenance = _trace(fixture, fixture.resolved, law=law)

    held = leaves(json.loads(document.model_dump_json()))
    named = frozenset(provenance.paths)

    if held - named:
        _fail(
            f"the document holds {sorted(held - named)}, which the provenance "
            "does not name",
            law=law,
        )

    if named - held:
        _fail(
            f"the provenance names {sorted(named - held)}, which the document "
            "does not hold",
            law=law,
        )


CONFIG_LOADER_LAWS: Final = (
    precedence_follows_the_declared_order,
    an_unknown_setting_is_refused_naming_its_path,
    composition_is_deterministic,
    recomposing_a_resolved_document_changes_nothing,
    provenance_covers_exactly_the_leaves,
)


class ConfigLoaderConformance:
    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    def test_precedence_follows_the_declared_order[M: dx.Model](
        self, loader_fixture: ConfigLoaderFixture[M]
    ) -> None:
        precedence_follows_the_declared_order(loader_fixture)

    def test_an_unknown_setting_is_refused_naming_its_path[M: dx.Model](
        self, loader_fixture: ConfigLoaderFixture[M]
    ) -> None:
        an_unknown_setting_is_refused_naming_its_path(loader_fixture)

    def test_composition_is_deterministic[M: dx.Model](
        self, loader_fixture: ConfigLoaderFixture[M]
    ) -> None:
        composition_is_deterministic(loader_fixture)

    def test_recomposing_a_resolved_document_changes_nothing[M: dx.Model](
        self, loader_fixture: ConfigLoaderFixture[M]
    ) -> None:
        recomposing_a_resolved_document_changes_nothing(loader_fixture)

    def test_provenance_covers_exactly_the_leaves[M: dx.Model](
        self, loader_fixture: ConfigLoaderFixture[M]
    ) -> None:
        provenance_covers_exactly_the_leaves(loader_fixture)
