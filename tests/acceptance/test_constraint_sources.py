"""Declared constraint sources train from documents and persist their budgets."""

from typing import TYPE_CHECKING

import pytest
from tests.acceptance.driver import leaf, train

from lofi.run import read_metrics

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


def declared_pin_kinds(value: JsonValue, /) -> list[str]:
    """Return the pin variant names from a resolved source declaration."""
    if not isinstance(value, list):
        msg = f"resolved constraint pins are not a list: {value!r}"
        raise TypeError(msg)

    kinds: list[str] = []

    for pin in value:
        if not isinstance(pin, dict):
            msg = f"a resolved constraint pin has no kind: {pin!r}"
            raise TypeError(msg)

        kind = pin.get("kind")

        if not isinstance(kind, str):
            msg = f"a resolved constraint pin kind is not text: {kind!r}"
            raise TypeError(msg)

        kinds.append(kind)

    return kinds


@pytest.mark.parametrize(
    ("document", "pin_kinds", "direction", "mass_share"),
    [
        (
            "declared-pins.yaml",
            ["item-feature", "span-pattern", "sentence-top"],
            "positive",
            1.0,
        ),
        ("declared-inverted.yaml", ["item-feature"], "inverted", 1.0),
        (
            "declared-quantile.yaml",
            ["item-feature", "span-pattern", "sentence-top"],
            "positive",
            0.5,
        ),
    ],
    ids=["pins", "inverted", "score-quantile"],
)
def test_each_declared_constraint_document_trains_and_persists_its_budget(
    document: str,
    pin_kinds: list[str],
    direction: str,
    mass_share: float,
    tmp_path: Path,
) -> None:
    point = train(document, tmp_path / document.removesuffix(".yaml"))
    metrics = read_metrics(point.directory)

    assert metrics is not None
    assert metrics.selection is not None
    assert metrics.selection.types_pinned > 0
    assert metrics.selection.tokens_covered > 0
    assert metrics.selection.mass_share == pytest.approx(mass_share)
    assert metrics.selection.clamp.negate is (direction == "inverted")
    assert (
        leaf(point.resolved, "data.datasets.train.constraint_source.kind") == "declared"
    )
    assert (
        declared_pin_kinds(
            leaf(point.resolved, "data.datasets.train.constraint_source.pins")
        )
        == pin_kinds
    )
