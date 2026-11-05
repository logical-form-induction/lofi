"""Tests for open synthetic corruption operator declarations."""

from __future__ import annotations

from importlib.metadata import EntryPoint
from math import sqrt
from random import Random
from typing import TYPE_CHECKING, Final, Literal, override

from tests.support.installed import install
from tests.unit.data.synthetic.test__corrupt import _assembly

from lofi.data.synthetic._corrupt import CorruptionContext, CorruptionOperator
from lofi.data.synthetic._values import Corruption, WeightedOperator
from lofi.data.synthetic.spec.operators import (
    CorruptionOperatorFactory,
    CorruptionOperatorPlugin,
    CorruptionOperatorSpec,
    WeightedCorruptionOperatorSpec,
)

if TYPE_CHECKING:
    import pytest

TAG: Final = "test-rotation"
GROUP: Final = "lofi.corruption"


class RotationSpec(CorruptionOperatorSpec, extra="forbid"):
    """A test-package operator with one declared variant field."""

    kind: Literal["test-rotation"] = "test-rotation"
    distance: int


class Rotation(CorruptionOperator):
    """Rotate forms by the declared distance."""

    __slots__ = ("_distance",)

    def __init__(self, distance: int) -> None:
        self._distance = distance

    @property
    @override
    def name(self) -> str:
        """Return the registered operator name."""
        return TAG

    @override
    def apply(self, random: Random, forms: tuple[str, ...]) -> tuple[str, ...] | None:
        """Rotate a nonempty sequence by the declared distance."""
        del random
        if not forms:
            return None
        at = self._distance % len(forms)
        return (*forms[at:], *forms[:at])


class RotationFactory(CorruptionOperatorFactory):
    """Build one rotation after its declaration has left the boundary."""

    __slots__ = ("_distance",)

    def __init__(self, distance: int) -> None:
        self._distance = distance

    @override
    def __call__(self, context: CorruptionContext) -> CorruptionOperator:
        del context
        return Rotation(self._distance)


def _compile(declaration: RotationSpec) -> CorruptionOperatorFactory:
    return RotationFactory(declaration.distance)


PLUGIN: Final = CorruptionOperatorPlugin(spec=RotationSpec, compiler=_compile)


class OtherOperator:
    """A second weighted operator used to identify draw frequency."""

    @property
    def name(self) -> str:
        """Return the test operator's distinct name."""
        return "test-other"

    def apply(self, random: Random, forms: tuple[str, ...]) -> tuple[str, ...] | None:
        """Return the input unchanged for this draw-only fixture."""
        del random
        return forms or None


def test_a_test_package_operator_is_built_through_its_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = EntryPoint(name=TAG, value=f"{__name__}:PLUGIN", group=GROUP)
    install(monkeypatch, GROUP.removeprefix("lofi."), (point,))
    assembly = _assembly()
    context = CorruptionContext(
        assembly.lexicon, assembly.grammar.matching, max_length=4
    )
    weighted = WeightedCorruptionOperatorSpec(
        operator=RotationSpec(distance=2), weight=3.0
    ).build(context)

    assert weighted.name == TAG
    assert weighted.weight == 3.0
    assert weighted.operator.apply(Random(0), ("a", "b", "c")) == ("c", "a", "b")

    request = Corruption(
        0.5,
        operators=(weighted, WeightedOperator(OtherOperator(), 1.0)),
        attempts=4,
        draw="bernoulli",
    )
    random = Random(17)
    draws = 4_000
    selected = sum(request.choose(random).name == TAG for _ in range(draws))
    expected = 0.75

    assert abs(selected / draws - expected) <= 4 * sqrt(
        expected * (1 - expected) / draws
    )


def test_a_validated_operator_compiles_to_a_context_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = EntryPoint(name=TAG, value=f"{__name__}:PLUGIN", group=GROUP)
    install(monkeypatch, GROUP.removeprefix("lofi."), (point,))
    declaration = RotationSpec(distance=2)
    compiled = WeightedCorruptionOperatorSpec(
        operator=declaration, weight=3.0
    ).compile()
    assembly = _assembly()
    context = CorruptionContext(
        assembly.lexicon, assembly.grammar.matching, max_length=4
    )

    weighted = compiled.build(context)

    assert weighted.name == TAG
    assert weighted.weight == 3.0
    assert weighted.operator.apply(Random(0), ("a", "b", "c")) == ("c", "a", "b")
