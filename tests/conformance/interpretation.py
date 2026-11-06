"""Conformance tests for derivation interpretations."""

from typing import TYPE_CHECKING

import pytest

from lofi.interpret.laws import INTERPRETATION_LAWS

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.interpret.laws import InterpretationFixture

__all__ = ["InterpretationSuite"]


class InterpretationSuite:
    """Conformance laws for derivation interpretations."""

    pytestmark = pytest.mark.conformance

    @pytest.mark.parametrize("law", INTERPRETATION_LAWS)
    def test_interpretation_law[Category, Value, Meaning, Weight](
        self,
        interpretation_conformance: InterpretationFixture[
            Category,
            Value,
            Meaning,
            Weight,
        ],
        law: Callable[
            [InterpretationFixture[Category, Value, Meaning, Weight]],
            None,
        ],
    ) -> None:
        law(interpretation_conformance)
