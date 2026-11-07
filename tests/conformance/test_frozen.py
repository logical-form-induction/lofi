"""The frozen-value suite run against every value class in the tree."""

from tests.conformance.frozen import FrozenValueConformance


class TestEveryFrozenValue(FrozenValueConformance):
    """Every frozen value the package declares."""
