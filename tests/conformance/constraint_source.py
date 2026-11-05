"""Reusable laws for constraint sources."""

from typing import TYPE_CHECKING, ClassVar, Final

import pytest

from lofi.data.laws import CONSTRAINT_SOURCE_LAWS

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from lofi.data.laws import ConstraintSourceFixture

__all__ = ["ConstraintSourceConformance"]

_LAW_NAMES: Final = tuple(law.__name__ for law in CONSTRAINT_SOURCE_LAWS)


class ConstraintSourceConformance:
    """Reusable constraint-source conformance laws.

    A subclass supplies one ``constraint_source_fixture`` fixture: the source at
    full coverage, the same evidence under a form-level rule, a builder for half
    coverage under a seed, the corpus, and what the evidence names.
    """

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    @pytest.mark.parametrize("law", CONSTRAINT_SOURCE_LAWS, ids=_LAW_NAMES)
    def test_the_source_holds_to_each_law[C: Hashable, L: Hashable](
        self,
        law: Callable[[ConstraintSourceFixture[C, L]], None],
        constraint_source_fixture: ConstraintSourceFixture[C, L],
    ) -> None:
        """Check one law against the subject.

        Parameters
        ----------
        law
            The law to run.
        constraint_source_fixture
            What the laws run against.
        """
        law(constraint_source_fixture)
