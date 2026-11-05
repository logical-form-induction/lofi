import hypothesis.strategies as st
import pytest

import lofi
from lofi.deduce import Counting
from lofi.deduce.laws import SemiringFixture, check_semiring_laws


def test_a_violated_law_is_caught_by_the_published_error() -> None:
    broken = SemiringFixture(
        semiring=Counting(),
        values=st.integers(min_value=0, max_value=1000),
        equal=lambda left, right: left == right + 1,
        max_examples=10,
    )

    with pytest.raises(lofi.LawViolationError, match="not commutative"):
        check_semiring_laws(broken)


def test_the_package_publishes_only_the_law_violation_error() -> None:
    assert lofi.__all__ == ["LawViolationError"]
