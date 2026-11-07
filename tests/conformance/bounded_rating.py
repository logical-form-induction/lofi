"""Reusable laws for scalar supervision consumed by bounded judgments."""

from typing import TYPE_CHECKING, ClassVar, NamedTuple, NoReturn

import pytest

from lofi._checks import LawViolationError

if TYPE_CHECKING:
    from lofi.judgment._ports import BoundedRating

__all__ = [
    "BoundedRatingSubject",
    "BoundedRatingSuite",
    "rating_exposes_one_scalar_and_its_related_items",
]


class BoundedRatingSubject(NamedTuple):
    """One bounded rating and its expected public values."""

    rating: BoundedRating
    value: float | None
    kind: str
    minimum: float
    maximum: float
    annotator: str | None
    related_items: tuple[str, ...]


def _fail(detail: str) -> NoReturn:
    """Raise a bounded-rating law violation."""
    msg = f"rating_exposes_one_scalar_and_its_related_items: {detail}"
    raise LawViolationError(msg)


def rating_exposes_one_scalar_and_its_related_items(
    subject: BoundedRatingSubject, /
) -> None:
    """Check the sole scalar-response surface and its typed item references."""
    rating = subject.rating
    observed = (
        rating.value,
        rating.scale.kind,
        rating.scale.minimum,
        rating.scale.maximum,
        rating.annotator,
        rating.related_items,
    )
    expected = (
        subject.value,
        subject.kind,
        subject.minimum,
        subject.maximum,
        subject.annotator,
        subject.related_items,
    )

    if observed != expected:
        _fail(f"found {observed!r}, expected {expected!r}")


class BoundedRatingSuite:
    """Reusable conformance law for a bounded scalar supervision value."""

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    def test_rating_exposes_one_scalar_and_its_related_items(
        self, bounded_rating_subject: BoundedRatingSubject
    ) -> None:
        """Run the scalar and related-item law."""
        rating_exposes_one_scalar_and_its_related_items(bounded_rating_subject)
