"""Expert-rating artifact dataset declarations."""

from typing import Literal

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec.components.data import DatasetSpec

__all__ = [
    "ExpertRatingDocument",
    "ExpertRatingsDatasetSpec",
    "ExpertRatingsDocument",
    "decode_expert_ratings",
]


class ExpertRatingDocument(dx.Model, extra="forbid"):
    """Decode one expert's judgment of one induced category."""

    category_id: str
    annotator: str
    rating: str
    errors: tuple[str, ...]


class ExpertRatingsDocument(dx.Model, extra="forbid"):
    """Decode expert judgments bound to one annotation package."""

    kind: Literal["expert-ratings"]
    package_digest: str
    ratings: tuple[ExpertRatingDocument, ...]


def decode_expert_ratings(document: str, /) -> ExpertRatingsDocument:
    """Validate one expert-ratings document at its dataset boundary."""
    return ExpertRatingsDocument.model_validate_json(document)


class ExpertRatingsDatasetSpec(DatasetSpec):
    """Declare a persisted expert-ratings artifact."""

    kind: Literal["expert-ratings"] = "expert-ratings"
    source: str

    @dx.validates("source")
    @classmethod
    def _source_is_named(cls, value: str) -> str:
        """Return a nonempty ratings source."""
        return ensure_name("an expert-ratings source", value)
