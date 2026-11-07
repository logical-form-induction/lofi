"""Expert ratings loaded through the artifact-dataset boundary."""

from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, final, override

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.data.datasets import ArtifactDatasetPlugin
from lofi.spec.datasets.expert import (
    ExpertRatingsDatasetSpec,
    decode_expert_ratings,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["PLUGIN", "ExpertRating", "ExpertRatings"]


class ExpertRating(NamedTuple):
    """One expert's rating and error labels for one induced category."""

    category_id: str
    annotator: str
    rating: str
    errors: tuple[str, ...]


@final
class ExpertRatings(FrozenValue):
    """Validated expert responses bound to one annotation-package digest."""

    _noun = "expert ratings"
    __slots__ = ("_package_digest", "_ratings")

    def __init__(self, package_digest: str, ratings: Iterable[ExpertRating], /) -> None:
        digest = ensure_name("an annotation-package digest", package_digest)

        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            msg = f"an annotation-package digest is not lowercase SHA-256: {digest!r}"
            raise ValueError(msg)

        checked: list[ExpertRating] = []
        seen: set[tuple[str, str]] = set()

        for position, response in enumerate(ratings):
            if not isinstance(response, ExpertRating):
                msg = f"expert response {position} is not an ExpertRating: {response!r}"
                raise TypeError(msg)

            category = ensure_name("an expert-rated category", response.category_id)
            annotator = ensure_name("an expert annotator", response.annotator)
            rating = ensure_name("an expert rating", response.rating)
            errors = tuple(
                ensure_name(f"expert error label {index}", error)
                for index, error in enumerate(response.errors)
            )

            if len(set(errors)) != len(errors):
                msg = (
                    f"expert response {(category, annotator)!r} repeats an error label"
                )
                raise ValueError(msg)

            key = category, annotator

            if key in seen:
                msg = f"expert {annotator!r} rates category {category!r} twice"
                raise ValueError(msg)

            seen.add(key)
            checked.append(ExpertRating(category, annotator, rating, errors))

        if not checked:
            msg = "an expert-ratings dataset contains no response"
            raise ValueError(msg)

        object.__setattr__(self, "_package_digest", digest)
        object.__setattr__(
            self,
            "_ratings",
            tuple(sorted(checked, key=lambda row: (row.category_id, row.annotator))),
        )

    @property
    def package_digest(self) -> str:
        """Return the annotation package these responses judge."""
        return self._package_digest

    @property
    def ratings(self) -> tuple[ExpertRating, ...]:
        """Return responses ordered by category and annotator."""
        return self._ratings

    @override
    def __repr__(self) -> str:
        """Return the response count and bound package digest."""
        return (
            f"ExpertRatings({self._package_digest!r}, {len(self._ratings)} responses)"
        )


def _load(declaration: ExpertRatingsDatasetSpec, /) -> ExpertRatings:
    """Validate and lower one expert-ratings JSON document."""
    document = decode_expert_ratings(
        Path(declaration.source).read_text(encoding="utf-8")
    )
    return ExpertRatings(
        document.package_digest,
        (
            ExpertRating(row.category_id, row.annotator, row.rating, row.errors)
            for row in document.ratings
        ),
    )


PLUGIN: Final = ArtifactDatasetPlugin(
    spec=ExpertRatingsDatasetSpec,
    artifact=ExpertRatings,
    load=_load,
)
