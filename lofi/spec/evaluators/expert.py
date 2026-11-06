"""Expert-category annotation and evaluator declarations."""

from typing import Literal, NamedTuple, cast, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec._ports import Dependency
from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.datasets.expert import ExpertRatingsDatasetSpec
from lofi.spec.evaluators.alignment import (
    AlignmentDeclaration,
    AlignmentSettings,
    AlignmentSpec,
)
from lofi.spec.evaluators.conventions import ConventionSettings, ConventionSpec

__all__ = [
    "ExpertCategoryEvaluatorSettings",
    "ExpertCategoryEvaluatorSpec",
]


class ExpertCategoryEvaluatorSettings(NamedTuple):
    """Resolved settings for expert evaluation of induced categories."""

    readout: str
    ratings: ExpertRatingsDatasetSpec | None
    taxonomy: tuple[str, ...]
    conventions: ConventionSettings
    alignment: AlignmentSettings


class ExpertCategoryEvaluatorSpec(EvaluatorSpec):
    """Declare expert agreement and a fixed error taxonomy over induced types."""

    kind: Literal["expert-categories"] = "expert-categories"
    readout: str
    ratings: ExpertRatingsDatasetSpec | None = None
    taxonomy: tuple[str, ...]
    conventions: ConventionSpec
    alignment: AlignmentSpec

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        """Return a named term-readout role."""
        return ensure_name("an expert evaluator's term-readout role", value)

    @dx.validates("taxonomy")
    @classmethod
    def _taxonomy_is_named_and_distinct(
        cls, values: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Return a nonempty, distinct error taxonomy."""
        held = tuple(
            ensure_name(f"expert error label {position}", value)
            for position, value in enumerate(values)
        )

        if not held:
            msg = "an expert evaluator declares no error taxonomy"
            raise ValueError(msg)

        if len(set(held)) != len(held):
            msg = f"an expert evaluator repeats an error label: {held!r}"
            raise ValueError(msg)

        return held

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the declared term-readout dependency."""
        return (Dependency(self.readout, "terms-readout"),)

    @override
    def metrics(self, labels: tuple[str, ...] = (), /) -> tuple[str, ...]:
        """Return agreement followed by one row per declared error label."""
        del labels
        return (
            "expert-rating-alpha",
            *(f"expert-error-{label}" for label in self.taxonomy),
        )

    def settings(self) -> ExpertCategoryEvaluatorSettings:
        """Return immutable settings consumed by the evaluator plugin."""
        return ExpertCategoryEvaluatorSettings(
            readout=self.readout,
            ratings=self.ratings,
            taxonomy=self.taxonomy,
            conventions=self.conventions.settings(),
            alignment=cast("AlignmentDeclaration", self.alignment).settings(),
        )
