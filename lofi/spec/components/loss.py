"""The loss section of a run and the component families its objectives declare."""

from typing import TYPE_CHECKING

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name, ensure_positive
from lofi.spec.components.base import ComponentSpec

if TYPE_CHECKING:
    from lofi.spec.components.data import TrainingBatchKind
    from lofi.spec.components.model import ModelSection

__all__ = [
    "LossSection",
    "LossSpec",
    "WeightScheduleSpec",
    "WeightedLossSpec",
]


class LossSpec(ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares a loss."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry point group lists the plugins."""
        return "loss"

    def grounding_signal(self) -> str | None:
        """Return the grounding signal this objective consumes, if any."""
        return None

    def grounding_query(self) -> str | None:
        """Return which proof questions grounding asks, if this is grounded."""
        return None

    def required_training_batches(self) -> frozenset[TrainingBatchKind]:
        """Return additional data capabilities this objective requires."""
        return frozenset()

    def refuse_model(self, model: ModelSection, path: str, /) -> None:
        """Refuse an objective naming a model component the model does not declare.

        Parameters
        ----------
        model : ModelSection
            The run's model section.
        path : str
            The declaration's location in the run document, which a refusal
            names.

        Raises
        ------
        ValueError
            If the objective names a space, span decoder or published mass the
            model does not declare.
        """
        del model, path


class WeightScheduleSpec(dx.Model, extra="forbid"):
    """Declares a linear ramp of a term's weight over the first optimizer steps.

    Parameters
    ----------
    start : float
        Finite, nonnegative weight the term takes before the first optimizer step.
    steps : int
        Positive number of optimizer steps over which the weight moves linearly
        from ``start`` to the term's declared weight, which it keeps afterwards.
    """

    start: float
    steps: int

    @dx.validates("start")
    @classmethod
    def _start_is_nonnegative(cls, start: float) -> float:
        """Validate the starting weight."""
        checked = ensure_finite("a schedule's starting weight", start)

        if checked < 0.0:
            msg = f"a schedule's starting weight is negative: {checked!r}"
            raise ValueError(msg)

        return checked

    @dx.validates("steps")
    @classmethod
    def _steps_are_positive(cls, steps: int) -> int:
        """Validate the ramp's optimizer-step count."""
        return ensure_positive("a schedule's steps", steps)


class WeightedLossSpec(dx.Model, extra="forbid"):
    """Declares one weighted term of the training objective.

    A term is built, reported and logged under its name. A term that declares no
    name takes its objective's registry tag, so two terms of one kind must name
    themselves apart.

    Parameters
    ----------
    objective : LossSpec
        Loss component.
    weight : float
        Finite, nonnegative multiplier applied to the term's value.
    name : str | None
        The nonempty name the term is reported under, without a slash, or None to
        use the objective's registry tag.
    schedule : WeightScheduleSpec | None
        How the multiplier ramps to ``weight`` over the first optimizer steps, or
        None for a multiplier that is ``weight`` throughout.
    """

    objective: LossSpec
    weight: float = 1.0
    schedule: WeightScheduleSpec | None = None
    name: str | None = None

    @dx.validates("name")
    @classmethod
    def _name_is_a_key_segment(cls, name: str | None) -> str | None:
        """Validate a term name.

        Parameters
        ----------
        name : str | None
            Name to validate.

        Returns
        -------
        str | None
            The name, or None.

        Raises
        ------
        TypeError
            If the name is not a string.
        ValueError
            If the name is empty or contains a slash, which separates the segments
            of a telemetry key.
        """
        if name is None:
            return None

        checked = ensure_name("a loss term name", name)

        if "/" in checked:
            msg = f"a loss term name contains no slash: {checked!r}"
            raise ValueError(msg)

        return checked

    @dx.validates("weight")
    @classmethod
    def _weight_is_nonnegative(cls, weight: float) -> float:
        """Validate an objective weight.

        Parameters
        ----------
        weight : float
            Weight to validate.

        Returns
        -------
        float
            Finite, nonnegative weight.

        Raises
        ------
        TypeError
            If the weight is not numeric.
        ValueError
            If the weight is negative, infinite, or NaN.
        """
        checked = ensure_finite("an objective weight", weight)

        if checked < 0.0:
            msg = f"an objective weight is negative: {checked!r}"
            raise ValueError(msg)

        return checked

    @property
    def tag(self) -> str:
        """Return the registry tag the objective variant was declared under."""
        return str(self.objective.model_dump()["kind"])

    @property
    def key(self) -> str:
        """Return the name the term is built, reported and logged under."""
        return self.tag if self.name is None else self.name


class LossSection(dx.Model, extra="forbid"):
    """Declares the training objective as weighted terms.

    Parameters
    ----------
    terms : tuple[WeightedLossSpec, ...]
        Weighted loss terms, in the order they are reported, at least one and no two
        under one name.
    """

    terms: tuple[WeightedLossSpec, ...]

    def grounding_query(self) -> str | None:
        """Return the sole proof-question policy declared by grounded terms."""
        queries = frozenset(
            query
            for term in self.terms
            if (query := term.objective.grounding_query()) is not None
        )

        if len(queries) > 1:
            msg = (
                "grounded objectives declare incompatible proof questions "
                f"{sorted(queries)!r}"
            )
            raise ValueError(msg)

        return None if not queries else next(iter(queries))

    @dx.validates("terms")
    @classmethod
    def _terms_are_present_and_distinct(
        cls, terms: tuple[WeightedLossSpec, ...]
    ) -> tuple[WeightedLossSpec, ...]:
        """Validate the weighted terms.

        Parameters
        ----------
        terms : tuple[WeightedLossSpec, ...]
            Terms to validate.

        Returns
        -------
        tuple[WeightedLossSpec, ...]
            The terms, in the order given.

        Raises
        ------
        ValueError
            If there is no term, or if two terms are reported under one name.
        """
        if not terms:
            msg = "a loss section declares at least one term"
            raise ValueError(msg)

        seen: set[str] = set()

        for term in terms:
            if term.key in seen:
                msg = (
                    f"two loss terms are named {term.key!r}; give each term of one "
                    "kind its own name"
                )
                raise ValueError(msg)

            seen.add(term.key)

        return terms
