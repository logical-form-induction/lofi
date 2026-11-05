"""The forced-choice response family over the latents of the alternatives.

A row of a forced-choice design shows several alternatives together, and the
linear predictor scores every one of them: the predictor runs over the
alternatives of a row, adjacent and in presentation order, and the family
reads it back as one row of scores per response row.

Without ties the alternatives are compared by a multinomial logit over those
scores, and the recorded level is the chosen alternative's place in
presentation order. With ties the row shows two alternatives, the family
takes the difference of their scores, and the cumulative family observes it
over three levels whose two cutpoints hold the tie between them.
"""

from typing import TYPE_CHECKING, final, override

from lofi.judgment._program import (
    Apply,
    Binary,
    Bind,
    Draw,
    Function,
    Gather,
    Literal,
    Name,
    ObserveChoice,
    Operator,
    Prior,
    Table,
    ensure_identifier,
)

if TYPE_CHECKING:
    from lofi.judgment._program import Expression, Step
    from lofi.judgment.cumulative import CumulativeFamily, DiscriminationGroup

__all__ = ["ChoiceFamily"]

_MINIMUM_ALTERNATIVES = 2
"""The minimum number of alternatives a row shows."""

_TIED_ALTERNATIVES = 2
"""The number of alternatives a tie is recorded between."""


@final
class ChoiceFamily:
    """A forced-choice response family.

    Parameters
    ----------
    alternatives : int
        Number of alternatives every row shows.
    discrimination : Prior | float | None
        Prior for a drawn slope on the scores, a positive value the slope is
        held at, or None to fix it at one.
    discrimination_group : DiscriminationGroup | None
        Grouping factor a drawn discrimination varies over, or None for one
        discrimination shared by every row.
    ordered : CumulativeFamily | None
        Family observing the difference of two scores when a tie is
        recordable, or None when the choice is between the alternatives
        alone.

    Raises
    ------
    ValueError
        If fewer than two alternatives are shown, a discrimination group is
        given without a drawn discrimination, or a tie is recorded over
        other than two alternatives and three categories.
    """

    __slots__ = (
        "_alternatives",
        "_discrimination",
        "_discrimination_group",
        "_ordered",
    )

    def __init__(
        self,
        alternatives: int,
        *,
        discrimination: Prior | float | None = None,
        discrimination_group: DiscriminationGroup | None = None,
        ordered: CumulativeFamily | None = None,
    ) -> None:
        if alternatives < _MINIMUM_ALTERNATIVES:
            msg = (
                f"a choice is made between at least {_MINIMUM_ALTERNATIVES} "
                f"alternatives: {alternatives}"
            )

            raise ValueError(msg)

        if discrimination_group is not None and not isinstance(discrimination, Prior):
            msg = (
                f"discrimination varies over {discrimination_group.index!r} and "
                f"the family does not draw it"
            )

            raise ValueError(msg)

        if ordered is not None and (
            alternatives != _TIED_ALTERNATIVES
            or ordered.categories != _TIED_ALTERNATIVES + 1
        ):
            msg = (
                f"a tie is recorded between {_TIED_ALTERNATIVES} alternatives over "
                f"{_TIED_ALTERNATIVES + 1} categories, and this family shows "
                f"{alternatives} over {ordered.categories}"
            )

            raise ValueError(msg)

        if ordered is not None and ordered.discrimination is not None:
            msg = (
                f"the choice family scales every alternative's score before it "
                f"takes their difference, and the family observing the difference "
                f"scales it again by {ordered.discrimination!r}"
            )

            raise ValueError(msg)

        self._alternatives = alternatives
        self._discrimination = discrimination
        self._discrimination_group = discrimination_group
        self._ordered = ordered

    @property
    def alternatives(self) -> int:
        """Return how many alternatives every row shows."""
        return self._alternatives

    @property
    def categories(self) -> int:
        """Return the number of recordable levels."""
        return self._alternatives if self._ordered is None else self._ordered.categories

    @property
    def discrimination(self) -> Prior | float | None:
        """Return the discrimination prior or held value, if any."""
        return self._discrimination

    @property
    def discrimination_group(self) -> DiscriminationGroup | None:
        """Return the grouping factor the discrimination varies over, if any."""
        return self._discrimination_group

    @property
    def ordered(self) -> CumulativeFamily | None:
        """Return the family observing the difference when a tie is recordable."""
        return self._ordered

    @property
    def index_object(self) -> str:
        """Return the name of the set indexing an alternative's place."""
        return "Alternative"

    @property
    def scalar_object(self) -> str:
        """Return the name of the one-element set a single draw is indexed by."""
        return "Unit"

    @property
    def cutpoint_object(self) -> str | None:
        """Return the set indexing the tie's cutpoints, if a tie is recordable."""
        return None if self._ordered is None else self._ordered.index_object

    @property
    def contrast_column(self) -> str | None:
        """Return the host column weighting each place, if a tie is recordable."""
        return None if self._ordered is None else "alternative_contrast"

    @property
    def contrast(self) -> tuple[float, ...]:
        """Return the weight of each place in the difference of the two scores.

        The difference runs from the first alternative to the last, so that
        a level rises as the last alternative is preferred.
        """
        return (-1.0, 1.0)

    @property
    def discrimination_site(self) -> str:
        """Return the discrimination site name."""
        return "choice_discrimination"

    @property
    def discrimination_rows_site(self) -> str:
        """Return the variable holding each row's discrimination, gathered by level."""
        return "choice_discrimination_rows"

    @property
    def scaled_site(self) -> str:
        """Return the site holding the scores after the discrimination scales them."""
        return "choice_scaled"

    @property
    def logit_site(self) -> str:
        """Return the site holding one row of scores per response row."""
        return "choice_logits"

    @property
    def contrast_site(self) -> str:
        """Return the site holding the weights as one row."""
        return "choice_contrast"

    @property
    def weighted_site(self) -> str:
        """Return the site holding each row's weighted scores."""
        return "choice_weighted"

    @property
    def difference_site(self) -> str:
        """Return the site holding the difference of each row's two scores."""
        return "choice_difference"

    @property
    def sites(self) -> tuple[str, ...]:
        """Return the sites the family draws, in draw order."""
        drawn = (
            (self.discrimination_site,)
            if isinstance(self._discrimination, Prior)
            else ()
        )

        return drawn + (() if self._ordered is None else self._ordered.sites)

    @property
    def bound_names(self) -> tuple[str, ...]:
        """Return variables bound by the family in binding order."""
        bound: list[str] = []

        if isinstance(self._discrimination, Prior):
            bound.append(self.discrimination_site)

        if self._discrimination_group is not None:
            bound.append(self.discrimination_rows_site)

        if self._discrimination is not None:
            bound.append(self.scaled_site)

        bound.append(self.logit_site)

        if self._ordered is None:
            return tuple(bound)

        bound += [self.contrast_site, self.weighted_site, self.difference_site]

        return (*bound, *self._ordered.bound_names)

    @property
    def host_columns(self) -> tuple[str, ...]:
        """Return the host-data columns the emitted steps read, without repeats."""
        columns: list[str] = []

        if self._discrimination_group is not None:
            columns.append(self._discrimination_group.index)

        contrast = self.contrast_column

        if contrast is not None:
            columns.append(contrast)

        if self._ordered is not None:
            columns.extend(self._ordered.host_columns)

        return tuple(dict.fromkeys(columns))

    def _discrimination_steps(self) -> tuple[Step, ...]:
        """Return the steps drawing the discrimination, if the family draws it.

        A shared discrimination is one draw over the one-element set. One
        varying over a group is drawn over the group's set and gathered by
        the group's index into one value per scored alternative.
        """
        if not isinstance(self._discrimination, Prior):
            return ()

        group = self._discrimination_group

        if group is None:
            return (
                Draw(
                    self.discrimination_site,
                    (self.scalar_object,),
                    self._discrimination,
                ),
            )

        return (
            Draw(self.discrimination_site, (group.index_object,), self._discrimination),
            Bind(
                self.discrimination_rows_site,
                Gather(self.discrimination_site, group.index),
            ),
        )

    def _scale(self) -> Expression:
        """Return the expression scaling each alternative's score.

        Raises
        ------
        ValueError
            If the family holds the discrimination at one, which scales
            nothing.
        """
        discrimination = self._discrimination

        if discrimination is None:
            msg = "a discrimination held at one scales no score"

            raise ValueError(msg)

        if not isinstance(discrimination, Prior):
            return Literal(discrimination)

        if self._discrimination_group is None:
            return Name(self.discrimination_site)

        return Name(self.discrimination_rows_site)

    def steps(self, *, predictor: str, response: str, plate: str) -> tuple[Step, ...]:
        """Return program steps that observe a choice through this family.

        Parameters
        ----------
        predictor : str
            Previously bound variable holding one score per alternative, the
            alternatives of a row adjacent and in presentation order.
        response : str
            Host column holding the chosen alternative's place.
        plate : str
            Finite set indexing response rows.

        Returns
        -------
        tuple[Step, ...]
            The steps in program order, the observation last.

        Raises
        ------
        ValueError
            If a name is invalid, names something the family binds or reads,
            or the plate is an object the family declares.
        """
        ensure_identifier(predictor, "a predictor")
        ensure_identifier(response, "a response")
        ensure_identifier(plate, "a response plate")

        taken = frozenset(self.bound_names) | frozenset(self.host_columns)

        for name, role in ((predictor, "a predictor"), (response, "a response")):
            if name in taken:
                msg = f"{role} must not be a name this family binds: {name!r}"

                raise ValueError(msg)

        if predictor == response:
            msg = f"a predictor and a response must differ: {predictor!r}"

            raise ValueError(msg)

        if plate in self._objects():
            msg = (
                f"a response plate must not be an object the family declares: {plate!r}"
            )

            raise ValueError(msg)

        emitted = list(self._discrimination_steps())
        scored = predictor

        if self._discrimination is not None:
            scored = self.scaled_site
            emitted.append(
                Bind(scored, Binary(Operator.MUL, self._scale(), Name(predictor)))
            )

        emitted.append(
            Bind(
                self.logit_site,
                Table(scored, plate, self.index_object, width=self._alternatives),
            )
        )
        ordered = self._ordered

        if ordered is None:
            emitted.append(ObserveChoice(response, plate, logits=self.logit_site))

            return tuple(emitted)

        contrast = self.contrast_column

        if contrast is None:
            msg = "a tie is observed over a difference and this family weights none"

            raise ValueError(msg)

        emitted += [
            Bind(
                self.contrast_site,
                Table(
                    contrast,
                    self.scalar_object,
                    self.index_object,
                    width=self._alternatives,
                ),
            ),
            Bind(
                self.weighted_site,
                Binary(Operator.MUL, Name(self.logit_site), Name(self.contrast_site)),
            ),
            Bind(
                self.difference_site,
                Binary(
                    Operator.MUL,
                    Literal(float(self._alternatives)),
                    Apply(Function.MEAN, Name(self.weighted_site)),
                ),
            ),
            *ordered.steps(
                predictor=self.difference_site, response=response, plate=plate
            ),
        ]

        return tuple(emitted)

    def _objects(self) -> frozenset[str]:
        """Return every finite set the family declares or indexes a draw by."""
        declared = {self.index_object, self.scalar_object}
        group = self._discrimination_group

        if group is not None:
            declared.add(group.index_object)

        ordered = self._ordered

        if ordered is not None:
            declared |= {ordered.index_object, ordered.scalar_object}

        return frozenset(declared)

    @override
    def __repr__(self) -> str:
        """Return the alternative count and whether a tie is recordable."""
        return (
            f"ChoiceFamily({self._alternatives!r}, ties={self._ordered is not None!r})"
        )
