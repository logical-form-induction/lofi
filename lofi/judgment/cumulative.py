"""Cumulative-response program steps.

The family emits the steps of a response program: the cutpoint draws and
bindings, an optional discrimination, and the ordered-logistic observation.
Nothing here builds or runs the program.
"""

import math
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
    ObserveOrdered,
    Operator,
    Prior,
    Table,
    ensure_identifier,
)

if TYPE_CHECKING:
    from lofi.judgment._program import Expression, Step

__all__ = [
    "DEFAULT_BASE_PRIOR",
    "DEFAULT_INCREMENT_PRIOR",
    "DEFAULT_SPREAD_PRIOR",
    "CumulativeFamily",
    "DiscriminationGroup",
    "PredictedParameter",
    "ThresholdGroup",
]

_MINIMUM_CATEGORIES = 2


#: Default base-threshold prior.
DEFAULT_BASE_PRIOR = Prior("Normal", 0.0, 5.0)

#: Default threshold-increment prior.
DEFAULT_INCREMENT_PRIOR = Prior("Normal", 0.0, 1.0)

#: Default prior on the spread of a group's threshold increments.
DEFAULT_SPREAD_PRIOR = Prior("HalfNormal", 1.0)


@final
class ThresholdGroup:
    """A grouping factor the thresholds vary over.

    Each level adds standardized offsets, scaled per threshold, to shared
    threshold increments. The program gathers each row's cutpoints through
    the declared host-data index.

    Parameters
    ----------
    index : str
        Host-data column holding each row's level, as a zero-based index.
    index_object : str
        Name of the finite set the levels index.
    spread : Prior
        Prior on the spread of each threshold's offsets across the levels.

    Raises
    ------
    ValueError
        If a name is not an identifier or the index and the index object
        coincide.
    """

    __slots__ = ("_index", "_index_object", "_spread")

    def __init__(
        self,
        index: str,
        index_object: str,
        *,
        spread: Prior = DEFAULT_SPREAD_PRIOR,
    ) -> None:
        self._index = ensure_identifier(index, "a threshold group's index")
        self._index_object = ensure_identifier(
            index_object, "a threshold group's index object"
        )

        if self._index == self._index_object:
            msg = (
                f"a threshold group's index and index object must differ: "
                f"{self._index!r}"
            )

            raise ValueError(msg)

        self._spread = spread

    @property
    def index(self) -> str:
        """Return the host-data column holding each row's level."""
        return self._index

    @property
    def index_object(self) -> str:
        """Return the finite set the levels index."""
        return self._index_object

    @property
    def spread(self) -> Prior:
        """Return the prior on each threshold's spread across the levels."""
        return self._spread

    @property
    def spread_site(self) -> str:
        """Return the site drawing one spread per threshold."""
        return f"{self._index}_cutpoint_spread"

    @property
    def offset_site(self) -> str:
        """Return the site drawing a standardized offset per level and threshold."""
        return f"{self._index}_cutpoint_offsets"

    @property
    def shift_site(self) -> str:
        """Return the variable holding the offsets as a level-by-threshold table."""
        return f"{self._index}_cutpoint_shifts"

    @property
    def cutpoint_site(self) -> str:
        """Return the variable holding one row of cutpoints per level."""
        return f"{self._index}_cutpoints"

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return (
            f"{type(self).__name__}({self._index!r}, {self._index_object!r}, "
            f"spread={self._spread!r})"
        )


@final
class DiscriminationGroup:
    """A grouping factor the discrimination varies over.

    Each level draws from the family's discrimination prior. The program
    gathers the row's discrimination through the declared host-data index.

    Parameters
    ----------
    index : str
        Host-data column holding each row's level, as a zero-based index.
    index_object : str
        Name of the finite set the levels index.

    Raises
    ------
    ValueError
        If a name is not an identifier or the index and the index object
        coincide.
    """

    __slots__ = ("_index", "_index_object")

    def __init__(self, index: str, index_object: str) -> None:
        self._index = ensure_identifier(index, "a discrimination group's index")
        self._index_object = ensure_identifier(
            index_object, "a discrimination group's index object"
        )

        if self._index == self._index_object:
            msg = (
                f"a discrimination group's index and index object must differ: "
                f"{self._index!r}"
            )

            raise ValueError(msg)

    @property
    def index(self) -> str:
        """Return the host-data column holding each row's level."""
        return self._index

    @property
    def index_object(self) -> str:
        """Return the finite set the levels index."""
        return self._index_object

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return f"{type(self).__name__}({self._index!r}, {self._index_object!r})"


@final
class PredictedParameter:
    """A parameter carrying a linear predictor the program binds elsewhere.

    The parameter is the exponential of the named variable, so a predictor
    of any sign gives a positive parameter and a predictor of zero gives
    one. The variable holds one value per response row, which is what lets
    the parameter vary from row to row rather than over the levels of a
    grouping factor.

    Parameters
    ----------
    variable : str
        Previously bound variable holding the parameter's logarithm, one
        value per response row.

    Raises
    ------
    ValueError
        If the name is not an identifier.
    """

    __slots__ = ("_variable",)

    def __init__(self, variable: str) -> None:
        self._variable = ensure_identifier(variable, "a predicted parameter's variable")

    @property
    def variable(self) -> str:
        """Return the variable holding the parameter's logarithm."""
        return self._variable

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return f"{type(self).__name__}({self._variable!r})"


def _group_names(group: ThresholdGroup) -> frozenset[str]:
    """Return every name a threshold group reads or binds in a program.

    Parameters
    ----------
    group : ThresholdGroup
        The group whose names are collected.
    """
    return frozenset(
        (
            group.index,
            group.spread_site,
            group.offset_site,
            group.shift_site,
            group.cutpoint_site,
        )
    )


@final
class CumulativeFamily:
    """An ordered-logit response family.

    Parameters
    ----------
    categories : int
        Number of ordered response categories.
    base : Prior
        Prior for the lowest threshold.
    increment : Prior
        Prior for each unconstrained threshold increment.
    discrimination : Prior | float | PredictedParameter | None
        Prior for a drawn discrimination, a positive value the discrimination
        is held at, a linear predictor it is the exponential of, or ``None``
        to fix it at one.
    group : ThresholdGroup | None
        Grouping factor the thresholds vary over, or ``None`` for one set of
        thresholds shared by every row.
    discrimination_group : DiscriminationGroup | None
        Grouping factor a drawn discrimination varies over, or ``None`` for
        one discrimination shared by every row.
    cutpoint_variance : float | None
        Population variance imposed on each row of cutpoints, or ``None`` to
        leave their scale free.

    Raises
    ------
    TypeError
        If ``categories``, a held discrimination, or a held cutpoint variance
        has the wrong type.
    ValueError
        If fewer than two categories are requested, a held discrimination is
        not a finite positive number, a held cutpoint variance is not finite
        and positive, a discrimination group is given without a drawn
        discrimination or beside a predicted one, a predicted discrimination
        reads a name the family binds, a group's index object is an object
        the family declares, or a group name conflicts with a family binding.
    """

    __slots__ = (
        "_base",
        "_categories",
        "_cutpoint_variance",
        "_discrimination",
        "_discrimination_group",
        "_group",
        "_increment",
    )

    def __init__(
        self,
        categories: int,
        *,
        base: Prior = DEFAULT_BASE_PRIOR,
        increment: Prior = DEFAULT_INCREMENT_PRIOR,
        discrimination: Prior | float | PredictedParameter | None = None,
        group: ThresholdGroup | None = None,
        discrimination_group: DiscriminationGroup | None = None,
        cutpoint_variance: float | None = None,
    ) -> None:
        if categories < _MINIMUM_CATEGORIES:
            msg = (
                f"an ordered response has at least {_MINIMUM_CATEGORIES} "
                f"categories: {categories}"
            )

            raise ValueError(msg)

        if isinstance(discrimination, bool):
            msg = f"a held discrimination is not a number: {discrimination!r}"

            raise TypeError(msg)

        if isinstance(discrimination, int | float) and not (
            math.isfinite(discrimination) and discrimination > 0.0
        ):
            msg = (
                f"a held discrimination is not a finite positive number: "
                f"{discrimination!r}"
            )

            raise ValueError(msg)

        if isinstance(cutpoint_variance, bool):
            msg = f"a held cutpoint variance is not a number: {cutpoint_variance!r}"

            raise TypeError(msg)

        if cutpoint_variance is not None and not (
            math.isfinite(cutpoint_variance) and cutpoint_variance > 0.0
        ):
            msg = (
                "a held cutpoint variance is not a finite positive number: "
                f"{cutpoint_variance!r}"
            )

            raise ValueError(msg)

        if group is not None:
            if group.index_object in (self.index_object, self.scalar_object):
                msg = (
                    f"a threshold group's index object must not be an object the "
                    f"family declares: {group.index_object!r}"
                )

                raise ValueError(msg)

            taken = sorted(_group_names(group) & self._own_names)

            if taken:
                msg = (
                    f"a threshold group must not name what the family binds: {taken!r}"
                )

                raise ValueError(msg)

        if isinstance(discrimination, PredictedParameter):
            if discrimination.variable in self._own_names:
                msg = (
                    f"a predicted discrimination must not read what the family "
                    f"binds: {discrimination.variable!r}"
                )

                raise ValueError(msg)

            if discrimination_group is not None:
                msg = (
                    f"discrimination follows the predictor "
                    f"{discrimination.variable!r} and cannot also vary over "
                    f"{discrimination_group.index!r}"
                )

                raise ValueError(msg)

        if discrimination_group is not None:
            if not isinstance(discrimination, Prior):
                msg = (
                    f"discrimination varies over {discrimination_group.index!r} "
                    f"and the family does not draw it"
                )

                raise ValueError(msg)

            if discrimination_group.index_object in (
                self.index_object,
                self.scalar_object,
            ):
                msg = (
                    f"a discrimination group's index object must not be an object "
                    f"the family declares: {discrimination_group.index_object!r}"
                )

                raise ValueError(msg)

            if discrimination_group.index in self._own_names:
                msg = (
                    f"a discrimination group must not name what the family binds: "
                    f"{discrimination_group.index!r}"
                )

                raise ValueError(msg)

        self._categories = categories
        self._cutpoint_variance = cutpoint_variance
        self._base = base
        self._increment = increment
        self._discrimination = discrimination
        self._group = group
        self._discrimination_group = discrimination_group

    @property
    def _own_names(self) -> frozenset[str]:
        """Return every variable the family may bind."""
        return frozenset(
            (
                self.base_site,
                self.increment_site,
                self.span_site,
                self.cutpoint_site,
                self.discrimination_site,
                self.discrimination_rows_site,
                self.discrimination_column_site,
                self.scaled_predictor_site,
                self.scaled_cutpoint_site,
                self.cutpoint_mean_site,
                self.cutpoint_mean_column_site,
                self.centered_cutpoint_site,
                self.cutpoint_variance_site,
                self.cutpoint_scale_site,
                self.cutpoint_scale_column_site,
            )
        )

    @property
    def categories(self) -> int:
        """Return the category count."""
        return self._categories

    @property
    def thresholds(self) -> int:
        """Return the threshold count."""
        return self._categories - 1

    @property
    def base(self) -> Prior:
        """Return the base-threshold prior."""
        return self._base

    @property
    def increment(self) -> Prior:
        """Return the threshold-increment prior."""
        return self._increment

    @property
    def discrimination(self) -> Prior | float | PredictedParameter | None:
        """Return the discrimination prior, held value, or predictor, if any."""
        return self._discrimination

    @property
    def group(self) -> ThresholdGroup | None:
        """Return the grouping factor the thresholds vary over, if any."""
        return self._group

    @property
    def discrimination_group(self) -> DiscriminationGroup | None:
        """Return the grouping factor the discrimination varies over, if any."""
        return self._discrimination_group

    @property
    def cutpoint_variance(self) -> float | None:
        """Return the population variance imposed on each cutpoint row."""
        return self._cutpoint_variance

    @property
    def index_object(self) -> str:
        """Return the threshold index-set name."""
        return "Cutpoint"

    @property
    def scalar_object(self) -> str:
        """Return the name of the one-element set a single draw is indexed by."""
        return "Unit"

    @property
    def base_site(self) -> str:
        """Return the base-threshold site name."""
        return "cutpoint_base"

    @property
    def increment_site(self) -> str:
        """Return the threshold-increment site name."""
        return "cutpoint_increments"

    @property
    def discrimination_site(self) -> str:
        """Return the discrimination site name."""
        return "discrimination"

    @property
    def discrimination_rows_site(self) -> str:
        """Return the variable holding each row's discrimination, gathered by level."""
        return "discrimination_rows"

    @property
    def discrimination_column_site(self) -> str:
        """Return the variable holding each row's discrimination as one column."""
        return "discrimination_column"

    @property
    def span_site(self) -> str:
        """Return the site holding each cutpoint's rise above the base."""
        return "cutpoint_spans"

    @property
    def cutpoint_site(self) -> str:
        """Return the cutpoint site name."""
        return "cutpoints"

    @property
    def cutpoint_mean_site(self) -> str:
        """Return the site holding each cutpoint row's mean."""
        return "cutpoint_means"

    @property
    def cutpoint_mean_column_site(self) -> str:
        """Return the site holding grouped cutpoint means as a column."""
        return "cutpoint_mean_column"

    @property
    def centered_cutpoint_site(self) -> str:
        """Return the site holding cutpoints centered within each row."""
        return "centered_cutpoints"

    @property
    def cutpoint_variance_site(self) -> str:
        """Return the site holding each cutpoint row's population variance."""
        return "cutpoint_variances"

    @property
    def cutpoint_scale_site(self) -> str:
        """Return the site holding each cutpoint row's identifying scale."""
        return "cutpoint_scales"

    @property
    def cutpoint_scale_column_site(self) -> str:
        """Return the site holding grouped cutpoint scales as a column."""
        return "cutpoint_scale_column"

    @property
    def scaled_predictor_site(self) -> str:
        """Return the scaled-predictor site name."""
        return "scaled_predictor"

    @property
    def scaled_cutpoint_site(self) -> str:
        """Return the scaled-cutpoint site name."""
        return "scaled_cutpoints"

    @property
    def sites(self) -> tuple[str, ...]:
        """Return the latent sites in draw order."""
        drawn = [self.base_site, self.increment_site]

        if self._group is not None:
            drawn += [self._group.spread_site, self._group.offset_site]

        if isinstance(self._discrimination, Prior):
            drawn.append(self.discrimination_site)

        return tuple(drawn)

    @property
    def bound_names(self) -> tuple[str, ...]:
        """Return variables bound by the family in binding order."""
        bound = [self.base_site, self.increment_site]

        if self._group is not None:
            bound += [
                self._group.spread_site,
                self._group.offset_site,
                self._group.shift_site,
            ]

        bound.append(self.span_site)

        if self._cutpoint_variance is not None:
            bound += [
                self.cutpoint_mean_site,
                self.centered_cutpoint_site,
                self.cutpoint_variance_site,
                self.cutpoint_scale_site,
            ]

            if self._group is not None:
                bound += [
                    self.cutpoint_mean_column_site,
                    self.cutpoint_scale_column_site,
                ]

        if self._group is not None:
            bound.append(self._group.cutpoint_site)

        bound.append(self.cutpoint_site)

        if isinstance(self._discrimination, Prior | PredictedParameter):
            bound.append(self.discrimination_site)

        if self._discrimination_group is not None:
            bound += [self.discrimination_rows_site, self.discrimination_column_site]
        elif isinstance(self._discrimination, PredictedParameter):
            bound.append(self.discrimination_column_site)

        if self._discrimination is not None:
            bound += [self.scaled_predictor_site, self.scaled_cutpoint_site]

        return tuple(bound)

    @property
    def host_columns(self) -> tuple[str, ...]:
        """Return the host-data columns the emitted steps read, without repeats."""
        columns = []

        if self._group is not None:
            columns.append(self._group.index)

        if self._discrimination_group is not None:
            columns.append(self._discrimination_group.index)

        return tuple(dict.fromkeys(columns))

    def _threshold_steps(self) -> tuple[Step, ...]:
        """Return the steps binding the cutpoints.

        Without a group the cutpoints are one row: the base plus the running
        sum of the softplus-transformed increments. With a group the
        increments of each level are the shared increments plus that level's
        spread-scaled offsets, giving one row of cutpoints per level, and the
        cutpoints of each observed row are gathered through the group's
        index.
        """
        emitted: list[Step] = [
            Draw(self.base_site, (self.scalar_object,), self._base),
            Draw(self.increment_site, (self.index_object,), self._increment),
        ]
        rising: Expression = Name(self.increment_site)

        if self._group is None:
            emitted.append(Bind(self.span_site, _running_softplus(rising)))

            if self._cutpoint_variance is None:
                emitted.append(
                    Bind(
                        self.cutpoint_site,
                        Binary(
                            Operator.ADD, Name(self.base_site), Name(self.span_site)
                        ),
                    )
                )
            else:
                emitted += self._identified_cutpoint_steps(
                    self.span_site, self.cutpoint_site
                )

            return tuple(emitted)

        group = self._group
        rising = Binary(
            Operator.ADD,
            rising,
            Binary(Operator.MUL, Name(group.spread_site), Name(group.shift_site)),
        )
        emitted += [
            Draw(group.spread_site, (self.index_object,), group.spread),
            Draw(
                group.offset_site,
                (group.index_object, self.index_object),
                Prior("Normal", 0.0, 1.0),
            ),
            Bind(
                group.shift_site,
                Table(
                    group.offset_site,
                    group.index_object,
                    self.index_object,
                    width=self.thresholds,
                ),
            ),
            Bind(self.span_site, _running_softplus(rising)),
        ]

        if self._cutpoint_variance is None:
            emitted.append(
                Bind(
                    group.cutpoint_site,
                    Binary(Operator.ADD, Name(self.base_site), Name(self.span_site)),
                )
            )
        else:
            emitted += self._identified_cutpoint_steps(
                self.span_site, group.cutpoint_site, grouped=group
            )

        emitted.append(
            Bind(self.cutpoint_site, Gather(group.cutpoint_site, group.index))
        )

        return tuple(emitted)

    def _identified_cutpoint_steps(
        self,
        source: str,
        output: str,
        *,
        grouped: ThresholdGroup | None = None,
    ) -> tuple[Step, ...]:
        """Scale each cutpoint row to the declared population variance.

        Parameters
        ----------
        source : str
            Site holding unscaled ordered cutpoint spans.
        output : str
            Site to bind to the identified cutpoints.
        grouped : ThresholdGroup | None
            Threshold group when ``source`` is a level-by-threshold table.
        """
        variance = self._cutpoint_variance

        if variance is None:
            msg = "cutpoints can be identified only when their variance is fixed"

            raise ValueError(msg)

        mean: Expression = Name(self.cutpoint_mean_site)
        steps: list[Step] = [
            Bind(self.cutpoint_mean_site, Apply(Function.MEAN, Name(source)))
        ]

        if grouped is not None:
            steps.append(
                Bind(
                    self.cutpoint_mean_column_site,
                    Table(
                        self.cutpoint_mean_site,
                        grouped.index_object,
                        self.scalar_object,
                        width=1,
                    ),
                )
            )
            mean = Name(self.cutpoint_mean_column_site)

        steps += [
            Bind(
                self.centered_cutpoint_site,
                Binary(Operator.SUB, Name(source), mean),
            ),
            Bind(
                self.cutpoint_variance_site,
                Apply(
                    Function.MEAN,
                    Apply(Function.SQUARE, Name(self.centered_cutpoint_site)),
                ),
            ),
            Bind(
                self.cutpoint_scale_site,
                Apply(
                    Function.SQRT,
                    Binary(
                        Operator.DIV,
                        Literal(variance),
                        Name(self.cutpoint_variance_site),
                    ),
                ),
            ),
        ]
        scale: Expression = Name(self.cutpoint_scale_site)

        if grouped is not None:
            steps.append(
                Bind(
                    self.cutpoint_scale_column_site,
                    Table(
                        self.cutpoint_scale_site,
                        grouped.index_object,
                        self.scalar_object,
                        width=1,
                    ),
                )
            )
            scale = Name(self.cutpoint_scale_column_site)

        steps.append(
            Bind(
                output,
                Binary(
                    Operator.ADD,
                    Name(self.base_site),
                    Binary(Operator.MUL, Name(self.centered_cutpoint_site), scale),
                ),
            )
        )

        return tuple(steps)

    def steps(self, *, predictor: str, response: str, plate: str) -> tuple[Step, ...]:
        """Return program steps that observe a response through this family.

        Parameters
        ----------
        predictor : str
            Previously bound linear-predictor variable.
        response : str
            Variable to bind to the observed response.
        plate : str
            Finite set indexing response rows.

        Returns
        -------
        tuple[Step, ...]
            The steps in program order, the observation last.

        Raises
        ------
        ValueError
            If a name is invalid, collides with a family-owned name, or creates
            an invalid predictor, response, or plate pairing.
        """
        ensure_identifier(predictor, "a predictor")
        ensure_identifier(response, "a response")
        ensure_identifier(plate, "a response plate")

        bound = frozenset(self.bound_names) | frozenset(self.host_columns)

        for name, role in ((predictor, "a predictor"), (response, "a response")):
            if name in bound:
                msg = f"{role} must not be a name this family binds: {name!r}"

                raise ValueError(msg)

            if (
                isinstance(self._discrimination, PredictedParameter)
                and name == self._discrimination.variable
            ):
                msg = (
                    f"{role} and the predictor a discrimination follows must "
                    f"differ: {name!r}"
                )

                raise ValueError(msg)

        if predictor == response:
            msg = f"a predictor and a response must differ: {predictor!r}"

            raise ValueError(msg)

        objects = {self.index_object, self.scalar_object}

        if self._group is not None:
            objects.add(self._group.index_object)

        if self._discrimination_group is not None:
            objects.add(self._discrimination_group.index_object)

        if plate in objects:
            msg = (
                f"a response plate must not be an object the family declares: {plate!r}"
            )

            raise ValueError(msg)

        emitted = list(self._threshold_steps())
        location, thresholds = predictor, self.cutpoint_site

        if self._discrimination is not None:
            location = self.scaled_predictor_site
            thresholds = self.scaled_cutpoint_site
            emitted += self._discrimination_steps(plate)
            row_scale, cutpoint_scale = self._discrimination_scales(
                self._discrimination
            )
            emitted += [
                Bind(location, Binary(Operator.MUL, row_scale, Name(predictor))),
                Bind(
                    thresholds,
                    Binary(Operator.MUL, cutpoint_scale, Name(self.cutpoint_site)),
                ),
            ]

        emitted.append(
            ObserveOrdered(response, plate, location=location, thresholds=thresholds)
        )

        return tuple(emitted)

    def _discrimination_steps(self, plate: str) -> tuple[Step, ...]:
        """Return the steps binding the discrimination, if the family has one.

        A shared discrimination is one draw over the one-element set. One
        varying over a group is drawn over the group's set, gathered by the
        group's index into one value per row, and read again as a
        one-column table so that it scales each row's cutpoints. One
        carrying a linear predictor is the exponential of that predictor,
        read the same two ways.

        Parameters
        ----------
        plate : str
            Finite set indexing response rows.
        """
        predicted = self._discrimination

        if isinstance(predicted, PredictedParameter):
            return (
                Bind(
                    self.discrimination_site,
                    Apply(Function.EXP, Name(predicted.variable)),
                ),
                Bind(
                    self.discrimination_column_site,
                    Table(self.discrimination_site, plate, self.scalar_object, width=1),
                ),
            )

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
            Bind(
                self.discrimination_column_site,
                Table(
                    self.discrimination_rows_site, plate, self.scalar_object, width=1
                ),
            ),
        )

    def _discrimination_scales(
        self, discrimination: Prior | float | PredictedParameter
    ) -> tuple[Expression, Expression]:
        """Return the expressions scaling the predictor and the cutpoints.

        A held or shared discrimination scales both by one value. One
        varying over a group, and one carrying a linear predictor, scale
        the predictor by each row's value and the cutpoints by the same
        values as a one-column table.

        Parameters
        ----------
        discrimination : Prior | float | PredictedParameter
            Prior of a drawn discrimination, the value it is held at, or
            the linear predictor it is the exponential of.
        """
        if isinstance(discrimination, PredictedParameter):
            return (
                Name(self.discrimination_site),
                Name(self.discrimination_column_site),
            )

        if not isinstance(discrimination, Prior):
            held = Literal(discrimination)

            return held, held

        if self._discrimination_group is None:
            shared = Name(self.discrimination_site)

            return shared, shared

        return (
            Name(self.discrimination_rows_site),
            Name(self.discrimination_column_site),
        )


def _running_softplus(increments: Expression) -> Expression:
    """Return the running sum of the softplus of increments.

    Softplus is nonnegative for every real increment, so the running sum
    never decreases along the threshold axis.

    Parameters
    ----------
    increments : Expression
        One unconstrained increment per threshold.
    """
    return Apply(Function.CUMSUM, Apply(Function.SOFTPLUS, increments))
