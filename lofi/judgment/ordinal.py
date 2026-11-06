"""The cumulative link head over ordered response levels."""

import math
from itertools import pairwise
from typing import TYPE_CHECKING, Never, final, override

import torch
from torch import nn
from torch.nn import functional

from lofi._checks import ensure_finite, ensure_non_negative
from lofi.judgment._spec import Cumulative

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.judgment._design import JudgmentDesign
    from lofi.judgment._ports import LatentPredictor
    from lofi.judgment._spec import ResponseSpec

__all__ = [
    "CumulativeLinkHead",
    "CumulativeParameters",
    "OrdinalHead",
    "RowEffects",
    "log_category_probabilities",
    "ordered_cutpoints",
]

#: Natural logarithm of two. The two forms below switch at its negation.
_LOG_TWO = math.log(2.0)


def _log1mexp(value: Tensor) -> Tensor:
    """Return the logarithm of one minus the exponential of a nonpositive value.

    Two formulas cover separate halves of the domain. The lower branch is
    clamped because ``torch.where`` evaluates both branches and its derivative
    is singular just below zero.

    Parameters
    ----------
    value : Tensor
        Nonpositive logarithm of a ratio of two cumulative probabilities.
    """
    lower = torch.clamp(value, max=-_LOG_TWO)

    return torch.where(
        value > -_LOG_TWO,
        torch.log(-torch.expm1(value)),
        torch.log1p(-torch.exp(lower)),
    )


def ordered_cutpoints(base: Tensor, spans: Tensor) -> Tensor:
    """Return nondecreasing cutpoints from a base and unconstrained spans.

    The lowest cutpoint sits one softplus-transformed span above the base, and
    each later cutpoint adds another. Softplus is nonnegative for every real
    span, so a negative span narrows the gap to zero and never reverses it.
    Spans with a leading row axis give one row of cutpoints per row of spans.

    Parameters
    ----------
    base : Tensor
        Location the cutpoints rise from.
    spans : Tensor
        One unconstrained span per cutpoint, shared or per row.
    """
    return base.unsqueeze(-1) + torch.cumsum(functional.softplus(spans), dim=-1)


def log_category_probabilities(latent: Tensor, cutpoints: Tensor) -> Tensor:
    """Return the log probability of each response level.

    Probabilities are differences between adjacent logistic cumulative values.
    The subtraction is performed in log space to avoid underflow; rows index
    latent scores and columns index response levels.

    Parameters
    ----------
    latent : Tensor
        Latent score for each response row.
    cutpoints : Tensor
        Nondecreasing cutpoints separating adjacent response levels, shared by
        every row or given as one row of cutpoints per latent score.
    """
    offsets = cutpoints - latent.unsqueeze(-1)
    log_cumulative = functional.logsigmoid(offsets)
    log_complement = functional.logsigmoid(-offsets)
    below = log_cumulative[:, :-1]
    above = log_cumulative[:, 1:]

    return torch.cat(
        (
            log_cumulative[:, :1],
            above + _log1mexp(below - above),
            log_complement[:, -1:],
        ),
        dim=1,
    )


def _ensure_float(field: str, value: Tensor, rank: int) -> None:
    """Check that a tensor holds floating-point values at the expected rank.

    Parameters
    ----------
    field : str
        Name of the value, used in the error message.
    value : Tensor
        Value to check.
    rank : int
        Number of axes the value is expected to have.

    Raises
    ------
    TypeError
        If the tensor holds values of another kind.
    ValueError
        If the tensor has another rank.
    """
    if not value.is_floating_point():
        msg = f"{field} holds {value.dtype}, which is not a floating-point kind"

        raise TypeError(msg)

    if value.ndim != rank:
        msg = f"{field} has rank {value.ndim}, expected {rank}"

        raise ValueError(msg)


@final
class CumulativeParameters:
    """Stores the unconstrained parameters of a cumulative link head.

    Every field is unconstrained. The head applies softplus to the spans and to
    the discrimination, so no field needs a positivity constraint.

    Parameters
    ----------
    base : Tensor
        Location the cutpoints rise from.
    spans : Tensor
        One unconstrained span per cutpoint.
    discrimination : Tensor | None
        Unconstrained discrimination, or None where the response family fixes
        discrimination at one.

    Raises
    ------
    TypeError
        If a field does not hold floating-point values.
    ValueError
        If a field has the wrong rank, no span is given, or two fields disagree
        about dtype.
    """

    __slots__ = ("_base", "_discrimination", "_spans")

    def __init__(
        self,
        base: Tensor,
        spans: Tensor,
        discrimination: Tensor | None = None,
    ) -> None:
        _ensure_float("a cumulative head's base", base, 0)
        _ensure_float("a cumulative head's spans", spans, 1)

        if spans.numel() == 0:
            msg = "a cumulative head needs one span per cutpoint and was given none"

            raise ValueError(msg)

        if spans.dtype is not base.dtype:
            msg = (
                f"a cumulative head's base holds {base.dtype} and its spans hold "
                f"{spans.dtype}"
            )

            raise ValueError(msg)

        if discrimination is not None:
            _ensure_float("a cumulative head's discrimination", discrimination, 0)

            if discrimination.dtype is not base.dtype:
                msg = (
                    f"a cumulative head's base holds {base.dtype} and its "
                    f"discrimination holds {discrimination.dtype}"
                )

                raise ValueError(msg)

        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_spans", spans)
        object.__setattr__(self, "_discrimination", discrimination)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "cumulative-head parameters are immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Raise AttributeError on field deletion."""
        del name
        msg = "cumulative-head parameters are immutable"

        raise AttributeError(msg)

    @property
    def base(self) -> Tensor:
        """Return the location the cutpoints rise from."""
        return self._base

    @property
    def spans(self) -> Tensor:
        """Return the unconstrained span of each cutpoint."""
        return self._spans

    @property
    def discrimination(self) -> Tensor | None:
        """Return the unconstrained discrimination, or None where it is fixed."""
        return self._discrimination

    @property
    def cutpoint_count(self) -> int:
        """Return the number of cutpoints these parameters describe."""
        return self._spans.numel()


@final
class RowEffects:
    """Stores the random-effect offsets of each response row.

    Cutpoint offsets act on unconstrained spans, preserving order; offset ``k``
    moves cutpoint ``k`` and every later cutpoint. Priors distinguish a common
    cutpoint shift from the opposite latent shift.

    Parameters
    ----------
    latent : Tensor
        Offset to the latent score of each row.
    spans : Tensor | None
        Offset to each cutpoint span of each row, or None where the effects
        move the latent score alone.

    Raises
    ------
    TypeError
        If a field does not hold floating-point values.
    ValueError
        If a field has the wrong rank, the two fields disagree about the number
        of rows, or the two fields disagree about dtype.
    """

    __slots__ = ("_latent", "_spans")

    def __init__(self, latent: Tensor, spans: Tensor | None = None) -> None:
        _ensure_float("a row effect's latent offset", latent, 1)

        if spans is not None:
            _ensure_float("a row effect's span offsets", spans, 2)

            if spans.shape[0] != latent.numel():
                msg = (
                    f"a row effect offsets {latent.numel()} latent scores and "
                    f"{spans.shape[0]} rows of spans"
                )

                raise ValueError(msg)

            if spans.dtype is not latent.dtype:
                msg = (
                    f"a row effect's latent offset holds {latent.dtype} and its "
                    f"span offsets hold {spans.dtype}"
                )

                raise ValueError(msg)

        object.__setattr__(self, "_latent", latent)
        object.__setattr__(self, "_spans", spans)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "row effects are immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Raise AttributeError on field deletion."""
        del name
        msg = "row effects are immutable"

        raise AttributeError(msg)

    @property
    def latent(self) -> Tensor:
        """Return the offset to each row's latent score."""
        return self._latent

    @property
    def spans(self) -> Tensor | None:
        """Return the offset to each row's cutpoint spans, if any."""
        return self._spans

    @property
    def rows(self) -> int:
        """Return the number of rows these effects offset."""
        return self._latent.numel()

    def __add__(self, other: RowEffects) -> RowEffects:
        """Return the effects that apply both sets of offsets.

        Parameters
        ----------
        other : RowEffects
            Offsets over the same rows, in the same dtype.

        Raises
        ------
        ValueError
            If the two effects offset different numbers of rows, hold
            different dtypes, or offset different numbers of spans.
        """
        if other.rows != self.rows:
            msg = f"row effects over {self.rows} rows were added to {other.rows}"

            raise ValueError(msg)

        if other._latent.dtype is not self._latent.dtype:
            msg = (
                f"row effects holding {self._latent.dtype} were added to "
                f"{other._latent.dtype}"
            )

            raise ValueError(msg)

        if (
            self._spans is not None
            and other._spans is not None
            and self._spans.shape[1] != other._spans.shape[1]
        ):
            msg = (
                f"row effects over {self._spans.shape[1]} spans were added to "
                f"{other._spans.shape[1]}"
            )

            raise ValueError(msg)

        if self._spans is None:
            spans = other._spans
        elif other._spans is None:
            spans = self._spans
        else:
            spans = self._spans + other._spans

        return RowEffects(self._latent + other._latent, spans)


@final
class CumulativeLinkHead:
    """A cumulative link head whose latent score comes from a declared predictor.

    Parameters
    ----------
    spec : ResponseSpec
        Response specification declaring the family, the level count and the
        predictor terms.
    predictor : LatentPredictor
        Evaluator for those terms.

    Raises
    ------
    TypeError
        If the specification declares another response family.
    ValueError
        If the evaluator implements terms the specification does not declare.
    """

    __slots__ = ("_discriminated", "_levels", "_predictor")

    def __init__(self, spec: ResponseSpec, predictor: LatentPredictor) -> None:
        family = spec.family

        if not isinstance(family, Cumulative):
            msg = (
                f"a cumulative-link head is defined for the cumulative family "
                f"alone, and this specification declares {family.family!r}"
            )

            raise TypeError(msg)

        if predictor.terms != spec.predictor:
            msg = (
                f"the evaluator implements {predictor.terms!r} against the "
                f"declared terms {spec.predictor!r}"
            )

            raise ValueError(msg)

        self._levels = spec.levels
        self._discriminated = family.draws_discrimination
        self._predictor = predictor

    @property
    def levels(self) -> int:
        """Return the number of ordered response levels."""
        return self._levels

    @property
    def cutpoint_count(self) -> int:
        """Return the number of cutpoints between adjacent levels."""
        return self._levels - 1

    @property
    def predictor(self) -> LatentPredictor:
        """Return the evaluator of the declared predictor terms."""
        return self._predictor

    def _check(self, parameters: CumulativeParameters) -> None:
        """Check that parameters describe this head's model.

        Parameters
        ----------
        parameters : CumulativeParameters
            Unconstrained parameters to pair with this head.

        Raises
        ------
        ValueError
            If the parameters give another number of cutpoints, or disagree with
            the declared family about drawing discrimination.
        """
        if parameters.cutpoint_count != self.cutpoint_count:
            msg = (
                f"a head over {self._levels} levels needs {self.cutpoint_count} "
                f"cutpoints and these parameters give {parameters.cutpoint_count}"
            )

            raise ValueError(msg)

        if (parameters.discrimination is not None) != self._discriminated:
            drawn = "draws" if self._discriminated else "does not draw"
            given = "given" if parameters.discrimination is not None else "not given"
            msg = (
                f"the declared family {drawn} discrimination and these parameters "
                f"have it {given}"
            )

            raise ValueError(msg)

    def discrimination(self, parameters: CumulativeParameters) -> Tensor:
        """Return the positive discrimination the head applies.

        A family that does not draw discrimination fixes it at one.

        Parameters
        ----------
        parameters : CumulativeParameters
            Unconstrained parameters of this head.
        """
        self._check(parameters)

        if parameters.discrimination is None:
            return torch.ones_like(parameters.base)

        return functional.softplus(parameters.discrimination)

    def cutpoints(
        self,
        parameters: CumulativeParameters,
        *,
        effects: RowEffects | None = None,
    ) -> Tensor:
        """Return the nondecreasing cutpoints on the latent scale.

        Without effects the result is one cutpoint per boundary between levels.
        With effects that offset the spans it is one such row per response row.

        Parameters
        ----------
        parameters : CumulativeParameters
            Unconstrained parameters of this head.
        effects : RowEffects | None
            Random-effect offsets of each response row, if any.
        """
        self._check(parameters)
        spans = parameters.spans

        if effects is not None and effects.spans is not None:
            if effects.spans.shape[1] != self.cutpoint_count:
                msg = (
                    f"a head over {self.cutpoint_count} cutpoints was given "
                    f"effects over {effects.spans.shape[1]}"
                )

                raise ValueError(msg)

            spans = spans + effects.spans

        rising = ordered_cutpoints(parameters.base, spans)

        if parameters.discrimination is None:
            return rising

        return functional.softplus(parameters.discrimination) * rising

    def _scaled(
        self,
        latent: Tensor,
        parameters: CumulativeParameters,
        effects: RowEffects | None,
    ) -> Tensor:
        """Return the latent score on the scale the cutpoints live on.

        Parameters
        ----------
        latent : Tensor
            One latent score per response row.
        parameters : CumulativeParameters
            Unconstrained parameters of this head.
        effects : RowEffects | None
            Random-effect offsets of each response row, if any.

        Raises
        ------
        ValueError
            If the latent score is not one score per row, or the effects offset
            another number of rows.
        """
        _ensure_float("a cumulative head's latent score", latent, 1)

        if effects is not None:
            if effects.rows != latent.numel():
                msg = (
                    f"{latent.numel()} latent scores were given effects over "
                    f"{effects.rows} rows"
                )

                raise ValueError(msg)

            latent = latent + effects.latent

        if parameters.discrimination is None:
            return latent

        return functional.softplus(parameters.discrimination) * latent

    def cumulative_probabilities(
        self,
        latent: Tensor,
        parameters: CumulativeParameters,
        *,
        effects: RowEffects | None = None,
    ) -> Tensor:
        """Return the probability of responding at or below each cutpoint.

        The result has one row per latent score and one column per cutpoint.

        Parameters
        ----------
        latent : Tensor
            One latent score per response row.
        parameters : CumulativeParameters
            Unconstrained parameters of this head.
        effects : RowEffects | None
            Random-effect offsets of each response row, if any.
        """
        cutpoints = self.cutpoints(parameters, effects=effects)
        scaled = self._scaled(latent, parameters, effects)

        return torch.sigmoid(cutpoints - scaled.unsqueeze(-1))

    def log_probabilities(
        self,
        latent: Tensor,
        parameters: CumulativeParameters,
        *,
        effects: RowEffects | None = None,
    ) -> Tensor:
        """Return the log probability of each response level.

        The result has one row per latent score and one column per level.

        Parameters
        ----------
        latent : Tensor
            One latent score per response row.
        parameters : CumulativeParameters
            Unconstrained parameters of this head.
        effects : RowEffects | None
            Random-effect offsets of each response row, if any.
        """
        return log_category_probabilities(
            self._scaled(latent, parameters, effects),
            self.cutpoints(parameters, effects=effects),
        )

    def probabilities(
        self,
        latent: Tensor,
        parameters: CumulativeParameters,
        *,
        effects: RowEffects | None = None,
    ) -> Tensor:
        """Return the probability of each response level.

        The result has one row per latent score and one column per level.

        Parameters
        ----------
        latent : Tensor
            One latent score per response row.
        parameters : CumulativeParameters
            Unconstrained parameters of this head.
        effects : RowEffects | None
            Random-effect offsets of each response row, if any.
        """
        return torch.exp(self.log_probabilities(latent, parameters, effects=effects))

    def latent_scores(self, design: JudgmentDesign) -> Tensor:
        """Return the declared predictor's score for each judgment row.

        Parameters
        ----------
        design : JudgmentDesign
            Dense columns of the judgments to score.

        Raises
        ------
        TypeError
            If the evaluator returns scores of a nonfloating kind.
        ValueError
            If the evaluator returns scores of another rank, or one score per row
            of another design.
        """
        latent = self._predictor.latent_scores(design)

        _ensure_float("a predictor's latent score", latent, 1)

        if latent.numel() != design.n_rows:
            msg = (
                f"a design of {design.n_rows} rows was scored with "
                f"{latent.numel()} latent scores"
            )

            raise ValueError(msg)

        return latent

    def log_likelihood(
        self,
        design: JudgmentDesign,
        parameters: CumulativeParameters,
        *,
        effects: RowEffects | None = None,
    ) -> Tensor:
        """Return the log probability of the response recorded in each row.

        Parameters
        ----------
        design : JudgmentDesign
            Dense columns of the judgments to score.
        parameters : CumulativeParameters
            Unconstrained parameters of this head.
        effects : RowEffects | None
            Random-effect offsets of each response row, if any.

        Raises
        ------
        ValueError
            If the design records another number of response levels.
        """
        if design.response_levels != self._levels:
            msg = (
                f"a head over {self._levels} levels was given a design over "
                f"{design.response_levels}"
            )

            raise ValueError(msg)

        log_probabilities = self.log_probabilities(
            self.latent_scores(design),
            parameters,
            effects=effects,
        )

        return log_probabilities.gather(1, design.responses.unsqueeze(1)).squeeze(1)


def _checked_levels(levels: int) -> int:
    """Return a number of ordered response levels a head can separate.

    Parameters
    ----------
    levels : int
        The number of ordered response levels.

    Raises
    ------
    TypeError
        If the count is not an integer.
    ValueError
        If the count is below two.
    """
    checked = ensure_non_negative("the number of levels", levels)

    if checked < 2:
        msg = f"a graded judgment has at least two levels, and {levels} were declared"

        raise ValueError(msg)

    return checked


def _checked_cutpoints(
    cutpoints: tuple[float, ...] | None, levels: int
) -> tuple[float, ...]:
    """Return strictly increasing initial cutpoints, one fewer than the levels.

    Parameters
    ----------
    cutpoints : tuple[float, ...] | None
        The initial cutpoints, or None for unit gaps centered on zero.
    levels : int
        The number of ordered response levels.

    Raises
    ------
    TypeError
        If a cutpoint is not a number.
    ValueError
        If a cutpoint is not finite, if there is not one fewer cutpoint than levels,
        or if the cutpoints do not strictly increase.
    """
    if cutpoints is None:
        return tuple(index - (levels - 2) / 2.0 for index in range(levels - 1))

    checked = tuple(
        ensure_finite(f"cutpoint {index}", cutpoint)
        for index, cutpoint in enumerate(cutpoints)
    )

    if len(checked) != levels - 1:
        msg = (
            f"{levels} levels are separated by {levels - 1} cutpoints, and "
            f"{len(checked)} were given"
        )

        raise ValueError(msg)

    if any(above <= below for below, above in pairwise(checked)):
        msg = f"cutpoints are strictly increasing, and these are not: {checked!r}"

        raise ValueError(msg)

    return checked


def _inverse_softplus(gap: float) -> float:
    """Return the span whose softplus is a positive gap.

    Parameters
    ----------
    gap : float
        A positive distance between adjacent cutpoints.
    """
    return math.log(math.expm1(gap))


@final
class OrdinalHead(nn.Module):
    """Trainable cutpoints giving the log probability of each response level.

    The cutpoints are held as a base and one unconstrained span per cutpoint,
    so they stay strictly increasing under any parameter values. A latent
    score is scored against them with the logistic cumulative link.

    Parameters
    ----------
    levels : int
        The number of ordered response levels.
    cutpoints : tuple[float, ...] | None
        The initial cutpoints, or None for unit gaps centered on zero.

    Raises
    ------
    TypeError
        If the level count is not an integer or a cutpoint is not a number.
    ValueError
        If fewer than two levels are declared, or if the cutpoints are not
        finite, are not one fewer than the levels, or do not strictly increase.
    """

    def __init__(self, levels: int, cutpoints: tuple[float, ...] | None = None) -> None:
        super().__init__()
        self._levels = _checked_levels(levels)
        rising = _checked_cutpoints(cutpoints, self._levels)
        gaps = (1.0, *(above - below for below, above in pairwise(rising)))
        self.base = nn.Parameter(torch.tensor(rising[0] - 1.0, dtype=torch.float64))
        self.spans = nn.Parameter(
            torch.tensor([_inverse_softplus(gap) for gap in gaps], dtype=torch.float64)
        )

    @property
    def levels(self) -> int:
        """Return the number of ordered response levels."""
        return self._levels

    @property
    def cutpoint_count(self) -> int:
        """Return the number of cutpoints between adjacent levels."""
        return self._levels - 1

    def cutpoints(self) -> Tensor:
        """Return the strictly increasing cutpoints between adjacent levels."""
        return ordered_cutpoints(self.base, self.spans)

    def log_probabilities(self, latent: Tensor) -> Tensor:
        """Return the log probability of each level at each latent score.

        The result has one row per latent score and one column per level.

        Parameters
        ----------
        latent : Tensor
            One latent score per response row.
        """
        return log_category_probabilities(latent, self.cutpoints())

    @override
    def __repr__(self) -> str:
        """Return the level count."""
        return f"OrdinalHead(levels={self._levels!r})"
