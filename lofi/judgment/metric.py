"""The metric head over a normalized response.

The head reads an ordered response as a number on an interval scale and
gives it a Gaussian density about the latent score. It holds no cutpoints,
so the distance between two adjacent levels is one everywhere and the only
correction for how a participant uses the scale is the normalization the
response is put through before the head sees it.
"""

import math
from typing import TYPE_CHECKING, Never, final, override

import torch
from torch.nn import functional

from lofi.judgment._spec import Metric, ScaleNormalization

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.judgment._design import JudgmentDesign
    from lofi.judgment._ports import LatentPredictor
    from lofi.judgment._spec import ResponseSpec
    from lofi.judgment.ordinal import RowEffects

__all__ = [
    "MetricHead",
    "MetricParameters",
    "item_response_weights",
    "normalized_responses",
    "ridit_scores",
]

#: Natural logarithm of two pi, the constant term of a Gaussian log density.
_LOG_TWO_PI = math.log(2.0 * math.pi)


def ridit_scores(design: JudgmentDesign, /) -> Tensor:
    """Return each response as its place in the participant's own scale use.

    A participant's responses are mapped onto the unit interval by their own
    empirical distribution: a level scores the midpoint of the interval that
    level occupies in it, so a participant who gave one level to every row
    scores one half on every row.

    Parameters
    ----------
    design : JudgmentDesign
        Rows whose recorded responses are scored.
    """
    participants = design.participants
    responses = design.responses
    counts = torch.zeros(
        len(design.participant_levels),
        design.response_levels,
        dtype=torch.float64,
        device=responses.device,
    )
    counts.index_put_(
        (participants, responses),
        torch.ones_like(responses, dtype=torch.float64),
        accumulate=True,
    )
    below = torch.cumsum(counts, dim=1) - counts
    table = (below + counts / 2.0) / counts.sum(dim=1, keepdim=True)

    return table[participants, responses]


def normalized_responses(
    design: JudgmentDesign,
    normalization: ScaleNormalization,
    /,
) -> Tensor:
    """Return the number each row's recorded level is observed as.

    Under the identity the zero-based level is the number itself. Under
    unit-interval scaling it is divided by the highest level, and under ridit
    scaling it is replaced by its place in the participant's own scale use.

    Parameters
    ----------
    design : JudgmentDesign
        Rows whose recorded responses are transformed.
    normalization : ScaleNormalization
        Transformation applied before the response is observed.

    Raises
    ------
    ValueError
        If the normalization is the logarithm, which an ordered level has no
        ratio scale for.
    """
    if normalization is ScaleNormalization.RIDIT:
        return ridit_scores(design)

    levels = design.responses.to(dtype=torch.float64)

    if normalization is ScaleNormalization.IDENTITY:
        return levels

    if normalization is ScaleNormalization.UNIT_INTERVAL:
        return levels / (design.response_levels - 1)

    msg = (
        f"{normalization.value!r} normalization takes the logarithm of a "
        f"ratio-scale response, and an ordered level is not one"
    )

    raise ValueError(msg)


def item_response_weights(
    design: JudgmentDesign,
    normalization: ScaleNormalization = ScaleNormalization.IDENTITY,
    /,
) -> Tensor:
    """Return one response weight per item, in item-index order.

    The identity channel assigns every item weight one, so passing it to an
    objective leaves the objective unchanged. Ridit and unit-interval channels
    first normalize every judgment row by the declared scale and then average
    the normalized rows belonging to each item.

    Parameters
    ----------
    design : JudgmentDesign
        Rows whose judgments supply the item weights.
    normalization : ScaleNormalization
        The channel applied to the judgments. Identity selects the no-op weight
        rather than treating the recorded level itself as a weight.

    Raises
    ------
    ValueError
        If the normalization is logarithmic, which an ordered level has no ratio
        scale for.
    """
    item_count = len(design.item_levels)

    if normalization is ScaleNormalization.IDENTITY:
        return torch.ones(
            item_count,
            dtype=torch.float64,
            device=design.items.device,
        )

    per_row = normalized_responses(design, normalization)
    totals = per_row.new_zeros(item_count).scatter_add(0, design.items, per_row)
    counts = torch.bincount(design.items, minlength=item_count).to(per_row)

    return totals / counts


@final
class MetricParameters:
    """Stores the unconstrained spread of a metric head.

    The head applies softplus to the spread, so the field needs no positivity
    constraint.

    Parameters
    ----------
    spread : Tensor
        Unconstrained spread of the response about the latent score.

    Raises
    ------
    TypeError
        If the spread does not hold floating-point values.
    ValueError
        If the spread is not a scalar.
    """

    __slots__ = ("_spread",)

    def __init__(self, spread: Tensor) -> None:
        if not spread.is_floating_point():
            msg = (
                f"a metric head's spread holds {spread.dtype}, which is not a "
                f"floating-point kind"
            )

            raise TypeError(msg)

        if spread.ndim != 0:
            msg = f"a metric head's spread has rank {spread.ndim}, expected 0"

            raise ValueError(msg)

        object.__setattr__(self, "_spread", spread)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "metric-head parameters are immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Raise AttributeError on field deletion."""
        del name
        msg = "metric-head parameters are immutable"

        raise AttributeError(msg)

    @property
    def spread(self) -> Tensor:
        """Return the unconstrained spread of the response."""
        return self._spread


@final
class MetricHead:
    """A Gaussian head over the normalized response of each row.

    Parameters
    ----------
    spec : ResponseSpec
        Response specification declaring the metric family, the level count,
        the normalization and the predictor terms.
    predictor : LatentPredictor
        Evaluator for those terms.

    Raises
    ------
    TypeError
        If the specification declares another response family.
    ValueError
        If the evaluator implements terms the specification does not declare,
        or the specification declares a normalization this head does not apply.
    """

    __slots__ = ("_levels", "_normalization", "_predictor")

    def __init__(self, spec: ResponseSpec, predictor: LatentPredictor) -> None:
        family = spec.family

        if not isinstance(family, Metric):
            msg = (
                f"a metric head is defined for the metric family alone, and this "
                f"specification declares {family.family!r}"
            )

            raise TypeError(msg)

        if predictor.terms != spec.predictor:
            msg = (
                f"the evaluator implements {predictor.terms!r} against the "
                f"declared terms {spec.predictor!r}"
            )

            raise ValueError(msg)

        if spec.normalization is ScaleNormalization.LOG:
            msg = (
                f"a metric head observes a normalized level, and "
                f"{spec.normalization.value!r} normalization takes the logarithm "
                f"of a ratio-scale response"
            )

            raise ValueError(msg)

        self._levels = spec.levels
        self._normalization = spec.normalization
        self._predictor = predictor

    @property
    def levels(self) -> int:
        """Return the number of ordered response levels."""
        return self._levels

    @property
    def normalization(self) -> ScaleNormalization:
        """Return the transformation the response is observed through."""
        return self._normalization

    @property
    def predictor(self) -> LatentPredictor:
        """Return the evaluator of the declared predictor terms."""
        return self._predictor

    def observations(self, design: JudgmentDesign, /) -> Tensor:
        """Return the number each row's recorded level is observed as.

        Parameters
        ----------
        design : JudgmentDesign
            Dense columns of the judgments to score.

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

        return normalized_responses(design, self._normalization)

    def spread(self, parameters: MetricParameters) -> Tensor:
        """Return the positive spread of the response about the latent score.

        Parameters
        ----------
        parameters : MetricParameters
            Unconstrained parameters of this head.
        """
        return functional.softplus(parameters.spread)

    def _located(self, latent: Tensor, effects: RowEffects | None) -> Tensor:
        """Return the mean of each row's response.

        Parameters
        ----------
        latent : Tensor
            One latent score per response row.
        effects : RowEffects | None
            Random-effect offsets of each response row, if any.

        Raises
        ------
        TypeError
            If the latent score is not of a floating-point kind.
        ValueError
            If the latent score is not one score per row, the effects offset
            another number of rows, or the effects offset cutpoint spans.
        """
        if not latent.is_floating_point():
            msg = (
                f"a metric head's latent score holds {latent.dtype}, which is "
                f"not a floating-point kind"
            )

            raise TypeError(msg)

        if latent.ndim != 1:
            msg = f"a metric head's latent score has rank {latent.ndim}, expected 1"

            raise ValueError(msg)

        if effects is None:
            return latent

        if effects.rows != latent.numel():
            msg = (
                f"{latent.numel()} latent scores were given effects over "
                f"{effects.rows} rows"
            )

            raise ValueError(msg)

        if effects.spans is not None:
            msg = (
                "a metric head has no cutpoints, and these effects offset "
                f"{effects.spans.shape[1]} cutpoint spans"
            )

            raise ValueError(msg)

        return latent + effects.latent

    def latent_scores(self, design: JudgmentDesign, /) -> Tensor:
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
            If the evaluator returns scores of another rank, or one score per
            row of another design.
        """
        latent = self._predictor.latent_scores(design)

        if not latent.is_floating_point():
            msg = (
                f"a predictor's latent score holds {latent.dtype}, which is not "
                f"a floating-point kind"
            )

            raise TypeError(msg)

        if latent.ndim != 1:
            msg = f"a predictor's latent score has rank {latent.ndim}, expected 1"

            raise ValueError(msg)

        if latent.numel() != design.n_rows:
            msg = (
                f"a design of {design.n_rows} rows was scored with "
                f"{latent.numel()} latent scores"
            )

            raise ValueError(msg)

        return latent

    def predictions(
        self,
        design: JudgmentDesign,
        /,
        *,
        effects: RowEffects | None = None,
    ) -> Tensor:
        """Return the response the head expects of each row.

        The expectation is the latent score itself, which is what makes the
        head metric: it predicts a point on the response scale rather than a
        distribution over the levels.

        Parameters
        ----------
        design : JudgmentDesign
            Dense columns of the judgments to score.
        effects : RowEffects | None
            Random-effect offsets of each response row, if any.
        """
        return self._located(self.latent_scores(design), effects)

    def log_likelihood(
        self,
        design: JudgmentDesign,
        parameters: MetricParameters,
        /,
        *,
        effects: RowEffects | None = None,
    ) -> Tensor:
        """Return the log density of the response recorded in each row.

        Parameters
        ----------
        design : JudgmentDesign
            Dense columns of the judgments to score.
        parameters : MetricParameters
            Unconstrained parameters of this head.
        effects : RowEffects | None
            Random-effect offsets of each response row, if any.

        Raises
        ------
        ValueError
            If the design records another number of response levels.
        """
        observed = self.observations(design)
        located = self._located(self.latent_scores(design), effects)
        spread = self.spread(parameters)
        standardized = (observed - located) / spread

        return -0.5 * (standardized**2 + _LOG_TWO_PI) - torch.log(spread)

    @override
    def __repr__(self) -> str:
        """Return the level count and the normalization."""
        return (
            f"MetricHead(levels={self._levels!r}, "
            f"normalization={self._normalization.value!r})"
        )
