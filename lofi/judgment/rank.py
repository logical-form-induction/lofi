"""Rank-consistent heads over ordered response levels."""

import math
from typing import TYPE_CHECKING, Never, cast, final, override

import torch
from torch.nn import functional

from lofi.judgment._spec import (
    RankConsistent,
    RankConsistentScheme,
    ScaleNormalization,
)
from lofi.judgment.mass import LogitLinking

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.judgment._design import JudgmentDesign
    from lofi.judgment._ports import LatentPredictor
    from lofi.judgment._spec import LinkTerm, ResponseSpec

__all__ = ["RankConsistentHead", "RankConsistentParameters"]

_LOG_TWO = math.log(2.0)


def _log1mexp(value: Tensor) -> Tensor:
    """Return the logarithm of one minus a nonpositive log probability.

    Parameters
    ----------
    value : Tensor
        Nonpositive logarithm whose complement is taken.
    """
    lower = torch.clamp(value, max=-_LOG_TWO)

    return torch.where(
        value > -_LOG_TWO,
        torch.log(-torch.expm1(value)),
        torch.log1p(-torch.exp(lower)),
    )


def _ensure_vector(field: str, value: Tensor) -> None:
    """Check that a parameter is a finite, nonempty floating-point vector.

    Parameters
    ----------
    field : str
        Parameter name used in a refusal message.
    value : Tensor
        Parameter value to check.

    Raises
    ------
    TypeError
        If the value is not floating point.
    ValueError
        If the value is not a finite, nonempty vector.
    """
    if not value.is_floating_point():
        msg = f"{field} holds {value.dtype}, which is not a floating-point kind"

        raise TypeError(msg)

    if value.ndim != 1 or value.numel() == 0:
        msg = f"{field} has shape {tuple(value.shape)!r}, expected a nonempty vector"

        raise ValueError(msg)

    if not bool(torch.isfinite(value).all()):
        msg = f"{field} contains a nonfinite value"

        raise ValueError(msg)


@final
class RankConsistentParameters:
    """The task weights and biases of a rank-consistent head.

    Parameters
    ----------
    weights : Tensor
        One shared task weight for CORAL or one weight per conditional task for
        CORN.
    biases : Tensor
        One bias per boundary between adjacent response levels.

    Raises
    ------
    TypeError
        If a field is not floating point.
    ValueError
        If a field is not a finite, nonempty vector, or the fields disagree
        about floating-point kind or device.
    """

    __slots__ = ("_biases", "_weights")

    def __init__(self, weights: Tensor, biases: Tensor) -> None:
        _ensure_vector("rank-consistent weights", weights)
        _ensure_vector("rank-consistent biases", biases)

        if weights.dtype is not biases.dtype:
            msg = (
                f"rank-consistent weights hold {weights.dtype} and biases hold "
                f"{biases.dtype}"
            )

            raise ValueError(msg)

        if weights.device != biases.device:
            msg = (
                f"rank-consistent weights are on {weights.device} and biases are "
                f"on {biases.device}"
            )

            raise ValueError(msg)

        object.__setattr__(self, "_weights", weights)
        object.__setattr__(self, "_biases", biases)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "rank-consistent parameters are immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Raise AttributeError on field deletion."""
        del name
        msg = "rank-consistent parameters are immutable"

        raise AttributeError(msg)

    @property
    def weights(self) -> Tensor:
        """Return the task weights."""
        return self._weights

    @property
    def biases(self) -> Tensor:
        """Return one bias per response-level boundary."""
        return self._biases

    @property
    def task_count(self) -> int:
        """Return the number of binary tasks carrying biases."""
        return self._biases.numel()


@final
class RankConsistentHead:
    """A CORAL or CORN head whose scores come from a declared predictor.

    Parameters
    ----------
    spec : ResponseSpec
        Response specification declaring a rank-consistent family.
    predictor : LatentPredictor
        Evaluator for the declared predictor terms.

    Raises
    ------
    TypeError
        If the specification declares another response family.
    ValueError
        If the evaluator implements other terms or the response is normalized.
    """

    __slots__ = ("_levels", "_predictor", "_scheme")

    def __init__(self, spec: ResponseSpec, predictor: LatentPredictor) -> None:
        family = spec.family

        if not isinstance(family, RankConsistent):
            msg = (
                "a rank-consistent head is defined for the rank_consistent family "
                f"alone, and this specification declares {family.family!r}"
            )

            raise TypeError(msg)

        if predictor.terms != spec.predictor:
            msg = (
                f"the evaluator implements {predictor.terms!r} against the "
                f"declared terms {spec.predictor!r}"
            )

            raise ValueError(msg)

        if spec.normalization is not ScaleNormalization.IDENTITY:
            msg = (
                "a rank-consistent head classifies the recorded level, and this "
                f"specification declares {spec.normalization.value!r} normalization"
            )

            raise ValueError(msg)

        LogitLinking(cast("LinkTerm", spec.predictor[0]), unit_slope=True)

        self._levels = spec.levels
        self._scheme = family.scheme
        self._predictor = predictor

    @property
    def levels(self) -> int:
        """Return the number of ordered response levels."""
        return self._levels

    @property
    def scheme(self) -> RankConsistentScheme:
        """Return the tying scheme among binary tasks."""
        return self._scheme

    @property
    def predictor(self) -> LatentPredictor:
        """Return the evaluator of the declared predictor terms."""
        return self._predictor

    @property
    def task_count(self) -> int:
        """Return the number of binary tasks between adjacent levels."""
        return self._levels - 1

    def _check(self, parameters: RankConsistentParameters) -> None:
        """Check that parameters implement this head's tasks.

        Parameters
        ----------
        parameters : RankConsistentParameters
            Parameters to pair with this head.

        Raises
        ------
        ValueError
            If the task counts disagree or CORAL biases are not decreasing.
        """
        if parameters.task_count != self.task_count:
            msg = (
                f"a head over {self._levels} levels needs {self.task_count} tasks "
                f"and these parameters give {parameters.task_count}"
            )

            raise ValueError(msg)

        expected_weights = (
            1 if self._scheme is RankConsistentScheme.CORAL else self.task_count
        )

        if parameters.weights.numel() != expected_weights:
            msg = (
                f"{self._scheme.value} over {self.task_count} tasks needs "
                f"{expected_weights} weights and these parameters give "
                f"{parameters.weights.numel()}"
            )

            raise ValueError(msg)

        if self._scheme is RankConsistentScheme.CORAL and not bool(
            torch.all(parameters.biases[:-1] > parameters.biases[1:])
        ):
            msg = "CORAL task biases must be strictly decreasing"

            raise ValueError(msg)

    def _check_design(self, design: JudgmentDesign) -> None:
        """Check that a design records this head's response levels.

        Parameters
        ----------
        design : JudgmentDesign
            Dense columns of the judgments to score.

        Raises
        ------
        ValueError
            If the design records another level count.
        """
        if design.response_levels != self._levels:
            msg = (
                f"a head over {self._levels} levels was given a design over "
                f"{design.response_levels}"
            )

            raise ValueError(msg)

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
            If the evaluator returns another rank or row count.
        """
        self._check_design(design)
        latent = self._predictor.latent_scores(design)

        if not latent.is_floating_point():
            msg = (
                f"a predictor's latent score holds {latent.dtype}, which is not "
                "a floating-point kind"
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

    def _logits(
        self,
        design: JudgmentDesign,
        parameters: RankConsistentParameters,
    ) -> Tensor:
        """Return one binary-task logit per row and response boundary.

        Parameters
        ----------
        design : JudgmentDesign
            Dense columns of the judgments to score.
        parameters : RankConsistentParameters
            Weights and biases of the binary tasks.
        """
        self._check(parameters)
        latent = self.latent_scores(design)

        if latent.dtype is not parameters.weights.dtype:
            msg = (
                f"latent scores hold {latent.dtype} and rank-consistent parameters "
                f"hold {parameters.weights.dtype}"
            )

            raise ValueError(msg)

        if latent.device != parameters.weights.device:
            msg = (
                f"latent scores are on {latent.device} and rank-consistent "
                f"parameters are on {parameters.weights.device}"
            )

            raise ValueError(msg)

        return latent.unsqueeze(1) * parameters.weights + parameters.biases

    def exceedance_probabilities(
        self,
        design: JudgmentDesign,
        parameters: RankConsistentParameters,
        /,
    ) -> Tensor:
        """Return the probability of exceeding each response boundary.

        Parameters
        ----------
        design : JudgmentDesign
            Dense columns of the judgments to score.
        parameters : RankConsistentParameters
            Weights and biases of the binary tasks.
        """
        logits = self._logits(design, parameters)

        if self._scheme is RankConsistentScheme.CORAL:
            return torch.sigmoid(logits)

        return torch.exp(torch.cumsum(functional.logsigmoid(logits), dim=1))

    def log_probabilities(
        self,
        design: JudgmentDesign,
        parameters: RankConsistentParameters,
        /,
    ) -> Tensor:
        """Return the log probability of every response level.

        Parameters
        ----------
        design : JudgmentDesign
            Dense columns of the judgments to score.
        parameters : RankConsistentParameters
            Weights and biases of the binary tasks.
        """
        logits = self._logits(design, parameters)

        if self._scheme is RankConsistentScheme.CORN:
            conditional = functional.logsigmoid(logits)
            previous = torch.cumsum(conditional, dim=1)

            return torch.cat(
                (
                    functional.logsigmoid(-logits[:, :1]),
                    previous[:, :-1] + functional.logsigmoid(-logits[:, 1:]),
                    previous[:, -1:],
                ),
                dim=1,
            )

        below = functional.logsigmoid(logits)
        above = functional.logsigmoid(-logits)
        bias_gaps = parameters.biases[1:] - parameters.biases[:-1]
        middle = below[:, :-1] + above[:, 1:] + _log1mexp(bias_gaps).unsqueeze(0)

        return torch.cat((above[:, :1], middle, below[:, -1:]), dim=1)

    def probabilities(
        self,
        design: JudgmentDesign,
        parameters: RankConsistentParameters,
        /,
    ) -> Tensor:
        """Return the probability of every response level.

        Parameters
        ----------
        design : JudgmentDesign
            Dense columns of the judgments to score.
        parameters : RankConsistentParameters
            Weights and biases of the binary tasks.
        """
        return torch.exp(self.log_probabilities(design, parameters))

    def log_likelihood(
        self,
        design: JudgmentDesign,
        parameters: RankConsistentParameters,
        /,
    ) -> Tensor:
        """Return the log probability of each row's recorded level.

        Parameters
        ----------
        design : JudgmentDesign
            Dense columns of the judgments to score.
        parameters : RankConsistentParameters
            Weights and biases of the binary tasks.
        """
        log_probabilities = self.log_probabilities(design, parameters)

        return log_probabilities.gather(1, design.responses.unsqueeze(1)).squeeze(1)

    @override
    def __repr__(self) -> str:
        """Return the level count and tying scheme."""
        return (
            f"RankConsistentHead(levels={self._levels!r}, "
            f"scheme={self._scheme.value!r})"
        )
