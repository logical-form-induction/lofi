"""Closed-form linear concept erasure and its paired report."""

from collections.abc import Sequence
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING, ClassVar, Final, Literal, override

import torch

from lofi._frozen import FrozenValue
from lofi.stats import Bootstrap, EffectSize, paired_effect_size

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.probe._values import Observation

__all__ = [
    "METRIC_DIRECTIONS",
    "ErasureReport",
    "LinearErasure",
    "MetricDirection",
    "fit_leace",
]


type MetricDirection = Literal["higher", "lower"]
"""Whether larger or smaller metric values mean better performance."""


METRIC_DIRECTIONS: Final[tuple[MetricDirection, ...]] = ("higher", "lower")
"""Declared metric directions."""


def _rank_two(name: str, value: Tensor, /, *, floating: bool) -> Tensor:
    """Return one nonempty finite matrix copied into host float64 storage.

    Parameters
    ----------
    name : str
        The matrix name used in refusal messages.
    value : Tensor
        The matrix to check.
    floating : bool
        Whether the matrix must hold floating-point values.

    Returns
    -------
    Tensor
        An independent host copy in fitting precision.

    Raises
    ------
    TypeError
        If the value is not a tensor or floating-point values are required but absent.
    ValueError
        If the value is not a nonempty finite matrix.
    """
    if not isinstance(value, torch.Tensor):
        msg = f"{name} are not a tensor: {value!r}"
        raise TypeError(msg)

    if value.ndim != 2:
        msg = f"{name} must have rank 2, not shape {tuple(value.shape)}"
        raise ValueError(msg)

    if int(value.shape[0]) == 0 or int(value.shape[1]) == 0:
        msg = f"{name} must be nonempty, not shape {tuple(value.shape)}"
        raise ValueError(msg)

    if floating and not value.is_floating_point():
        msg = f"{name} must be floating point, not {value.dtype}"
        raise TypeError(msg)

    if value.is_complex():
        msg = f"{name} must be real, not {value.dtype}"
        raise TypeError(msg)

    copied = value.detach().to(device="cpu", dtype=torch.float64).clone()

    if not bool(torch.isfinite(copied).all()):
        msg = f"{name} contain a nonfinite value"
        raise ValueError(msg)

    return copied


def _one_hot(concepts: Tensor) -> Tensor:
    """Return a checked one-hot concept matrix in fitting precision.

    Parameters
    ----------
    concepts : Tensor
        One row per representation and one column per concept value.

    Returns
    -------
    Tensor
        An independent host copy in fitting precision.

    Raises
    ------
    TypeError
        If the concepts are not a real tensor.
    ValueError
        If the concepts are empty, nonfinite, not a matrix, or not one-hot.
    """
    checked = _rank_two("concepts", concepts, floating=False)
    binary = (checked == 0.0) | (checked == 1.0)

    if not bool(binary.all()) or not bool((checked.sum(dim=1) == 1.0).all()):
        msg = "concepts must be one-hot, with exactly one unit entry in every row"
        raise ValueError(msg)

    return checked


def _spectral_factors(covariance: Tensor) -> tuple[Tensor, Tensor]:
    """Return the square root and pseudoinverse square root of a covariance.

    Parameters
    ----------
    covariance : Tensor
        A symmetric positive-semidefinite covariance matrix.

    Returns
    -------
    tuple[Tensor, Tensor]
        The square root followed by its Moore-Penrose pseudoinverse.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    tolerance = (
        torch.finfo(torch.float64).eps * max(covariance.shape) * eigenvalues.abs().max()
    )
    positive = eigenvalues > tolerance
    square_roots = torch.zeros_like(eigenvalues)
    inverse_roots = torch.zeros_like(eigenvalues)
    square_roots[positive] = eigenvalues[positive].sqrt()
    inverse_roots[positive] = eigenvalues[positive].rsqrt()

    return (
        eigenvectors @ torch.diag(square_roots) @ eigenvectors.T,
        eigenvectors @ torch.diag(inverse_roots) @ eigenvectors.T,
    )


def _column_projector(matrix: Tensor) -> Tensor:
    """Return the orthogonal projector onto a matrix's column space.

    Parameters
    ----------
    matrix : Tensor
        The matrix whose column space is projected onto.

    Returns
    -------
    Tensor
        A square projector over the matrix's row axis.
    """
    left, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    tolerance = (
        torch.finfo(torch.float64).eps * max(matrix.shape) * singular_values.max()
    )
    basis = left[:, singular_values > tolerance]

    return basis @ basis.T


class LinearErasure(FrozenValue):
    """An affine LEACE map applied over an activation's final axis.

    Parameters
    ----------
    mean : Tensor
        The fitted representation mean.
    projection : Tensor
        The column vector projector, applied transposed to row-shaped activations.

    Raises
    ------
    TypeError
        If either parameter is not a floating-point tensor.
    ValueError
        If the mean and projection have incompatible shapes, contain nonfinite values,
        or the projection is not idempotent.
    """

    __slots__ = ("_mean", "_projection")

    _noun = "linear erasure"

    tag: ClassVar[str] = "linear-erasure"

    def __init__(self, mean: Tensor, projection: Tensor) -> None:
        if not isinstance(mean, torch.Tensor):
            msg = f"a linear erasure's mean is not a tensor: {mean!r}"
            raise TypeError(msg)

        if mean.ndim != 1 or int(mean.shape[0]) == 0:
            msg = f"a linear erasure's mean must have rank 1, not {tuple(mean.shape)}"
            raise ValueError(msg)

        if not mean.is_floating_point():
            msg = f"a linear erasure's mean must be floating point, not {mean.dtype}"
            raise TypeError(msg)

        checked_projection = _rank_two(
            "a linear erasure's projection", projection, floating=True
        )
        checked_mean = mean.detach().to(device="cpu", dtype=torch.float64).clone()

        if not bool(torch.isfinite(checked_mean).all()):
            msg = "a linear erasure's mean contains a nonfinite value"
            raise ValueError(msg)

        width = int(checked_mean.shape[0])

        if tuple(checked_projection.shape) != (width, width):
            msg = (
                f"a linear erasure over {width} coordinates needs a square projection, "
                f"not shape {tuple(checked_projection.shape)}"
            )
            raise ValueError(msg)

        if not torch.allclose(
            checked_projection @ checked_projection,
            checked_projection,
            atol=1e-10,
            rtol=1e-10,
        ):
            msg = "a linear erasure's projection is not idempotent"
            raise ValueError(msg)

        object.__setattr__(self, "_mean", checked_mean)
        object.__setattr__(self, "_projection", checked_projection)

    @property
    def mean(self) -> Tensor:
        """Return an independent copy of the fitted representation mean."""
        return self._mean.clone()

    @property
    def projection(self) -> Tensor:
        """Return an independent copy of the fitted column vector projector."""
        return self._projection.clone()

    @property
    def width(self) -> int:
        """Return the final-axis width this map accepts."""
        return int(self._mean.shape[0])

    def stands_in_for(self, observation: Observation, /) -> Tensor:
        """Return an activation with fitted linear concept information erased.

        Parameters
        ----------
        observation : Observation
            The activation to transform over its final axis.

        Returns
        -------
        Tensor
            An out-of-place affine transformation of the activation, in its original
            shape, dtype, and placement.

        Raises
        ------
        TypeError
            If the source activation is not floating point.
        ValueError
            If the source is scalar, nonfinite, or has the wrong final-axis width.
        """
        source = observation.tensor

        if not source.is_floating_point():
            msg = f"an erasure source must be floating point, not {source.dtype}"
            raise TypeError(msg)

        if source.ndim == 0 or int(source.shape[-1]) != self.width:
            msg = (
                f"an erasure over {self.width} coordinates cannot act on the innermost "
                f"axis of a source of shape {tuple(source.shape)}"
            )
            raise ValueError(msg)

        if not bool(torch.isfinite(source).all()):
            msg = "an erasure source contains a nonfinite value"
            raise ValueError(msg)

        mean = self._mean.to(device=source.device, dtype=source.dtype)
        projection = self._projection.to(device=source.device, dtype=source.dtype)

        return (source - mean) @ projection.T + mean

    @override
    def __repr__(self) -> str:
        """Return the final-axis width this erasure acts over."""
        return f"LinearErasure(width={self.width})"


def fit_leace(representations: Tensor, concepts: Tensor, /) -> LinearErasure:
    """Fit closed-form linear concept erasure in float64 precision.

    Parameters
    ----------
    representations : Tensor
        One representation per row and one activation coordinate per column.
    concepts : Tensor
        One row per representation and one column per concept value, one-hot encoded.

    Returns
    -------
    LinearErasure
        The fitted affine activation intervention.

    Raises
    ------
    TypeError
        If representations are not floating point or either input is not a real tensor.
    ValueError
        If either input is malformed, empty, nonfinite, or the row counts disagree.
    """
    fitted = _rank_two("representations", representations, floating=True)
    encoded = _one_hot(concepts)

    if int(fitted.shape[0]) != int(encoded.shape[0]):
        msg = (
            f"representations have {int(fitted.shape[0])} rows but concepts have "
            f"{int(encoded.shape[0])} rows"
        )
        raise ValueError(msg)

    mean = fitted.mean(dim=0)
    centered = fitted - mean
    centered_concepts = encoded - encoded.mean(dim=0)
    covariance = centered.T @ centered / len(centered)
    cross_covariance = centered.T @ centered_concepts / len(centered)
    square_root, inverse_root = _spectral_factors(covariance)
    concept_space = _column_projector(inverse_root @ cross_covariance)
    projection = (
        torch.eye(int(fitted.shape[1]), dtype=torch.float64)
        - square_root @ concept_space @ inverse_root
    )

    return LinearErasure(mean, projection)


type _ScorePair = tuple[Sequence[float], Sequence[float]]


def _scores(name: str, pair: _ScorePair) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return one named pair of finite, nonempty, aligned score sequences.

    Parameters
    ----------
    name : str
        The comparison name used in refusal messages.
    pair : _ScorePair
        Scores before and after erasure, aligned by item.

    Returns
    -------
    tuple[tuple[float, ...], tuple[float, ...]]
        Immutable copies of the two score sequences.

    Raises
    ------
    TypeError
        If the pair is malformed or a score is not a real number.
    ValueError
        If either sequence is empty, the lengths disagree, or a score is nonfinite.
    """
    if not isinstance(pair, tuple) or len(pair) != 2:
        msg = f"an erasure report's {name} comparison is not a before-after pair"
        raise TypeError(msg)

    copied: list[tuple[float, ...]] = []

    for condition, values in zip(("before", "after"), pair, strict=True):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            msg = (
                f"an erasure report's {name} {condition} scores are not a sequence: "
                f"{values!r}"
            )
            raise TypeError(msg)

        held: list[float] = []

        for position, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, Real):
                msg = (
                    f"an erasure report's {name} {condition} score {position} is not "
                    f"a real number: {value!r}"
                )
                raise TypeError(msg)

            number = float(value)

            if not isfinite(number):
                msg = (
                    f"an erasure report's {name} {condition} score {position} is "
                    f"nonfinite: {number!r}"
                )
                raise ValueError(msg)

            held.append(number)

        if not held:
            msg = f"an erasure report's {name} {condition} scores are empty"
            raise ValueError(msg)

        copied.append(tuple(held))

    before, after = copied

    if len(before) != len(after):
        msg = (
            f"an erasure report's {name} comparison has {len(before)} before scores "
            f"and {len(after)} after scores"
        )
        raise ValueError(msg)

    return before, after


def _direction(name: str, value: str) -> MetricDirection:
    """Return one comparison's declared metric direction.

    Parameters
    ----------
    name : str
        The comparison name used in refusal messages.
    value : str
        The proposed direction.

    Returns
    -------
    MetricDirection
        The direction, unchanged.

    Raises
    ------
    ValueError
        If the value is not a declared direction.
    """
    if value not in METRIC_DIRECTIONS:
        msg = f"an erasure report's {name} metric direction is not declared: {value!r}"
        raise ValueError(msg)

    return value


def _drop(
    name: str,
    pair: _ScorePair,
    direction: MetricDirection,
    bootstrap: Bootstrap,
) -> EffectSize:
    """Return one named paired standardized drop with its bootstrap interval.

    Parameters
    ----------
    name : str
        The comparison name used in refusal messages.
    pair : _ScorePair
        Scores before and after erasure, aligned by item.
    direction : MetricDirection
        Whether larger or smaller values mean better performance. Scores are oriented
        to performance before the drop is estimated, so degradation is positive in
        either direction.
    bootstrap : Bootstrap
        The required resampling configuration.

    Returns
    -------
    EffectSize
        Paired Cliff's delta and its bootstrap result.
    """
    before, after = _scores(name, pair)
    sign = 1.0 if direction == "higher" else -1.0

    return paired_effect_size(
        tuple(sign * value for value in before),
        tuple(sign * value for value in after),
        bootstrap,
        estimator="paired-cliffs-delta",
    )


class ErasureReport(FrozenValue):
    """Three paired drops that separate behavior, decoding, and a control concept.

    Parameters
    ----------
    behavioral : _ScorePair
        Per-item behavioral scores before and after erasure.
    decodability : _ScorePair
        Per-item target-decoding scores before and after erasure.
    control : _ScorePair
        Per-item unrelated-control scores before and after erasure.
    behavioral_direction : MetricDirection
        Whether larger or smaller behavioral scores mean better behavior.
    decodability_direction : MetricDirection
        Whether larger or smaller target-decoding scores mean better decoding.
    control_direction : MetricDirection
        Whether larger or smaller unrelated-control scores mean better decoding.
    bootstrap : Bootstrap
        The required resampling configuration used for all three intervals.

    Raises
    ------
    TypeError
        If the bootstrap or a comparison is malformed.
    ValueError
        If a comparison is empty, misaligned, or nonfinite.
    """

    __slots__ = (
        "_behavioral_direction",
        "_behavioral_drop",
        "_control_direction",
        "_control_drop",
        "_decodability_direction",
        "_decodability_drop",
    )

    _noun = "erasure report"

    def __init__(
        self,
        behavioral: _ScorePair,
        decodability: _ScorePair,
        control: _ScorePair,
        *,
        behavioral_direction: MetricDirection,
        decodability_direction: MetricDirection,
        control_direction: MetricDirection,
        bootstrap: Bootstrap,
    ) -> None:
        if not isinstance(bootstrap, Bootstrap):
            msg = f"an erasure report's bootstrap is not one: {bootstrap!r}"
            raise TypeError(msg)

        behavioral_orientation = _direction("behavioral", behavioral_direction)
        decodability_orientation = _direction("decodability", decodability_direction)
        control_orientation = _direction("control", control_direction)

        object.__setattr__(self, "_behavioral_direction", behavioral_orientation)
        object.__setattr__(self, "_decodability_direction", decodability_orientation)
        object.__setattr__(self, "_control_direction", control_orientation)
        object.__setattr__(
            self,
            "_behavioral_drop",
            _drop("behavioral", behavioral, behavioral_orientation, bootstrap),
        )
        object.__setattr__(
            self,
            "_decodability_drop",
            _drop(
                "decodability",
                decodability,
                decodability_orientation,
                bootstrap,
            ),
        )
        object.__setattr__(
            self,
            "_control_drop",
            _drop("control", control, control_orientation, bootstrap),
        )

    @property
    def behavioral_direction(self) -> MetricDirection:
        """Return which direction means better behavior."""
        return self._behavioral_direction

    @property
    def decodability_direction(self) -> MetricDirection:
        """Return which direction means better target decodability."""
        return self._decodability_direction

    @property
    def control_direction(self) -> MetricDirection:
        """Return which direction means better unrelated-control decodability."""
        return self._control_direction

    @property
    def behavioral_drop(self) -> EffectSize:
        """Return the model-behavior drop after erasure."""
        return self._behavioral_drop

    @property
    def decodability_drop(self) -> EffectSize:
        """Return the erased concept's decodability drop."""
        return self._decodability_drop

    @property
    def control_drop(self) -> EffectSize:
        """Return the unrelated concept's decodability drop."""
        return self._control_drop

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report has the same three paired drops."""
        if type(other) is not ErasureReport:
            return NotImplemented

        return (
            self._behavioral_direction == other._behavioral_direction
            and self._decodability_direction == other._decodability_direction
            and self._control_direction == other._control_direction
            and self._behavioral_drop == other._behavioral_drop
            and self._decodability_drop == other._decodability_drop
            and self._control_drop == other._control_drop
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the three paired drops."""
        return hash(
            (
                ErasureReport,
                self._behavioral_direction,
                self._decodability_direction,
                self._control_direction,
                self._behavioral_drop,
                self._decodability_drop,
                self._control_drop,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the three paired drops in constructor order."""
        return (
            f"ErasureReport({self._behavioral_direction!r}, "
            f"{self._decodability_direction!r}, {self._control_direction!r}, "
            f"{self._behavioral_drop!r}, {self._decodability_drop!r}, "
            f"{self._control_drop!r})"
        )
