"""Model comparison by leave-one-out expected log predictive density.

The substrate's comparison table supplies each model's expected log predictive
density, standard error, effective parameter count, rank and weight, and each
pair's difference, its standard error and the probability that the better-ranked
model is the better one. Its leave-one-out summary supplies the per-observation
Pareto shape of the importance sampling. Every figure is reported at the
precision the table gives it.
"""

from itertools import combinations
from math import isfinite
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, cast, final, override

from pandas import DataFrame
from quivers.diagnostics import compare, loo

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi.judgment import ModelComparisonSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from xarray import DataArray, DataTree

    from lofi._types import JsonValue

__all__ = [
    "WEIGHTING_METHODS",
    "ModelComparison",
    "ModelRow",
    "PairwiseDelta",
    "ParetoGate",
    "WeightingMethod",
    "compare_declared_models",
    "compare_models",
]


class _ParetoLoo(Protocol):
    """A pointwise LOO result exposing Pareto shape estimates."""

    @property
    def pareto_k(self) -> DataArray:
        """Return one Pareto shape estimate per observation."""
        ...


type WeightingMethod = Literal["stacking", "BB-pseudo-BMA", "pseudo-BMA"]
"""A model-weight estimator the substrate offers."""

WEIGHTING_METHODS: tuple[WeightingMethod, ...] = (
    "stacking",
    "BB-pseudo-BMA",
    "pseudo-BMA",
)
"""The model-weight estimators, in the substrate's order."""


@final
class ParetoGate:
    """A reliability rule over the Pareto shape diagnostic of a fit.

    A fit is reliable when the share of its observations whose Pareto shape
    exceeds the threshold is at most the declared fraction.

    Parameters
    ----------
    fraction : float
        Largest share of observations allowed above the threshold.
    threshold : float
        Pareto shape above which an observation's importance weights are
        untrustworthy.

    Raises
    ------
    TypeError
        If a field is not numeric.
    ValueError
        If ``fraction`` is not a proportion or ``threshold`` is not positive.
    """

    __slots__ = ("_fraction", "_threshold")

    def __init__(self, *, fraction: float, threshold: float = 0.7) -> None:
        self._fraction = ensure_finite("a Pareto gate's fraction", fraction)
        self._threshold = ensure_finite("a Pareto gate's threshold", threshold)

        if not 0.0 <= self._fraction <= 1.0:
            msg = f"a Pareto gate's fraction is not a proportion: {fraction!r}"
            raise ValueError(msg)

        if self._threshold <= 0.0:
            msg = f"a Pareto gate's threshold is not positive: {threshold!r}"
            raise ValueError(msg)

    @property
    def fraction(self) -> float:
        """Return the largest share of observations allowed above the threshold."""
        return self._fraction

    @property
    def threshold(self) -> float:
        """Return the Pareto shape above which an observation is untrustworthy."""
        return self._threshold

    def exceeds(self, shape: float) -> bool:
        """Return whether one observation's Pareto shape is untrustworthy.

        Parameters
        ----------
        shape : float
            The observation's estimated Pareto shape. A shape that could not be
            estimated counts as exceeding, since nothing shows its importance
            weights are trustworthy.

        Returns
        -------
        bool
            Whether the shape is above the threshold or not finite.
        """
        return not isfinite(shape) or shape > self._threshold

    def passes(self, exceeding: int, observations: int) -> bool:
        """Return whether a fit with these counts is reliable.

        Parameters
        ----------
        exceeding : int
            Observations whose Pareto shape exceeds the threshold.
        observations : int
            Observations the fit was scored on.

        Returns
        -------
        bool
            Whether the exceeding share is at most the declared fraction.

        Raises
        ------
        TypeError
            If a count is not an integer.
        ValueError
            If a count is negative, there are no observations, or more exceed
            the threshold than were scored.
        """
        above = ensure_non_negative("a fit's exceeding count", exceeding)
        scored = ensure_non_negative("a fit's observation count", observations)

        if scored == 0:
            msg = "a fit scored on no observations has no Pareto shapes"
            raise ValueError(msg)

        if above > scored:
            msg = f"{above} observations exceed the threshold out of {scored}"
            raise ValueError(msg)

        return above / scored <= self._fraction

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a gate with the same two parts."""
        if type(other) is not ParetoGate:
            return NotImplemented

        return self._fraction == other._fraction and self._threshold == other._threshold

    @override
    def __hash__(self) -> int:
        """Return a hash over the fraction and the threshold."""
        return hash((ParetoGate, self._fraction, self._threshold))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this gate."""
        return f"ParetoGate(fraction={self._fraction!r}, threshold={self._threshold!r})"


class ModelRow(NamedTuple):
    """One fitted model's leave-one-out summary."""

    name: str
    rank: int
    elpd: float
    se: float
    effective_parameters: float
    weight: float
    observations: int
    exceeding: int
    reliable: bool

    @property
    def elpd_per_observation(self) -> float:
        """Return expected log predictive density per scored observation.

        Raises
        ------
        TypeError
            If the total density is not numeric or the observation count is
            not an integer.
        ValueError
            If the total density is not finite or the observation count is not
            positive.
        """
        total = ensure_finite("a model row's ELPD", self.elpd)
        observations = ensure_non_negative(
            "a model row's observation count", self.observations
        )

        if observations == 0:
            msg = "a model row scored on no observations"
            raise ValueError(msg)

        return ensure_finite("a model row's ELPD per observation", total / observations)

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the row as serializable report columns."""
        return {
            "model": self.name,
            "rank": self.rank,
            "elpd": self.elpd,
            "elpd_per_observation": self.elpd_per_observation,
            "se": self.se,
            "effective_parameters": self.effective_parameters,
            "weight": self.weight,
            "observations": self.observations,
            "exceeding": self.exceeding,
            "reliable": self.reliable,
        }


class PairwiseDelta(NamedTuple):
    """The leave-one-out difference between two fitted models.

    The difference is signed from the left model: positive when the left model
    has the higher expected log predictive density. When the two densities are
    exactly equal, the difference and its standard error are zero and the left
    model's probability of being better is one half.
    """

    left: str
    right: str
    delta: float
    se: float
    probability_left_better: float
    flagged: bool

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the delta as serializable report columns."""
        return {
            "left": self.left,
            "right": self.right,
            "delta": self.delta,
            "se": self.se,
            "probability_left_better": self.probability_left_better,
            "flagged": self.flagged,
        }


def _ranked(models: tuple[ModelRow, ...]) -> tuple[ModelRow, ...]:
    """Return the model rows, checked to be ranked consecutively from zero."""
    checked = tuple(models)

    for row in checked:
        if not isinstance(row, ModelRow):
            msg = f"a model row is not one: {row!r}"
            raise TypeError(msg)

        _ = row.elpd_per_observation

    if [row.rank for row in checked] != list(range(len(checked))):
        msg = f"model ranks are not consecutive from zero: {checked!r}"
        raise ValueError(msg)

    return checked


def _paired(
    deltas: tuple[PairwiseDelta, ...], models: tuple[ModelRow, ...]
) -> tuple[PairwiseDelta, ...]:
    """Return the deltas, checked to cover each model pair once with the right flag."""
    checked = tuple(deltas)

    for delta in checked:
        if not isinstance(delta, PairwiseDelta):
            msg = f"a pairwise delta is not one: {delta!r}"
            raise TypeError(msg)

    reliable = {row.name: row.reliable for row in models}
    pairs = [(delta.left, delta.right) for delta in checked]

    if pairs != list(combinations(reliable, 2)):
        msg = f"pairwise deltas do not cover the model pairs once: {pairs!r}"
        raise ValueError(msg)

    for delta in checked:
        if delta.flagged != (not reliable[delta.left] or not reliable[delta.right]):
            msg = f"a delta's flag disagrees with its models' reliability: {delta!r}"
            raise ValueError(msg)

    return checked


@final
class ModelComparison:
    """A ranked comparison of fitted models with pairwise differences.

    Parameters
    ----------
    models : tuple[ModelRow, ...]
        One row per model, in rank order.
    deltas : tuple[PairwiseDelta, ...]
        One difference per unordered pair, higher-ranked model on the left.
    gate : ParetoGate
        Reliability rule the rows were judged under.
    method : WeightingMethod
        Estimator the weights came from.

    Raises
    ------
    TypeError
        If a member has an invalid type.
    ValueError
        If ranks are not consecutive from zero, a delta names a model the rows
        do not, a delta's flag disagrees with the rows' reliability, or the
        pairs are not exactly the unordered pairs of models.
    """

    __slots__ = ("_deltas", "_gate", "_method", "_models")

    def __init__(
        self,
        models: tuple[ModelRow, ...],
        deltas: tuple[PairwiseDelta, ...],
        *,
        gate: ParetoGate,
        method: WeightingMethod,
    ) -> None:
        if not isinstance(gate, ParetoGate):
            msg = f"a model comparison's gate is not one: {gate!r}"
            raise TypeError(msg)

        if method not in WEIGHTING_METHODS:
            msg = f"a model comparison's weighting method is unknown: {method!r}"
            raise ValueError(msg)

        self._models = _ranked(models)
        self._deltas = _paired(deltas, self._models)
        self._gate = gate
        self._method: WeightingMethod = method

    @property
    def models(self) -> tuple[ModelRow, ...]:
        """Return the model rows in rank order."""
        return self._models

    @property
    def deltas(self) -> tuple[PairwiseDelta, ...]:
        """Return one difference per unordered pair of models."""
        return self._deltas

    @property
    def gate(self) -> ParetoGate:
        """Return the reliability rule the rows were judged under."""
        return self._gate

    @property
    def method(self) -> WeightingMethod:
        """Return the estimator the weights came from."""
        return self._method

    @property
    def ranking(self) -> tuple[str, ...]:
        """Return the model names from best to worst."""
        return tuple(row.name for row in self._models)

    @property
    def unreliable(self) -> tuple[str, ...]:
        """Return the names of the models the gate rejected, in rank order."""
        return tuple(row.name for row in self._models if not row.reliable)

    def rows(self) -> list[dict[str, JsonValue]]:
        """Return the comparison as long-format report rows.

        Returns
        -------
        list[dict[str, JsonValue]]
            One row per model, then one per pairwise difference, each naming its
            kind, the weighting method and the gate.
        """
        shared: dict[str, JsonValue] = {
            "method": self._method,
            "threshold": self._gate.threshold,
            "fraction": self._gate.fraction,
        }

        return [
            *({"kind": "model", **shared, **row.to_spec()} for row in self._models),
            *({"kind": "delta", **shared, **delta.to_spec()} for delta in self._deltas),
        ]

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a comparison with the same four parts."""
        if type(other) is not ModelComparison:
            return NotImplemented

        return (
            self._models == other._models
            and self._deltas == other._deltas
            and self._gate == other._gate
            and self._method == other._method
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the rows, the deltas, the gate and the method."""
        return hash(
            (ModelComparison, self._models, self._deltas, self._gate, self._method)
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this comparison."""
        return (
            f"ModelComparison({self._models!r}, {self._deltas!r}, "
            f"gate={self._gate!r}, method={self._method!r})"
        )


def _table(
    fits: Mapping[str, DataTree],
    *,
    method: WeightingMethod,
    var_name: str | None,
    reference: str | None,
) -> dict[str, dict[str, float]]:
    """Return the substrate's comparison table as numeric columns keyed by model."""
    table = compare(
        fits,
        method=method,
        var_name=var_name,
        reference=reference,
        round_to="none",
    )

    if not isinstance(table, DataFrame):
        msg = f"the substrate's comparison is not a table: {table!r}"
        raise TypeError(msg)

    return {
        str(name): {
            str(column): float(value)
            for column, value in row.items()
            if not isinstance(value, str)
        }
        for name, row in table.to_dict(orient="index").items()
    }


def _column(row: Mapping[str, float], name: str, column: str) -> float:
    """Return one finite entry of a model's table row."""
    value = row[column]

    if not isfinite(value):
        msg = f"model {name!r} has no finite {column}: {value!r}"
        raise ValueError(msg)

    return value


def _probability_left_better(row: Mapping[str, float], name: str) -> float:
    """Return the probability that a model is better than the pair's reference.

    The substrate leaves the probability blank when the two expected log
    predictive densities are exactly equal, where the normal approximation it
    applies everywhere else gives one half.
    """
    if row["elpd_diff"] == 0.0:
        return 0.5

    return _column(row, name, "p_better")


def _exceeding(
    fit: DataTree, *, gate: ParetoGate, var_name: str | None
) -> tuple[int, int]:
    """Return how many observations the gate finds untrustworthy, and how many exist."""
    scored = cast("_ParetoLoo", loo(fit, pointwise=True, var_name=var_name))
    shapes = [float(shape) for shape in scored.pareto_k.to_numpy().ravel()]

    return sum(1 for shape in shapes if gate.exceeds(shape)), len(shapes)


def compare_models(
    fits: Mapping[str, DataTree],
    *,
    gate: ParetoGate,
    method: WeightingMethod = "stacking",
    var_name: str | None = None,
) -> ModelComparison:
    """Rank fitted models by leave-one-out expected log predictive density.

    The ranks and weights come from one comparison over every fit. Each pairwise
    difference comes from a separate comparison of the two fits with the
    lower-ranked one as reference, so it is signed from the higher-ranked one.

    Parameters
    ----------
    fits : Mapping[str, DataTree]
        Fitted models by name, each with a log-likelihood group.
    gate : ParetoGate
        Reliability rule over each fit's Pareto shapes. A difference involving a
        fit the gate rejects is flagged.
    method : WeightingMethod
        Estimator of the model weights.
    var_name : str | None
        Observed variable to compare on, when a fit's log-likelihood group holds
        more than one.

    Returns
    -------
    ModelComparison
        The ranked rows and every pairwise difference.

    Raises
    ------
    TypeError
        If the gate is not one, a name is not a string, or the substrate returns
        no table.
    ValueError
        If fewer than two fits are named, a name is empty, the method is unknown,
        or a reported figure is not finite.
    """
    if not isinstance(gate, ParetoGate):
        msg = f"a model comparison's gate is not one: {gate!r}"
        raise TypeError(msg)

    if method not in WEIGHTING_METHODS:
        msg = f"a model comparison's weighting method is unknown: {method!r}"
        raise ValueError(msg)

    named = {
        ensure_name("a fitted model's name", name): fit for name, fit in fits.items()
    }

    if len(named) < 2:
        msg = f"a model comparison needs at least two fits and was given {len(named)}"
        raise ValueError(msg)

    table = _table(named, method=method, var_name=var_name, reference=None)
    models: list[ModelRow] = []

    for name, fit in named.items():
        row = table[name]
        exceeding, observations = _exceeding(fit, gate=gate, var_name=var_name)
        models.append(
            ModelRow(
                name=name,
                rank=int(_column(row, name, "rank")),
                elpd=_column(row, name, "elpd"),
                se=_column(row, name, "se"),
                effective_parameters=_column(row, name, "p"),
                weight=_column(row, name, "weight"),
                observations=observations,
                exceeding=exceeding,
                reliable=gate.passes(exceeding, observations),
            )
        )

    models.sort(key=lambda row: row.rank)
    deltas: list[PairwiseDelta] = []

    for left, right in combinations(models, 2):
        pair = _table(
            {left.name: named[left.name], right.name: named[right.name]},
            method=method,
            var_name=var_name,
            reference=right.name,
        )
        row = pair[left.name]
        deltas.append(
            PairwiseDelta(
                left=left.name,
                right=right.name,
                delta=_column(row, left.name, "elpd_diff"),
                se=_column(row, left.name, "dse"),
                probability_left_better=_probability_left_better(row, left.name),
                flagged=not left.reliable or not right.reliable,
            )
        )

    return ModelComparison(tuple(models), tuple(deltas), gate=gate, method=method)


def compare_declared_models(
    fits: Mapping[str, DataTree],
    declaration: ModelComparisonSpec,
    /,
    *,
    var_name: str | None = None,
) -> ModelComparison:
    """Run the predictive comparison declared by a judgment-fit document.

    The declaration is the sole source of the weighting method and both parts
    of the Pareto reliability gate. Its criterion is currently PSIS-LOO, the
    criterion implemented by :func:`compare_models`.
    """
    if not isinstance(declaration, ModelComparisonSpec):
        msg = f"a model-comparison declaration is not one: {declaration!r}"
        raise TypeError(msg)

    named = frozenset(fits)
    declared = frozenset(declaration.candidates)

    if named != declared:
        missing = tuple(sorted(declared - named))
        unexpected = tuple(sorted(named - declared))
        msg = (
            "the fitted models do not match the declared candidates: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
        raise ValueError(msg)

    return compare_models(
        fits,
        gate=ParetoGate(
            fraction=declaration.pareto_fraction,
            threshold=declaration.pareto_threshold,
        ),
        method=declaration.weighting.value,
        var_name=var_name,
    )
