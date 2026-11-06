"""Metrics for induced and reference clusterings."""

from math import comb, exp, lgamma, log
from types import MappingProxyType
from typing import TYPE_CHECKING, override

from lofi._checks import ensure_finite, ensure_name
from lofi.eval._values import MetricResult, Resampling

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.eval._alignment import Contingency

__all__ = [
    "ClusteringScore",
    "adjusted_mutual_information",
    "adjusted_rand",
    "clustering_scores",
    "completeness",
    "homogeneity",
    "v_measure",
]


_INVENTORY = "induced_labels"
"""The component used to report uncorrected score size."""


class ClusteringScore:
    """A clustering score and its chance-correction status.

    Attributes
    ----------
    name : str
        Score name.
    result : MetricResult
        Metric result.
    chance_corrected : bool
        Whether the score subtracts its chance value.

    Parameters
    ----------
    name : str
        Score name.
    result : MetricResult
        Metric result.
    chance_corrected : bool
        Whether the score subtracts its chance value.

    Raises
    ------
    TypeError
        If an input has an invalid type.
    ValueError
        If the name is empty or an uncorrected score omits inventory size.
    """

    __slots__ = ("_chance_corrected", "_name", "_result")

    def __init__(
        self, name: str, result: MetricResult, *, chance_corrected: bool
    ) -> None:
        if not isinstance(result, MetricResult):
            msg = f"a clustering score's result is not one: {result!r}"
            raise TypeError(msg)

        if not isinstance(chance_corrected, bool):
            msg = (
                f"a clustering score's chance-correction flag is not a bool: "
                f"{chance_corrected!r}"
            )
            raise TypeError(msg)

        self._name = ensure_name("a clustering score's name", name)
        self._result = result
        self._chance_corrected = chance_corrected

        if not chance_corrected and _INVENTORY not in result.components:
            msg = (
                f"the score {name!r} is not chance corrected and reports no "
                f"{_INVENTORY!r} component, so its value cannot be read against "
                f"another inventory"
            )
            raise ValueError(msg)

    @property
    def name(self) -> str:
        """Return the score name."""
        return self._name

    @property
    def result(self) -> MetricResult:
        """Return the value, the token count behind it and its decomposition."""
        return self._result

    @property
    def chance_corrected(self) -> bool:
        """Return whether the value was corrected against the marginals alone."""
        return self._chance_corrected

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a score with the same three fields."""
        if type(other) is not ClusteringScore:
            return NotImplemented

        return (
            self._name == other._name
            and self._result == other._result
            and self._chance_corrected == other._chance_corrected
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the name, the result and the flag."""
        return hash((ClusteringScore, self._name, self._result, self._chance_corrected))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this score."""
        return (
            f"ClusteringScore({self._name!r}, {self._result!r}, "
            f"chance_corrected={self._chance_corrected!r})"
        )


def homogeneity(contingency: Contingency, *, unit: str) -> ClusteringScore:
    """Return cluster homogeneity.

    Parameters
    ----------
    contingency : Contingency
        Label co-occurrence counts.
    unit : str
        Resampling-unit name.

    Returns
    -------
    ClusteringScore
        Homogeneity score.

    Raises
    ------
    TypeError
        If an input has an invalid type.
    ValueError
        If ``unit`` is empty.
    """
    counts = _Counts(contingency, unit)

    return counts.score(
        "homogeneity",
        _conditional_share(counts.gold_entropy, counts.gold_given_induced),
        chance_corrected=False,
    )


def completeness(contingency: Contingency, *, unit: str) -> ClusteringScore:
    """Return cluster completeness.

    Parameters
    ----------
    contingency : Contingency
        Label co-occurrence counts.
    unit : str
        Resampling-unit name.

    Returns
    -------
    ClusteringScore
        Completeness score.

    Raises
    ------
    TypeError
        If an input has an invalid type.
    ValueError
        If ``unit`` is empty.
    """
    counts = _Counts(contingency, unit)

    return counts.score(
        "completeness",
        _conditional_share(counts.induced_entropy, counts.induced_given_gold),
        chance_corrected=False,
    )


def v_measure(
    contingency: Contingency, *, unit: str, beta: float = 1.0
) -> ClusteringScore:
    """Return beta-weighted homogeneity and completeness.

    Parameters
    ----------
    contingency : Contingency
        Label co-occurrence counts.
    unit : str
        Resampling-unit name.
    beta : float
        Nonnegative completeness weight.

    Returns
    -------
    ClusteringScore
        V-measure score.

    Raises
    ------
    TypeError
        If an input has an invalid type.
    ValueError
        If ``unit`` is empty or ``beta`` is negative or nonfinite.
    """
    weight = ensure_finite("a v-measure's beta", beta)

    if weight < 0.0:
        msg = f"a v-measure's beta is negative: {beta!r}"
        raise ValueError(msg)

    counts = _Counts(contingency, unit)
    parts = (
        _conditional_share(counts.gold_entropy, counts.gold_given_induced),
        _conditional_share(counts.induced_entropy, counts.induced_given_gold),
    )

    return counts.score(
        "v-measure",
        _weighted_harmonic_mean(parts, weight),
        chance_corrected=False,
        components={"beta": weight, "homogeneity": parts[0], "completeness": parts[1]},
    )


def adjusted_mutual_information(
    contingency: Contingency, *, unit: str
) -> ClusteringScore:
    """Return chance-corrected mutual information.

    Parameters
    ----------
    contingency : Contingency
        Label co-occurrence counts.
    unit : str
        Resampling-unit name.

    Returns
    -------
    ClusteringScore
        Adjusted mutual information.

    Raises
    ------
    TypeError
        If an input has an invalid type.
    ValueError
        If ``unit`` is empty.
    """
    counts = _Counts(contingency, unit)
    observed = counts.mutual_information
    expected = counts.expected_mutual_information
    room = (counts.induced_entropy + counts.gold_entropy) / 2.0 - expected

    return counts.score(
        "adjusted-mutual-information",
        None if room == 0.0 else (observed - expected) / room,
        chance_corrected=True,
        components={"mutual_information": observed, "expected": expected},
    )


def adjusted_rand(contingency: Contingency, *, unit: str) -> ClusteringScore:
    """Return the chance-corrected Rand index.

    Parameters
    ----------
    contingency : Contingency
        Label co-occurrence counts.
    unit : str
        Resampling-unit name.

    Returns
    -------
    ClusteringScore
        Adjusted Rand score.

    Raises
    ------
    TypeError
        If an input has an invalid type.
    ValueError
        If ``unit`` is empty.
    """
    counts = _Counts(contingency, unit)
    pairs = counts.total_pairs
    joined = sum(comb(count, 2) for count in counts.cells)
    induced_pairs = sum(comb(count, 2) for count in counts.induced_totals)
    gold_pairs = sum(comb(count, 2) for count in counts.gold_totals)
    expected = 0.0 if pairs == 0 else induced_pairs * gold_pairs / pairs
    room = (induced_pairs + gold_pairs) / 2.0 - expected

    return counts.score(
        "adjusted-rand",
        None if room == 0.0 else (joined - expected) / room,
        chance_corrected=True,
        components={"joined_pairs": float(joined), "expected": expected},
    )


def clustering_scores(
    contingency: Contingency, *, unit: str, beta: float = 1.0
) -> Mapping[str, ClusteringScore]:
    """Return all clustering scores keyed by name.

    Parameters
    ----------
    contingency : Contingency
        Label co-occurrence counts.
    unit : str
        Resampling-unit name.
    beta : float
        Nonnegative completeness weight for V-measure.

    Returns
    -------
    Mapping[str, ClusteringScore]
        Read-only score mapping.

    Raises
    ------
    TypeError
        If an input has an invalid type.
    ValueError
        If ``unit`` is empty or ``beta`` is negative or nonfinite.
    """
    scores = (
        homogeneity(contingency, unit=unit),
        completeness(contingency, unit=unit),
        v_measure(contingency, unit=unit, beta=beta),
        adjusted_mutual_information(contingency, unit=unit),
        adjusted_rand(contingency, unit=unit),
    )

    return MappingProxyType({score.name: score for score in scores})


def _conditional_share(entropy: float, conditional: float) -> float:
    """Return normalized mutual information.

    Parameters
    ----------
    entropy : float
        Marginal entropy.
    conditional : float
        Conditional entropy.

    Returns
    -------
    float
        Share of marginal entropy explained.
    """
    if entropy == 0.0:
        return 1.0

    return 1.0 - conditional / entropy


def _weighted_harmonic_mean(parts: tuple[float, float], beta: float) -> float:
    """Return a beta-weighted harmonic mean.

    Parameters
    ----------
    parts : tuple[float, float]
        Homogeneity and completeness.
    beta : float
        Completeness weight.

    Returns
    -------
    float
        Weighted harmonic mean.
    """
    first, second = parts
    denominator = beta * first + second

    if denominator == 0.0:
        return 0.0

    return (1.0 + beta) * first * second / denominator


def _entropy(totals: Sequence[int], total: int) -> float:
    """Return marginal entropy in nats.

    Parameters
    ----------
    totals : Sequence[int]
        Marginal counts.
    total : int
        Sum of the counts.

    Returns
    -------
    float
        Entropy in nats.
    """
    if total == 0:
        return 0.0

    return -sum((count / total) * log(count / total) for count in totals if count > 0)


class _Counts:
    """Caches contingency marginals and cells.

    Parameters
    ----------
    contingency : Contingency
        Label co-occurrence counts.
    unit : str
        Resampling-unit name.

    Raises
    ------
    TypeError
        If ``unit`` is not text.
    ValueError
        If ``unit`` is empty.
    """

    __slots__ = (
        "_cells",
        "_gold_totals",
        "_induced_totals",
        "_resampling",
        "_shape",
        "_total",
    )

    def __init__(self, contingency: Contingency, unit: str) -> None:
        induced = contingency.induced_labels
        gold = contingency.gold_labels
        self._cells = tuple(
            contingency.count(row, column) for row in induced for column in gold
        )
        self._induced_totals = tuple(
            sum(contingency.count(row, column) for column in gold) for row in induced
        )
        self._gold_totals = tuple(
            sum(contingency.count(row, column) for row in induced) for column in gold
        )
        self._total = contingency.total
        self._shape = (len(induced), len(gold))
        self._resampling = Resampling(ensure_name("a scored unit", unit))

    @property
    def cells(self) -> tuple[int, ...]:
        """Return every cell of the table, row by row."""
        return self._cells

    @property
    def induced_totals(self) -> tuple[int, ...]:
        """Return how much mass each induced label holds."""
        return self._induced_totals

    @property
    def gold_totals(self) -> tuple[int, ...]:
        """Return how much mass each gold label holds."""
        return self._gold_totals

    @property
    def total_pairs(self) -> int:
        """Return how many unordered pairs of units the table holds."""
        return comb(self._total, 2)

    @property
    def induced_entropy(self) -> float:
        """Return the entropy of the induced labeling, in nats."""
        return _entropy(self._induced_totals, self._total)

    @property
    def gold_entropy(self) -> float:
        """Return the entropy of the gold labeling, in nats."""
        return _entropy(self._gold_totals, self._total)

    @property
    def mutual_information(self) -> float:
        """Return the mutual information of the two labelings, in nats."""
        return self.induced_entropy + self.gold_entropy - self._joint_entropy()

    @property
    def gold_given_induced(self) -> float:
        """Return gold entropy conditioned on the induced labels."""
        return self._joint_entropy() - self.induced_entropy

    @property
    def induced_given_gold(self) -> float:
        """Return induced entropy conditioned on the gold labels."""
        return self._joint_entropy() - self.gold_entropy

    @property
    def expected_mutual_information(self) -> float:
        """Return expected mutual information under fixed marginals.

        Returns
        -------
        float
            Expected mutual information in nats.
        """
        total = self._total

        if total == 0:
            return 0.0

        expected = 0.0

        for induced in self._induced_totals:
            for gold in self._gold_totals:
                expected += _expected_cell(induced, gold, total)

        return expected

    def score(
        self,
        name: str,
        value: float | None,
        *,
        chance_corrected: bool,
        components: Mapping[str, float] | None = None,
    ) -> ClusteringScore:
        """Return a score with inventory sizes.

        Parameters
        ----------
        name : str
            Score name.
        value : float | None
            Score value, or ``None`` when undefined.
        chance_corrected : bool
            Whether the score subtracts its chance value.
        components : Mapping[str, float] | None
            Additional metric components.

        Returns
        -------
        ClusteringScore
            Score with metric metadata.
        """
        induced, gold = self._shape

        return ClusteringScore(
            name,
            MetricResult(
                None if self._total == 0 else value,
                self._total,
                self._resampling,
                components={
                    _INVENTORY: float(induced),
                    "gold_labels": float(gold),
                    **(components or {}),
                },
            ),
            chance_corrected=chance_corrected,
        )

    def _joint_entropy(self) -> float:
        """Return the entropy of the two labelings taken together, in nats."""
        return _entropy(self._cells, self._total)


def _expected_cell(induced: int, gold: int, total: int) -> float:
    """Return one cell's expected mutual-information share.

    Parameters
    ----------
    induced : int
        Induced-label marginal count.
    gold : int
        Gold-label marginal count.
    total : int
        Total item count.

    Returns
    -------
    float
        Expected mutual-information contribution.
    """
    if induced == 0 or gold == 0:
        return 0.0

    constant = (
        lgamma(induced + 1)
        + lgamma(gold + 1)
        + lgamma(total - induced + 1)
        + lgamma(total - gold + 1)
        - lgamma(total + 1)
    )
    share = 0.0

    for count in range(max(1, induced + gold - total), min(induced, gold) + 1):
        weight = exp(
            constant
            - lgamma(count + 1)
            - lgamma(induced - count + 1)
            - lgamma(gold - count + 1)
            - lgamma(total - induced - gold + count + 1)
        )
        share += (
            (count / total) * (log(count) + log(total) - log(induced) - log(gold))
        ) * weight

    return share
