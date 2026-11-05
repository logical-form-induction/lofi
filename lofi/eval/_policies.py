"""Label-alignment policies."""

from typing import TYPE_CHECKING

from lofi.eval._alignment import Alignment, SplitLeakError
from lofi.eval._ports import TIE_BREAKS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.eval._alignment import Contingency
    from lofi.eval._ports import TieBreak

__all__ = ["ManyToOne", "OneToOne"]


class ManyToOne:
    """Maps each induced label to its most frequent gold label.

    Parameters
    ----------
    tie_break : TieBreak
        Order for equally frequent gold labels.

    Raises
    ------
    ValueError
        If ``tie_break`` is not declared.
    """

    __slots__ = ("_tie_break",)

    def __init__(self, *, tie_break: TieBreak) -> None:
        self._tie_break: TieBreak = _declared(tie_break)

    @property
    def name(self) -> str:
        """Return the policy name."""
        return "many-to-one"

    @property
    def tie_break(self) -> TieBreak:
        """Return the total order equally scoring gold labels are settled under."""
        return self._tie_break

    @property
    def injective(self) -> bool:
        """Return whether the fitted mapping is injective."""
        return False

    def fit(self, contingency: Contingency, /) -> Alignment:
        """Return the most frequent gold label for each induced label.

        Parameters
        ----------
        contingency : Contingency
            Label co-occurrence counts.

        Returns
        -------
        Alignment
            Fitted many-to-one correspondence.

        Raises
        ------
        SplitLeakError
            If ``contingency`` uses a prohibited fitting split.
        """
        _fittable(contingency)
        marginals = _marginals(contingency)

        return Alignment(
            {
                induced: _best_gold(contingency, induced, marginals, self._tie_break)
                for induced in contingency.induced_labels
                if _row_total(contingency, induced) > 0
            },
            contingency,
        )


class OneToOne:
    """Pairs labels so that each gold label is used at most once.

    Parameters
    ----------
    tie_break : TieBreak
        Order for equally scoring assignments.

    Raises
    ------
    ValueError
        If ``tie_break`` is not declared.
    """

    __slots__ = ("_tie_break",)

    def __init__(self, *, tie_break: TieBreak) -> None:
        self._tie_break: TieBreak = _declared(tie_break)

    @property
    def name(self) -> str:
        """Return the policy name."""
        return "one-to-one"

    @property
    def tie_break(self) -> TieBreak:
        """Return the total order equally scoring assignments are settled under."""
        return self._tie_break

    @property
    def injective(self) -> bool:
        """Return whether the fitted mapping is injective."""
        return True

    def fit(self, contingency: Contingency, /) -> Alignment:
        """Return the optimal one-to-one label assignment.

        Parameters
        ----------
        contingency : Contingency
            Label co-occurrence counts.

        Returns
        -------
        Alignment
            Fitted one-to-one correspondence.

        Raises
        ------
        SplitLeakError
            If ``contingency`` uses a prohibited fitting split.
        """
        _fittable(contingency)
        induced = contingency.induced_labels
        gold = contingency.gold_labels

        if not induced or not gold:
            return Alignment({}, contingency)

        values = _assignment_values(contingency, tie_break=self._tie_break)
        taken = _Assignment(_costs(values)).solve()

        return Alignment(
            {
                induced[row]: gold[column]
                for row, column in enumerate(taken)
                if row < len(induced)
                and column < len(gold)
                and contingency.count(induced[row], gold[column]) > 0
            },
            contingency,
        )


def _fittable(contingency: Contingency) -> None:
    """Refuse counts accumulated on the held-out publication split."""
    if contingency.split == "test":
        msg = "an alignment policy cannot fit on the 'test' split"
        raise SplitLeakError(msg)


def _declared(tie_break: TieBreak) -> TieBreak:
    """Return a declared tie-breaking order.

    Parameters
    ----------
    tie_break : TieBreak
        Candidate tie-breaking order.

    Returns
    -------
    TieBreak
        Validated order.

    Raises
    ------
    ValueError
        If ``tie_break`` is not declared.
    """
    if tie_break not in TIE_BREAKS:
        msg = (
            f"a policy's tie-breaking order is not a declared one: "
            f"{tie_break!r}, which is not in {TIE_BREAKS!r}"
        )
        raise ValueError(msg)

    return tie_break


def _marginals(contingency: Contingency) -> tuple[int, ...]:
    """Return gold-label marginals in label order.

    Parameters
    ----------
    contingency : Contingency
        Label co-occurrence counts.

    Returns
    -------
    tuple[int, ...]
        Gold-label marginal counts.
    """
    return tuple(
        sum(contingency.count(induced, gold) for induced in contingency.induced_labels)
        for gold in contingency.gold_labels
    )


def _row_total(contingency: Contingency, induced: str) -> int:
    """Return one induced label's marginal count.

    Parameters
    ----------
    contingency : Contingency
        Label co-occurrence counts.
    induced : str
        Induced label.

    Returns
    -------
    int
        Induced-label marginal count.
    """
    return sum(contingency.count(induced, gold) for gold in contingency.gold_labels)


def _best_gold(
    contingency: Contingency,
    induced: str,
    marginals: tuple[int, ...],
    tie_break: TieBreak,
) -> str:
    """Return an induced label's gold match after tie-breaking.

    Parameters
    ----------
    contingency : Contingency
        Label co-occurrence counts.
    induced : str
        Induced label.
    marginals : tuple[int, ...]
        Gold-label marginal counts.
    tie_break : TieBreak
        Tie-breaking order.

    Returns
    -------
    str
        Selected gold label.
    """
    gold = contingency.gold_labels
    counts = [contingency.count(induced, label) for label in gold]
    best = max(counts)
    tied = [index for index, count in enumerate(counts) if count == best]

    if tie_break == "largest-gold-marginal":
        return gold[min(tied, key=lambda index: (-marginals[index], index))]

    return gold[min(tied)]


def _assignment_values(
    contingency: Contingency, *, tie_break: TieBreak
) -> list[list[int]]:
    """Return the assignment value matrix.

    Parameters
    ----------
    contingency : Contingency
        Label co-occurrence counts.
    tie_break : TieBreak
        Tie-breaking order encoded in cell values.

    Returns
    -------
    list[list[int]]
        Square maximum-value matrix.
    """
    induced = contingency.induced_labels
    gold = contingency.gold_labels
    size = max(len(induced), len(gold))
    marginals = _marginals(contingency)

    base = len(gold) + 1
    order_scale = base**size
    marginal_scale = order_scale if tie_break == "largest-gold-marginal" else 0
    count_scale = order_scale * (size * contingency.total + 1) + marginal_scale

    return [
        [
            _entry(
                contingency.count(induced[row], gold[column]),
                marginals[column],
                len(gold) - column,
                scales=(count_scale, marginal_scale, base ** (size - 1 - row)),
            )
            if row < len(induced) and column < len(gold)
            else 0
            for column in range(size)
        ]
        for row in range(size)
    ]


def _entry(
    count: int, marginal: int, precedence: int, *, scales: tuple[int, int, int]
) -> int:
    """Return one value-matrix cell.

    Parameters
    ----------
    count : int
        Label-pair count.
    marginal : int
        Gold-label marginal count.
    precedence : int
        Lexical tie-breaking precedence.
    scales : tuple[int, int, int]
        Integer scales for the three priorities.

    Returns
    -------
    int
        Encoded cell value.
    """
    count_scale, marginal_scale, order_scale = scales

    return count * count_scale + marginal * marginal_scale + precedence * order_scale


def _costs(values: Sequence[Sequence[int]]) -> list[list[int]]:
    """Convert maximum values to minimum costs.

    Parameters
    ----------
    values : Sequence[Sequence[int]]
        Square maximum-value matrix.

    Returns
    -------
    list[list[int]]
        Equivalent minimum-cost matrix.
    """
    largest = max(max(row) for row in values)

    return [[largest - value for value in row] for row in values]


class _Assignment:
    """Solves minimum-cost perfect matching.

    Parameters
    ----------
    cost : Sequence[Sequence[int]]
        Square assignment-cost matrix.
    """

    __slots__ = (
        "_column_potential",
        "_cost",
        "_least",
        "_matched",
        "_predecessor",
        "_reached",
        "_row_potential",
        "_size",
        "_used",
    )

    def __init__(self, cost: Sequence[Sequence[int]]) -> None:
        self._cost = cost
        self._size = len(cost)
        self._row_potential = [0] * (self._size + 1)
        self._column_potential = [0] * (self._size + 1)
        self._matched = [0] * (self._size + 1)
        self._least = [0] * (self._size + 1)
        self._predecessor = [0] * (self._size + 1)
        self._used = [False] * (self._size + 1)
        self._reached = [False] * (self._size + 1)

    def solve(self) -> tuple[int, ...]:
        """Return the column selected for each row.

        Returns
        -------
        tuple[int, ...]
            Minimum-cost perfect assignment.
        """
        for row in range(1, self._size + 1):
            self._augment(row)

        taken = [0] * self._size

        for column in range(1, self._size + 1):
            if self._matched[column] != 0:
                taken[self._matched[column] - 1] = column - 1

        return tuple(taken)

    def _augment(self, row: int) -> None:
        """Augment the matching to include one row.

        Parameters
        ----------
        row : int
            One-based row index.
        """
        self._matched[0] = row
        self._least = [0] * (self._size + 1)
        self._predecessor = [0] * (self._size + 1)
        self._used = [False] * (self._size + 1)
        self._reached = [False] * (self._size + 1)
        column = 0

        while True:
            self._used[column] = True
            column = self._step(column)

            if self._matched[column] == 0:
                break

        while column != 0:
            previous = self._predecessor[column]
            self._matched[column] = self._matched[previous]
            column = previous

    def _step(self, column: int) -> int:
        """Relax free columns and return the nearest.

        Parameters
        ----------
        column : int
            Current one-based column index.

        Returns
        -------
        int
            Nearest free-column candidate.
        """
        source = self._matched[column]
        delta = 0
        nearest = 0

        for candidate in range(1, self._size + 1):
            if self._used[candidate]:
                continue

            reduced = (
                self._cost[source - 1][candidate - 1]
                - self._row_potential[source]
                - self._column_potential[candidate]
            )

            if not self._reached[candidate] or reduced < self._least[candidate]:
                self._least[candidate] = reduced
                self._predecessor[candidate] = column
                self._reached[candidate] = True

            if nearest == 0 or self._least[candidate] < delta:
                delta = self._least[candidate]
                nearest = candidate

        self._shift(delta)

        return nearest

    def _shift(self, delta: int) -> None:
        """Update assignment potentials.

        Parameters
        ----------
        delta : int
            Reduced-cost shift.
        """
        for candidate in range(self._size + 1):
            if self._used[candidate]:
                self._row_potential[self._matched[candidate]] += delta
                self._column_potential[candidate] -= delta
            else:
                self._least[candidate] -= delta
