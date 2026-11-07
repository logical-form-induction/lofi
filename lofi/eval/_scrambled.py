"""Scrambled signature assignments and their empirical predictive-fit null."""

from math import factorial, prod
from random import Random
from typing import TYPE_CHECKING, final, override

from lofi._checks import ensure_finite, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.spec._signature import SignatureAssignments

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "EmpiricalTwinNull",
    "SignatureAssignments",
    "TwinFit",
    "empirical_twin_null",
    "sample_exchangeable_signature_twins",
    "sample_signature_twins",
]


@final
class TwinFit(FrozenValue):
    """One signature assignment and its held-out expected log predictive density.

    Parameters
    ----------
    assignments : SignatureAssignments
        Assignment fitted to obtain the predictive density.
    elpd : float
        Finite held-out expected log predictive density.

    Raises
    ------
    TypeError
        If the assignments are not signature assignments or the predictive
        density is not numeric.
    ValueError
        If the predictive density is not finite.
    """

    _noun = "twin fit"

    __slots__ = ("_assignments", "_elpd")

    def __init__(self, assignments: SignatureAssignments, elpd: float) -> None:
        if not isinstance(assignments, SignatureAssignments):
            msg = (
                "a twin fit's assignments are not signature assignments: "
                f"{assignments!r}"
            )
            raise TypeError(msg)

        object.__setattr__(self, "_assignments", assignments)
        object.__setattr__(self, "_elpd", ensure_finite("a twin fit's ELPD", elpd))

    @property
    def assignments(self) -> SignatureAssignments:
        """Return the fitted signature assignments."""
        return self._assignments

    @property
    def elpd(self) -> float:
        """Return the held-out expected log predictive density."""
        return self._elpd

    def _fields(self) -> tuple[SignatureAssignments, float]:
        """Return the assignment and predictive density."""
        return self._assignments, self._elpd

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another fit has the same assignment and density."""
        if type(other) is not TwinFit:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the assignment and predictive density."""
        return hash((TwinFit, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this fit."""
        return f"TwinFit({self._assignments!r}, {self._elpd!r})"


@final
class EmpiricalTwinNull(FrozenValue):
    """An observed fit against an ordered empirical distribution of twin fits.

    Parameters
    ----------
    observed : TwinFit
        Fit under the observed assignments.
    twins : Sequence[TwinFit]
        Distinct non-identity twin fits in draw order.
    declared_twins : int
        Minimum number of twin fits the comparison declares.

    Raises
    ------
    TypeError
        If a fit is not a twin fit or the declared count is not an integer.
    ValueError
        If fewer than three twins are declared, fewer fits are supplied, a twin
        is the identity or a duplicate, its family inventories or parameter
        count differ, or a computed delta is not finite.
    """

    _noun = "empirical twin null"

    __slots__ = ("_declared_twins", "_deltas", "_observed", "_quantile", "_twins")

    def __init__(
        self,
        observed: TwinFit,
        twins: Sequence[TwinFit],
        *,
        declared_twins: int,
    ) -> None:
        if not isinstance(observed, TwinFit):
            msg = (
                f"an empirical twin null's observed fit is not a twin fit: {observed!r}"
            )
            raise TypeError(msg)

        declared = ensure_non_negative(
            "an empirical twin null's declared twin count", declared_twins
        )

        if declared < 3:
            msg = (
                f"an empirical twin null declares fewer than three twins: {declared!r}"
            )
            raise ValueError(msg)

        held = tuple(twins)

        if len(held) < declared:
            msg = (
                f"an empirical twin null has {len(held)} fits for "
                f"{declared} declared twins"
            )
            raise ValueError(msg)

        seen: set[SignatureAssignments] = set()
        deltas: list[float] = []
        below = 0
        tied = 0

        for position, twin in enumerate(held):
            if not isinstance(twin, TwinFit):
                msg = f"empirical twin fit {position} is not a twin fit: {twin!r}"
                raise TypeError(msg)

            _validate_twin(observed.assignments, twin.assignments, position=position)

            if twin.assignments in seen:
                msg = f"empirical twin fit {position} repeats an earlier assignment"
                raise ValueError(msg)

            seen.add(twin.assignments)
            deltas.append(
                ensure_finite(
                    f"empirical twin delta {position}", observed.elpd - twin.elpd
                )
            )

            if twin.elpd < observed.elpd:
                below += 1
            elif twin.elpd == observed.elpd:
                tied += 1

        object.__setattr__(self, "_observed", observed)
        object.__setattr__(self, "_twins", held)
        object.__setattr__(self, "_declared_twins", declared)
        object.__setattr__(self, "_deltas", tuple(deltas))
        object.__setattr__(self, "_quantile", (below + 0.5 * tied) / len(held))

    @property
    def observed(self) -> TwinFit:
        """Return the fit under the observed assignments."""
        return self._observed

    @property
    def twins(self) -> tuple[TwinFit, ...]:
        """Return every twin fit in draw order."""
        return self._twins

    @property
    def declared_twins(self) -> int:
        """Return the minimum declared number of twin fits."""
        return self._declared_twins

    @property
    def deltas(self) -> tuple[float, ...]:
        """Return observed-minus-twin ELPDs in draw order."""
        return self._deltas

    @property
    def quantile(self) -> float:
        """Return the observed ELPD's midrank quantile among the twin ELPDs."""
        return self._quantile

    def _fields(
        self,
    ) -> tuple[TwinFit, tuple[TwinFit, ...], int, tuple[float, ...], float]:
        """Return all retained and computed fields."""
        return (
            self._observed,
            self._twins,
            self._declared_twins,
            self._deltas,
            self._quantile,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another null has the same ordered fields."""
        if type(other) is not EmpiricalTwinNull:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over every retained and computed field."""
        return hash((EmpiricalTwinNull, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this empirical null."""
        return (
            f"EmpiricalTwinNull({self._observed!r}, {self._twins!r}, "
            f"declared_twins={self._declared_twins!r})"
        )


def _validate_twin(
    observed: SignatureAssignments,
    twin: SignatureAssignments,
    *,
    position: int,
) -> None:
    """Validate one twin against the observed signature.

    Parameters
    ----------
    observed : SignatureAssignments
        Assignments whose inventories must be preserved.
    twin : SignatureAssignments
        Candidate non-identity assignments.
    position : int
        Position used in refusal messages.

    Raises
    ------
    ValueError
        If the candidate changes a family inventory or no assignment at all.
    """
    observed_inventories = (
        frozenset(observed.primitives),
        frozenset(observed.constructors),
        frozenset(observed.combinators),
    )
    twin_inventories = (
        frozenset(twin.primitives),
        frozenset(twin.constructors),
        frozenset(twin.combinators),
    )

    if twin_inventories != observed_inventories:
        msg = f"empirical twin fit {position} changes a signature-family inventory"
        raise ValueError(msg)

    if twin == observed:
        msg = f"empirical twin fit {position} is the identity assignment"
        raise ValueError(msg)


def _unrank_permutation(values: tuple[str, ...], rank: int) -> tuple[str, ...]:
    """Return one permutation from its zero-based lexicographic rank.

    Parameters
    ----------
    values : tuple[str, ...]
        Values in identity order.
    rank : int
        Rank below the factorial of the inventory size.

    Returns
    -------
    tuple[str, ...]
        The permutation at that rank.
    """
    remaining = list(values)
    permutation: list[str] = []

    for width in range(len(values), 0, -1):
        block = factorial(width - 1)
        slot, rank = divmod(rank, block)
        permutation.append(remaining.pop(slot))

    return tuple(permutation)


def _assignment_at(observed: SignatureAssignments, rank: int) -> SignatureAssignments:
    """Return the joint within-family permutation at a nonzero rank.

    Parameters
    ----------
    observed : SignatureAssignments
        Identity ordering and family inventories.
    rank : int
        Mixed-radix rank in the complete permutation space.

    Returns
    -------
    SignatureAssignments
        The assignments at that rank.
    """
    primitive_space = factorial(len(observed.primitives))
    constructor_space = factorial(len(observed.constructors))
    primitive_rank = rank % primitive_space
    remaining = rank // primitive_space
    constructor_rank = remaining % constructor_space
    combinator_rank = remaining // constructor_space

    return SignatureAssignments(
        _unrank_permutation(observed.primitives, primitive_rank),
        _unrank_permutation(observed.constructors, constructor_rank),
        _unrank_permutation(observed.combinators, combinator_rank),
    )


def _sample_ranks(population: int, count: int, rng: Random) -> tuple[int, ...]:
    """Return distinct zero-based ranks in exactly ``count`` draws.

    Parameters
    ----------
    population : int
        Finite population size.
    count : int
        Number of distinct ranks to draw.
    rng : Random
        Local random-number stream.

    Returns
    -------
    tuple[int, ...]
        Distinct ranks in draw order.
    """
    remapped: dict[int, int] = {}
    ordered: list[int] = []

    for offset in range(count):
        remaining = population - offset
        selected = rng.randrange(remaining)
        chosen = remapped.get(selected, selected)
        remapped[selected] = remapped.get(remaining - 1, remaining - 1)
        ordered.append(chosen)

    return tuple(ordered)


def _shape_groups(shapes: Sequence[str], size: int, /) -> tuple[tuple[int, ...], ...]:
    """Return positions grouped by their declared exchangeable shape."""
    held = tuple(shapes)

    if len(held) != size:
        msg = f"an exchangeability shape has {len(held)} slots, not {size}"
        raise ValueError(msg)

    groups: dict[str, list[int]] = {}
    for position, shape in enumerate(held):
        groups.setdefault(shape, []).append(position)

    return tuple(tuple(positions) for positions in groups.values())


def _exchangeable_assignment_at(
    observed: SignatureAssignments,
    rank: int,
    primitive_groups: Sequence[Sequence[int]],
    constructor_groups: Sequence[Sequence[int]],
    /,
) -> SignatureAssignments:
    """Return one mixed-radix permutation within exchangeable slot groups."""

    def permute(
        values: tuple[str, ...], groups: Sequence[Sequence[int]]
    ) -> tuple[str, ...]:
        nonlocal rank
        reordered = list(values)

        for group in groups:
            positions = tuple(group)
            space = factorial(len(positions))
            group_rank = rank % space
            rank //= space
            names = _unrank_permutation(
                tuple(values[position] for position in positions), group_rank
            )
            for position, name in zip(positions, names, strict=True):
                reordered[position] = name

        return tuple(reordered)

    primitives = permute(observed.primitives, primitive_groups)
    constructors = permute(observed.constructors, constructor_groups)
    combinators = permute(
        observed.combinators, (tuple(range(len(observed.combinators))),)
    )
    return SignatureAssignments(primitives, constructors, combinators)


def sample_exchangeable_signature_twins(
    observed: SignatureAssignments,
    *,
    primitive_shapes: Sequence[str],
    constructor_shapes: Sequence[str],
    count: int,
    seed: int,
) -> tuple[SignatureAssignments, ...]:
    """Sample twins while permuting primitives and constructors only by shape."""
    if not isinstance(observed, SignatureAssignments):
        msg = f"sampled twin assignments are not signature assignments: {observed!r}"
        raise TypeError(msg)

    requested = ensure_non_negative("a requested twin count", count)
    local_seed = ensure_non_negative("a signature-twin seed", seed)

    if requested < 3:
        msg = f"a signature-twin sample requests fewer than three twins: {requested!r}"
        raise ValueError(msg)

    primitive_groups = _shape_groups(primitive_shapes, len(observed.primitives))
    constructor_groups = _shape_groups(constructor_shapes, len(observed.constructors))
    groups = (
        *primitive_groups,
        *constructor_groups,
        tuple(range(len(observed.combinators))),
    )
    space = prod(factorial(len(group)) for group in groups) - 1

    if requested > space:
        msg = (
            f"a signature has {space} shape-exchangeable non-identity scrambles, "
            f"fewer than the {requested} requested twins"
        )
        raise ValueError(msg)

    ranks = _sample_ranks(space, requested, Random(local_seed))
    return tuple(
        _exchangeable_assignment_at(
            observed, rank + 1, primitive_groups, constructor_groups
        )
        for rank in ranks
    )


def sample_signature_twins(
    observed: SignatureAssignments,
    *,
    count: int,
    seed: int,
) -> tuple[SignatureAssignments, ...]:
    """Sample distinct non-identity within-family signature permutations.

    Parameters
    ----------
    observed : SignatureAssignments
        Identity assignments.
    count : int
        Number of distinct twins, which must be at least three.
    seed : int
        Nonnegative seed for the local random-number stream.

    Returns
    -------
    tuple[SignatureAssignments, ...]
        Sampled twins in draw order.

    Raises
    ------
    TypeError
        If the assignments are not signature assignments or a count is not an
        integer.
    ValueError
        If the count is below three, the seed is negative, or the exact finite
        scramble space cannot supply the requested number of distinct twins.
    """
    if not isinstance(observed, SignatureAssignments):
        msg = f"sampled twin assignments are not signature assignments: {observed!r}"
        raise TypeError(msg)

    requested = ensure_non_negative("a requested twin count", count)
    local_seed = ensure_non_negative("a signature-twin seed", seed)

    if requested < 3:
        msg = f"a signature-twin sample requests fewer than three twins: {requested!r}"
        raise ValueError(msg)

    space = observed.scramble_space

    if requested > space:
        msg = (
            f"a signature has {space} distinct non-identity scrambles, "
            f"fewer than the {requested} requested twins"
        )
        raise ValueError(msg)

    rng = Random(local_seed)
    ranks = _sample_ranks(space, requested, rng)

    return tuple(_assignment_at(observed, rank + 1) for rank in ranks)


def empirical_twin_null(
    observed: TwinFit,
    twins: Sequence[TwinFit],
    *,
    declared_twins: int,
) -> EmpiricalTwinNull:
    """Reduce observed and scrambled-signature fits to their empirical null.

    Parameters
    ----------
    observed : TwinFit
        Fit under the observed assignments.
    twins : Sequence[TwinFit]
        Distinct non-identity twin fits in draw order.
    declared_twins : int
        Minimum number of twins the comparison declares.

    Returns
    -------
    EmpiricalTwinNull
        Every fit, observed-minus-twin delta, and the observed midrank quantile.

    Raises
    ------
    TypeError
        If a fit or the declared count has an invalid type.
    ValueError
        If the fits do not form the declared inventory-matched empirical null.
    """
    return EmpiricalTwinNull(observed, twins, declared_twins=declared_twins)
