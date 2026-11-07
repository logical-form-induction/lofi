"""Synthetic samplers for bounded term types."""

from itertools import product
from random import Random
from typing import TYPE_CHECKING, Final, assert_never, overload

from lofi._checks import ensure_name, ensure_non_negative, ensure_positive
from lofi._frozen import FrozenValue
from lofi.data._sampling import (
    ArityBranching,
    PowerChildDepth,
    SizeExhaustiveWeighting,
    TypeDistribution,
    TypeSampleReport,
    UniformChildDepth,
    UniformExhaustiveWeighting,
    term_depth,
)
from lofi.registry import ComponentPlugin
from lofi.spec.type_samplers.distribution import (
    ArityBranchingSettings,
    PowerChildDepthSettings,
    UniformChildDepthSettings,
)
from lofi.spec.type_samplers.exhaustive import (
    ExhaustiveTypeSamplerSettings,
    ExhaustiveTypeSamplerSpec,
    SizeExhaustiveWeightingSettings,
    UniformExhaustiveWeightingSettings,
)
from lofi.spec.type_samplers.recursive import (
    RecursiveTypeSamplerSettings,
    RecursiveTypeSamplerSpec,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from lofi.data._sampling import ExhaustiveWeighting
    from lofi.terms import Operator, Signature, Sort, Term, TermStore

__all__ = [
    "ExhaustiveTypeInventory",
    "ExhaustiveTypeSampler",
    "RecursiveTypeSampler",
    "TypeSamplerContext",
    "build_arity_branching",
    "build_child_depth",
]


class TypeSamplerContext(FrozenValue):
    """The unified term store and root sort a type sampler reads."""

    _noun = "type sampler context"
    __slots__ = ("_sort", "_store")

    def __init__(self, store: TermStore, sort: Sort) -> None:
        named = ensure_name("a sampled root sort", sort)

        if named not in store.signature.sorts:
            msg = f"sampled sort {named!r} is not in {store.signature.sorts!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_sort", named)

    @property
    def store(self) -> TermStore:
        """Return the store sampled terms are interned in."""
        return self._store

    @property
    def sort(self) -> Sort:
        """Return the required root sort of every sampled term."""
        return self._sort


def build_child_depth(
    settings: UniformChildDepthSettings | PowerChildDepthSettings, /
) -> UniformChildDepth | PowerChildDepth:
    """Build the interior child-depth distribution from validated settings."""
    match settings:
        case UniformChildDepthSettings():
            return UniformChildDepth()
        case PowerChildDepthSettings(exponent=exponent):
            return PowerChildDepth(exponent)
        case unreachable:
            assert_never(unreachable)


def build_arity_branching(settings: ArityBranchingSettings, /) -> ArityBranching:
    """Build immutable per-arity branching weights from validated settings."""
    return ArityBranching({row.arity: row.deepest for row in settings.by_arity})


def _build_exhaustive_weighting(
    settings: UniformExhaustiveWeightingSettings | SizeExhaustiveWeightingSettings,
) -> ExhaustiveWeighting:
    """Build one exhaustive-inventory weighting from validated settings."""
    match settings:
        case UniformExhaustiveWeightingSettings():
            return UniformExhaustiveWeighting()
        case SizeExhaustiveWeightingSettings(exponent=exponent):
            return SizeExhaustiveWeighting(exponent)
        case unreachable:
            assert_never(unreachable)


def _weighted_choice[T](
    random: Random, values: Sequence[T], weights: Sequence[float]
) -> T:
    """Return one value sampled by nonnegative weights."""
    total = sum(weights)

    if total <= 0.0:
        msg = "candidate weights do not contain positive mass"
        raise ValueError(msg)

    point = random.random() * total

    for value, weight in zip(values, weights, strict=True):
        point -= weight

        if point < 0.0:
            return value

    return values[-1]


def _validate_request(
    store: TermStore,
    sort: Sort,
    count: int,
    distribution: TypeDistribution,
) -> int:
    """Return a sampling count validated against its store."""
    held = ensure_non_negative("count", count)
    names = {op.name for op in store.signature.operators}
    unknown = distribution.constructor_weights.keys() - names

    if unknown:
        msg = f"constructor weights name undeclared operators: {sorted(unknown)!r}"
        raise ValueError(msg)

    for term in distribution.bias_types:
        if not store.contains(term):
            msg = f"bias type {term!r} does not belong to the sampler store"
            raise ValueError(msg)

        if term.op.sort != sort:
            msg = f"bias type {term!r} has root sort {term.op.sort!r}, not {sort!r}"
            raise ValueError(msg)

        if term_depth(term) > distribution.max_depth:
            msg = f"bias type {term!r} exceeds max_depth {distribution.max_depth}"
            raise ValueError(msg)

    return held


def _report(terms: Sequence[Term]) -> TypeSampleReport:
    """Return depth and constructor counts for a sampled batch."""
    depths: dict[int, int] = {}
    constructors: dict[str, int] = {}

    def visit(term: Term) -> None:
        constructors[term.op.name] = constructors.get(term.op.name, 0) + 1

        for child in term.args:
            visit(child)

    for term in terms:
        depth = term_depth(term)
        depths[depth] = depths.get(depth, 0) + 1
        visit(term)

    return TypeSampleReport(len(terms), depths, constructors)


def _bias_or_none(random: Random, distribution: TypeDistribution) -> Term | None:
    """Return a reserved-bias term when that mass is selected."""
    if distribution.bias_types and random.random() < distribution.bias_mass:
        return random.choice(distribution.bias_types)

    return None


class ExhaustiveTypeInventory(FrozenValue):
    """A stable finite inventory of payload-free interned terms.

    Parameters
    ----------
    store : TermStore
        Store used to intern every candidate.
    max_size : int
        Inclusive node bound.

    Raises
    ------
    TypeError
        If ``max_size`` is not an integer.
    ValueError
        If ``max_size`` is not positive.
    """

    _noun = "type inventory"

    __slots__ = ("_candidates", "_depths", "_max_size", "_sort", "_store")

    def __init__(self, store: TermStore, max_size: int, *, sort: Sort) -> None:
        held_size = ensure_positive("max_size", max_size)
        context = TypeSamplerContext(store, sort)
        candidates = _enumerate(store, held_size, sort=context.sort)
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_sort", context.sort)
        object.__setattr__(self, "_max_size", held_size)
        object.__setattr__(self, "_candidates", candidates)
        object.__setattr__(self, "_depths", tuple(map(term_depth, candidates)))

    @property
    def store(self) -> TermStore:
        """Return the store that owns every candidate."""
        return self._store

    @property
    def signature(self) -> Signature:
        """Return the candidate term signature."""
        return self._store.signature

    @property
    def sort(self) -> Sort:
        """Return the root sort shared by every candidate."""
        return self._sort

    @property
    def max_size(self) -> int:
        """Return the inclusive node bound."""
        return self._max_size

    @property
    def candidates(self) -> tuple[Term, ...]:
        """Return every candidate in stable inventory order."""
        return self._candidates

    def at_depth(self, depth: int, /) -> tuple[Term, ...]:
        """Return candidates at one exact one-based depth.

        Parameters
        ----------
        depth : int
            Exact root-to-leaf depth.

        Raises
        ------
        TypeError
            If ``depth`` is not an integer.
        ValueError
            If ``depth`` is not positive.
        """
        held = ensure_positive("depth", depth)

        return tuple(
            term
            for term, candidate_depth in zip(
                self._candidates, self._depths, strict=True
            )
            if candidate_depth == held
        )

    def through_depth(self, max_depth: int, /) -> tuple[Term, ...]:
        """Return candidates no deeper than an inclusive bound.

        Parameters
        ----------
        max_depth : int
            Inclusive root-to-leaf depth bound.

        Raises
        ------
        TypeError
            If ``max_depth`` is not an integer.
        ValueError
            If ``max_depth`` is not positive.
        """
        held = ensure_positive("max_depth", max_depth)

        return tuple(
            term
            for term, candidate_depth in zip(
                self._candidates, self._depths, strict=True
            )
            if candidate_depth <= held
        )

    def __iter__(self) -> Iterator[Term]:
        """Return candidates in stable inventory order."""
        return iter(self._candidates)

    def __len__(self) -> int:
        """Return the candidate count."""
        return len(self._candidates)

    @overload
    def __getitem__(self, index: int, /) -> Term: ...

    @overload
    def __getitem__(self, index: slice, /) -> tuple[Term, ...]: ...

    def __getitem__(self, index: int | slice, /) -> Term | tuple[Term, ...]:
        """Return one candidate or a stable candidate slice."""
        return self._candidates[index]


def _enumerate(store: TermStore, max_size: int, *, sort: Sort) -> tuple[Term, ...]:
    """Return unique payload-free terms of one root sort in stable order."""
    by_size_sort: dict[tuple[int, Sort], list[Term]] = {}
    ordered: list[Term] = []
    seen: set[Term] = set()

    for size in range(1, max_size + 1):
        for op in store.signature.operators:
            if op.arity == 0 and size == 1:
                leaves = (
                    (store.op(op),)
                    if not op.has_payload
                    else (
                        tuple(
                            store.op(op, payload=payload)
                            for payload in range(
                                len(store.signature.vocabulary(op.sort))
                            )
                        )
                        if op.is_constant
                        else ()
                    )
                )

                for term in leaves:
                    if term in seen:
                        continue

                    seen.add(term)
                    by_size_sort.setdefault((size, op.sort), []).append(term)

                    if op.sort == sort:
                        ordered.append(term)

                continue

            if op.arity == 0 or size == 1:
                continue

            for sizes in _compositions(size - 1, op.arity):
                pools = tuple(
                    by_size_sort.get((child_size, child_sort), ())
                    for child_size, child_sort in zip(sizes, op.arg_sorts, strict=True)
                )

                if not all(pools):
                    continue

                for args in product(*pools):
                    term = store.op(op, *args)

                    if term in seen:
                        continue

                    seen.add(term)
                    by_size_sort.setdefault((size, op.sort), []).append(term)
                    if op.sort == sort:
                        ordered.append(term)

    return tuple(ordered)


class RecursiveTypeSampler:
    """A seeded recursive sampler for interned terms.

    Parameters
    ----------
    store : TermStore
        Store used to intern sampled terms.
    max_size : int
        Maximum number of nodes in an inventoried candidate.
    """

    __slots__ = ("_inventory",)

    def __init__(self, store: TermStore, max_size: int, *, sort: Sort) -> None:
        self._inventory = ExhaustiveTypeInventory(store, max_size, sort=sort)

    @property
    def inventory(self) -> ExhaustiveTypeInventory:
        """Return the finite candidate universe."""
        return self._inventory

    @property
    def signature(self) -> Signature:
        """Return the sampled term signature."""
        return self._inventory.signature

    def sample(
        self,
        count: int,
        distribution: TypeDistribution,
        /,
        *,
        seed: int,
    ) -> tuple[tuple[Term, ...], TypeSampleReport]:
        """Return a seeded recursive sample and its observed counts.

        Parameters
        ----------
        count : int
            Number of terms to sample.
        distribution : TypeDistribution
            Requested depth, constructor weights, branching skew, and bias.
        seed : int
            Seed for the local random generator.

        Returns
        -------
        tuple[tuple[Term, ...], TypeSampleReport]
            Interned terms and their observed depth and constructor counts.

        Raises
        ------
        TypeError
            If ``count`` is not an integer.
        ValueError
            If the request is invalid for the store or no requested term can
            be drawn.
        """
        store = self._inventory.store
        held_count = _validate_request(
            store,
            self._inventory.sort,
            count,
            distribution,
        )
        random = Random(seed)
        possible = self._possible(distribution.target_depth)
        terms: list[Term] = []

        for _ in range(held_count):
            biased = _bias_or_none(random, distribution)
            terms.append(
                biased
                if biased is not None
                else self._draw(
                    random,
                    distribution,
                    possible,
                    distribution.target_depth,
                    self._inventory.sort,
                )
            )

        result = tuple(terms)

        return result, _report(result)

    def _possible(self, maximum: int) -> dict[tuple[Sort, int], tuple[Operator, ...]]:
        """Return constructors indexed by exact depth and result sort."""
        possible: dict[tuple[Sort, int], tuple[Operator, ...]] = {}

        for depth in range(1, maximum + 1):
            for sort in self.signature.sorts:
                choices = []

                for op in self.signature.operators:
                    if op.sort != sort:
                        continue

                    is_leaf = (
                        depth == 1
                        and op.arity == 0
                        and (
                            not op.has_payload
                            or (
                                op.is_constant
                                and len(self.signature.vocabulary(op.sort)) > 0
                            )
                        )
                    )
                    is_branch = (
                        depth > 1
                        and op.arity > 0
                        and all(
                            any(
                                possible.get((child_sort, child_depth), ())
                                for child_depth in range(1, depth)
                            )
                            for child_sort in op.arg_sorts
                        )
                        and any(
                            possible.get((child_sort, depth - 1), ())
                            for child_sort in op.arg_sorts
                        )
                    )

                    if is_leaf or is_branch:
                        choices.append(op)

                possible[(sort, depth)] = tuple(choices)

        return possible

    def _draw(
        self,
        random: Random,
        distribution: TypeDistribution,
        possible: dict[tuple[Sort, int], tuple[Operator, ...]],
        depth: int,
        sort: Sort | None,
    ) -> Term:
        """Return one term at an exact depth and optional result sort."""
        choices = tuple(
            op
            for candidate_sort in self.signature.sorts
            if sort is None or candidate_sort == sort
            for op in possible.get((candidate_sort, depth), ())
        )

        if not choices:
            qualifier = "any sort" if sort is None else f"sort {sort!r}"
            msg = (
                f"the signature has no payload-free term of depth {depth} "
                f"for {qualifier}"
            )
            raise ValueError(msg)

        weights = tuple(
            distribution.constructor_weights.get(op.name, 1.0) for op in choices
        )
        op = _weighted_choice(random, choices, weights)

        if op.arity == 0:
            payload = (
                random.randrange(len(self.signature.vocabulary(op.sort)))
                if op.is_constant
                else None
            )
            return self._inventory.store.op(op, payload=payload)

        valid_deepest = tuple(
            at
            for at, child_sort in enumerate(op.arg_sorts)
            if possible.get((child_sort, depth - 1), ())
        )
        branching = distribution.branching.weights(op.arity)
        deepest = _weighted_choice(
            random,
            valid_deepest,
            tuple(branching[at] for at in valid_deepest),
        )
        child_depths = []

        for at, child_sort in enumerate(op.arg_sorts):
            if at == deepest:
                child_depths.append(depth - 1)
                continue

            feasible = tuple(
                child_depth
                for child_depth in range(1, depth)
                if possible.get((child_sort, child_depth), ())
            )
            child_depths.append(
                _weighted_choice(
                    random,
                    feasible,
                    distribution.child_depth.weights(feasible),
                )
            )

        args = tuple(
            self._draw(random, distribution, possible, child_depth, child_sort)
            for child_sort, child_depth in zip(op.arg_sorts, child_depths, strict=True)
        )

        return self._inventory.store.op(op, *args)


class ExhaustiveTypeSampler:
    """A seeded sampler over a finite inventory of payload-free terms.

    Parameters
    ----------
    store : TermStore
        Store used to intern sampled terms.
    max_size : int
        Maximum number of nodes in an inventoried term.
    weighting : ExhaustiveWeighting
        Relative mass over candidates in the finite inventory.

    Raises
    ------
    TypeError
        If ``max_size`` is not an integer.
    ValueError
        If ``max_size`` is not positive.
    """

    __slots__ = ("_inventory", "_weighting")

    def __init__(
        self,
        store: TermStore,
        max_size: int,
        *,
        sort: Sort,
        weighting: ExhaustiveWeighting,
    ) -> None:
        self._inventory = ExhaustiveTypeInventory(store, max_size, sort=sort)
        self._weighting = weighting

    @property
    def inventory(self) -> ExhaustiveTypeInventory:
        """Return the finite candidate universe."""
        return self._inventory

    @property
    def signature(self) -> Signature:
        """Return the sampled term signature."""
        return self._inventory.signature

    @property
    def max_size(self) -> int:
        """Return the exhaustive node bound."""
        return self._inventory.max_size

    @property
    def weighting(self) -> ExhaustiveWeighting:
        """Return the inventory weighting."""
        return self._weighting

    def sample(
        self,
        count: int,
        distribution: TypeDistribution,
        /,
        *,
        seed: int,
    ) -> tuple[tuple[Term, ...], TypeSampleReport]:
        """Return a seeded inventory sample and its observed counts.

        Parameters
        ----------
        count : int
            Number of terms to sample.
        distribution : TypeDistribution
            Requested depth, constructor weights, and bias.
        seed : int
            Seed for the local random generator.

        Returns
        -------
        tuple[tuple[Term, ...], TypeSampleReport]
            Interned terms and their observed depth and constructor counts.

        Raises
        ------
        TypeError
            If ``count`` is not an integer.
        ValueError
            If the request is invalid for the store or finite inventory.
        """
        held_count = _validate_request(
            self._inventory.store,
            self._inventory.sort,
            count,
            distribution,
        )
        random = Random(seed)

        for term in distribution.bias_types:
            if term.size > self.max_size:
                msg = f"bias type {term!r} exceeds max_size {self.max_size}"
                raise ValueError(msg)

        candidates = self._inventory.at_depth(distribution.target_depth)

        if held_count and not candidates:
            msg = (
                "the exhaustive inventory has no term at target depth "
                f"{distribution.target_depth}"
            )
            raise ValueError(msg)

        weights = self._weighting.weights(candidates)
        terms = tuple(
            biased
            if (biased := _bias_or_none(random, distribution)) is not None
            else _weighted_choice(random, candidates, weights)
            for _ in range(held_count)
        )

        return terms, _report(terms)


def _compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
    """Return positive integer compositions in lexical order."""
    if parts == 1:
        if total > 0:
            yield (total,)
        return

    for first in range(1, total):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def _build_recursive(
    settings: RecursiveTypeSamplerSettings, context: TypeSamplerContext, /
) -> RecursiveTypeSampler:
    """Build a recursive sampler with its finite candidate inventory.

    Parameters
    ----------
    settings : RecursiveTypeSamplerSettings
        The node bound of the candidate inventory.
    context : TypeSamplerContext
        The store and sort recursive draws use.
    """
    return RecursiveTypeSampler(
        context.store,
        settings.max_size,
        sort=context.sort,
    )


def _build_exhaustive(
    settings: ExhaustiveTypeSamplerSettings, context: TypeSamplerContext, /
) -> ExhaustiveTypeSampler:
    """Build a sampler over its finite exhaustive inventory.

    Parameters
    ----------
    settings : ExhaustiveTypeSamplerSettings
        The node bound of the inventory and the weighting over it.
    context : TypeSamplerContext
        The store and sort inventoried candidates use.
    """
    return ExhaustiveTypeSampler(
        context.store,
        settings.max_size,
        sort=context.sort,
        weighting=_build_exhaustive_weighting(settings.weighting),
    )


RECURSIVE_PLUGIN: Final = ComponentPlugin(
    spec=RecursiveTypeSamplerSpec, build=_build_recursive
)
"""The recursive type sampler's plugin."""

EXHAUSTIVE_PLUGIN: Final = ComponentPlugin(
    spec=ExhaustiveTypeSamplerSpec, build=_build_exhaustive
)
"""The exhaustive type sampler's plugin."""
