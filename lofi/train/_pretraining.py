"""Synthetic pretraining steps over the type and rule-pair samplers."""

from __future__ import annotations

from hashlib import blake2b
from random import Random
from types import MappingProxyType
from typing import TYPE_CHECKING, Never, final, override

from lofi._checks import (
    ensure_name,
    ensure_non_negative,
    ensure_positive,
    refuse_reassignment,
)
from lofi._frozen import FrozenValue
from lofi.data import TypeDistribution, term_depth
from lofi.terms import TreeEditDistance, ball
from lofi.train._optimizing import Optimizing, over_no_span, stepped_result
from lofi.train._steps import EvalResult, StepOutcome

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi.data import (
        ArityBranching,
        ChildDepthDistribution,
        InventoriedTypeSampler,
        PairDistribution,
        PairSampler,
        PairSampleReport,
        SampledPair,
        TypeSampler,
        TypeSampleReport,
    )
    from lofi.judge import Action, ActionSet
    from lofi.terms import BatchEncoding, Term, TermCodec
    from lofi.train._losses import LossAssembly
    from lofi.train._optimization import OptimizationResult
    from lofi.train._ports import OptimizationPolicy
    from lofi.train._steps import TrainResult

__all__ = [
    "ControllerStep",
    "DistanceBatcher",
    "DistanceEncoderStep",
    "Draw",
    "LinearDepthCurriculum",
    "PairDraws",
    "RuleBatch",
    "TermsBatch",
    "TermsStep",
    "Trained",
    "TrainedController",
    "TypeBatch",
    "distance_batch_seed",
    "scheduled_depth",
    "terms_batch_seed",
]


@final
class Draw:
    """Records one batch request.

    Parameters
    ----------
    seed : int
        The seed the sampler draws under.
    count : int
        How many items to draw.
    step : int
        The step of the run the batch trains, counting from zero.
    horizon : int
        How many steps the run has.

    Raises
    ------
    TypeError
        If a field is not an integer.
    ValueError
        If the count or the step is negative, or the horizon is not above zero.
    """

    __slots__ = ("_count", "_horizon", "_seed", "_step")

    def __init__(self, seed: int, count: int, *, step: int, horizon: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            msg = f"a seed is an integer, not {seed!r}"

            raise TypeError(msg)

        if ensure_non_negative("horizon", horizon) == 0:
            msg = "a run of no steps has no step to draw for"

            raise ValueError(msg)

        object.__setattr__(self, "_seed", seed)
        object.__setattr__(self, "_count", ensure_non_negative("count", count))
        object.__setattr__(self, "_step", ensure_non_negative("step", step))
        object.__setattr__(self, "_horizon", horizon)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("draw")

    @property
    def seed(self) -> int:
        """Return the seed."""
        return self._seed

    @property
    def count(self) -> int:
        """Return how many items to draw."""
        return self._count

    @property
    def step(self) -> int:
        """Return the step of the run the batch trains."""
        return self._step

    @property
    def horizon(self) -> int:
        """Return how many steps the run has."""
        return self._horizon

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another draw requests the same batch at the same position."""
        if type(other) is not Draw:
            return NotImplemented

        return (
            self._seed == other._seed
            and self._count == other._count
            and self._step == other._step
            and self._horizon == other._horizon
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the seed, the count and the position."""
        return hash((Draw, self._seed, self._count, self._step, self._horizon))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this draw."""
        return (
            f"Draw({self._seed!r}, {self._count!r}, step={self._step!r}, "
            f"horizon={self._horizon!r})"
        )


class PairDraws:
    """The values a rule-pair step reads.

    Parameters
    ----------
    pair_distribution : PairDistribution
        The distribution the labeled pairs are drawn from.
    draw : Draw
        The batch request.
    """

    __slots__ = ("_draw", "_pair_distribution")

    def __init__(self, pair_distribution: PairDistribution, draw: Draw) -> None:
        object.__setattr__(self, "_pair_distribution", pair_distribution)
        object.__setattr__(self, "_draw", draw)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("pair draw")

    @property
    def pair_distribution(self) -> PairDistribution:
        """Return the distribution the labeled pairs are drawn from."""
        return self._pair_distribution

    @property
    def draw(self) -> Draw:
        """Return the batch request."""
        return self._draw

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value requests the same draw and distribution."""
        if type(other) is not PairDraws:
            return NotImplemented

        return (
            self._pair_distribution == other._pair_distribution
            and self._draw == other._draw
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the distribution and the draw."""
        return hash((PairDraws, self._pair_distribution, self._draw))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this value."""
        return f"PairDraws({self._pair_distribution!r}, {self._draw!r})"


@final
class TypeBatch:
    """A batch of drawn types and the pairs formed among them.

    Parameters
    ----------
    terms : BatchEncoding
        The linearization of every drawn type, in draw order.
    pairs : tuple[tuple[Term, Term], ...]
        The pairs of drawn types, in pair order.
    left : BatchEncoding
        The linearization of the first type of each pair.
    right : BatchEncoding
        The linearization of the second type of each pair.
    """

    __slots__ = ("_left", "_pairs", "_right", "_terms")

    def __init__(
        self,
        terms: BatchEncoding,
        pairs: tuple[tuple[Term, Term], ...],
        left: BatchEncoding,
        right: BatchEncoding,
    ) -> None:
        object.__setattr__(self, "_terms", terms)
        object.__setattr__(self, "_pairs", pairs)
        object.__setattr__(self, "_left", left)
        object.__setattr__(self, "_right", right)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("type batch")

    @property
    def terms(self) -> BatchEncoding:
        """Return the linearized types, one row per drawn type."""
        return self._terms

    @property
    def pairs(self) -> tuple[tuple[Term, Term], ...]:
        """Return the two types of each pair, in pair order."""
        return self._pairs

    @property
    def left(self) -> BatchEncoding:
        """Return the linearization of the first type of each pair."""
        return self._left

    @property
    def right(self) -> BatchEncoding:
        """Return the linearization of the second type of each pair."""
        return self._right

    def __len__(self) -> int:
        """Return how many types the batch holds."""
        return len(self._terms.rows)

    @override
    def __repr__(self) -> str:
        """Return the type and pair counts."""
        return f"TypeBatch(types={len(self)}, pairs={len(self._pairs)})"


@final
class RuleBatch:
    """A batch of labeled rule instances at one step of a run.

    Parameters
    ----------
    instances : tuple[SampledPair[Term, Action], ...]
        Premise tuples, conclusions and admitted actions in batch order.
    step : int
        The step of the run the batch trains, counting from zero.
    horizon : int
        How many steps the run has.
    """

    __slots__ = ("_horizon", "_instances", "_step")

    def __init__(
        self,
        instances: tuple[SampledPair[Term, Action], ...],
        *,
        step: int,
        horizon: int,
    ) -> None:
        object.__setattr__(self, "_instances", instances)
        object.__setattr__(self, "_step", step)
        object.__setattr__(self, "_horizon", horizon)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("rule batch")

    @property
    def instances(self) -> tuple[SampledPair[Term, Action], ...]:
        """Return premise tuples, conclusions and admitted actions in batch order."""
        return self._instances

    @property
    def step(self) -> int:
        """Return the step of the run the batch trains."""
        return self._step

    @property
    def horizon(self) -> int:
        """Return how many steps the run has."""
        return self._horizon

    def __len__(self) -> int:
        """Return how many instances the batch holds."""
        return len(self._instances)

    @override
    def __repr__(self) -> str:
        """Return the instance count and the position in the run."""
        return (
            f"RuleBatch(instances={len(self)}, step={self._step!r}, "
            f"horizon={self._horizon!r})"
        )


@final
class TermsBatch:
    """One autoencoding and combinator batch for a represented space."""

    __slots__ = ("_rules", "_types")

    def __init__(self, types: TypeBatch, rules: RuleBatch) -> None:
        if len(rules) not in (0, len(types)):
            msg = (
                f"a terms batch holds {len(types)} autoencoding terms and "
                f"{len(rules)} combinator pairs"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_types", types)
        object.__setattr__(self, "_rules", rules)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("terms batch")

    @property
    def terms(self) -> BatchEncoding:
        """Return the generated terms used for autoencoding."""
        return self._types.terms

    @property
    def instances(self) -> tuple[SampledPair[Term, Action], ...]:
        """Return the generated rule instances in batch order."""
        return self._rules.instances

    @property
    def step(self) -> int:
        """Return the zero-based optimizer position."""
        return self._rules.step

    @property
    def horizon(self) -> int:
        """Return the optimizer-step horizon."""
        return self._rules.horizon

    def __len__(self) -> int:
        """Return the number of autoencoding terms and combinator pairs."""
        return len(self._types)

    @override
    def __repr__(self) -> str:
        """Return the represented training counts and position."""
        return (
            f"TermsBatch(items={len(self)}, step={self.step}, horizon={self.horizon})"
        )


@final
class Trained:
    """Names the checkpoint role a step trained.

    Parameters
    ----------
    role : str
        The role the trained component's state is checkpointed under.

    Raises
    ------
    TypeError
        If the role is not a string.
    ValueError
        If the role is empty.
    """

    __slots__ = ("_role",)

    def __init__(self, role: str) -> None:
        object.__setattr__(self, "_role", ensure_name("role", role))

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("trained role")

    @property
    def role(self) -> str:
        """Return the checkpoint role."""
        return self._role

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value names the same role."""
        if type(other) is not Trained:
            return NotImplemented

        return self._role == other._role

    @override
    def __hash__(self) -> int:
        """Return a hash over the role."""
        return hash((Trained, self._role))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this value."""
        return f"Trained({self._role!r})"


@final
class TrainedController:
    """Names the checkpoint role a controller step trained, with its label map.

    Parameters
    ----------
    role : str
        The role the controller's state is checkpointed under.
    labels : Mapping[str, int]
        The column each action's weight occupies, keyed by the action's name.

    Raises
    ------
    TypeError
        If the role is not a string.
    ValueError
        If the role is empty.
    """

    __slots__ = ("_labels", "_role")

    def __init__(self, role: str, labels: Mapping[str, int]) -> None:
        object.__setattr__(self, "_role", ensure_name("role", role))
        object.__setattr__(self, "_labels", MappingProxyType(dict(labels)))

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("trained controller")

    @property
    def role(self) -> str:
        """Return the checkpoint role."""
        return self._role

    @property
    def labels(self) -> Mapping[str, int]:
        """Return the column of each action, keyed by the action's name."""
        return self._labels

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value names the same role under the same labels."""
        if type(other) is not TrainedController:
            return NotImplemented

        return self._role == other._role and dict(self._labels) == dict(other._labels)

    @override
    def __hash__(self) -> int:
        """Return a hash over the role and the labels."""
        labels = tuple(sorted(self._labels.items()))

        return hash((TrainedController, self._role, labels))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this value."""
        return f"TrainedController({self._role!r}, {dict(self._labels)!r})"


def _label_map(action_set: ActionSet) -> Mapping[str, int]:
    """Return the column each action occupies, keyed by the action's name.

    Parameters
    ----------
    action_set : ActionSet
        The actions a controller weighs, in its column order.
    """
    return MappingProxyType(
        {action.name: column for column, action in enumerate(action_set.actions)}
    )


def _type_metrics(report: TypeSampleReport) -> dict[str, float]:
    """Return the scalars a type sample reports.

    Parameters
    ----------
    report : TypeSampleReport
        What the sampler observed.

    Returns
    -------
    dict[str, float]
        The draw count, and the mean root depth where anything was drawn.
    """
    metrics = {"sample/draws": float(report.draws)}

    if report.draws:
        weighted = sum(depth * count for depth, count in report.depths.items())
        metrics["sample/mean_depth"] = weighted / report.draws

    return metrics


def _pair_metrics(report: PairSampleReport) -> dict[str, float]:
    """Return the scalars a pair sample reports.

    Parameters
    ----------
    report : PairSampleReport
        What the sampler observed.

    Returns
    -------
    dict[str, float]
        The emitted count and achieved rule-arity proportions.
    """
    return {
        "sample/emitted": float(report.emitted),
        **{
            f"sample/arity/{arity}": proportion
            for arity, proportion in report.arity_distribution.items()
        },
    }


def _type_batch(types: Sequence[Term], codec: TermCodec) -> TypeBatch:
    """Return the batch of drawn types, paired off two at a time in draw order.

    Parameters
    ----------
    types : Sequence[Term]
        The drawn types.
    codec : TermCodec
        What linearizes them.

    Returns
    -------
    TypeBatch
        Every type once among the terms, and the disjoint consecutive pairs, so that a
        trailing unpaired type appears among the terms alone.
    """
    pairs = tuple(zip(types[0::2], types[1::2], strict=False))

    return TypeBatch(
        codec.encode_batch(types),
        pairs,
        codec.encode_batch([left for left, _ in pairs]),
        codec.encode_batch([right for _, right in pairs]),
    )


def distance_batch_seed(stream_seed: int, position: int, /) -> int:
    """Return the deterministic batch seed for one global optimizer position.

    Parameters
    ----------
    stream_seed : int
        Seed of the declared distance-batches stream.
    position : int
        Global optimizer position, counting from zero.
    """
    held_position = ensure_non_negative("optimizer position", position)

    if isinstance(stream_seed, bool) or not isinstance(stream_seed, int):
        msg = f"a stream seed is an integer, not {stream_seed!r}"
        raise TypeError(msg)

    digest = blake2b(
        f"{stream_seed}:{held_position}".encode(),
        digest_size=4,
        person=b"lofi.dist",
    ).digest()

    return int.from_bytes(digest, "big")


def terms_batch_seed(stream_seed: int, position: int, /) -> int:
    """Return the deterministic term-batch seed at one optimizer position."""
    held_position = ensure_non_negative("optimizer position", position)

    if isinstance(stream_seed, bool) or not isinstance(stream_seed, int):
        msg = f"a stream seed is an integer, not {stream_seed!r}"
        raise TypeError(msg)

    digest = blake2b(
        f"{stream_seed}:{held_position}".encode(),
        digest_size=4,
        person=b"lofi.term",
    ).digest()

    return int.from_bytes(digest, "big")


@final
class LinearDepthCurriculum(FrozenValue):
    """Linear growth from one starting depth through a finite inventory."""

    _noun = "linear depth curriculum"

    __slots__ = ("_start_depth", "_steps")

    def __init__(self, start_depth: int, steps: int) -> None:
        object.__setattr__(
            self, "_start_depth", ensure_positive("start_depth", start_depth)
        )
        object.__setattr__(self, "_steps", ensure_positive("curriculum steps", steps))

    @property
    def start_depth(self) -> int:
        """Return the first available inventory depth."""
        return self._start_depth

    @property
    def steps(self) -> int:
        """Return the number of positions over which depth grows."""
        return self._steps


def scheduled_depth(
    depths: Sequence[int],
    position: int,
    curriculum: LinearDepthCurriculum | None,
    /,
) -> int:
    """Return the available inventory depth at one optimizer position.

    Parameters
    ----------
    depths : Sequence[int]
        Nonempty available depths in increasing order.
    position : int
        Global optimizer position, counting from zero.
    curriculum : LinearDepthCurriculum | None
        Growth declaration, or ``None`` to expose the full inventory immediately.
    """
    available = tuple(sorted(set(depths)))

    if not available:
        msg = "a depth curriculum requires a nonempty inventory"
        raise ValueError(msg)

    held_position = ensure_non_negative("optimizer position", position)

    if curriculum is None:
        return available[-1]

    available = tuple(depth for depth in available if depth >= curriculum.start_depth)

    if not available:
        msg = (
            f"depth curriculum starts at {curriculum.start_depth}, beyond the "
            "finite inventory"
        )
        raise ValueError(msg)

    horizon = curriculum.steps

    if horizon == 1:
        return available[-1]

    progress = min(held_position, horizon - 1)
    index = (progress * (len(available) - 1)) // (horizon - 1)

    return available[index]


@final
class DistanceBatcher:
    """Build deterministic type pairs from an inventoried sampler."""

    __slots__ = (
        "_branching",
        "_child_depth",
        "_constructor_weights",
        "_curriculum",
        "_depths",
        "_distance",
        "_radius",
        "_sampler",
    )

    def __init__(
        self,
        sampler: InventoriedTypeSampler,
        *,
        radius: int | None,
        curriculum: LinearDepthCurriculum | None,
        branching: ArityBranching,
        child_depth: ChildDepthDistribution,
        constructor_weights: Mapping[str, float],
    ) -> None:
        if radius is not None:
            ensure_non_negative("radius", radius)

        self._sampler = sampler
        self._radius = radius
        self._curriculum = curriculum
        self._branching = branching
        self._child_depth = child_depth
        self._constructor_weights = MappingProxyType(dict(constructor_weights))
        self._depths = tuple(term_depth(candidate) for candidate in sampler.inventory)
        scheduled_depth(self._depths, 0, curriculum)
        self._distance = TreeEditDistance()

    @property
    def sampler(self) -> InventoriedTypeSampler:
        """Return the installed inventoried sampler."""
        return self._sampler

    @property
    def radius(self) -> int | None:
        """Return the optional closed-ball radius."""
        return self._radius

    @property
    def curriculum(self) -> LinearDepthCurriculum | None:
        """Return the optional linear-growth declaration."""
        return self._curriculum

    def batch(self, codec: TermCodec, draw: Draw, /) -> TypeBatch:
        """Return deterministic pairs for one global optimizer position.

        Parameters
        ----------
        codec : TermCodec
            What linearizes sampled terms.
        draw : Draw
            Pair count, seed, and global position.
        """
        inventory = self._sampler.inventory
        depth = scheduled_depth(self._depths, draw.step, self._curriculum)
        distribution = TypeDistribution(
            depth,
            self._constructor_weights,
            branching=self._branching,
            child_depth=self._child_depth,
            max_depth=depth,
        )

        if self._radius is None:
            sampled, _ = self._sampler.sample(
                2 * draw.count,
                distribution,
                seed=draw.seed,
            )
            pairs = tuple(zip(sampled[0::2], sampled[1::2], strict=True))
        else:
            centers, _ = self._sampler.sample(
                draw.count,
                distribution,
                seed=draw.seed,
            )
            candidates = inventory.through_depth(depth)
            random = Random(draw.seed)
            pairs: list[tuple[Term, Term]] = []

            for center in centers:
                neighbors = ball(
                    center,
                    candidates,
                    self._distance,
                    radius=self._radius,
                )

                if not neighbors:
                    msg = (
                        f"the installed sampler drew {center!r} outside its finite "
                        f"depth-{depth} inventory ball"
                    )
                    raise ValueError(msg)

                pairs.append((center, random.choice(neighbors)))

        held_pairs = tuple(pairs)
        terms = tuple(term for pair in held_pairs for term in pair)

        return TypeBatch(
            codec.encode_batch(terms),
            held_pairs,
            codec.encode_batch([left for left, _ in held_pairs]),
            codec.encode_batch([right for _, right in held_pairs]),
        )


@final
class DistanceEncoderStep:
    """Train a term encoder on one deterministic distance batch."""

    __slots__ = ("_batcher", "_codec", "_name", "_optimizing", "_role")

    def __init__(
        self,
        batcher: DistanceBatcher,
        codec: TermCodec,
        assembly: LossAssembly[TypeBatch],
        policy: OptimizationPolicy[Tensor],
        *,
        role: str,
        name: str | None = None,
    ) -> None:
        self._batcher = batcher
        self._codec = codec
        self._optimizing: Optimizing[TypeBatch] = Optimizing(assembly, policy)
        self._role = ensure_name("role", role)
        self._name = ensure_name("name", _named(name, self._role))

    @property
    def name(self) -> str:
        """Return the step name."""
        return self._name

    @property
    def role(self) -> str:
        """Return the checkpoint role."""
        return self._role

    @property
    def batcher(self) -> DistanceBatcher:
        """Return the deterministic pair batcher."""
        return self._batcher

    @property
    def assembly(self) -> LossAssembly[TypeBatch]:
        """Return the weighted distance objective."""
        return self._optimizing.assembly

    def run(self, draw: Draw, /) -> StepOutcome[Trained, TrainResult]:
        """Build one pair batch and take one optimizer step.

        Parameters
        ----------
        draw : Draw
            Pair count, deterministic seed, and global position.
        """
        batch = self._batcher.batch(self._codec, draw)
        reported, trained = self._optimizing.run(
            over_no_span(batch, len(batch.pairs)), step=draw.step
        )
        metrics = {
            "sample/pairs": float(len(batch.pairs)),
            **trained,
        }

        return StepOutcome(
            Trained(self._role),
            stepped_result(draw.step, len(batch.pairs), reported, metrics),
        )

    def evaluate(self, draw: Draw, /) -> EvalResult:
        """Evaluate one held-out pair batch without taking an optimizer step.

        Parameters
        ----------
        draw : Draw
            Pair count, deterministic seed, and global position.
        """
        batch = self._batcher.batch(self._codec, draw)
        evaluated = self._optimizing.assembly.evaluate(
            over_no_span(batch, len(batch.pairs)), step=draw.step
        )
        return EvalResult(len(batch.pairs), metrics=evaluated.metrics)

    @override
    def __repr__(self) -> str:
        """Return the name, role, and assembly."""
        return (
            f"DistanceEncoderStep(name={self._name!r}, role={self._role!r}, "
            f"assembly={self.assembly!r})"
        )


def _rule_batch(
    sampler: PairSampler[Term, Action],
    inputs: PairDraws,
) -> tuple[RuleBatch, PairSampleReport]:
    """Draw one batch of labeled rule instances.

    Parameters
    ----------
    sampler : PairSampler[Term, Action]
        What draws the instances.
    inputs : PairDraws
        The distribution and the batch request.

    Returns
    -------
    tuple[RuleBatch, PairSampleReport]
        The batch, and what the sampler observed drawing it.
    """
    draw = inputs.draw
    pairs, report = sampler.sample(draw.count, inputs.pair_distribution, seed=draw.seed)
    batch = RuleBatch(pairs, step=draw.step, horizon=draw.horizon)

    return batch, report


@final
class TermsStep:
    """Train a represented space on generated terms and combinator pairs.

    Each run draws one batch of terms of the space's sort under the request's
    seed, and, where the step holds a pair sampler, one batch of rule instances,
    and takes one optimizer step on their joint loss.

    Parameters
    ----------
    types : TypeSampler
        What draws the terms.
    type_distribution : TypeDistribution
        The distribution the terms are drawn from.
    pairs : PairSampler[Term, Action] | None
        What draws the rule instances, or None to train no combinators.
    pair_distribution : PairDistribution | None
        The distribution the rule instances are drawn from, held exactly when
        ``pairs`` is.
    codec : TermCodec
        What linearizes the terms.
    assembly : LossAssembly[TermsBatch]
        The autoencoding and combinator objectives.
    policy : OptimizationPolicy[Tensor]
        What runs the backward pass and the optimizer.
    roles : Sequence[str]
        The component roles the step trains, in checkpoint order.
    name : str
        The step's name.

    Raises
    ------
    ValueError
        If no role is named, a name is empty, or only one of the pair sampler
        and its distribution is held.
    """

    __slots__ = (
        "_assembly",
        "_codec",
        "_name",
        "_optimizing",
        "_pair_distribution",
        "_pairs",
        "_roles",
        "_type_distribution",
        "_types",
    )

    def __init__(
        self,
        types: TypeSampler,
        type_distribution: TypeDistribution,
        pairs: PairSampler[Term, Action] | None,
        pair_distribution: PairDistribution | None,
        codec: TermCodec,
        assembly: LossAssembly[TermsBatch],
        policy: OptimizationPolicy[Tensor],
        *,
        roles: Sequence[str],
        name: str,
    ) -> None:
        held_roles = tuple(ensure_name("a trained term role", role) for role in roles)

        if not held_roles:
            msg = "a terms step trains at least one component role"
            raise ValueError(msg)

        if (pairs is None) != (pair_distribution is None):
            msg = "a terms step holds both a pair sampler and its distribution"
            raise ValueError(msg)

        self._types = types
        self._type_distribution = type_distribution
        self._pairs = pairs
        self._pair_distribution = pair_distribution
        self._codec = codec
        self._assembly = assembly
        self._optimizing: Optimizing[TermsBatch] = Optimizing(assembly, policy)
        self._roles = held_roles
        self._name = ensure_name("name", name)

    @property
    def name(self) -> str:
        """Return the step name."""
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        """Return required value names."""
        return frozenset({"draw"})

    @property
    def produces(self) -> frozenset[str]:
        """Return the trained component roles."""
        return frozenset(self._roles)

    @property
    def roles(self) -> tuple[str, ...]:
        """Return the component roles trained together by this step."""
        return self._roles

    @property
    def assembly(self) -> LossAssembly[TermsBatch]:
        """Return the autoencoding and combinator objectives."""
        return self._assembly

    def _draw(self, draw: Draw, /) -> tuple[TermsBatch, dict[str, float]]:
        """Draw one deterministic generated batch and its sampling metrics."""
        terms, type_report = self._types.sample(
            draw.count,
            self._type_distribution,
            seed=draw.seed,
        )
        type_batch = _type_batch(terms, self._codec)
        metrics = {
            f"terms/{key}": value for key, value in _type_metrics(type_report).items()
        }

        if self._pairs is None or self._pair_distribution is None:
            rule_batch = RuleBatch((), step=draw.step, horizon=draw.horizon)
        else:
            rule_batch, pair_report = _rule_batch(
                self._pairs,
                PairDraws(self._pair_distribution, draw),
            )
            metrics.update(
                (f"pairs/{key}", value)
                for key, value in _pair_metrics(pair_report).items()
            )

        return TermsBatch(type_batch, rule_batch), metrics

    def evaluate(self, draw: Draw, /) -> EvalResult:
        """Evaluate one held-out generated batch without taking an optimizer step."""
        batch, _ = self._draw(draw)
        evaluated = self._assembly.evaluate(
            over_no_span(batch, len(batch)), step=draw.step
        )
        return EvalResult(len(batch), metrics=evaluated.metrics)

    def run(self, draw: Draw, /) -> StepOutcome[tuple[Trained, ...], TrainResult]:
        """Draw one generated batch and take one joint optimizer step.

        Parameters
        ----------
        draw : Draw
            The batch request.

        Returns
        -------
        StepOutcome[tuple[Trained, ...], TrainResult]
            The trained roles, and the result over the drawn terms with the
            sample, optimizer and loss scalars; a request for no terms takes no
            step.
        """
        batch, metrics = self._draw(draw)
        reported: OptimizationResult | None = None

        if len(batch):
            reported, trained = self._optimizing.run(
                over_no_span(batch, len(batch)), step=draw.step
            )
            metrics.update(trained)

        return StepOutcome(
            tuple(Trained(role) for role in self._roles),
            stepped_result(draw.step, len(batch), reported, metrics),
        )

    @override
    def __repr__(self) -> str:
        """Return trained roles and the installed objective assembly."""
        return (
            f"TermsStep(name={self._name!r}, roles={self._roles!r}, "
            f"assembly={self._assembly!r})"
        )


@final
class ControllerStep:
    """Pretrains a controller on labeled rule instances.

    Each run draws one batch of instances under the request's seed, linearizes the
    daughters and the conclusion through the codec, and takes one optimizer step on
    the assembled loss. The label map the checkpoint records is the column order of
    the declared action set.

    Parameters
    ----------
    sampler : PairSampler[Term, Action]
        What draws the instances.
    action_set : ActionSet
        The actions the controller weighs, in its column order.
    assembly : LossAssembly[RuleBatch]
        The weighted objectives the controller trains under.
    policy : OptimizationPolicy[Tensor]
        What runs the backward pass and the optimizer.
    role : str
        The role the controller's state is checkpointed under.
    name : str | None
        The step's name, or None to name it after the role.

    Raises
    ------
    TypeError
        If the role or the name is not a string.
    ValueError
        If the role or the name is empty.
    """

    __slots__ = ("_labels", "_name", "_optimizing", "_role", "_sampler")

    def __init__(
        self,
        sampler: PairSampler[Term, Action],
        action_set: ActionSet,
        assembly: LossAssembly[RuleBatch],
        policy: OptimizationPolicy[Tensor],
        *,
        role: str,
        name: str | None = None,
    ) -> None:
        self._sampler = sampler
        self._labels = _label_map(action_set)
        self._optimizing: Optimizing[RuleBatch] = Optimizing(assembly, policy)
        self._role = ensure_name("role", role)
        self._name = ensure_name("name", _named(name, self._role))

    @property
    def name(self) -> str:
        """Return the step name."""
        return self._name

    @property
    def role(self) -> str:
        """Return the checkpoint role."""
        return self._role

    @property
    def labels(self) -> Mapping[str, int]:
        """Return the column of each action, keyed by the action's name."""
        return self._labels

    @property
    def requires(self) -> frozenset[str]:
        """Return required value names."""
        return frozenset({"pair_distribution", "draw"})

    @property
    def produces(self) -> frozenset[str]:
        """Return produced value names."""
        return frozenset({self._role})

    @property
    def assembly(self) -> LossAssembly[RuleBatch]:
        """Return the weighted objectives."""
        return self._optimizing.assembly

    def evaluate(self, inputs: PairDraws, /) -> EvalResult:
        """Evaluate one held-out generated rule batch without fitting it."""
        batch, _ = _rule_batch(self._sampler, inputs)
        evaluated = self._optimizing.assembly.evaluate(
            over_no_span(batch, len(batch)), step=inputs.draw.step
        )
        return EvalResult(len(batch), metrics=evaluated.metrics)

    def run(self, inputs: PairDraws, /) -> StepOutcome[TrainedController, TrainResult]:
        """Draw one batch of rule instances and take one optimizer step on it.

        Parameters
        ----------
        inputs : PairDraws
            The distribution and the batch request.

        Returns
        -------
        StepOutcome[TrainedController, TrainResult]
            The checkpoint role with the label map, and the result over the drawn
            instances with the sample, optimizer and loss scalars; a request for no
            instances takes no step.
        """
        batch, report = _rule_batch(self._sampler, inputs)
        metrics = _pair_metrics(report)
        reported: OptimizationResult | None = None

        if len(batch):
            reported, trained = self._optimizing.run(
                over_no_span(batch, len(batch)), step=inputs.draw.step
            )
            metrics.update(trained)

        return StepOutcome(
            TrainedController(self._role, self._labels),
            stepped_result(inputs.draw.step, len(batch), reported, metrics),
        )

    @override
    def __repr__(self) -> str:
        """Return the name, the role, the labels and the assembly."""
        return (
            f"ControllerStep(name={self._name!r}, role={self._role!r}, "
            f"labels={dict(self._labels)!r}, assembly={self.assembly!r})"
        )


def _named(name: str | None, role: str) -> str:
    """Return a step's name, which is the role's pretraining where none is given.

    Parameters
    ----------
    name : str | None
        The name given, if any.
    role : str
        The checkpoint role the step trains.
    """
    return f"pretrain:{role}" if name is None else name
