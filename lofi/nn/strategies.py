"""Shared strategies over frontier term decoders."""

from typing import TYPE_CHECKING, NamedTuple

import torch

from lofi._checks import NumericalFailureError, ensure_finite, ensure_positive
from lofi.nn._values import LogDistribution
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.ports import NoAdmissibleChoiceError
from lofi.nn.sampling import SamplingStrategy, sample_from_logits
from lofi.terms import ChoiceIndex, Frontier, HoleContext, TermStore, replace_at

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor

    from lofi.nn._values import Encoded
    from lofi.nn.ports import FrontierDecoder, TeacherForcedDecoder
    from lofi.terms import BatchEncoding, Path, Sort, Term

__all__ = [
    "beam",
    "exact_matches",
    "exhaustive",
    "greedy",
    "paired_exhaustive",
    "sample",
    "score",
]

_CHOICE_AXES = ("choice",)


class _Candidate[S](NamedTuple):
    """One decoder state paired with its partial term and path weight."""

    skeleton: Term
    state: S
    decisions: tuple[int, ...]
    log_weight: Tensor


class _PairedCandidate[LeftS, RightS](NamedTuple):
    """One term frontier carrying two decoders' aligned states and weights."""

    skeleton: Term
    left_state: LeftS
    right_state: RightS
    decisions: tuple[int, ...]
    left_log_weight: Tensor
    right_log_weight: Tensor


def _rank[S](candidate: _Candidate[S]) -> tuple[float, tuple[int, ...]]:
    return -float(candidate.log_weight.detach()), candidate.decisions


def _next_path[S](candidate: _Candidate[S], max_nodes: int, /) -> HoleContext:
    skeleton = candidate.skeleton
    probe = Frontier(skeleton, dict.fromkeys(_hole_paths(skeleton)), 0.0)
    return probe.context(max_nodes)


def _hole_paths(term: Term, /) -> tuple[Path, ...]:
    pending: list[tuple[Term, Path]] = [(term, ())]
    found: list[Path] = []

    while pending:
        node, path = pending.pop()

        if node.op.is_hole:
            found.append(path)
            continue

        pending.extend(
            (child, (*path, side))
            for side, child in reversed(tuple(enumerate(node.args)))
        )

    return tuple(sorted(found, key=lambda path: (len(path), path)))


def _forced(clamp: Term | None, path: Path, index: ChoiceIndex, /) -> int | None:
    if clamp is None:
        return None

    node = clamp

    for side in path:
        if node.op.is_hole:
            return None

        if side >= len(node.args):
            msg = "the clamp has no node at the next frontier path"
            raise ValueError(msg)

        node = node.args[side]

    if node.op.is_hole:
        return None

    return index.index(node.op, node.payload)


def _legal(
    index: ChoiceIndex,
    *,
    context: HoleContext,
    forced: int | None,
) -> tuple[int, ...]:
    return tuple(
        choice
        for choice, admitted in enumerate(index.legal(context))
        if admitted and (forced is None or forced == choice)
    )


def _row(distribution: LogDistribution, index: ChoiceIndex, /) -> Tensor:
    """Return one choice row in which every weight is finite or negative infinity.

    Raises
    ------
    NumericalFailureError
        If a choice weight the mask admits is not a number or is positive infinity.
    ValueError
        If the distribution does not weigh the index's choices.
    """
    if distribution.axes != _CHOICE_AXES:
        msg = f"a frontier decoder weighs {_CHOICE_AXES}, not {distribution.axes}"
        raise ValueError(msg)

    if distribution.normalized_over != _CHOICE_AXES[0]:
        msg = "a frontier decoder normalizes over the choice axis"
        raise ValueError(msg)

    if distribution.size(_CHOICE_AXES[0]) != index.width:
        msg = (
            f"the decoder weighs {distribution.size(_CHOICE_AXES[0])} choices "
            f"beside the index's {index.width}"
        )
        raise ValueError(msg)

    values = distribution.tensor

    if distribution.mask is not None:
        values = torch.where(distribution.mask, values, values.new_full((), -torch.inf))

    diverged = values.isnan() | values.isposinf()

    if bool(diverged.any()):
        quantity = "a frontier decoder's choice weight"
        raise NumericalFailureError(quantity, float(values[diverged][0].detach()))

    return values


def _initial[S](
    decoder: FrontierDecoder[S],
    inputs: Encoded,
    store: TermStore,
    root_sort: Sort,
    /,
) -> _Candidate[S]:
    root = store.hole(root_sort)
    return _Candidate(root, decoder.start(inputs), (), inputs.tensor.new_zeros(()))


def _frontier_store(store: TermStore, /) -> TermStore:
    """Return a store whose private partial terms may contain typed holes."""
    return TermStore(store.signature.with_holes(), max_nodes=store.max_nodes)


def _materialize(source: TermStore, target: TermStore, term: Term, /) -> Term:
    """Intern one completed private-frontier term in the caller's store."""
    arguments = tuple(_materialize(source, target, argument) for argument in term.args)
    operator = target.signature.by_name(term.op.name)

    if term.op.is_free_variable:
        payload = term.payload

        if payload is None:
            msg = f"free variable {term.op.name!r} has no atom"
            raise ValueError(msg)

        return target.fvar(term.op.sort, source.atom_name(payload))

    return target.op(operator, *arguments, payload=term.payload)


def _admitted[S](
    candidate: _Candidate[S],
    decoder: FrontierDecoder[S],
    source: TermStore,
    target: TermStore,
    /,
) -> bool:
    """Return whether a completed private frontier belongs to the decoder language."""
    return decoder.admits(_materialize(source, target, candidate.skeleton))


def _successors[S](
    candidate: _Candidate[S],
    decoder: FrontierDecoder[S],
    index: ChoiceIndex,
    store: TermStore,
    /,
    *,
    max_nodes: int,
    clamp: Term | None,
) -> tuple[_Candidate[S], ...]:
    context = _next_path(candidate, max_nodes)
    legal = _legal(
        index,
        context=context,
        forced=_forced(clamp, context.path, index),
    )
    try:
        row = _row(decoder.weigh(candidate.state, context=context), index)
    except NoAdmissibleChoiceError:
        return ()
    following = []

    for choice in legal:
        value = row[choice]

        if bool(value.isneginf()):
            continue

        following.append(
            _successor(
                candidate,
                decoder,
                index,
                store,
                context=context,
                choice=choice,
                value=value,
            )
        )

    return tuple(following)


def _successor[S](
    candidate: _Candidate[S],
    decoder: FrontierDecoder[S],
    index: ChoiceIndex,
    store: TermStore,
    /,
    *,
    context: HoleContext,
    choice: int,
    value: Tensor,
) -> _Candidate[S]:
    """Advance one frontier through an already-weighed admitted choice."""
    selected = index.choice(choice)
    children = tuple(
        store.hole(child_sort) for child_sort in selected.operator.arg_sorts
    )
    replacement = index.instantiate(store, choice, *children)
    skeleton = replace_at(store, candidate.skeleton, context.path, replacement)
    chosen = torch.tensor(choice, device=value.device)
    return _Candidate(
        skeleton,
        decoder.fill(candidate.state, choices=chosen),
        (*candidate.decisions, choice),
        candidate.log_weight + value,
    )


def _paired_successors[LeftS, RightS](
    candidate: _PairedCandidate[LeftS, RightS],
    left_decoder: FrontierDecoder[LeftS],
    right_decoder: FrontierDecoder[RightS],
    index: ChoiceIndex,
    store: TermStore,
    /,
    *,
    max_nodes: int,
    clamp: Term | None,
) -> tuple[_PairedCandidate[LeftS, RightS], ...]:
    """Advance two decoder states through the same legal symbolic choices."""
    context = _next_path(
        _Candidate(
            candidate.skeleton,
            candidate.left_state,
            candidate.decisions,
            candidate.left_log_weight,
        ),
        max_nodes,
    )
    legal = _legal(
        index,
        context=context,
        forced=_forced(clamp, context.path, index),
    )
    try:
        left_row = _row(
            left_decoder.weigh(candidate.left_state, context=context), index
        )
        right_row = _row(
            right_decoder.weigh(candidate.right_state, context=context), index
        )
    except NoAdmissibleChoiceError:
        return ()
    following = []

    for choice in legal:
        left_value = left_row[choice]
        right_value = right_row[choice]

        if bool(left_value.isneginf()) and bool(right_value.isneginf()):
            continue

        selected = index.choice(choice)
        children = tuple(store.hole(sort) for sort in selected.operator.arg_sorts)
        replacement = index.instantiate(store, choice, *children)
        skeleton = replace_at(store, candidate.skeleton, context.path, replacement)
        left_choice = torch.tensor(choice, device=left_value.device)
        right_choice = torch.tensor(choice, device=right_value.device)
        following.append(
            _PairedCandidate(
                skeleton,
                left_decoder.fill(candidate.left_state, choices=left_choice),
                right_decoder.fill(candidate.right_state, choices=right_choice),
                (*candidate.decisions, choice),
                candidate.left_log_weight + left_value,
                candidate.right_log_weight + right_value,
            )
        )

    return tuple(following)


def _weighted[S](
    candidates: Iterable[_Candidate[S]],
    source: TermStore,
    target: TermStore,
    /,
    *,
    like: Tensor,
    estimate: Estimate,
) -> WeightedTerms:
    ranked = tuple(sorted(candidates, key=_rank))
    terms = tuple(
        _materialize(source, target, candidate.skeleton) for candidate in ranked
    )
    weights = (
        torch.stack(tuple(candidate.log_weight for candidate in ranked))
        if ranked
        else like.new_empty((0,))
    )
    held = weights.exp().sum()
    residual = (1.0 - held).clamp_min(0.0).log()
    return WeightedTerms(terms, weights, residual, estimate)


def _teacher_forced_rows(
    decoder: TeacherForcedDecoder,
    inputs: Encoded,
    terms: BatchEncoding,
    /,
    *,
    index: ChoiceIndex,
) -> LogDistribution:
    """Return validated teacher-forced rows for the supplied terms."""
    rows = decoder.rows(inputs, terms=terms)

    if rows.axes != ("term", "slot", "choice"):
        msg = (
            f"teacher-forced rows run over ('term', 'slot', 'choice'), not {rows.axes}"
        )
        raise ValueError(msg)

    if rows.normalized_over != "choice":
        msg = "teacher-forced rows normalize over the choice axis"
        raise ValueError(msg)

    if rows.size("choice") != index.width:
        msg = (
            f"teacher-forced rows weigh {rows.size('choice')} choices beside the "
            f"scoring index's {index.width}"
        )
        raise ValueError(msg)

    return rows


def _term_choices(
    rows: LogDistribution,
    terms: BatchEncoding,
    index: ChoiceIndex,
    /,
) -> tuple[Tensor, Tensor]:
    """Return aligned payload-aware choices and occupied slots."""
    choices = torch.tensor(
        tuple(index.codes(row) for row in terms.rows),
        device=rows.tensor.device,
    ).reshape(len(terms.rows), terms.width)
    occupied = torch.tensor(terms.mask, device=rows.tensor.device)

    if tuple(choices.shape) != tuple(rows.tensor.shape[:2]):
        msg = (
            f"the terms occupy shape {tuple(choices.shape)} beside teacher rows "
            f"of shape {tuple(rows.tensor.shape[:2])}"
        )
        raise ValueError(msg)

    return choices, occupied


def exact_matches(
    decoder: TeacherForcedDecoder,
    inputs: Encoded,
    terms: BatchEncoding,
    /,
    *,
    index: ChoiceIndex,
) -> Tensor:
    """Return whether each supplied term is the teacher-forced row argmax."""
    rows = _teacher_forced_rows(decoder, inputs, terms, index=index)

    if not terms.rows:
        return torch.ones(0, dtype=torch.bool, device=rows.tensor.device)

    choices, occupied = _term_choices(rows, terms, index)
    return ((rows.tensor.detach().argmax(dim=-1) == choices) | ~occupied).all(dim=-1)


def score(
    decoder: TeacherForcedDecoder,
    inputs: Encoded,
    terms: BatchEncoding,
    /,
    *,
    index: ChoiceIndex,
) -> LogDistribution:
    """Return one shared decoder log weight for each supplied term.

    The teacher-forced decoder owns the conditional rows. This function alone
    selects each term's choices and sums its occupied slots, so consumers cannot
    silently disagree about the definition of a term weight.
    """
    rows = _teacher_forced_rows(decoder, inputs, terms, index=index)

    if not terms.rows:
        return LogDistribution(rows.tensor.sum(dim=(1, 2)), ("term",))

    choices, occupied = _term_choices(rows, terms, index)

    selected = rows.tensor.gather(2, choices.unsqueeze(-1)).squeeze(-1)
    weights = torch.where(occupied, selected, selected.new_zeros(())).sum(dim=1)
    mask = rows.mask

    if mask is not None and mask.ndim > 1:
        mask = mask.any(dim=tuple(range(1, mask.ndim)))

    return LogDistribution(weights, ("term",), mask)


def beam[S](
    decoder: FrontierDecoder[S],
    inputs: Encoded,
    /,
    *,
    index: ChoiceIndex,
    store: TermStore,
    root_sort: Sort,
    max_nodes: int,
    width: int,
    clamp: Term | None = None,
) -> WeightedTerms:
    """Return the highest-weight completed terms kept by a bounded beam."""
    bound = ensure_positive("max_nodes", max_nodes)
    kept = ensure_positive("beam width", width)
    if index.signature is not store.signature:
        msg = "the choice index and term store must share one signature"
        raise ValueError(msg)
    frontier_store = _frontier_store(store)
    active = (_initial(decoder, inputs, frontier_store, root_sort),)

    for _step in range(bound):
        if all(not _hole_paths(candidate.skeleton) for candidate in active):
            break

        following: list[_Candidate[S]] = []

        for candidate in active:
            if _hole_paths(candidate.skeleton):
                following.extend(
                    successor
                    for successor in _successors(
                        candidate,
                        decoder,
                        index,
                        frontier_store,
                        max_nodes=bound,
                        clamp=clamp,
                    )
                    if _hole_paths(successor.skeleton)
                    or _admitted(successor, decoder, frontier_store, store)
                )
            else:
                following.append(candidate)

        active = tuple(sorted(following, key=_rank)[:kept])

        if not active:
            break

    complete = tuple(
        candidate for candidate in active if not _hole_paths(candidate.skeleton)
    )
    return _weighted(
        complete,
        frontier_store,
        store,
        like=inputs.tensor,
        estimate=Estimate.TRUNCATED,
    )


def greedy[S](
    decoder: FrontierDecoder[S],
    inputs: Encoded,
    /,
    *,
    index: ChoiceIndex,
    store: TermStore,
    root_sort: Sort,
    max_nodes: int,
    clamp: Term | None = None,
) -> WeightedTerms:
    """Return the term obtained by taking the highest-weight choice at each hole."""
    return beam(
        decoder,
        inputs,
        index=index,
        store=store,
        root_sort=root_sort,
        max_nodes=max_nodes,
        width=1,
        clamp=clamp,
    )


def exhaustive[S](
    decoder: FrontierDecoder[S],
    inputs: Encoded,
    /,
    *,
    index: ChoiceIndex,
    store: TermStore,
    root_sort: Sort,
    max_nodes: int,
    residual_mass: float,
    exact_residual: float,
    cap: int,
    clamp: Term | None = None,
) -> WeightedTerms:
    """Enumerate terms until the residual bound or term cap stops the descent."""
    bound = ensure_positive("max_nodes", max_nodes)
    limit = ensure_positive("term cap", cap)
    residual_bound = ensure_finite("residual mass", residual_mass)

    if not 0.0 <= residual_bound < 1.0:
        msg = f"a residual mass lies in [0, 1), not at {residual_mass!r}"
        raise ValueError(msg)

    exactness = ensure_finite("exactness residual", exact_residual)

    if not 0.0 <= exactness < 1.0:
        msg = f"an exactness residual lies in [0, 1), not at {exact_residual!r}"
        raise ValueError(msg)

    if index.signature is not store.signature:
        msg = "the choice index and term store must share one signature"
        raise ValueError(msg)
    frontier_store = _frontier_store(store)
    active = (_initial(decoder, inputs, frontier_store, root_sort),)
    complete: list[_Candidate[S]] = []
    pruned = False

    for _step in range(bound):
        following: list[_Candidate[S]] = []

        for candidate in active:
            for successor in _successors(
                candidate,
                decoder,
                index,
                frontier_store,
                max_nodes=bound,
                clamp=clamp,
            ):
                if _hole_paths(successor.skeleton):
                    following.append(successor)
                elif _admitted(successor, decoder, frontier_store, store):
                    complete.append(successor)

        complete.sort(key=_rank)

        if len(complete) >= limit:
            pruned = len(complete) > limit or bool(following)
            complete = complete[:limit]
            active = tuple(following)
            break

        active = tuple(following)

        if not active:
            break

        active_mass = sum(
            float(candidate.log_weight.detach().exp()) for candidate in active
        )

        if complete and active_mass <= residual_bound:
            break

    provisional = _weighted(
        complete,
        frontier_store,
        store,
        like=inputs.tensor,
        estimate=Estimate.TRUNCATED,
    )
    exact = (
        not pruned and not active and bool(provisional.log_residual.exp() <= exactness)
    )

    if exact:
        return WeightedTerms(
            provisional.terms,
            provisional.log_weights,
            provisional.log_residual,
            Estimate.EXACT,
        )

    return provisional


def paired_exhaustive[LeftS, RightS](
    left_decoder: FrontierDecoder[LeftS],
    left_inputs: Encoded,
    right_decoder: FrontierDecoder[RightS],
    right_inputs: Encoded,
    /,
    *,
    index: ChoiceIndex,
    store: TermStore,
    root_sort: Sort,
    max_nodes: int,
    residual_mass: float,
    exact_residual: float,
    cap: int,
    clamp: Term | None = None,
) -> tuple[WeightedTerms, WeightedTerms]:
    """Enumerate one term support under two aligned decoder frontiers.

    The two decoders advance through each legal choice together. Thus the returned
    weights name identical terms in identical order without independently building
    and joining two marginal beams. Enumeration stops only when both active masses
    meet the declared residual bound, or when the shared cap truncates the support.
    """
    bound = ensure_positive("max_nodes", max_nodes)
    limit = ensure_positive("term cap", cap)
    residual_bound = ensure_finite("residual mass", residual_mass)

    if not 0.0 <= residual_bound < 1.0:
        msg = f"a residual mass lies in [0, 1), not at {residual_mass!r}"
        raise ValueError(msg)

    exactness = ensure_finite("exactness residual", exact_residual)

    if not 0.0 <= exactness < 1.0:
        msg = f"an exactness residual lies in [0, 1), not at {exact_residual!r}"
        raise ValueError(msg)

    if index.signature is not store.signature:
        msg = "the choice index and term store must share one signature"
        raise ValueError(msg)

    frontier_store = _frontier_store(store)
    root = frontier_store.hole(root_sort)
    active = (
        _PairedCandidate(
            root,
            left_decoder.start(left_inputs),
            right_decoder.start(right_inputs),
            (),
            left_inputs.tensor.new_zeros(()),
            right_inputs.tensor.new_zeros(()),
        ),
    )
    complete: list[_PairedCandidate[LeftS, RightS]] = []
    pruned = False

    for _step in range(bound):
        following: list[_PairedCandidate[LeftS, RightS]] = []

        for candidate in active:
            for successor in _paired_successors(
                candidate,
                left_decoder,
                right_decoder,
                index,
                frontier_store,
                max_nodes=bound,
                clamp=clamp,
            ):
                if _hole_paths(successor.skeleton):
                    following.append(successor)
                elif left_decoder.admits(
                    _materialize(frontier_store, store, successor.skeleton)
                ) and right_decoder.admits(
                    _materialize(frontier_store, store, successor.skeleton)
                ):
                    complete.append(successor)

        complete.sort(
            key=lambda candidate: (
                -float(candidate.left_log_weight.detach()),
                candidate.decisions,
            )
        )

        if len(complete) >= limit:
            pruned = len(complete) > limit or bool(following)
            complete = complete[:limit]
            active = tuple(following)
            break

        active = tuple(following)

        if not active:
            break

        left_active = sum(
            float(candidate.left_log_weight.detach().exp()) for candidate in active
        )
        right_active = sum(
            float(candidate.right_log_weight.detach().exp()) for candidate in active
        )

        omitted_union = left_active + right_active - left_active * right_active

        if complete and omitted_union <= residual_bound:
            break

    terms = tuple(
        _materialize(frontier_store, store, candidate.skeleton)
        for candidate in complete
    )
    left_weights = (
        torch.stack(tuple(candidate.left_log_weight for candidate in complete))
        if complete
        else left_inputs.tensor.new_empty((0,))
    )
    right_weights = (
        torch.stack(tuple(candidate.right_log_weight for candidate in complete))
        if complete
        else right_inputs.tensor.new_empty((0,))
    )
    left_residual = (1.0 - left_weights.exp().sum()).clamp_min(0.0).log()
    right_residual = (1.0 - right_weights.exp().sum()).clamp_min(0.0).log()
    exact = (
        not pruned
        and not active
        and bool(left_residual.exp() <= exactness)
        and bool(right_residual.exp() <= exactness)
    )
    estimate = Estimate.EXACT if exact else Estimate.TRUNCATED

    return (
        WeightedTerms(terms, left_weights, left_residual, estimate),
        WeightedTerms(terms, right_weights, right_residual, estimate),
    )


def sample[S](
    decoder: FrontierDecoder[S],
    inputs: Encoded,
    /,
    *,
    index: ChoiceIndex,
    store: TermStore,
    root_sort: Sort,
    max_nodes: int,
    restriction: SamplingStrategy,
    draws: int,
    clamp: Term | None = None,
    generator: torch.Generator | None = None,
) -> WeightedTerms:
    """Draw admitted terms under a declared restriction.

    A requested draw whose partial frontier reaches no policy-admitted choice is
    rejected. Sampling makes at most ``max_nodes`` attempts per requested draw, so
    admission cannot turn a finite decoding declaration into an unbounded retry.
    """
    bound = ensure_positive("max_nodes", max_nodes)
    count = ensure_positive("draws", draws)
    if index.signature is not store.signature:
        msg = "the choice index and term store must share one signature"
        raise ValueError(msg)
    reached: dict[Term, _Candidate[S]] = {}
    accepted = 0
    frontier_store = _frontier_store(store)

    for _attempt in range(count * bound):
        candidate = _initial(decoder, inputs, frontier_store, root_sort)

        for _step in range(bound):
            if not _hole_paths(candidate.skeleton):
                if _admitted(candidate, decoder, frontier_store, store):
                    reached.setdefault(candidate.skeleton, candidate)
                    accepted += 1
                break

            context = _next_path(candidate, bound)
            legal = _legal(
                index,
                context=context,
                forced=_forced(clamp, context.path, index),
            )
            try:
                distribution = decoder.weigh(candidate.state, context=context)
            except NoAdmissibleChoiceError:
                break
            row = _row(distribution, index)
            offered = torch.zeros_like(row, dtype=torch.bool)
            offered[list(legal)] = row[list(legal)].isfinite()
            if not bool(offered.any()):
                break

            sampling_row = row.masked_fill(~offered, -torch.inf)
            sampling_row = sampling_row - sampling_row.logsumexp(dim=0)
            restricted = LogDistribution(
                sampling_row,
                _CHOICE_AXES,
                offered,
                normalized_over=_CHOICE_AXES[0],
            )
            choice = int(
                sample_from_logits(
                    restricted,
                    strategy=restriction,
                    over=_CHOICE_AXES[0],
                    generator=generator,
                ).item()
            )
            candidate = _successor(
                candidate,
                decoder,
                index,
                frontier_store,
                context=context,
                choice=choice,
                value=row[choice],
            )

            if not _hole_paths(candidate.skeleton):
                if _admitted(candidate, decoder, frontier_store, store):
                    reached.setdefault(candidate.skeleton, candidate)
                    accepted += 1
                break

        if accepted == count:
            break

    return _weighted(
        reached.values(),
        frontier_store,
        store,
        like=inputs.tensor,
        estimate=Estimate.SAMPLED,
    )
