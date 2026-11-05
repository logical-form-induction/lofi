"""Complete-term coherence over declared represented spaces."""

from math import log
from typing import TYPE_CHECKING, Final, assert_never, cast, override

import torch

from lofi._checks import ensure_name, ensure_non_negative
from lofi.nn._values import Encoded
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.loss._approximation import ApproximationBound
from lofi.nn.loss._divergence import KL, CrossEntropy, FenchelYoung
from lofi.nn.loss._errors import MissingOutputError
from lofi.nn.loss._paired import PairedTermDivergence
from lofi.nn.loss._parser import SPAN_AXES, SPLIT_AXES
from lofi.nn.loss._ports import (
    CoherenceEvaluation,
    LossComponentBindings,
    LossTerm,
    ScheduledBatch,
)
from lofi.nn.ports import FrontierDecoder
from lofi.nn.spaces import (
    CombinatorFamily,
    RelationKind,
    Space,
    SpaceRelationBinding,
)
from lofi.registry import ComponentPlugin
from lofi.spec.losses.coherence import (
    CoherenceDivergenceSettings,
    CoherenceEstimator,
    CoherenceLossSettings,
    CoherenceLossSpec,
    CrossEntropyDivergenceSettings,
    FenchelYoungDivergenceSettings,
    KLDivergenceSettings,
)

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.deduce import SpanIndex
    from lofi.nn._values import LogDistribution
    from lofi.nn.loss._divergence import Divergence
    from lofi.nn.loss._ports import PublishedOutputs
    from lofi.nn.loss._readout import TermsReadout
    from lofi.terms import Term

__all__ = ["CoherenceObjective", "MarginalWarmup"]

_TERM_AXES: Final[tuple[str, ...]] = ("term", "width")


class MarginalWarmup:
    """Move span weights linearly from one to constituent marginals."""

    __slots__ = ("_steps",)

    def __init__(self, steps: int) -> None:
        if ensure_non_negative("the warmup's step count", steps) == 0:
            msg = "a warmup lasts at least one step"
            raise ValueError(msg)

        self._steps = steps

    @property
    def steps(self) -> int:
        """Return the step at which marginals take the whole weight."""
        return self._steps

    def fraction(self, step: int) -> float:
        """Return the marginal share at one nonnegative training step."""
        return min(1.0, ensure_non_negative("a step", step) / self._steps)

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not MarginalWarmup:
            return NotImplemented

        return self._steps == other._steps

    @override
    def __hash__(self) -> int:
        return hash((MarginalWarmup, self._steps))

    @override
    def __repr__(self) -> str:
        return f"MarginalWarmup({self._steps!r})"


def _required[T](value: T | None, field: str, *, objective: str) -> T:
    """Return one required published value or name its absence."""
    if value is None:
        raise MissingOutputError(
            objective=objective, missing=frozenset({field}), published=frozenset()
        )

    return value


def _cell_mask(cells: Encoded) -> Tensor | None:
    """Return which whole cells are live."""
    mask = cells.mask

    if mask is None:
        return None

    if mask.ndim > len(SPAN_AXES):
        msg = f"a cell is decoded whole, and a mask of rank {mask.ndim} marks part"
        raise ValueError(msg)

    items, count, _ = cells.tensor.shape
    widened = mask.reshape(tuple(mask.shape) + (1,) * (len(SPAN_AXES) - mask.ndim))

    return widened.expand(items, count)


def _per_split(
    weight: LogDistribution, spans: SpanIndex
) -> tuple[Tensor, Tensor | None]:
    """Return one controller weight at every declared cutpoint."""
    tensor = weight.tensor
    mask = weight.mask
    axes = weight.axes

    if axes == SPAN_AXES:
        tensor = tensor.unsqueeze(-1).expand(-1, -1, spans.max_splits)
    elif axes != SPLIT_AXES:
        msg = f"an action weight runs over {SPAN_AXES} or {SPLIT_AXES}, not {axes}"
        raise ValueError(msg)

    expected = (spans.num_spans, spans.max_splits)

    if tuple(tensor.shape[1:]) != expected:
        msg = (
            f"an action weight holds {tuple(tensor.shape[1:])}, and the span index "
            f"addresses {expected}"
        )
        raise ValueError(msg)

    if mask is None:
        return tensor, None

    widened = mask.reshape(tuple(mask.shape) + (1,) * (len(SPLIT_AXES) - mask.ndim))

    return tensor, widened.expand_as(tensor)


def _per_span(
    weight: LogDistribution, spans: SpanIndex
) -> tuple[Tensor, Tensor | None]:
    """Return one unary-action weight at every declared span."""
    tensor = weight.tensor
    mask = weight.mask

    if weight.axes == SPLIT_AXES:
        tensor = tensor.logsumexp(dim=-1)
        mask = None if mask is None else mask.any(dim=-1)
    elif weight.axes != SPAN_AXES:
        msg = f"a unary action weight runs over {SPAN_AXES}, not {weight.axes}"
        raise ValueError(msg)

    if tuple(tensor.shape[1:]) != (spans.num_spans,):
        msg = (
            f"a unary action weight holds {tuple(tensor.shape[1:])}, and the span "
            f"index addresses {(spans.num_spans,)}"
        )
        raise ValueError(msg)

    return tensor, mask


def _mixture(
    source: WeightedTerms, branches: tuple[WeightedTerms, ...], /
) -> WeightedTerms:
    """Mix one decoded branch under each listed source-term weight."""
    if len(branches) != len(source.terms):
        msg = (
            f"a unary mixture has {len(source.terms)} source terms and "
            f"{len(branches)} branches"
        )
        raise ValueError(msg)

    by_term: dict[Term, list[Tensor]] = {}
    residuals = [source.log_residual]
    estimates = [source.estimate]

    for source_weight, branch in zip(source.log_weights, branches, strict=True):
        residuals.append(source_weight + branch.log_residual)
        estimates.append(branch.estimate)

        for term, weight in zip(branch.terms, branch.log_weights, strict=True):
            by_term.setdefault(term, []).append(source_weight + weight)

    terms = tuple(by_term)
    weights = source.log_weights.new_empty((0,))

    if terms:
        weights = torch.stack(
            [torch.stack(by_term[term]).logsumexp(0) for term in terms]
        )

    residual = torch.stack(residuals).logsumexp(0).clamp_max(0.0)
    estimate = (
        Estimate.SAMPLED
        if Estimate.SAMPLED in estimates
        else Estimate.TRUNCATED
        if Estimate.TRUNCATED in estimates
        else Estimate.EXACT
    )

    return WeightedTerms(terms, weights, residual, estimate)


def _restrict_to_domain(
    source: WeightedTerms,
    source_rows: tuple[int, ...],
    /,
) -> WeightedTerms:
    """Keep a partial rule's known premises and the source truncation mass."""
    rows = tuple(dict.fromkeys(source_rows))
    weights = torch.stack(tuple(source.log_weights[row] for row in rows))

    return WeightedTerms(
        tuple(source.terms[row] for row in rows),
        weights,
        source.log_residual,
        source.estimate,
    )


def _uniform_mixture(branches: tuple[WeightedTerms, ...], /) -> WeightedTerms:
    """Mix equally weighted conclusions of one multivalued symbolic rule."""
    if not branches:
        msg = "a uniform unary mixture has no branch"
        raise ValueError(msg)

    shift = branches[0].log_residual.new_tensor(-log(len(branches)))
    by_term: dict[Term, list[Tensor]] = {}
    estimates: list[Estimate] = []
    residuals: list[Tensor] = []
    for branch in branches:
        estimates.append(branch.estimate)
        residuals.append(shift + branch.log_residual)
        for term, weight in zip(branch.terms, branch.log_weights, strict=True):
            by_term.setdefault(term, []).append(shift + weight)

    terms = tuple(by_term)
    weights = shift.new_empty((0,))
    if terms:
        weights = torch.stack(
            [torch.stack(by_term[term]).logsumexp(0) for term in terms]
        )
    estimate = (
        Estimate.SAMPLED
        if Estimate.SAMPLED in estimates
        else Estimate.TRUNCATED
        if Estimate.TRUNCATED in estimates
        else Estimate.EXACT
    )
    return WeightedTerms(
        terms,
        weights,
        torch.stack(residuals).logsumexp(0).clamp_max(0.0),
        estimate,
    )


def _span_weights(marginal: Tensor, fraction: float) -> Tensor:
    """Interpolate each span weight from one to its marginal."""
    return (1.0 - fraction) + fraction * marginal


def _checked_item_weights(weights: Tensor | None, items: int, like: Tensor) -> Tensor:
    """Return one finite nonnegative weight per batch item."""
    if weights is None:
        return like.new_ones(items)

    if not isinstance(weights, torch.Tensor):
        msg = "coherence item weights must be a tensor"
        raise TypeError(msg)

    if weights.ndim != 1 or weights.numel() != items:
        shape = tuple(weights.shape)
        msg = f"coherence item weights have shape {shape}, not ({items},)"
        raise ValueError(msg)

    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        msg = "coherence item weights must be finite and nonnegative"
        raise ValueError(msg)

    return weights.to(device=like.device, dtype=like.dtype)


def _family(space: Space, /) -> CombinatorFamily:
    """Return the composing family of one represented space."""
    family = space.combinators

    if family is None:
        msg = f"space for sort {space.sort!r} declares no combinator family"
        raise ValueError(msg)

    return family


def _frontier(decoder: object, /, *, role: str) -> FrontierDecoder[object]:
    """Return one decoder after checking paired-frontier capability."""
    if not isinstance(decoder, FrontierDecoder):
        msg = f"the {role} decoder does not implement frontier decoding"
        raise TypeError(msg)

    return cast("FrontierDecoder[object]", decoder)


def _projected_cells(cells: Encoded, relation: SpaceRelationBinding | None) -> Encoded:
    """Return chart cells in the space whose decoders coherence compares."""
    if relation is None or relation.kind is RelationKind.ERASE_THEN_DECODE:
        return cells

    items, count, width = cells.tensor.shape
    flattened = Encoded(cells.tensor.reshape(items * count, width), _TERM_AXES)
    projected = relation.project(flattened)

    return Encoded(
        projected.tensor.reshape(items, count, projected.tensor.shape[-1]),
        cells.axes,
        cells.mask,
    )


def _valid_pairs(
    cells: Encoded,
    spans: SpanIndex,
    split_valid: Tensor,
    marginal_built: Tensor,
) -> Tensor:
    """Return each split whose parent and children are available."""
    items = int(cells.tensor.shape[0])
    structure = torch.tensor(
        spans.split_mask, dtype=torch.bool, device=cells.tensor.device
    )
    valid = structure.unsqueeze(0).expand(items, -1, -1)
    valid = valid & split_valid & marginal_built.unsqueeze(-1)
    live = _cell_mask(cells)

    if live is not None:
        parent = torch.arange(spans.num_spans, device=cells.tensor.device)
        left = torch.tensor(
            spans.left_child_ids, dtype=torch.long, device=cells.tensor.device
        )
        right = torch.tensor(
            spans.right_child_ids, dtype=torch.long, device=cells.tensor.device
        )
        valid = valid & live[:, parent].unsqueeze(-1) & live[:, left] & live[:, right]

    return valid


def _evaluations(
    recorded: list[list[tuple[str, int, float]]] | None, /
) -> tuple[CoherenceEvaluation, ...]:
    """Freeze optional detached contributions in batch order."""
    if recorded is None:
        return ()

    return tuple(
        CoherenceEvaluation(index, tuple(contributions))
        for index, contributions in enumerate(recorded)
    )


class CoherenceObjective[BatchT: ScheduledBatch]:
    """Compare complete parent terms with each declared child composition."""

    __slots__ = (
        "_approximation",
        "_divergence",
        "_estimator",
        "_name",
        "_readout",
        "_relation",
        "_space",
        "_warmup",
    )

    def __init__(
        self,
        *,
        readout: TermsReadout,
        space: Space,
        relation: SpaceRelationBinding | None,
        divergence: Divergence,
        warmup: MarginalWarmup | None,
        estimator: CoherenceEstimator,
        name: str,
        approximation: ApproximationBound | None,
    ) -> None:
        if readout.decoding is None:
            msg = "coherence requires a complete-term decoding readout"
            raise ValueError(msg)

        if relation is not None and relation.source is not space:
            msg = "a coherence relation must start at its declared source space"
            raise ValueError(msg)

        if estimator is CoherenceEstimator.TRUNCATED and approximation is None:
            msg = "truncated coherence requires a residual-mass bound"
            raise ValueError(msg)

        self._readout = readout
        self._space = space
        self._relation = relation
        self._divergence = divergence
        self._warmup = warmup
        self._estimator = estimator
        self._name = ensure_name("the objective's name", name)
        self._approximation = approximation

    @property
    def name(self) -> str:
        """Return the reporting name of this objective."""
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        """Return the parser publications this objective consumes."""
        required = {"spans", "cells", "splits", "marginals", "actions"}

        if self._warmup is not None:
            required.add("batch")

        return frozenset(required)

    @property
    def readout(self) -> TermsReadout:
        """Return the complete-term strategy used for paired decoding."""
        return self._readout

    @property
    def space(self) -> Space:
        """Return the declared source space."""
        return self._space

    @property
    def relation(self) -> SpaceRelationBinding | None:
        """Return the source-to-target relation, or none within one space."""
        return self._relation

    @property
    def estimator(self) -> CoherenceEstimator:
        """Return whether paired decoding must be exact or may truncate."""
        return self._estimator

    @property
    def approximation(self) -> ApproximationBound | None:
        """Return the coherence residual budget, or none for exact estimation."""
        return self._approximation

    @property
    def warmup(self) -> MarginalWarmup | None:
        """Return the marginal warmup, or none for immediate marginal weights."""
        return self._warmup

    def _decoding_space(self) -> Space:
        """Return the space whose decoders produce the compared terms."""
        relation = self._relation

        if relation is None or relation.kind is RelationKind.ERASE_THEN_DECODE:
            return self._space

        return relation.target

    def _pairing(self) -> PairedTermDivergence:
        """Return the complete-term comparison implied by the relation."""
        relation = self._relation

        if relation is None or relation.kind is not RelationKind.ERASE_THEN_DECODE:
            return PairedTermDivergence(self._divergence)

        def erase(term: Term) -> Term:
            return relation.erase(term)

        return PairedTermDivergence(self._divergence, reference=erase, candidate=erase)

    def _consistency(
        self,
        source_input: Encoded,
        target_input: Encoded,
        /,
    ) -> LossTerm | None:
        """Compare category and semantic identity decoders when declared."""
        relation = self._relation

        if relation is None or relation.kind is not RelationKind.CONSISTENCY:
            return None

        source_decoder = _frontier(relation.source.identity, role="source identity")
        target_decoder = _frontier(relation.target.identity, role="target identity")
        approximation = self._member_approximation()
        reference = self._readout.decode_input(
            source_decoder,
            source_input,
            space=relation.source,
            approximation=approximation,
        )
        candidate = self._readout.decode_input(
            target_decoder,
            target_input,
            space=relation.target,
            approximation=approximation,
        )

        if self._estimator is CoherenceEstimator.EXACT and (
            reference.estimate is not Estimate.EXACT
            or candidate.estimate is not Estimate.EXACT
        ):
            msg = "exact coherence exhausted neither identity frontier"
            raise ValueError(msg)

        return PairedTermDivergence(
            self._divergence,
            reference=relation.erase,
        )(reference, candidate)

    def _member_approximation(self) -> ApproximationBound | None:
        """Return a safe member budget for two-frontier residual unions."""
        approximation = self._approximation
        return None if approximation is None else approximation.paired_member()

    def _fraction(self, outputs: PublishedOutputs[BatchT]) -> float:
        """Return the current marginal share."""
        if self._warmup is None:
            return 1.0

        return self._warmup.fraction(outputs.batch.step)

    def _evaluate_binary(
        self,
        outputs: PublishedOutputs[BatchT],
        /,
        *,
        item_weights: Tensor | None = None,
        record: bool,
    ) -> tuple[LossTerm, tuple[CoherenceEvaluation, ...]]:
        """Return the paired divergence and optional detached item records."""
        spans = outputs.spans
        cells = _required(outputs.cells, "cells", objective=self._name)
        splits = _required(outputs.splits, "splits", objective=self._name)
        marginals = _required(outputs.marginals, "marginals", objective=self._name)
        effective_cells = _projected_cells(cells, self._relation)
        space = self._decoding_space()
        family = _family(space)
        reference_decoder = _frontier(space.identity, role="identity")
        binary = tuple(family.decoders)
        recorded: list[list[tuple[str, int, float]]] | None = (
            [[] for _ in range(int(effective_cells.tensor.shape[0]))]
            if record
            else None
        )
        zero = effective_cells.tensor.new_zeros(())

        if not binary:
            return LossTerm(zero, 0, residual=zero), _evaluations(recorded)

        candidate_decoders = tuple(
            _frontier(family.decoders[action], role=action.name) for action in binary
        )
        action_rows = tuple(
            _per_split(outputs.actions[action], spans) for action in binary
        )
        valid = _valid_pairs(effective_cells, spans, splits.valid, marginals.built)

        for _, mask in action_rows:
            if mask is not None:
                valid = valid & mask

        items = int(effective_cells.tensor.shape[0])
        checked_items = _checked_item_weights(
            item_weights, items, effective_cells.tensor
        )
        item, span, split = valid.nonzero(as_tuple=True)

        if item.numel() == 0:
            return LossTerm(zero, 0, residual=zero), _evaluations(recorded)

        left_ids = torch.tensor(
            spans.left_child_ids, dtype=torch.long, device=valid.device
        )
        right_ids = torch.tensor(
            spans.right_child_ids, dtype=torch.long, device=valid.device
        )
        left = left_ids[span, split]
        right = right_ids[span, split]
        action_logits = torch.stack([row[item, span, split] for row, _ in action_rows])
        has_action = action_logits.logsumexp(0) > -torch.inf
        item = item[has_action]
        span = span[has_action]
        split = split[has_action]
        left = left[has_action]
        right = right[has_action]
        action_logits = action_logits[:, has_action]

        if item.numel() == 0:
            return LossTerm(zero, 0, residual=zero), _evaluations(recorded)

        marginal = marginals.constituent.tensor[item, span]
        split_probability = splits.log_probabilities.tensor[item, span, split].exp()
        share = (
            _span_weights(marginal, self._fraction(outputs))
            * split_probability
            * checked_items[item]
        )
        positive = share > 0
        item = item[positive]
        span = span[positive]
        left = left[positive]
        right = right[positive]
        share = share[positive]
        action_probabilities = action_logits[:, positive].softmax(0)

        if item.numel() == 0:
            return LossTerm(zero, 0, residual=zero), _evaluations(recorded)

        pairing = self._pairing()
        total = zero
        residual_total = zero
        residual_weight = zero
        member_approximation = self._member_approximation()

        for pair in range(item.numel()):
            width = int(effective_cells.tensor.shape[-1])
            parent_input = Encoded(
                effective_cells.tensor[item[pair], span[pair]].reshape(1, width),
                _TERM_AXES,
            )
            left_input = Encoded(
                effective_cells.tensor[item[pair], left[pair]].reshape(1, width),
                _TERM_AXES,
            )
            right_input = Encoded(
                effective_cells.tensor[item[pair], right[pair]].reshape(1, width),
                _TERM_AXES,
            )
            merge = family.merge

            if merge is None:
                msg = "binary coherence has no declared daughter merge"
                raise RuntimeError(msg)

            candidate_input = merge.merge(left_input, right_input)

            if self._relation is not None and (
                self._relation.kind is RelationKind.CONSISTENCY
            ):
                source_width = int(cells.tensor.shape[-1])
                source_input = Encoded(
                    cells.tensor[item[pair], span[pair]].reshape(1, source_width),
                    _TERM_AXES,
                )
                consistency = self._consistency(source_input, parent_input)

                if consistency is None or consistency.residual is None:
                    msg = "a declared consistency relation reported no residual mass"
                    raise ValueError(msg)

                total = total + share[pair] * consistency.value
                residual_total = (
                    residual_total + share[pair].detach() * consistency.residual
                )
                residual_weight = residual_weight + share[pair].detach()

                if recorded is not None:
                    batch_item = int(item[pair])
                    start, end = spans.span_of(int(span[pair]))
                    recorded[batch_item].append(
                        (
                            "consistency",
                            end - start,
                            float((share[pair] * consistency.value).detach()),
                        )
                    )

            for column, decoder in enumerate(candidate_decoders):
                reference, candidate = self._readout.paired_inputs(
                    reference_decoder,
                    decoder,
                    parent_input,
                    candidate_input,
                    space=space,
                    approximation=member_approximation,
                )

                if self._estimator is CoherenceEstimator.EXACT and (
                    reference.estimate is not Estimate.EXACT
                    or candidate.estimate is not Estimate.EXACT
                ):
                    msg = "exact coherence exhausted neither paired frontier"
                    raise ValueError(msg)

                compared = pairing(reference, candidate)
                probability = action_probabilities[column, pair]
                weight = share[pair] * probability
                contribution = torch.where(weight > 0, weight * compared.value, zero)
                total = total + contribution

                if recorded is not None:
                    batch_item = int(item[pair])
                    start, end = spans.span_of(int(span[pair]))
                    recorded[batch_item].append(
                        (
                            binary[column].name,
                            end - start,
                            float(contribution.detach()),
                        )
                    )

                if compared.residual is None:
                    msg = "a paired term divergence did not report residual mass"
                    raise ValueError(msg)

                residual_total = residual_total + weight.detach() * compared.residual
                residual_weight = residual_weight + weight.detach()

        count = int(item.unique().numel())
        share_total = share.sum()
        floor = torch.finfo(share.dtype).tiny
        controller_logs = action_probabilities.clamp_min(floor).log()
        controller_entropy = (
            -(action_probabilities * controller_logs).sum(0).mean().detach()
        )

        evaluations = _evaluations(recorded)
        return (
            LossTerm(
                total / count,
                count,
                residual=residual_total / residual_weight,
                diagnostics={
                    "controller_entropy": controller_entropy,
                    "signal_share": (
                        marginal[positive]
                        * split_probability[positive]
                        * checked_items[item]
                    ).sum()
                    / share_total,
                },
            ),
            evaluations,
        )

    def _evaluate_unary(
        self,
        outputs: PublishedOutputs[BatchT],
        /,
        *,
        item_weights: Tensor | None,
        record: bool,
    ) -> tuple[LossTerm, tuple[CoherenceEvaluation, ...]]:
        """Return coherence between each cell and its unary-rule re-encodings."""
        spans = outputs.spans
        cells = _required(outputs.cells, "cells", objective=self._name)
        marginals = _required(outputs.marginals, "marginals", objective=self._name)
        space = self._decoding_space()
        family = _family(space)
        unary = family.reencodable_unary
        items = int(cells.tensor.shape[0])
        recorded: list[list[tuple[str, int, float]]] | None = (
            [[] for _ in range(items)] if record else None
        )
        zero = cells.tensor.new_zeros(())

        if not unary:
            return LossTerm(zero, 0, residual=zero), _evaluations(recorded)

        effective_cells = _projected_cells(cells, self._relation)
        action_rows = tuple(
            _per_span(outputs.actions[action], spans) for action in unary
        )
        logits = torch.stack(
            [
                tensor.masked_fill(
                    ~mask,
                    -torch.inf,
                )
                if mask is not None
                else tensor
                for tensor, mask in action_rows
            ]
        )
        valid = marginals.built & torch.isfinite(logits).any(dim=0)
        live = _cell_mask(effective_cells)

        if live is not None:
            valid = valid & live

        item, span = valid.nonzero(as_tuple=True)

        if item.numel() == 0:
            return LossTerm(zero, 0, residual=zero), _evaluations(recorded)

        checked_items = _checked_item_weights(
            item_weights, items, effective_cells.tensor
        )
        marginal = marginals.constituent.tensor[item, span]
        share = _span_weights(marginal, self._fraction(outputs)) * checked_items[item]
        positive = share > 0
        item = item[positive]
        span = span[positive]
        share = share[positive]
        signal = marginal[positive] * checked_items[item]
        action_logits = logits[:, item, span]

        if item.numel() == 0:
            return LossTerm(zero, 0, residual=zero), _evaluations(recorded)

        identity = _frontier(space.identity, role="identity")
        pairing = self._pairing()
        total = zero
        residual_total = zero
        share_total = zero
        signal_total = zero
        controller_entropy_total = zero
        evaluated_cells = 0
        evaluated_items: set[int] = set()
        member_approximation = self._member_approximation()

        for cell in range(item.numel()):
            width = int(effective_cells.tensor.shape[-1])
            parent_input = Encoded(
                effective_cells.tensor[item[cell], span[cell]].reshape(1, width),
                _TERM_AXES,
            )
            reference = self._readout.decode_input(
                identity,
                parent_input,
                space=space,
                approximation=member_approximation,
            )

            if self._estimator is CoherenceEstimator.EXACT and (
                reference.estimate is not Estimate.EXACT
            ):
                msg = "exact unary coherence did not exhaust the premise frontier"
                raise ValueError(msg)

            candidates: list[tuple[int, WeightedTerms, WeightedTerms]] = []

            for column, action in enumerate(unary):
                if not bool(torch.isfinite(action_logits[column, cell])):
                    continue

                reencoded = family.reencode(action, reference.terms)

                if not reencoded.source_rows:
                    continue

                encoded = reencoded.encoded
                domain_rows = tuple(dict.fromkeys(reencoded.source_rows))
                domain_reference = _restrict_to_domain(reference, domain_rows)

                if member_approximation is not None:
                    member_approximation.check_terms(
                        domain_reference,
                        component=f"coherence objective {self._name!r} unary premise",
                    )

                conclusions_by_source: dict[int, list[int]] = {}
                for target, source in enumerate(reencoded.source_rows):
                    conclusions_by_source.setdefault(source, []).append(target)
                branches = tuple(
                    _uniform_mixture(
                        tuple(
                            self._readout.decode_input(
                                identity,
                                Encoded(
                                    encoded.tensor[target].reshape(
                                        1, encoded.tensor.shape[-1]
                                    ),
                                    _TERM_AXES,
                                ),
                                space=space,
                                approximation=member_approximation,
                            )
                            for target in conclusions_by_source[source]
                        )
                    )
                    for source in domain_rows
                )
                candidate = _mixture(domain_reference, branches)

                if self._estimator is CoherenceEstimator.EXACT and (
                    candidate.estimate is not Estimate.EXACT
                ):
                    msg = "exact unary coherence did not exhaust a conclusion frontier"
                    raise ValueError(msg)

                candidates.append((column, domain_reference, candidate))

            if not candidates:
                continue

            probabilities = torch.stack(
                [action_logits[column, cell] for column, _, _ in candidates]
            ).softmax(0)
            floor = torch.finfo(probabilities.dtype).tiny
            controller_entropy_total = controller_entropy_total + (
                -(probabilities * probabilities.clamp_min(floor).log()).sum().detach()
            )
            share_total = share_total + share[cell].detach()
            signal_total = signal_total + signal[cell].detach()
            evaluated_cells += 1
            evaluated_items.add(int(item[cell]))

            for probability, (column, domain_reference, candidate) in zip(
                probabilities, candidates, strict=True
            ):
                paired = pairing(domain_reference, candidate)
                compared = LossTerm(
                    paired.value,
                    paired.count,
                    residual=(
                        candidate.log_residual.new_zeros(())
                        if candidate.estimate is Estimate.EXACT
                        else candidate.log_residual.exp()
                    ),
                    diagnostics=paired.diagnostics,
                )
                weight = share[cell] * probability
                contribution = torch.where(weight > 0, weight * compared.value, zero)
                total = total + contribution

                if recorded is not None:
                    batch_item = int(item[cell])
                    start, end = spans.span_of(int(span[cell]))
                    recorded[batch_item].append(
                        (
                            unary[column].name,
                            end - start,
                            float(contribution.detach()),
                        )
                    )

                if compared.residual is None:
                    msg = "a unary term divergence did not report residual mass"
                    raise ValueError(msg)

                residual_total = residual_total + weight.detach() * compared.residual

        count = len(evaluated_items)

        if count == 0:
            return LossTerm(zero, 0, residual=zero), _evaluations(recorded)

        return (
            LossTerm(
                total / count,
                count,
                residual=residual_total / share_total,
                diagnostics={
                    "controller_entropy": controller_entropy_total / evaluated_cells,
                    "signal_share": signal_total / share_total,
                },
            ),
            _evaluations(recorded),
        )

    def _evaluate(
        self,
        outputs: PublishedOutputs[BatchT],
        /,
        *,
        item_weights: Tensor | None = None,
        record: bool,
    ) -> tuple[LossTerm, tuple[CoherenceEvaluation, ...]]:
        """Return the binary and unary coherence contributions together."""
        binary, binary_evaluations = self._evaluate_binary(
            outputs, item_weights=item_weights, record=record
        )
        unary, unary_evaluations = self._evaluate_unary(
            outputs, item_weights=item_weights, record=record
        )
        evaluations = tuple(
            CoherenceEvaluation(
                left.item,
                (*left.contributions, *right.contributions),
            )
            for left, right in zip(binary_evaluations, unary_evaluations, strict=True)
        )

        if binary.count == 0:
            return unary, evaluations
        if unary.count == 0:
            return binary, evaluations

        diagnostics = {
            name: (binary.diagnostics[name] + unary.diagnostics[name]) / 2
            for name in binary.diagnostics.keys() & unary.diagnostics.keys()
        }
        residual = None

        if binary.residual is not None and unary.residual is not None:
            residual = torch.maximum(binary.residual, unary.residual)

        return (
            LossTerm(
                binary.value + unary.value,
                max(binary.count, unary.count),
                residual=residual,
                diagnostics=diagnostics,
            ),
            evaluations,
        )

    def evaluate(
        self,
        outputs: PublishedOutputs[BatchT],
        /,
        *,
        item_weights: Tensor | None = None,
    ) -> LossTerm:
        """Return the action-, split- and marginal-weighted paired divergence."""
        term, _ = self._evaluate(outputs, item_weights=item_weights, record=False)

        return self._checked(term)

    def _checked(self, term: LossTerm, /) -> LossTerm:
        """Return a coherence term after checking a truncating residual."""
        if (
            self._estimator is CoherenceEstimator.TRUNCATED
            and self._approximation is not None
            and term.residual is not None
        ):
            self._approximation.check_residual(
                term.residual,
                component=f"coherence objective {self._name!r}",
            )

        return term

    def evaluation(
        self, outputs: PublishedOutputs[BatchT], /
    ) -> tuple[CoherenceEvaluation, ...]:
        """Return detached per-item coherence for held-out reporting."""
        term, evaluations = self._evaluate(outputs, record=True)
        self._checked(term)
        return evaluations

    @override
    def __repr__(self) -> str:
        relation = None if self._relation is None else self._relation.kind.value
        return (
            f"CoherenceObjective(name={self._name!r}, sort={self._space.sort!r}, "
            f"relation={relation!r}, estimator={self._estimator.value!r}, "
            f"divergence={self._divergence!r}, warmup={self._warmup!r})"
        )


def _divergence(settings: CoherenceDivergenceSettings, /) -> Divergence:
    """Return the runtime divergence one declaration describes."""
    match settings:
        case KLDivergenceSettings():
            return KL
        case CrossEntropyDivergenceSettings():
            return CrossEntropy()
        case FenchelYoungDivergenceSettings(alpha=alpha):
            return FenchelYoung(alpha)
        case unreachable:
            assert_never(unreachable)


def _build(
    settings: CoherenceLossSettings, context: LossComponentBindings, /
) -> CoherenceObjective[ScheduledBatch]:
    """Build complete-term coherence from declared model roles."""
    source = context.space(settings.space)
    target = context.readout_space(settings.readout)
    relation = None

    if target is not source:
        relation = context.relation(settings.space)

        if relation.target is not target:
            msg = (
                f"space {settings.space!r} relates to sort {relation.target.sort!r}, "
                f"not the readout's sort {target.sort!r}"
            )
            raise ValueError(msg)

    readout = context.terms_readout(settings.readout)
    approximation = None

    if settings.estimator is CoherenceEstimator.TRUNCATED:
        approximation = (
            ApproximationBound(settings.residual_mass)
            if settings.residual_mass is not None
            else readout.run_approximation
        )

    return CoherenceObjective(
        readout=readout,
        space=source,
        relation=relation,
        divergence=_divergence(settings.divergence),
        warmup=None if not settings.warmup else MarginalWarmup(settings.warmup),
        estimator=settings.estimator,
        name=context.name,
        approximation=approximation,
    )


PLUGIN: Final = ComponentPlugin(spec=CoherenceLossSpec, build=_build)
"""The complete-term coherence objective plugin."""
