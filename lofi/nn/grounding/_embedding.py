"""Inference grounding from expected-form or sentence vectors."""

from typing import TYPE_CHECKING, Final, Literal, final, override

import torch
from torch import nn

from lofi.lf import InferenceCell, InferenceLabel, InferenceLabeling
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.entailment import EntailmentHead, PairedVectors
from lofi.nn.forms import WeightedTerms
from lofi.nn.grounding._values import (
    CellMass,
    Composition,
    EmbeddingVectors,
    EvidenceLedger,
    GradientPath,
    GroundedBatch,
    GroundingMetadata,
    checked_composition,
    composition_from_settings,
)
from lofi.nn.ports import MarginalDecoder, SoftTermEncoder, SubwordPooler
from lofi.nn.sites import ComponentId, SiteMap
from lofi.nn.spaces import Space
from lofi.registry import ComponentPlugin
from lofi.spec.groundings import ExpectedEmbeddingSettings, ExpectedEmbeddingSpec
from lofi.terms import BatchEncoding, Encoding, Operator

if TYPE_CHECKING:
    from torch import Tensor

    from lofi._types import ReadoutId
    from lofi.nn.build import BuildContext
    from lofi.nn.grounding._ports import GroundingInput

__all__ = ["PLUGIN", "EmbeddingVectors", "ExpectedEmbeddingGrounding"]


def _expected(space: Space, forms: WeightedTerms, /) -> Encoded | None:
    if not forms.terms:
        return None

    encoded = space.encoder.encode(space.codec.codec.encode_batch(forms.terms))

    if encoded.axes != ("term", "width"):
        msg = (
            "an expected-form grounding reads term encodings over "
            f"('term', 'width'), not {encoded.axes!r}"
        )
        raise ValueError(msg)

    weights = forms.log_weights.softmax(0).to(encoded.tensor)
    vector = (weights.unsqueeze(1) * encoded.tensor).sum(0, keepdim=True)
    return Encoded(vector, ("pair", "width"))


def _soft_expected(space: Space, rows: LogDistribution, /) -> Encoded:
    """Encode choice marginals without selecting or enumerating complete terms."""
    encoder = space.encoder

    if not isinstance(encoder, SoftTermEncoder):
        msg = "soft expected-form grounding requires a SoftTermEncoder"
        raise TypeError(msg)

    expected_axes = ("term", "slot", "choice")

    if rows.axes != expected_axes or rows.normalized_over != "choice":
        msg = (
            f"soft expected-form rows run over {expected_axes} and normalize over "
            f"'choice', not {rows.axes!r} and {rows.normalized_over!r}"
        )
        raise ValueError(msg)

    skeleton = space.codec.codec.skeleton

    if skeleton is None:
        msg = "soft expected-form grounding requires a path-addressed layout"
        raise ValueError(msg)

    pad = space.choices.PAD
    occupied = rows.tensor[..., pad + 1 :].isfinite().any(dim=-1)
    placeholders = tuple(
        next(
            (
                symbol
                for symbol, admitted in enumerate(per_slot)
                if admitted and symbol != pad
            ),
            pad,
        )
        for per_slot in space.codec.codec.legal_symbols
    )
    structure = BatchEncoding(
        tuple(
            Encoding(
                tuple(
                    placeholders[slot] if live else pad
                    for slot, live in enumerate(per_term)
                ),
                skeleton.parents,
                skeleton.depths,
                tuple(per_term),
            )
            for per_term in occupied.detach().cpu().tolist()
        )
    )
    encoded = encoder.encode_soft(rows, structure=structure)

    if encoded.axes != ("term", "width"):
        msg = (
            "a soft expected-form grounding reads encodings over "
            f"('term', 'width'), not {encoded.axes!r}"
        )
        raise ValueError(msg)

    return Encoded(encoded.tensor, ("pair", "width"), encoded.mask)


def _forms(batch: GroundingInput, name: ReadoutId, /) -> WeightedTerms:
    forms = batch.forms.get(name)

    if forms is None:
        offered = ", ".join(sorted(batch.forms)) or "nothing"
        msg = (
            f"expected-embedding grounding reads forms {name!r}, and the step "
            f"published {offered}"
        )
        raise ValueError(msg)

    return forms


def _choice_rows(batch: GroundingInput, name: ReadoutId, /) -> LogDistribution:
    """Return one pair side's payload-aware choice marginals."""
    rows = batch.choice_rows.get(name)

    if rows is None:
        offered = ", ".join(sorted(batch.choice_rows)) or "nothing"
        msg = (
            f"soft expected-embedding grounding reads choice rows {name!r}, and "
            f"the step published {offered}"
        )
        raise ValueError(msg)

    return rows


def _validate_soft_space(space: Space, /) -> None:
    """Refuse a represented space whose marginals do not have fixed contexts."""
    if not isinstance(space.identity, MarginalDecoder):
        msg = "soft expected-form grounding requires a MarginalDecoder"
        raise TypeError(msg)

    if not isinstance(space.encoder, SoftTermEncoder):
        msg = "soft expected-form grounding requires a SoftTermEncoder"
        raise TypeError(msg)

    codec = space.codec.codec
    skeleton = codec.skeleton

    if skeleton is None or not space.codec.path_addressed:
        msg = "soft expected-form grounding requires a path-addressed layout"
        raise ValueError(msg)

    index = codec.index
    signature = index.signature

    if space.sort not in signature.sorts:
        msg = f"soft expected-form grounding names undeclared root sort {space.sort!r}"
        raise ValueError(msg)

    offered = tuple(
        tuple(
            index.operator(symbol)
            for symbol, admitted in enumerate(per_slot)
            if admitted and symbol != index.PAD
        )
        for per_slot in codec.legal_symbols
    )
    reachable: list[tuple[Operator, ...]] = []

    for slot, parent in enumerate(skeleton.parents):
        if parent == Encoding.NO_PARENT:
            sorts = frozenset({space.sort})
        else:
            side = sum(prior == parent for prior in skeleton.parents[:slot])
            sorts = frozenset(
                operator.arg_sorts[side]
                for operator in reachable[parent]
                if side < operator.arity
            )

        if len(sorts) > 1:
            msg = (
                f"soft expected-form grounding cannot marginalize {space.sort!r}: "
                f"slot {slot} reaches sorts {tuple(sorted(sorts))!r}"
            )
            raise ValueError(msg)

        if not sorts:
            reachable.append(())
            continue

        expected = next(iter(sorts))
        operators = tuple(
            operator for operator in offered[slot] if operator.sort == expected
        )

        if not operators:
            msg = (
                f"soft expected-form grounding cannot marginalize {space.sort!r}: "
                f"slot {slot} reaches uninhabited sort {expected!r}"
            )
            raise ValueError(msg)

        if any(operator.is_binder for operator in operators):
            msg = (
                f"soft expected-form grounding cannot marginalize binder-reachable "
                f"sort {space.sort!r}"
            )
            raise ValueError(msg)

        reachable.append(operators)


@final
class ExpectedEmbeddingGrounding(nn.Module):
    """Classify pairs from expected logical form or sentence vectors."""

    def __init__(
        self,
        head: EntailmentHead[InferenceLabel],
        /,
        *,
        vectors: EmbeddingVectors,
        expectation: Literal["enumerate", "soft"],
        composition: Composition,
        space: Space | None,
    ) -> None:
        super().__init__()

        if not isinstance(vectors, EmbeddingVectors):
            msg = f"embedding vectors are an EmbeddingVectors value, not {vectors!r}"
            raise TypeError(msg)

        if vectors is EmbeddingVectors.EXPECTED_FORM and space is None:
            msg = "expected-form grounding requires its represented form space"
            raise ValueError(msg)

        if vectors is EmbeddingVectors.SENTENCE and space is not None:
            msg = "sentence grounding reads no represented form space"
            raise ValueError(msg)

        if vectors is EmbeddingVectors.SENTENCE and expectation != "enumerate":
            msg = "sentence-vector grounding has no soft term expectation"
            raise ValueError(msg)

        if vectors is EmbeddingVectors.EXPECTED_FORM and expectation == "soft":
            if space is None:
                msg = "soft expected-form grounding has no represented form space"
                raise ValueError(msg)

            _validate_soft_space(space)

        self.head = head
        self._vectors = vectors
        self._expectation = expectation
        self._composition = checked_composition(composition)
        self._space = space

    @property
    def labels(self) -> tuple[InferenceLabel, ...]:
        return self.head.labels

    @property
    def requires(self) -> frozenset[str]:
        return frozenset(
            {"choice_rows" if self._expectation == "soft" else "forms"}
            if self._vectors is EmbeddingVectors.EXPECTED_FORM
            else {"sentences"}
        )

    @property
    def vectors(self) -> EmbeddingVectors:
        return self._vectors

    @property
    def composition(self) -> Composition:
        return self._composition

    @property
    def expectation(self) -> Literal["enumerate", "soft"]:
        """Return how expected form vectors are computed."""
        return self._expectation

    @property
    def grounding_metadata(self) -> GroundingMetadata:
        """Return the stable direct-classifier grounding identity."""
        return GroundingMetadata(
            "expected-embedding",
            scorer="entailment-head",
            source=self._vectors.value,
            prover=None,
            prover_version=None,
            budget=None,
        )

    def _sentence(self, batch: GroundingInput, /) -> tuple[LogDistribution, Tensor]:
        pairs = batch.sentences

        if pairs is None:
            msg = "sentence grounding requires published paired sentence vectors"
            raise ValueError(msg)

        count = len(batch.pairs.premises)

        if pairs.pair_count != count:
            msg = (
                f"sentence vectors hold {pairs.pair_count} pairs, and the batch "
                f"declares {count}"
            )
            raise ValueError(msg)

        return self.head.predict(pairs), pairs.premise.tensor.new_zeros(count)

    def _expected_forms(
        self, batch: GroundingInput, /
    ) -> tuple[LogDistribution, Tensor]:
        space = self._space

        if space is None:
            msg = "expected-form grounding has no represented form space"
            raise RuntimeError(msg)

        left_rows: list[Tensor] = []
        right_rows: list[Tensor] = []
        pair_positions: list[int] = []
        residual_rows: list[Tensor] = []
        count = len(batch.pairs.premises)
        counts = (
            count,
            len(batch.pairs.hypotheses),
            len(batch.pairs.alignments),
            len(batch.pairs.labels),
        )

        if len(set(counts)) != 1:
            msg = (
                "an embedding grounding has one side and alignment set per pair, "
                f"not {counts}"
            )
            raise ValueError(msg)

        for pair in range(count):
            left = _forms(batch, batch.pairs.premises[pair])
            right = _forms(batch, batch.pairs.hypotheses[pair])
            left_vector = _expected(space, left)
            alignments = batch.pairs.alignments[pair]

            if not alignments:
                msg = "each grounded pair declares at least one alignment"
                raise ValueError(msg)

            if left_vector is None:
                continue

            for alignment in alignments:
                renamed = WeightedTerms(
                    tuple(alignment.apply(space.store, term) for term in right.terms),
                    right.log_weights,
                    right.log_residual,
                    right.estimate,
                )
                right_vector = _expected(space, renamed)

                if right_vector is None:
                    continue

                left_rows.append(left_vector.tensor[0])
                right_rows.append(right_vector.tensor[0])
                pair_positions.append(pair)
                retained = (1.0 - left.log_residual.exp()) * (
                    1.0 - right.log_residual.exp()
                )
                residual_rows.append(1.0 - retained)

        like = next(iter(batch.forms.values())).log_weights
        rows = like.new_full((count, len(self.labels)), -torch.inf)
        residual = like.new_ones(count)
        decided = torch.zeros(count, dtype=torch.bool, device=like.device)

        if left_rows:
            left = Encoded(torch.stack(left_rows), ("pair", "width"))
            right = Encoded(torch.stack(right_rows), ("pair", "width"))
            scored = self.head.predict(PairedVectors(left, right)).tensor

            for pair in range(count):
                positions = [
                    position
                    for position, owner in enumerate(pair_positions)
                    if owner == pair
                ]

                if not positions:
                    continue

                chosen = scored[positions]
                rows[pair] = torch.stack(
                    [
                        self._composition.over_alignments(chosen[:, label])
                        for label in range(len(self.labels))
                    ]
                )
                residual[pair] = self._composition.over_residuals(
                    torch.stack([residual_rows[position] for position in positions])
                )
                decided[pair] = True

        return (
            LogDistribution(rows, ("pair", "label"), decided, normalized_over="label"),
            residual,
        )

    def _soft_forms(self, batch: GroundingInput, /) -> tuple[LogDistribution, Tensor]:
        """Classify pairs from mean-field form encodings without enumeration."""
        space = self._space

        if space is None:
            msg = "soft expected-form grounding has no represented form space"
            raise RuntimeError(msg)

        count = len(batch.pairs.premises)
        counts = (
            count,
            len(batch.pairs.hypotheses),
            len(batch.pairs.alignments),
            len(batch.pairs.labels),
        )

        if len(set(counts)) != 1:
            msg = (
                "an embedding grounding has one side and alignment set per pair, "
                f"not {counts}"
            )
            raise ValueError(msg)

        left_rows: list[Tensor] = []
        right_rows: list[Tensor] = []

        for pair in range(count):
            alignments = batch.pairs.alignments[pair]

            if len(alignments) != 1 or len(alignments[0]) != 0:
                msg = (
                    "soft expectation requires exactly one identity alignment per "
                    f"pair, not {alignments!r}"
                )
                raise ValueError(msg)

            left = _soft_expected(
                space, _choice_rows(batch, batch.pairs.premises[pair])
            )
            right = _soft_expected(
                space, _choice_rows(batch, batch.pairs.hypotheses[pair])
            )
            left_rows.append(left.tensor[0])
            right_rows.append(right.tensor[0])

        if not left_rows:
            like = next(self.head.parameters())
            empty = like.new_empty((0, len(self.labels)))
            return (
                LogDistribution(empty, ("pair", "label"), normalized_over="label"),
                like.new_empty((0,)),
            )

        like = left_rows[0]
        probabilities = self.head.predict(
            PairedVectors(
                Encoded(torch.stack(left_rows), ("pair", "width")),
                Encoded(torch.stack(right_rows), ("pair", "width")),
            )
        )
        return probabilities, like.new_zeros(count)

    def ground(self, batch: GroundingInput, /) -> GroundedBatch:
        """Return a complete differentiable distribution over inference labels."""
        if self._vectors is EmbeddingVectors.SENTENCE:
            probabilities, residual = self._sentence(batch)
        elif self._expectation == "soft":
            probabilities, residual = self._soft_forms(batch)
        else:
            probabilities, residual = self._expected_forms(batch)

        decided = (
            probabilities.mask
            if probabilities.mask is not None
            else torch.ones(
                probabilities.tensor.shape[0],
                dtype=torch.bool,
                device=probabilities.tensor.device,
            )
        )
        cells = probabilities.tensor.new_zeros(
            (probabilities.tensor.shape[0], len(InferenceCell))
        )

        return GroundedBatch(
            probabilities,
            labels=self.labels,
            complete=True,
            residual=residual,
            decided=decided,
            cells=CellMass(cells),
            gradient_path=GradientPath.PATHWISE,
            baseline=None,
            auxiliary=None,
            ledger=EvidenceLedger.empty(),
            variance=None,
            reward=None,
        )

    def site_map(self) -> SiteMap:
        return SiteMap.custom(ComponentId("expected-embedding-grounding"), {})

    def open_step(self) -> None:
        """Open the next step; this scorer holds no step-local state."""

    @override
    def __repr__(self) -> str:
        return (
            f"ExpectedEmbeddingGrounding(vectors={self._vectors.value!r}, "
            f"expectation={self._expectation!r})"
        )


def _build(
    settings: ExpectedEmbeddingSettings, context: BuildContext, /
) -> ExpectedEmbeddingGrounding:
    """Build an inference classifier at the width owned by its vector producer."""
    vectors = EmbeddingVectors(settings.vectors)
    labeling = context.producer("labeling")

    if not isinstance(labeling, InferenceLabeling):
        msg = f"an embedding grounding reads InferenceLabeling, not {labeling!r}"
        raise TypeError(msg)

    if vectors is EmbeddingVectors.EXPECTED_FORM:
        produced = context.producer(settings.readout)

        if not isinstance(produced, Space):
            msg = f"expected-form grounding reads a represented Space, not {produced!r}"
            raise TypeError(msg)

        space = produced
        width = space.encoder.output_width
    else:
        produced = context.producer("sentences")

        if not isinstance(produced, SubwordPooler):
            msg = f"sentence grounding reads a SubwordPooler, not {produced!r}"
            raise TypeError(msg)

        space = None
        width = produced.output_width

    return ExpectedEmbeddingGrounding(
        EntailmentHead(labeling.labels, width=width, hidden=settings.hidden),
        vectors=vectors,
        expectation=settings.expectation,
        composition=composition_from_settings(*settings.composition),
        space=space,
    )


PLUGIN: Final = ComponentPlugin(spec=ExpectedEmbeddingSpec, build=_build)
"""The expected-form or sentence-vector inference-grounding plugin."""
