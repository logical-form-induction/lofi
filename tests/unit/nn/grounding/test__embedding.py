from types import MappingProxyType
from typing import Final

import pytest
import torch

from lofi._types import ReadoutId
from lofi.lf import THREE_WAY, ArgumentRenaming, InferenceLabel
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.arity import ArityHeadScheme
from lofi.nn.codec import NeuralCodecView
from lofi.nn.encoders import TreeRnnEncoder
from lofi.nn.entailment import EntailmentHead, PairedVectors
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.grounding import (
    EmbeddingVectors,
    ExpectedEmbeddingGrounding,
    GradientPath,
    GroundingInput,
    GroundingQuery,
    Marginal,
    Phase,
)
from lofi.nn.sites import ComponentId
from lofi.nn.spaces import Space
from lofi.terms import (
    BatchEncoding,
    BindingSite,
    ChoiceIndex,
    CompleteTree,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
    TermStore,
)

ATOM: Final = Operator("a", "T", family="primitive")
OTHER_ATOM: Final = Operator("b", "T", family="primitive")
PAIR: Final = Operator("pair", "T", arg_sorts=("T", "T"), family="constructor")
SIGNATURE: Final = Signature("soft-grounding", ("T",), (ATOM, OTHER_ATOM, PAIR))
INDEX: Final = OperatorIndex(SIGNATURE)
LAYOUT: Final = CompleteTree(SIGNATURE, SIGNATURE.operators, order="bf", depth=2)
CODEC: Final = TermCodec(INDEX, LAYOUT)
VIEW: Final = NeuralCodecView(CODEC)
CHOICES: Final = ChoiceIndex(SIGNATURE, depth_bounds={"T": LAYOUT.depth})
STORE: Final = TermStore(SIGNATURE)


class MarginalIdentity:
    def marginal_rows(self, inputs: Encoded, /, *, root_sort: str) -> LogDistribution:
        del inputs, root_sort
        msg = "grounding must consume already-published marginals"
        raise AssertionError(msg)


def represented_space() -> Space:
    torch.manual_seed(17)
    encoder = TreeRnnEncoder(
        codec=VIEW,
        choices=CHOICES,
        component=ComponentId("soft-grounding-encoder"),
        scheme=ArityHeadScheme(width=5, reached=SIGNATURE.operators),
        cell="gru",
        layers=1,
        norm_cap=None,
        inventory="uncached",
    ).to(torch.float64)

    return Space(
        "T",
        codec=VIEW,
        choices=CHOICES,
        store=STORE,
        max_nodes=VIEW.capacity,
        encoder=encoder,
        identity=MarginalIdentity(),
        combinators=None,
    )


def root_only_space(*, sort: str = "Root") -> Space:
    signature = Signature(
        "root-only-soft-grounding",
        ("Root", "Branch"),
        (
            Operator("root", "Root"),
            Operator("leaf", "Branch"),
            Operator("branch", "Branch", arg_sorts=("Branch", "Branch")),
        ),
    )
    index = OperatorIndex(signature)
    layout = CompleteTree(signature, signature.operators, order="bf", depth=2)
    codec = NeuralCodecView(TermCodec(index, layout))
    choices = ChoiceIndex(signature, depth_bounds={"Root": 1, "Branch": layout.depth})
    encoder = TreeRnnEncoder(
        codec=codec,
        choices=choices,
        component=ComponentId("root-only-soft-grounding"),
        scheme=ArityHeadScheme(width=4, reached=signature.operators),
        cell="gru",
        layers=1,
        norm_cap=None,
        inventory="uncached",
    )

    return Space(
        sort,
        codec=codec,
        choices=choices,
        store=TermStore(signature),
        max_nodes=codec.capacity,
        encoder=encoder,
        identity=MarginalIdentity(),
        combinators=None,
    )


def one_hot(terms: BatchEncoding) -> LogDistribution:
    codes = torch.tensor(
        tuple(CHOICES.codes(row) for row in terms.rows), dtype=torch.int64
    )
    probabilities = torch.nn.functional.one_hot(codes, CHOICES.width).double()
    return LogDistribution(
        probabilities.log(),
        ("term", "slot", "choice"),
        normalized_over="choice",
    )


class Pairs:
    premises = (ReadoutId("premise"), ReadoutId("premise"))
    hypotheses = (ReadoutId("hypothesis"), ReadoutId("hypothesis"))
    alignments = (
        (ArgumentRenaming.identity(),),
        (ArgumentRenaming.identity(),),
    )
    labels = (InferenceLabel.ENTAILS, InferenceLabel.NEUTRAL)


class OnePair:
    premises = (ReadoutId("premise"),)
    hypotheses = (ReadoutId("hypothesis"),)
    alignments = ((ArgumentRenaming.identity(),),)
    labels = (InferenceLabel.ENTAILS,)


def test_sentence_grounding_publishes_complete_pathwise_predictions() -> None:
    grounding = ExpectedEmbeddingGrounding(
        EntailmentHead(THREE_WAY.labels, width=3, hidden=4),
        vectors=EmbeddingVectors.SENTENCE,
        expectation="enumerate",
        composition=Marginal(),
        space=None,
    )
    sentences = PairedVectors(
        Encoded(torch.randn(2, 3), ("pair", "width")),
        Encoded(torch.randn(2, 3), ("pair", "width")),
    )
    result = grounding.ground(
        GroundingInput(
            MappingProxyType({}),
            choice_rows=MappingProxyType({}),
            pairs=Pairs(),
            query=GroundingQuery.FULL,
            sentences=sentences,
            phase=Phase.TRAIN,
        )
    )

    assert result.complete
    assert result.gradient_path is GradientPath.PATHWISE
    assert result.decided.tolist() == [True, True]
    assert result.residual.tolist() == pytest.approx([0.0, 0.0])
    assert result.log_probabilities.tensor.exp().sum(1).tolist() == pytest.approx(
        [1.0, 1.0]
    )


def test_sentence_grounding_requires_sentence_vectors() -> None:
    grounding = ExpectedEmbeddingGrounding(
        EntailmentHead(THREE_WAY.labels, width=3),
        vectors=EmbeddingVectors.SENTENCE,
        expectation="enumerate",
        composition=Marginal(),
        space=None,
    )

    with pytest.raises(ValueError, match="requires published"):
        grounding.ground(
            GroundingInput(
                MappingProxyType({}),
                choice_rows=MappingProxyType({}),
                pairs=Pairs(),
                query=GroundingQuery.FULL,
                sentences=None,
                phase=Phase.EVALUATE,
            )
        )


def test_one_hot_soft_expectation_matches_complete_term_enumeration() -> None:
    space = represented_space()
    encoder = space.encoder
    head = EntailmentHead(THREE_WAY.labels, width=encoder.output_width, hidden=7).to(
        torch.float64
    )
    enumerate_grounding = ExpectedEmbeddingGrounding(
        head,
        vectors=EmbeddingVectors.EXPECTED_FORM,
        expectation="enumerate",
        composition=Marginal(),
        space=space,
    )
    soft_grounding = ExpectedEmbeddingGrounding(
        head,
        vectors=EmbeddingVectors.EXPECTED_FORM,
        expectation="soft",
        composition=Marginal(),
        space=space,
    )
    premise = STORE.op(PAIR, STORE.op(ATOM), STORE.op(OTHER_ATOM))
    hypothesis = STORE.op(PAIR, STORE.op(OTHER_ATOM), STORE.op(ATOM))
    premise_encoding = CODEC.encode_batch((premise,))
    hypothesis_encoding = CODEC.encode_batch((hypothesis,))
    forms = MappingProxyType(
        {
            ReadoutId("premise"): WeightedTerms(
                (premise,),
                torch.zeros(1, dtype=torch.float64),
                torch.tensor(-torch.inf),
                Estimate.EXACT,
            ),
            ReadoutId("hypothesis"): WeightedTerms(
                (hypothesis,),
                torch.zeros(1, dtype=torch.float64),
                torch.tensor(-torch.inf),
                Estimate.EXACT,
            ),
        }
    )
    rows = MappingProxyType(
        {
            ReadoutId("premise"): one_hot(premise_encoding),
            ReadoutId("hypothesis"): one_hot(hypothesis_encoding),
        }
    )

    enumerated = enumerate_grounding.ground(
        GroundingInput(
            forms,
            choice_rows=MappingProxyType({}),
            pairs=OnePair(),
            query=GroundingQuery.FULL,
            sentences=None,
            phase=Phase.TRAIN,
        )
    )
    softened = soft_grounding.ground(
        GroundingInput(
            MappingProxyType({}),
            choice_rows=rows,
            pairs=OnePair(),
            query=GroundingQuery.FULL,
            sentences=None,
            phase=Phase.TRAIN,
        )
    )

    assert softened.residual.tolist() == pytest.approx([0.0])
    assert torch.allclose(
        softened.log_probabilities.tensor,
        enumerated.log_probabilities.tensor,
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "alignments",
    [
        (
            ArgumentRenaming.identity(),
            ArgumentRenaming.identity(),
        ),
        (ArgumentRenaming({"x": "y"}),),
    ],
    ids=("multiple", "nonidentity"),
)
def test_soft_expectation_refuses_unsupported_alignments(
    alignments: tuple[ArgumentRenaming, ...],
) -> None:
    space = represented_space()
    grounding = ExpectedEmbeddingGrounding(
        EntailmentHead(THREE_WAY.labels, width=space.encoder.output_width).to(
            torch.float64
        ),
        vectors=EmbeddingVectors.EXPECTED_FORM,
        expectation="soft",
        composition=Marginal(),
        space=space,
    )

    class UnsupportedPair:
        premises = (ReadoutId("premise"),)
        hypotheses = (ReadoutId("hypothesis"),)
        labels = (InferenceLabel.ENTAILS,)

        @property
        def alignments(self) -> tuple[tuple[ArgumentRenaming, ...], ...]:
            return (alignments,)

    with pytest.raises(ValueError, match="exactly one identity alignment"):
        grounding.ground(
            GroundingInput(
                MappingProxyType({}),
                choice_rows=MappingProxyType({}),
                pairs=UnsupportedPair(),
                query=GroundingQuery.FULL,
                sentences=None,
                phase=Phase.TRAIN,
            )
        )


def test_soft_expectation_refuses_a_binder_reachable_layout_at_wiring() -> None:
    binder = Operator(
        "bind",
        "T",
        arg_sorts=("T",),
        binding=(BindingSite((0,), "T"),),
    )
    signature = Signature("binding-soft-grounding", ("T",), (ATOM, binder))
    index = OperatorIndex(signature)
    layout = CompleteTree(signature, signature.operators, order="bf", depth=2)
    codec = NeuralCodecView(TermCodec(index, layout))
    choices = ChoiceIndex(signature, depth_bounds={"T": layout.depth})
    encoder = TreeRnnEncoder(
        codec=codec,
        choices=choices,
        component=ComponentId("binding-soft-grounding"),
        scheme=ArityHeadScheme(width=4, reached=signature.operators),
        cell="gru",
        layers=1,
        norm_cap=None,
        inventory="uncached",
    )
    space = Space(
        "T",
        codec=codec,
        choices=choices,
        store=TermStore(signature),
        max_nodes=codec.capacity,
        encoder=encoder,
        identity=MarginalIdentity(),
        combinators=None,
    )

    with pytest.raises(ValueError, match="binder-reachable"):
        ExpectedEmbeddingGrounding(
            EntailmentHead(THREE_WAY.labels, width=encoder.output_width),
            vectors=EmbeddingVectors.EXPECTED_FORM,
            expectation="soft",
            composition=Marginal(),
            space=space,
        )


def test_soft_expectation_refuses_choice_dependent_child_sorts_at_wiring() -> None:
    signature = Signature(
        "mixed-soft-grounding",
        ("Root", "Left", "Right"),
        (
            Operator("left", "Root", arg_sorts=("Left",)),
            Operator("right", "Root", arg_sorts=("Right",)),
            Operator("l", "Left"),
            Operator("r", "Right"),
        ),
    )
    index = OperatorIndex(signature)
    layout = CompleteTree(signature, signature.operators, order="bf", depth=2)
    codec = NeuralCodecView(TermCodec(index, layout))
    choices = ChoiceIndex(signature, depth_bounds={"Root": 2, "Left": 1, "Right": 1})
    encoder = TreeRnnEncoder(
        codec=codec,
        choices=choices,
        component=ComponentId("mixed-soft-grounding"),
        scheme=ArityHeadScheme(width=4, reached=signature.operators),
        cell="gru",
        layers=1,
        norm_cap=None,
        inventory="uncached",
    )
    space = Space(
        "Root",
        codec=codec,
        choices=choices,
        store=TermStore(signature),
        max_nodes=codec.capacity,
        encoder=encoder,
        identity=MarginalIdentity(),
        combinators=None,
    )

    with pytest.raises(ValueError, match="reaches sorts"):
        ExpectedEmbeddingGrounding(
            EntailmentHead(THREE_WAY.labels, width=encoder.output_width),
            vectors=EmbeddingVectors.EXPECTED_FORM,
            expectation="soft",
            composition=Marginal(),
            space=space,
        )


def test_soft_expectation_allows_unreachable_descendant_slots() -> None:
    space = root_only_space()
    grounding = ExpectedEmbeddingGrounding(
        EntailmentHead(THREE_WAY.labels, width=space.encoder.output_width),
        vectors=EmbeddingVectors.EXPECTED_FORM,
        expectation="soft",
        composition=Marginal(),
        space=space,
    )

    assert grounding.requires == frozenset({"choice_rows"})


def test_soft_expectation_refuses_an_undeclared_root_sort_at_wiring() -> None:
    space = root_only_space(sort="Missing")

    with pytest.raises(ValueError, match="undeclared root sort 'Missing'"):
        ExpectedEmbeddingGrounding(
            EntailmentHead(THREE_WAY.labels, width=space.encoder.output_width),
            vectors=EmbeddingVectors.EXPECTED_FORM,
            expectation="soft",
            composition=Marginal(),
            space=space,
        )
