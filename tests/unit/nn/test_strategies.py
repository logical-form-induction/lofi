from typing import Final, NamedTuple, override

import pytest
import torch

from lofi._checks import NumericalFailureError
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.ports import NoAdmissibleChoiceError
from lofi.nn.sampling import KEEP_OFFERED, SamplingStrategy, sampling_strategy
from lofi.nn.strategies import (
    beam,
    exact_matches,
    exhaustive,
    greedy,
    paired_exhaustive,
    sample,
    score,
)
from lofi.spec.decoding import SamplingSettings
from lofi.terms import (
    BatchEncoding,
    ChoiceIndex,
    Encoding,
    HoleContext,
    Operator,
    Signature,
    Term,
    TermStore,
    Vocabulary,
)


class State(NamedTuple):
    decisions: tuple[int, ...]


SIGNATURE: Final = Signature(
    "strategies-test",
    ("Root", "Leaf"),
    (
        Operator("leaf", "Leaf"),
        Operator("other", "Leaf"),
        Operator("wrap", "Root", arg_sorts=("Leaf",)),
    ),
)
CHOICES: Final = ChoiceIndex(
    SIGNATURE,
    depth_bounds={"Root": 2, "Leaf": 2},
)
STORE: Final = TermStore(SIGNATURE)
ENCODED: Final = Encoded(torch.tensor([[0.25]]), ("term", "width"))


class Decoder:
    def __init__(self) -> None:
        logits = torch.full((CHOICES.width,), -torch.inf, dtype=torch.float64)
        logits[CHOICES.index("wrap")] = 0.0
        self._root = logits.log_softmax(dim=0)
        logits = torch.full((CHOICES.width,), -torch.inf, dtype=torch.float64)
        logits[CHOICES.index("leaf")] = 2.0
        logits[CHOICES.index("other")] = 1.0
        self._leaf = logits.log_softmax(dim=0)

    def start(self, inputs: Encoded, /) -> State:
        assert inputs is ENCODED
        return State(())

    def weigh(self, state: State, /, *, context: HoleContext) -> LogDistribution:
        del context
        row = self._root if not state.decisions else self._leaf
        mask = row.isfinite()
        return LogDistribution(row, ("choice",), mask, normalized_over="choice")

    def fill(self, state: State, /, *, choices: torch.Tensor) -> State:
        return State((*state.decisions, int(choices.item())))

    def admits(self, term: Term, /) -> bool:
        del term
        return True

    def rows(self, inputs: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        assert inputs is ENCODED
        assert len(terms.rows) == 1
        rows = torch.stack((self._root, self._leaf)).unsqueeze(0)
        return LogDistribution(
            rows,
            ("term", "slot", "choice"),
            normalized_over="choice",
        )


def test_shared_scoring_preserves_an_empty_term_batch() -> None:
    class EmptyRows:
        def rows(self, inputs: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
            del terms
            rows = inputs.tensor[:, :1].unsqueeze(-1).expand(-1, 2, CHOICES.width)
            return LogDistribution(
                rows,
                ("term", "slot", "choice"),
                normalized_over="choice",
            )

    empty = BatchEncoding(())
    inputs = Encoded(torch.empty((0, 1), requires_grad=True), ("term", "width"))
    weighed = score(EmptyRows(), inputs, empty, index=CHOICES)

    assert weighed.axes == ("term",)
    assert weighed.tensor.shape == (0,)
    assert weighed.tensor.grad_fn is not None


def test_shared_scoring_reduces_choice_legality_to_one_mask_per_term() -> None:
    class MaskedRows(Decoder):
        @override
        def rows(self, inputs: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
            rows = super().rows(inputs, terms=terms)
            return LogDistribution(
                rows.tensor,
                rows.axes,
                rows.tensor.isfinite(),
                normalized_over="choice",
            )

    term = STORE.op(
        SIGNATURE.operator(SIGNATURE.code("wrap")),
        STORE.op(SIGNATURE.operator(SIGNATURE.code("leaf"))),
    )
    weighed = score(
        MaskedRows(),
        ENCODED,
        BatchEncoding(
            (
                Encoding(
                    (SIGNATURE.code("wrap"), SIGNATURE.code("leaf")),
                    (Encoding.NO_PARENT, 0),
                    (0, 1),
                    (True, True),
                ),
            )
        ),
        index=CHOICES,
    )

    assert STORE.contains(term)
    assert weighed.mask is not None
    assert weighed.mask.shape == (1,)
    assert bool(weighed.mask[0])


def test_teacher_forced_exact_matching_requires_every_occupied_slot() -> None:
    def encoded(leaf: str) -> BatchEncoding:
        return BatchEncoding(
            (
                Encoding(
                    (SIGNATURE.code("wrap"), SIGNATURE.code(leaf)),
                    (Encoding.NO_PARENT, 0),
                    (0, 1),
                    (True, True),
                ),
            )
        )

    assert exact_matches(Decoder(), ENCODED, encoded("leaf"), index=CHOICES).tolist()
    assert not any(
        exact_matches(Decoder(), ENCODED, encoded("other"), index=CHOICES).tolist()
    )


def test_shared_scoring_refuses_a_foreign_choice_index() -> None:
    foreign_signature = Signature(
        "foreign-strategies-test",
        ("Root", "Leaf"),
        (
            Operator("leaf", "Leaf"),
            Operator("other", "Leaf"),
            Operator("extra", "Leaf"),
            Operator("wrap", "Root", arg_sorts=("Leaf",)),
        ),
    )
    foreign = ChoiceIndex(
        foreign_signature,
        depth_bounds={"Root": 2, "Leaf": 2},
    )
    terms = BatchEncoding(
        (
            Encoding(
                (SIGNATURE.code("wrap"), SIGNATURE.code("leaf")),
                (Encoding.NO_PARENT, 0),
                (0, 1),
                (True, True),
            ),
        )
    )

    with pytest.raises(ValueError, match="scoring index's"):
        score(Decoder(), ENCODED, terms, index=foreign)


def test_exhaustive_weights_and_residual_partition_probability_mass() -> None:
    result = exhaustive(
        Decoder(),
        ENCODED,
        index=CHOICES,
        store=STORE,
        root_sort="Root",
        max_nodes=2,
        residual_mass=0.0,
        exact_residual=1e-6,
        cap=8,
    )

    assert isinstance(result, WeightedTerms)
    assert result.estimate is Estimate.EXACT
    assert len(result.terms) == 2
    assert all(STORE.contains(term) for term in result.terms)
    assert torch.allclose(
        result.log_weights.exp().sum() + result.log_residual.exp(),
        torch.tensor(1.0, dtype=torch.float64),
    )


def test_an_exhaustive_cap_reports_the_pruned_mass_as_truncated() -> None:
    result = exhaustive(
        Decoder(),
        ENCODED,
        index=CHOICES,
        store=STORE,
        root_sort="Root",
        max_nodes=2,
        residual_mass=0.0,
        exact_residual=1e-6,
        cap=1,
    )

    assert result.estimate is Estimate.TRUNCATED
    assert len(result) == 1
    assert result.log_residual.exp() > 0.0


def test_paired_exhaustive_advances_one_shared_term_support() -> None:
    left, right = paired_exhaustive(
        Decoder(),
        ENCODED,
        Decoder(),
        ENCODED,
        index=CHOICES,
        store=STORE,
        root_sort="Root",
        max_nodes=2,
        residual_mass=0.0,
        exact_residual=1e-6,
        cap=8,
    )

    assert left.estimate is Estimate.EXACT
    assert right.estimate is Estimate.EXACT
    assert left.terms == right.terms
    assert all(STORE.contains(term) for term in left.terms)
    assert torch.equal(left.log_weights, right.log_weights)


def test_paired_exhaustive_persists_both_residuals_at_one_shared_cap() -> None:
    left, right = paired_exhaustive(
        Decoder(),
        ENCODED,
        Decoder(),
        ENCODED,
        index=CHOICES,
        store=STORE,
        root_sort="Root",
        max_nodes=2,
        residual_mass=0.0,
        exact_residual=1e-6,
        cap=1,
    )

    assert left.estimate is Estimate.TRUNCATED
    assert right.estimate is Estimate.TRUNCATED
    assert left.terms == right.terms
    assert left.log_residual.exp() > 0.0
    assert right.log_residual.exp() > 0.0


class DivergedDecoder(Decoder):
    @override
    def weigh(self, state: State, /, *, context: HoleContext) -> LogDistribution:
        row = super().weigh(state, context=context).tensor.clone()

        if state.decisions:
            row[CHOICES.index("other")] = torch.nan

        return LogDistribution(row, ("choice",), normalized_over="choice")


def _decode(search: str, decoder: Decoder, /) -> WeightedTerms:
    match search:
        case "exhaustive":
            return exhaustive(
                decoder,
                ENCODED,
                index=CHOICES,
                store=STORE,
                root_sort="Root",
                max_nodes=2,
                residual_mass=0.0,
                exact_residual=1e-6,
                cap=8,
            )
        case "beam":
            return beam(
                decoder,
                ENCODED,
                index=CHOICES,
                store=STORE,
                root_sort="Root",
                max_nodes=2,
                width=2,
            )
        case "greedy":
            return greedy(
                decoder,
                ENCODED,
                index=CHOICES,
                store=STORE,
                root_sort="Root",
                max_nodes=2,
            )
        case _:
            return sample(
                decoder,
                ENCODED,
                index=CHOICES,
                store=STORE,
                root_sort="Root",
                max_nodes=2,
                restriction=sampling_strategy(SamplingSettings("top-k", 1.0, 2, None)),
                draws=1,
            )


@pytest.mark.parametrize("search", ["exhaustive", "beam", "greedy", "sample"])
def test_a_diverged_choice_weight_is_refused_rather_than_skipped(search: str) -> None:
    with pytest.raises(NumericalFailureError, match="not finite"):
        _decode(search, DivergedDecoder())


def test_paired_exhaustive_refuses_a_diverged_choice_weight_on_either_side() -> None:
    with pytest.raises(NumericalFailureError, match="not finite"):
        paired_exhaustive(
            Decoder(),
            ENCODED,
            DivergedDecoder(),
            ENCODED,
            index=CHOICES,
            store=STORE,
            root_sort="Root",
            max_nodes=2,
            residual_mass=0.0,
            exact_residual=1e-6,
            cap=8,
        )


def test_beam_weights_are_attached_and_equal_exhaustive_path_weights() -> None:
    decoder = Decoder()
    complete = exhaustive(
        decoder,
        ENCODED,
        index=CHOICES,
        store=STORE,
        root_sort="Root",
        max_nodes=2,
        residual_mass=0.0,
        exact_residual=1e-6,
        cap=8,
    )
    selected = beam(
        decoder,
        ENCODED,
        index=CHOICES,
        store=STORE,
        root_sort="Root",
        max_nodes=2,
        width=1,
    )

    assert selected.estimate is Estimate.TRUNCATED
    assert selected.terms == complete.terms[:1]
    assert torch.equal(selected.log_weights, complete.log_weights[:1])

    term = selected.terms[0]
    encoded = BatchEncoding(
        (
            Encoding(
                (SIGNATURE.code(term.op), SIGNATURE.code(term.args[0].op)),
                (Encoding.NO_PARENT, 0),
                (0, 1),
                (True, True),
            ),
        )
    )
    assert torch.equal(
        score(decoder, ENCODED, encoded, index=CHOICES).tensor,
        selected.log_weights,
    )


def test_a_beam_discards_completed_terms_refused_by_the_decoder_policy() -> None:
    class OtherDecoder(Decoder):
        @override
        def admits(self, term: Term, /) -> bool:
            return term.args[0].op.name == "other"

    selected = beam(
        OtherDecoder(),
        ENCODED,
        index=CHOICES,
        store=STORE,
        root_sort="Root",
        max_nodes=2,
        width=1,
    )

    assert selected.terms == (
        STORE.op(SIGNATURE.by_name("wrap"), STORE.op(SIGNATURE.by_name("other"))),
    )


def test_greedy_and_sample_build_well_sorted_terms() -> None:
    greedy_result = greedy(
        Decoder(),
        ENCODED,
        index=CHOICES,
        store=STORE,
        root_sort="Root",
        max_nodes=2,
    )
    sampled = sample(
        Decoder(),
        ENCODED,
        index=CHOICES,
        store=STORE,
        root_sort="Root",
        max_nodes=2,
        restriction=sampling_strategy(SamplingSettings("top-k", 1.0, 1, None)),
        draws=3,
    )

    assert greedy_result.terms[0].op.sort == "Root"
    assert greedy_result.terms[0].args[0].op.sort == "Leaf"
    assert sampled.estimate is Estimate.SAMPLED
    assert all(term == greedy_result.terms[0] for term in sampled.terms)


def test_sample_retries_a_dead_frontier_within_its_declared_bounds() -> None:
    class RetryingDecoder(Decoder):
        def __init__(self) -> None:
            super().__init__()
            self.starts = 0
            self.dead = False

        @override
        def start(self, inputs: Encoded, /) -> State:
            self.dead = self.starts == 0
            self.starts += 1
            return super().start(inputs)

        @override
        def weigh(self, state: State, /, *, context: HoleContext) -> LogDistribution:
            if self.dead and state.decisions:
                msg = "the first sampled frontier has no admitted leaf"
                raise NoAdmissibleChoiceError(msg)

            return super().weigh(state, context=context)

    decoder = RetryingDecoder()
    result = sample(
        decoder,
        ENCODED,
        index=CHOICES,
        store=STORE,
        root_sort="Root",
        max_nodes=2,
        restriction=sampling_strategy(SamplingSettings("top-k", 1.0, 1, None)),
        draws=1,
    )

    assert decoder.starts == 2
    assert len(result) == 1


def test_a_clamp_forces_the_named_subterm() -> None:
    clamp = STORE.op(
        SIGNATURE.operator(SIGNATURE.code("wrap")),
        STORE.op(SIGNATURE.operator(SIGNATURE.code("other"))),
    )
    result = greedy(
        Decoder(),
        ENCODED,
        index=CHOICES,
        store=STORE,
        root_sort="Root",
        max_nodes=2,
        clamp=clamp,
    )

    assert result.terms == (clamp,)


def test_a_sample_clamp_normalizes_the_draw_but_keeps_the_model_weight() -> None:
    clamp = STORE.op(
        SIGNATURE.operator(SIGNATURE.code("wrap")),
        STORE.op(SIGNATURE.operator(SIGNATURE.code("other"))),
    )
    result = sample(
        Decoder(),
        ENCODED,
        index=CHOICES,
        store=STORE,
        root_sort="Root",
        max_nodes=2,
        restriction=SamplingStrategy(KEEP_OFFERED),
        draws=1,
        clamp=clamp,
    )

    expected = torch.tensor([2.0, 1.0], dtype=torch.float64).log_softmax(dim=0)[1]
    assert result.terms == (clamp,)
    assert torch.allclose(result.log_weights, expected.unsqueeze(0))


def test_payload_choices_build_the_selected_constant() -> None:
    signature = Signature(
        "payload",
        ("T",),
        (),
        vocabularies=(Vocabulary("T", ("constant",)),),
    )
    store = TermStore(signature)
    choices = ChoiceIndex(signature, depth_bounds={"T": 1})
    chosen = choices.index("#const[T]", 0)

    class PayloadDecoder:
        def start(self, inputs: Encoded, /) -> State:
            assert inputs is ENCODED
            return State(())

        def weigh(self, state: State, /, *, context: HoleContext) -> LogDistribution:
            del state, context
            row = torch.full((choices.width,), -torch.inf, dtype=torch.float64)
            row[chosen] = 0.0
            return LogDistribution(
                row,
                ("choice",),
                row.isfinite(),
                normalized_over="choice",
            )

        def fill(self, state: State, /, *, choices: torch.Tensor) -> State:
            return State((*state.decisions, int(choices.item())))

        def admits(self, term: Term, /) -> bool:
            del term
            return True

        def rows(self, inputs: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
            assert inputs is ENCODED
            assert len(terms.rows) == 1
            row = torch.full((1, 1, choices.width), -torch.inf, dtype=torch.float64)
            row[0, 0, chosen] = 0.0
            return LogDistribution(
                row,
                ("term", "slot", "choice"),
                normalized_over="choice",
            )

    result = greedy(
        PayloadDecoder(),
        ENCODED,
        index=choices,
        store=store,
        root_sort="T",
        max_nodes=1,
    )

    assert result.terms == (store.const("T", "constant"),)
    encoded = BatchEncoding(
        (
            Encoding(
                (signature.code("#const[T]"),),
                (Encoding.NO_PARENT,),
                (0,),
                (True,),
                (0,),
            ),
        )
    )
    assert torch.equal(
        score(PayloadDecoder(), ENCODED, encoded, index=choices).tensor,
        torch.zeros(1, dtype=torch.float64),
    )


@pytest.mark.parametrize("exact_residual", [-0.1, 1.0])
def test_an_exactness_residual_outside_the_unit_interval_is_refused(
    exact_residual: float,
) -> None:
    with pytest.raises(ValueError, match="exactness residual"):
        exhaustive(
            Decoder(),
            ENCODED,
            index=CHOICES,
            store=STORE,
            root_sort="Root",
            max_nodes=2,
            residual_mass=0.0,
            exact_residual=exact_residual,
            cap=8,
        )
