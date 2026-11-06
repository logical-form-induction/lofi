from typing import TYPE_CHECKING, cast, override

import pytest
import torch

from lofi.judge import Action, ActionSet
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.ports import ForwardMode, ForwardOptions
from lofi.nn.spaces import ChartCombine, CombinatorFamily, Concatenate

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.judge import RuleSet
    from lofi.nn.ports import TermEncoder
    from lofi.terms import Term

from lofi.terms import (
    BatchEncoding,
    ChoiceIndex,
    CompleteTree,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
    TermStore,
)

IDENTITY = Action("identity")
APPLY = Action("apply", "apply")
COMPOSE = Action("compose", "compose")
ACTIONS = ActionSet((IDENTITY, APPLY, COMPOSE), IDENTITY)
RAISE = Action("raise", "raise")
UNARY_ACTIONS = ActionSet((IDENTITY, RAISE), IDENTITY)
OPTIONS = ForwardOptions(ForwardMode.OBSERVE)
SIGNATURE = Signature("space-fixture", ("T",), (Operator("a", "T"),))
CODEC = TermCodec(
    OperatorIndex(SIGNATURE),
    CompleteTree(SIGNATURE, SIGNATURE.operators, order="df", depth=1),
)
CHOICES = ChoiceIndex(SIGNATURE, depth_bounds={"T": CODEC.capacity})
STORE = TermStore(SIGNATURE)
TERM = STORE.op(SIGNATURE.by_name("a"))


class FixtureRule:
    def __init__(self, name: str, arity: int) -> None:
        self.name = name
        self.arity = arity
        self.directions = frozenset({"forward"})
        self.forward_is_complete = True

    def apply(self, inputs: tuple[Term, ...], /) -> tuple[Term, ...]:
        return (inputs[0],)


class MultivaluedFixtureRule(FixtureRule):
    """Return two licensed conclusions for one unary premise."""

    @override
    def apply(self, inputs: tuple[Term, ...], /) -> tuple[Term, ...]:
        return (inputs[0], inputs[0])


class RecordingEncoder:
    output_width = 2

    def __init__(self) -> None:
        self.received: BatchEncoding | None = None

    def encode(self, terms: BatchEncoding, /) -> Encoded:
        self.received = terms
        values = torch.tensor([[float(row.symbols[0]), 1.0] for row in terms.rows])

        return Encoded(values, ("term", "width"))


class RecordingDecoder:
    def __init__(self) -> None:
        self.received: Encoded | None = None
        self.scored: tuple[Encoded, BatchEncoding] | None = None

    def marginal_rows(self, encoded: Encoded, /, *, root_sort: str) -> LogDistribution:
        del root_sort
        self.received = encoded

        return LogDistribution(
            torch.log_softmax(
                torch.ones(encoded.tensor.shape[0], CODEC.capacity, CHOICES.width),
                dim=-1,
            ),
            ("term", "slot", "choice"),
            normalized_over="choice",
        )

    def rows(
        self,
        encoded: Encoded,
        /,
        *,
        terms: BatchEncoding,
    ) -> LogDistribution:
        """Record one teacher-forced row request."""
        self.scored = encoded, terms

        return LogDistribution(
            torch.log_softmax(
                torch.ones(encoded.tensor.shape[0], CODEC.capacity, CHOICES.width),
                dim=-1,
            ),
            ("term", "slot", "choice"),
            normalized_over="choice",
        )


class RecordingChart:
    def __init__(self) -> None:
        self.received: tuple[Tensor, Tensor] | None = None

    def compose(self, left: Tensor, right: Tensor) -> Tensor:
        self.received = left, right

        return left + 10.0 * right


def _family(
    merge: Concatenate | ChartCombine,
) -> tuple[CombinatorFamily, RecordingDecoder]:
    decoder = RecordingDecoder()
    encoder = RecordingEncoder()

    return (
        CombinatorFamily(
            {
                APPLY: decoder,
                COMPOSE: decoder,
            },
            actions=ACTIONS,
            rules=cast(
                "RuleSet[Term]",
                (FixtureRule("apply", 2), FixtureRule("compose", 2)),
            ),
            merge=merge,
            codec=CODEC,
            encoder=cast("TermEncoder", encoder),
            identity=decoder,
            index=CHOICES,
            root_sort="T",
        ),
        decoder,
    )


def test_concatenate_is_the_declared_decoder_input() -> None:
    family, decoder = _family(Concatenate(2))
    left = Encoded(torch.tensor([[1.0, 2.0]]), ("term", "width"))
    right = Encoded(torch.tensor([[3.0, 4.0]]), ("term", "width"))

    family.compose(APPLY, left, right=right)

    assert decoder.received is not None
    assert decoder.received.axes == left.axes
    assert torch.equal(decoder.received.tensor, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))


def test_chart_combine_reads_the_published_composition() -> None:
    chart = RecordingChart()
    family, decoder = _family(ChartCombine(chart, 2))
    left = Encoded(torch.tensor([[1.0, 2.0]]), ("term", "width"))
    right = Encoded(torch.tensor([[3.0, 4.0]]), ("term", "width"))

    family.compose(COMPOSE, left, right=right)

    assert chart.received == (left.tensor, right.tensor)
    assert decoder.received is not None
    assert torch.equal(decoder.received.tensor, left.tensor + 10.0 * right.tensor)


def test_a_family_scores_parent_terms_after_its_declared_merge() -> None:
    family, decoder = _family(Concatenate(2))

    scored = family.scorers[APPLY].score(
        ((TERM, TERM),),
        conclusions=(TERM,),
        options=OPTIONS,
    )

    assert scored.axes == ("term",)
    assert decoder.scored is not None
    merged, received_terms = decoder.scored
    assert torch.equal(merged.tensor, torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
    assert received_terms == CODEC.encode_batch((TERM,))


def test_a_unary_action_reencodes_its_symbolic_conclusion_through_the_identity() -> (
    None
):
    identity = RecordingDecoder()
    encoder = RecordingEncoder()
    family = CombinatorFamily(
        {},
        actions=UNARY_ACTIONS,
        rules=cast("RuleSet[Term]", (FixtureRule("raise", 1),)),
        merge=None,
        codec=CODEC,
        encoder=cast("TermEncoder", encoder),
        identity=identity,
        index=CHOICES,
        root_sort="T",
    )

    scored = family.scorers[RAISE].score(
        ((TERM,),), conclusions=(TERM,), options=OPTIONS
    )

    assert scored.axes == ("term",)
    assert encoder.received == CODEC.encode_batch((TERM,))
    assert identity.scored is not None
    assert identity.scored[1] == CODEC.encode_batch((TERM,))


def test_teacher_forcing_scores_one_licensed_multivalued_unary_conclusion() -> None:
    identity = RecordingDecoder()
    encoder = RecordingEncoder()
    family = CombinatorFamily(
        {},
        actions=UNARY_ACTIONS,
        rules=cast("RuleSet[Term]", (MultivaluedFixtureRule("raise", 1),)),
        merge=None,
        codec=CODEC,
        encoder=cast("TermEncoder", encoder),
        identity=identity,
        index=CHOICES,
        root_sort="T",
    )

    scored = family.scorers[RAISE].score(
        ((TERM,),), conclusions=(TERM,), options=OPTIONS
    )

    assert scored.axes == ("term",)
    assert encoder.received == CODEC.encode_batch((TERM,))


def test_only_forward_complete_unary_actions_are_reencodable() -> None:
    reverse_only = FixtureRule("raise", 1)
    reverse_only.directions = frozenset({"reverse"})
    reverse_only.forward_is_complete = False
    identity = RecordingDecoder()
    family = CombinatorFamily(
        {},
        actions=UNARY_ACTIONS,
        rules=cast("RuleSet[Term]", (reverse_only,)),
        merge=None,
        codec=CODEC,
        encoder=cast("TermEncoder", RecordingEncoder()),
        identity=identity,
        index=CHOICES,
        root_sort="T",
    )

    assert family.unary == (RAISE,)
    assert family.reencodable_unary == ()


def test_a_family_covers_exactly_the_composing_actions() -> None:
    decoder = RecordingDecoder()

    with pytest.raises(ValueError, match="binary actions compose"):
        CombinatorFamily(
            {APPLY: decoder},
            actions=ACTIONS,
            rules=cast(
                "RuleSet[Term]",
                (FixtureRule("apply", 2), FixtureRule("compose", 2)),
            ),
            merge=Concatenate(2),
            codec=CODEC,
            encoder=cast("TermEncoder", RecordingEncoder()),
            identity=decoder,
            index=CHOICES,
            root_sort="T",
        )

    with pytest.raises(ValueError, match="not binary actions"):
        CombinatorFamily(
            {APPLY: decoder, COMPOSE: decoder, IDENTITY: decoder},
            actions=ACTIONS,
            rules=cast(
                "RuleSet[Term]",
                (FixtureRule("apply", 2), FixtureRule("compose", 2)),
            ),
            merge=Concatenate(2),
            codec=CODEC,
            encoder=cast("TermEncoder", RecordingEncoder()),
            identity=decoder,
            index=CHOICES,
            root_sort="T",
        )


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            Encoded(torch.zeros(1, 2), ("term", "width")),
            Encoded(torch.zeros(1, 2), ("item", "width")),
            "different axes or shapes",
        ),
        (
            Encoded(torch.zeros(1, 2), ("term", "width"), torch.tensor([True])),
            Encoded(torch.zeros(1, 2), ("term", "width")),
            "different masks",
        ),
    ],
)
def test_a_merge_requires_aligned_daughters(
    left: Encoded, right: Encoded, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        Concatenate(2).merge(left, right)
