import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, NamedTuple

import pytest
import torch
from torch import Tensor

import lofi.nn.loss._ports as loss_ports
from lofi._types import ReadoutId
from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.codec import DISTRIBUTION_AXES
from lofi.nn.loss._grammaticality import GrammaticalityObjective
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._ports import (
    BoundObjective,
    CoherenceEvaluation,
    CombinatorDecoder,
    CombinatorScorer,
    ConstrainedItems,
    FormCredit,
    LabeledItems,
    LossComponentBindings,
    LossTerm,
    Objective,
    PublishedOutputs,
    ResolvableConstraint,
    RuleInstances,
    ScheduledBatch,
    ScheduledRuleInstances,
    SpanTokenDecoder,
    SurfaceItems,
    TokenizedItems,
    TypeItems,
    TypePairs,
)
from lofi.nn.loss._readout import MASS_AXES
from lofi.nn.loss._values import (
    OUTPUT_FIELDS,
    CellChart,
    ObjectiveOutputs,
    RecordingOutputs,
    SpanReadout,
)
from lofi.nn.ports import ForwardMode, ForwardOptions
from lofi.nn.tokens.embedder import ReservedTokens, TokenVocabulary
from lofi.terms import BatchEncoding, Encoding, Term

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.nn.loss._readout import TermsReadout
    from lofi.nn.spaces import Space, SpaceRelationBinding


class CompleteLossBindings:
    """A structurally complete loss-component binding context."""

    @property
    def name(self) -> str:
        """Return the declared term name."""
        return "complete"

    def terms_readout(self, role: str, /) -> TermsReadout:
        """Raise after accepting a term-readout role.

        Parameters
        ----------
        role : str
            Role to resolve.
        """
        raise LookupError(role)

    def space(self, role: str, /) -> Space:
        """Raise after accepting a represented-space role.

        Parameters
        ----------
        role : str
            Role to resolve.
        """
        raise LookupError(role)

    def readout_space(self, role: str, /) -> Space:
        """Raise after accepting a term-readout role."""
        raise LookupError(role)

    def relation(self, role: str, /) -> SpaceRelationBinding:
        """Raise after accepting a represented-space relation role."""
        raise LookupError(role)

    def span_decoder(self, role: str, /) -> SpanTokenDecoder:
        """Raise after accepting a span-decoder role.

        Parameters
        ----------
        role : str
            Role to resolve.
        """
        raise LookupError(role)

    def proof_reward(self) -> loss_ports.ProofRewardBindings:
        """Raise because the protocol test resolves no live proof service."""
        raise LookupError


def test_loss_component_bindings_requires_every_exact_role_capability() -> None:
    bindings: LossComponentBindings = CompleteLossBindings()
    partial = SimpleNamespace(terms_readout=bindings.terms_readout)
    roles_only = SimpleNamespace(
        terms_readout=bindings.terms_readout,
        space=bindings.space,
        readout_space=bindings.readout_space,
    )

    nameless = SimpleNamespace(
        terms_readout=bindings.terms_readout,
        space=bindings.space,
        readout_space=bindings.readout_space,
        relation=bindings.relation,
        span_decoder=bindings.span_decoder,
        proof_reward=bindings.proof_reward,
    )

    assert isinstance(bindings, LossComponentBindings)
    assert bindings.name == "complete"
    assert not isinstance(partial, LossComponentBindings)
    assert not isinstance(roles_only, LossComponentBindings)
    assert not isinstance(nameless, LossComponentBindings)


def test_a_coherence_evaluation_is_frozen_and_keeps_detached_contributions() -> None:
    evaluation = CoherenceEvaluation(2, (("application", 3, 0.25),))

    assert evaluation.item == 2
    assert evaluation.contributions == (("application", 3, 0.25),)
    assert not hasattr(evaluation, "__dict__")

    with pytest.raises(AttributeError):
        evaluation.item = 4


@pytest.mark.parametrize(
    ("item", "contributions"),
    [
        (-1, (("application", 1, 0.0),)),
        (0, (("", 1, 0.0),)),
        (0, (("application", 0, 0.0),)),
        (0, (("application", 1, float("nan")),)),
    ],
)
def test_a_coherence_evaluation_refuses_malformed_contributions(
    item: int, contributions: tuple[tuple[str, int, float], ...]
) -> None:
    with pytest.raises(ValueError, match=r"coherence (evaluation|contribution)"):
        CoherenceEvaluation(item, contributions)


def test_a_term_keeps_what_it_was_built_from() -> None:
    value = torch.tensor(2.5)
    residual = torch.tensor(0.125)

    term = LossTerm(
        value,
        8,
        residual=residual,
        diagnostics={"entropy_gap": torch.tensor(1.0)},
    )

    assert term.value is value
    assert term.count == 8
    assert term.residual is residual
    assert term.diagnostics["entropy_gap"].item() == 1.0


def test_a_term_defaults_to_no_residual_and_no_diagnostics() -> None:
    term = LossTerm(torch.tensor(1.0), 4)

    assert term.residual is None
    assert len(term.diagnostics) == 0


def test_diagnostics_is_a_snapshot_rather_than_the_caller_s_mapping() -> None:
    source = {"entropy_gap": torch.tensor(0.5)}

    term = LossTerm(torch.tensor(1.0), 3, diagnostics=source)
    source["entropy_gap"] = torch.tensor(9.0)
    source["collapse_rate"] = torch.tensor(1.0)

    assert term.diagnostics is not source
    assert term.diagnostics["entropy_gap"].item() == 0.5
    assert "collapse_rate" not in term.diagnostics


def test_a_term_reads_back_what_it_was_built_from_and_holds_nothing_else() -> None:
    term = LossTerm(torch.tensor(1.5), 2)

    assert term.value.item() == 1.5
    assert term.count == 2
    assert term.residual is None
    assert not hasattr(term, "__dict__")


def test_inapplicable_is_a_zero_that_follows_the_data() -> None:
    like = torch.ones(2, 3, dtype=torch.float64)

    term = LossTerm.inapplicable(like)

    assert term.count == 0
    assert term.value.ndim == 0
    assert term.value.item() == 0.0
    assert term.value.dtype is torch.float64
    assert term.value.device == like.device
    assert term.residual is None
    assert len(term.diagnostics) == 0


def test_inapplicable_reaches_its_input_with_an_all_zero_gradient() -> None:
    like = torch.randn(4, 5, requires_grad=True)

    LossTerm.inapplicable(like).value.backward()

    assert like.grad is not None
    assert like.grad.shape == like.shape
    assert not like.grad.any()


def test_inapplicable_reads_no_value_so_a_masked_input_stays_finite() -> None:
    like = torch.tensor([0.0, float("-inf")], requires_grad=True)

    term = LossTerm.inapplicable(like)
    term.value.backward()

    assert term.value.item() == 0.0
    assert like.grad is not None
    assert torch.isfinite(like.grad).all()


def test_inapplicable_takes_a_tensor_of_any_rank() -> None:
    for like in (torch.tensor(3.0), torch.empty(0), torch.ones(2, 2).t()):
        assert LossTerm.inapplicable(like).value.item() == 0.0


def test_the_debug_checks_reject_a_malformed_term(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(loss_ports, on=True)

    with pytest.raises(ValueError, match="value must be a scalar"):
        loss_ports.LossTerm(torch.zeros(3), 1)

    with pytest.raises(ValueError, match="cannot be negative"):
        loss_ports.LossTerm(torch.tensor(0.0), -1)

    with pytest.raises(ValueError, match="residual must be a scalar"):
        loss_ports.LossTerm(torch.tensor(0.0), 1, residual=torch.zeros(2))

    with pytest.raises(ValueError, match="diagnostics"):
        loss_ports.LossTerm(torch.tensor(0.0), 1, diagnostics={"gap": torch.zeros(2)})


def test_without_the_debug_checks_a_term_is_built_unexamined(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(loss_ports, on=False)

    term = loss_ports.LossTerm(torch.zeros(3), -1)

    assert term.value.shape == (3,)
    assert term.count == -1


class Outputs:
    def __init__(self, scores: torch.Tensor) -> None:
        self.scores = scores


class ScoreSum:
    name = "score_sum"
    requires = frozenset({"scores"})

    def evaluate(self, published: Outputs, /) -> LossTerm:
        return LossTerm(published.scores.sum(), published.scores.numel())


def test_an_objective_is_reached_through_the_protocol_alone() -> None:
    objective: Objective[Outputs] = ScoreSum()
    published = Outputs(torch.tensor([1.0, 2.0, 3.0]))

    term = objective.evaluate(published)

    assert objective.name == "score_sum"
    assert objective.requires == frozenset({"scores"})
    assert term.value.item() == 6.0
    assert term.count == 3
    assert term.residual is None


def test_a_value_missing_a_declared_member_is_not_an_objective() -> None:
    assert isinstance(ScoreSum(), Objective)
    assert not isinstance(
        SimpleNamespace(name="nameless", requires=frozenset()), Objective
    )


def published_outputs() -> ObjectiveOutputs[tuple[str, ...]]:
    inside = LogDistribution(torch.full((2, 3), -1.0), ("item", "span"))

    return ObjectiveOutputs(
        spans=SpanIndex(3),
        chart=CellChart(inside),
        batch=("first", "second"),
    )


class Anchor:
    name = "anchor"
    requires = frozenset({"batch"})

    def bind(self, context: float, /) -> Objective[PublishedOutputs[tuple[str, ...]]]:
        return Scaled(context)


class Scaled:
    name = "anchor"
    requires = frozenset({"batch"})

    def __init__(self, scale: float) -> None:
        self._scale = scale

    def evaluate(self, outputs: PublishedOutputs[tuple[str, ...]], /) -> LossTerm:
        return LossTerm(torch.tensor(self._scale * len(outputs.batch)), 1)


def test_a_bundle_and_its_proxy_are_both_what_an_objective_is_written_against() -> None:
    outputs = published_outputs()
    view = RecordingOutputs(outputs, objective="anchor", requires=frozenset({"batch"}))

    assert isinstance(outputs, PublishedOutputs)
    assert isinstance(view, PublishedOutputs)


def test_a_bundle_without_scored_forms_is_not_what_an_objective_reads() -> None:
    outputs = published_outputs()
    without = SimpleNamespace(
        **{
            field: getattr(outputs, field)
            for field in OUTPUT_FIELDS
            if field != "forms"
        }
    )

    assert not isinstance(without, PublishedOutputs)
    assert isinstance(SimpleNamespace(**vars(without), forms={}), PublishedOutputs)


def test_an_unbound_objective_is_not_yet_an_objective() -> None:
    unbound = Anchor()

    assert isinstance(unbound, BoundObjective)
    assert not isinstance(unbound, Objective)


def test_binding_hands_back_an_objective_that_scores_a_step() -> None:
    bound = Anchor().bind(3.0)

    term = bound.evaluate(published_outputs())

    assert isinstance(bound, Objective)
    assert term.value.item() == 6.0
    assert term.count == 1


class ResolvesSpans:
    def to_spans(
        self,
        batch_item_ids: tuple[str, ...],
        /,
        *,
        readout: ReadoutId,
    ) -> tuple[()]:
        del batch_item_ids, readout

        return ()


def test_a_constraint_resolver_is_structural() -> None:
    assert isinstance(ResolvesSpans(), ResolvableConstraint)


class Rows:
    """A batch implementing only the tokenized-items protocol."""

    def __init__(self, ids: torch.Tensor, lengths: tuple[int, ...]) -> None:
        self._ids = ids
        self._lengths = lengths

    @property
    def token_ids(self) -> torch.Tensor:
        return self._ids

    @property
    def lengths(self) -> tuple[int, ...]:
        return self._lengths


class Words:
    """A batch implementing only the surface-items protocol."""

    def __init__(self, items: tuple[tuple[str, ...], ...]) -> None:
        self._items = items

    @property
    def words(self) -> tuple[tuple[str, ...], ...]:
        return self._items


SPAN_VOCABULARY = TokenVocabulary(
    ("a", "b"), reserved=ReservedTokens({"<unk>": 0, "<eos>": 1}), unknown="<unk>"
)


class Flat:
    """A decoder implementing only the span-token-decoder protocol."""

    @property
    def vocabulary(self) -> TokenVocabulary:
        return SPAN_VOCABULARY

    @property
    def end(self) -> str:
        return "<eos>"

    def score_tokens(
        self, prefix: Encoded, /, *, targets: torch.Tensor
    ) -> LogDistribution:
        rows, positions = targets.shape
        scores = prefix.tensor.new_zeros((rows, positions, SPAN_VOCABULARY.size))

        return LogDistribution(
            scores.log_softmax(dim=2),
            ("span", "position", "token"),
            normalized_over="token",
        )


def test_a_batch_naming_its_words_answers_to_the_surface_items_port() -> None:
    batch: SurfaceItems = Words((("a", "b"), ("b",)))

    assert isinstance(batch, SurfaceItems)
    assert not isinstance(Anchor(), SurfaceItems)
    assert batch.words == (("a", "b"), ("b",))


def test_a_batch_naming_its_token_rows_answers_to_the_tokenized_items_port() -> None:
    ids = torch.tensor([[1, 2, 0], [2, 0, 0]], dtype=torch.int64)
    batch: TokenizedItems = Rows(ids, (2, 1))

    assert isinstance(batch, TokenizedItems)
    assert not isinstance(Anchor(), TokenizedItems)
    assert not isinstance(Words((("a",),)), TokenizedItems)
    assert not isinstance(SimpleNamespace(token_ids=ids), TokenizedItems)
    assert not isinstance(SimpleNamespace(lengths=(2, 1)), TokenizedItems)
    assert batch.lengths == (2, 1)
    assert torch.equal(batch.token_ids, ids)


def test_a_span_token_decoder_scores_each_position_of_each_span() -> None:
    decoder: SpanTokenDecoder = Flat()
    prefix = Encoded(torch.zeros(2, 3, dtype=torch.float64), ("span", "width"))
    scored = decoder.score_tokens(prefix, targets=torch.zeros(2, 4, dtype=torch.int64))

    assert isinstance(decoder, SpanTokenDecoder)
    assert not isinstance(Anchor(), SpanTokenDecoder)
    assert not isinstance(SimpleNamespace(vocabulary=SPAN_VOCABULARY), SpanTokenDecoder)
    assert not isinstance(
        SimpleNamespace(vocabulary=SPAN_VOCABULARY, score_tokens=decoder.score_tokens),
        SpanTokenDecoder,
    )
    assert not isinstance(
        SimpleNamespace(score_tokens=decoder.score_tokens), SpanTokenDecoder
    )
    assert decoder.vocabulary.size == 4
    assert scored.axes == ("span", "position", "token")
    assert tuple(scored.tensor.shape) == (2, 4, 4)


ROW = Encoding((1, 0, 0), (-1, 0, 0), (0, 1, 1), (True, False, False))
ROWS = BatchEncoding((ROW,))


def test_a_batch_of_linearized_types_answers_to_the_type_items_port() -> None:
    batch: TypeItems = SimpleNamespace(terms=ROWS)

    assert isinstance(batch, TypeItems)
    assert not isinstance(Anchor(), TypeItems)
    assert batch.terms.width == 3


def test_a_batch_of_type_pairs_holds_each_pair_as_terms_and_as_rows() -> None:
    batch: TypePairs = SimpleNamespace(pairs=(), left=ROWS, right=ROWS)

    assert isinstance(batch, TypePairs)
    assert not isinstance(SimpleNamespace(pairs=(), left=ROWS), TypePairs)
    assert batch.left.rows == batch.right.rows == (ROW,)


def test_a_batch_of_constrained_items_holds_its_localized_constraints() -> None:
    batch: ConstrainedItems[str, str] = SimpleNamespace(constraints=())

    assert isinstance(batch, ConstrainedItems)
    assert not isinstance(SimpleNamespace(occurrences=()), ConstrainedItems)
    assert batch.constraints == ()


def test_a_batch_of_rule_instances_names_the_actions_admitting_each() -> None:
    batch: RuleInstances = SimpleNamespace(instances=())
    scheduled: ScheduledRuleInstances = SimpleNamespace(
        instances=(), step=3, horizon=10
    )

    assert isinstance(batch, RuleInstances)
    assert not isinstance(batch, ScheduledRuleInstances)
    assert isinstance(scheduled, ScheduledRuleInstances)
    assert isinstance(scheduled, RuleInstances)
    assert not isinstance(SimpleNamespace(actions=((),)), RuleInstances)
    assert batch.instances == ()
    assert (scheduled.step, scheduled.horizon) == (3, 10)


class ProductComposer:
    @property
    def modes(self) -> frozenset[ForwardMode]:
        return frozenset({ForwardMode.MARGINALIZE})

    def compose(
        self, left: Encoded, /, *, right: Encoded, options: ForwardOptions
    ) -> LogDistribution:
        del options
        pairs = left.tensor.shape[0]
        scores = (left.tensor * right.tensor).sum(dim=-1)
        logits = scores.reshape(pairs, 1, 1).expand(pairs, 2, 4)
        steps = torch.arange(4, dtype=logits.dtype)

        return LogDistribution(
            (logits * steps).log_softmax(dim=-1),
            DISTRIBUTION_AXES,
            normalized_over="symbol",
        )


class ScoreOnly:
    @property
    def modes(self) -> frozenset[ForwardMode]:
        return frozenset({ForwardMode.OBSERVE})

    def score(self, left: Encoded, /) -> torch.Tensor:
        return left.tensor.sum()


def children(pairs: int) -> tuple[Encoded, Encoded]:
    generator = torch.Generator().manual_seed(pairs)
    left = torch.randn(pairs, 3, generator=generator)
    right = torch.randn(pairs, 3, generator=generator)

    return Encoded(left, ("term", "width")), Encoded(right, ("term", "width"))


def test_a_composer_reading_a_distribution_off_a_pair_satisfies_the_protocol() -> None:
    composer = ProductComposer()
    left, right = children(5)
    options = ForwardOptions(ForwardMode.MARGINALIZE)

    assert isinstance(composer, CombinatorDecoder)
    assert options.mode in composer.modes

    composed = composer.compose(left, right=right, options=options)

    assert composed.axes == DISTRIBUTION_AXES
    assert composed.normalized_over == "symbol"
    assert composed.tensor.shape == (5, 2, 4)


def test_a_decoder_without_a_composition_is_not_a_combinator_decoder() -> None:
    assert not isinstance(ScoreOnly(), CombinatorDecoder)


class ProductScorer:
    @property
    def arity(self) -> int:
        return 2

    @property
    def slots(self) -> int:
        return 3

    @property
    def modes(self) -> frozenset[ForwardMode]:
        return frozenset({ForwardMode.OBSERVE})

    def score(
        self,
        premises: object,
        /,
        *,
        conclusions: object,
        options: ForwardOptions,
    ) -> LogDistribution:
        del premises, conclusions, options
        weights = -torch.ones(2)

        return LogDistribution(weights, ("term",))

    def exact_matches(
        self,
        premises: object,
        /,
        *,
        conclusions: object,
        options: ForwardOptions,
    ) -> torch.Tensor:
        del premises, conclusions, options
        return torch.ones(2, dtype=torch.bool)


def test_a_scorer_weighing_a_parent_against_a_pair_satisfies_the_protocol() -> None:
    scorer = ProductScorer()

    assert isinstance(scorer, CombinatorScorer)
    assert not isinstance(scorer, CombinatorDecoder)
    assert not isinstance(ProductComposer(), CombinatorScorer)

    scored = scorer.score(
        ((), ()),
        conclusions=(object(), object()),
        options=ForwardOptions(ForwardMode.OBSERVE),
    )

    assert scored.axes == ("term",)
    assert scored.tensor.shape == (2,)
    assert bool((scored.tensor <= -1.0).all())


def test_a_batch_naming_its_step_satisfies_the_scheduled_batch_protocol() -> None:
    scheduled: ScheduledBatch = SimpleNamespace(items=("one",), step=7)

    assert isinstance(scheduled, ScheduledBatch)
    assert not isinstance(SimpleNamespace(items=("one",)), ScheduledBatch)
    assert not isinstance(("one",), ScheduledBatch)
    assert scheduled.step == 7


class Sentences(NamedTuple):
    """A batch implementing only the labeled-items protocol."""

    lengths: tuple[int, ...]
    labels: tuple[bool | None, ...]


class Bare(NamedTuple):
    """A batch that names its items and nothing else."""

    ids: tuple[str, ...]


SPANS = SpanIndex(2)
ROOT_MASS = ReadoutId("sentences")


def pair(log_masses: Tensor) -> SpanReadout:
    rest = torch.log(-torch.expm1(log_masses))

    return SpanReadout(
        SPANS,
        LogDistribution(
            torch.stack((log_masses, rest), dim=-1),
            MASS_AXES,
            normalized_over="outcome",
        ),
    )


def test_a_batch_with_lengths_and_labels_satisfies_the_protocol() -> None:
    assert isinstance(Sentences((2, 1), (True, None)), LabeledItems)
    assert not isinstance(Bare(("first",)), LabeledItems)


def test_the_two_members_the_protocol_declares_are_all_the_objective_reads() -> None:
    log_masses = torch.full((2, SPANS.num_spans), -0.5).double()
    log_masses[0, SPANS.span_id(0, 2)] = -0.25
    log_masses[1, SPANS.span_id(0, 1)] = -1.0
    batch: LabeledItems = Sentences((2, 1), (True, False))
    outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(LogDistribution(torch.zeros(2, SPANS.num_spans), SPAN_AXES)),
        batch=batch,
        readouts={ROOT_MASS: pair(log_masses)},
    )

    term = GrammaticalityObjective(
        readout=ROOT_MASS, noise=0.0, name="grammaticality"
    ).evaluate(outputs)

    assert term.count == 2
    assert term.value.item() == pytest.approx(
        (0.25 - math.log1p(-math.exp(-1.0))) / 2.0, rel=1e-12
    )


class Half:
    def credit(self, form: Term, /) -> float:
        del form

        return 0.5


class Unpaid:
    def score(self, form: Term, /) -> float:
        del form

        return 0.5


def test_a_scorer_paying_one_form_satisfies_the_form_credit_protocol() -> None:
    credit: FormCredit = Half()

    assert isinstance(credit, FormCredit)
    assert not isinstance(Unpaid(), FormCredit)
