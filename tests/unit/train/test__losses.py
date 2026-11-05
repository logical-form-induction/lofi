from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import didactic.api as dx
import pytest
import torch
from tests.conformance.bound_objective import (
    LossComponentBindingsFixture,
    LossComponentBindingsLawSuite,
)

from lofi._types import ReadoutId
from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.codec import NeuralCodecView
from lofi.nn.loss._errors import MissingOutputError
from lofi.nn.loss._mapping import SOFTMAX
from lofi.nn.loss._ports import (
    LossComponentBindings,
    LossTerm,
)
from lofi.nn.loss._readout import (
    TermsReadout,
)
from lofi.nn.loss._values import CellChart, ObjectiveOutputs
from lofi.nn.sites import ComponentId
from lofi.nn.spaces import RelationKind, Space, SpaceRelationBinding
from lofi.nn.tokens.decoder import RecurrentSpanDecoder
from lofi.nn.tokens.embedder import ReservedTokens, TokenVocabulary
from lofi.spec.readouts import DecodedMeaningSettings, ReadoutSource
from lofi.spec.readouts.terms import ReadoutMass
from lofi.terms import (
    ChoiceIndex,
    CompleteTree,
    Operator,
    OperatorIndex,
    Signature,
    Term,
    TermCodec,
    TermStore,
)
from lofi.train._losses import (
    InactiveLossError,
    LossAssembly,
    LossBindingContext,
    LossResult,
    MisnamedObjectiveError,
    WeightedObjective,
    WeightedTerm,
    WeightSchedule,
    metric_key,
)
from lofi.train._steps import TrainResult

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.nn.loss._ports import PublishedOutputs
    from lofi.nn.ports import TermEncoder

type Batch = tuple[str, ...]
type Outputs = PublishedOutputs[Batch]

ATOM = Operator("atom", "T", family="primitive")
SIGNATURE = Signature("loss-bindings", ("T",), (ATOM,))
STORE = TermStore(SIGNATURE)
VIEW = NeuralCodecView(
    TermCodec(
        OperatorIndex(SIGNATURE),
        CompleteTree(SIGNATURE, SIGNATURE.operators, order="bf", depth=1),
    )
)
CHOICES = ChoiceIndex(SIGNATURE, depth_bounds={"T": 1})


class BindingDecoder:
    """A decoder resolved through a represented space and its term readout."""

    def marginal_rows(self, encoded: Encoded, /, *, root_sort: str) -> LogDistribution:
        """Refuse evaluation because the binding laws only resolve identity.

        Parameters
        ----------
        encoded : Encoded
            Vectors the decoder would read.
        """
        del encoded, root_sort

        raise AssertionError


class BindingEncoder:
    """A term encoder retained by the represented-space fixture."""

    output_width = 2


class IdentityMorphism:
    """The identity term map between the fixture's equal spaces."""

    def __call__(self, term: Term, /) -> Term:
        """Return the supplied term unchanged."""
        return term


class TestLossBindingContextLaws(LossComponentBindingsLawSuite):
    @pytest.fixture
    def binding_conformance(self) -> LossComponentBindingsFixture:
        decoder = BindingDecoder()
        readout = TermsReadout(
            decoder,
            codec=VIEW,
            index=CHOICES,
            reads=ReadoutSource.INSIDE,
            decoding=None,
            key=ReadoutId("types"),
            targets={},
            strategy=ReadoutMass.FACTORIZED,
            mapping=SOFTMAX,
            store=None,
            root_sort="T",
            max_nodes=None,
            clamp=None,
            meanings=DecodedMeaningSettings("decoded"),
            proof_terms=None,
            approximation=None,
            run_approximation=None,
        )
        space = Space(
            "T",
            codec=VIEW,
            choices=CHOICES,
            store=STORE,
            max_nodes=VIEW.capacity,
            encoder=cast("TermEncoder", BindingEncoder()),
            identity=decoder,
            combinators=None,
        )
        relation = SpaceRelationBinding(
            space,
            space,
            kind=RelationKind.CONSISTENCY,
            mapping=None,
            erasure=IdentityMorphism(),
        )
        span_decoder = RecurrentSpanDecoder(
            TokenVocabulary(
                ("word",), reserved=ReservedTokens({"<unk>": 0}), unknown="<unk>"
            ),
            width=2,
            cell="gru",
            layers=1,
            state_width=None,
            head=(),
            end="<eos>",
            component=ComponentId("reconstruction"),
        )
        context = LossBindingContext(
            name="coherence-secondary",
            readouts={"types": readout},
            spaces={"categories": space},
            readout_spaces={"types": space},
            relations={"categories": relation},
            span_decoders={"reconstruction": span_decoder},
            proof_reward=None,
        )

        return LossComponentBindingsFixture(
            context,
            "coherence-secondary",
            "types",
            readout,
            "categories",
            space,
            "categories",
            relation,
            "reconstruction",
            span_decoder,
        )


def bundle() -> ObjectiveOutputs[Batch]:
    inside = LogDistribution(torch.full((2, 3), -1.0), ("item", "span"))

    return ObjectiveOutputs(
        spans=SpanIndex(3),
        chart=CellChart(inside),
        batch=("first", "second"),
    )


class Fixed:
    def __init__(
        self,
        name: str,
        value: float,
        *,
        count: int = 1,
        residual: float | None = None,
        diagnostics: Mapping[str, float] | None = None,
    ) -> None:
        self.name = name
        self.requires = frozenset({"batch"})
        self._value = value
        self._count = count
        self._residual = residual
        self._diagnostics = dict(diagnostics or {})

    def evaluate(self, outputs: Outputs, /) -> LossTerm:
        scale = float(len(outputs.batch))
        leaf = torch.tensor(self._value, requires_grad=True)

        return LossTerm(
            leaf * scale / 2.0,
            self._count,
            residual=None if self._residual is None else torch.tensor(self._residual),
            diagnostics={
                name: torch.tensor(value) for name, value in self._diagnostics.items()
            },
        )


class Inapplicable:
    def __init__(self, name: str) -> None:
        self.name = name
        self.requires = frozenset({"batch"})

    def evaluate(self, outputs: Outputs, /) -> LossTerm:
        like = torch.zeros(len(outputs.batch), requires_grad=True)

        return LossTerm.inapplicable(like)


def weighted(
    name: str,
    weight: float,
    value: float,
    *,
    residual: float | None = None,
    diagnostics: Mapping[str, float] | None = None,
) -> WeightedObjective[Batch]:
    objective = Fixed(name, value, residual=residual, diagnostics=diagnostics)

    return WeightedObjective(name, weight, objective)


def assembly(*entries: WeightedObjective[Batch]) -> LossAssembly[Batch]:
    return LossAssembly(entries)


def models_built(call: Callable[[], LossResult]) -> int:
    counted = 0
    original = dx.Model.__init__

    def counting(model: dx.Model, /, **fields: int) -> None:
        nonlocal counted
        counted += 1

        original(model, **fields)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dx.Model, "__init__", counting)
        call()

    return counted


def test_the_total_recomputes_as_the_sum_of_each_active_weight_times_value() -> None:
    built = assembly(
        weighted("coherence", 0.5, 2.0),
        weighted("pinning", 2.0, 3.0),
        WeightedObjective("inference", 100.0, Inapplicable("inference")),
    )

    result = built.evaluate(bundle(), step=0)
    recomputed = sum(
        record.weight * record.term.value.item()
        for record in result.terms
        if not record.inactive
    )

    assert result.total.item() == pytest.approx(recomputed)
    assert result.total.item() == pytest.approx(0.5 * 2.0 + 2.0 * 3.0)


def test_the_total_stays_on_the_graph_of_every_active_term() -> None:
    built = assembly(weighted("coherence", 0.5, 2.0), weighted("pinning", 2.0, 3.0))

    result = built.evaluate(bundle(), step=0)

    assert result.total.requires_grad
    assert result.total.grad_fn is not None


def test_each_term_records_its_own_inactive_marker() -> None:
    built = assembly(
        weighted("coherence", 1.0, 2.0),
        WeightedObjective("pinning", 1.0, Inapplicable("pinning")),
        weighted("inference", 1.0, 3.0),
    )

    result = built.evaluate(bundle(), step=0)

    assert [record.inactive for record in result.terms] == [0, 1, 0]
    assert [record.term.count for record in result.terms] == [1, 0, 1]


def test_the_inactive_count_is_the_number_of_zero_count_terms() -> None:
    built = assembly(
        weighted("coherence", 1.0, 2.0),
        WeightedObjective("pinning", 1.0, Inapplicable("pinning")),
        WeightedObjective("inference", 1.0, Inapplicable("inference")),
    )

    result = built.evaluate(bundle(), step=0)

    assert result.inactive == 2
    assert result.inactive == sum(record.inactive for record in result.terms)
    assert result.inactive == sum(record.term.count == 0 for record in result.terms)


def test_an_all_inactive_assembly_raises_rather_than_returning_a_zero() -> None:
    built = assembly(
        WeightedObjective("coherence", 1.0, Inapplicable("coherence")),
        WeightedObjective("pinning", 1.0, Inapplicable("pinning")),
    )

    with pytest.raises(InactiveLossError) as caught:
        built.evaluate(bundle(), step=0)

    assert caught.value.names == ("coherence", "pinning")
    assert "coherence" in str(caught.value)
    assert "pinning" in str(caught.value)


def test_one_active_term_is_enough_for_a_result() -> None:
    built = assembly(
        WeightedObjective("coherence", 1.0, Inapplicable("coherence")),
        weighted("pinning", 3.0, 2.0),
    )

    result = built.evaluate(bundle(), step=0)

    assert result.inactive == 1
    assert result.total.item() == pytest.approx(6.0)


def test_every_base_metric_key_is_exactly_loss_slash_name() -> None:
    built = assembly(
        WeightedObjective("coherence", 1.0, Fixed("coherence", 2.0)),
        WeightedObjective("span-pinning", 1.0, Fixed("span-pinning", 3.0)),
    )

    result = built.evaluate(bundle(), step=0)

    assert set(result.metrics) == {"loss/coherence", "loss/span-pinning"}
    assert [record.key for record in result.terms] == [
        "loss/coherence",
        "loss/span-pinning",
    ]
    assert result.metrics["loss/coherence"] == pytest.approx(2.0)
    assert result.metrics["loss/span-pinning"] == pytest.approx(3.0)


def test_the_key_comes_from_the_name_and_not_from_the_class_name() -> None:
    class CoherenceObjective(Fixed):
        pass

    built = assembly(
        WeightedObjective("kl-coherence", 1.0, CoherenceObjective("kl-coherence", 2.0))
    )

    result = built.evaluate(bundle(), step=0)

    assert set(result.metrics) == {"loss/kl-coherence"}


def test_a_diagnostic_lands_under_the_term_it_belongs_to() -> None:
    built = assembly(
        weighted("coherence", 1.0, 2.0, diagnostics={"entropy": 0.25}),
        weighted("pinning", 1.0, 3.0, residual=0.125),
    )

    result = built.evaluate(bundle(), step=0)

    assert set(result.metrics) == {
        "loss/coherence",
        "loss/coherence/entropy",
        "loss/pinning",
        "loss/pinning/residual",
    }
    assert result.metrics["loss/coherence/entropy"] == pytest.approx(0.25)
    assert result.metrics["loss/pinning/residual"] == pytest.approx(0.125)


def test_each_term_names_the_key_its_residual_is_logged_under() -> None:
    built = assembly(
        weighted("coherence", 1.0, 2.0),
        weighted("pinning", 1.0, 3.0, residual=0.125),
    )

    result = built.evaluate(bundle(), step=0)

    assert dict(built.residual_keys) == {
        "coherence": "loss/coherence/residual",
        "pinning": "loss/pinning/residual",
    }
    assert built.residual_keys["pinning"] in result.metrics
    assert built.residual_keys["coherence"] not in result.metrics


def test_an_inactive_term_publishes_no_metric() -> None:
    built = assembly(
        weighted("coherence", 1.0, 2.0),
        WeightedObjective("pinning", 1.0, Inapplicable("pinning")),
    )

    result = built.evaluate(bundle(), step=0)

    assert set(result.metrics) == {"loss/coherence"}


def test_a_metric_is_the_unweighted_value_of_its_term() -> None:
    built = assembly(weighted("coherence", 7.0, 2.0))

    result = built.evaluate(bundle(), step=0)

    assert result.metrics["loss/coherence"] == pytest.approx(2.0)
    assert result.total.item() == pytest.approx(14.0)


def test_the_metrics_are_plain_finite_scalars_a_train_result_accepts() -> None:
    built = assembly(
        weighted("coherence", 1.0, 2.0, diagnostics={"entropy": 0.25}),
        weighted("pinning", 1.0, 3.0),
    )

    result = built.evaluate(bundle(), step=0)
    reported = TrainResult(3, 2, metrics=result.metrics)

    assert all(type(value) is float for value in result.metrics.values())
    assert dict(reported.metrics) == dict(result.metrics)


def test_metric_keys_are_derived_by_one_function() -> None:
    assert metric_key("coherence") == "loss/coherence"
    assert metric_key("coherence", "entropy") == "loss/coherence/entropy"

    with pytest.raises(ValueError, match="name"):
        metric_key("")

    with pytest.raises(ValueError, match="diagnostic"):
        metric_key("coherence", "")


def test_a_weighted_objective_whose_name_is_not_the_declared_one_is_refused() -> None:
    with pytest.raises(MisnamedObjectiveError) as caught:
        WeightedObjective("coherence", 1.0, Fixed("kl", 2.0))

    assert caught.value.declared == "coherence"
    assert caught.value.name == "kl"
    assert "coherence" in str(caught.value)
    assert "kl" in str(caught.value)


def test_a_weighted_objective_requires_a_nonempty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        WeightedObjective("", 1.0, Fixed("", 2.0))


def test_a_weighted_objective_exposes_what_it_was_given() -> None:
    objective = Fixed("coherence", 2.0)
    entry = WeightedObjective("coherence", 0.5, objective)

    with pytest.raises(AttributeError, match="immutable"):
        entry.__setattr__("_weight", 1.0)

    assert entry.name == "coherence"
    assert entry.weight == 0.5
    assert entry.objective is objective
    assert "coherence" in repr(entry)


def test_an_assembly_keeps_its_terms_in_the_order_given() -> None:
    built = assembly(
        weighted("pinning", 0.5, 2.0),
        weighted("coherence", 2.0, 3.0),
    )

    assert built.names == ("pinning", "coherence")
    assert built.weights == (0.5, 2.0)
    assert built.requires == frozenset({"batch"})
    assert len(built) == 2
    assert [entry.name for entry in built] == ["pinning", "coherence"]
    assert "pinning" in repr(built)


def test_two_terms_of_one_kind_report_under_their_own_names() -> None:
    built = assembly(
        weighted("coherence", 1.0, 2.0, diagnostics={"entropy": 0.25}),
        weighted("coherence-secondary", 0.5, 3.0, diagnostics={"entropy": 0.5}),
    )

    result = built.evaluate(bundle(), step=0)

    assert built.names == ("coherence", "coherence-secondary")
    assert result.metrics == {
        "loss/coherence": pytest.approx(2.0),
        "loss/coherence/entropy": pytest.approx(0.25),
        "loss/coherence-secondary": pytest.approx(3.0),
        "loss/coherence-secondary/entropy": pytest.approx(0.5),
    }
    assert result.total.item() == pytest.approx(3.5)


def test_a_loss_binding_context_needs_a_nonempty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        LossBindingContext(
            name="",
            readouts={},
            spaces={},
            readout_spaces={},
            relations={},
            span_decoders={},
            proof_reward=None,
        )


def test_an_assembly_with_no_term_is_refused() -> None:
    with pytest.raises(ValueError, match="no objectives"):
        LossAssembly(())


def test_an_assembly_refuses_two_terms_of_one_name() -> None:
    with pytest.raises(ValueError, match="two objectives are named"):
        assembly(weighted("coherence", 1.0, 2.0), weighted("coherence", 2.0, 3.0))


@pytest.mark.parametrize("weight", [-0.5, float("inf"), float("nan")])
def test_an_assembly_refuses_a_weight_that_cannot_scale_a_term(weight: float) -> None:
    with pytest.raises(ValueError, match="weight"):
        assembly(weighted("coherence", weight, 2.0))


def test_an_assembly_refuses_an_objective_reading_what_the_graph_omits() -> None:
    with pytest.raises(MissingOutputError):
        LossAssembly(
            (weighted("coherence", 1.0, 2.0),),
            published=frozenset({"chart"}),
        )


def test_a_loss_binding_context_is_an_immutable_snapshot_of_role_mappings() -> None:
    readouts = {}
    spaces = {}
    readout_spaces = {}
    relations = {}
    context = LossBindingContext(
        name="coherence",
        readouts=readouts,
        spaces=spaces,
        readout_spaces=readout_spaces,
        relations=relations,
        span_decoders={},
        proof_reward=None,
    )

    with pytest.raises(AttributeError, match="immutable"):
        context.__setattr__("_readouts", context.readouts)

    assert isinstance(context, LossComponentBindings)
    assert isinstance(context.readouts, MappingProxyType)
    assert isinstance(context.spaces, MappingProxyType)
    assert context.readouts is not readouts
    assert context.spaces is not spaces
    assert dict(context.readouts) == {}
    assert dict(context.spaces) == {}
    assert context.name == "coherence"


def test_a_context_without_a_declared_loss_component_names_the_absence() -> None:
    context = LossBindingContext(
        name="coherence",
        readouts={},
        spaces={},
        readout_spaces={},
        relations={},
        span_decoders={},
        proof_reward=None,
    )

    with pytest.raises(ValueError, match="term-readout role 'types' is not built"):
        context.terms_readout("types")

    with pytest.raises(ValueError, match="space role 'categories' is not built"):
        context.space("categories")

    with pytest.raises(ValueError, match="has no represented space"):
        context.readout_space("types")

    with pytest.raises(ValueError, match="has no relation"):
        context.relation("categories")

    with pytest.raises(ValueError, match="'reconstruction' is not built"):
        context.span_decoder("reconstruction")


def test_evaluating_a_step_builds_no_boundary_object() -> None:
    built = assembly(weighted("coherence", 1.0, 2.0), weighted("pinning", 1.0, 3.0))

    assert models_built(lambda: built.evaluate(bundle(), step=0)) == 0


def test_a_weighted_term_is_immutable() -> None:
    record = WeightedTerm("coherence", 0.5, LossTerm(torch.tensor(2.0), 1))

    with pytest.raises(AttributeError, match="immutable"):
        record.__setattr__("_weight", 1.0)

    assert record.weight == 0.5
    assert record.key == "loss/coherence"
    assert record.inactive == 0
    assert "coherence" in repr(record)


def test_a_result_is_immutable_and_hands_out_snapshots() -> None:
    term = WeightedTerm("coherence", 1.0, LossTerm(torch.tensor(2.0), 1))
    metrics = {"loss/coherence": 2.0}
    result = LossResult(torch.tensor(2.0), [term], 0, metrics)
    metrics["loss/pinning"] = 3.0

    with pytest.raises(AttributeError, match="immutable"):
        result.__setattr__("_inactive", 1)

    assert set(result.metrics) == {"loss/coherence"}
    assert result.terms == (term,)
    assert "inactive=0" in repr(result)


def test_a_result_refuses_an_inactive_count_off_its_records() -> None:
    active = WeightedTerm("coherence", 1.0, LossTerm(torch.tensor(2.0), 1))

    with pytest.raises(ValueError, match="inactive"):
        LossResult(torch.tensor(2.0), [active], 1, {})


def test_the_inactive_error_names_the_terms_and_needs_at_least_one() -> None:
    error = InactiveLossError(("coherence",))

    assert error.names == ("coherence",)

    with pytest.raises(ValueError, match="at least one term"):
        InactiveLossError(())


@pytest.mark.parametrize(
    ("step", "expected"), [(0, 0.2), (1, 0.4), (2, 0.6), (4, 1.0), (9, 1.0)]
)
def test_a_schedule_ramps_linearly_to_the_declared_weight_and_holds(
    step: int, expected: float
) -> None:
    assert WeightSchedule(0.2, 4).at(1.0, step) == pytest.approx(expected)


def test_a_schedule_may_ramp_down_to_the_declared_weight() -> None:
    schedule = WeightSchedule(2.0, 2)

    assert [schedule.at(1.0, step) for step in range(4)] == [2.0, 1.5, 1.0, 1.0]


@pytest.mark.parametrize(("start", "steps"), [(-0.1, 2), (float("nan"), 2), (0.0, 0)])
def test_a_schedule_refuses_a_negative_start_or_no_steps(
    start: float, steps: int
) -> None:
    with pytest.raises(ValueError, match="schedule"):
        WeightSchedule(start, steps)


def test_a_schedule_is_a_frozen_value_compared_by_its_start_and_steps() -> None:
    schedule = WeightSchedule(0.5, 3)

    assert schedule == WeightSchedule(0.5, 3)
    assert hash(schedule) == hash(WeightSchedule(0.5, 3))
    assert schedule != WeightSchedule(0.5, 4)
    assert schedule != WeightSchedule(0.25, 3)
    assert (schedule.start, schedule.steps) == (0.5, 3)
    assert repr(schedule) == "WeightSchedule(0.5, 3)"

    with pytest.raises(AttributeError):
        schedule.__setattr__("_start", 1.0)


def test_a_scheduled_term_enters_the_total_and_the_metrics_at_its_ramped_weight() -> (
    None
):
    scheduled = WeightedObjective(
        "coherence",
        1.0,
        Fixed("coherence", 2.0),
        schedule=WeightSchedule(0.0, 4),
    )
    built = assembly(scheduled, weighted("pinning", 2.0, 3.0))
    totals = []
    weights = []

    for step in (0, 2, 4, 8):
        result = built.evaluate(bundle(), step=step)
        totals.append(result.total.item())
        weights.append(result.metrics[metric_key("coherence", "weight")])

        assert metric_key("pinning", "weight") not in result.metrics
        assert result.terms[0].weight == weights[-1]

    assert weights == [0.0, 0.5, 1.0, 1.0]
    assert totals == pytest.approx([w * 2.0 + 2.0 * 3.0 for w in weights])
    assert scheduled.weight == 1.0
    assert scheduled.weight_at(2) == 0.5


def test_an_unscheduled_term_takes_its_weight_at_every_step() -> None:
    term = weighted("coherence", 0.75, 2.0)

    assert term.schedule is None
    assert {term.weight_at(step) for step in (0, 5, 500)} == {0.75}


def test_an_inactive_scheduled_term_publishes_no_weight() -> None:
    built = assembly(
        weighted("coherence", 1.0, 2.0),
        WeightedObjective(
            "inference",
            1.0,
            Inapplicable("inference"),
            schedule=WeightSchedule(0.0, 4),
        ),
    )

    result = built.evaluate(bundle(), step=2)

    assert metric_key("inference", "weight") not in result.metrics
    assert result.terms[1].weight == 0.5
