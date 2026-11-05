from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, cast, override

import didactic.api as dx
import pytest
import torch
from tests.conformance.bound_objective import (
    BoundObjectiveFixture,
    BoundObjectiveLawSuite,
)
from tests.conformance.configured_component import (
    ConfiguredComponentConformance,
    ConfiguredComponentFixture,
)
from tests.conformance.objective import ObjectiveLawSuite

import lofi.nn.loss._anchor as anchor_module
from lofi._types import ReadoutId
from lofi.deduce import SpanIndex
from lofi.lexicon import ClampMode
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.codec import NeuralCodecView
from lofi.nn.loss._anchor import (
    VectorAnchorObjective,
)
from lofi.nn.loss._constraint import BatchLocalConstraint, BatchLocalNegativeBracket
from lofi.nn.loss._errors import UncomputedCellError
from lofi.nn.loss._linearize import LinearizedTargets
from lofi.nn.loss._mapping import SOFTMAX
from lofi.nn.loss._ports import (
    BoundObjective,
    Objective,
    PublishedOutputs,
)
from lofi.nn.loss._readout import (
    CELL_AXES,
    TermsReadout,
)
from lofi.nn.loss._values import CellChart, ObjectiveOutputs
from lofi.nn.loss.laws import (
    ObjectiveFixture,
    the_value_is_monotone_where_the_fixture_says,
)
from lofi.spec.losses.vector_anchor import (
    AnchorAggregateName,
    AnchorDistanceName,
    VectorAnchorLossSpec,
)
from lofi.spec.readouts import DecodedMeaningSettings, ReadoutSource
from lofi.spec.readouts.terms import ReadoutMass
from lofi.terms import (
    BatchEncoding,
    ChoiceIndex,
    CompleteTree,
    Encoding,
    Identity,
    Operator,
    OperatorIndex,
    Signature,
    TargetSet,
    TermCodec,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch import Tensor

    from lofi.nn.loss._ports import (
        LossTerm,
        ProofRewardBindings,
        SpanTokenDecoder,
    )
    from lofi.nn.ports import TermEncoder
    from lofi.nn.spaces import Space, SpaceRelationBinding
    from lofi.spec import ConfiguredSpec
    from lofi.terms import Term

NOMINALS: Final = ReadoutId("nominal/types")
CLAUSES: Final = ReadoutId("clausal.types")
SPANS: Final = SpanIndex(2)
READOUT_SIGNATURE: Final = Signature(
    "anchor-readout", ("T",), (Operator("a", "T", family="primitive"),)
)
READOUT_CODEC: Final = NeuralCodecView(
    TermCodec(
        OperatorIndex(READOUT_SIGNATURE),
        CompleteTree(
            READOUT_SIGNATURE, READOUT_SIGNATURE.operators, order="bf", depth=1
        ),
    )
)
READOUT_CHOICES: Final = ChoiceIndex(READOUT_SIGNATURE, depth_bounds={"T": 1})

NOMINAL_TARGET = TargetSet(("N", "NP"), Identity())
CLAUSAL_TARGET = TargetSet(("S",), Identity())

type Constraint = BatchLocalConstraint[str, str]


class Batch(NamedTuple):
    """A batch with localized type constraints."""

    constraints: tuple[Constraint, ...]


class Brackets(NamedTuple):
    """A batch localizing only negative brackets."""

    constraints: tuple[BatchLocalNegativeBracket[str], ...]


type Outputs = ObjectiveOutputs[Batch]


def encoding(symbol: int) -> Encoding:
    return Encoding((symbol,), (Encoding.NO_PARENT,), (0,), (True,))


TARGETS: Final = MappingProxyType(
    {
        NOMINALS: BatchEncoding((encoding(1), encoding(2))),
        CLAUSES: BatchEncoding((encoding(3),)),
    }
)


class Encoder:
    """A fixed lookup encoder for target terms."""

    def __init__(
        self,
        vectors: Tensor | None = None,
        *,
        mask: Tensor | None = None,
        axes: tuple[str, ...] = ("term", "width"),
    ) -> None:
        self.vectors = (
            torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                dtype=torch.float64,
                requires_grad=True,
            )
            if vectors is None
            else vectors
        )
        self.mask = mask
        self.axes = axes
        self.calls = 0

    def encode(self, terms: BatchEncoding, /) -> Encoded:
        self.calls += 1
        rows = torch.tensor([row.symbols[0] for row in terms.rows])

        return Encoded(self.vectors[rows], self.axes, self.mask)


def encoder(
    vectors: Tensor | None = None,
    *,
    mask: Tensor | None = None,
    axes: tuple[str, ...] = ("term", "width"),
) -> tuple[Encoder, TermEncoder]:
    built = Encoder(vectors, mask=mask, axes=axes)

    return built, cast("TermEncoder", built)


def constraint(
    readout: ReadoutId,
    *,
    item: int,
    span: tuple[int, int],
    negate: bool = False,
    penalty: float | None = None,
) -> Constraint:
    target = NOMINAL_TARGET if readout == NOMINALS else CLAUSAL_TARGET

    return BatchLocalConstraint(
        target,
        (span,),
        (item,),
        "token",
        readout=readout,
        mode=ClampMode.HARD if penalty is None else ClampMode.SOFT,
        penalty=penalty,
        negate=negate,
    )


CONSTRAINTS: Final = (
    constraint(NOMINALS, item=0, span=(0, 1)),
    constraint(CLAUSES, item=1, span=(0, 1)),
    constraint(NOMINALS, item=0, span=(0, 2), negate=True),
)

LIVE: Final = torch.tensor([[True, True, True], [True, True, False]])
DISTANCE: Final = AnchorDistanceName.COSINE
AGGREGATE: Final = AnchorAggregateName.NEAREST
MARGIN: Final = 1.0


def cell_values() -> Tensor:
    return torch.tensor(
        [
            [[0.8, 0.6], [-0.6, -0.8], [-0.8, -0.6]],
            [[0.6, 0.8], [0.8, -0.6], [7.0, 9.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )


def assemble(
    cells: Tensor | None = None,
    *,
    constraints: tuple[Constraint, ...] = CONSTRAINTS,
    mask: Tensor | None = LIVE,
    axes: tuple[str, ...] = CELL_AXES,
) -> Outputs:
    values = cell_values() if cells is None else cells

    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(LogDistribution(torch.full((2, 3), -1.0), ("item", "span"))),
        batch=Batch(constraints),
        cells=Encoded(values, axes, mask),
    )


def bound(
    targets: Mapping[ReadoutId, BatchEncoding] = TARGETS,
) -> tuple[Encoder, Objective[PublishedOutputs[Batch]]]:
    built, context = encoder()
    objective = VectorAnchorObjective[str, str](
        targets=targets,
        name="vector-anchor",
        distance=DISTANCE,
        aggregate=AGGREGATE,
        margin=MARGIN,
    ).bind(context)

    return built, cast("Objective[PublishedOutputs[Batch]]", objective)


def manual(outputs: Outputs) -> float:
    targets = {
        NOMINALS: torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        CLAUSES: torch.tensor([[-1.0, 0.0]], dtype=torch.float64),
    }
    losses = []
    cells = outputs.cells

    assert cells is not None

    for held in outputs.batch.constraints:
        for item in held.item_positions:
            for start, end in held.spans:
                cell = cells.tensor[item, SPANS.span_id(start, end)]
                similarities = torch.nn.functional.cosine_similarity(
                    cell.unsqueeze(0), targets[held.readout], dim=-1
                )
                distance = (1.0 - similarities).min()
                loss = (
                    torch.relu(distance.new_tensor(MARGIN) - distance)
                    if held.negate
                    else distance
                )
                losses.append(loss * (1.0 if held.penalty is None else held.penalty))

    return float(torch.stack(losses).mean().detach())


def permuted(outputs: Outputs) -> Outputs:
    cells = outputs.cells

    assert cells is not None

    reordered = tuple(
        BatchLocalConstraint(
            held.targets,
            held.spans,
            tuple(1 - item for item in held.item_positions),
            held.scope,
            readout=held.readout,
            mode=held.mode,
            penalty=held.penalty,
            negate=held.negate,
        )
        for held in reversed(outputs.batch.constraints)
    )

    return assemble(
        cells.tensor.flip(0).detach().requires_grad_(),
        constraints=reordered,
        mask=cells.mask.flip(0) if cells.mask is not None else None,
    )


def refilled(outputs: Outputs) -> Outputs:
    cells = outputs.cells

    assert cells is not None
    values = cells.tensor.detach().clone()
    values[~LIVE] = cells.tensor.new_tensor([101.0, -203.0])

    return assemble(values.requires_grad_(), constraints=outputs.batch.constraints)


def inapplicable() -> Outputs:
    return assemble(constraints=())


def models_built(call: Callable[[], LossTerm]) -> int:
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


def fixture() -> ObjectiveFixture[Batch]:
    _, objective = bound()
    far = cell_values().detach()
    near = far.clone()
    far[0, SPANS.span_id(0, 1)] = torch.tensor([-1.0, -1.0])
    near[0, SPANS.span_id(0, 1)] = torch.tensor([1.0, 0.01])
    one = (constraint(NOMINALS, item=0, span=(0, 1)),)

    return ObjectiveFixture(
        objective=objective,
        outputs=assemble,
        inputs=lambda outputs: (cast("Encoded", outputs.cells).tensor,),
        unweighted=manual,
        monotone=lambda: (
            assemble(far.clone().requires_grad_(), constraints=one),
            assemble(near.clone().requires_grad_(), constraints=one),
        ),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        inapplicable=inapplicable,
        tolerance=1e-9,
    )


class TestVectorAnchorObjectiveLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Batch]:
        return fixture()

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that moving a cell toward a permitted target lowers the value.

        Parameters
        ----------
        conformance : ObjectiveFixture[BatchT]
            What the law runs against.
        """
        the_value_is_monotone_where_the_fixture_says(conformance)


class TestVectorAnchorBindingLaws(BoundObjectiveLawSuite):
    @pytest.fixture
    def bound_conformance(
        self,
    ) -> BoundObjectiveFixture[TermEncoder, PublishedOutputs[Batch]]:
        built, context = encoder()
        before = built.vectors.detach().clone()
        recipe = VectorAnchorObjective[str, str](
            targets=TARGETS,
            name="vector-anchor",
            distance=DISTANCE,
            aggregate=AGGREGATE,
            margin=MARGIN,
        )

        return BoundObjectiveFixture(
            recipe,
            context,
            lambda _: torch.equal(built.vectors, before),
        )


def test_binding_returns_a_new_objective_and_detaches_targets_once() -> None:
    unbound = VectorAnchorObjective[str, str](
        targets=TARGETS,
        name="vector-anchor",
        distance=DISTANCE,
        aggregate=AGGREGATE,
        margin=MARGIN,
    )
    built, context = encoder()

    objective = unbound.bind(context)
    objective.evaluate(assemble()).value.backward()

    assert isinstance(unbound, BoundObjective)
    assert not isinstance(unbound, Objective)
    assert isinstance(objective, Objective)
    assert built.calls == len(TARGETS)
    assert built.vectors.grad is None
    assert isinstance(objective, torch.nn.Module)
    assert all(not tensor.requires_grad for tensor in objective.buffers())


def test_positive_constraints_use_the_nearest_target() -> None:
    _, objective = bound()
    cells = cell_values().detach()
    cells[0, SPANS.span_id(0, 1)] = torch.tensor([1.0, 0.0])
    only = (constraint(NOMINALS, item=0, span=(0, 1)),)

    term = objective.evaluate(assemble(cells.requires_grad_(), constraints=only))

    assert term.value.item() == pytest.approx(0.0, abs=1e-12)
    assert term.count == 1


def test_negated_constraints_enforce_the_declared_margin_from_the_nearest_target() -> (
    None
):
    _, objective = bound()
    only = (constraint(NOMINALS, item=0, span=(0, 1), negate=True),)
    sitting = cell_values().detach()
    sitting[0, SPANS.span_id(0, 1)] = torch.tensor([1.0, 0.0])
    away = sitting.clone()
    away[0, SPANS.span_id(0, 1)] = torch.tensor([-1.0, -1.0])

    on_one = objective.evaluate(
        assemble(sitting.requires_grad_(), constraints=only)
    ).value
    far_from_both = objective.evaluate(
        assemble(away.requires_grad_(), constraints=only)
    ).value

    assert on_one.item() == pytest.approx(MARGIN)
    assert far_from_both.item() == 0.0
    assert on_one > far_from_both


def test_distance_aggregation_and_margin_each_change_the_loss() -> None:
    cells = cell_values().detach()
    cells[0, SPANS.span_id(0, 1)] = torch.tensor([0.9, 0.1])
    _, context = encoder()

    def value(
        distance: AnchorDistanceName,
        aggregate: AnchorAggregateName,
        margin: float,
        *,
        negate: bool = False,
    ) -> float:
        objective = VectorAnchorObjective[str, str](
            targets=TARGETS,
            name="vector-anchor",
            distance=distance,
            aggregate=aggregate,
            margin=margin,
        ).bind(context)
        constraints = (constraint(NOMINALS, item=0, span=(0, 1), negate=negate),)
        return objective.evaluate(
            assemble(cells.clone().requires_grad_(), constraints=constraints)
        ).value.item()

    cosine_nearest = value(AnchorDistanceName.COSINE, AnchorAggregateName.NEAREST, 1.0)
    euclidean_nearest = value(
        AnchorDistanceName.SQUARED_EUCLIDEAN, AnchorAggregateName.NEAREST, 1.0
    )
    cosine_mean = value(AnchorDistanceName.COSINE, AnchorAggregateName.MEAN, 1.0)
    narrow_margin = value(
        AnchorDistanceName.COSINE, AnchorAggregateName.NEAREST, 0.5, negate=True
    )
    wide_margin = value(
        AnchorDistanceName.COSINE, AnchorAggregateName.NEAREST, 1.5, negate=True
    )

    assert cosine_nearest != pytest.approx(euclidean_nearest)
    assert cosine_nearest != pytest.approx(cosine_mean)
    assert narrow_margin != pytest.approx(wide_margin)


def test_target_identity_survives_mapping_and_constraint_reordering() -> None:
    forward_targets = {NOMINALS: TARGETS[NOMINALS], CLAUSES: TARGETS[CLAUSES]}
    reverse_targets = {CLAUSES: TARGETS[CLAUSES], NOMINALS: TARGETS[NOMINALS]}
    _, first = bound(forward_targets)
    _, second = bound(reverse_targets)
    outputs = assemble()
    reversed_constraints = assemble(constraints=CONSTRAINTS[::-1])

    assert first.evaluate(outputs).value.item() == pytest.approx(
        second.evaluate(reversed_constraints).value.item()
    )
    assert isinstance(first, torch.nn.Module)
    assert isinstance(second, torch.nn.Module)
    assert tuple(first.state_dict()) == tuple(second.state_dict())
    assert tuple(first.state_dict()) == tuple(
        f"target_{readout.encode('utf-8').hex()}" for readout in sorted(TARGETS)
    )


def test_the_gradient_reaches_constrained_cells_and_no_target() -> None:
    built, objective = bound()
    outputs = assemble()

    objective.evaluate(outputs).value.backward()

    cells = cast("Encoded", outputs.cells).tensor
    touched = torch.zeros_like(cells, dtype=torch.bool)

    for held in CONSTRAINTS:
        for item in held.item_positions:
            for start, end in held.spans:
                touched[item, SPANS.span_id(start, end)] = True

    assert cells.grad is not None
    assert bool(cells.grad[touched].ne(0.0).any())
    assert not bool(cells.grad[~touched].any())
    assert built.vectors.grad is None


def test_a_soft_anchor_reports_and_applies_its_gradient_scale() -> None:
    _, objective = bound()
    hard_outputs = assemble(constraints=(constraint(NOMINALS, item=0, span=(0, 1)),))
    soft_outputs = assemble(
        constraints=(constraint(NOMINALS, item=0, span=(0, 1), penalty=0.25),)
    )
    hard = objective.evaluate(hard_outputs)
    soft = objective.evaluate(soft_outputs)

    hard.value.backward()
    soft.value.backward()

    hard_cells = cast("Encoded", hard_outputs.cells).tensor
    soft_cells = cast("Encoded", soft_outputs.cells).tensor

    assert hard_cells.grad is not None
    assert soft_cells.grad is not None
    assert soft.value.item() == pytest.approx(0.25 * hard.value.item())
    assert torch.allclose(soft_cells.grad, 0.25 * hard_cells.grad)
    assert hard.diagnostics["gradient_scale"].item() == 1.0
    assert soft.diagnostics["gradient_scale"].item() == 0.25


def test_a_constraint_with_no_bound_target_is_refused() -> None:
    _, objective = bound({NOMINALS: TARGETS[NOMINALS]})
    unknown = (constraint(CLAUSES, item=0, span=(0, 1)),)

    with pytest.raises(ValueError, match="no target"):
        objective.evaluate(assemble(constraints=unknown))


def test_a_negative_bracket_is_refused_as_bearing_no_anchor_target() -> None:
    _, objective = bound()
    brackets = cast("Objective[PublishedOutputs[Brackets]]", objective)
    outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(LogDistribution(torch.full((2, 3), -1.0), ("item", "span"))),
        batch=Brackets((BatchLocalNegativeBracket(((0, 1),), (0,), "token"),)),
        cells=Encoded(cell_values(), CELL_AXES, LIVE),
    )

    with pytest.raises(TypeError, match="negative bracket"):
        brackets.evaluate(outputs)


def test_a_constraint_on_a_masked_cell_is_refused() -> None:
    _, objective = bound()
    masked = (constraint(NOMINALS, item=1, span=(0, 2)),)

    with pytest.raises(UncomputedCellError):
        objective.evaluate(assemble(constraints=masked))


def test_a_mask_dividing_cell_features_is_refused() -> None:
    _, objective = bound()
    divided = LIVE.unsqueeze(-1).expand(2, 3, 2)

    with pytest.raises(ValueError, match="divides a cell's features"):
        objective.evaluate(assemble(mask=divided))


def test_an_incompatible_target_mask_is_refused_at_binding() -> None:
    unbound = VectorAnchorObjective[str, str](
        targets={NOMINALS: TARGETS[NOMINALS]},
        name="vector-anchor",
        distance=DISTANCE,
        aggregate=AGGREGATE,
        margin=MARGIN,
    )
    _, context = encoder(mask=torch.tensor([True, False]))

    with pytest.raises(ValueError, match="masked a term"):
        unbound.bind(context)


def test_an_empty_target_set_is_refused() -> None:
    with pytest.raises(ValueError, match="has no term"):
        VectorAnchorObjective[str, str](
            targets={NOMINALS: BatchEncoding(())},
            name="vector-anchor",
            distance=DISTANCE,
            aggregate=AGGREGATE,
            margin=MARGIN,
        )


def test_a_batch_with_no_constraint_gets_a_connected_zero() -> None:
    _, objective = bound()

    term = objective.evaluate(inapplicable())

    assert term.count == 0
    assert term.value.item() == 0.0
    assert term.value.grad_fn is not None


def test_the_value_matches_the_independent_formula() -> None:
    _, objective = bound()
    outputs = assemble()

    term = objective.evaluate(outputs)

    assert term.value.item() == pytest.approx(manual(outputs), rel=1e-12)
    assert term.count == len(CONSTRAINTS)


def test_the_bound_objective_reads_only_its_declared_outputs() -> None:
    _, objective = bound()

    assert objective.requires == frozenset({"batch", "cells", "spans"})
    assert objective.evaluate(assemble()).count == len(CONSTRAINTS)


class DistributionDecoder:
    """A decoder declaring the capability a term readout requires."""

    def marginal_rows(self, inputs: Encoded, /, *, root_sort: str) -> LogDistribution:
        raise AssertionError(inputs, root_sort)


class Binding:
    """A term readout and its represented-space encoder."""

    def __init__(self) -> None:
        self.readout = TermsReadout(
            DistributionDecoder(),
            codec=READOUT_CODEC,
            index=READOUT_CHOICES,
            reads=ReadoutSource.INSIDE,
            decoding=None,
            key=ReadoutId("types"),
            targets={
                readout: LinearizedTargets(
                    terms,
                    0,
                    "identity",
                    cast(
                        "TargetSet[Term]",
                        {NOMINALS: NOMINAL_TARGET, CLAUSES: CLAUSAL_TARGET}[readout],
                    ),
                )
                for readout, terms in TARGETS.items()
            },
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
        self.encoder, self.context = encoder()
        self.requests: list[str] = []

    @property
    def name(self) -> str:
        return "anchored-types"

    def terms_readout(self, role: str, /) -> TermsReadout:
        self.requests.append(role)

        if role != "anchored":
            msg = f"no term readout is bound as {role!r}"
            raise ValueError(msg)

        return self.readout

    def space(self, role: str, /) -> Space:
        msg = f"vector anchors must not resolve represented space {role!r}"
        raise AssertionError(msg)

    def readout_space(self, role: str, /) -> Space:
        if role != "anchored":
            msg = f"no represented space is bound to term readout {role!r}"
            raise ValueError(msg)

        return cast("Space", self)

    def relation(self, role: str, /) -> SpaceRelationBinding:
        msg = f"vector anchors must not resolve space relation {role!r}"
        raise AssertionError(msg)

    def span_decoder(self, role: str, /) -> SpanTokenDecoder:
        msg = f"vector anchors must not resolve span decoder {role!r}"
        raise AssertionError(msg)

    def proof_reward(self) -> ProofRewardBindings:
        msg = "vector anchors must not resolve proof-reward bindings"
        raise AssertionError(msg)


DECLARATION: Final = VectorAnchorLossSpec(readout="anchored")


def configured(context: Binding | None = None) -> Objective[PublishedOutputs[Batch]]:
    return cast(
        "Objective[PublishedOutputs[Batch]]",
        anchor_module.PLUGIN.construct(
            DECLARATION, Binding() if context is None else context
        ),
    )


class TestConfiguredVectorAnchorLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Batch]:
        return fixture()._replace(objective=configured())

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that moving a cell toward a permitted target lowers the value.

        Parameters
        ----------
        conformance : ObjectiveFixture[BatchT]
            What the law runs against.
        """
        the_value_is_monotone_where_the_fixture_says(conformance)


class TestVectorAnchorPlugin(ConfiguredComponentConformance):
    @pytest.fixture
    def configured_component_fixture(self) -> ConfiguredComponentFixture:
        context = Binding()
        before = context.encoder.vectors.detach().clone()
        _, direct = bound()

        def observe(built: object) -> dict[str, bool]:
            objective = cast("Objective[PublishedOutputs[Batch]]", built)
            outputs = assemble()

            return {
                "named for its term": objective.name == "anchored-types",
                "reads the declared readout": context.requests[-1] == "anchored",
                "anchors at the encoder's vectors": objective.evaluate(
                    outputs
                ).value.item()
                == pytest.approx(direct.evaluate(outputs).value.item(), rel=1e-12),
            }

        return ConfiguredComponentFixture(
            declaration=cast("ConfiguredSpec[object]", DECLARATION),
            context=context,
            expected_type=type(direct),
            observe=observe,
            unchanged=lambda: torch.equal(context.encoder.vectors, before),
        )


def test_the_plugin_anchors_every_target_set_of_the_declared_readout() -> None:
    context = Binding()
    objective = configured(context)

    assert objective.name == "anchored-types"
    assert objective.requires == frozenset({"batch", "cells", "spans"})
    assert context.requests == ["anchored"]
    assert context.encoder.calls == len(TARGETS)


def test_the_plugin_preserves_the_declared_distance_aggregate_and_margin() -> None:
    context = Binding()
    declaration = VectorAnchorLossSpec.model_validate(
        {
            "readout": "anchored",
            "distance": "squared-euclidean",
            "aggregate": "mean",
            "margin": 0.25,
        }
    )
    built = cast(
        "VectorAnchorObjective[str, str]",
        anchor_module.PLUGIN.construct(declaration, context),
    )

    assert built.distance is AnchorDistanceName.SQUARED_EUCLIDEAN
    assert built.aggregate is AnchorAggregateName.MEAN
    assert built.margin == 0.25
