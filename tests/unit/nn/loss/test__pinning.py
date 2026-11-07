import math
from collections.abc import Iterable, Mapping
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, cast, override

import didactic.api as dx
import pytest
import torch
from tests.conformance.configured_component import (
    ConfiguredComponentConformance,
    ConfiguredComponentFixture,
)
from tests.conformance.objective import ObjectiveLawSuite
from tests.language import language

import lofi.nn.loss._pinning as pinning_module
from lofi._types import ItemId, ReadoutId
from lofi.deduce import (
    Edge,
    ItemKey,
    LogInside,
    PackedForest,
    RootGoal,
    SpanIndex,
    outside_from_edges,
    target_mass,
)
from lofi.lexicon import ClampMode
from lofi.lf import term_codec
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.codec import NeuralCodecView
from lofi.nn.decoders import RecursiveTermDecoder, TreeDecoderSettings
from lofi.nn.loss._constraint import (
    BatchLocalConstraint,
    BatchLocalNegativeBracket,
    ConstraintOccurrence,
    LogicalFormConstraint,
)
from lofi.nn.loss._errors import (
    ImpossibleNegativeBracketError,
    ImproperMassError,
    UncomputedCellError,
)
from lofi.nn.loss._linearize import (
    LinearizedTargets,
    linearize_targets,
)
from lofi.nn.loss._mapping import SOFTMAX
from lofi.nn.loss._parser import (
    SPAN_AXES,
    SPLIT_AXES,
    SpanMarginals,
)
from lofi.nn.loss._pinning import SpanConstraintObjective
from lofi.nn.loss._readout import (
    CELL_AXES,
    MASS_AXES,
    TERMS_READOUT_AXES,
    TermsReadout,
    log_complement,
    log_mass,
)
from lofi.nn.loss._suite import ObjectiveSuite
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
    RecordingOutputs,
    SpanReadout,
    SpanTerms,
)
from lofi.nn.loss.laws import (
    ObjectiveFixture,
    the_value_is_monotone_where_the_fixture_says,
)
from lofi.nn.sites import ComponentId
from lofi.nn.type_directed import UNRESTRICTED_ADMISSION
from lofi.spec.losses.pinning import PinningLossSpec
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
    Term,
    TermCodec,
    TermStore,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from torch import Tensor

    from lofi.judge import Action
    from lofi.nn.forms import WeightedTerms
    from lofi.nn.loss._ports import (
        LossTerm,
        ProofRewardBindings,
        SpanTokenDecoder,
    )
    from lofi.nn.ports import MarginalDecoder
    from lofi.nn.spaces import Space, SpaceRelationBinding
    from lofi.spec import ConfiguredSpec
    from lofi.terms import Matching


FORM_LANGUAGE: Final = language(("bo", "likes"), types={"bo": "e", "likes": "e -> t"})

SIGNATURE: Final = Signature(
    "pinning-fixture",
    ("Cat",),
    tuple(
        Operator(name, "Cat", family="primitive")
        for name in ("S", "NP", "VP", "D", "N", "V")
    ),
)
INDEX: Final = OperatorIndex(SIGNATURE)
CHOICES: Final = ChoiceIndex(SIGNATURE, depth_bounds={"Cat": 0})
CODEC: Final = TermCodec(
    INDEX, CompleteTree(SIGNATURE, SIGNATURE.operators, order="df", depth=1)
)
VIEW: Final = NeuralCodecView(CODEC)
STORE: Final = TermStore(SIGNATURE)
MATCHING: Final[Matching[Term]] = Identity()


def category(name: str) -> Term:
    return STORE.op(SIGNATURE.by_name(name))


NOUNS: Final = TargetSet((category("N"), category("NP")), MATCHING)
VERBS: Final = TargetSet((category("V"), category("VP")), MATCHING)
SENTENCES: Final = TargetSet((category("S"),), MATCHING)
NOUN_MASS: Final = ReadoutId("nouns")
VERB_MASS: Final = ReadoutId("verbs")
TYPES: Final = ReadoutId("types")
DECLARED: Final[Mapping[ReadoutId, TargetSet[Term]]] = {
    NOUN_MASS: NOUNS,
    VERB_MASS: VERBS,
}

LENGTH: Final = 3
ITEMS: Final = 2
LENGTHS: Final = (3, 2)
SPANS: Final = SpanIndex(LENGTH)
FILL: Final = -9.0
SCOPE: Final = "token"

type Constraint = BatchLocalConstraint[Term, str]


class Constrained(NamedTuple):
    """A batch of items with the constraints localized to their positions."""

    ids: tuple[str, ...]
    constraints: tuple[Constraint, ...]


type Outputs = ObjectiveOutputs[Constrained]

IDS: Final = ("first", "second")


def live_spans(lengths: tuple[int, ...] = LENGTHS) -> Tensor:
    mask = torch.zeros(len(lengths), SPANS.num_spans, dtype=torch.bool)

    for item, length in enumerate(lengths):
        for start, end in SPANS.spans_by_length():
            if end <= length:
                mask[item, SPANS.span_id(start, end)] = True

    return mask


LIVE: Final = live_spans()


def objective(
    readouts: Iterable[ReadoutId] = tuple(DECLARED),
    *,
    name: str = "pinning",
) -> SpanConstraintObjective[Term, str]:
    return SpanConstraintObjective(readouts=readouts, name=name)


def constraint(
    targets: TargetSet[Term],
    spans: tuple[tuple[int, int], ...],
    positions: tuple[int, ...],
    *,
    readout: ReadoutId | None = None,
    negate: bool = False,
    penalty: float | None = None,
) -> Constraint:
    assigned = VERB_MASS if targets is VERBS else NOUN_MASS

    return BatchLocalConstraint(
        targets,
        spans,
        positions,
        SCOPE,
        readout=assigned if readout is None else readout,
        mode=ClampMode.HARD if penalty is None else ClampMode.SOFT,
        penalty=penalty,
        negate=negate,
    )


CONSTRAINTS: Final = (
    constraint(NOUNS, ((0, 1),), (0, 1)),
    constraint(VERBS, ((1, 2),), (0,), negate=True),
    constraint(NOUNS, ((0, 2),), (1,)),
)


def random_log_masses(seed: int, items: int = ITEMS) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    masses = 0.05 + 0.9 * torch.rand(
        items, SPANS.num_spans, generator=generator, dtype=torch.float64
    )

    return masses.log()


def pair_of(
    log_masses: Tensor,
    *,
    computed: Tensor = LIVE,
    fill: float = FILL,
) -> SpanReadout:
    rest = torch.log(-torch.expm1(log_masses))
    pair = torch.stack((log_masses, rest), dim=-1)
    padded = torch.where(computed.unsqueeze(-1), pair, fill)

    return SpanReadout(
        SPANS,
        LogDistribution(
            padded.detach().requires_grad_(),
            MASS_AXES,
            computed,
            normalized_over="outcome",
        ),
    )


def assemble(
    masses: Mapping[ReadoutId, Tensor],
    *,
    constraints: tuple[Constraint, ...] = CONSTRAINTS,
    computed: Tensor = LIVE,
    fill: float = FILL,
    ids: tuple[str, ...] = IDS,
) -> Outputs:
    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(torch.where(computed, -1.0, fill), SPAN_AXES, computed)
        ),
        batch=Constrained(ids, constraints),
        readouts={
            key: pair_of(held, computed=computed, fill=fill)
            for key, held in masses.items()
        },
    )


def masses(seed: int = 0) -> dict[ReadoutId, Tensor]:
    return {NOUN_MASS: random_log_masses(seed), VERB_MASS: random_log_masses(seed + 1)}


def bundle() -> Outputs:
    return assemble(masses())


def log_masses_of(outputs: Outputs) -> dict[ReadoutId, Tensor]:
    return {key: log_mass(held.distribution) for key, held in outputs.readouts.items()}


def computed_of(outputs: Outputs) -> Tensor:
    return next(iter(outputs.readouts.values())).computed


def permuted(outputs: Outputs) -> Outputs:
    order = torch.tensor([1, 0])
    swapped = tuple(
        constraint(
            held.targets,
            held.spans,
            tuple(1 - position for position in held.item_positions),
            readout=held.readout,
            penalty=held.penalty,
            negate=held.negate,
        )
        for held in outputs.batch.constraints
    )

    return assemble(
        {key: held[order] for key, held in log_masses_of(outputs).items()},
        constraints=swapped,
        computed=computed_of(outputs)[order],
        ids=tuple(reversed(outputs.batch.ids)),
    )


def refilled(outputs: Outputs) -> Outputs:
    return assemble(
        log_masses_of(outputs),
        constraints=outputs.batch.constraints,
        computed=computed_of(outputs),
        fill=FILL / 3.0,
        ids=outputs.batch.ids,
    )


def unconstrained() -> Outputs:
    return assemble(masses(), constraints=())


def inputs_of(outputs: Outputs) -> tuple[Tensor, ...]:
    return tuple(outputs.readouts[key].distribution.tensor for key in DECLARED)


def instances(
    constraints: tuple[Constraint, ...],
) -> Iterator[tuple[Constraint, int, tuple[int, int]]]:
    for held in constraints:
        for item in held.item_positions:
            for span in held.spans:
                yield held, item, span


def by_hand(outputs: Outputs) -> float:
    read: list[float] = []

    for held, item, (start, end) in instances(outputs.batch.constraints):
        cell = outputs.readouts[held.readout].at(item, start, end)
        value = log_complement(cell) if held.negate else log_mass(cell)
        weight = 1.0 if held.penalty is None else held.penalty
        read.append(-float(value.detach()) * weight)

    return sum(read) / len(read) if read else 0.0


def with_mass(mass: float, *, elsewhere: float = 0.5) -> dict[ReadoutId, Tensor]:
    held = torch.full((ITEMS, SPANS.num_spans), math.log(elsewhere)).double()

    for constrained, item, (start, end) in instances(CONSTRAINTS):
        held[item, SPANS.span_id(start, end)] = math.log(
            1.0 - mass if constrained.negate else mass
        )

    return {NOUN_MASS: held, VERB_MASS: held.clone()}


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


def fixture() -> ObjectiveFixture[Constrained]:
    return ObjectiveFixture(
        objective=objective(),
        outputs=bundle,
        inputs=inputs_of,
        unweighted=by_hand,
        monotone=lambda: (assemble(with_mass(0.2)), assemble(with_mass(0.9))),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        inapplicable=unconstrained,
        tolerance=1e-9,
    )


class TestSpanConstraintObjectiveLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Constrained]:
        return fixture()

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that less mass on the permitted sets scores above more."""
        the_value_is_monotone_where_the_fixture_says(conformance)


def test_the_value_is_the_mean_negative_log_mass_over_the_instances() -> None:
    outputs = bundle()

    term = objective().evaluate(outputs)

    assert term.value.item() == pytest.approx(by_hand(outputs), rel=1e-12)
    assert term.count == 4
    assert term.value.item() > 0.0


def test_the_count_is_the_number_of_item_and_span_pairs() -> None:
    wide = constraint(NOUNS, ((0, 1), (1, 2)), (0, 1))
    narrow = constraint(VERBS, ((0, 2),), (0,))

    term = objective().evaluate(assemble(masses(), constraints=(wide, narrow)))

    assert term.count == 5


def test_raising_the_mass_of_a_permitted_set_strictly_lowers_the_value() -> None:
    pinned = objective()
    values = [
        pinned.evaluate(assemble(with_mass(mass))).value.item()
        for mass in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
    ]

    assert all(later < earlier for earlier, later in pairwise(values))
    assert values[0] == pytest.approx(-math.log(0.1), rel=1e-12)


def test_the_value_over_two_constraints_is_the_mean_over_both() -> None:
    first = constraint(NOUNS, ((0, 1),), (0,))
    second = constraint(VERBS, ((1, 2),), (1,))
    held = masses()
    pinned = objective()

    both = pinned.evaluate(assemble(held, constraints=(first, second)))
    alone = [
        pinned.evaluate(assemble(held, constraints=(one,))).value.item()
        for one in (first, second)
    ]
    first_mass = held[NOUN_MASS][0, SPANS.span_id(0, 1)].item()
    second_mass = held[VERB_MASS][1, SPANS.span_id(1, 2)].item()

    assert alone[0] != pytest.approx(alone[1])
    assert both.count == 2
    assert both.value.item() == pytest.approx(sum(alone) / 2.0, rel=1e-12)
    assert both.value.item() == pytest.approx(
        -(first_mass + second_mass) / 2.0, rel=1e-12
    )


def test_a_negated_constraint_is_scored_on_the_mass_off_the_set() -> None:
    held = masses()
    log_m = held[VERB_MASS][0, SPANS.span_id(1, 2)].item()
    permitted = constraint(VERBS, ((1, 2),), (0,))
    excluded = constraint(VERBS, ((1, 2),), (0,), negate=True)
    pinned = objective()

    on = pinned.evaluate(assemble(held, constraints=(permitted,))).value.item()
    off = pinned.evaluate(assemble(held, constraints=(excluded,))).value.item()

    assert on == pytest.approx(-log_m, rel=1e-12)
    assert off == pytest.approx(-math.log1p(-math.exp(log_m)), rel=1e-12)
    assert off != pytest.approx(on)


def test_a_negated_constraint_rewards_mass_leaving_the_set() -> None:
    excluded = constraint(VERBS, ((1, 2),), (0,), negate=True)
    pinned = objective()
    values = []

    for mass in (0.9, 0.5, 0.1):
        held = masses()
        held[VERB_MASS][0, SPANS.span_id(1, 2)] = math.log(mass)
        values.append(pinned.evaluate(assemble(held, constraints=(excluded,))).value)

    assert values[0].item() > values[1].item() > values[2].item()
    assert values[2].item() == pytest.approx(-math.log(0.9), rel=1e-12)


def test_a_soft_constraint_scales_its_value_and_gradient_by_its_penalty() -> None:
    hard_outputs = assemble(
        masses(seed=8),
        constraints=(constraint(NOUNS, ((0, 1),), (0,)),),
    )
    soft_outputs = assemble(
        masses(seed=8),
        constraints=(constraint(NOUNS, ((0, 1),), (0,), penalty=0.25),),
    )
    pinned = objective()
    hard = pinned.evaluate(hard_outputs)
    soft = pinned.evaluate(soft_outputs)

    hard.value.backward()
    soft.value.backward()

    hard_gradient = inputs_of(hard_outputs)[0].grad
    soft_gradient = inputs_of(soft_outputs)[0].grad

    assert hard_gradient is not None
    assert soft_gradient is not None
    assert soft.value.item() == pytest.approx(0.25 * hard.value.item(), rel=1e-12)
    assert torch.allclose(soft_gradient, 0.25 * hard_gradient)
    assert hard.diagnostics["gradient_scale"].item() == 1.0
    assert soft.diagnostics["gradient_scale"].item() == 0.25


def test_the_satisfied_diagnostic_is_the_mean_satisfied_probability() -> None:
    term = objective().evaluate(assemble(with_mass(0.25)))

    assert term.diagnostics["satisfied"].item() == pytest.approx(0.25, rel=1e-12)
    assert term.diagnostics["satisfied"].grad_fn is None


def test_a_batch_with_no_constraint_gets_the_zero_term() -> None:
    term = objective().evaluate(unconstrained())

    assert term.count == 0
    assert term.value.item() == 0.0
    assert term.value.grad_fn is not None


def test_the_gradient_reaches_the_constrained_cells_alone() -> None:
    outputs = bundle()

    objective().evaluate(outputs).value.backward()

    touched = {key: torch.zeros(ITEMS, SPANS.num_spans, 2).bool() for key in DECLARED}
    keys = {NOUNS: NOUN_MASS, VERBS: VERB_MASS}

    for held, item, (start, end) in instances(CONSTRAINTS):
        outcome = 1 if held.negate else 0
        touched[keys[held.targets]][item, SPANS.span_id(start, end), outcome] = True

    for key in DECLARED:
        gradient = outputs.readouts[key].distribution.tensor.grad

        assert gradient is not None
        assert bool((gradient[touched[key]] != 0.0).all())
        assert not bool(gradient[~touched[key]].any())


class ReadKeys[K, V](Mapping[K, V]):
    """A mapping that records which keys are read through it."""

    def __init__(self, held: Mapping[K, V]) -> None:
        self._held = held
        self.read: list[K] = []

    @override
    def __getitem__(self, key: K) -> V:
        self.read.append(key)

        return self._held[key]

    @override
    def __iter__(self) -> Iterator[K]:
        return iter(self._held)

    @override
    def __len__(self) -> int:
        return len(self._held)


class Published(NamedTuple):
    """A step's outputs whose readouts are read through a recording mapping."""

    spans: SpanIndex
    chart: CellChart
    batch: Constrained
    readouts: ReadKeys[ReadoutId, SpanReadout]
    cells: Encoded | None = None
    outside_cells: Encoded | None = None
    splits: None = None
    scores: None = None
    marginals: None = None
    actions: Mapping[Action, LogDistribution] = MappingProxyType({})
    choice_rows: Mapping[ReadoutId, SpanReadout] = MappingProxyType({})
    terms: Mapping[ReadoutId, SpanTerms] = MappingProxyType({})
    forms: Mapping[ReadoutId, WeightedTerms] = MappingProxyType({})
    entailment: None = None
    entropy: Encoded | None = None
    mbr: Encoded | None = None


def test_the_objective_reads_the_batch_and_the_mass_ids_alone() -> None:
    pinned = objective()
    outputs = bundle()
    proxy = RecordingOutputs(outputs, objective=pinned.name, requires=pinned.requires)

    term = pinned.evaluate(proxy)

    assert proxy.touched == {"batch", "readouts"}
    assert term.count == 4


def test_the_objective_reads_the_mass_ids_and_never_the_types() -> None:
    types = SpanReadout(
        SPANS,
        LogDistribution(
            torch.zeros(ITEMS, SPANS.num_spans, 1, CODEC.n_symbols)
            .double()
            .log_softmax(dim=-1),
            TERMS_READOUT_AXES,
            normalized_over="symbol",
        ),
    )
    recorded = ReadKeys(
        {TYPES: types, **{key: pair_of(mass) for key, mass in masses().items()}}
    )
    outputs = Published(
        SPANS,
        CellChart(LogDistribution(torch.where(LIVE, -1.0, FILL), SPAN_AXES, LIVE)),
        Constrained(IDS, CONSTRAINTS),
        recorded,
        Encoded(torch.zeros(ITEMS, SPANS.num_spans, 2), CELL_AXES, LIVE),
    )

    term = objective().evaluate(outputs)

    assert set(recorded) == {TYPES, NOUN_MASS, VERB_MASS}
    assert set(recorded.read) == {NOUN_MASS, VERB_MASS}
    assert term.count == 4


def test_a_never_computed_cell_is_a_named_error() -> None:
    beyond = constraint(NOUNS, ((2, 3),), (1,))

    with pytest.raises(UncomputedCellError) as caught:
        objective().evaluate(assemble(masses(), constraints=(beyond,)))

    assert caught.value.item == 1
    assert caught.value.span == (2, 3)


def test_a_span_outside_the_input_is_refused() -> None:
    outside = constraint(NOUNS, ((0, LENGTH + 1),), (0,))

    with pytest.raises(ValueError, match="not a span of an input"):
        objective().evaluate(assemble(masses(), constraints=(outside,)))


def test_an_item_the_batch_does_not_hold_is_refused() -> None:
    absent = constraint(NOUNS, ((0, 1),), (ITEMS,))

    with pytest.raises(IndexError, match="no item"):
        objective().evaluate(assemble(masses(), constraints=(absent,)))


@pytest.mark.parametrize("negate", [False, True])
@pytest.mark.parametrize("published", [0.1, math.nan])
def test_a_log_mass_above_zero_or_not_a_number_is_refused_rather_than_a_nan(
    published: float, *, negate: bool
) -> None:
    pair = torch.full((ITEMS, SPANS.num_spans, 2), -1.0).double()
    pair[0, SPANS.span_id(0, 1), 1 if negate else 0] = published
    pair[1, SPANS.span_id(0, 2), 1 if negate else 0] = published
    outputs: Outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(torch.where(LIVE, -1.0, FILL), SPAN_AXES, LIVE)
        ),
        batch=Constrained(
            IDS,
            (
                constraint(NOUNS, ((1, 2),), (0,), negate=negate),
                constraint(NOUNS, ((0, 1), (0, 2)), (0, 1), negate=negate),
            ),
        ),
        readouts={
            NOUN_MASS: SpanReadout(SPANS, LogDistribution(pair, MASS_AXES, LIVE)),
            VERB_MASS: pair_of(random_log_masses(1)),
        },
    )

    with pytest.raises(ImproperMassError) as caught:
        objective().evaluate(outputs)

    assert caught.value.readout == NOUN_MASS
    assert caught.value.item == 0
    assert caught.value.span == (0, 1)
    assert math.isnan(caught.value.value) or caught.value.value == pytest.approx(0.1)
    assert "at most one" in str(caught.value)


def test_a_published_mass_of_exactly_one_is_accepted() -> None:
    held = masses()
    held[NOUN_MASS][0, SPANS.span_id(0, 1)] = 0.0
    pinned = constraint(NOUNS, ((0, 1),), (0,))

    term = objective().evaluate(assemble(held, constraints=(pinned,)))

    assert term.value.item() == 0.0


def test_a_constraint_on_an_undeclared_readout_is_refused() -> None:
    stray = constraint(
        SENTENCES,
        ((0, 3),),
        (0,),
        readout=ReadoutId("sentences"),
    )

    with pytest.raises(ValueError, match="not declared"):
        objective().evaluate(assemble(masses(), constraints=(stray,)))


def test_a_declared_id_the_step_did_not_publish_is_refused() -> None:
    with pytest.raises(ValueError, match="reads the mass 'verbs'"):
        objective().evaluate(assemble({NOUN_MASS: random_log_masses(0)}))


def test_a_declared_id_no_constraint_reads_is_still_required() -> None:
    only_nouns = tuple(held for held in CONSTRAINTS if held.targets is NOUNS)

    with pytest.raises(ValueError, match="reads the mass 'verbs'"):
        objective().evaluate(
            assemble({NOUN_MASS: random_log_masses(0)}, constraints=only_nouns)
        )


def test_the_fixtures_transform_the_outputs_they_are_handed() -> None:
    held = masses(seed=7)
    outputs = assemble(held)

    swapped = permuted(outputs)
    refilled_outputs = refilled(outputs)

    assert swapped.batch.ids == tuple(reversed(IDS))
    assert torch.equal(
        log_mass(swapped.readouts[NOUN_MASS].distribution)[LIVE[[1, 0]]],
        held[NOUN_MASS][[1, 0]][LIVE[[1, 0]]],
    )
    assert torch.equal(
        log_mass(refilled_outputs.readouts[VERB_MASS].distribution)[LIVE],
        held[VERB_MASS][LIVE],
    )
    assert bool(
        (
            log_mass(refilled_outputs.readouts[VERB_MASS].distribution)[~LIVE]
            == FILL / 3.0
        ).all()
    )
    assert refilled_outputs.batch.constraints == CONSTRAINTS


def test_target_provenance_does_not_change_the_assigned_readout() -> None:
    held = masses()
    pinned = objective()
    declared = constraint(NOUNS, ((0, 2),), (1,))
    reordered = TargetSet((category("NP"), category("N")), Identity())
    same = constraint(reordered, ((0, 2),), (1,), readout=NOUN_MASS)

    assert same.targets is reordered
    assert same.readout == declared.readout == NOUN_MASS
    assert (
        pinned.evaluate(assemble(held, constraints=(same,))).value.item()
        == pinned.evaluate(assemble(held, constraints=(declared,))).value.item()
    )


def test_the_assigned_readout_routes_a_constraint_without_target_matching() -> None:
    held = masses()
    noun_read = constraint(NOUNS, ((0, 2),), (1,), readout=NOUN_MASS)
    verb_read = constraint(NOUNS, ((0, 2),), (1,), readout=VERB_MASS)
    pinned = objective()
    noun_value = pinned.evaluate(assemble(held, constraints=(noun_read,))).value.item()
    verb_value = pinned.evaluate(assemble(held, constraints=(verb_read,))).value.item()
    span_id = SPANS.span_id(0, 2)

    assert noun_value == pytest.approx(-held[NOUN_MASS][1, span_id].item())
    assert verb_value == pytest.approx(-held[VERB_MASS][1, span_id].item())
    assert noun_value != pytest.approx(verb_value)


def test_repeating_a_declared_readout_is_refused() -> None:
    with pytest.raises(ValueError, match="more than once"):
        objective((NOUN_MASS, NOUN_MASS))


def test_the_instances_of_one_mass_are_read_in_one_gather() -> None:
    gathers: list[int] = []
    original = SpanReadout.gather

    def counting(
        readout: SpanReadout, cells: Sequence[tuple[int, int, int]]
    ) -> LogDistribution:
        gathers.append(len(cells))

        return original(readout, cells)

    def never(readout: SpanReadout, item: int, i: int, j: int) -> LogDistribution:
        del readout, item, i, j
        msg = "the objective reads cells one at a time"
        raise AssertionError(msg)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SpanReadout, "gather", counting)
        patch.setattr(SpanReadout, "at", never)
        term = objective().evaluate(bundle())

    assert term.count == 4
    assert sorted(gathers) == [1, 3]


def test_a_readout_that_is_not_a_mass_pair_is_refused() -> None:
    outputs: Outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(torch.where(LIVE, -1.0, FILL), SPAN_AXES, LIVE)
        ),
        batch=Constrained(IDS, CONSTRAINTS),
        readouts={
            NOUN_MASS: SpanReadout(
                SPANS,
                LogDistribution(torch.full((ITEMS, SPANS.num_spans), -1.0), SPAN_AXES),
            ),
            VERB_MASS: pair_of(random_log_masses(1)),
        },
    )

    with pytest.raises(ValueError, match="reads a mass pair over"):
        objective().evaluate(outputs)


def test_the_objective_declares_what_it_reads_and_what_it_pins() -> None:
    pinned = objective()

    assert pinned.name == "pinning"
    assert pinned.requires == frozenset({"batch", "readouts"})
    assert pinned.readouts == tuple(DECLARED)
    assert not hasattr(pinned, "targets")
    assert "nouns" in repr(pinned)
    assert objective(name="clamp").name == "clamp"


def test_no_mass_readout_selects_the_negative_bracket_route() -> None:
    pinned = objective(())

    assert pinned.readouts == ()
    assert pinned.requires == {"batch", "marginals"}


def test_distinct_ids_are_accepted_without_target_sets() -> None:
    declared = objective((NOUN_MASS, ReadoutId("nominals")))

    assert declared.readouts == (NOUN_MASS, ReadoutId("nominals"))


def test_an_empty_id_or_name_is_refused() -> None:
    with pytest.raises(ValueError, match="readout id"):
        objective((ReadoutId(""),))

    with pytest.raises(ValueError, match="name"):
        objective(name="")


def test_the_term_is_weighted_by_the_suite_beside_another() -> None:
    outputs = bundle()
    nouns = objective((NOUN_MASS,), name="nouns-only")
    both = objective()
    only_nouns = tuple(held for held in CONSTRAINTS if held.targets is NOUNS)
    suite = ObjectiveSuite([(0.5, nouns), (2.0, both)])

    result = suite.evaluate(assemble(masses(), constraints=only_nouns))

    assert result.total.item() == pytest.approx(
        0.5 * nouns.evaluate(assemble(masses(), constraints=only_nouns)).value.item()
        + 2.0 * both.evaluate(assemble(masses(), constraints=only_nouns)).value.item(),
        rel=1e-12,
    )
    assert "pinning/satisfied" in result.diagnostics
    assert (
        result.terms["pinning"].count
        == len(tuple(instances(only_nouns)))
        == both.evaluate(outputs).count - 1
    )


VIEW: Final = NeuralCodecView(CODEC)
INPUT_WIDTH: Final = 4


def decoder() -> RecursiveTermDecoder:
    torch.manual_seed(5)
    built = RecursiveTermDecoder(
        codec=VIEW,
        choices=CHOICES,
        admission=UNRESTRICTED_ADMISSION,
        component=ComponentId("identity"),
        settings=TreeDecoderSettings(INPUT_WIDTH, 5),
        choice_heads="flat",
    )

    return built.to(torch.float64)


def cells() -> Encoded:
    steps = torch.arange(ITEMS * SPANS.num_spans * INPUT_WIDTH, dtype=torch.float64)
    values = torch.cos(steps).reshape(ITEMS, SPANS.num_spans, INPUT_WIDTH)

    return Encoded(
        torch.where(LIVE.unsqueeze(-1), values, 7.0).detach().requires_grad_(),
        CELL_AXES,
        LIVE,
    )


def marginals(interior: float = 0.3) -> SpanMarginals:
    widths = torch.tensor([j - i for i, j in SPANS.spans_by_length()])
    built = LIVE & (widths > 1)
    constituent = torch.zeros(ITEMS, SPANS.num_spans, dtype=torch.float64)
    constituent[0, SPANS.span_id(0, 2)] = interior
    constituent[0, SPANS.span_id(1, 3)] = 1.0 - interior
    constituent[0, SPANS.span_id(0, 3)] = 1.0
    constituent[1, SPANS.span_id(0, 2)] = 1.0
    rule = torch.zeros(ITEMS, SPANS.num_spans, SPANS.max_splits, dtype=torch.float64)
    valid = torch.zeros(ITEMS, SPANS.num_spans, SPANS.max_splits, dtype=torch.bool)

    for item, length in enumerate(LENGTHS):
        for i, j in SPANS.spans_by_length(min_length=2):
            if j > length:
                continue

            for k in SPANS.splits(i, j):
                span, column = SPANS.split_id(i, j, k)
                valid[item, span, column] = True

    rule[0, SPANS.span_id(0, 2), 0] = interior
    rule[0, SPANS.span_id(1, 3), 0] = 1.0 - interior
    rule[0, SPANS.span_id(0, 3), 0] = interior
    rule[0, SPANS.span_id(0, 3), 1] = 1.0 - interior
    rule[1, SPANS.span_id(0, 2), 0] = 1.0

    return SpanMarginals(
        SPANS,
        Encoded(constituent.requires_grad_(), SPAN_AXES, built),
        Encoded(rule, SPLIT_AXES, valid),
        torch.tensor(LENGTHS),
    )


def bracket_outputs(
    interior: float,
    *,
    span: tuple[int, int] = (0, 2),
    item: int = 0,
) -> Outputs:
    """Return one batch carrying a negative bracket and parser marginals."""
    weighted = marginals(interior)
    batch = Constrained(
        IDS,
        cast(
            "tuple[Constraint, ...]",
            (BatchLocalNegativeBracket((span,), (item,), SCOPE),),
        ),
    )

    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(torch.zeros(ITEMS, SPANS.num_spans), SPAN_AXES)
        ),
        batch=batch,
        marginals=weighted,
    )


def test_a_negative_bracket_reads_the_parser_constituent_complement_exactly() -> None:
    outputs = bracket_outputs(0.3)

    term = objective(()).evaluate(outputs)

    assert term.count == 1
    assert term.value.item() == pytest.approx(-math.log1p(-0.3), rel=1e-12)
    assert term.diagnostics["satisfied"].item() == pytest.approx(0.7, rel=1e-12)


def test_a_negative_bracket_gradient_reaches_the_constituent_marginal() -> None:
    outputs = bracket_outputs(0.3)
    weighted = outputs.marginals

    assert weighted is not None
    objective(()).evaluate(outputs).value.backward()
    gradient = weighted.constituent.tensor.grad

    assert gradient is not None
    touched = SPANS.span_id(0, 2)
    assert gradient[0, touched].item() == pytest.approx(1.0 / 0.7, rel=1e-12)
    assert not bool(gradient[0, torch.arange(SPANS.num_spans) != touched].any())
    assert not bool(gradient[1].any())


def test_a_negative_bracket_at_zero_has_exact_value_and_finite_gradient() -> None:
    outputs = bracket_outputs(0.0)
    weighted = outputs.marginals

    assert weighted is not None
    term = objective(()).evaluate(outputs)
    term.value.backward()
    gradient = weighted.constituent.tensor.grad

    assert term.value.item() == 0.0
    assert gradient is not None
    assert gradient[0, SPANS.span_id(0, 2)].item() == 1.0
    assert bool(gradient.isfinite().all())


@pytest.mark.parametrize(
    ("interior", "item", "span"),
    [(1.0, 0, (0, 2)), (0.3, 0, (0, 1)), (0.3, 1, (0, 2))],
    ids=["unit-internal-mass", "token", "whole-input"],
)
def test_a_negative_bracket_on_a_certain_constituent_is_refused_by_name(
    interior: float,
    item: int,
    span: tuple[int, int],
) -> None:
    outputs = bracket_outputs(interior, span=span, item=item)
    weighted = outputs.marginals

    assert weighted is not None

    with pytest.raises(ImpossibleNegativeBracketError) as raised:
        objective(()).evaluate(outputs)

    assert raised.value.item == item
    assert raised.value.span == span
    assert weighted.constituent.tensor.grad is None


def test_a_negative_bracket_refuses_a_padded_span() -> None:
    outputs = bracket_outputs(0.3, span=(2, 3), item=1)
    weighted = outputs.marginals

    assert weighted is not None

    with pytest.raises(UncomputedCellError) as raised:
        objective(()).evaluate(outputs)

    assert raised.value.item == 1
    assert raised.value.span == (2, 3)
    assert weighted.constituent.tensor.grad is None


def test_the_bracket_route_rejects_a_target_bearing_constraint() -> None:
    with pytest.raises(ValueError, match=r"no readout.*target-bearing"):
        objective(()).evaluate(bundle())


def test_the_target_route_rejects_a_negative_bracket() -> None:
    with pytest.raises(ValueError, match="cannot score a negative bracket"):
        objective().evaluate(bracket_outputs(0.3))


def published_through(
    built: MarginalDecoder,
    *,
    strategy: ReadoutMass,
    held: Encoded,
    weighted: SpanMarginals,
    constraints: tuple[Constraint, ...] = CONSTRAINTS,
) -> Outputs:
    readout = TermsReadout(
        built,
        codec=VIEW,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=TYPES,
        targets={
            key: linearize_targets(targets, CODEC) for key, targets in DECLARED.items()
        },
        strategy=strategy,
        mapping=SOFTMAX,
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    )

    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(torch.where(LIVE, -1.0, FILL).double(), SPAN_AXES, LIVE)
        ),
        batch=Constrained(IDS, constraints),
        readouts=readout.publish(held, spans=SPANS, marginals=weighted),
    )


@pytest.mark.parametrize("strategy", list(ReadoutMass))
def test_the_strategy_is_the_readouts_and_the_gradient_reaches_the_cells(
    strategy: ReadoutMass,
) -> None:
    built = decoder()
    held = cells()
    weighted = marginals()
    outputs = published_through(built, strategy=strategy, held=held, weighted=weighted)
    pinned = objective()

    term = pinned.evaluate(outputs)
    term.value.backward()

    assert math.isfinite(term.value.item())
    assert term.value.item() > 0.0
    assert term.value.item() == pytest.approx(by_hand(outputs), rel=1e-12)
    assert term.count == 4
    assert held.tensor.grad is not None
    assert bool(held.tensor.grad.isfinite().all())
    assert bool(held.tensor.grad[0, SPANS.span_id(0, 1)].abs().sum() > 0.0)
    assert not bool(held.tensor.grad[~LIVE].any())
    assert weighted.constituent.tensor.grad is not None
    assert bool(weighted.constituent.tensor.grad[1, SPANS.span_id(0, 2)] == 0.0)
    reached = [
        parameter.grad for parameter in built.parameters() if parameter.grad is not None
    ]

    assert reached
    assert all(bool(gradient.isfinite().all()) for gradient in reached)


def test_the_two_strategies_give_one_value_on_primitive_targets() -> None:
    built = decoder()
    pinned = objective()
    values = [
        pinned.evaluate(
            published_through(
                built, strategy=strategy, held=cells(), weighted=marginals()
            )
        ).value.item()
        for strategy in ReadoutMass
    ]

    assert values[0] == pytest.approx(values[1], rel=1e-9)


class Table:
    """A decoder reading a fixed distribution over the primitives at every cell."""

    def __init__(self, table: Tensor) -> None:
        self._table = table

    def marginal_rows(self, encoded: Encoded, /, *, root_sort: str) -> LogDistribution:
        del root_sort
        at = encoded.tensor.long()
        rows = self._table[at[:, 0], at[:, 1]].unsqueeze(1)

        return LogDistribution(
            rows, ("term", "slot", "choice"), normalized_over="choice"
        )

    def rows(self, encoded: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        del terms
        return self.marginal_rows(encoded, root_sort="Cat")


def addresses() -> Encoded:
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(ITEMS), torch.arange(SPANS.num_spans), indexing="ij"
        ),
        dim=-1,
    ).to(torch.float64)

    return Encoded(grid, CELL_AXES, LIVE)


def peaked_on(name: str) -> Tensor:
    table = torch.full(
        (ITEMS, SPANS.num_spans, CODEC.n_symbols), -torch.inf, dtype=torch.float64
    )
    table[..., CODEC.encode(category(name)).symbols[0]] = 0.0

    return table


@pytest.mark.parametrize("strategy", list(ReadoutMass))
def test_a_phrase_constraint_is_not_satisfied_by_type_mass_at_an_unused_cell(
    strategy: ReadoutMass,
) -> None:
    phrase = constraint(NOUNS, ((0, 2),), (0,))
    pinned = objective()
    values = {
        interior: pinned.evaluate(
            published_through(
                Table(peaked_on("NP")),
                strategy=strategy,
                held=addresses(),
                weighted=marginals(interior),
                constraints=(phrase,),
            )
        ).value.item()
        for interior in (1e-6, 0.5, 1.0 - 1e-6)
    }

    assert values[1e-6] == pytest.approx(-math.log(1e-6), rel=1e-6)
    assert values[0.5] == pytest.approx(math.log(2.0), rel=1e-9)
    assert values[1.0 - 1e-6] == pytest.approx(1e-6, rel=1e-3)


@pytest.mark.parametrize("strategy", list(ReadoutMass))
def test_a_leaf_constraint_is_satisfied_by_type_mass_alone(
    strategy: ReadoutMass,
) -> None:
    leaf = constraint(NOUNS, ((0, 1),), (0,))
    pinned = objective()

    term = pinned.evaluate(
        published_through(
            Table(peaked_on("N")),
            strategy=strategy,
            held=addresses(),
            weighted=marginals(1e-6),
            constraints=(leaf,),
        )
    )

    assert term.value.item() == pytest.approx(0.0, abs=1e-12)


class Production(NamedTuple):
    """One binary rewrite of the oracle grammar with its weight."""

    parent: str
    left: str
    right: str
    weight: float


PRODUCTIONS: Final = (
    Production("S", "NP", "VP", 0.9),
    Production("NP", "D", "N", 0.8),
    Production("VP", "V", "NP", 0.7),
    Production("NP", "NP", "NP", 0.2),
    Production("VP", "VP", "NP", 0.3),
)
LEXICON: Final[Mapping[str, Mapping[str, float]]] = {
    "dogs": {"N": 0.6, "NP": 0.4},
    "see": {"V": 0.7, "VP": 0.3},
    "fish": {"N": 0.3, "NP": 0.3, "V": 0.2, "VP": 0.2},
    "walk": {"N": 0.4, "NP": 0.4, "V": 0.2},
}
WORDS: Final = ("dogs", "see", "fish", "walk")
GOLD: Final = (
    ((0, 4), "S"),
    ((0, 1), "NP"),
    ((1, 4), "VP"),
    ((1, 2), "V"),
    ((2, 4), "NP"),
    ((2, 3), "NP"),
    ((3, 4), "NP"),
)


def forest(words: tuple[str, ...]) -> PackedForest[Term, float]:
    index = SpanIndex(len(words))
    built = PackedForest[Term, float](index, LogInside())

    for position, word in enumerate(words):
        for reading, weight in LEXICON[word].items():
            built.add(
                ItemKey((position, position + 1), category(reading), "lex"),
                Edge(word, (), category(reading), math.log(weight)),
            )

    for i, j in index.spans_by_length(min_length=2):
        for k in index.splits(i, j):
            for left in built.items_at((i, k)):
                for right in built.items_at((k, j)):
                    for rule in PRODUCTIONS:
                        if left.reading is category(
                            rule.left
                        ) and right.reading is category(rule.right):
                            built.add(
                                ItemKey((i, j), category(rule.parent), "step"),
                                Edge(
                                    "-".join(rule[:3]),
                                    (left, right),
                                    category(rule.parent),
                                    math.log(rule.weight),
                                    split=k,
                                ),
                            )

    return built


def concentrated_forest() -> PackedForest[Term, float]:
    """Return the gold derivation as a one-derivation packed forest."""
    built = PackedForest[Term, float](SpanIndex(len(WORDS)), LogInside())
    leaves = (
        ((0, 1), "NP", "dogs"),
        ((1, 2), "V", "see"),
        ((2, 3), "NP", "fish"),
        ((3, 4), "NP", "walk"),
    )

    for span, name, word in leaves:
        built.add(
            ItemKey(span, category(name), "lex"),
            Edge(word, (), category(name), 0.0),
        )

    steps = (
        ((2, 4), "NP", ((2, 3), "NP", "lex"), ((3, 4), "NP", "lex"), 3),
        ((1, 4), "VP", ((1, 2), "V", "lex"), ((2, 4), "NP", "step"), 2),
        ((0, 4), "S", ((0, 1), "NP", "lex"), ((1, 4), "VP", "step"), 1),
    )

    for span, parent, left, right, split in steps:
        children = tuple(
            ItemKey(child_span, category(child_name), kind)
            for child_span, child_name, kind in (left, right)
        )
        built.add(
            ItemKey(span, category(parent), "step"),
            Edge(parent, children, category(parent), 0.0, split=split),
        )

    return built


def ambiguous_forest() -> PackedForest[Term, float]:
    """Return two disjoint derivations packed under one sentence item."""
    built = concentrated_forest()
    alternatives = (
        ((0, 1), "D", "dogs"),
        ((1, 2), "N", "see"),
        ((2, 3), "V", "fish"),
        ((3, 4), "D", "walk"),
    )

    for span, name, word in alternatives:
        built.add(
            ItemKey(span, category(name), "alternative"),
            Edge(word, (), category(name), 0.0),
        )

    left = ItemKey((0, 2), category("NP"), "alternative")
    right = ItemKey((2, 4), category("VP"), "alternative")
    built.add(
        left,
        Edge(
            "alternative-np",
            (
                ItemKey((0, 1), category("D"), "alternative"),
                ItemKey((1, 2), category("N"), "alternative"),
            ),
            category("NP"),
            0.0,
            split=1,
        ),
    )
    built.add(
        right,
        Edge(
            "alternative-vp",
            (
                ItemKey((2, 3), category("V"), "alternative"),
                ItemKey((3, 4), category("D"), "alternative"),
            ),
            category("VP"),
            0.0,
            split=3,
        ),
    )
    built.add(
        ItemKey((0, 4), category("S"), "step"),
        Edge("alternative-s", (left, right), category("S"), 0.0, split=2),
    )

    return built


def test_pinning_every_gold_constituent_is_the_span_marginal_supervised_loss() -> None:
    chart = ambiguous_forest()
    outside = outside_from_edges(chart, RootGoal(SENTENCES))
    index = chart.index
    singletons = {
        op.name: TargetSet((category(op.name),), MATCHING) for op in SIGNATURE.operators
    }
    declared = {ReadoutId(f"gold-{name}"): held for name, held in singletons.items()}
    live = torch.ones(1, index.num_spans, dtype=torch.bool)
    readouts = {}

    for key, held in declared.items():
        log_masses = torch.full((1, index.num_spans), -torch.inf, dtype=torch.float64)

        for i, j in index.spans_by_length():
            mass = target_mass(chart, outside, held, (i, j))
            log_masses[0, index.span_id(i, j)] = (
                min(math.log(mass), 0.0) if mass > 0.0 else -math.inf
            )

        rest = torch.log(-torch.expm1(log_masses))
        readouts[key] = SpanReadout(
            index,
            LogDistribution(
                torch.stack((log_masses, rest), dim=-1),
                MASS_AXES,
                live,
                normalized_over="outcome",
            ),
        )

    constraints = tuple(
        constraint(
            singletons[name],
            (span,),
            (0,),
            readout=ReadoutId(f"gold-{name}"),
        )
        for span, name in GOLD
    )
    outputs: Outputs = ObjectiveOutputs(
        spans=index,
        chart=CellChart(LogDistribution(torch.zeros(1, index.num_spans), SPAN_AXES)),
        batch=Constrained((" ".join(WORDS),), constraints),
        readouts=readouts,
    )
    gold_marginals = [
        outside.constituent_marginal(
            ItemKey(span, category(name), "lex" if span[1] - span[0] == 1 else "step")
        )
        for span, name in GOLD
    ]

    term = objective(tuple(declared)).evaluate(outputs)

    assert term.count == len(GOLD) == 2 * len(WORDS) - 1
    assert all(0.0 < marginal <= 1.0 for marginal in gold_marginals)
    assert sum(marginal < 1.0 for marginal in gold_marginals) >= 2
    assert gold_marginals[0] == pytest.approx(1.0)
    marginal_nll = -sum(math.log(marginal) for marginal in gold_marginals)
    joint_nll = outside.partition

    assert term.value.item() * term.count == pytest.approx(marginal_nll, rel=1e-9)
    assert marginal_nll != pytest.approx(joint_nll, abs=1e-4)
    assert term.value.item() > 0.0


def test_constituent_marginal_pinning_matches_joint_likelihood_when_concentrated() -> (
    None
):
    chart = concentrated_forest()
    outside = outside_from_edges(chart, RootGoal(SENTENCES))
    index = chart.index
    singletons = {
        name: TargetSet((category(name),), MATCHING)
        for name in {category_name for _, category_name in GOLD}
    }
    declared = {ReadoutId(f"gold-{name}"): held for name, held in singletons.items()}
    live = torch.ones(1, index.num_spans, dtype=torch.bool)
    readouts = {}

    for name, held in singletons.items():
        log_masses = torch.full((1, index.num_spans), -torch.inf, dtype=torch.float64)

        for i, j in index.spans_by_length():
            mass = target_mass(chart, outside, held, (i, j))
            log_masses[0, index.span_id(i, j)] = (
                min(math.log(mass), 0.0) if mass > 0.0 else -math.inf
            )

        readouts[ReadoutId(f"gold-{name}")] = SpanReadout(
            index,
            LogDistribution(
                torch.stack((log_masses, torch.log(-torch.expm1(log_masses))), dim=-1),
                MASS_AXES,
                live,
                normalized_over="outcome",
            ),
        )

    constraints = tuple(
        constraint(
            singletons[name],
            (span,),
            (0,),
            readout=ReadoutId(f"gold-{name}"),
        )
        for span, name in GOLD
    )
    outputs: Outputs = ObjectiveOutputs(
        spans=index,
        chart=CellChart(LogDistribution(torch.zeros(1, index.num_spans), SPAN_AXES)),
        batch=Constrained((" ".join(WORDS),), constraints),
        readouts=readouts,
    )

    term = objective(tuple(declared)).evaluate(outputs)
    joint_nll = outside.partition

    assert term.count == len(GOLD)
    assert term.value.item() * term.count == pytest.approx(joint_nll, abs=1e-4)


FORMS: Final = FORM_LANGUAGE.store
FORM_CODEC: Final = term_codec(
    FORM_LANGUAGE,
    CompleteTree(FORMS.signature, FORMS.signature.operators, order="df", depth=3),
)
FORM_VIEW: Final = NeuralCodecView(FORM_CODEC)
FORM_CHOICES: Final = ChoiceIndex(
    FORMS.signature,
    depth_bounds=dict.fromkeys(FORMS.signature.sorts, FORM_CODEC.capacity),
)
FORM_MATCHING: Final[Matching[Term]] = Identity()
FORM_MASS: Final = ReadoutId("likes-bo")
FORM_TYPES: Final = ReadoutId("forms")
LIKES_BO: Final = FORM_LANGUAGE.apply(
    FORM_LANGUAGE.constant("likes"), FORM_LANGUAGE.constant("bo")
)
DENIED: Final = FORM_LANGUAGE.apply(FORM_LANGUAGE.constant("not"), LIKES_BO)
PEAK: Final = 0.9
PINNED_SPANS: Final = (("first", (0, 1)), ("first", (0, 2)), ("second", (0, 2)))
INTERIOR: Final = 0.3


class FormTable:
    """A decoder reading a fixed distribution over every slot at every cell."""

    def __init__(self, table: Tensor) -> None:
        self._table = table

    def marginal_rows(self, encoded: Encoded, /, *, root_sort: str) -> LogDistribution:
        del root_sort
        at = encoded.tensor.long()
        rows = self._table[at[:, 0], at[:, 1]]

        return LogDistribution(
            rows,
            ("term", "slot", "choice"),
            normalized_over="choice",
        )

    def rows(self, encoded: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        del terms
        return self.marginal_rows(encoded, root_sort=FORM_LANGUAGE.term_sort)


def peaked_form(form: Term, peak: float = PEAK) -> Tensor:
    encoded = FORM_CODEC.encode(form)
    codes = FORM_CHOICES.codes(encoded)
    spread = math.log((1.0 - peak) / (FORM_CHOICES.width - 1))
    table = torch.full(
        (ITEMS, SPANS.num_spans, FORM_CODEC.capacity, FORM_CHOICES.width),
        spread,
        dtype=torch.float64,
    )

    for slot, choice in enumerate(codes):
        table[:, :, slot, choice] = math.log(peak)

    return table


def random_forms(seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.rand(
        ITEMS,
        SPANS.num_spans,
        FORM_CODEC.capacity,
        FORM_CHOICES.width,
        generator=generator,
        dtype=torch.float64,
    )

    return logits.log_softmax(dim=-1)


def form_constraint(
    form: Term = LIKES_BO, *, negate: bool = False
) -> LogicalFormConstraint[str]:
    return LogicalFormConstraint(
        form,
        tuple(ConstraintOccurrence(ItemId(item), span) for item, span in PINNED_SPANS),
        SCOPE,
        matching=FORM_MATCHING,
        negate=negate,
    )


def pinned_forms(
    form: Term = LIKES_BO, *, negate: bool = False
) -> tuple[Constraint, ...]:
    return form_constraint(form, negate=negate).to_spans(
        tuple(ItemId(item) for item in IDS), readout=FORM_MASS
    )


def form_readout(table: Tensor, *, strategy: ReadoutMass) -> TermsReadout:
    return TermsReadout(
        FormTable(table),
        codec=FORM_VIEW,
        index=FORM_CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=FORM_TYPES,
        targets={FORM_MASS: linearize_targets(form_constraint().targets, FORM_CODEC)},
        strategy=strategy,
        mapping=SOFTMAX,
        store=None,
        root_sort=FORM_LANGUAGE.term_sort,
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    )


def published_forms(
    table: Tensor,
    *,
    strategy: ReadoutMass = ReadoutMass.FACTORIZED,
    constraints: tuple[Constraint, ...],
) -> Outputs:
    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(torch.where(LIVE, -1.0, FILL).double(), SPAN_AXES, LIVE)
        ),
        batch=Constrained(IDS, constraints),
        readouts=form_readout(table, strategy=strategy).publish(
            addresses(), spans=SPANS, marginals=marginals(INTERIOR)
        ),
    )


def form_log_masses(table: Tensor) -> Tensor:
    published = form_readout(table, strategy=ReadoutMass.FACTORIZED).publish(
        addresses(), spans=SPANS, marginals=marginals(INTERIOR)
    )

    return log_mass(published[FORM_MASS].distribution).detach()


def form_mass_by_hand(
    clamped: Term,
    peaked: Term,
    item: str,
    span: tuple[int, int],
    strategy: ReadoutMass,
) -> float:
    wanted = FORM_CODEC.encode(clamped)
    table = FORM_CODEC.encode(peaked)
    wanted_choices = FORM_CHOICES.codes(wanted)
    table_choices = FORM_CHOICES.codes(table)
    spread = (1.0 - PEAK) / (FORM_CHOICES.width - 1)

    def choice_symbol(choice: int) -> int:
        if choice == FORM_CHOICES.PAD:
            return FORM_CODEC.index.PAD

        return FORM_CODEC.index.index(FORM_CHOICES.operator(choice))

    def slot_mass(slot: int, symbol: int) -> float:
        peaked_choice = table_choices[slot]

        if strategy is ReadoutMass.COUPLED:
            return PEAK if wanted_choices[slot] == peaked_choice else spread

        choices = tuple(
            choice
            for choice in range(FORM_CHOICES.width)
            if choice_symbol(choice) == symbol
        )
        return sum(PEAK if choice == peaked_choice else spread for choice in choices)

    mass = math.prod(
        slot_mass(slot, symbol)
        for slot, (symbol, occupied) in enumerate(
            zip(wanted.symbols, wanted.occupied, strict=True)
        )
        if occupied
    )
    start, end = span
    interior = end - start > 1 and end < LENGTHS[IDS.index(item)]

    return mass * (INTERIOR if interior else 1.0)


def form_objective() -> SpanConstraintObjective[Term, str]:
    return objective((FORM_MASS,), name="meaning")


def test_a_pinned_form_localizes_through_the_shared_constraint_runtime() -> None:
    localized = pinned_forms()

    assert all(type(local) is BatchLocalConstraint for local in localized)
    assert sorted(
        (item, span)
        for local in localized
        for item in local.item_positions
        for span in local.spans
    ) == [(0, (0, 1)), (0, (0, 2)), (1, (0, 2))]
    assert all(local.targets.categories == (LIKES_BO,) for local in localized)
    assert all(local.readout == FORM_MASS for local in localized)


@pytest.mark.parametrize("strategy", list(ReadoutMass))
def test_a_readout_matching_the_clamped_form_scores_strictly_below_one_that_does_not(
    strategy: ReadoutMass,
) -> None:
    pinned = form_objective()
    localized = pinned_forms()

    matched = pinned.evaluate(
        published_forms(peaked_form(LIKES_BO), strategy=strategy, constraints=localized)
    )
    elsewhere = pinned.evaluate(
        published_forms(peaked_form(DENIED), strategy=strategy, constraints=localized)
    )

    expected = {
        peaked: -sum(
            math.log(form_mass_by_hand(LIKES_BO, peaked, item, span, strategy))
            for item, span in PINNED_SPANS
        )
        / len(PINNED_SPANS)
        for peaked in (LIKES_BO, DENIED)
    }

    assert matched.count == elsewhere.count == len(PINNED_SPANS)
    assert matched.value.item() < elsewhere.value.item()
    assert matched.value.item() == pytest.approx(expected[LIKES_BO], rel=1e-9)
    assert elsewhere.value.item() == pytest.approx(expected[DENIED], rel=1e-9)
    assert expected[LIKES_BO] < expected[DENIED]


@pytest.mark.parametrize("strategy", list(ReadoutMass))
def test_the_negated_arm_reverses_the_ordering_over_the_clamped_form(
    strategy: ReadoutMass,
) -> None:
    pinned = form_objective()
    forbidden = pinned_forms(negate=True)

    matched = pinned.evaluate(
        published_forms(peaked_form(LIKES_BO), strategy=strategy, constraints=forbidden)
    )
    elsewhere = pinned.evaluate(
        published_forms(peaked_form(DENIED), strategy=strategy, constraints=forbidden)
    )

    expected = {
        peaked: -sum(
            math.log1p(-form_mass_by_hand(LIKES_BO, peaked, item, span, strategy))
            for item, span in PINNED_SPANS
        )
        / len(PINNED_SPANS)
        for peaked in (LIKES_BO, DENIED)
    }

    assert all(local.negate for local in forbidden)
    assert matched.value.item() > elsewhere.value.item()
    assert matched.value.item() == pytest.approx(expected[LIKES_BO], rel=1e-9)
    assert elsewhere.value.item() == pytest.approx(expected[DENIED], rel=1e-9)


def test_payload_aware_coupling_differs_from_symbol_factorization() -> None:
    pinned = form_objective()
    localized = pinned_forms()
    table = random_forms(3)

    values = [
        pinned.evaluate(
            published_forms(table, strategy=strategy, constraints=localized)
        ).value.item()
        for strategy in ReadoutMass
    ]

    assert values[0] != pytest.approx(values[1], rel=1e-9)
    assert values[0] > 0.0


def test_a_pinned_form_is_scored_by_the_type_clamps_arithmetic() -> None:
    localized = pinned_forms()
    outputs = published_forms(random_forms(4), constraints=localized)
    satisfied = [
        outputs.readouts[FORM_MASS].at(item, start, end).tensor[0].exp().item()
        for local in localized
        for item in local.item_positions
        for start, end in local.spans
    ]

    term = form_objective().evaluate(outputs)

    assert term.count == len(satisfied) == len(PINNED_SPANS)
    assert term.value.item() == pytest.approx(by_hand(outputs), rel=1e-12)
    assert term.diagnostics["satisfied"].item() == pytest.approx(
        sum(satisfied) / len(satisfied), rel=1e-9
    )


def form_bundle(seed: int = 5) -> Outputs:
    return assemble(
        {FORM_MASS: form_log_masses(random_forms(seed))}, constraints=pinned_forms()
    )


def form_fixture() -> ObjectiveFixture[Constrained]:
    return ObjectiveFixture(
        objective=form_objective(),
        outputs=form_bundle,
        inputs=lambda outputs: (outputs.readouts[FORM_MASS].distribution.tensor,),
        unweighted=by_hand,
        monotone=lambda: (
            assemble(
                {FORM_MASS: form_log_masses(peaked_form(DENIED))},
                constraints=pinned_forms(),
            ),
            assemble(
                {FORM_MASS: form_log_masses(peaked_form(LIKES_BO))},
                constraints=pinned_forms(),
            ),
        ),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        inapplicable=lambda: assemble(
            {FORM_MASS: form_log_masses(random_forms(5))}, constraints=()
        ),
        tolerance=1e-9,
    )


class TestLogicalFormClampLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Constrained]:
        return form_fixture()

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that a readout matching the clamped form scores below a mismatch."""
        the_value_is_monotone_where_the_fixture_says(conformance)


class DistributionDecoder:
    """A decoder declaring the capability a term readout requires."""

    def marginal_rows(self, inputs: Encoded, /, *, root_sort: str) -> LogDistribution:
        raise AssertionError(inputs, root_sort)


PINNED_TARGET: Final = LinearizedTargets(
    BatchEncoding(
        (
            Encoding(
                symbols=(1,),
                parents=(Encoding.NO_PARENT,),
                depths=(0,),
                occupied=(True,),
            ),
        )
    ),
    0,
    "identity",
    SENTENCES,
)


def pinned_readout(*targets: str) -> TermsReadout:
    return TermsReadout(
        DistributionDecoder(),
        codec=VIEW,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=ReadoutId("distribution"),
        targets={ReadoutId(target): PINNED_TARGET for target in targets},
        strategy=ReadoutMass.FACTORIZED,
        mapping=SOFTMAX,
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    )


class Bindings:
    """One term readout installed under the role the declaration names."""

    def __init__(self, readout: TermsReadout, *, role: str = "types") -> None:
        self.readout = readout
        self.role = role
        self.requests: list[str] = []

    @property
    def name(self) -> str:
        return "pinned-types"

    def terms_readout(self, role: str, /) -> TermsReadout:
        self.requests.append(role)

        if role != self.role:
            msg = f"no term readout is bound as {role!r}"
            raise ValueError(msg)

        return self.readout

    def space(self, role: str, /) -> Space:
        msg = f"pinning must not resolve represented space {role!r}"
        raise AssertionError(msg)

    def readout_space(self, role: str, /) -> Space:
        msg = f"pinning must not resolve a readout space for {role!r}"
        raise AssertionError(msg)

    def relation(self, role: str, /) -> SpaceRelationBinding:
        msg = f"pinning must not resolve space relation {role!r}"
        raise AssertionError(msg)

    def span_decoder(self, role: str, /) -> SpanTokenDecoder:
        msg = f"pinning must not resolve span decoder {role!r}"
        raise AssertionError(msg)

    def proof_reward(self) -> ProofRewardBindings:
        msg = "pinning must not resolve proof-reward bindings"
        raise AssertionError(msg)


PINNING: Final = PinningLossSpec(readout="constraints")


class TestPinningPlugin(ConfiguredComponentConformance):
    @pytest.fixture
    def configured_component_fixture(self) -> ConfiguredComponentFixture:
        readout = pinned_readout("root", "argument")
        context = Bindings(readout, role="constraints")

        def observe(built: object) -> dict[str, bool]:
            held = cast("SpanConstraintObjective[Term, str]", built)

            return {
                "named for its term": held.name == "pinned-types",
                "reads the declared readout": context.requests[-1] == "constraints",
                "pins to every target set the readout holds": held.readouts
                == (ReadoutId("root"), ReadoutId("argument")),
            }

        return ConfiguredComponentFixture(
            declaration=cast("ConfiguredSpec[object]", PINNING),
            context=context,
            expected_type=SpanConstraintObjective,
            observe=observe,
            unchanged=lambda: context.readout is readout,
        )


def test_the_plugin_builds_bracket_pinning_without_resolving_a_readout() -> None:
    context = Bindings(pinned_readout("unused"), role="must-not-be-read")

    built = pinning_module.PLUGIN.construct(PinningLossSpec(), context)

    assert built.name == "pinned-types"
    assert built.readouts == ()
    assert built.requires == {"batch", "marginals"}
    assert context.requests == []


def test_the_plugin_refuses_a_readout_without_any_mass_target() -> None:
    with pytest.raises(ValueError, match="reads at least one mass readout"):
        pinning_module.PLUGIN.construct(
            PINNING, Bindings(pinned_readout(), role="constraints")
        )
