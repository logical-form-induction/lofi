import math
from collections.abc import Mapping
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

import lofi.nn.loss._grammaticality as grammaticality_module
from lofi._types import ReadoutId
from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.codec import NeuralCodecView
from lofi.nn.decoders import RecursiveTermDecoder, TreeDecoderSettings
from lofi.nn.loss._constraint import BatchLocalConstraint
from lofi.nn.loss._errors import (
    ImproperMassError,
    UncomputedCellError,
)
from lofi.nn.loss._grammaticality import GrammaticalityObjective
from lofi.nn.loss._linearize import linearize_targets
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
from lofi.spec.losses.grammaticality import GrammaticalityLossSpec
from lofi.spec.readouts import DecodedMeaningSettings, ReadoutSource
from lofi.spec.readouts.terms import ReadoutMass
from lofi.terms import (
    ChoiceIndex,
    CompleteTree,
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
    from lofi.nn.spaces import Space, SpaceRelationBinding
    from lofi.spec import ConfiguredSpec
    from lofi.terms import Matching


SIGNATURE: Final = Signature(
    "grammaticality-fixture",
    ("Cat",),
    tuple(Operator(name, "Cat", family="primitive") for name in ("S", "NP", "VP")),
)
INDEX: Final = OperatorIndex(SIGNATURE)
CHOICES: Final = ChoiceIndex(SIGNATURE, depth_bounds={"Cat": 0})
CODEC: Final = TermCodec(
    INDEX, CompleteTree(SIGNATURE, SIGNATURE.operators, order="df", depth=1)
)
STORE: Final = TermStore(SIGNATURE)
MATCHING: Final[Matching[Term]] = Identity()


def category(name: str) -> Term:
    return STORE.op(SIGNATURE.by_name(name))


SENTENCES: Final = TargetSet((category("S"),), MATCHING)
ROOT_MASS: Final = ReadoutId("sentences")
TYPES: Final = ReadoutId("types")

LENGTH: Final = 3
ITEMS: Final = 3
LENGTHS: Final = (3, 2, 3)
LABELS: Final[tuple[bool | None, ...]] = (True, False, True)
SPANS: Final = SpanIndex(LENGTH)
FILL: Final = -9.0
IDS: Final = ("first", "second", "third")


class Labeled(NamedTuple):
    """A batch of items each labeled for grammaticality, or left unlabeled."""

    ids: tuple[str, ...]
    lengths: tuple[int, ...]
    labels: tuple[bool | None, ...]


type Outputs = ObjectiveOutputs[Labeled]


def live_spans(lengths: tuple[int, ...] = LENGTHS) -> Tensor:
    mask = torch.zeros(len(lengths), SPANS.num_spans, dtype=torch.bool)

    for item, length in enumerate(lengths):
        for start, end in SPANS.spans_by_length():
            if end <= length:
                mask[item, SPANS.span_id(start, end)] = True

    return mask


LIVE: Final = live_spans()


def roots(lengths: tuple[int, ...] = LENGTHS) -> list[int]:
    return [SPANS.span_id(0, length) for length in lengths]


def objective(
    readout: ReadoutId = ROOT_MASS,
    *,
    noise: float = 0.0,
    name: str = "grammaticality",
) -> GrammaticalityObjective:
    return GrammaticalityObjective(readout=readout, noise=noise, name=name)


class NoBindings:
    """A binding context whose every lookup fails, for an objective reading none."""

    @property
    def name(self) -> str:
        return "judged-sentences"

    def terms_readout(self, role: str, /) -> TermsReadout:
        msg = f"grammaticality must not resolve term readout {role!r}"
        raise AssertionError(msg)

    def space(self, role: str, /) -> Space:
        msg = f"grammaticality must not resolve represented space {role!r}"
        raise AssertionError(msg)

    def readout_space(self, role: str, /) -> Space:
        msg = f"grammaticality must not resolve a readout space for {role!r}"
        raise AssertionError(msg)

    def relation(self, role: str, /) -> SpaceRelationBinding:
        msg = f"grammaticality must not resolve space relation {role!r}"
        raise AssertionError(msg)

    def span_decoder(self, role: str, /) -> SpanTokenDecoder:
        msg = f"grammaticality must not resolve span decoder {role!r}"
        raise AssertionError(msg)

    def proof_reward(self) -> ProofRewardBindings:
        msg = "grammaticality must not resolve proof-reward bindings"
        raise AssertionError(msg)


DECLARATION: Final = GrammaticalityLossSpec(readout="sentences", noise=0.2)


def installed() -> GrammaticalityObjective:
    return grammaticality_module.PLUGIN.construct(DECLARATION, NoBindings())


class TestGrammaticalityPlugin(ConfiguredComponentConformance):
    @pytest.fixture
    def configured_component_fixture(self) -> ConfiguredComponentFixture:
        def observe(built: object) -> dict[str, bool]:
            held = cast("GrammaticalityObjective", built)

            return {
                "named for its term": held.name == "judged-sentences",
                "reads the declared publication": held.readout == ROOT_MASS,
                "holds the declared noise": held.noise == 0.2,
            }

        return ConfiguredComponentFixture(
            declaration=cast("ConfiguredSpec[object]", DECLARATION),
            context=NoBindings(),
            expected_type=GrammaticalityObjective,
            observe=observe,
            unchanged=lambda: True,
        )


def test_the_plugin_builds_the_objective_a_direct_construction_builds() -> None:
    built = installed()
    direct = objective(noise=0.2, name="judged-sentences")
    outputs = bundle()
    actual = built.evaluate(outputs)
    expected = direct.evaluate(outputs)

    assert built.name == direct.name == "judged-sentences"
    assert built.readout == direct.readout == ROOT_MASS
    assert built.noise == direct.noise == 0.2
    assert built.requires == direct.requires
    assert actual.count == expected.count
    torch.testing.assert_close(actual.value, expected.value)
    assert actual.diagnostics.keys() == expected.diagnostics.keys()

    for name in actual.diagnostics:
        torch.testing.assert_close(actual.diagnostics[name], expected.diagnostics[name])


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
    log_masses: Tensor,
    *,
    lengths: tuple[int, ...] = LENGTHS,
    labels: tuple[bool | None, ...] = LABELS,
    computed: Tensor = LIVE,
    fill: float = FILL,
    ids: tuple[str, ...] = IDS,
    key: ReadoutId = ROOT_MASS,
) -> Outputs:
    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(torch.where(computed, -1.0, fill), SPAN_AXES, computed)
        ),
        batch=Labeled(ids, lengths, labels),
        readouts={key: pair_of(log_masses, computed=computed, fill=fill)},
    )


def bundle(seed: int = 0) -> Outputs:
    return assemble(random_log_masses(seed))


def log_masses_of(outputs: Outputs) -> Tensor:
    return log_mass(outputs.readouts[ROOT_MASS].distribution)


def computed_of(outputs: Outputs) -> Tensor:
    return outputs.readouts[ROOT_MASS].computed


def permuted(outputs: Outputs) -> Outputs:
    order = torch.tensor([2, 0, 1])
    batch = outputs.batch

    return assemble(
        log_masses_of(outputs)[order],
        lengths=tuple(batch.lengths[position] for position in order.tolist()),
        labels=tuple(batch.labels[position] for position in order.tolist()),
        computed=computed_of(outputs)[order],
        ids=tuple(batch.ids[position] for position in order.tolist()),
    )


def refilled(outputs: Outputs) -> Outputs:
    return assemble(
        log_masses_of(outputs),
        lengths=outputs.batch.lengths,
        labels=outputs.batch.labels,
        computed=computed_of(outputs),
        fill=FILL / 3.0,
        ids=outputs.batch.ids,
    )


def unlabeled() -> Outputs:
    return assemble(random_log_masses(0), labels=(None,) * ITEMS)


def inputs_of(outputs: Outputs) -> tuple[Tensor, ...]:
    return (outputs.readouts[ROOT_MASS].distribution.tensor,)


def scores_of(outputs: Outputs) -> list[float]:
    read = log_masses_of(outputs).detach()

    return [
        math.exp(read[item, span].item())
        for item, span in enumerate(roots(outputs.batch.lengths))
    ]


def by_hand(outputs: Outputs, *, noise: float = 0.0) -> float:
    read: list[float] = []

    for score, label in zip(scores_of(outputs), outputs.batch.labels, strict=True):
        if label is None:
            continue

        agreeing = score if label else 1.0 - score
        read.append(-math.log((1.0 - noise) * agreeing + noise * (1.0 - agreeing)))

    return sum(read) / len(read) if read else 0.0


def with_score(
    score: float, *, elsewhere: float = 0.5, lengths: tuple[int, ...] = LENGTHS
) -> Tensor:
    held = torch.full((len(lengths), SPANS.num_spans), math.log(elsewhere)).double()

    for item, span in enumerate(roots(lengths)):
        held[item, span] = math.log(score)

    return held


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


def fixture() -> ObjectiveFixture[Labeled]:
    grammatical: tuple[bool | None, ...] = (True,) * ITEMS

    return ObjectiveFixture(
        objective=objective(),
        outputs=bundle,
        inputs=inputs_of,
        unweighted=by_hand,
        monotone=lambda: (
            assemble(with_score(0.2), labels=grammatical),
            assemble(with_score(0.9), labels=grammatical),
        ),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        inapplicable=unlabeled,
        tolerance=1e-9,
    )


class TestGrammaticalityObjectiveLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Labeled]:
        return fixture()

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that less root mass on grammatical items scores above more."""
        the_value_is_monotone_where_the_fixture_says(conformance)


def test_the_value_is_the_binary_cross_entropy_against_the_labels() -> None:
    outputs = bundle()

    term = objective().evaluate(outputs)

    assert term.value.item() == pytest.approx(by_hand(outputs), rel=1e-12)
    assert term.count == ITEMS
    assert term.value.item() > 0.0


def test_the_count_is_the_number_of_labeled_items() -> None:
    term = objective().evaluate(
        assemble(random_log_masses(0), labels=(True, None, False))
    )

    assert term.count == 2


type Constraint = BatchLocalConstraint[Term, str]


class MatchedConstraints(NamedTuple):
    """A labeled batch that also lists matching root constraints."""

    ids: tuple[str, ...]
    lengths: tuple[int, ...]
    labels: tuple[bool | None, ...]
    constraints: tuple[Constraint, ...]


def root_constraints(
    lengths: tuple[int, ...], labels: tuple[bool | None, ...]
) -> tuple[Constraint, ...]:
    return tuple(
        BatchLocalConstraint(
            SENTENCES,
            ((0, length),),
            (item,),
            "top",
            readout=ROOT_MASS,
            negate=not label,
        )
        for item, (length, label) in enumerate(zip(lengths, labels, strict=True))
        if label is not None
    )


@pytest.mark.parametrize(
    ("seed", "labels"),
    [
        (0, (True, False, True)),
        (1, (False, True, False)),
        (2, (True, True, False)),
        (3, (False, False, True)),
    ],
)
def test_the_value_equals_the_matching_root_constraint_objective(
    seed: int, labels: tuple[bool | None, ...]
) -> None:
    log_masses = random_log_masses(seed)
    readouts = {ROOT_MASS: pair_of(log_masses)}
    outputs: ObjectiveOutputs[MatchedConstraints] = ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(torch.where(LIVE, -1.0, FILL), SPAN_AXES, LIVE)
        ),
        batch=MatchedConstraints(
            IDS, LENGTHS, labels, root_constraints(LENGTHS, labels)
        ),
        readouts=readouts,
    )
    constraint_objective = SpanConstraintObjective[Term, str](
        readouts=(ROOT_MASS,), name="pinning"
    )

    constrained = constraint_objective.evaluate(outputs)
    proper = objective().evaluate(outputs)

    assert any(labels)
    assert not all(labels)
    assert constrained.count == proper.count == ITEMS
    assert abs(constrained.value.item() - proper.value.item()) <= 1e-6
    assert proper.value.item() == pytest.approx(constrained.value.item(), rel=1e-12)


def test_raising_the_root_mass_on_grammatical_items_strictly_lowers_the_value() -> None:
    grammatical: tuple[bool | None, ...] = (True,) * ITEMS
    scored = objective()
    values = [
        scored.evaluate(assemble(with_score(score), labels=grammatical)).value.item()
        for score in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
    ]

    assert all(later < earlier for earlier, later in pairwise(values))
    assert values[0] == pytest.approx(-math.log(0.1), rel=1e-12)


def test_raising_the_root_mass_on_ungrammatical_items_strictly_raises_the_value() -> (
    None
):
    ungrammatical: tuple[bool | None, ...] = (False,) * ITEMS
    scored = objective()
    values = [
        scored.evaluate(assemble(with_score(score), labels=ungrammatical)).value.item()
        for score in (0.1, 0.5, 0.9)
    ]

    assert all(later > earlier for earlier, later in pairwise(values))
    assert values[-1] == pytest.approx(-math.log(0.1), rel=1e-12)


@pytest.mark.parametrize("noise", [0.05, 0.2, 0.45])
def test_the_label_noise_arm_mixes_the_flipped_label_at_the_declared_rate(
    noise: float,
) -> None:
    outputs = bundle()

    noisy = objective(noise=noise).evaluate(outputs)
    plain = objective().evaluate(outputs)

    assert noisy.value.item() == pytest.approx(by_hand(outputs, noise=noise), rel=1e-12)
    assert noisy.value.item() != pytest.approx(plain.value.item())
    assert noisy.count == plain.count


def test_a_noise_rate_of_zero_is_the_plain_cross_entropy() -> None:
    outputs = bundle()

    assert objective(noise=0.0).evaluate(outputs).value.item() == pytest.approx(
        objective().evaluate(outputs).value.item(), rel=1e-12
    )


@pytest.mark.parametrize("noise", [0.1, 0.3, 0.49])
def test_the_noisy_value_still_falls_as_grammatical_root_mass_rises(
    noise: float,
) -> None:
    grammatical: tuple[bool | None, ...] = (True,) * ITEMS
    scored = objective(noise=noise)
    values = [
        scored.evaluate(assemble(with_score(score), labels=grammatical)).value.item()
        for score in (0.1, 0.5, 0.9, 0.999)
    ]

    assert all(later < earlier for earlier, later in pairwise(values))
    assert values[-1] > 0.0
    assert values[-1] == pytest.approx(
        -math.log((1.0 - noise) * 0.999 + noise * 0.001), rel=1e-12
    )


def test_a_noisy_term_is_bounded_where_the_plain_term_is_not() -> None:
    grammatical: tuple[bool | None, ...] = (True,) * ITEMS
    tiny = 1e-300
    outputs = assemble(with_score(tiny), labels=grammatical)

    noisy = objective(noise=0.1).evaluate(outputs).value.item()
    plain = objective().evaluate(outputs).value.item()

    assert noisy == pytest.approx(-math.log(0.1), rel=1e-9)
    assert plain > 600.0


@pytest.mark.parametrize("noise", [-0.1, 0.5, 0.75, 1.0, math.nan, math.inf])
def test_a_noise_rate_the_label_cannot_survive_is_refused(noise: float) -> None:
    with pytest.raises(ValueError, match="label-noise rate"):
        objective(noise=noise)


def test_a_noise_rate_that_is_not_a_number_is_refused() -> None:
    with pytest.raises(TypeError, match="label-noise rate"):
        objective(noise=True)


def test_the_dynamic_range_of_the_score_is_a_diagnostic_over_the_whole_batch() -> None:
    outputs = assemble(random_log_masses(3), labels=(True, None, False))
    scores = scores_of(outputs)

    term = objective().evaluate(outputs)

    assert term.count == 2
    assert set(term.diagnostics) == {"range", "lowest", "highest"}
    assert term.diagnostics["lowest"].item() == pytest.approx(min(scores), rel=1e-12)
    assert term.diagnostics["highest"].item() == pytest.approx(max(scores), rel=1e-12)
    assert term.diagnostics["range"].item() == pytest.approx(
        max(scores) - min(scores), rel=1e-12
    )
    assert max(scores) - min(scores) > 0.0
    assert all(diagnostic.grad_fn is None for diagnostic in term.diagnostics.values())


def test_a_batch_with_one_score_everywhere_has_no_range() -> None:
    term = objective().evaluate(assemble(with_score(0.4)))

    assert term.diagnostics["range"].item() == 0.0
    assert term.diagnostics["lowest"].item() == pytest.approx(0.4, rel=1e-12)


def test_an_unlabeled_batch_gets_the_zero_term_and_still_reports_the_range() -> None:
    outputs = unlabeled()
    scores = scores_of(outputs)

    term = objective().evaluate(outputs)

    assert term.count == 0
    assert term.value.item() == 0.0
    assert term.value.grad_fn is not None
    assert term.diagnostics["range"].item() == pytest.approx(
        max(scores) - min(scores), rel=1e-12
    )


def test_an_empty_batch_gets_the_zero_term() -> None:
    outputs = assemble(
        random_log_masses(0, items=0),
        lengths=(),
        labels=(),
        computed=live_spans(()),
        ids=(),
    )

    term = objective().evaluate(outputs)

    assert term.count == 0
    assert term.value.item() == 0.0
    assert term.value.grad_fn is not None
    assert not term.diagnostics


def test_a_batch_whose_labels_do_not_match_its_lengths_is_refused() -> None:
    with pytest.raises(ValueError, match="labels 2 items"):
        objective().evaluate(assemble(random_log_masses(0), labels=(True, False)))


def test_the_gradient_reaches_the_root_cells_alone() -> None:
    outputs = assemble(random_log_masses(0), labels=(True, None, False))

    objective().evaluate(outputs).value.backward()

    gradient = outputs.readouts[ROOT_MASS].distribution.tensor.grad
    touched = torch.zeros(ITEMS, SPANS.num_spans, 2, dtype=torch.bool)
    touched[0, SPANS.span_id(0, 3), 0] = True
    touched[2, SPANS.span_id(0, 3), 1] = True

    assert gradient is not None
    assert bool((gradient[touched] != 0.0).all())
    assert not bool(gradient[~touched].any())


def test_the_noisy_gradient_reaches_both_outcomes_of_a_labeled_root() -> None:
    outputs = assemble(random_log_masses(0), labels=(True, None, None))

    objective(noise=0.2).evaluate(outputs).value.backward()

    gradient = outputs.readouts[ROOT_MASS].distribution.tensor.grad

    assert gradient is not None
    assert bool((gradient[0, SPANS.span_id(0, 3)] != 0.0).all())
    assert not bool(gradient[1:].any())


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
    batch: Labeled
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


def test_the_objective_reads_the_batch_and_the_mass_id_alone() -> None:
    scored = objective()
    outputs = bundle()
    proxy = RecordingOutputs(outputs, objective=scored.name, requires=scored.requires)

    term = scored.evaluate(proxy)

    assert proxy.touched == {"batch", "readouts"}
    assert term.count == ITEMS


def test_the_objective_reads_the_mass_id_and_never_the_types() -> None:
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
    recorded = ReadKeys({TYPES: types, ROOT_MASS: pair_of(random_log_masses(0))})
    outputs = Published(
        SPANS,
        CellChart(LogDistribution(torch.where(LIVE, -1.0, FILL), SPAN_AXES, LIVE)),
        Labeled(IDS, LENGTHS, LABELS),
        recorded,
        Encoded(torch.zeros(ITEMS, SPANS.num_spans, 2), CELL_AXES, LIVE),
    )

    term = objective().evaluate(outputs)

    assert set(recorded) == {TYPES, ROOT_MASS}
    assert set(recorded.read) == {ROOT_MASS}
    assert term.count == ITEMS


def test_the_roots_are_read_in_one_gather() -> None:
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

    assert term.count == ITEMS
    assert gathers == [ITEMS]


def test_a_root_the_readout_never_computed_is_a_named_error() -> None:
    with pytest.raises(UncomputedCellError) as caught:
        objective().evaluate(assemble(random_log_masses(0), lengths=(3, 3, 3)))

    assert caught.value.item == 1
    assert caught.value.span == (0, 3)


def test_a_length_the_chart_does_not_reach_is_refused() -> None:
    with pytest.raises(ValueError, match="not a span of an input"):
        objective().evaluate(assemble(random_log_masses(0), lengths=(3, LENGTH + 1, 3)))


def test_an_empty_item_is_refused() -> None:
    with pytest.raises(ValueError, match="not a span of an input"):
        objective().evaluate(assemble(random_log_masses(0), lengths=(3, 0, 3)))


@pytest.mark.parametrize("outcome", [0, 1])
@pytest.mark.parametrize("published", [0.1, math.nan])
def test_a_log_mass_above_zero_or_not_a_number_is_refused_rather_than_a_nan(
    published: float, outcome: int
) -> None:
    pair = torch.full((ITEMS, SPANS.num_spans, 2), -1.0).double()
    pair[1, SPANS.span_id(0, 2), outcome] = published
    outputs: Outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(torch.where(LIVE, -1.0, FILL), SPAN_AXES, LIVE)
        ),
        batch=Labeled(IDS, LENGTHS, (True, None, True)),
        readouts={
            ROOT_MASS: SpanReadout(SPANS, LogDistribution(pair, MASS_AXES, LIVE))
        },
    )

    with pytest.raises(ImproperMassError) as caught:
        objective().evaluate(outputs)

    assert caught.value.readout == ROOT_MASS
    assert caught.value.item == 1
    assert caught.value.span == (0, 2)
    assert math.isnan(caught.value.value) or caught.value.value == pytest.approx(0.1)


def test_a_published_score_of_exactly_one_is_accepted_on_a_grammatical_item() -> None:
    held = random_log_masses(0)
    held[0, SPANS.span_id(0, 3)] = 0.0

    term = objective().evaluate(assemble(held, labels=(True, None, None)))

    assert term.value.item() == 0.0
    assert term.diagnostics["highest"].item() == 1.0


def test_a_declared_id_the_step_did_not_publish_is_refused() -> None:
    with pytest.raises(ValueError, match="reads the root mass 'sentences'"):
        objective().evaluate(assemble(random_log_masses(0), key=ReadoutId("other")))


def test_a_readout_that_is_not_a_mass_pair_is_refused() -> None:
    outputs: Outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(torch.where(LIVE, -1.0, FILL), SPAN_AXES, LIVE)
        ),
        batch=Labeled(IDS, LENGTHS, LABELS),
        readouts={
            ROOT_MASS: SpanReadout(
                SPANS,
                LogDistribution(torch.full((ITEMS, SPANS.num_spans), -1.0), SPAN_AXES),
            )
        },
    )

    with pytest.raises(ValueError, match="reads a mass pair over"):
        objective().evaluate(outputs)


def test_the_objective_declares_what_it_reads_and_what_it_scores() -> None:
    scored = objective(noise=0.1)

    assert scored.name == "grammaticality"
    assert scored.requires == frozenset({"batch", "readouts"})
    assert scored.readout == ROOT_MASS
    assert scored.noise == 0.1
    assert "sentences" in repr(scored)
    assert "0.1" in repr(scored)
    assert objective(name="acceptability").name == "acceptability"


def test_an_empty_id_or_name_is_refused() -> None:
    with pytest.raises(ValueError, match="readout id"):
        objective(ReadoutId(""))

    with pytest.raises(ValueError, match="name"):
        objective(name="")


def test_the_term_is_weighted_by_the_suite_beside_another() -> None:
    outputs = bundle()
    plain = objective()
    noisy = objective(noise=0.2, name="noisy")
    suite = ObjectiveSuite([(0.5, plain), (2.0, noisy)])

    result = suite.evaluate(bundle())

    assert result.total.item() == pytest.approx(
        0.5 * plain.evaluate(outputs).value.item()
        + 2.0 * noisy.evaluate(outputs).value.item(),
        rel=1e-12,
    )
    assert "grammaticality/range" in result.diagnostics
    assert "noisy/range" in result.diagnostics
    assert result.terms["grammaticality"].count == ITEMS


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


def marginals() -> SpanMarginals:
    widths = torch.tensor([j - i for i, j in SPANS.spans_by_length()])
    built = LIVE & (widths > 1)
    constituent = torch.zeros(ITEMS, SPANS.num_spans, dtype=torch.float64)
    rule = torch.zeros(ITEMS, SPANS.num_spans, SPANS.max_splits, dtype=torch.float64)
    valid = torch.zeros(ITEMS, SPANS.num_spans, SPANS.max_splits, dtype=torch.bool)

    for item, length in enumerate(LENGTHS):
        constituent[item, SPANS.span_id(0, length)] = 1.0

        for i, j in SPANS.spans_by_length(min_length=2):
            if j > length:
                continue

            for k in SPANS.splits(i, j):
                span, column = SPANS.split_id(i, j, k)
                valid[item, span, column] = True

        if length == LENGTH:
            constituent[item, SPANS.span_id(0, 2)] = 0.3
            constituent[item, SPANS.span_id(1, 3)] = 0.7
            rule[item, SPANS.span_id(0, 2), 0] = 0.3
            rule[item, SPANS.span_id(1, 3), 0] = 0.7
            rule[item, SPANS.span_id(0, 3), 0] = 0.3
            rule[item, SPANS.span_id(0, 3), 1] = 0.7

        else:
            rule[item, SPANS.span_id(0, 2), 0] = 1.0

    return SpanMarginals(
        SPANS,
        Encoded(constituent, SPAN_AXES, built),
        Encoded(rule, SPLIT_AXES, valid),
        torch.tensor(LENGTHS),
    )


@pytest.mark.parametrize("strategy", list(ReadoutMass))
def test_the_score_is_the_published_root_mass_of_the_readout(
    strategy: ReadoutMass,
) -> None:
    built = decoder()
    held = cells()
    readout = TermsReadout(
        built,
        codec=VIEW,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=TYPES,
        targets={ROOT_MASS: linearize_targets(SENTENCES, CODEC)},
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
    published = readout.publish(held, spans=SPANS, marginals=marginals())
    outputs: Outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(torch.where(LIVE, -1.0, FILL).double(), SPAN_AXES, LIVE)
        ),
        batch=Labeled(IDS, LENGTHS, LABELS),
        readouts=published,
    )
    root_types = published[TYPES].gather(
        [(item, 0, length) for item, length in enumerate(LENGTHS)]
    )
    symbol = CODEC.encode(category("S")).symbols[0]
    expected = root_types.tensor[:, 0, symbol].detach().exp().tolist()

    term = objective().evaluate(outputs)
    term.value.backward()

    assert math.isfinite(term.value.item())
    assert term.value.item() == pytest.approx(by_hand(outputs), rel=1e-12)
    assert scores_of(outputs) == pytest.approx(expected, rel=1e-9)
    assert term.diagnostics["range"].item() == pytest.approx(
        max(expected) - min(expected), rel=1e-9
    )
    assert held.tensor.grad is not None
    assert bool(held.tensor.grad.isfinite().all())

    for item, span in enumerate(roots()):
        assert bool(held.tensor.grad[item, span].abs().sum() > 0.0)

    assert not bool(held.tensor.grad[~LIVE].any())
    reached = [
        parameter.grad for parameter in built.parameters() if parameter.grad is not None
    ]

    assert reached
    assert all(bool(gradient.isfinite().all()) for gradient in reached)


def test_the_fixtures_transform_the_outputs_they_are_handed() -> None:
    held = random_log_masses(7)
    outputs = assemble(held)

    swapped = permuted(outputs)
    refilled_outputs = refilled(outputs)
    order = [2, 0, 1]

    assert swapped.batch.ids == tuple(IDS[position] for position in order)
    assert swapped.batch.lengths == tuple(LENGTHS[position] for position in order)
    assert swapped.batch.labels == tuple(LABELS[position] for position in order)
    assert torch.equal(log_masses_of(swapped)[LIVE[order]], held[order][LIVE[order]])
    assert torch.equal(log_masses_of(refilled_outputs)[LIVE], held[LIVE])
    assert bool((log_masses_of(refilled_outputs)[~LIVE] == FILL / 3.0).all())
    assert refilled_outputs.batch == outputs.batch
