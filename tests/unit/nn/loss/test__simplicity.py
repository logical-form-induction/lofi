import math
from typing import TYPE_CHECKING, Final, cast, override

import didactic.api as dx
import pytest
import torch
from tests.conformance.configured_component import (
    ConfiguredComponentConformance,
    ConfiguredComponentFixture,
)
from tests.conformance.objective import ObjectiveLawSuite

import lofi.nn.loss._simplicity as simplicity_module
from lofi._types import ReadoutId
from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.codec import NeuralCodecView
from lofi.nn.loss._description_cost import (
    DescriptionCost,
    PerConstructorDescriptionCost,
    UniformDescriptionCost,
)
from lofi.nn.loss._mapping import SOFTMAX
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._readout import (
    TERMS_READOUT_AXES,
    TermsReadout,
)
from lofi.nn.loss._simplicity import (
    SimplicityPriorObjective,
    expected_description_lengths,
)
from lofi.nn.loss._suite import ObjectiveSuite
from lofi.nn.loss._values import CellChart, ObjectiveOutputs, SpanReadout
from lofi.nn.loss.laws import (
    ObjectiveFixture,
    the_value_is_monotone_where_the_fixture_says,
)
from lofi.spec.losses.simplicity import SimplicityLossSpec, SimplicityScope
from lofi.spec.readouts import DecodedMeaningSettings, ReadoutSource
from lofi.spec.readouts.terms import ReadoutMass
from lofi.terms import (
    ChoiceIndex,
    CompleteTree,
    Encoding,
    Operator,
    OperatorIndex,
    PreorderArityLayout,
    Signature,
    TermCodec,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.nn.loss._ports import (
        LossTerm,
        ProofRewardBindings,
        SpanTokenDecoder,
    )
    from lofi.nn.spaces import Space, SpaceRelationBinding
    from lofi.spec import ConfiguredSpec

type Batch = tuple[str, ...]
type Outputs = ObjectiveOutputs[Batch]


SIGNATURE: Final = Signature(
    "simplicity-fixture",
    ("Cat",),
    (
        Operator("N", "Cat", family="primitive"),
        Operator("S", "Cat", family="primitive"),
        Operator("/", "Cat", arg_sorts=("Cat", "Cat"), family="constructor"),
    ),
)
INDEX: Final = OperatorIndex(SIGNATURE)
LAYOUT: Final = CompleteTree(SIGNATURE, SIGNATURE.operators, order="df", depth=2)
CODEC: Final[TermCodec] = TermCodec(INDEX, LAYOUT)
VIEW: Final = NeuralCodecView(CODEC)
CHOICES: Final = ChoiceIndex(SIGNATURE, depth_bounds={"Cat": CODEC.capacity})
SLOTS: Final = CODEC.capacity
SYMBOLS: Final = CODEC.n_symbols
PAD: Final = INDEX.PAD
N: Final = INDEX.index("N")
S: Final = INDEX.index("S")
LENGTH: Final = 3
ITEMS: Final = 2
LENGTHS: Final = (3, 2)
SPANS: Final = SpanIndex(LENGTH)
BATCH: Final[Batch] = ("first", "second")
PADDING: Final = -9.0
KEY: Final = ReadoutId("types")
OCCUPANCIES: Final = (0.1, 0.3, 0.6, 0.9)


def root_slot() -> int:
    skeleton = CODEC.skeleton

    assert skeleton is not None

    return skeleton.parents.index(Encoding.NO_PARENT)


ROOT: Final = root_slot()


def live_spans(lengths: tuple[int, ...] = LENGTHS) -> Tensor:
    mask = torch.zeros(len(lengths), SPANS.num_spans, dtype=torch.bool)

    for item, length in enumerate(lengths):
        for start, end in SPANS.spans_by_length():
            if end <= length:
                mask[item, SPANS.span_id(start, end)] = True

    return mask


LIVE: Final = live_spans()
LEXICAL: Final = torch.tensor(
    [SPANS.span_id(i, j) for i, j in SPANS.leaves()], dtype=torch.long
)


def objective(
    scope: SimplicityScope = SimplicityScope.LEXICON,
    *,
    readout: ReadoutId = KEY,
    codec: TermCodec = CODEC,
    symbol_cost: DescriptionCost | None = None,
) -> SimplicityPriorObjective[Batch]:
    return SimplicityPriorObjective(
        readout=readout,
        codec=codec,
        scope=scope,
        symbol_cost=(
            UniformDescriptionCost(1.0) if symbol_cost is None else symbol_cost
        ),
        name="simplicity",
    )


def random_scores(seed: int, items: int = ITEMS) -> Tensor:
    generator = torch.Generator().manual_seed(seed)

    return torch.randn(
        items, SPANS.num_spans, SLOTS, SYMBOLS, generator=generator, dtype=torch.float64
    ).log_softmax(dim=-1)


def with_padding(padding: float, *, elsewhere: int = N) -> Tensor:
    """Return scores giving the padding code one probability at every slot.

    The rest of the mass goes to one symbol, so the distribution is exact.
    """
    scores = torch.full((ITEMS, SPANS.num_spans, SLOTS, SYMBOLS), -math.inf)
    scores = scores.to(torch.float64)
    scores[..., PAD] = math.log(padding)
    scores[..., elsewhere] = math.log1p(-padding)

    return scores


def assemble(
    scores: Tensor,
    *,
    computed: Tensor = LIVE,
    fill: float = PADDING,
    batch: Batch = BATCH,
) -> Outputs:
    padded = torch.where(computed.reshape(*computed.shape, 1, 1), scores, fill)

    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(torch.where(computed, -1.0, fill), SPAN_AXES, computed)
        ),
        batch=batch,
        readouts={
            KEY: SpanReadout(
                SPANS,
                LogDistribution(
                    padded.detach().requires_grad_(),
                    TERMS_READOUT_AXES,
                    computed,
                    normalized_over="symbol",
                ),
            )
        },
    )


def bundle() -> Outputs:
    return assemble(random_scores(0))


def permuted(outputs: Outputs) -> Outputs:
    del outputs
    order = torch.tensor([1, 0])

    return assemble(
        random_scores(0)[order],
        computed=LIVE[order],
        batch=tuple(reversed(BATCH)),
    )


def refilled(outputs: Outputs) -> Outputs:
    del outputs

    return assemble(random_scores(0), fill=PADDING / 3.0)


def empty_bundle() -> Outputs:
    return assemble(
        random_scores(0), computed=torch.zeros(ITEMS, SPANS.num_spans).bool()
    )


def inputs_of(outputs: Outputs) -> tuple[Tensor, ...]:
    return (outputs.readouts[KEY].distribution.tensor,)


def by_hand(
    outputs: Outputs, scope: SimplicityScope = SimplicityScope.LEXICON
) -> float:
    published = outputs.readouts[KEY]
    total = 0.0
    count = 0

    for item in range(published.items):
        for start, end in SPANS.spans_by_length():
            if scope is SimplicityScope.LEXICON and end - start != 1:
                continue

            if not bool(published.computed[item, SPANS.span_id(start, end)]):
                continue

            cell = published.at(item, start, end).tensor.detach()
            count += 1

            for slot in range(SLOTS):
                total += 1.0 - math.exp(float(cell[slot, PAD]))

    return total / count if count else 0.0


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
    return ObjectiveFixture(
        objective=objective(),
        outputs=bundle,
        inputs=inputs_of,
        unweighted=by_hand,
        monotone=lambda: (assemble(with_padding(0.05)), assemble(with_padding(0.9))),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        inapplicable=empty_bundle,
        tolerance=1e-9,
    )


class TestSimplicityPriorObjectiveLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Batch]:
        return fixture()

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that a chart peaked off padding scores above a padding-heavy one."""
        the_value_is_monotone_where_the_fixture_says(conformance)


def test_rewards_padding_mass() -> None:
    heavy = objective().evaluate(assemble(with_padding(0.9))).value.item()
    peaked = objective().evaluate(assemble(with_padding(0.1))).value.item()

    assert heavy < peaked


def test_the_value_is_what_the_cell_by_cell_reckoning_gives() -> None:
    outputs = bundle()

    assert objective().evaluate(outputs).value.item() == pytest.approx(
        by_hand(outputs), rel=1e-9
    )


def test_the_unaggregated_kernel_sums_to_the_objectives_exact_numerator() -> None:
    outputs = bundle()
    published = outputs.readouts[KEY]
    lengths, priced = expected_description_lengths(
        published,
        outputs.spans,
        CODEC,
        SimplicityScope.LEXICON,
        UniformDescriptionCost(1.0),
    )
    term = objective().evaluate(outputs)
    expected = torch.zeros_like(published.computed)
    expected[:, LEXICAL] = published.computed[:, LEXICAL]

    assert lengths.shape == published.computed.shape
    assert torch.equal(priced, expected)
    assert lengths[priced].sum().item() == pytest.approx(
        term.value.item() * term.count, rel=1e-12
    )


def test_the_value_is_linear_in_the_expected_occupied_node_count() -> None:
    values = [
        objective().evaluate(assemble(with_padding(1.0 - occupancy))).value.item()
        for occupancy in OCCUPANCIES
    ]
    expected = [SLOTS * occupancy for occupancy in OCCUPANCIES]

    assert values == pytest.approx(expected, rel=1e-12)


def test_the_root_constructor_is_part_of_the_description_length() -> None:
    scores = with_padding(0.4)
    rerooted = scores.clone()
    rerooted[..., ROOT, :] = torch.tensor([-math.inf] * SYMBOLS, dtype=torch.float64)
    rerooted[..., ROOT, S] = 0.0

    rerooted_value = objective().evaluate(assemble(rerooted)).value.item()
    original = objective().evaluate(assemble(scores)).value.item()

    assert rerooted_value - original == pytest.approx(0.4, rel=1e-12)


def test_a_node_costs_the_same_whatever_symbol_it_holds() -> None:
    on_nouns = objective().evaluate(assemble(with_padding(0.4, elsewhere=N)))
    on_sentences = objective().evaluate(assemble(with_padding(0.4, elsewhere=S)))
    mixed = with_padding(0.4, elsewhere=N)
    mixed[..., N] = math.log(0.3)
    mixed[..., S] = math.log(0.3)

    assert on_nouns.value.item() == on_sentences.value.item()
    assert objective().evaluate(assemble(mixed)).value.item() == pytest.approx(
        on_nouns.value.item(), rel=1e-12
    )


def test_the_lexicon_alone_is_priced_by_default() -> None:
    scores = random_scores(0)
    interior = scores.clone()
    interior[:, SPANS.span_id(0, LENGTH)] = with_padding(0.99)[:, 0]
    priced = objective()

    assert priced.scope is SimplicityScope.LEXICON
    assert priced.evaluate(assemble(interior)).value.item() == (
        priced.evaluate(assemble(scores)).value.item()
    )
    assert priced.evaluate(assemble(scores)).count == int(LIVE[:, LEXICAL].sum())


def test_every_cell_is_priced_under_the_wider_scope() -> None:
    scores = random_scores(0)
    interior = scores.clone()
    interior[:, SPANS.span_id(0, LENGTH)] = with_padding(0.99)[:, 0]
    priced = objective(SimplicityScope.ALL_CELLS)

    assert priced.evaluate(assemble(interior)).value.item() != pytest.approx(
        priced.evaluate(assemble(scores)).value.item()
    )
    assert priced.evaluate(assemble(scores)).value.item() == pytest.approx(
        by_hand(assemble(scores), SimplicityScope.ALL_CELLS), rel=1e-9
    )
    assert priced.evaluate(assemble(scores)).count == int(LIVE.sum())


def test_the_padding_diagnostic_is_the_mean_padding_probability() -> None:
    term = objective().evaluate(assemble(with_padding(0.4)))

    assert term.diagnostics["padding"].item() == pytest.approx(0.4, rel=1e-12)
    assert term.diagnostics["padding"].grad_fn is None


def test_the_objective_declares_what_it_reads() -> None:
    priced = objective()

    assert priced.requires == frozenset({"spans", "readouts"})
    assert priced.name == "simplicity"
    assert priced.readout == KEY
    assert "LEXICON" in repr(priced)


def test_the_term_sums_with_another_at_its_declared_weight_through_the_suite() -> None:
    outputs = bundle()
    lexical = objective()
    everywhere = SimplicityPriorObjective(
        readout=KEY,
        codec=CODEC,
        scope=SimplicityScope.ALL_CELLS,
        symbol_cost=UniformDescriptionCost(1.0),
        name="simplicity-all",
    )
    suite = ObjectiveSuite([(0.25, lexical), (4.0, everywhere)])

    result = suite.evaluate(outputs)

    assert result.total.item() == pytest.approx(
        0.25 * lexical.evaluate(outputs).value.item()
        + 4.0 * everywhere.evaluate(outputs).value.item(),
        rel=1e-12,
    )
    assert "simplicity/padding" in result.diagnostics


def test_a_preorder_codec_is_priced_over_every_slot() -> None:
    preorder = TermCodec(INDEX, PreorderArityLayout(SLOTS))
    priced = objective(codec=preorder)
    scores = with_padding(0.4)
    rerooted = scores.clone()
    rerooted[..., 0, :] = torch.tensor([-math.inf] * SYMBOLS, dtype=torch.float64)
    rerooted[..., 0, S] = 0.0

    assert priced.evaluate(assemble(scores)).value.item() == pytest.approx(
        SLOTS * 0.6, rel=1e-12
    )
    assert priced.evaluate(assemble(rerooted)).value.item() != pytest.approx(
        priced.evaluate(assemble(scores)).value.item()
    )


def test_per_constructor_costs_change_the_expected_description_length() -> None:
    scores = with_padding(0.4, elsewhere=N)
    uniform = objective().evaluate(assemble(scores)).value.item()
    keyed = (
        objective(
            symbol_cost=PerConstructorDescriptionCost({"N": 0.25, "S": 1.0, "/": 2.0})
        )
        .evaluate(assemble(scores))
        .value.item()
    )

    assert keyed == pytest.approx(uniform * 0.25, rel=1e-12)


def test_constructor_cost_sweep_shifts_the_trained_category_distribution() -> None:
    def trained(symbol_cost: DescriptionCost) -> Tensor:
        logits = torch.zeros(SYMBOLS, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.SGD((logits,), lr=0.25)

        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            scores = logits.log_softmax(-1).reshape(1, 1, 1, SYMBOLS)
            scores = scores.expand(ITEMS, SPANS.num_spans, SLOTS, SYMBOLS)
            padded = torch.where(
                LIVE.reshape(ITEMS, SPANS.num_spans, 1, 1), scores, PADDING
            )
            outputs: Outputs = ObjectiveOutputs(
                spans=SPANS,
                chart=CellChart(
                    LogDistribution(torch.where(LIVE, -1.0, PADDING), SPAN_AXES, LIVE)
                ),
                batch=BATCH,
                readouts={
                    KEY: SpanReadout(
                        SPANS,
                        LogDistribution(
                            padded,
                            TERMS_READOUT_AXES,
                            LIVE,
                            normalized_over="symbol",
                        ),
                    )
                },
            )
            objective(SimplicityScope.ALL_CELLS, symbol_cost=symbol_cost).evaluate(
                outputs
            ).value.backward()
            optimizer.step()

        return logits.softmax(-1).detach()

    uniform = trained(UniformDescriptionCost(1.0))
    keyed = trained(PerConstructorDescriptionCost({"N": 0.25, "S": 0.25, "/": 3.0}))
    slash = INDEX.index("/")

    assert uniform[slash].item() == pytest.approx(uniform[N].item(), rel=1e-12)
    assert keyed[slash] < keyed[N]


def test_a_readout_id_the_step_did_not_publish_is_refused() -> None:
    with pytest.raises(ValueError, match="reads the readout 'forms'"):
        objective(readout=ReadoutId("forms")).evaluate(bundle())


def test_a_readout_over_other_axes_is_refused() -> None:
    flat: Outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(LogDistribution(torch.full((ITEMS, 6), -1.0), SPAN_AXES)),
        batch=BATCH,
        readouts={
            KEY: SpanReadout(
                SPANS, LogDistribution(torch.full((ITEMS, 6), -1.0), SPAN_AXES)
            )
        },
    )

    with pytest.raises(ValueError, match="prices a readout over"):
        objective().evaluate(flat)


@pytest.mark.parametrize(
    ("slots", "symbols"), [(SLOTS + 1, SYMBOLS), (SLOTS, SYMBOLS + 1)]
)
def test_a_readout_over_other_slots_or_symbols_than_the_codec_is_refused(
    slots: int, symbols: int
) -> None:
    scores = torch.zeros(ITEMS, SPANS.num_spans, slots, symbols).log_softmax(-1)
    outputs: Outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(LogDistribution(torch.full((ITEMS, 6), -1.0), SPAN_AXES)),
        batch=BATCH,
        readouts={
            KEY: SpanReadout(
                SPANS,
                LogDistribution(scores, TERMS_READOUT_AXES, normalized_over="symbol"),
            )
        },
    )

    with pytest.raises(ValueError, match="the codec lays out"):
        objective().evaluate(outputs)


def test_a_blank_readout_id_or_name_is_refused() -> None:
    with pytest.raises(ValueError, match="readout id"):
        objective(readout=ReadoutId(""))

    with pytest.raises(ValueError, match="name"):
        SimplicityPriorObjective(
            readout=KEY,
            codec=CODEC,
            name="",
            scope=SimplicityScope.LEXICON,
            symbol_cost=UniformDescriptionCost(1.0),
        )


class CodecDecoder:
    """A decoder declaring a distribution over the fixture codec's symbols."""

    def marginal_rows(self, inputs: Encoded, /, *, root_sort: str) -> LogDistribution:
        raise AssertionError(inputs, root_sort)


class Binding:
    """One term readout installed under the role the declaration names."""

    def __init__(self) -> None:
        self.readout = TermsReadout(
            CodecDecoder(),
            codec=VIEW,
            index=CHOICES,
            reads=ReadoutSource.INSIDE,
            decoding=None,
            key=KEY,
            targets={},
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
        self.requests: list[str] = []

    @property
    def name(self) -> str:
        return "simple-types"

    def terms_readout(self, role: str, /) -> TermsReadout:
        self.requests.append(role)

        if role != "priced":
            msg = f"no term readout is bound as {role!r}"
            raise ValueError(msg)

        return self.readout

    def space(self, role: str, /) -> Space:
        msg = f"simplicity must not resolve represented space {role!r}"
        raise AssertionError(msg)

    def readout_space(self, role: str, /) -> Space:
        msg = f"simplicity must not resolve a readout space for {role!r}"
        raise AssertionError(msg)

    def relation(self, role: str, /) -> SpaceRelationBinding:
        msg = f"simplicity must not resolve space relation {role!r}"
        raise AssertionError(msg)

    def span_decoder(self, role: str, /) -> SpanTokenDecoder:
        msg = f"simplicity must not resolve span decoder {role!r}"
        raise AssertionError(msg)

    def proof_reward(self) -> ProofRewardBindings:
        msg = "simplicity must not resolve proof-reward bindings"
        raise AssertionError(msg)


def declaration(scope: str = "lexicon") -> SimplicityLossSpec:
    return SimplicityLossSpec.model_validate({"readout": "priced", "scope": scope})


def configured(
    scope: str = "lexicon", context: Binding | None = None
) -> SimplicityPriorObjective[Batch]:
    return simplicity_module.PLUGIN.construct(
        declaration(scope), Binding() if context is None else context
    )


class TestConfiguredSimplicityPriorLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Batch]:
        return fixture()._replace(objective=configured())

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that a chart peaked off padding scores above a padding-heavy one."""
        the_value_is_monotone_where_the_fixture_says(conformance)


class TestSimplicityPlugin(ConfiguredComponentConformance):
    @pytest.fixture
    def configured_component_fixture(self) -> ConfiguredComponentFixture:
        context = Binding()
        readout = context.readout

        def observe(built: object) -> dict[str, bool]:
            held = cast("SimplicityPriorObjective[Batch]", built)

            return {
                "named for its term": held.name == "simple-types",
                "reads the declared readout's key": held.readout == KEY,
                "holds the declared scope": held.scope is SimplicityScope.ALL_CELLS,
            }

        return ConfiguredComponentFixture(
            declaration=cast("ConfiguredSpec[object]", declaration("all-cells")),
            context=context,
            expected_type=SimplicityPriorObjective,
            observe=observe,
            unchanged=lambda: context.readout is readout,
        )


@pytest.mark.parametrize(
    ("scope", "expected"),
    [("lexicon", SimplicityScope.LEXICON), ("all-cells", SimplicityScope.ALL_CELLS)],
)
def test_the_plugin_binds_the_readout_key_and_codec_of_the_declared_role(
    scope: str, expected: SimplicityScope
) -> None:
    context = Binding()
    bound = configured(scope, context)
    outputs = bundle()

    assert bound.name == "simple-types"
    assert context.requests == ["priced"]
    assert bound.readout == KEY
    assert bound.scope is expected
    assert bound.evaluate(outputs).value.item() == pytest.approx(
        objective(expected).evaluate(outputs).value.item(), rel=1e-12
    )


def test_the_plugin_resolves_the_declared_constructor_cost_policy() -> None:
    context = Binding()
    declaration = SimplicityLossSpec.model_validate(
        {
            "readout": "priced",
            "scope": "all-cells",
            "symbol_cost": {
                "kind": "per-constructor",
                "costs": {"N": 0.25, "S": 1.0, "/": 2.0},
            },
        }
    )
    built = simplicity_module.PLUGIN.construct(declaration, context)
    direct = objective(
        SimplicityScope.ALL_CELLS,
        symbol_cost=PerConstructorDescriptionCost({"N": 0.25, "S": 1.0, "/": 2.0}),
    )

    assert built.evaluate(bundle()).value.item() == pytest.approx(
        direct.evaluate(bundle()).value.item(), rel=1e-12
    )
