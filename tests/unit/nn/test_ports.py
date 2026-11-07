from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, Self, cast

import pytest
import torch

from lofi.data import Alignment
from lofi.judge import Action, ActionSet
from lofi.nn._values import (
    ChartReadouts,
    Encoded,
    LogDistribution,
    TokenEncoding,
)
from lofi.nn.arity import ArityHeadScheme, RoleFillerScheme
from lofi.nn.ports import (
    ArityScheme,
    ChartScorer,
    Consumption,
    Controller,
    ForwardMode,
    ForwardOptions,
    FrontierDecoder,
    Instrumentable,
    InstrumentScope,
    MarginalDecoder,
    Observable,
    Patch,
    Reading,
    ReceiverEdge,
    RelationKind,
    Relaxation,
    SoftTermEncoder,
    StandardDeclaration,
    StructuralRelation,
    SubwordPooler,
    SyntheticTreeEncoder,
    TeacherForcedDecoder,
    TermEncoder,
    TokenEncoder,
    describe_structure,
    ensure_mode,
)
from lofi.nn.sampling import KEEP_OFFERED, SamplingStrategy
from lofi.nn.sites import (
    Access,
    ComponentId,
    ComponentKind,
    DTypeClass,
    ReceiverSide,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)
from lofi.nn.tokens.pool import MeanSubwordPooler
from lofi.probe import Noise, Observation, Shuffle, Zero
from lofi.terms import BatchEncoding, Encoding, HoleContext, Operator, Term

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

PAD = 0
ALPHABET = 4


def row(
    symbols: tuple[int, ...], parents: tuple[int, ...], depths: tuple[int, ...]
) -> Encoding:
    return Encoding(symbols, parents, depths, tuple(code != PAD for code in symbols))


ROOTED = (Encoding.NO_PARENT, 0, 0)
LEVELS = (0, 1, 1)

TERMS = BatchEncoding(
    (
        row((3, 1, 2), ROOTED, LEVELS),
        row((2, 1, PAD), ROOTED, LEVELS),
    )
)


def observable(
    component: str,
    kind: ComponentKind,
    *points: tuple[Role, tuple[int | str, ...], tuple[str, ...], DTypeClass],
) -> SiteMap:
    named = ComponentId(component)

    return SiteMap(
        named,
        kind,
        {
            Site(named, role, key): SiteSpec(
                axes, dtype_class, differentiable=True, access=Access.OBSERVE
            )
            for role, key, axes, dtype_class in points
        },
    )


def codes(terms: BatchEncoding) -> torch.Tensor:
    return torch.tensor(terms.symbols, dtype=torch.int64)


def one_hot(terms: BatchEncoding) -> LogDistribution:
    hot = torch.nn.functional.one_hot(codes(terms), ALPHABET).double()

    return LogDistribution(
        hot.log(), ("term", "slot", "symbol"), normalized_over="symbol"
    )


class CodeSumEncoder:
    @property
    def output_width(self) -> int:
        return 1

    def encode(self, terms: BatchEncoding, /) -> Encoded:
        totals = codes(terms).sum(dim=1, keepdim=True)

        return Encoded(totals.double(), ("term", "width"))

    def encode_nodes(self, terms: BatchEncoding, /) -> Encoded:
        return Encoded(
            codes(terms).unsqueeze(-1).double(),
            ("term", "slot", "width"),
            torch.tensor(terms.mask),
        )

    def encode_soft(
        self,
        symbols: LogDistribution,
        /,
        *,
        structure: BatchEncoding,
    ) -> Encoded:
        del structure
        alphabet = torch.arange(symbols.size("symbol"), dtype=torch.float64)
        over_symbols = symbols.axis("symbol")
        expected = (symbols.tensor.exp() * alphabet).sum(dim=over_symbols)

        return Encoded(
            expected.sum(dim=symbols.axis("slot"), keepdim=True), ("term", "width")
        )

    def site_map(self) -> SiteMap:
        return observable(
            "code-sum-encoder",
            ComponentKind.TERM_ENCODER,
            (Role.TERM_ENCODING, (), ("term", "width"), DTypeClass.REAL),
            (
                Role.NODE_ENCODING,
                ("node",),
                ("term", "slot", "width"),
                DTypeClass.REAL,
            ),
        )


class SyntheticCodeSum:
    """A structural continuation fixture."""

    output_width = 1

    def continue_synthetic(
        self,
        structure: BatchEncoding,
        leaves: Encoded,
        /,
        *,
        leaf_slots: tuple[int, ...],
    ) -> Encoded:
        del structure, leaf_slots
        return leaves

    def validate_synthetic_continuation(self) -> None:
        """Accept this complete one-coordinate state."""


def test_synthetic_tree_continuation_is_a_published_encoder_capability() -> None:
    assert isinstance(SyntheticCodeSum(), SyntheticTreeEncoder)


class SlotRows:
    def __init__(self, slots: int, choices: int) -> None:
        self._slots = slots
        self._choices = choices

    def _log_probabilities(self, encoded: Encoded) -> torch.Tensor:
        alphabet = torch.arange(self._choices, dtype=torch.float64)
        slots = torch.arange(1, self._slots + 1, dtype=torch.float64).unsqueeze(-1)
        logits = encoded.tensor.unsqueeze(-1) * alphabet * slots

        return logits.log_softmax(dim=-1)

    def rows(self, encoded: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        del terms
        return LogDistribution(
            self._log_probabilities(encoded),
            ("term", "slot", "choice"),
            normalized_over="choice",
        )

    def marginal_rows(self, encoded: Encoded, /, *, root_sort: str) -> LogDistribution:
        del root_sort
        return LogDistribution(
            self._log_probabilities(encoded),
            ("term", "slot", "choice"),
            normalized_over="choice",
        )

    def site_map(self) -> SiteMap:
        return observable(
            "slot-decoder",
            ComponentKind.TERM_DECODER,
            (
                Role.SYMBOL_DISTRIBUTION,
                (),
                ("term", "slot", "choice"),
                DTypeClass.LOG_PROBABILITY,
            ),
            (
                Role.UNBOUND_VECTOR,
                (),
                ("term", "slot", "width"),
                DTypeClass.REAL,
            ),
        )


class StepwiseSlotDecoder:
    def start(self, encoded: Encoded, /) -> torch.Tensor:
        return encoded.tensor

    def weigh(self, state: torch.Tensor, /, *, context: HoleContext) -> LogDistribution:
        del context
        logits = state.sum() * torch.arange(ALPHABET, dtype=state.dtype)

        return LogDistribution(
            logits.log_softmax(dim=0),
            ("choice",),
            normalized_over="choice",
        )

    def fill(self, state: torch.Tensor, /, *, choices: torch.Tensor) -> torch.Tensor:
        return state + choices

    def admits(self, term: Term, /) -> bool:
        del term
        return True


JOIN = Action("join", "concatenation")
KEEP = Action("keep")
GAP_ACTIONS = ActionSet((JOIN, KEEP), KEEP)


class GapController:
    def __init__(self, actions: ActionSet = GAP_ACTIONS) -> None:
        self._actions = actions

    @property
    def action_set(self) -> ActionSet:
        return self._actions

    @property
    def input_width(self) -> int:
        return 1

    @property
    def input(self) -> str:
        return "inside"

    def weigh(self, left: Encoded, /, *, right: Encoded) -> LogDistribution:
        gap = (left.tensor - right.tensor).abs().sum(dim=-1)
        total = (left.tensor + right.tensor).abs().sum(dim=-1)

        return LogDistribution(torch.stack((-gap, -total), dim=-1), ("cell", "action"))

    def site_map(self) -> SiteMap:
        return observable(
            "gap-controller",
            ComponentKind.CONTROLLER,
            (
                Role.COMBINATOR_LOGITS,
                (),
                ("cell", "action"),
                DTypeClass.LOG_PROBABILITY,
            ),
        )


class CharacterEncoder:
    def __init__(self, rows: int) -> None:
        self._rows = rows

    @property
    def output_width(self) -> int:
        return 1

    def encode_words(self, items: Sequence[Sequence[str]], /) -> TokenEncoding:
        vectors = torch.zeros(len(items), self._rows, self.output_width)
        mask = torch.zeros(len(items), self._rows, dtype=torch.bool)
        alignments: list[Alignment] = []

        for item, words in enumerate(items):
            spans: list[tuple[int, int] | None] = []
            at = 0

            for position, word in enumerate(words):
                end = at + len(word)

                if end > self._rows:
                    spans.extend([None] * (len(words) - position))
                    break

                for offset, letter in enumerate(word):
                    vectors[item, at + offset, 0] = float(ord(letter))

                mask[item, at:end] = True
                spans.append((at, end))
                at = end

            alignments.append(Alignment(tuple(spans)))

        rows = Encoded(vectors, ("item", "row", "width"), mask)

        return TokenEncoding(rows, tuple(alignments), words=items)

    def site_map(self) -> SiteMap:
        return observable(
            "character-encoder",
            ComponentKind.TOKEN_ENCODER,
            (
                Role.TOKEN_EMBEDDING,
                (),
                ("item", "row", "width"),
                DTypeClass.REAL,
            ),
        )


class MeanPooler:
    def __init__(self, width: int) -> None:
        self._width = width

    @property
    def output_width(self) -> int:
        return self._width

    def pool(self, encoding: TokenEncoding, /) -> Encoded:
        alignments = encoding.alignments
        words = max(alignment.word_count for alignment in alignments)
        rows = encoding.vectors.tensor
        pooled = rows.new_zeros((len(alignments), words, self._width))
        mask = torch.zeros(len(alignments), words, dtype=torch.bool)

        for item, alignment in enumerate(alignments):
            for word, span in enumerate(alignment.rows):
                if span is None:
                    continue

                start, end = span
                pooled[item, word] = rows[item, start:end].mean(dim=0)
                mask[item, word] = True

        return Encoded(pooled, ("item", "word", "width"), mask)

    def site_map(self) -> SiteMap:
        return observable(
            "mean-pooler",
            ComponentKind.SUBWORD_POOLER,
            (
                Role.WORD_VECTOR,
                (),
                ("item", "word", "width"),
                DTypeClass.REAL,
            ),
        )


class TotalScorer:
    @property
    def publishes(self) -> frozenset[str]:
        return frozenset({"cells", "log_z"})

    def readouts(self, leaves: Encoded, /) -> ChartReadouts:
        values = leaves.tensor[..., 0]

        if leaves.mask is not None:
            values = values.masked_fill(~leaves.mask, float("-inf"))

        total = LogDistribution(values.logsumexp(dim=-1), ("item",))

        return ChartReadouts(
            cells=Encoded(leaves.tensor, ("item", "span", "width"), leaves.mask),
            log_z=total,
        )

    def site_map(self) -> SiteMap:
        return observable(
            "total-scorer",
            ComponentKind.PARSER,
            (
                Role.CHART_CELL,
                (),
                ("item", "span", "width"),
                DTypeClass.REAL,
            ),
        )


def encoder() -> CodeSumEncoder:
    return CodeSumEncoder()


def leaves_of(items: Sequence[Sequence[str]], rows: int) -> Encoded:
    return MeanPooler(1).pool(CharacterEncoder(rows).encode_words(items))


def test_an_encoder_hands_back_a_vector_per_term_at_its_declared_width() -> None:
    encoded: TermEncoder = encoder()

    terms = encoded.encode(TERMS)

    assert terms.axes == ("term", "width")
    assert terms.size("term") == len(TERMS.rows)
    assert terms.size("width") == encoded.output_width
    assert terms.tensor.squeeze(-1).tolist() == [6.0, 3.0]


def test_an_encoder_hands_back_a_vector_per_node_masked_by_occupancy() -> None:
    encoded: TermEncoder = encoder()

    nodes = encoded.encode_nodes(TERMS)

    assert nodes.axes == ("term", "slot", "width")
    assert nodes.size("width") == encoded.output_width
    assert nodes.mask is not None
    assert nodes.mask.tolist() == [[True, True, True], [True, True, False]]


def test_soft_encoding_a_concentrated_distribution_agrees_with_hard_encoding() -> None:
    encoded: SoftTermEncoder = encoder()

    soft = encoded.encode_soft(one_hot(TERMS), structure=TERMS)
    hard = encoded.encode(TERMS)

    assert soft.axes == hard.axes
    assert torch.allclose(soft.tensor, hard.tensor, atol=1e-6)


def test_soft_encoding_is_a_separate_encoder_capability() -> None:
    hard_only = SimpleNamespace(
        output_width=encoder().output_width,
        encode=encoder().encode,
        encode_nodes=encoder().encode_nodes,
        site_map=encoder().site_map,
    )

    assert isinstance(hard_only, TermEncoder)
    assert not isinstance(hard_only, SoftTermEncoder)
    assert isinstance(encoder(), SoftTermEncoder)


def test_a_teacher_forced_decoder_reports_choice_rows_for_supplied_terms() -> None:
    decoder: TeacherForcedDecoder = SlotRows(TERMS.width, ALPHABET)

    rows = decoder.rows(encoder().encode(TERMS), terms=TERMS)

    assert rows.axes == ("term", "slot", "choice")
    assert rows.normalized_over == "choice"
    assert torch.allclose(
        rows.tensor.exp().sum(dim=-1),
        torch.ones(len(TERMS.rows), TERMS.width, dtype=torch.float64),
    )


def test_a_marginal_decoder_reports_choice_rows_without_supplied_terms() -> None:
    decoder: MarginalDecoder = SlotRows(TERMS.width, ALPHABET)

    rows = decoder.marginal_rows(encoder().encode(TERMS), root_sort="term")

    assert rows.axes == ("term", "slot", "choice")
    assert rows.normalized_over == "choice"
    assert torch.allclose(
        rows.tensor.exp().sum(dim=-1),
        torch.ones(len(TERMS.rows), TERMS.width, dtype=torch.float64),
    )


class HalfScheme:
    @property
    def width(self) -> int:
        return 1

    @property
    def ranks(self) -> tuple[int, ...]:
        return (1,)

    def factor(self, parent: Encoded, /, *, arity: int) -> tuple[Encoded, ...]:
        del arity

        return (parent,)


def test_an_arity_scheme_declares_both_directions() -> None:
    reached = (Operator("atom", "T"), Operator("wrap", "T", arg_sorts=("T",)))

    assert isinstance(RoleFillerScheme(width=2, reached=reached), ArityScheme)
    assert isinstance(ArityHeadScheme(width=2, reached=reached), ArityScheme)
    assert not isinstance(HalfScheme(), ArityScheme)


class Undeclared:
    """An object lacking one frontier operation."""

    def start(self, encoded: Encoded, /) -> torch.Tensor:
        return encoded.tensor

    def weigh(self, state: torch.Tensor, /) -> LogDistribution:
        logits = state.sum() * torch.arange(ALPHABET, dtype=state.dtype)

        return LogDistribution(
            logits.log_softmax(dim=0), ("choice",), normalized_over="choice"
        )


def test_a_frontier_decoder_declares_its_state_operations() -> None:
    decoder: FrontierDecoder[torch.Tensor] = StepwiseSlotDecoder()
    state = decoder.fill(
        decoder.start(Encoded(torch.zeros(1), ("width",))),
        choices=torch.tensor(2),
    )

    assert isinstance(decoder, FrontierDecoder)
    assert not isinstance(SlotRows(TERMS.width, ALPHABET), FrontierDecoder)
    assert torch.equal(state, torch.tensor([2.0]))
    context = HoleContext((), "term", MappingProxyType({}), MappingProxyType({}), 1)
    assert decoder.weigh(state, context=context).axes == ("choice",)


def test_a_frontier_decoder_requires_every_state_operation() -> None:
    assert not isinstance(Undeclared(), FrontierDecoder)


def test_a_decoder_declares_the_structure_it_reads_as_a_mapping() -> None:
    declaration = MappingProxyType({StructuralRelation.POSITION: Consumption.EMBEDDING})

    assert dict(declaration) == {StructuralRelation.POSITION: Consumption.EMBEDDING}


def test_the_relations_come_in_two_kinds_with_one_value_per_slot_or_per_pair() -> None:
    unary = {
        relation
        for relation in StructuralRelation
        if relation.kind is RelationKind.UNARY
    }
    pairwise = set(StructuralRelation) - unary

    assert unary == {
        StructuralRelation.POSITION,
        StructuralRelation.DEPTH,
        StructuralRelation.ARITY,
    }
    assert pairwise == {
        StructuralRelation.PARENT,
        StructuralRelation.ANCESTOR,
        StructuralRelation.SIBLING,
        StructuralRelation.TREE_DISTANCE,
        StructuralRelation.SAME_CONSTITUENT,
        StructuralRelation.SPAN_CONTAINS,
    }
    assert all(relation.kind is RelationKind.PAIRWISE for relation in pairwise)
    assert StructuralRelation("tree_distance") is StructuralRelation.TREE_DISTANCE


def test_a_declaration_is_described_relation_by_relation_in_its_own_order() -> None:
    assert describe_structure({}) == "{}"
    assert (
        describe_structure(
            {
                StructuralRelation.ANCESTOR: Consumption.MASK,
                StructuralRelation.POSITION: Consumption.EMBEDDING,
            }
        )
        == "{ancestor: mask, position: embedding}"
    )


def test_the_standard_declarations_are_the_five_rungs_of_the_ablation() -> None:
    rungs = {
        StandardDeclaration.POSITIONS: {
            StructuralRelation.POSITION: Consumption.EMBEDDING
        },
        StandardDeclaration.POSITIONS_WITH_DEPTH: {
            StructuralRelation.POSITION: Consumption.EMBEDDING,
            StructuralRelation.DEPTH: Consumption.EMBEDDING,
        },
        StandardDeclaration.ANCESTOR_MASK: {
            StructuralRelation.POSITION: Consumption.EMBEDDING,
            StructuralRelation.ANCESTOR: Consumption.MASK,
        },
        StandardDeclaration.ANCESTOR_AND_SIBLING_BIAS: {
            StructuralRelation.POSITION: Consumption.EMBEDDING,
            StructuralRelation.ANCESTOR: Consumption.BIAS,
            StructuralRelation.SIBLING: Consumption.BIAS,
        },
        StandardDeclaration.PARENT_RECURRENCE: {
            StructuralRelation.POSITION: Consumption.EMBEDDING,
            StructuralRelation.PARENT: Consumption.RECURRENCE,
        },
    }

    assert list(rungs) == list(StandardDeclaration)

    for standard, declaration in rungs.items():
        assert dict(standard.declaration) == declaration

        with pytest.raises(TypeError, match="does not support item assignment"):
            cast("dict[StructuralRelation, Consumption]", standard.declaration)[
                StructuralRelation.ARITY
            ] = Consumption.EMBEDDING


def test_a_controller_weighs_one_column_per_action() -> None:
    controller: Controller = GapController()
    cells = Encoded(torch.tensor([[2.0], [1.0]]), ("cell", "width"))

    weights = controller.weigh(cells, right=cells)

    assert weights.axes == ("cell", "action")
    assert controller.input_width == cells.size("width")
    assert weights.size("action") == len(controller.action_set.actions)
    assert weights.tensor[:, 0].tolist() == [0.0, 0.0]
    assert bool((weights.tensor[:, 0] > weights.tensor[:, 1]).all())


def test_a_token_encoder_returns_rows_and_the_words_they_encode() -> None:
    tokens: TokenEncoder = CharacterEncoder(8)

    encoded = tokens.encode_words([["ab", "c"]])

    assert encoded.vectors.size("width") == tokens.output_width
    assert encoded.lengths.tolist() == [3]
    assert encoded.alignments[0].rows == ((0, 2), (2, 3))
    assert encoded.vectors.tensor[0, :3, 0].tolist() == [97.0, 98.0, 99.0]


def test_a_word_that_does_not_fit_the_budget_is_reported_rather_than_shifted() -> None:
    tokens: TokenEncoder = CharacterEncoder(4)

    encoded = tokens.encode_words([["abc", "de", "f"]])

    assert encoded.alignments[0].rows == ((0, 3), None, None)
    assert encoded.truncated == ((1, 2),)
    assert encoded.truncation is not None
    assert encoded.truncation[0].words == ("de", "f")
    assert not encoded.alignments[0].is_total


def test_a_pooler_averages_the_rows_the_alignment_gives_each_word() -> None:
    pooler: SubwordPooler = MeanPooler(1)

    pooled = pooler.pool(CharacterEncoder(8).encode_words([["ab", "c"]]))

    assert pooled.axes == ("item", "word", "width")
    assert pooled.size("width") == pooler.output_width
    assert pooled.tensor[0, :, 0].tolist() == [97.5, 99.0]
    assert pooled.mask is not None
    assert pooled.mask.tolist() == [[True, True]]


def test_a_pooler_masks_off_a_word_no_row_encoded() -> None:
    pooler: SubwordPooler = MeanPooler(1)

    pooled = pooler.pool(CharacterEncoder(4).encode_words([["abc", "de", "f"]]))

    assert pooled.mask is not None
    assert pooled.mask.tolist() == [[True, False, False]]


def test_a_scorer_fills_the_readouts_it_publishes_and_leaves_the_rest_absent() -> None:
    scorer: ChartScorer = TotalScorer()
    leaves = leaves_of([["ab", "c"]], 8)

    published = scorer.readouts(leaves)

    assert scorer.publishes == frozenset({"cells", "log_z"})
    assert published.cells is not None
    assert published.cells.axes == ("item", "span", "width")
    assert torch.equal(published.cells.tensor, leaves.tensor)
    assert published.log_z is not None
    assert published.log_z.axes == ("item",)
    assert published.inside is None
    assert published.alpha is None
    assert published.mu is None
    assert published.actions is None


def test_a_scorer_leaves_a_padded_word_out_of_its_total() -> None:
    scorer: ChartScorer = TotalScorer()

    padded = scorer.readouts(leaves_of([["abc", "de", "f"]], 4))
    live = scorer.readouts(leaves_of([["abc"]], 4))

    assert padded.log_z is not None
    assert live.log_z is not None
    assert torch.allclose(padded.log_z.tensor, live.log_z.tensor)


def test_every_port_publishes_the_points_an_observer_may_name() -> None:
    subjects: tuple[TermEncoder | Controller | TokenEncoder | SubwordPooler, ...] = (
        encoder(),
        GapController(),
        CharacterEncoder(8),
        MeanPooler(1),
    )
    published = [subject.site_map() for subject in (*subjects, TotalScorer())]

    assert all(published)

    for one in published:
        for site, spec in one.items():
            assert isinstance(site, Site)
            assert spec.rank == len(spec.axes)
            assert Access.OBSERVE in spec.access


def test_a_published_point_names_the_axes_the_component_actually_returns() -> None:
    controller: Controller = GapController()
    cells = Encoded(torch.tensor([[2.0], [1.0]]), ("cell", "width"))
    logits = Site(ComponentId("gap-controller"), Role.COMBINATOR_LOGITS)

    spec = controller.site_map()[logits]
    weights = controller.weigh(cells, right=cells)

    assert spec.axes == weights.axes
    assert weights.tensor.shape[spec.position("action")] == len(
        controller.action_set.actions
    )
    assert Site(ComponentId("gap-controller"), Role.CHART_CELL) not in (
        controller.site_map()
    )


def test_independent_decoder_capabilities_can_overlap_on_one_component() -> None:
    rows = SlotRows(TERMS.width, ALPHABET)

    assert isinstance(rows, TeacherForcedDecoder)
    assert isinstance(rows, MarginalDecoder)
    assert not isinstance(rows, FrontierDecoder)


def test_non_decoder_shapes_answer_to_one_port_and_no_other() -> None:
    ports = (
        TermEncoder,
        Controller,
        TokenEncoder,
        SubwordPooler,
        ChartScorer,
    )
    subjects = (
        encoder(),
        GapController(),
        CharacterEncoder(8),
        MeanPooler(1),
        TotalScorer(),
    )

    for subject, port in zip(subjects, ports, strict=True):
        assert [held for held in ports if isinstance(subject, held)] == [port]


def test_a_shape_missing_a_declared_member_answers_to_no_port() -> None:
    built = encoder()
    half_an_encoder = SimpleNamespace(output_width=1, encode=built.encode)
    assert not isinstance(half_an_encoder, TermEncoder)
    assert isinstance(built, TermEncoder)

    rows = SlotRows(TERMS.width, ALPHABET)
    assert not isinstance(SimpleNamespace(), TeacherForcedDecoder)
    assert not isinstance(SimpleNamespace(), MarginalDecoder)
    assert isinstance(SimpleNamespace(rows=rows.rows), TeacherForcedDecoder)
    assert isinstance(
        SimpleNamespace(marginal_rows=rows.marginal_rows), MarginalDecoder
    )


def a_pass(
    mode: ForwardMode, rate: float, temperature: float, *, flag: bool
) -> ForwardOptions:
    plain = ForwardOptions(mode)

    if not plain.declares("sampling_rate"):
        return plain

    return ForwardOptions(
        mode,
        sampling_rate=rate,
        relaxation=Relaxation(gumbel_temperature=temperature, straight_through=flag),
    )


def test_the_forward_modes_are_three_and_nothing_else() -> None:
    assert {mode.name for mode in ForwardMode} == {"OBSERVE", "SAMPLE", "MARGINALIZE"}

    with pytest.raises(ValueError, match="not a valid ForwardMode"):
        ForwardMode("sample")

    with pytest.raises(TypeError, match="runs in a ForwardMode"):
        ForwardOptions(cast("ForwardMode", "sample"))


def test_a_pass_takes_only_the_settings_its_mode_declares() -> None:
    drawn = ForwardOptions(
        ForwardMode.SAMPLE,
        sampling_rate=0.25,
        relaxation=Relaxation(gumbel_temperature=0.5, straight_through=True),
    )

    assert drawn.sampling_rate == 0.25
    assert drawn.relaxation == Relaxation(gumbel_temperature=0.5, straight_through=True)

    for barren in (ForwardMode.OBSERVE, ForwardMode.MARGINALIZE):
        with pytest.raises(ValueError, match="declares no option"):
            ForwardOptions(barren, sampling_rate=0.5)

        with pytest.raises(ValueError, match="declares no option"):
            ForwardOptions(
                barren,
                relaxation=Relaxation(gumbel_temperature=1.0, straight_through=True),
            )

        with pytest.raises(ValueError, match="declares no option"):
            ForwardOptions(
                barren, sampling=SamplingStrategy(KEEP_OFFERED, temperature=2.0)
            )


def test_reading_a_setting_a_mode_does_not_declare_is_refused() -> None:
    watched = ForwardOptions(ForwardMode.OBSERVE)

    assert not watched.declares("sampling_rate")
    assert ForwardOptions(ForwardMode.SAMPLE).declares("relaxation")

    with pytest.raises(ValueError, match="OBSERVE declares no option"):
        _ = watched.sampling_rate

    with pytest.raises(ValueError, match="OBSERVE declares no option"):
        _ = watched.relaxation

    with pytest.raises(ValueError, match="OBSERVE declares no option"):
        _ = watched.sampling


def test_a_sampling_pass_left_open_takes_its_mode_s_own_defaults() -> None:
    drawn = ForwardOptions(ForwardMode.SAMPLE)

    assert drawn.sampling_rate == 1.0
    assert drawn.sampling == SamplingStrategy(KEEP_OFFERED)
    assert drawn.relaxation == Relaxation(
        gumbel_temperature=1.0, straight_through=False
    )


def test_a_rate_and_a_temperature_are_checked_where_they_are_named() -> None:
    with pytest.raises(ValueError, match=r"lies in \(0, 1\]"):
        ForwardOptions(ForwardMode.SAMPLE, sampling_rate=0.0)

    with pytest.raises(ValueError, match=r"lies in \(0, 1\]"):
        ForwardOptions(ForwardMode.SAMPLE, sampling_rate=1.5)

    with pytest.raises(ValueError, match="is above zero"):
        ForwardOptions(
            ForwardMode.SAMPLE,
            relaxation=Relaxation(gumbel_temperature=0.0, straight_through=False),
        )

    with pytest.raises(ValueError, match="not finite"):
        ForwardOptions(
            ForwardMode.SAMPLE,
            relaxation=Relaxation(
                gumbel_temperature=float("nan"), straight_through=False
            ),
        )

    with pytest.raises(TypeError, match="not a number"):
        ForwardOptions(ForwardMode.SAMPLE, sampling_rate=cast("float", "half"))

    with pytest.raises(TypeError, match="flag is a boolean"):
        ForwardOptions(
            ForwardMode.SAMPLE,
            relaxation=Relaxation(
                gumbel_temperature=1.0, straight_through=cast("bool", 1)
            ),
        )


def test_two_passes_describing_the_same_run_are_one_value() -> None:
    drawn = ForwardOptions(ForwardMode.SAMPLE, sampling_rate=0.5)
    same = ForwardOptions(ForwardMode.SAMPLE, sampling_rate=0.5)

    assert drawn == same
    assert hash(drawn) == hash(same)
    assert drawn != ForwardOptions(ForwardMode.SAMPLE, sampling_rate=0.25)
    assert drawn != ForwardOptions(ForwardMode.MARGINALIZE)
    assert drawn.__eq__(ForwardMode.SAMPLE) is NotImplemented
    assert repr(ForwardOptions(ForwardMode.OBSERVE)) == (
        "ForwardOptions(ForwardMode.OBSERVE)"
    )
    assert "sampling_rate=0.5" in repr(drawn)

    for field in ("mode", "sampling_rate"):
        with pytest.raises(AttributeError):
            setattr(drawn, field, None)


def test_a_declared_mode_passes_the_refusal_untouched() -> None:
    watched = ForwardOptions(ForwardMode.OBSERVE)

    assert ensure_mode(watched, frozenset(ForwardMode)) is watched

    with pytest.raises(ValueError, match="runs in no mode"):
        ensure_mode(watched, frozenset())


def test_a_controller_reads_its_columns_off_the_action_set_it_was_built_with() -> None:
    wider = ActionSet((JOIN, KEEP, Action("split", "division")), KEEP)
    controller = GapController(wider)

    assert controller.action_set is wider
    assert controller.action_set.identity in controller.action_set.actions
    assert controller.action_set.index("split") == 2
    assert GapController().action_set.actions == (JOIN, KEEP)


WATCHED = ComponentId("watched-pooler")
POOLED_AXES = ("item", "word", "width")


class WatchScope:
    def __init__(
        self,
        component: WatchedPooler,
        sites: frozenset[Site],
        patches: Mapping[Site, Patch[Observation]],
    ) -> None:
        self._component = component
        self._sites = sites
        self._patches = patches
        self._readings: dict[Site, Observation] = {}

    @property
    def readings(self) -> Mapping[Site, Observation]:
        return MappingProxyType(dict(self._readings))

    def __enter__(self) -> Self:
        self._component.scope = self

        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        del kind, error, traceback

        self._component.scope = None

    def visit(self, site: Site, pooled: Encoded) -> Encoded:
        reading = Observation(site, pooled.tensor, axes=pooled.axes, mask=pooled.mask)

        if site in self._sites:
            self._readings[site] = reading

        patch = self._patches.get(site)

        if patch is None:
            return pooled

        return Encoded(patch.stands_in_for(reading), pooled.axes, pooled.mask)


class WatchedPooler:
    def __init__(self, width: int) -> None:
        self._width = width
        self.scope: WatchScope | None = None

    @property
    def output_width(self) -> int:
        return self._width

    def pool(self, encoding: TokenEncoding, /) -> Encoded:
        pooled = MeanPooler(self._width).pool(encoding)
        scope = self.scope

        if scope is None:
            return pooled

        return scope.visit(Site(WATCHED, Role.WORD_VECTOR), pooled)

    def site_map(self) -> SiteMap:
        return observable(
            "watched-pooler",
            ComponentKind.SUBWORD_POOLER,
            (Role.WORD_VECTOR, (), POOLED_AXES, DTypeClass.REAL),
        )

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[Observation]]
    ) -> WatchScope:
        published = frozenset(self.site_map())
        unknown = (sites | frozenset(patches)) - published

        if unknown:
            msg = f"this pooler publishes {sorted(map(repr, published))}, not {unknown}"
            raise ValueError(msg)

        return WatchScope(self, sites, patches)


def watched() -> tuple[WatchedPooler, Site, TokenEncoding]:
    return (
        WatchedPooler(1),
        Site(WATCHED, Role.WORD_VECTOR),
        CharacterEncoder(8).encode_words([["ab", "c"]]),
    )


def test_every_slot_is_observable_and_the_real_components_are_too() -> None:
    subjects = (
        encoder(),
        SlotRows(TERMS.width, ALPHABET),
        GapController(),
        CharacterEncoder(8),
        MeanPooler(1),
        TotalScorer(),
        MeanSubwordPooler(1, component=ComponentId("mean-pooler")),
    )

    for subject in subjects:
        assert isinstance(subject, Observable)
        assert isinstance(subject.site_map(), Mapping)

    assert not isinstance(SimpleNamespace(output_width=1), Observable)
    assert not isinstance(MeanPooler(1), Instrumentable)


def test_the_probe_layers_own_values_answer_to_the_reading_and_patch_shapes() -> None:
    site = Site(WATCHED, Role.WORD_VECTOR)
    reading = Observation(site, torch.ones(2, 3), axes=("word", "width"))

    assert isinstance(reading, Reading)
    assert reading.site == site
    assert not isinstance(reading, Patch)

    for patch in (Zero(), Noise(0.5, 3), Shuffle(5)):
        assert isinstance(patch, Patch)
        assert not isinstance(patch, Reading)
        assert patch.stands_in_for(reading).shape == reading.tensor.shape

    assert torch.equal(Zero().stands_in_for(reading), torch.zeros(2, 3))


def test_a_receiver_edge_exposes_one_immutable_public_axis_coordinate() -> None:
    edge = ReceiverEdge(4, 2, 7, 1, ReceiverSide.RIGHT)

    assert edge.position == 4
    assert edge.source == 2
    assert edge.target == 7
    assert edge.split == 1
    assert edge.side is ReceiverSide.RIGHT
    assert dict(edge.coordinates) == {"edge": 4}

    with pytest.raises(TypeError):
        cast("dict[str, int]", edge.coordinates)["edge"] = 5


def test_a_scope_reads_the_points_it_was_opened_over_and_leaves_nothing_behind() -> (
    None
):
    pooler, site, encoding = watched()
    before = pooler.pool(encoding).tensor.clone()
    scope = pooler.instrument(frozenset({site}), patches={})

    assert isinstance(scope, InstrumentScope)
    assert isinstance(pooler, Instrumentable)
    assert isinstance(pooler, SubwordPooler)
    assert not scope.readings

    with scope as opened:
        pooler.pool(encoding)

        assert opened is scope

    assert set(scope.readings) == {site}
    assert scope.readings[site].axes == pooler.site_map()[site].axes
    assert torch.equal(scope.readings[site].tensor, before)
    assert torch.equal(pooler.pool(encoding).tensor, before)
    assert pooler.scope is None


def test_a_patch_stands_in_for_a_point_only_for_the_length_of_the_scope() -> None:
    pooler, site, encoding = watched()
    before = pooler.pool(encoding).tensor.clone()

    with pooler.instrument(frozenset(), patches={site: Zero()}):
        inside = pooler.pool(encoding).tensor.clone()

    assert bool(before.abs().sum() > 0)
    assert torch.equal(inside, torch.zeros_like(before))
    assert torch.equal(pooler.pool(encoding).tensor, before)


def fail_inside(pooler: WatchedPooler, site: Site, encoding: TokenEncoding) -> None:
    with pooler.instrument(frozenset({site}), patches={site: Zero()}):
        pooler.pool(encoding)
        msg = "a pass failed"

        raise RuntimeError(msg)


def test_a_scope_restores_the_component_when_the_pass_inside_it_raises() -> None:
    pooler, site, encoding = watched()
    before = pooler.pool(encoding).tensor.clone()

    with pytest.raises(RuntimeError, match="a pass failed"):
        fail_inside(pooler, site, encoding)

    assert pooler.scope is None
    assert torch.equal(pooler.pool(encoding).tensor, before)


def test_a_scope_over_a_point_the_component_does_not_publish_is_refused() -> None:
    pooler, _, _ = watched()
    elsewhere = Site(WATCHED, Role.CHART_CELL)

    with pytest.raises(ValueError, match="this pooler publishes"):
        pooler.instrument(frozenset({elsewhere}), patches={})

    with pytest.raises(ValueError, match="this pooler publishes"):
        pooler.instrument(frozenset(), patches={elsewhere: Zero()})
