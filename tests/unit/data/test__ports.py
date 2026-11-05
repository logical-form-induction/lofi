import json
from typing import TYPE_CHECKING, NamedTuple

import torch
from lairs.integrations.codecs import (
    BratCodec,
    BratIso,
    ConlluCodec,
    ConlluIso,
    CorpusFragment,
    FragmentRecord,
)

from lofi._types import ItemId
from lofi.data._batch import Alignment, Batch, Mask
from lofi.data._constraint import SpanConstraint, SpanConstraints
from lofi.data._constraint_sources import ConstraintSelection
from lofi.data._hashed import ItemHasher
from lofi.data._inventory import Inventory
from lofi.data._join import ItemLookup
from lofi.data._ports import (
    AmbiguityRule,
    Codec,
    ConstraintSource,
    CorpusReader,
    Featurizer,
    FeaturizerComponent,
    FeaturizerResources,
    InventoriedTypeSampler,
    Iso,
    IsoCodec,
    LabelMembership,
    PairSampler,
    Source,
    StepRecovery,
    TypeInventory,
    TypeSampler,
)
from lofi.data._report import LoadReport
from lofi.data._sampling import PairDistribution, PairSampleReport
from lofi.data._selection import (
    RankBand,
    ScopeKind,
    SelectionPolicy,
    SelectionReport,
    UnionRule,
)
from lofi.data._supervision import Example, Supervision, SupervisionArm, Unlabeled
from lofi.data._values import Item, Token, TokenSpan
from lofi.data.synthetic import RecursiveTypeSampler
from lofi.terms import Operator, Signature, TermStore

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

type Line = tuple[str, str, str]


class EmptyPairSampler:
    """A test double with the pair-sampler call shape."""

    def sample(
        self,
        count: int,
        distribution: PairDistribution,
        /,
        *,
        seed: int,
    ) -> tuple[tuple[object, ...], PairSampleReport]:
        del distribution, seed
        del count
        return (), PairSampleReport(0, {})


def test_synthetic_samplers_satisfy_the_structural_ports() -> None:
    signature = Signature("port-sampling", ("Cat",), (Operator("n", "Cat"),))
    sampler = RecursiveTypeSampler(TermStore(signature), max_size=7, sort="Cat")

    assert isinstance(sampler, TypeSampler)
    assert isinstance(sampler, InventoriedTypeSampler)
    assert isinstance(sampler.inventory, TypeInventory)
    assert isinstance(EmptyPairSampler(), PairSampler)


def fragment_of(lines: tuple[Line, ...], source: str) -> CorpusFragment:
    records = tuple(
        FragmentRecord(local_id=local_id, nsid=nsid, value_json=value_json)
        for local_id, nsid, value_json in lines
    )

    return CorpusFragment(records=records, source=source)


class LineIso:
    def forward(self, source: tuple[Line, ...], /) -> CorpusFragment:
        return fragment_of(source, "line")

    def backward(self, target: CorpusFragment, /) -> tuple[Line, ...]:
        return tuple(
            (record.local_id, record.nsid, record.value_json)
            for record in target.records
        )


class LineCodec:
    @property
    def name(self) -> str:
        return "line"

    @property
    def iso(self) -> Iso[tuple[Line, ...], CorpusFragment]:
        return LineIso()

    def decode(
        self,
        src: str | bytes,
        *,
        into: CorpusFragment | None = None,
    ) -> CorpusFragment:
        text = src.decode() if isinstance(src, bytes) else src
        lines = tuple(
            (local_id, nsid, value_json)
            for local_id, nsid, value_json in (
                line.split("\t") for line in text.splitlines() if line
            )
        )
        already = () if into is None else self.iso.backward(into)

        return fragment_of(already + lines, self.name)

    def encode(self, records: Iterable[FragmentRecord]) -> str:
        return "".join(
            f"{record.local_id}\t{record.nsid}\t{record.value_json}\n"
            for record in records
        )


def test_a_codec_the_dataset_layer_registers_satisfies_the_bound_port() -> None:
    assert isinstance(BratCodec(), Codec)
    assert isinstance(ConlluCodec(), Codec)


def test_a_translation_the_dataset_layer_ships_satisfies_the_declared_one() -> None:
    assert isinstance(BratIso(), Iso)
    assert isinstance(ConlluIso(), Iso)


def test_a_codec_publishing_its_translation_satisfies_the_narrower_port() -> None:
    codec = LineCodec()

    assert isinstance(codec, Codec)
    assert isinstance(codec, IsoCodec)


def test_a_codec_publishing_no_translation_fails_the_narrower_port() -> None:
    assert not isinstance(BratCodec(), IsoCodec)
    assert not isinstance(ConlluCodec(), IsoCodec)


def test_neither_port_admits_a_value_that_merely_decodes_or_encodes() -> None:
    assert not isinstance(json.JSONDecoder(), Codec)
    assert not isinstance(json.JSONDecoder(), Iso)
    assert not isinstance(json.JSONEncoder(), Codec)
    assert not isinstance(json.JSONEncoder(), Iso)


def test_a_translation_and_a_codec_are_told_apart_by_shape() -> None:
    assert not isinstance(LineIso(), Codec)
    assert not isinstance(LineCodec(), Iso)


def test_a_codec_reads_records_out_of_its_own_external_form() -> None:
    codec = LineCodec()
    decoded = codec.decode('e1\tpub.layers.expression\t{"text": "rain"}\n')

    assert len(decoded.records) == 1
    assert decoded.records[0].local_id == "e1"
    assert decoded.records[0].nsid == "pub.layers.expression"
    assert json.loads(decoded.records[0].value_json) == {"text": "rain"}
    assert decoded.source == codec.name


def test_a_record_travels_as_json_text_rather_than_as_a_generated_class() -> None:
    codec = LineCodec()
    value = {"text": "rain", "tokens": [{"index": 0, "text": "rain"}]}
    written = codec.encode(
        (
            FragmentRecord(
                local_id="e1",
                nsid="pub.layers.expression",
                value_json=json.dumps(value),
            ),
        )
    )

    assert json.loads(codec.decode(written).records[0].value_json) == value


def test_decoding_into_a_fragment_puts_its_records_first() -> None:
    codec = LineCodec()
    first = codec.decode("e1\tpub.layers.expression\t{}\n")
    both = codec.decode("s1\tpub.layers.segmentation\t{}\n", into=first)

    assert [record.local_id for record in both.records] == ["e1", "s1"]
    assert [record.local_id for record in first.records] == ["e1"]


def test_a_codec_translates_between_the_parsed_shape_and_the_fragment() -> None:
    codec = LineCodec()
    lines: tuple[Line, ...] = (("e1", "pub.layers.expression", "{}"),)
    fragment = codec.iso.forward(lines)

    assert codec.iso.backward(fragment) == lines
    assert fragment == codec.decode(codec.encode(fragment.records))


ITEM = Item(ItemId("s/1"), (Token(0, "rain"),), split="train")


class OneItemReader:
    name = "one-item"
    grammar = None
    lexicon = None

    def splits(self) -> tuple[str, ...]:
        return ("train",)

    def items(self, split: str, /) -> Source[Item]:
        return (ITEM,) if split == "train" else ()

    def examples(self, split: str, /) -> Source[Example[Supervision[str]]]:
        return tuple(Example(item, Unlabeled()) for item in self.items(split))

    def supervision_kinds(self) -> frozenset[SupervisionArm]:
        return frozenset({Unlabeled})

    def report(self) -> LoadReport:
        return LoadReport(1, 1)


class PaddingFeaturizer:
    accepts: frozenset[SupervisionArm] = frozenset({Unlabeled})
    selection: SupervisionArm = Unlabeled

    def __call__(self, examples: Sequence[Example[Supervision[str]]], /) -> Batch:
        lengths = tuple(len(example.item.tokens) for example in examples)
        width = max(lengths, default=0)
        rows = torch.zeros(len(examples), width, 1)
        present = torch.arange(width).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)

        return Batch(
            rows,
            lengths,
            mask=Mask(present),
            alignments=tuple(
                Alignment(tuple((index, index + 1) for index in range(length)))
                for length in lengths
            ),
            item_ids=tuple(example.item.item_id for example in examples),
        )


class Tensorizer:
    """Tensorizes the resource-port fixture."""

    def __call__(
        self,
        rows: Sequence[tuple[int, ...]],
        /,
        *,
        padding: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = max((len(row) for row in rows), default=0)
        values = torch.full((len(rows), width), padding, dtype=torch.long)
        present = torch.zeros((len(rows), width), dtype=torch.bool)

        for index, row in enumerate(rows):
            if row:
                values[index, : len(row)] = torch.tensor(row)
                present[index, : len(row)] = True

        return values, present


class Resources(NamedTuple):
    """Dataset resources for one featurizer."""

    hasher: ItemHasher
    tensorizer: Tensorizer
    lookup: ItemLookup


class BareCall:
    def __call__(
        self,
        examples: Sequence[Example[Supervision[str]]],
        /,
    ) -> Sequence[Example[Supervision[str]]]:
        return examples


class ArithmeticRecovery:
    """A small observed-step recovery relation."""

    def recover(
        self,
        inputs: Sequence[int],
        output: int,
        /,
    ) -> tuple[str, ...]:
        """Return the labels whose arithmetic relation matches."""
        return ("sum", "combined") if sum(inputs) == output else ()


class SmallLabels:
    """A small integer label inventory."""

    def contains(self, label: int, /) -> bool:
        """Return whether the label is in the test inventory."""
        return label in {1, 2, 3}


def test_a_re_readable_source_satisfies_the_port_and_a_generator_does_not() -> None:
    assert isinstance((ITEM,), Source)
    assert isinstance([ITEM], Source)
    assert not isinstance((item for item in (ITEM,)), Source)
    assert not isinstance(iter((ITEM,)), Source)


def test_a_step_recovery_returns_stable_labels_for_one_observed_step() -> None:
    recovery = ArithmeticRecovery()
    inputs = [1, 2]

    assert isinstance(recovery, StepRecovery)
    assert recovery.recover(inputs, 3) == ("sum", "combined")
    assert recovery.recover(inputs, 4) == ()
    assert inputs == [1, 2]


def test_a_label_inventory_satisfies_the_membership_port() -> None:
    labels = SmallLabels()

    assert isinstance(labels, LabelMembership)
    assert labels.contains(2)
    assert not labels.contains(4)
    assert not isinstance(ArithmeticRecovery(), LabelMembership)


def test_a_reader_declaring_the_eight_members_satisfies_the_port() -> None:
    reader = OneItemReader()

    assert isinstance(reader, CorpusReader)
    assert reader.name == "one-item"
    assert reader.grammar is None
    assert reader.lexicon is None
    assert reader.splits() == ("train",)
    assert tuple(reader.items("train")) == (ITEM,)
    assert next(iter(reader.examples("train"))).supervision == Unlabeled()
    assert reader.supervision_kinds() == frozenset({Unlabeled})
    assert reader.report().emitted == 1


def test_the_reader_port_refuses_a_value_that_merely_reads_something() -> None:
    assert not isinstance(json.JSONDecoder(), CorpusReader)
    assert not isinstance(LineCodec(), CorpusReader)
    assert not isinstance(BareCall(), CorpusReader)
    assert not isinstance(PaddingFeaturizer(), CorpusReader)


def test_a_featurizer_declares_its_arms_and_produces_a_batch() -> None:
    featurizer = PaddingFeaturizer()
    batch = featurizer([Example(ITEM, Unlabeled())])

    assert isinstance(featurizer, Featurizer)
    assert featurizer.accepts == frozenset({Unlabeled})
    assert len(batch) == 1
    assert batch.lengths == (1,)
    assert batch.mask is not None
    assert bool(batch.mask.values[0, 0]) is True
    assert batch.item_ids == (ITEM.item_id,)


def test_the_featurizer_port_refuses_anything_that_merely_has_a_call() -> None:
    assert not isinstance(BareCall(), Featurizer)
    assert not isinstance(OneItemReader(), Featurizer)
    assert not isinstance(torch.nn.Linear(1, 1), Featurizer)


def test_dataset_resources_and_featurizers_are_distinct_wiring_components() -> None:
    resources = Resources(
        ItemHasher(
            Inventory("words", ("rain",), ("<pad>",)),
            Inventory("categories", ()),
        ),
        Tensorizer(),
        ItemLookup((ITEM,)),
    )
    featurizer = PaddingFeaturizer()

    assert isinstance(resources, FeaturizerResources)
    assert not isinstance(resources, FeaturizerComponent)
    assert isinstance(featurizer, FeaturizerComponent)
    assert not isinstance(featurizer, FeaturizerResources)


def test_a_reader_and_a_featurizer_agree_on_one_arm_before_a_run_is_wired() -> None:
    reader = OneItemReader()
    featurizer = PaddingFeaturizer()

    assert reader.supervision_kinds() & featurizer.accepts


class FirstRule:
    """Resolves a form to the first type observed, at the form level."""

    @property
    def scope(self) -> ScopeKind:
        return ScopeKind.FORM

    def permitted(self, masses: Mapping[str, float], /) -> frozenset[str]:
        return frozenset({next(iter(masses))})


class WholeItemSource:
    """Constrains every item to be a sentence over its whole span."""

    @property
    def scope(self) -> ScopeKind:
        return ScopeKind.OCCURRENCE

    @property
    def policy(self) -> SelectionPolicy[str, str]:
        return SelectionPolicy(UnionRule())

    def select(
        self, examples: Iterable[Example[Supervision[str]]], /
    ) -> ConstraintSelection[str, str]:
        held = tuple(examples)
        emitted = tuple(
            Example(
                example.item,
                SpanConstraints(
                    (
                        SpanConstraint(
                            TokenSpan(0, len(example.item.tokens)), frozenset({"S"})
                        ),
                    ),
                    len(example.item.tokens),
                ),
            )
            for example in held
        )

        return ConstraintSelection(
            emitted,
            LoadReport(len(held), len(emitted)),
            SelectionReport(1, 1, 1.0, {}, selector=RankBand(0.0, 1.0)),
        )


def test_a_rule_declaring_a_scope_and_a_resolution_satisfies_the_port() -> None:
    rule = FirstRule()

    assert isinstance(rule, AmbiguityRule)
    assert rule.permitted({"N": 1.0, "NP": 2.0}) == {"N"}
    assert not isinstance(BareCall(), AmbiguityRule)
    assert not isinstance(WholeItemSource(), AmbiguityRule)


def test_a_source_declaring_a_scope_a_policy_and_a_selection_satisfies_the_port() -> (
    None
):
    source = WholeItemSource()
    selection = source.select([Example(ITEM, Unlabeled())])
    emitted, report = selection.examples, selection.report

    assert isinstance(source, ConstraintSource)
    assert emitted[0].supervision.constraints[0].span == TokenSpan(0, 1)
    assert report == LoadReport(1, 1)
    assert not isinstance(OneItemReader(), ConstraintSource)
    assert not isinstance(FirstRule(), ConstraintSource)
