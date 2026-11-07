from typing import TYPE_CHECKING, ClassVar, Final, override

import pytest
from hypothesis import strategies as st
from lairs.integrations.codecs import CorpusFragment, FragmentRecord

from lofi._checks import LawViolationError
from lofi._types import ItemId
from lofi.data import (
    Alphabet,
    ConstraintSelection,
    ConstraintSource,
    CorpusReader,
    DeclaredSource,
    HashedItem,
    Inventory,
    Item,
    ItemHasher,
    PerOccurrenceRule,
    Scalar,
    Scale,
    ScopeKind,
    SelectionPolicy,
    Source,
    SpanConstraint,
    SpanConstraints,
    SupervisionArm,
    Token,
    TokenSpan,
    UnionRule,
    Unlabeled,
)
from lofi.data._report import LoadReport
from lofi.data._supervision import Example, Supervision
from lofi.data.laws import (
    CODEC_LAWS,
    CONSTRAINT_SOURCE_LAWS,
    ISO_LAWS,
    ITEM_LAWS,
    READER_LAWS,
    CodecFixture,
    ConstraintSourceFixture,
    IsoFixture,
    ItemFixture,
    ReaderFixture,
    a_missing_annotation_is_a_typed_absent_case,
    a_reserved_symbol_keeps_its_index_as_the_inventory_grows,
    a_type_level_constraint_binds_every_occurrence,
    ambiguity_is_preserved_and_not_resolved,
    check_codec_laws,
    check_constraint_source_laws,
    check_iso_laws,
    check_item_laws,
    check_reader_laws,
    decoding_does_not_mutate,
    emission_is_deterministic_under_a_seed,
    full_coverage_yields_every_constituent,
    ids_are_distinct_under_a_repeated_surface,
    the_inventory_does_not_grow_while_a_corpus_is_read,
    the_report_is_total,
    the_scale_rides_on_the_response,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

type Line = tuple[str, str, str]

EXAMPLES: Final = 25

NAMES: Final = st.text(alphabet="abc", min_size=1, max_size=4)

VALUES: Final = st.sampled_from(('{"a": 1}', '{"b": [2]}', "{}"))


def lines() -> st.SearchStrategy[Line]:
    return st.tuples(NAMES, NAMES, VALUES)


def sources() -> st.SearchStrategy[str]:
    return st.lists(lines(), max_size=3).map(
        lambda drawn: "".join(f"{one}\t{two}\t{three}\n" for one, two, three in drawn)
    )


def parsed(src: str | bytes) -> tuple[Line, ...]:
    text = src.decode() if isinstance(src, bytes) else src

    return tuple(
        (local_id, nsid, value_json)
        for local_id, nsid, value_json in (
            line.split("\t") for line in text.splitlines() if line
        )
    )


def written(records: Iterable[Line]) -> str:
    return "".join(f"{one}\t{two}\t{three}\n" for one, two, three in records)


def fragment_of(drawn: tuple[Line, ...]) -> CorpusFragment:
    records = tuple(
        FragmentRecord(local_id=local_id, nsid=nsid, value_json=value_json)
        for local_id, nsid, value_json in drawn
    )

    return CorpusFragment(records=records, source="line")


def records_of(fragment: CorpusFragment) -> tuple[FragmentRecord, ...]:
    return fragment.records


def lines_of(fragment: CorpusFragment) -> tuple[Line, ...]:
    return tuple(
        (record.local_id, record.nsid, record.value_json) for record in fragment.records
    )


def fragments() -> st.SearchStrategy[CorpusFragment]:
    return st.lists(lines(), max_size=3).map(tuple).map(fragment_of)


class LineIso:
    def forward(self, source: tuple[Line, ...], /) -> CorpusFragment:
        return fragment_of(source)

    def backward(self, target: CorpusFragment, /) -> tuple[Line, ...]:
        return lines_of(target)


class ForgetfulIso(LineIso):
    @override
    def backward(self, target: CorpusFragment, /) -> tuple[Line, ...]:
        return tuple(
            (local_id, "", value_json) for local_id, _, value_json in lines_of(target)
        )


class LineCodec:
    @property
    def name(self) -> str:
        return "line"

    def decode(
        self,
        src: str | bytes,
        *,
        into: CorpusFragment | None = None,
    ) -> CorpusFragment:
        already = () if into is None else lines_of(into)

        return fragment_of(already + parsed(src))

    def encode(self, records: Iterable[FragmentRecord]) -> str:
        return written(
            (record.local_id, record.nsid, record.value_json) for record in records
        )


class LosingCodec(LineCodec):
    @override
    def encode(self, records: Iterable[FragmentRecord]) -> str:
        return written((record.local_id, record.nsid, "") for record in records)


class ReplacingCodec(LineCodec):
    @override
    def decode(
        self,
        src: str | bytes,
        *,
        into: CorpusFragment | None = None,
    ) -> CorpusFragment:
        return fragment_of(parsed(src))


class SilentCodec(LineCodec):
    @override
    def decode(
        self,
        src: str | bytes,
        *,
        into: CorpusFragment | None = None,
    ) -> CorpusFragment:
        return fragment_of(())


class Records:
    def __init__(self, records: tuple[Line, ...]) -> None:
        self.records = records


class MutatingCodec:
    @property
    def name(self) -> str:
        return "mutating"

    def decode(self, src: str | bytes, *, into: Records | None = None) -> Records:
        if into is None:
            return Records(parsed(src))

        into.records = into.records + parsed(src)

        return Records(into.records)

    def encode(self, records: Iterable[Line]) -> str:
        return written(records)


def iso_fixture(iso: LineIso) -> IsoFixture[tuple[Line, ...], CorpusFragment]:
    return IsoFixture(
        iso,
        st.lists(lines(), min_size=1, max_size=3).map(tuple),
        fragments().filter(lambda fragment: bool(fragment.records)),
        EXAMPLES,
    )


def codec_fixture(codec: LineCodec) -> CodecFixture[CorpusFragment, FragmentRecord]:
    return CodecFixture(codec, sources(), records_of, EXAMPLES)


type IsoLaw = Callable[[IsoFixture[tuple[Line, ...], CorpusFragment]], None]

type CodecLaw = Callable[[CodecFixture[CorpusFragment, FragmentRecord]], None]

ISO_LAW_NAMES: Final = tuple(law.__name__ for law in ISO_LAWS)

CODEC_LAW_NAMES: Final = tuple(law.__name__ for law in CODEC_LAWS)


@pytest.mark.parametrize("law", ISO_LAWS, ids=ISO_LAW_NAMES)
def test_a_faithful_translation_holds_to_every_law_of_its_own(law: IsoLaw) -> None:
    law(iso_fixture(LineIso()))


@pytest.mark.parametrize("law", ISO_LAWS, ids=ISO_LAW_NAMES)
def test_a_translation_that_drops_a_field_fails_every_law(law: IsoLaw) -> None:
    with pytest.raises(LawViolationError, match="inverts"):
        law(iso_fixture(ForgetfulIso()))


def test_the_aggregate_runs_the_translation_laws_in_one_call() -> None:
    check_iso_laws(iso_fixture(LineIso()))

    with pytest.raises(LawViolationError, match="inverts"):
        check_iso_laws(iso_fixture(ForgetfulIso()))


@pytest.mark.parametrize("law", CODEC_LAWS, ids=CODEC_LAW_NAMES)
def test_a_faithful_codec_holds_to_every_law_of_its_own(law: CodecLaw) -> None:
    law(codec_fixture(LineCodec()))


def test_the_aggregate_runs_the_codec_laws_in_one_call() -> None:
    check_codec_laws(codec_fixture(LineCodec()))

    with pytest.raises(LawViolationError, match="decoding round trips"):
        check_codec_laws(codec_fixture(LosingCodec()))


def test_a_codec_that_writes_a_record_without_its_value_fails_the_round_trip() -> None:
    with pytest.raises(LawViolationError, match="decoding round trips"):
        check_codec_laws(codec_fixture(LosingCodec()))


def test_a_codec_that_forgets_what_it_was_reading_into_fails() -> None:
    with pytest.raises(LawViolationError, match="decoding into extends"):
        check_codec_laws(codec_fixture(ReplacingCodec()))


def test_a_codec_that_finds_nothing_fails_rather_than_passing_vacuously() -> None:
    with pytest.raises(LawViolationError, match="found no record"):
        check_codec_laws(codec_fixture(SilentCodec()))


def test_a_codec_that_changes_the_fragment_it_was_handed_fails() -> None:
    fixture: CodecFixture[Records, Line] = CodecFixture(
        MutatingCodec(),
        sources().filter(bool),
        lambda fragment: fragment.records,
        EXAMPLES,
    )

    with pytest.raises(LawViolationError, match="decoding does not mutate"):
        decoding_does_not_mutate(fixture)


type Record = tuple[str, str, tuple[str, ...]]

CORPUS: Final[tuple[Record, ...]] = (
    ("train", "s/1", ("the", "child", "slept")),
    ("train", "s/2", ("the", "child", "slept")),
    ("test", "s/3", ("the", "dog", "barked")),
)

READER_LAW_NAMES: Final = tuple(law.__name__ for law in READER_LAWS)

type ReaderLaw = Callable[[ReaderFixture[str, str]], None]


def item_of(record: Record) -> Item:
    split, item_id, surfaces = record

    return Item(
        ItemId(item_id),
        tuple(Token(index, surface) for index, surface in enumerate(surfaces)),
        split=split,
    )


class Corpus:
    name = "fixture"
    grammar = None
    lexicon = None

    def __init__(self, records: tuple[Record, ...] = CORPUS) -> None:
        self._records = records

    def splits(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(split for split, _, _ in self._records))

    def items(self, split: str, /) -> Source[Item]:
        return tuple(item_of(record) for record in self._records if record[0] == split)

    def examples(self, split: str, /) -> Source[Example[Supervision[str]]]:
        return tuple(Example(item, Unlabeled()) for item in self.items(split))

    def supervision_kinds(self) -> frozenset[SupervisionArm]:
        return frozenset({Unlabeled})

    def report(self) -> LoadReport:
        return LoadReport(len(self._records), len(self._records))


class DrainingCorpus(Corpus):
    @override
    def items(self, split: str, /) -> Source[Item]:
        return _Drained(tuple(super().items(split)))


class MiscountingCorpus(Corpus):
    @override
    def items(self, split: str, /) -> Source[Item]:
        return _Overstated(tuple(super().items(split)))


class DriftingCorpus(Corpus):
    loads = 0

    def __init__(self, records: tuple[Record, ...] = CORPUS) -> None:
        super().__init__(records)
        DriftingCorpus.loads += 1
        self._load = DriftingCorpus.loads

    @override
    def items(self, split: str, /) -> Source[Item]:
        return tuple(
            Item(ItemId(f"{item.item_id}/{self._load}"), item.tokens, split=item.split)
            for item in super().items(split)
        )


class SurfaceKeyedCorpus(Corpus):
    @override
    def items(self, split: str, /) -> Source[Item]:
        return tuple(
            Item(
                ItemId(" ".join(token.surface for token in item.tokens)),
                item.tokens,
                split=item.split,
            )
            for item in super().items(split)
        )


class RepeatedSupervisionCorpus(Corpus):
    @override
    def items(self, split: str, /) -> Source[Item]:
        items = tuple(super().items(split))

        return items + items[:1]

    @override
    def examples(self, split: str, /) -> Source[Example[Supervision[str]]]:
        return tuple(
            Example(item, Scalar(float(index + 1), RATING))
            for index, item in enumerate(self.items(split))
        )


class UnderreportingCorpus(Corpus):
    @override
    def report(self) -> LoadReport:
        return LoadReport(
            len(self._records),
            len(self._records) - 1,
            {"invalid record": 1},
        )


class _Overstated:
    def __init__(self, items: tuple[Item, ...]) -> None:
        self._items = items

    def __iter__(self) -> Iterator[Item]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items) + 1


class _Drained:
    def __init__(self, items: tuple[Item, ...]) -> None:
        self._count = len(items)
        self._iterator = iter(items)

    def __iter__(self) -> Iterator[Item]:
        return self._iterator

    def __len__(self) -> int:
        return self._count


def reader_fixture(
    corpus: Callable[[], CorpusReader[str, str]],
) -> ReaderFixture[
    str,
    str,
]:
    return ReaderFixture(corpus)


@pytest.mark.parametrize("law", READER_LAWS, ids=READER_LAW_NAMES)
def test_a_faithful_reader_holds_to_every_reader_law(law: ReaderLaw) -> None:
    law(reader_fixture(Corpus))


def test_the_aggregate_runs_the_reader_laws_in_one_call() -> None:
    check_reader_laws(reader_fixture(Corpus))

    with pytest.raises(LawViolationError, match="re-readable"):
        check_reader_laws(reader_fixture(DrainingCorpus))


def test_a_reader_handing_out_a_generator_fails_re_readability() -> None:
    with pytest.raises(LawViolationError, match="on the second"):
        check_reader_laws(reader_fixture(DrainingCorpus))


def test_a_source_that_overstates_its_own_size_fails_re_readability() -> None:
    with pytest.raises(LawViolationError, match="reports a length"):
        check_reader_laws(reader_fixture(MiscountingCorpus))


def test_a_reader_minting_a_fresh_id_per_load_fails_stability() -> None:
    with pytest.raises(LawViolationError, match="ids are stable"):
        check_reader_laws(reader_fixture(DriftingCorpus))


def test_a_reader_keying_ids_on_the_surface_fails_distinctness() -> None:
    with pytest.raises(LawViolationError, match="for more than one item"):
        check_reader_laws(reader_fixture(SurfaceKeyedCorpus))


def test_repeated_examples_over_one_item_do_not_fail_distinctness() -> None:
    ids_are_distinct_under_a_repeated_surface(reader_fixture(RepeatedSupervisionCorpus))


def test_a_corpus_repeating_no_surface_cannot_test_distinctness() -> None:
    unrepeated = CORPUS[1:]

    with pytest.raises(LawViolationError, match="read alike"):
        check_reader_laws(reader_fixture(lambda: Corpus(unrepeated)))


def test_a_reader_whose_report_undercounts_what_it_emits_fails_totality() -> None:
    with pytest.raises(LawViolationError, match="accounts for every item"):
        check_reader_laws(reader_fixture(UnderreportingCorpus))


def test_an_empty_corpus_fails_rather_than_passing_every_law_vacuously() -> None:
    for law in READER_LAWS:
        with pytest.raises(LawViolationError, match=r"no item|read alike"):
            law(reader_fixture(lambda: Corpus(())))


ITEM_SURFACES: Final = ("the", "child", "sleeps")

ITEM_CATEGORIES: Final = ("NP/N", "N", "S\\NP", "S")

ITEM_RESERVED: Final = ("<unknown>",)

RATING: Final = Scale("ordinal", 1.0, 7.0)

ITEM_TOKENS: Final = tuple(Token(at, word) for at, word in enumerate(ITEM_SURFACES))

type ItemLaw = Callable[[ItemFixture], None]


def item_lexicon(reserved: tuple[str, ...] = ITEM_RESERVED) -> Inventory:
    return Inventory("surfaces", ITEM_SURFACES, reserved)


def item_categories() -> Inventory:
    return Inventory("categories", ITEM_CATEGORIES)


def an_item(handle: str, text: str | None = "the child sleeps") -> Item:
    return Item(ItemId(handle), ITEM_TOKENS, text=text)


def rated() -> Example[Supervision[str]]:
    return Example(an_item("rated"), Scalar(6.0, RATING))


def unlabeled() -> Example[Supervision[str]]:
    return Example(an_item("unlabeled"), Unlabeled())


def textless() -> Example[Supervision[str]]:
    return Example(an_item("textless", None), Unlabeled())


def ambiguous() -> Example[Supervision[str]]:
    permitted = (
        frozenset({"NP/N", "N"}),
        frozenset({"N"}),
        frozenset({"S\\NP", "S"}),
    )

    return Example(an_item("ambiguous"), SpanConstraints.from_positions(permitted))


def item_examples() -> Iterator[Example[Supervision[str]]]:
    yield rated()
    yield unlabeled()
    yield ambiguous()
    yield textless()


class Resolving(ItemHasher):
    __slots__ = ()

    @override
    def __call__(self, example: Example[Supervision[str]], /) -> HashedItem:
        hashed = super().__call__(example)

        return HashedItem(
            hashed.item_id,
            hashed.alphabet,
            hashed.symbols,
            tuple(frozenset({min(beam)}) for beam in hashed.analyses),
            hashed.response,
        )


class PartlyResolving(ItemHasher):
    __slots__ = ()

    @override
    def __call__(self, example: Example[Supervision[str]], /) -> HashedItem:
        hashed = super().__call__(example)
        analyses = list(hashed.analyses)

        for at, beam in enumerate(analyses):
            if len(beam) > 1:
                analyses[at] = frozenset({min(beam)})

                break

        return HashedItem(
            hashed.item_id,
            hashed.alphabet,
            hashed.symbols,
            tuple(analyses),
            hashed.response,
        )


class Scaleless(ItemHasher):
    __slots__ = ()

    @override
    def __call__(self, example: Example[Supervision[str]], /) -> HashedItem:
        hashed = super().__call__(example)

        return HashedItem(
            hashed.item_id, hashed.alphabet, hashed.symbols, hashed.analyses
        )


class Growing(ItemHasher):
    __slots__ = ("_grown",)

    def __init__(self, lexicon: Inventory, categories: Inventory) -> None:
        super().__init__(lexicon, categories)
        self._grown = lexicon

    @property
    @override
    def lexicon(self) -> Inventory:
        return self._grown

    @property
    @override
    def alphabet(self) -> Alphabet:
        return Alphabet(self._grown.size, self.categories.size)

    @override
    def __call__(self, example: Example[Supervision[str]], /) -> HashedItem:
        self._grown = self._grown.extended(
            token.surface for token in example.item.tokens
        )

        return HashedItem(
            example.item.item_id,
            self.alphabet,
            tuple(self._grown.index(token.surface) for token in example.item.tokens),
        )


def item_fixture(
    hasher: ItemHasher | None = None,
    examples: Callable[[], Iterator[Example[Supervision[str]]]] = item_examples,
    growth: tuple[str, ...] = ("dog", "runs"),
) -> ItemFixture:
    return ItemFixture(
        ItemHasher(item_lexicon(), item_categories()) if hasher is None else hasher,
        examples,
        ambiguous(),
        unlabeled(),
        textless(),
        growth,
    )


@pytest.mark.parametrize("law", ITEM_LAWS, ids=[law.__name__ for law in ITEM_LAWS])
def test_a_faithful_adapter_holds_to_every_item_law(law: ItemLaw) -> None:
    law(item_fixture())


def test_the_aggregate_runs_the_item_laws_in_one_call() -> None:
    check_item_laws(item_fixture())


def test_an_adapter_that_drops_the_scale_fails_the_first_lesson() -> None:
    with pytest.raises(LawViolationError, match="the scale rides on the response"):
        the_scale_rides_on_the_response(
            item_fixture(Scaleless(item_lexicon(), item_categories()))
        )


def test_a_corpus_with_no_response_cannot_test_the_first_lesson() -> None:
    with pytest.raises(LawViolationError, match="no example included a response"):
        the_scale_rides_on_the_response(
            item_fixture(examples=lambda: iter([unlabeled()]))
        )


def test_an_adapter_that_grows_its_inventory_fails_the_third_lesson() -> None:
    growing = Growing(Inventory("surfaces", ("the",), ITEM_RESERVED), item_categories())

    with pytest.raises(LawViolationError, match="does not grow"):
        the_inventory_does_not_grow_while_a_corpus_is_read(item_fixture(growing))


def test_a_fixture_supplying_no_example_cannot_test_the_third_lesson() -> None:
    with pytest.raises(LawViolationError, match="supplied no example"):
        the_inventory_does_not_grow_while_a_corpus_is_read(
            item_fixture(examples=lambda: iter([]))
        )


def test_an_adapter_that_resolves_ambiguity_fails_the_fourth_lesson() -> None:
    with pytest.raises(LawViolationError, match="ambiguity is preserved"):
        ambiguity_is_preserved_and_not_resolved(
            item_fixture(Resolving(item_lexicon(), item_categories()))
        )


def test_an_adapter_that_resolves_one_position_of_several_fails_the_fourth() -> None:
    partial = PartlyResolving(item_lexicon(), item_categories())
    hashed = partial(ambiguous())

    assert hashed.is_ambiguous

    with pytest.raises(LawViolationError, match="hashed to 1 of them"):
        ambiguity_is_preserved_and_not_resolved(item_fixture(partial))


def test_a_corpus_offering_no_alternative_cannot_test_the_fourth_lesson() -> None:
    fixture = item_fixture()._replace(ambiguous=unlabeled())

    with pytest.raises(LawViolationError, match="no position more than one"):
        ambiguity_is_preserved_and_not_resolved(fixture)


def test_an_inventory_reserving_nothing_cannot_test_the_fifth_lesson() -> None:
    bare = ItemHasher(item_lexicon(()), item_categories())

    with pytest.raises(LawViolationError, match="reserves no symbol"):
        a_reserved_symbol_keeps_its_index_as_the_inventory_grows(item_fixture(bare))


def test_a_growth_that_adds_nothing_cannot_test_the_fifth_lesson() -> None:
    with pytest.raises(LawViolationError, match="hold still under"):
        a_reserved_symbol_keeps_its_index_as_the_inventory_grows(
            item_fixture(growth=("the",))
        )


def test_an_example_named_unlabeled_that_has_a_rating_fails_the_sixth() -> None:
    fixture = item_fixture()._replace(unlabeled=rated())

    with pytest.raises(LawViolationError, match="hashed to the response"):
        a_missing_annotation_is_a_typed_absent_case(fixture)


def test_an_example_named_textless_that_contains_text_fails_the_sixth() -> None:
    fixture = item_fixture()._replace(textless=unlabeled())

    with pytest.raises(LawViolationError, match="contains text"):
        a_missing_annotation_is_a_typed_absent_case(fixture)


def constrained(identifier: str, *surfaces: str) -> Example[Supervision[str]]:
    return Example(
        Item(
            ItemId(f"at://example.invalid/constrained/{identifier}"),
            tuple(
                Token(position, surface) for position, surface in enumerate(surfaces)
            ),
        ),
        Unlabeled(),
    )


CONSTRAINED: Final = (
    constrained("1", "every", "person", "left"),
    constrained("2", "every", "person", "saw", "every", "person"),
    constrained("3", "nobody", "left"),
    constrained("4", "nobody", "saw", "nobody"),
)

BY_ITEM: Final = {
    CONSTRAINED[0].item.item_id: (
        SpanConstraint(TokenSpan(0, 2), frozenset({"NP"})),
        SpanConstraint(TokenSpan(2, 3), frozenset({"S\\NP"})),
    ),
    CONSTRAINED[1].item.item_id: (
        SpanConstraint(TokenSpan(0, 2), frozenset({"NP"})),
        SpanConstraint(TokenSpan(3, 5), frozenset({"NP"})),
        SpanConstraint(TokenSpan(2, 5), frozenset({"S\\NP"})),
    ),
    CONSTRAINED[3].item.item_id: (
        SpanConstraint(TokenSpan(1, 3), frozenset({"S\\NP"})),
        SpanConstraint(TokenSpan(0, 1), frozenset({"NP"})),
    ),
}

BY_FORM: Final = {
    ("every", "person"): frozenset({"NP"}),
    ("left",): frozenset({"S\\NP"}),
    ("nobody",): frozenset({"NP"}),
    ("saw",): frozenset({"(S\\NP)/NP"}),
}


def by_item(coverage: float = 1.0, seed: int = 0) -> DeclaredSource[str, str]:
    return DeclaredSource(
        SelectionPolicy(PerOccurrenceRule(), coverage=coverage, seed=seed),
        by_item=BY_ITEM,
    )


def by_form() -> DeclaredSource[str, str]:
    return DeclaredSource(SelectionPolicy(UnionRule()), by_form=BY_FORM)


class _Wrapped:
    def __init__(self, source: ConstraintSource[str, str]) -> None:
        self._source = source

    @property
    def scope(self) -> ScopeKind:
        return self._source.scope

    @property
    def policy(self) -> SelectionPolicy[str, str]:
        return self._source.policy

    def select(
        self, examples: Iterable[Example[Supervision[str]]], /
    ) -> ConstraintSelection[str, str]:
        return self._source.select(examples)


class Dropping(_Wrapped):
    """Loses the last constraint of the first constrained item."""

    @override
    def select(
        self, examples: Iterable[Example[Supervision[str]]], /
    ) -> ConstraintSelection[str, str]:
        selection = super().select(examples)
        emitted = selection.examples
        first = emitted[0]
        arm = first.supervision

        return ConstraintSelection(
            (
                Example(
                    first.item,
                    SpanConstraints(arm.constraints[:-1], arm.token_count),
                ),
                *emitted[1:],
            ),
            selection.report,
            selection.budget,
        )


class Doubling(_Wrapped):
    """Emits the first constrained item's first constraint twice."""

    @override
    def select(
        self, examples: Iterable[Example[Supervision[str]]], /
    ) -> ConstraintSelection[str, str]:
        selection = super().select(examples)
        emitted = selection.examples
        first = emitted[0]
        arm = first.supervision

        return ConstraintSelection(
            (
                Example(
                    first.item,
                    SpanConstraints(
                        (arm.constraints[0], *arm.constraints), arm.token_count
                    ),
                ),
                *emitted[1:],
            ),
            selection.report,
            selection.budget,
        )


class Withholding(_Wrapped):
    """Leaves the second item out of the emission and the report."""

    @override
    def select(
        self, examples: Iterable[Example[Supervision[str]]], /
    ) -> ConstraintSelection[str, str]:
        selection = super().select(examples)
        emitted = selection.examples
        kept = (emitted[0], *emitted[2:])

        return ConstraintSelection(
            kept,
            LoadReport(len(kept), len(kept)),
            selection.budget,
        )


class Reordering(_Wrapped):
    """Emits the items in reverse order."""

    @override
    def select(
        self, examples: Iterable[Example[Supervision[str]]], /
    ) -> ConstraintSelection[str, str]:
        selection = super().select(examples)

        return ConstraintSelection(
            tuple(reversed(selection.examples)), selection.report, selection.budget
        )


class Overreporting(_Wrapped):
    """Reports one more record seen than the corpus holds, as skipped."""

    @override
    def select(
        self, examples: Iterable[Example[Supervision[str]]], /
    ) -> ConstraintSelection[str, str]:
        selection = super().select(examples)
        report = selection.report

        return ConstraintSelection(
            selection.examples,
            LoadReport(report.records_seen + 1, report.emitted, {"phantom": 1}),
            selection.budget,
        )


class Underreporting(_Wrapped):
    """Reports one fewer item emitted than came back, as skipped."""

    @override
    def select(
        self, examples: Iterable[Example[Supervision[str]]], /
    ) -> ConstraintSelection[str, str]:
        selection = super().select(examples)
        report = selection.report

        return ConstraintSelection(
            selection.examples,
            LoadReport(report.records_seen, report.emitted - 1, {"lost": 1}),
            selection.budget,
        )


class Unbinding(_Wrapped):
    """Leaves the second item's form-level constraints out."""

    @override
    def select(
        self, examples: Iterable[Example[Supervision[str]]], /
    ) -> ConstraintSelection[str, str]:
        selection = super().select(examples)
        emitted = selection.examples
        second = emitted[1]

        return ConstraintSelection(
            (
                emitted[0],
                Example(
                    second.item,
                    SpanConstraints((), second.supervision.token_count),
                ),
                *emitted[2:],
            ),
            selection.report,
            selection.budget,
        )


class Misanchoring(_Wrapped):
    """Anchors every form-level constraint one token to the right where it can."""

    @override
    def select(
        self, examples: Iterable[Example[Supervision[str]]], /
    ) -> ConstraintSelection[str, str]:
        selection = super().select(examples)
        emitted = selection.examples
        moved: list[Example[SpanConstraints[str]]] = []

        for example in emitted:
            arm = example.supervision
            constraints = tuple(
                SpanConstraint(
                    TokenSpan(held.span.start + 1, held.span.end + 1),
                    held.permitted,
                    held.scope,
                )
                if held.span.end < arm.token_count
                else held
                for held in arm.constraints
            )
            moved.append(
                Example(example.item, SpanConstraints(constraints, arm.token_count))
            )

        return ConstraintSelection(tuple(moved), selection.report, selection.budget)


class Counting(_Wrapped):
    """Emits under a seed that advances on every call, across every instance."""

    calls: ClassVar[int] = 0

    @override
    def select(
        self, examples: Iterable[Example[Supervision[str]]], /
    ) -> ConstraintSelection[str, str]:
        Counting.calls += 1
        policy = self._source.policy.with_seed(Counting.calls)

        return DeclaredSource(policy, by_item=BY_ITEM).select(examples)


class Unseeded(_Wrapped):
    """Draws its covered fraction under one fixed seed for every policy."""

    @override
    def select(
        self, examples: Iterable[Example[Supervision[str]]], /
    ) -> ConstraintSelection[str, str]:
        policy = self._source.policy.with_seed(0)

        return DeclaredSource(policy, by_item=BY_ITEM).select(examples)


class Colliding(_Wrapped):
    """Draws under seed 0 when given seed 1, and honors every other seed."""

    @override
    def select(
        self, examples: Iterable[Example[Supervision[str]]], /
    ) -> ConstraintSelection[str, str]:
        policy = self._source.policy

        if policy.seed == 1:
            policy = policy.with_seed(0)

        return DeclaredSource(policy, by_item=BY_ITEM).select(examples)


def constraint_source_fixture(
    full: ConstraintSource[str, str] | None = None,
    typed: ConstraintSource[str, str] | None = None,
    half: Callable[[int], ConstraintSource[str, str]] | None = None,
) -> ConstraintSourceFixture[str, str]:
    return ConstraintSourceFixture(
        by_item() if full is None else full,
        by_form() if typed is None else typed,
        by_item if half is None else half,
        CONSTRAINED,
        {
            example.item.item_id: frozenset(BY_ITEM.get(example.item.item_id, ()))
            for example in CONSTRAINED
        },
    )


@pytest.mark.parametrize(
    "law", CONSTRAINT_SOURCE_LAWS, ids=[law.__name__ for law in CONSTRAINT_SOURCE_LAWS]
)
def test_a_faithful_constraint_source_holds_to_every_law(
    law: Callable[[ConstraintSourceFixture[str, str]], None],
) -> None:
    law(constraint_source_fixture(half=lambda seed: by_item(0.5, seed)))


def test_the_aggregate_runs_the_constraint_source_laws_in_one_call() -> None:
    check_constraint_source_laws(
        constraint_source_fixture(half=lambda seed: by_item(0.5, seed))
    )

    with pytest.raises(LawViolationError, match="every constituent"):
        check_constraint_source_laws(constraint_source_fixture(Dropping(by_item())))


def test_a_source_dropping_a_constraint_fails_full_coverage() -> None:
    with pytest.raises(LawViolationError, match="where the evidence names"):
        full_coverage_yields_every_constituent(
            constraint_source_fixture(Dropping(by_item()))
        )


def test_a_source_emitting_a_constraint_twice_fails_full_coverage() -> None:
    with pytest.raises(LawViolationError, match="one constraint twice"):
        full_coverage_yields_every_constituent(
            constraint_source_fixture(Doubling(by_item()))
        )


def test_a_source_withholding_an_item_fails_full_coverage() -> None:
    with pytest.raises(LawViolationError, match="emitted the items"):
        full_coverage_yields_every_constituent(
            constraint_source_fixture(Withholding(by_item()))
        )


def test_evidence_naming_nothing_cannot_test_full_coverage() -> None:
    fixture = constraint_source_fixture()._replace(
        expected=dict.fromkeys(fixture_ids(), frozenset[SpanConstraint[str]]())
    )

    with pytest.raises(LawViolationError, match="names no constraint"):
        full_coverage_yields_every_constituent(fixture)


def fixture_ids() -> tuple[ItemId, ...]:
    return tuple(example.item.item_id for example in CONSTRAINED)


def test_a_form_level_source_that_skips_an_item_holding_the_form_fails_binding() -> (
    None
):
    with pytest.raises(LawViolationError, match="emitted without its constraint"):
        a_type_level_constraint_binds_every_occurrence(
            constraint_source_fixture(typed=Unbinding(by_form()))
        )


def test_a_form_level_source_anchored_off_its_form_fails_binding() -> None:
    with pytest.raises(LawViolationError, match="which spells"):
        a_type_level_constraint_binds_every_occurrence(
            constraint_source_fixture(typed=Misanchoring(by_form()))
        )


def test_a_source_emitting_no_form_level_constraint_cannot_test_binding() -> None:
    with pytest.raises(LawViolationError, match="no form-level constraint"):
        a_type_level_constraint_binds_every_occurrence(
            constraint_source_fixture(typed=by_item())
        )


def test_a_source_reseeding_itself_on_every_call_fails_determinism() -> None:
    with pytest.raises(LawViolationError, match="under seed 0 differ"):
        emission_is_deterministic_under_a_seed(
            constraint_source_fixture(half=lambda seed: Counting(by_item(0.5, seed)))
        )


def test_a_source_ignoring_its_seed_fails_determinism() -> None:
    with pytest.raises(LawViolationError, match="seeds 0 to 7 emitted the same"):
        emission_is_deterministic_under_a_seed(
            constraint_source_fixture(half=lambda seed: Unseeded(by_item(0.5, seed)))
        )


def test_two_seeds_that_happen_to_coincide_do_not_fail_determinism() -> None:
    emission_is_deterministic_under_a_seed(
        constraint_source_fixture(half=lambda seed: Colliding(by_item(0.5, seed)))
    )


def test_a_source_ignoring_its_coverage_fails_determinism() -> None:
    with pytest.raises(LawViolationError, match="constraints of"):
        emission_is_deterministic_under_a_seed(
            constraint_source_fixture(half=lambda seed: by_item(1.0, seed))
        )


def test_a_source_seeing_more_than_the_corpus_fails_totality() -> None:
    with pytest.raises(LawViolationError, match="records of a corpus"):
        the_report_is_total(constraint_source_fixture(Overreporting(by_item())))


def test_a_source_reporting_fewer_than_it_emitted_fails_totality() -> None:
    with pytest.raises(LawViolationError, match="came back"):
        the_report_is_total(constraint_source_fixture(Underreporting(by_item())))


def test_a_source_emitting_out_of_corpus_order_fails_totality() -> None:
    with pytest.raises(LawViolationError, match="corpus's order"):
        the_report_is_total(constraint_source_fixture(Reordering(by_item())))
