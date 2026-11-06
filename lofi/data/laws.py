"""Conformance laws for data-layer adapters."""

from collections import Counter
from typing import TYPE_CHECKING, Final, NamedTuple

from hypothesis import given, settings

from lofi._checks import fail_law
from lofi._types import ItemId
from lofi.data._constraint import SpanConstraints
from lofi.data._hashed import HashedItem
from lofi.data._supervision import GoldDerivation, Scalar

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence

    from hypothesis.strategies import SearchStrategy

    from lofi.data._constraint import SpanConstraint
    from lofi.data._hashed import ItemHasher
    from lofi.data._ports import Codec, ConstraintSource, CorpusReader, Iso
    from lofi.data._supervision import Example, Supervision
    from lofi.data._values import Item

__all__ = [
    "CODEC_LAWS",
    "CONSTRAINT_SOURCE_LAWS",
    "ISO_LAWS",
    "ITEM_LAWS",
    "READER_LAWS",
    "CodecFixture",
    "ConstraintSourceFixture",
    "IsoFixture",
    "ItemFixture",
    "ReaderFixture",
    "a_missing_annotation_is_a_typed_absent_case",
    "a_reserved_symbol_keeps_its_index_as_the_inventory_grows",
    "a_type_level_constraint_binds_every_occurrence",
    "ambiguity_is_preserved_and_not_resolved",
    "backward_inverts_forward",
    "check_codec_laws",
    "check_constraint_source_laws",
    "check_iso_laws",
    "check_item_laws",
    "check_reader_laws",
    "decoding_does_not_mutate",
    "decoding_into_extends",
    "decoding_round_trips",
    "emission_is_deterministic_under_a_seed",
    "forward_inverts_backward",
    "full_coverage_yields_every_constituent",
    "ids_are_distinct_under_a_repeated_surface",
    "ids_are_stable_across_loads",
    "sources_are_re_readable",
    "the_hashed_level_refuses_a_code_its_alphabet_cannot_reach",
    "the_inventory_does_not_grow_while_a_corpus_is_read",
    "the_report_accounts_for_every_item",
    "the_report_is_total",
    "the_scale_rides_on_the_response",
]


class IsoFixture[Source, Target](NamedTuple):
    """Inputs for isomorphism conformance laws.

    Parameters
    ----------
    iso : Iso[Source, Target]
        The translation under test.
    sources : SearchStrategy[Source]
        A strategy over the side ``forward`` starts from.
    targets : SearchStrategy[Target]
        A strategy over the side ``backward`` starts from. It must draw
        targets the translation can reach, since a target no source maps to
        says nothing about whether the two directions invert each other.
    max_examples : int
        The example budget each law gives Hypothesis.
    """

    iso: Iso[Source, Target]
    sources: SearchStrategy[Source]
    targets: SearchStrategy[Target]
    max_examples: int = 200


class CodecFixture[Fragment, Record](NamedTuple):
    """Inputs for codec conformance laws.

    Parameters
    ----------
    codec : Codec[Fragment, Record]
        The codec under test.
    sources : SearchStrategy[str | bytes]
        A strategy over external sources the codec decodes. It must draw
        sources the codec finds records in, since a codec that decodes every
        source to nothing passes a round trip without doing anything.
    records_of : Callable[[Fragment], tuple[Record, ...]]
        Read the records back out of a fragment. A fragment is the codec's own
        pivot rather than this module's, so the way into it is supplied here.
    max_examples : int
        The example budget each law gives Hypothesis.
    """

    codec: Codec[Fragment, Record]
    sources: SearchStrategy[str | bytes]
    records_of: Callable[[Fragment], tuple[Record, ...]]
    max_examples: int = 100


def _budget(max_examples: int) -> settings:
    """Return the Hypothesis settings every law runs under.

    Parameters
    ----------
    max_examples : int
        The example budget the fixture declared.
    """
    # laws do not impose per-example deadlines
    return settings(max_examples=max_examples, deadline=None)


def backward_inverts_forward[Source, Target](
    fixture: IsoFixture[Source, Target],
) -> None:
    """Check that translating a source and back returns the source.

    Parameters
    ----------
    fixture : IsoFixture[Source, Target]
        The translation under test and its strategies.

    Raises
    ------
    LawViolationError
        If some source does not come back.
    """
    iso = fixture.iso

    @given(fixture.sources)
    @_budget(fixture.max_examples)
    def law(source: Source) -> None:
        returned = iso.backward(iso.forward(source))

        if returned != source:
            detail = f"{source!r} came back as {returned!r}"

            fail_law(detail, law="backward inverts forward")

    law()


def forward_inverts_backward[Source, Target](
    fixture: IsoFixture[Source, Target],
) -> None:
    """Check that translating a target back and forward returns the target.

    Parameters
    ----------
    fixture : IsoFixture[Source, Target]
        The translation under test and its strategies.

    Raises
    ------
    LawViolationError
        If some target does not come back.
    """
    iso = fixture.iso

    @given(fixture.targets)
    @_budget(fixture.max_examples)
    def law(target: Target) -> None:
        returned = iso.forward(iso.backward(target))

        if returned != target:
            detail = f"{target!r} came back as {returned!r}"

            fail_law(detail, law="forward inverts backward")

    law()


ISO_LAWS: Final = (backward_inverts_forward, forward_inverts_backward)
"""Round-trip laws in conformance-case order."""


def check_iso_laws[Source, Target](fixture: IsoFixture[Source, Target]) -> None:
    """Run every isomorphism law against one translation.

    Parameters
    ----------
    fixture : IsoFixture[Source, Target]
        The translation under test and its strategies.

    Raises
    ------
    LawViolationError
        On the first law the translation fails.
    """
    for law in ISO_LAWS:
        law(fixture)


def decoding_round_trips[Fragment, Record](
    fixture: CodecFixture[Fragment, Record],
) -> None:
    """Check that encoding what a source decoded to decodes back to the same.

    Parameters
    ----------
    fixture : CodecFixture[Fragment, Record]
        The codec under test, its strategy and its way into a fragment.

    Raises
    ------
    LawViolationError
        If some fragment does not come back, or if no drawn source held a
        record, which leaves the law with nothing to have tested.
    """
    codec = fixture.codec
    records_of = fixture.records_of
    seen: list[int] = []

    @given(fixture.sources)
    @_budget(fixture.max_examples)
    def law(source: str | bytes) -> None:
        decoded = codec.decode(source)
        seen.append(len(records_of(decoded)))
        returned = codec.decode(codec.encode(records_of(decoded)))

        if returned != decoded:
            detail = f"{source!r} decoded to {decoded!r} and back to {returned!r}"

            fail_law(detail, law="decoding round trips")

    law()

    if not any(seen):
        detail = f"{codec.name} found no record in any source it was given"

        fail_law(detail, law="decoding round trips")


def decoding_into_extends[Fragment, Record](
    fixture: CodecFixture[Fragment, Record],
) -> None:
    """Check that decoding into a fragment keeps that fragment's records first.

    Parameters
    ----------
    fixture : CodecFixture[Fragment, Record]
        The codec under test, its strategy and its way into a fragment.

    Raises
    ------
    LawViolationError
        If some extension drops or reorders what it was given, or if no drawn
        source held a record, which leaves the law with nothing to have tested.
    """
    codec = fixture.codec
    records_of = fixture.records_of
    seen: list[int] = []

    @given(fixture.sources, fixture.sources)
    @_budget(fixture.max_examples)
    def law(first: str | bytes, second: str | bytes) -> None:
        already = records_of(codec.decode(first))
        seen.append(len(already))
        extended = records_of(codec.decode(second, into=codec.decode(first)))

        if extended[: len(already)] != already:
            detail = (
                f"decoding {second!r} into the {len(already)} records of "
                f"{first!r} returned {extended!r}"
            )

            fail_law(detail, law="decoding into extends")

    law()

    if not any(seen):
        detail = f"{codec.name} found no record in any source it was given"

        fail_law(detail, law="decoding into extends")


def decoding_does_not_mutate[Fragment, Record](
    fixture: CodecFixture[Fragment, Record],
) -> None:
    """Check that decoding into a fragment leaves that fragment as it was.

    Parameters
    ----------
    fixture : CodecFixture[Fragment, Record]
        The codec under test, its strategy and its way into a fragment.

    Raises
    ------
    LawViolationError
        If some fragment is changed by being extended.
    """
    codec = fixture.codec
    records_of = fixture.records_of

    @given(fixture.sources, fixture.sources)
    @_budget(fixture.max_examples)
    def law(first: str | bytes, second: str | bytes) -> None:
        target = codec.decode(first)
        before = records_of(target)
        codec.decode(second, into=target)
        after = records_of(target)

        if after != before:
            detail = (
                f"decoding {second!r} into the fragment of {first!r} left it "
                f"holding {after!r} rather than {before!r}"
            )

            fail_law(detail, law="decoding does not mutate")

    law()


CODEC_LAWS: Final = (
    decoding_round_trips,
    decoding_into_extends,
    decoding_does_not_mutate,
)
"""Codec laws in conformance-case order."""


def check_codec_laws[Fragment, Record](fixture: CodecFixture[Fragment, Record]) -> None:
    """Run every codec law against one codec.

    Parameters
    ----------
    fixture : CodecFixture[Fragment, Record]
        The codec under test, its strategy and its way into a fragment.

    Raises
    ------
    LawViolationError
        On the first law the codec fails.
    """
    for law in CODEC_LAWS:
        law(fixture)


class ReaderFixture[L: Hashable, G](NamedTuple):
    """Inputs for corpus-reader conformance laws.

    Parameters
    ----------
    load : Callable[[], CorpusReader[L, G]]
        Read the corpus and return a reader over it. Two calls are two loads,
        so a fixture handing back one cached reader says nothing about whether
        ids survive a reload.
    """

    load: Callable[[], CorpusReader[L, G]]


def _all_items[L: Hashable, G](reader: CorpusReader[L, G]) -> tuple[Item, ...]:
    """Return every item of every split, in the order the reader lists them.

    Parameters
    ----------
    reader : CorpusReader[L, G]
        The reader to read.

    Returns
    -------
    tuple[Item, ...]
        The items, split by split.
    """
    return tuple(item for split in reader.splits() for item in reader.items(split))


def _all_examples[L: Hashable, G](
    reader: CorpusReader[L, G],
) -> tuple[Example[Supervision[L]], ...]:
    """Return every example of every split."""
    return tuple(
        example for split in reader.splits() for example in reader.examples(split)
    )


def sources_are_re_readable[L: Hashable, G](fixture: ReaderFixture[L, G]) -> None:
    """Check that a split repeats items and examples exactly.

    Parameters
    ----------
    fixture : ReaderFixture[L, G]
        How to load the corpus.

    Raises
    ------
    LawViolationError
        If two readings of one source differ, if a source's length disagrees
        with what it yields, or if the corpus holds no item, which leaves the
        law with nothing to have tested.
    """
    reader = fixture.load()
    seen = 0

    for split in reader.splits():
        for name, source in (
            ("items", reader.items(split)),
            ("examples", reader.examples(split)),
        ):
            first = tuple(source)
            second = tuple(source)
            seen += len(first)

            if first != second:
                detail = (
                    f"{name} of {split!r} read {len(first)} on the first pass "
                    f"and {len(second)} on the second"
                )

                fail_law(detail, law="sources are re-readable")

            if len(source) != len(first):
                detail = (
                    f"{name} of {split!r} reports a length of {len(source)} "
                    f"and yields {len(first)}"
                )

                fail_law(detail, law="sources are re-readable")

    if not seen:
        detail = f"{reader.name} handed out no item in any of its splits"

        fail_law(detail, law="sources are re-readable")


def ids_are_stable_across_loads[L: Hashable, G](fixture: ReaderFixture[L, G]) -> None:
    """Check that two loads of one corpus mint the same ids in the same order.

    Parameters
    ----------
    fixture : ReaderFixture[L, G]
        How to load the corpus.

    Raises
    ------
    LawViolationError
        If the two loads disagree about the splits or about any split's ids, or
        if the corpus holds no item.
    """
    one = fixture.load()
    other = fixture.load()

    if one.splits() != other.splits():
        detail = f"one load found {one.splits()} and the other {other.splits()}"

        fail_law(detail, law="ids are stable across loads")

    seen = 0

    for split in one.splits():
        first = tuple(item.item_id for item in one.items(split))
        second = tuple(item.item_id for item in other.items(split))
        seen += len(first)

        if first != second:
            detail = f"{split!r} loaded as {first!r} and then as {second!r}"

            fail_law(detail, law="ids are stable across loads")

    if not seen:
        detail = f"{one.name} handed out no item in any of its splits"

        fail_law(detail, law="ids are stable across loads")


def ids_are_distinct_under_a_repeated_surface[L: Hashable, G](
    fixture: ReaderFixture[L, G],
) -> None:
    """Check that items with the same surface form retain distinct identifiers.

    Parameters
    ----------
    fixture : ReaderFixture[L, G]
        How to load the corpus.

    Raises
    ------
    LawViolationError
        If two items share an id, or if no two items share a surface form.
    """
    reader = fixture.load()
    items = _all_items(reader)
    examples = _all_examples(reader)
    repeated_ids = [
        item_id
        for item_id, count in Counter(item.item_id for item in items).items()
        if count > 1
        and len({example for example in examples if example.item.item_id == item_id})
        != count
    ]

    if repeated_ids:
        detail = f"{reader.name} minted {repeated_ids[0]!r} for more than one item"

        fail_law(detail, law="ids are distinct under a repeated surface")

    distinct = tuple(dict.fromkeys(items))
    surfaces = Counter(
        tuple(token.surface for token in item.tokens) for item in distinct
    )

    if not any(count > 1 for count in surfaces.values()):
        detail = (
            f"no two of the {len(distinct)} items {reader.name} handed out read "
            "alike, so the fixture's corpus cannot tell a reader that mints an "
            "id from the surface form from one that does not"
        )

        fail_law(detail, law="ids are distinct under a repeated surface")


def the_report_accounts_for_every_item[L: Hashable, G](
    fixture: ReaderFixture[L, G],
) -> None:
    """Check that the report's emitted count is what the reader handed out.

    Parameters
    ----------
    fixture : ReaderFixture[L, G]
        How to load the corpus.

    Raises
    ------
    LawViolationError
        If the report's emitted count differs from the number of items handed
        out, or if the corpus holds no item.
    """
    reader = fixture.load()
    emitted = sum(len(reader.items(split)) for split in reader.splits())
    report = reader.report()

    if report.emitted != emitted:
        detail = (
            f"{reader.name} reports {report.emitted} emitted out of "
            f"{report.records_seen} seen and hands out {emitted} items"
        )

        fail_law(detail, law="the report accounts for every item")

    if not emitted:
        detail = f"{reader.name} handed out no item in any of its splits"

        fail_law(detail, law="the report accounts for every item")


READER_LAWS: Final = (
    sources_are_re_readable,
    ids_are_stable_across_loads,
    ids_are_distinct_under_a_repeated_surface,
    the_report_accounts_for_every_item,
)
"""In-process reader laws in conformance-case order."""


def check_reader_laws[L: Hashable, G](fixture: ReaderFixture[L, G]) -> None:
    """Run every in-process reader law against one corpus.

    Parameters
    ----------
    fixture : ReaderFixture[L, G]
        How to load the corpus.

    Raises
    ------
    LawViolationError
        On the first law the reader fails.
    """
    for law in READER_LAWS:
        law(fixture)


_PROBE_ID: Final = ItemId("the-item-a-law-tries-to-build")
"""The item identifier used by rejection laws."""


def _alternatives(example: Example[Supervision[str]]) -> tuple[frozenset[str], ...]:
    """Return the categories a corpus offers at each position of one example.

    Parameters
    ----------
    example : Example[Supervision[str]]
        The item and what its corpus says about it.

    Returns
    -------
    tuple[frozenset[str], ...]
        One set per token position, empty overall for an arm that offers no
        category at all.
    """
    supervision = example.supervision

    if isinstance(supervision, SpanConstraints):
        return tuple(constraint.permitted for constraint in supervision.constraints)

    if isinstance(supervision, GoldDerivation):
        derivation = supervision.derivation
        leaves = sorted(
            (node for node in derivation.nodes.values() if node.is_lexical),
            key=lambda node: node.span.start,
        )

        return tuple(frozenset({node.label}) for node in leaves)

    return ()


class ItemFixture(NamedTuple):
    """Inputs for item-encoding conformance laws.

    Parameters
    ----------
    hasher : ItemHasher
        Adapter from symbolic examples to hashed items.
    examples : Callable[[], Iterator[Example[Supervision[str]]]]
        Factory for a fresh iterator over supported examples.
    ambiguous : Example[Supervision[str]]
        Example with more than one category at a position.
    unlabeled : Example[Supervision[str]]
        Example without supervision.
    textless : Example[Supervision[str]]
        Example whose item has no text.
    growth : tuple[str, ...]
        Ordinary symbols used to extend the lexical inventory.
    """

    hasher: ItemHasher
    examples: Callable[[], Iterator[Example[Supervision[str]]]]
    ambiguous: Example[Supervision[str]]
    unlabeled: Example[Supervision[str]]
    textless: Example[Supervision[str]]
    growth: tuple[str, ...]


def the_scale_rides_on_the_response(fixture: ItemFixture) -> None:
    """Check that a response arrives with the scale it was collected on.

    Parameters
    ----------
    fixture : ItemFixture
        The adapter and the examples to put through it.

    Raises
    ------
    LawViolationError
        If a hashed response is not the response on its example, if an
        example with no response hashes to one that has one, or if no example
        has a response at all.
    """
    hasher = fixture.hasher
    rated = 0

    for example in fixture.examples():
        hashed = hasher(example)
        response = example.supervision

        if not isinstance(response, Scalar):
            if hashed.response is not None:
                detail = (
                    f"{example.item.item_id!r} has {response!r} and hashed "
                    f"to a response of {hashed.response!r}"
                )

                fail_law(detail, law="the scale rides on the response")

            continue

        rated += 1

        if hashed.response != response:
            detail = (
                f"{example.item.item_id!r} has response {response!r} but "
                f"hashed to {hashed.response!r}"
            )

            fail_law(detail, law="the scale rides on the response")

    if not rated:
        detail = (
            "no example included a response, so the fixture cannot tell an "
            "adapter that reads the scale off the item from one that supplies "
            "a scale of its own"
        )

        fail_law(detail, law="the scale rides on the response")


def the_hashed_level_refuses_a_code_its_alphabet_cannot_reach(
    fixture: ItemFixture,
) -> None:
    """Check that a hashed item cannot be built holding an unreachable code.

    Parameters
    ----------
    fixture : ItemFixture
        The adapter, whose alphabet says which codes are reachable.

    Raises
    ------
    LawViolationError
        If a code at either alphabet's ceiling is accepted, or if a negative
        one is.
    """
    alphabet = fixture.hasher.alphabet

    for name, built in (
        (
            "a surface code at the ceiling",
            lambda: HashedItem(_PROBE_ID, alphabet, (alphabet.lexical,)),
        ),
        (
            "a negative surface code",
            lambda: HashedItem(_PROBE_ID, alphabet, (-1,)),
        ),
        (
            "a category code at the ceiling",
            lambda: HashedItem(
                _PROBE_ID, alphabet, (0,), (frozenset({alphabet.categorial}),)
            ),
        ),
    ):
        try:
            built()
        except ValueError:
            continue

        detail = f"{name} was accepted against an alphabet of {alphabet!r}"

        fail_law(detail, law="the hashed level refuses an unreachable code")


def the_inventory_does_not_grow_while_a_corpus_is_read(fixture: ItemFixture) -> None:
    """Check that corpus reading preserves declared inventories.

    Parameters
    ----------
    fixture : ItemFixture
        The adapter and the examples to put through it.

    Raises
    ------
    LawViolationError
        If the alphabet moves while the corpus is read, if a minted code falls
        outside it, or if no example was read at all.
    """
    hasher = fixture.hasher
    before = hasher.alphabet
    seen = 0

    for example in fixture.examples():
        hashed = hasher(example)
        seen += 1

        if hasher.alphabet != before:
            detail = (
                f"the alphabet was {before!r} and is {hasher.alphabet!r} after "
                f"reading {example.item.item_id!r}"
            )

            fail_law(detail, law="the inventory does not grow while a corpus is read")

        for code in hashed.symbols:
            if not 0 <= code < before.lexical:
                detail = (
                    f"{example.item.item_id!r} hashed to the surface code "
                    f"{code} against an inventory of {before.lexical}"
                )

                fail_law(
                    detail, law="the inventory does not grow while a corpus is read"
                )

    if not seen:
        detail = "the fixture supplied no example, so nothing was read at all"

        fail_law(detail, law="the inventory does not grow while a corpus is read")


def ambiguity_is_preserved_and_not_resolved(fixture: ItemFixture) -> None:
    """Check that every category a corpus offered a position survives hashing.

    Parameters
    ----------
    fixture : ItemFixture
        The adapter and an example containing an ambiguous position.

    Raises
    ------
    LawViolationError
        If the example is unambiguous or hashing drops an offered alternative.
    """
    hasher = fixture.hasher
    example = fixture.ambiguous
    offered = _alternatives(example)

    if not any(len(candidates) > 1 for candidates in offered):
        detail = (
            f"the example named as ambiguous, {example.item.item_id!r}, offers "
            "no position more than one category"
        )

        fail_law(detail, law="ambiguity is preserved and not resolved")

    hashed = hasher(example)

    if len(hashed.analyses) != len(offered):
        detail = (
            f"{example.item.item_id!r} was offered categories at "
            f"{len(offered)} positions and hashed to {len(hashed.analyses)}"
        )

        fail_law(detail, law="ambiguity is preserved and not resolved")

    for position, (candidates, codes) in enumerate(
        zip(offered, hashed.analyses, strict=True)
    ):
        if len(codes) != len(candidates):
            detail = (
                f"position {position} of {example.item.item_id!r} was offered "
                f"{sorted(candidates)} and hashed to {len(codes)} of them"
            )

            fail_law(detail, law="ambiguity is preserved and not resolved")

    if not hashed.is_ambiguous:
        detail = (
            f"{example.item.item_id!r} was offered a position more than one "
            "category and hashed to a form that reports no ambiguity"
        )

        fail_law(detail, law="ambiguity is preserved and not resolved")


def a_reserved_symbol_keeps_its_index_as_the_inventory_grows(
    fixture: ItemFixture,
) -> None:
    """Check that growing an inventory moves no code already declared.

    Parameters
    ----------
    fixture : ItemFixture
        The adapter, whose lexical inventory is grown, and the symbols to grow
        it by.

    Raises
    ------
    LawViolationError
        If the inventory reserves nothing, if the growth adds nothing, or if
        some code moves.
    """
    lexicon = fixture.hasher.lexicon

    if not lexicon.reserved:
        detail = (
            f"the inventory {lexicon.name} reserves no symbol, so a prefix "
            "that held still could not be told from one that did not"
        )

        fail_law(detail, law="a reserved symbol keeps its index")

    grown = lexicon.extended(fixture.growth)

    if grown.size <= lexicon.size:
        detail = (
            f"{lexicon.name} declares {lexicon.size} codes and grew to "
            f"{grown.size}, so nothing was added to hold still under"
        )

        fail_law(detail, law="a reserved symbol keeps its index")

    for symbol in (*lexicon.reserved, *lexicon.ordinary):
        if grown.index(symbol) != lexicon.index(symbol):
            detail = (
                f"{symbol!r} was coded {lexicon.index(symbol)} and is coded "
                f"{grown.index(symbol)} in the grown inventory"
            )

            fail_law(detail, law="a reserved symbol keeps its index")

    for symbol in lexicon.reserved:
        if not grown.is_reserved(symbol):
            detail = f"{symbol!r} was reserved and is ordinary after the growth"

            fail_law(detail, law="a reserved symbol keeps its index")


def a_missing_annotation_is_a_typed_absent_case(fixture: ItemFixture) -> None:
    """Check that absent corpus values remain absent.

    Parameters
    ----------
    fixture : ItemFixture
        The adapter and the two examples the law needs a witness of.

    Raises
    ------
    LawViolationError
        If the example the fixture names as unlabeled has an analysis or a
        response, or if the one it names as textless has text or fails to
        hash.
    """
    hasher = fixture.hasher
    unlabeled = hasher(fixture.unlabeled)

    if unlabeled.analyses:
        detail = (
            f"{fixture.unlabeled.item.item_id!r} is unlabeled and hashed to "
            f"the analyses {unlabeled.analyses!r}"
        )

        fail_law(detail, law="a missing annotation is a typed absent case")

    if unlabeled.response is not None:
        detail = (
            f"{fixture.unlabeled.item.item_id!r} is unlabeled and hashed to "
            f"the response {unlabeled.response!r}"
        )

        fail_law(detail, law="a missing annotation is a typed absent case")

    if fixture.textless.item.text is not None:
        detail = (
            f"the example named as textless, "
            f"{fixture.textless.item.item_id!r}, contains text"
        )

        fail_law(detail, law="a missing annotation is a typed absent case")

    hashed = hasher(fixture.textless)

    if len(hashed.symbols) != len(fixture.textless.item.tokens):
        detail = (
            f"{fixture.textless.item.item_id!r} holds "
            f"{len(fixture.textless.item.tokens)} tokens and hashed to "
            f"{len(hashed.symbols)} codes"
        )

        fail_law(detail, law="a missing annotation is a typed absent case")


ITEM_LAWS: Final = (
    the_scale_rides_on_the_response,
    the_hashed_level_refuses_a_code_its_alphabet_cannot_reach,
    the_inventory_does_not_grow_while_a_corpus_is_read,
    ambiguity_is_preserved_and_not_resolved,
    a_reserved_symbol_keeps_its_index_as_the_inventory_grows,
    a_missing_annotation_is_a_typed_absent_case,
)
"""Item-interface laws in conformance-case order."""


def check_item_laws(fixture: ItemFixture) -> None:
    """Run every item law against one adapter and its corpus.

    Parameters
    ----------
    fixture : ItemFixture
        The adapter, the examples and the three witnesses.

    Raises
    ------
    LawViolationError
        On the first law the adapter fails.
    """
    for law in ITEM_LAWS:
        law(fixture)


class ConstraintSourceFixture[C: Hashable, L: Hashable](NamedTuple):
    """Inputs for constraint-source conformance laws.

    Parameters
    ----------
    full : ConstraintSource[C, L]
        The source at full coverage over every frequency, at its own scope.
    typed : ConstraintSource[C, L]
        The same evidence at full coverage under a rule binding at the form level.
    half : Callable[[int], ConstraintSource[C, L]]
        Build the source at half coverage under a seed.
    corpus : tuple[Example[Supervision[C]], ...]
        The examples every source reads, in item order.
    expected : Mapping[ItemId, frozenset[SpanConstraint[L]]]
        The constraints full coverage names for each readable item, an item that
        names none included with an empty set.
    """

    full: ConstraintSource[C, L]
    typed: ConstraintSource[C, L]
    half: Callable[[int], ConstraintSource[C, L]]
    corpus: tuple[Example[Supervision[C]], ...]
    expected: Mapping[ItemId, frozenset[SpanConstraint[L]]]


def _emitted_ids[L: Hashable](
    emitted: Sequence[Example[SpanConstraints[L]]],
) -> tuple[ItemId, ...]:
    """Return the ids of constrained examples in emission order.

    Parameters
    ----------
    emitted : Sequence[Example[SpanConstraints[L]]]
        What a source emitted.
    """
    return tuple(example.item.item_id for example in emitted)


def full_coverage_yields_every_constituent[C: Hashable, L: Hashable](
    fixture: ConstraintSourceFixture[C, L],
) -> None:
    """Check that full coverage emits exactly what the evidence names, item by item.

    Parameters
    ----------
    fixture : ConstraintSourceFixture[C, L]
        The source, the corpus and what the corpus's evidence names.

    Raises
    ------
    LawViolationError
        If an item is emitted with a constraint the evidence does not name or
        without one it does, if a constraint is emitted twice for one item, if
        the emitted items are not exactly the expected ones, or if the evidence
        names no constraint at all, which leaves the law with nothing to test.
    """
    law = "full coverage yields every constituent"

    if not any(fixture.expected.values()):
        fail_law("the fixture names no constraint under full coverage", law=law)

    emitted = fixture.full.select(fixture.corpus).examples
    ids = _emitted_ids(emitted)

    if set(ids) != set(fixture.expected) or len(set(ids)) != len(ids):
        detail = (
            f"full coverage emitted the items {ids!r} where the evidence names "
            f"{tuple(fixture.expected)!r}"
        )

        fail_law(detail, law=law)

    for example in emitted:
        constraints = example.supervision.constraints
        named = fixture.expected[example.item.item_id]

        if len(constraints) != len(set(constraints)):
            detail = (
                f"{example.item.item_id!r} was emitted with one constraint twice: "
                f"{constraints!r}"
            )

            fail_law(detail, law=law)

        if set(constraints) != named:
            detail = (
                f"{example.item.item_id!r} was emitted with {set(constraints)!r} "
                f"where the evidence names {named!r}"
            )

            fail_law(detail, law=law)


def a_type_level_constraint_binds_every_occurrence[C: Hashable, L: Hashable](
    fixture: ConstraintSourceFixture[C, L],
) -> None:
    """Check that a form-level constraint reaches every occurrence and no other span.

    Parameters
    ----------
    fixture : ConstraintSourceFixture[C, L]
        The source under a form-level rule, and the corpus.

    Raises
    ------
    LawViolationError
        If a form-level constraint is anchored at a span not spelling its form,
        if an item holding the form was emitted without the constraint, if the
        emitted arm's instances miss an occurrence of the form or add a span
        that is not one, or if no form-level constraint was emitted, which
        leaves the law with nothing to test.
    """
    law = "a type-level constraint binds every occurrence"
    emitted = fixture.typed.select(fixture.corpus).examples
    bound: dict[tuple[str, ...], frozenset[L]] = {}

    for example in emitted:
        item = example.item
        surfaces = tuple(token.surface for token in item.tokens)
        instances = example.supervision.instances(item.tokens)

        for constraint in example.supervision.constraints:
            form = constraint.scope.form

            if form is None:
                continue

            span = constraint.span
            bound[form] = constraint.permitted

            if surfaces[span.start : span.end] != form:
                detail = (
                    f"{item.item_id!r} anchors the form {list(form)} at "
                    f"[{span.start}, {span.end}), which spells "
                    f"{list(surfaces[span.start : span.end])}"
                )

                fail_law(detail, law=law)

            occurrences = {
                start
                for start in range(len(surfaces) - len(form) + 1)
                if surfaces[start : start + len(form)] == form
            }
            reached = {
                instance.span.start
                for instance in instances
                if instance.permitted == constraint.permitted
                and instance.span.length == len(form)
                and surfaces[instance.span.start : instance.span.end] == form
            }

            if reached != occurrences:
                detail = (
                    f"{item.item_id!r} binds {list(form)} at {sorted(reached)!r} "
                    f"where it occurs at {sorted(occurrences)!r}"
                )

                fail_law(detail, law=law)

    if not bound:
        fail_law("the form-level source emitted no form-level constraint", law=law)

    for example in emitted:
        item = example.item
        surfaces = tuple(token.surface for token in item.tokens)
        held = {
            constraint.scope.form
            for constraint in example.supervision.constraints
            if constraint.scope.form is not None
        }

        for form in bound:
            occurs = any(
                surfaces[start : start + len(form)] == form
                for start in range(len(surfaces) - len(form) + 1)
            )

            if occurs and form not in held:
                detail = (
                    f"{item.item_id!r} holds the form {list(form)} and was "
                    "emitted without its constraint"
                )

                fail_law(detail, law=law)


def emission_is_deterministic_under_a_seed[C: Hashable, L: Hashable](
    fixture: ConstraintSourceFixture[C, L],
) -> None:
    """Check that a seed fixes the emission and that coverage draws under it.

    Parameters
    ----------
    fixture : ConstraintSourceFixture[C, L]
        The source at full and at half coverage, and the corpus.

    Raises
    ------
    LawViolationError
        If two emissions at full coverage differ, if two emissions at half
        coverage under one seed differ, if half coverage emits as many
        constraints as full coverage, or if half coverage emits the same
        constraints under every one of eight seeds.
    """
    law = "emission is deterministic under a seed"
    first = fixture.full.select(fixture.corpus).examples
    second = fixture.full.select(fixture.corpus).examples

    if first != second:
        fail_law("two emissions at full coverage differ", law=law)

    seeds = range(8)
    once = fixture.half(seeds[0]).select(fixture.corpus).examples
    again = fixture.half(seeds[0]).select(fixture.corpus).examples

    if once != again:
        detail = f"two emissions at half coverage under seed {seeds[0]} differ"

        fail_law(detail, law=law)

    covered = sum(len(example.supervision.constraints) for example in once)
    total = sum(len(example.supervision.constraints) for example in first)

    if covered >= total:
        detail = f"half coverage emitted {covered} constraints of {total}"

        fail_law(detail, law=law)

    if all(
        fixture.half(seed).select(fixture.corpus).examples == once for seed in seeds[1:]
    ):
        detail = f"half coverage under seeds {seeds[0]} to {seeds[-1]} emitted the same"

        fail_law(detail, law=law)


def the_report_is_total[C: Hashable, L: Hashable](
    fixture: ConstraintSourceFixture[C, L],
) -> None:
    """Check that the report saw the corpus and emitted what came back, in order.

    A report's emitted and skipped counts sum to what it saw by construction, so
    the law checks the two ends the report cannot check itself: what it saw is
    the corpus, and what it emitted is what was returned.

    Parameters
    ----------
    fixture : ConstraintSourceFixture[C, L]
        The source and the corpus.

    Raises
    ------
    LawViolationError
        If the report saw other than the corpus or emitted other than what was
        returned, or if the emitted items are not a subsequence of the corpus.
    """
    law = "the report is total"
    selection = fixture.full.select(fixture.corpus)
    emitted, report = selection.examples, selection.report

    if report.records_seen != len(fixture.corpus):
        detail = (
            f"the report saw {report.records_seen} records of a corpus of "
            f"{len(fixture.corpus)}"
        )

        fail_law(detail, law=law)

    if report.emitted != len(emitted):
        detail = f"the report emitted {report.emitted} where {len(emitted)} came back"

        fail_law(detail, law=law)

    order = tuple(example.item.item_id for example in fixture.corpus)
    ids = _emitted_ids(emitted)
    kept = tuple(item_id for item_id in order if item_id in set(ids))

    if ids != kept:
        detail = f"the emitted items {ids!r} are not the corpus's order {order!r}"

        fail_law(detail, law=law)


CONSTRAINT_SOURCE_LAWS: Final = (
    full_coverage_yields_every_constituent,
    a_type_level_constraint_binds_every_occurrence,
    emission_is_deterministic_under_a_seed,
    the_report_is_total,
)
"""Constraint-source laws in conformance-case order."""


def check_constraint_source_laws[C: Hashable, L: Hashable](
    fixture: ConstraintSourceFixture[C, L],
) -> None:
    """Run every constraint-source law against one source.

    Parameters
    ----------
    fixture : ConstraintSourceFixture[C, L]
        The source, the corpus and what the corpus's evidence names.

    Raises
    ------
    LawViolationError
        On the first law the source fails.
    """
    for law in CONSTRAINT_SOURCE_LAWS:
        law(fixture)
