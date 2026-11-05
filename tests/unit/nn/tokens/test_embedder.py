from typing import TYPE_CHECKING, Final, cast

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.conformance._sites import SiteObservation
from tests.conformance.token_encoder import (
    TokenEncoderConformance,
    TokenEncoderSuite,
)

from lofi.nn._values import TruncationReport
from lofi.nn.sites import (
    Access,
    ComponentId,
    ComponentKind,
    DTypeClass,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)
from lofi.nn.tokens.embedder import (
    LoadReport,
    ReservedTokens,
    StaticTokenEmbedder,
    TokenVocabulary,
    from_glove_text,
    from_word2vec_text,
)
from lofi.nn.tokens.pool import MeanSubwordPooler
from lofi.nn.tokens.window import REFUSE, REPORT, ContextOverflowError, ContextWindow
from lofi.probe import ensure_publishes

if TYPE_CHECKING:
    from pathlib import Path

EMBEDDER: Final = ComponentId("embedder")
ROW_AXES: Final = ("item", "row", "width")

# the unknown token uses row 2.
RESERVED: Final = ReservedTokens({"<pad>": 0, "<bos>": 1, "<unk>": 2})
WORDS: Final = ("dog", "cat", "runs")

# three rows name vocabulary tokens.
GLOVE: Final = "dog 1.0 2.0\nfox 3.0 4.0\ncat 5.0 6.0\nwolf 7.0 8.0\nruns 9.0 10.0"


def vocabulary(unknown: str = "<unk>") -> TokenVocabulary:
    return TokenVocabulary(WORDS, reserved=RESERVED, unknown=unknown)


def written(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.write_text(text, encoding="utf-8", newline="\n")

    return path


def embedder(window: ContextWindow | None = None) -> StaticTokenEmbedder:
    built = vocabulary()
    table = torch.arange(float(built.size * 2)).reshape(built.size, 2)

    return StaticTokenEmbedder(
        table,
        vocabulary=built,
        component=EMBEDDER,
        window=window,
        trainable=False,
        load_report=None,
    )


class TestAStaticEmbedderUnderAReportingWindow(TokenEncoderSuite):
    @pytest.fixture
    def conformance(self) -> TokenEncoderConformance[StaticTokenEmbedder]:
        batch = (("dog", "cat", "runs"), ("cat",))

        def build() -> StaticTokenEmbedder:
            return embedder(ContextWindow(3, overflow=REPORT))

        return TokenEncoderConformance(
            build=build,
            items=batch,
            overlong=(("dog", "cat", "runs", "zzz"),),
            budget=3,
            sites=SiteObservation(
                build=build,
                observe=lambda subject: {
                    Site(EMBEDDER, Role.TOKEN_EMBEDDING): subject.encode_words(
                        batch
                    ).vectors
                },
                read=lambda subject: (subject.encode_words(batch).vectors.tensor,),
            ),
        )


def items() -> st.SearchStrategy[tuple[tuple[str, ...], ...]]:
    words = st.lists(st.sampled_from([*WORDS, "zzz"]), min_size=0, max_size=6).map(
        tuple
    )

    return st.lists(words, min_size=1, max_size=4).map(tuple)


def test_a_reserved_index_is_the_owners_and_two_owners_may_disagree() -> None:
    one = TokenVocabulary(WORDS, reserved=RESERVED, unknown="<unk>")
    other = TokenVocabulary(
        WORDS, reserved=ReservedTokens({"<unk>": 4, "<pad>": 3}), unknown="<unk>"
    )

    assert one.unknown_index == 2
    assert other.unknown_index == 4
    assert one.index_of("zzz") == 2
    assert other.index_of("zzz") == 4
    assert one.index_of("dog") != other.index_of("dog")


def test_two_reserved_tokens_may_not_share_an_index() -> None:
    with pytest.raises(ValueError, match="both"):
        ReservedTokens({"<unk>": 1, "<pad>": 1})


def test_a_reserved_index_is_a_non_negative_integer_under_a_name() -> None:
    with pytest.raises(ValueError, match="negative"):
        ReservedTokens({"<unk>": -1})

    with pytest.raises(ValueError, match="empty"):
        ReservedTokens({"": 0})

    with pytest.raises(TypeError, match="not an integer"):
        ReservedTokens({"<unk>": cast("int", 1.0)})


def test_the_reserved_tokens_report_the_order_of_the_indices_they_took() -> None:
    assert RESERVED.names == ("<pad>", "<bos>", "<unk>")
    assert RESERVED.indices == frozenset({0, 1, 2})
    assert RESERVED.count == 3
    assert RESERVED.holds("<unk>")
    assert not RESERVED.holds("dog")
    assert ReservedTokens({"<unk>": 2, "<bos>": 1, "<pad>": 0}) == RESERVED
    assert hash(RESERVED) == hash(ReservedTokens({"<pad>": 0, "<bos>": 1, "<unk>": 2}))
    assert "<unk>" in repr(RESERVED)


def test_a_reserved_token_that_was_not_set_aside_cannot_be_read() -> None:
    with pytest.raises(KeyError, match="not reserved"):
        RESERVED.index_of("<mask>")


def test_the_unknown_token_is_one_the_owner_set_aside() -> None:
    with pytest.raises(ValueError, match="not set"):
        vocabulary("<mask>")


def test_the_free_indices_go_to_the_vocabularys_own_tokens_in_order() -> None:
    built = vocabulary()

    assert built.size == 6
    assert [built.index_of(word) for word in WORDS] == [3, 4, 5]
    assert built.tokens == WORDS
    assert built.reserved == RESERVED
    assert built.holds("dog")
    assert not built.holds("<unk>")
    assert built.index_of("<bos>") == 1
    assert "unknown_index=2" in repr(built)


def test_a_vocabulary_refuses_a_token_it_could_not_index() -> None:
    with pytest.raises(ValueError, match="twice"):
        TokenVocabulary(("dog", "dog"), reserved=RESERVED, unknown="<unk>")

    with pytest.raises(ValueError, match="both"):
        TokenVocabulary(("<pad>",), reserved=RESERVED, unknown="<unk>")

    with pytest.raises(ValueError, match="no row has"):
        TokenVocabulary(
            ("dog",), reserved=ReservedTokens({"<unk>": 9}), unknown="<unk>"
        )

    with pytest.raises(ValueError, match="empty"):
        TokenVocabulary(("",), reserved=RESERVED, unknown="<unk>")


def test_a_report_accounts_for_every_row_it_read() -> None:
    report = LoadReport(rows_read=5, rows_matched=3, rows_dropped=2)

    assert (report.rows_read, report.rows_matched, report.rows_dropped) == (5, 3, 2)
    assert report == LoadReport(rows_read=5, rows_matched=3, rows_dropped=2)
    assert hash(report) == hash(LoadReport(rows_read=5, rows_matched=3, rows_dropped=2))
    assert "rows_matched=3" in repr(report)

    with pytest.raises(ValueError, match="do not account"):
        LoadReport(rows_read=5, rows_matched=3, rows_dropped=1)

    with pytest.raises(ValueError, match="negative"):
        LoadReport(rows_read=1, rows_matched=2, rows_dropped=-1)


def test_the_glove_loader_reports_what_it_read_matched_and_dropped(
    tmp_path: Path,
) -> None:
    built = vocabulary()
    embedder, report = from_glove_text(
        written(tmp_path, "vectors.txt", GLOVE),
        vocabulary=built,
        component=EMBEDDER,
        trainable=False,
    )

    assert (report.rows_read, report.rows_matched, report.rows_dropped) == (5, 3, 2)
    assert embedder.output_width == 2
    assert embedder.encode_words([["dog"]]).vectors.tensor.tolist() == [[[1.0, 2.0]]]
    assert embedder.encode_words([["runs"]]).vectors.tensor.tolist() == [[[9.0, 10.0]]]
    assert embedder.encode_words([["fox"]]).vectors.tensor.tolist() == [[[0.0, 0.0]]]
    assert embedder.load_report == report
    assert report.measurements() == {
        "static_vectors/rows_read": 5.0,
        "static_vectors/rows_matched": 3.0,
        "static_vectors/rows_dropped": 2.0,
    }
    assert embedder.report() == report.measurements()


def test_unmatched_rows_are_initialized_before_loaded_rows_are_overlaid(
    tmp_path: Path,
) -> None:
    embedder, _ = from_glove_text(
        written(tmp_path, "vectors.txt", "dog 1.0 2.0"),
        vocabulary=vocabulary(),
        component=EMBEDDER,
        trainable=False,
        oov_init="normal",
        oov_init_std=0.5,
        seed=17,
    )

    assert embedder.encode_words([["dog"]]).vectors.tensor.tolist() == [[[1.0, 2.0]]]
    assert not torch.equal(
        embedder.encode_words([["fox"]]).vectors.tensor,
        torch.zeros(1, 1, 2),
    )


def test_lowercase_matching_refuses_ambiguous_vocabulary_and_file_rows(
    tmp_path: Path,
) -> None:
    ambiguous = TokenVocabulary(("Dog", "dog"), reserved=RESERVED, unknown="<unk>")

    with pytest.raises(ValueError, match="lowercasing collides on vocabulary"):
        from_glove_text(
            written(tmp_path, "ambiguous.txt", "dog 1.0"),
            vocabulary=ambiguous,
            component=EMBEDDER,
            trainable=False,
            lowercase=True,
        )

    with pytest.raises(ValueError, match="lowercasing collides on vector rows"):
        from_glove_text(
            written(tmp_path, "rows.txt", "DOG 1.0\ndog 2.0"),
            vocabulary=vocabulary(),
            component=EMBEDDER,
            trainable=False,
            lowercase=True,
        )


def test_the_word2vec_loader_reads_the_same_rows_behind_a_header(
    tmp_path: Path,
) -> None:
    built = vocabulary()
    text = f"5 2\n{GLOVE}"
    embedder, report = from_word2vec_text(
        written(tmp_path, "vectors.vec", text),
        vocabulary=built,
        component=EMBEDDER,
        trainable=False,
    )
    plain, other = from_glove_text(
        written(tmp_path, "vectors.txt", GLOVE),
        vocabulary=built,
        component=EMBEDDER,
        trainable=False,
    )

    assert report == other
    assert torch.equal(
        embedder.encode_words([list(WORDS)]).vectors.tensor,
        plain.encode_words([list(WORDS)]).vectors.tensor,
    )


@pytest.mark.parametrize("layout", ["glove", "word2vec"])
def test_each_text_loader_honors_the_requested_gradient_policy(
    layout: str, tmp_path: Path
) -> None:
    text = GLOVE if layout == "glove" else f"5 2\n{GLOVE}"
    path = written(tmp_path, f"vectors-{layout}.txt", text)
    loader = from_glove_text if layout == "glove" else from_word2vec_text
    frozen, _ = loader(
        path, vocabulary=vocabulary(), component=EMBEDDER, trainable=False
    )
    trainable, _ = loader(
        path,
        vocabulary=vocabulary(),
        component=EMBEDDER,
        trainable=True,
    )

    trainable.encode_words([["dog"]]).vectors.tensor.sum().backward()

    assert all(not parameter.requires_grad for parameter in frozen.parameters())
    assert all(parameter.requires_grad for parameter in trainable.parameters())
    assert all(parameter.grad is not None for parameter in trainable.parameters())


def test_a_word2vec_header_that_the_rows_contradict_is_refused(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="declares 9 rows"):
        from_word2vec_text(
            written(tmp_path, "rows.vec", f"9 2\n{GLOVE}"),
            vocabulary=vocabulary(),
            component=EMBEDDER,
            trainable=False,
        )

    with pytest.raises(ValueError, match="width of 7"):
        from_word2vec_text(
            written(tmp_path, "width.vec", f"5 7\n{GLOVE}"),
            vocabulary=vocabulary(),
            component=EMBEDDER,
            trainable=False,
        )

    with pytest.raises(ValueError, match="opens with"):
        from_word2vec_text(
            written(tmp_path, "head.vec", f"5\n{GLOVE}"),
            vocabulary=vocabulary(),
            component=EMBEDDER,
            trainable=False,
        )


def test_a_file_no_table_could_be_built_from_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no vector"):
        from_glove_text(
            written(tmp_path, "empty.txt", "\n\n"),
            vocabulary=vocabulary(),
            component=EMBEDDER,
            trainable=False,
        )

    with pytest.raises(ValueError, match="has no vector"):
        from_glove_text(
            written(tmp_path, "bare.txt", "dog\n"),
            vocabulary=vocabulary(),
            component=EMBEDDER,
            trainable=False,
        )

    with pytest.raises(ValueError, match="columns"):
        from_glove_text(
            written(tmp_path, "ragged.txt", "dog 1.0 2.0\ncat 3.0\n"),
            vocabulary=vocabulary(),
            component=EMBEDDER,
            trainable=False,
        )


def test_the_unknown_row_is_the_one_the_vocabulary_names(tmp_path: Path) -> None:
    text = "<unk> -1.0 -1.0\ndog 1.0 2.0"
    here = TokenVocabulary(WORDS, reserved=RESERVED, unknown="<unk>")
    there = TokenVocabulary(
        WORDS, reserved=ReservedTokens({"<unk>": 0, "<pad>": 1}), unknown="<unk>"
    )
    path = written(tmp_path, "unk.txt", text)

    first, _ = from_glove_text(
        path, vocabulary=here, component=EMBEDDER, trainable=False
    )
    second, _ = from_glove_text(
        path, vocabulary=there, component=EMBEDDER, trainable=False
    )

    assert here.unknown_index != there.unknown_index
    assert first.encode_words([["zzz"]]).vectors.tensor.tolist() == [[[0.0, 0.0]]]
    assert second.encode_words([["zzz"]]).vectors.tensor.tolist() == [[[0.0, 0.0]]]
    assert first.vocabulary.unknown_index == 2
    assert second.vocabulary.unknown_index == 0


def test_a_reserved_row_a_file_names_is_matched_like_any_other(
    tmp_path: Path,
) -> None:
    built = TokenVocabulary(
        ("dog",), reserved=ReservedTokens({"<unk>": 0}), unknown="<unk>"
    )
    embedder, report = from_glove_text(
        written(tmp_path, "vectors.txt", "<unk> -1.0\ndog 1.0\n"),
        vocabulary=built,
        component=EMBEDDER,
        trainable=False,
    )

    assert report.rows_matched == 1
    assert report.rows_dropped == 1
    assert embedder.encode_words([["zzz"]]).vectors.tensor.tolist() == [[[0.0]]]


def test_each_word_reaches_one_row_and_the_alignment_says_which(
    tmp_path: Path,
) -> None:
    embedder, _ = from_glove_text(
        written(tmp_path, "vectors.txt", GLOVE),
        vocabulary=vocabulary(),
        component=EMBEDDER,
        trainable=False,
    )
    encoding = embedder.encode_words([["dog", "runs"], ["cat"]])

    assert encoding.vectors.axes == ROW_AXES
    assert encoding.vectors.tensor.shape == (2, 2, 2)
    assert encoding.lengths.tolist() == [2, 1]
    assert encoding.alignments[0].rows == ((0, 1), (1, 2))
    assert encoding.alignments[1].rows == ((0, 1),)
    assert encoding.truncated == ((), ())
    assert encoding.vectors.mask is not None
    assert encoding.vectors.mask.tolist() == [[True, True], [True, False]]
    assert encoding.vectors.tensor[1, 1].tolist() == [0.0, 0.0]


def test_a_table_the_vocabulary_does_not_describe_is_refused() -> None:
    built = vocabulary()

    with pytest.raises(ValueError, match="table of 2"):
        StaticTokenEmbedder(
            torch.zeros(2, 3),
            vocabulary=built,
            component=EMBEDDER,
            trainable=False,
            window=None,
            load_report=None,
        )

    with pytest.raises(ValueError, match="rank 2"):
        StaticTokenEmbedder(
            torch.zeros(built.size),
            vocabulary=built,
            component=EMBEDDER,
            trainable=False,
            window=None,
            load_report=None,
        )

    with pytest.raises(ValueError, match="at least one column"):
        StaticTokenEmbedder(
            torch.zeros(built.size, 0),
            vocabulary=built,
            component=EMBEDDER,
            trainable=False,
            window=None,
            load_report=None,
        )

    with pytest.raises(TypeError, match="real numbers"):
        StaticTokenEmbedder(
            torch.zeros(built.size, 2, dtype=torch.int64),
            vocabulary=built,
            component=EMBEDDER,
            trainable=False,
            window=None,
            load_report=None,
        )


def test_a_static_table_takes_no_gradient_unless_it_was_asked_to() -> None:
    built = vocabulary()
    table = torch.arange(float(built.size * 2)).reshape(built.size, 2)
    still = StaticTokenEmbedder(
        table,
        vocabulary=built,
        component=EMBEDDER,
        trainable=False,
        window=None,
        load_report=None,
    )
    moving = StaticTokenEmbedder(
        table,
        vocabulary=built,
        component=EMBEDDER,
        trainable=True,
        window=None,
        load_report=None,
    )

    moving.encode_words([["dog"]]).vectors.tensor.sum().backward()

    assert all(not parameter.requires_grad for parameter in still.parameters())
    assert still.encode_words([["dog"]]).vectors.tensor.grad_fn is None
    assert all(parameter.requires_grad for parameter in moving.parameters())
    assert all(parameter.grad is not None for parameter in moving.parameters())
    assert torch.equal(
        still.encode_words([["dog"]]).vectors.tensor,
        moving.encode_words([["dog"]]).vectors.tensor.detach(),
    )


def test_an_empty_batch_still_carries_a_mask_and_an_alignment(
    tmp_path: Path,
) -> None:
    embedder, _ = from_glove_text(
        written(tmp_path, "vectors.txt", GLOVE),
        vocabulary=vocabulary(),
        component=EMBEDDER,
        trainable=False,
    )
    encoding = embedder.encode_words([])

    assert encoding.vectors.tensor.shape == (0, 0, 2)
    assert encoding.alignments == ()


def test_an_embedder_without_a_window_embeds_an_item_of_any_length_whole() -> None:
    encoding = embedder().encode_words([WORDS * 5])

    assert embedder().window is None
    assert encoding.lengths.tolist() == [15]
    assert encoding.alignments[0].is_total
    assert encoding.truncation == (TruncationReport((), ()),)


def test_a_window_that_refuses_overflow_returns_no_tensor_for_the_batch() -> None:
    bounded = embedder(ContextWindow(2, overflow=REFUSE))

    with pytest.raises(ContextOverflowError, match="3 words needs 3 rows") as caught:
        bounded.encode_words([["dog"], ["dog", "cat", "runs"]])

    assert caught.value.words == ("dog", "cat", "runs")
    assert bounded.encode_words([["dog"], ["cat", "runs"]]).lengths.tolist() == [1, 2]


def test_a_window_that_reports_overflow_names_the_dropped_suffix() -> None:
    bounded = embedder(ContextWindow(2, overflow=REPORT))

    encoding = bounded.encode_words([["dog", "cat", "runs", "zzz"], ["cat"]])

    assert encoding.vectors.tensor.shape == (2, 2, 2)
    assert encoding.lengths.tolist() == [2, 1]
    assert encoding.alignments[0].rows == ((0, 1), (1, 2), None, None)
    assert encoding.truncated == ((2, 3), ())
    assert encoding.truncation == (
        TruncationReport((2, 3), ("runs", "zzz")),
        TruncationReport((), ()),
    )
    assert encoding.words == (("dog", "cat", "runs", "zzz"), ("cat",))


def test_a_two_sentence_batch_of_unequal_length_pools_under_its_mask() -> None:
    pooler = MeanSubwordPooler(width=2, component=ComponentId("pooler"))
    encoding = embedder().encode_words([["dog", "cat", "runs"], ["runs"]])
    pooled = pooler.pool(encoding)
    mask = pooled.mask

    assert encoding.vectors.mask is not None
    assert encoding.vectors.mask.tolist() == [[True, True, True], [True, False, False]]
    assert mask is not None
    assert mask.tolist() == [[True, True, True], [True, False, False]]
    assert torch.equal(pooled.tensor[0], encoding.vectors.tensor[0])
    assert torch.equal(pooled.tensor[1, 0], encoding.vectors.tensor[1, 0])
    assert torch.equal(pooled.tensor[1, 1:], torch.zeros(2, 2))


@settings(max_examples=50)
@given(batch=items(), rows=st.integers(min_value=1, max_value=4))
def test_every_ragged_batch_declares_its_lengths_and_names_what_it_dropped(
    batch: tuple[tuple[str, ...], ...], rows: int
) -> None:
    encoding = embedder(ContextWindow(rows, overflow=REPORT)).encode_words(batch)
    mask = encoding.vectors.mask
    reports = encoding.truncation

    assert mask is not None
    assert reports is not None
    assert encoding.vectors.size("row") <= rows
    assert encoding.lengths.tolist() == mask.sum(dim=1).tolist()

    for item, words in enumerate(batch):
        kept = min(len(words), rows)
        alignment = encoding.alignments[item]

        assert int(encoding.lengths[item]) == kept
        assert alignment.word_count == len(words)
        assert alignment.rows == tuple((word, word + 1) for word in range(kept)) + (
            None,
        ) * (len(words) - kept)
        assert reports[item].words == words[kept:]
        assert reports[item].positions == tuple(range(kept, len(words)))
        assert not bool(mask[item, kept:].any())


def test_the_published_point_is_the_rows_a_pass_produced(tmp_path: Path) -> None:
    embedder, _ = from_glove_text(
        written(tmp_path, "vectors.txt", GLOVE),
        vocabulary=vocabulary(),
        component=EMBEDDER,
        trainable=False,
    )
    published = embedder.site_map()
    site = Site(EMBEDDER, Role.TOKEN_EMBEDDING)

    assert set(published) == {site}
    assert published.component == EMBEDDER
    assert published.kind is ComponentKind.TOKEN_ENCODER
    assert published[site] == SiteSpec(
        ROW_AXES, DTypeClass.REAL, differentiable=True, access=Access.OBSERVE
    )
    assert ensure_publishes(published) is published


@pytest.mark.parametrize(
    ("axes", "dtype"),
    [
        (("item", "width", "row"), DTypeClass.REAL),
        (ROW_AXES, DTypeClass.INTEGER),
    ],
)
def test_the_token_encoder_contract_refuses_a_mutated_real_site(
    axes: tuple[str, ...], dtype: DTypeClass
) -> None:
    published = embedder().site_map()
    site = Site(EMBEDDER, Role.TOKEN_EMBEDDING)
    mutated = SiteMap(
        published.component,
        published.kind,
        {
            site: SiteSpec(
                axes,
                dtype,
                differentiable=True,
                access=Access.OBSERVE,
            )
        },
    )

    with pytest.raises(ValueError, match=r"runs over|holds"):
        ensure_publishes(mutated)


@settings(max_examples=25)
@given(
    rows=st.lists(
        st.text(alphabet="abcd", min_size=1, max_size=3), min_size=1, max_size=8
    ),
    held=st.lists(
        st.text(alphabet="abcd", min_size=1, max_size=3), min_size=1, max_size=6
    ),
)
def test_every_row_read_is_one_matched_or_one_dropped(
    tmp_path_factory: pytest.TempPathFactory,
    rows: list[str],
    held: list[str],
) -> None:
    tokens = tuple(dict.fromkeys(held))
    built = TokenVocabulary(
        tokens, reserved=ReservedTokens({"<unk>": 0}), unknown="<unk>"
    )
    text = "\n".join(f"{token} {number}.0" for number, token in enumerate(rows))
    path = written(tmp_path_factory.mktemp("vectors"), "vectors.txt", text)

    _, report = from_glove_text(
        path, vocabulary=built, component=EMBEDDER, trainable=False
    )
    expected = sum(1 for token in rows if built.holds(token))

    assert report.rows_read == len(rows)
    assert report.rows_matched == expected
    assert report.rows_read == report.rows_matched + report.rows_dropped
