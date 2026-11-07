"""Tests for the strict CoNLL-2008 dependency-gold boundary."""

from copy import copy, deepcopy
from pickle import dumps, loads
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from lofi._types import ItemId
from lofi.data import AttachmentTree
from lofi.data.corpora.conll2008 import (
    CcgbankSentenceIdentity,
    Conll2008AlignmentError,
    Conll2008Sentence,
    MalformedConll2008Error,
    align_conll2008,
    read_conll2008,
)


def _row(
    identifier: str,
    form: str,
    pos: str,
    head: str,
    label: str,
    *extra: str,
) -> str:
    """Return one ten-column gold row with optional trailing columns."""
    return "\t".join(
        (identifier, form, "_", pos, "_", "_", "_", "_", head, label, *extra)
    )


FIRST = "\n".join(
    (
        _row("1", "The", "DT", "2", "det", "retained-extra"),
        _row("2", "sleeps", "VBZ", "0", "root"),
    )
)
SECOND = "\n".join(
    (
        _row("1", "Dogs", "NNS", "2", "nsubj"),
        _row("2", "run", "VBP", "0", "root"),
    )
)


@pytest.mark.parametrize("terminal", ["", "\n\n"])
def test_the_reader_retains_every_sentence_with_or_without_a_terminal_separator(
    tmp_path: Path, terminal: str
) -> None:
    source = tmp_path / "gold.conll"
    source.write_text(f"{FIRST}\n\n{SECOND}{terminal}", encoding="utf-8", newline="\n")

    first, second = read_conll2008(source)

    assert first.ordinal == 1
    assert first.forms == ("The", "sleeps")
    assert first.pos == ("DT", "VBZ")
    assert first.tree.heads == (2, 0)
    assert first.labels == ("det", "root")
    assert second.ordinal == 2
    assert second.forms == ("Dogs", "run")
    assert second.pos == ("NNS", "VBP")
    assert second.tree.heads == (2, 0)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("", "contains no sentence"),
        ("\n", "empty sentence"),
        (f"{FIRST}\n\n\n", "empty sentence"),
        ("1\tword", "at least 10"),
        (_row("x", "word", "NN", "0", "root"), "ID.*not an integer"),
        (_row("2", "word", "NN", "0", "root"), "expected consecutive ID 1"),
        (
            "\n".join(
                (
                    _row("1", "a", "NN", "0", "root"),
                    _row("1", "b", "NN", "1", "dep"),
                )
            ),
            "expected consecutive ID 2",
        ),
        (_row("1", "", "NN", "0", "root"), "FORM.*missing"),
        (_row("1", "word", "_", "0", "root"), "GPOS.*missing"),
        (_row("1", "word", "NN", "x", "root"), "HEAD.*not an integer"),
        (_row("1", "word", "NN", "0", "_"), "DEPREL.*missing"),
        (_row("1", "word", "NN", "2", "dep"), "outside 1 tokens"),
        (_row("1", "word", "NN", "1", "dep"), "heads itself"),
        (
            "\n".join(
                (
                    _row("1", "a", "NN", "2", "dep"),
                    _row("2", "b", "NN", "1", "dep"),
                )
            ),
            "0 tokens attach to the root",
        ),
        (
            "\n".join(
                (
                    _row("1", "a", "NN", "0", "root"),
                    _row("2", "b", "NN", "0", "root"),
                )
            ),
            "2 tokens attach to the root",
        ),
        (
            "\n".join(
                (
                    _row("1", "a", "NN", "0", "root"),
                    _row("2", "b", "NN", "3", "dep"),
                    _row("3", "c", "NN", "2", "dep"),
                )
            ),
            "does not reach the root",
        ),
    ],
)
def test_the_reader_refuses_malformed_rows_sentences_and_trees(
    tmp_path: Path, document: str, message: str
) -> None:
    source = tmp_path / "bad.conll"
    source.write_text(document, encoding="utf-8", newline="\n")

    with pytest.raises(MalformedConll2008Error, match=message):
        read_conll2008(source)


def test_the_reader_refuses_non_utf8_and_a_missing_source(tmp_path: Path) -> None:
    source = tmp_path / "bad.conll"
    source.write_bytes(b"\xff")

    with pytest.raises(MalformedConll2008Error, match="not UTF-8"):
        read_conll2008(source)

    with pytest.raises(ValueError, match="not a readable file"):
        read_conll2008(tmp_path / "missing.conll")


def test_a_sentence_is_frozen_and_validates_its_parallel_fields() -> None:
    tree = AttachmentTree(("a",), (0,))
    sentence = Conll2008Sentence(1, tree, ("NN",), ("root",))

    assert copy(sentence) is sentence
    assert deepcopy(sentence) is sentence
    assert sentence == Conll2008Sentence(1, tree, ("NN",), ("root",))
    assert hash(sentence) == hash(Conll2008Sentence(1, tree, ("NN",), ("root",)))
    assert loads(dumps(sentence)) == sentence
    assert sentence.__eq__(object()) is NotImplemented

    field = "_ordinal"
    with pytest.raises(AttributeError, match="immutable"):
        setattr(sentence, field, 2)
    with pytest.raises(ValueError, match="POS tags"):
        Conll2008Sentence(1, tree, (), ("root",))
    with pytest.raises(ValueError, match="dependency labels"):
        Conll2008Sentence(1, tree, ("NN",), ())
    with pytest.raises(ValueError, match="has no token"):
        Conll2008Sentence(1, AttachmentTree((), ()), (), ())


def _sentences() -> tuple[Conll2008Sentence, ...]:
    return (
        Conll2008Sentence(
            1,
            AttachmentTree(("The", "sleeps"), (2, 0)),
            ("DT", "VBZ"),
            ("det", "root"),
        ),
        Conll2008Sentence(
            2,
            AttachmentTree(("Dogs", "run"), (2, 0)),
            ("NNS", "VBP"),
            ("nsubj", "root"),
        ),
    )


def _identities() -> tuple[CcgbankSentenceIdentity, ...]:
    return (
        CcgbankSentenceIdentity(ItemId("wsj_0001.1"), ("The", "sleeps"), ("DT", "VBZ")),
        CcgbankSentenceIdentity(ItemId("wsj_0001.2"), ("Dogs", "run"), ("NNS", "VBP")),
    )


def test_exact_alignment_attaches_ordered_ccgbank_identifiers() -> None:
    aligned = align_conll2008(_sentences(), _identities())

    assert tuple(entry.item_id for entry in aligned) == (
        ItemId("wsj_0001.1"),
        ItemId("wsj_0001.2"),
    )
    assert tuple(entry.sentence.ordinal for entry in aligned) == (1, 2)
    assert aligned[0].sentence.tree.heads == (2, 0)
    assert loads(dumps(aligned)) == aligned


def test_exact_alignment_refuses_counts_order_forms_pos_and_repeated_ids() -> None:
    sentences = _sentences()
    identities = _identities()

    with pytest.raises(Conll2008AlignmentError, match="nonempty"):
        align_conll2008((), ())
    with pytest.raises(Conll2008AlignmentError, match=r"has 2 sentences.*has 1"):
        align_conll2008(sentences, identities[:1])
    with pytest.raises(Conll2008AlignmentError, match="forms at position 1"):
        align_conll2008(sentences, tuple(reversed(identities)))
    with pytest.raises(Conll2008AlignmentError, match="POS tags at position 1"):
        align_conll2008(
            sentences,
            (
                CcgbankSentenceIdentity(
                    identities[0].item_id, identities[0].forms, ("XX", "VBZ")
                ),
                identities[1],
            ),
        )
    with pytest.raises(Conll2008AlignmentError, match="repeat an item id"):
        align_conll2008(
            sentences,
            (
                identities[0],
                CcgbankSentenceIdentity(
                    identities[0].item_id, identities[1].forms, identities[1].pos
                ),
            ),
        )
    with pytest.raises(Conll2008AlignmentError, match="source ordinal 2"):
        align_conll2008(
            (
                Conll2008Sentence(
                    2,
                    sentences[0].tree,
                    sentences[0].pos,
                    sentences[0].labels,
                ),
            ),
            identities[:1],
        )

    with pytest.raises(TypeError, match="gold at position 1"):
        align_conll2008(
            cast("tuple[Conll2008Sentence, ...]", (object(),)), identities[:1]
        )
    with pytest.raises(TypeError, match="identity at position 1"):
        align_conll2008(
            sentences[:1], cast("tuple[CcgbankSentenceIdentity, ...]", (object(),))
        )
