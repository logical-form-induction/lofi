"""Strict reading and CCGbank alignment of CoNLL-2008 dependency gold."""

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, final, override

from lofi._checks import ensure_name, ensure_positive
from lofi._frozen import FrozenValue
from lofi._types import ItemId
from lofi.data import AttachmentTree

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "AlignedConll2008Sentence",
    "CcgbankSentenceIdentity",
    "Conll2008AlignmentError",
    "Conll2008Sentence",
    "MalformedConll2008Error",
    "align_conll2008",
    "read_conll2008",
]

_COLUMN_COUNT = 10
_FORM = 1
_GPOS = 3
_HEAD = 8
_DEPREL = 9


class MalformedConll2008Error(ValueError):
    """Error raised when a CoNLL-2008 source violates the read boundary."""


class Conll2008AlignmentError(ValueError):
    """Error raised when dependency gold does not exactly align with CCGbank."""


class _Row(NamedTuple):
    """One validated token row before its sentence tree is built."""

    form: str
    pos: str
    head: int
    label: str


@final
class Conll2008Sentence(FrozenValue):
    """One dependency-gold sentence in its source-relative order.

    Parameters
    ----------
    ordinal : int
        One-based sentence position in the source file.
    tree : AttachmentTree
        Validated single-rooted dependency tree.
    pos : Sequence[str]
        Gold part of speech tags in token order.
    labels : Sequence[str]
        Gold dependency labels in token order.
    """

    _noun = "CoNLL-2008 sentence"
    __slots__ = ("_labels", "_ordinal", "_pos", "_tree")

    def __init__(
        self,
        ordinal: int,
        tree: AttachmentTree,
        pos: Sequence[str],
        labels: Sequence[str],
    ) -> None:
        if not isinstance(tree, AttachmentTree):
            msg = f"a CoNLL-2008 dependency tree is not one: {tree!r}"
            raise TypeError(msg)
        if tree.length == 0:
            msg = "a CoNLL-2008 sentence has no token"
            raise ValueError(msg)

        held_pos = tuple(
            ensure_name(f"the POS tag at position {position}", tag)
            for position, tag in enumerate(pos, start=1)
        )
        held_labels = tuple(
            ensure_name(f"the dependency label at position {position}", label)
            for position, label in enumerate(labels, start=1)
        )

        if len(held_pos) != tree.length:
            msg = f"{tree.length} tokens are given {len(held_pos)} POS tags"
            raise ValueError(msg)
        if len(held_labels) != tree.length:
            msg = f"{tree.length} tokens are given {len(held_labels)} dependency labels"
            raise ValueError(msg)

        object.__setattr__(
            self, "_ordinal", ensure_positive("a CoNLL-2008 sentence ordinal", ordinal)
        )
        object.__setattr__(self, "_tree", tree)
        object.__setattr__(self, "_pos", held_pos)
        object.__setattr__(self, "_labels", held_labels)

    @property
    def ordinal(self) -> int:
        """Return the one-based sentence position in the source."""
        return self._ordinal

    @property
    def tree(self) -> AttachmentTree:
        """Return the validated dependency tree."""
        return self._tree

    @property
    def forms(self) -> tuple[str, ...]:
        """Return surface forms in token order."""
        return self._tree.forms

    @property
    def pos(self) -> tuple[str, ...]:
        """Return gold part of speech tags in token order."""
        return self._pos

    @property
    def labels(self) -> tuple[str, ...]:
        """Return gold dependency labels in token order."""
        return self._labels

    def _fields(
        self,
    ) -> tuple[int, AttachmentTree, tuple[str, ...], tuple[str, ...]]:
        """Return fields in declaration order."""
        return self._ordinal, self._tree, self._pos, self._labels

    @override
    def __reduce__(
        self,
    ) -> tuple[
        type[Conll2008Sentence],
        tuple[int, AttachmentTree, tuple[str, ...], tuple[str, ...]],
    ]:
        """Return the reconstruction call."""
        return Conll2008Sentence, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another sentence has the same source fields."""
        if type(other) is not Conll2008Sentence:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the source fields."""
        return hash((Conll2008Sentence, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the reconstruction call."""
        return f"Conll2008Sentence{self._fields()!r}"


@final
class CcgbankSentenceIdentity(FrozenValue):
    """The ordered CCGbank identity and token fields used for exact alignment."""

    _noun = "CCGbank sentence identity"
    __slots__ = ("_forms", "_item_id", "_pos")

    def __init__(
        self,
        item_id: ItemId,
        forms: Sequence[str],
        pos: Sequence[str],
    ) -> None:
        held_forms = tuple(
            ensure_name(f"the CCGbank form at position {position}", form)
            for position, form in enumerate(forms, start=1)
        )
        held_pos = tuple(
            ensure_name(f"the CCGbank POS tag at position {position}", tag)
            for position, tag in enumerate(pos, start=1)
        )
        if not held_forms:
            msg = "a CCGbank sentence identity has no token"
            raise ValueError(msg)
        if len(held_forms) != len(held_pos):
            msg = f"{len(held_forms)} CCGbank forms are given {len(held_pos)} POS tags"
            raise ValueError(msg)

        object.__setattr__(
            self, "_item_id", ItemId(ensure_name("a CCGbank item id", item_id))
        )
        object.__setattr__(self, "_forms", held_forms)
        object.__setattr__(self, "_pos", held_pos)

    @property
    def item_id(self) -> ItemId:
        """Return the CCGbank sentence identifier."""
        return self._item_id

    @property
    def forms(self) -> tuple[str, ...]:
        """Return CCGbank surface forms in token order."""
        return self._forms

    @property
    def pos(self) -> tuple[str, ...]:
        """Return CCGbank gold part of speech tags in token order."""
        return self._pos

    def _fields(self) -> tuple[ItemId, tuple[str, ...], tuple[str, ...]]:
        """Return fields in declaration order."""
        return self._item_id, self._forms, self._pos

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another identity has the same fields."""
        if type(other) is not CcgbankSentenceIdentity:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the identity fields."""
        return hash((CcgbankSentenceIdentity, self._fields()))

    @override
    def __reduce__(
        self,
    ) -> tuple[
        type[CcgbankSentenceIdentity],
        tuple[ItemId, tuple[str, ...], tuple[str, ...]],
    ]:
        """Return the reconstruction call."""
        return CcgbankSentenceIdentity, self._fields()

    @override
    def __repr__(self) -> str:
        """Return the reconstruction call."""
        return f"CcgbankSentenceIdentity{self._fields()!r}"


@final
class AlignedConll2008Sentence(FrozenValue):
    """One CoNLL-2008 sentence attached to its exact CCGbank identifier."""

    _noun = "aligned CoNLL-2008 sentence"
    __slots__ = ("_item_id", "_sentence")

    def __init__(self, item_id: ItemId, sentence: Conll2008Sentence) -> None:
        if not isinstance(sentence, Conll2008Sentence):
            msg = f"aligned CoNLL-2008 gold is not a sentence: {sentence!r}"
            raise TypeError(msg)
        object.__setattr__(
            self, "_item_id", ItemId(ensure_name("an aligned CCGbank item id", item_id))
        )
        object.__setattr__(self, "_sentence", sentence)

    @property
    def item_id(self) -> ItemId:
        """Return the aligned CCGbank identifier."""
        return self._item_id

    @property
    def sentence(self) -> Conll2008Sentence:
        """Return the aligned dependency-gold sentence."""
        return self._sentence

    def _fields(self) -> tuple[ItemId, Conll2008Sentence]:
        """Return fields in declaration order."""
        return self._item_id, self._sentence

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another alignment has the same fields."""
        if type(other) is not AlignedConll2008Sentence:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the alignment fields."""
        return hash((AlignedConll2008Sentence, self._fields()))

    @override
    def __reduce__(
        self,
    ) -> tuple[type[AlignedConll2008Sentence], tuple[ItemId, Conll2008Sentence]]:
        """Return the reconstruction call."""
        return AlignedConll2008Sentence, self._fields()

    @override
    def __repr__(self) -> str:
        """Return the reconstruction call."""
        return f"AlignedConll2008Sentence{self._fields()!r}"


def _integer(value: str, subject: str, /) -> int:
    """Return a tabular integer or raise the corpus error."""
    try:
        return int(value)
    except ValueError as error:
        msg = f"{subject} is not an integer: {value!r}"
        raise MalformedConll2008Error(msg) from error


def _required(value: str, subject: str, /) -> str:
    """Return a populated gold field or raise the corpus error."""
    if not value or value == "_":
        msg = f"{subject} is missing"
        raise MalformedConll2008Error(msg)
    return value


def _sentence(rows: Sequence[_Row], ordinal: int, /) -> Conll2008Sentence:
    """Build one validated sentence from its token rows."""
    try:
        tree = AttachmentTree(
            tuple(row.form for row in rows),
            tuple(row.head for row in rows),
        )
    except (TypeError, ValueError) as error:
        msg = f"CoNLL-2008 sentence {ordinal} has an invalid dependency tree: {error}"
        raise MalformedConll2008Error(msg) from error

    return Conll2008Sentence(
        ordinal,
        tree,
        tuple(row.pos for row in rows),
        tuple(row.label for row in rows),
    )


def read_conll2008(source: Path, /) -> tuple[Conll2008Sentence, ...]:
    """Read one UTF-8 CoNLL-2008 dependency-gold file.

    Sentences are separated by exactly one blank line. A final sentence is
    retained whether or not the source ends with its blank-line separator.

    Parameters
    ----------
    source : Path
        Tabular CoNLL-2008 file.

    Raises
    ------
    ValueError
        If the path is not a file.
    MalformedConll2008Error
        If UTF-8 decoding, a token row, sentence separation, or dependency-tree
        validation fails.
    """
    path = Path(source)
    if not path.is_file():
        msg = f"a CoNLL-2008 source is not a readable file: {path}"
        raise ValueError(msg)

    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        msg = f"CoNLL-2008 source {path} is not UTF-8"
        raise MalformedConll2008Error(msg) from error

    sentences: list[Conll2008Sentence] = []
    rows: list[_Row] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        if line == "":
            if not rows:
                msg = (
                    f"CoNLL-2008 source {path} has an empty sentence "
                    f"at line {line_number}"
                )
                raise MalformedConll2008Error(msg)
            sentences.append(_sentence(rows, len(sentences) + 1))
            rows = []
            continue

        columns = line.split("\t")
        if len(columns) < _COLUMN_COUNT:
            msg = (
                f"CoNLL-2008 row {line_number} has {len(columns)} columns; "
                f"at least {_COLUMN_COUNT} are required"
            )
            raise MalformedConll2008Error(msg)

        identifier = _integer(columns[0], f"CoNLL-2008 row {line_number} ID")
        expected = len(rows) + 1
        if identifier != expected:
            msg = (
                f"CoNLL-2008 row {line_number} has ID {identifier}; "
                f"expected consecutive ID {expected}"
            )
            raise MalformedConll2008Error(msg)

        rows.append(
            _Row(
                _required(columns[_FORM], f"CoNLL-2008 row {line_number} FORM"),
                _required(columns[_GPOS], f"CoNLL-2008 row {line_number} GPOS"),
                _integer(columns[_HEAD], f"CoNLL-2008 row {line_number} HEAD"),
                _required(columns[_DEPREL], f"CoNLL-2008 row {line_number} DEPREL"),
            )
        )

    if rows:
        sentences.append(_sentence(rows, len(sentences) + 1))

    if not sentences:
        msg = f"CoNLL-2008 source {path} contains no sentence"
        raise MalformedConll2008Error(msg)

    return tuple(sentences)


def align_conll2008(
    sentences: Sequence[Conll2008Sentence],
    ccgbank: Sequence[CcgbankSentenceIdentity],
    /,
) -> tuple[AlignedConll2008Sentence, ...]:
    """Align ordered dependency gold to ordered CCGbank identities exactly.

    Parameters
    ----------
    sentences : Sequence[Conll2008Sentence]
        Dependency-gold sentences in source order.
    ccgbank : Sequence[CcgbankSentenceIdentity]
        CCGbank identifiers, forms, and gold POS in the corresponding order.

    Raises
    ------
    Conll2008AlignmentError
        If either sequence is empty, counts differ, a CCGbank identifier
        repeats, or a source position has different forms or POS tags.
    """
    held_sentences = tuple(sentences)
    held_ccgbank = tuple(ccgbank)

    if not held_sentences or not held_ccgbank:
        msg = "CoNLL-2008 alignment requires nonempty gold and CCGbank sequences"
        raise Conll2008AlignmentError(msg)
    if len(held_sentences) != len(held_ccgbank):
        msg = (
            f"CoNLL-2008 gold has {len(held_sentences)} sentences and CCGbank "
            f"has {len(held_ccgbank)}"
        )
        raise Conll2008AlignmentError(msg)

    for position, sentence in enumerate(held_sentences, start=1):
        if not isinstance(sentence, Conll2008Sentence):
            msg = f"CoNLL-2008 gold at position {position} is not a sentence"
            raise TypeError(msg)
    for position, identity in enumerate(held_ccgbank, start=1):
        if not isinstance(identity, CcgbankSentenceIdentity):
            msg = f"CCGbank identity at position {position} is not one"
            raise TypeError(msg)

    identifiers = tuple(identity.item_id for identity in held_ccgbank)
    if len(set(identifiers)) != len(identifiers):
        msg = "ordered CCGbank identities repeat an item id"
        raise Conll2008AlignmentError(msg)

    aligned: list[AlignedConll2008Sentence] = []
    for position, (sentence, identity) in enumerate(
        zip(held_sentences, held_ccgbank, strict=True), start=1
    ):
        if sentence.ordinal != position:
            msg = (
                f"CoNLL-2008 sentence at position {position} has source ordinal "
                f"{sentence.ordinal}"
            )
            raise Conll2008AlignmentError(msg)
        if sentence.forms != identity.forms:
            msg = (
                f"CoNLL-2008 forms at position {position} do not match CCGbank "
                f"item {identity.item_id!r}"
            )
            raise Conll2008AlignmentError(msg)
        if sentence.pos != identity.pos:
            msg = (
                f"CoNLL-2008 POS tags at position {position} do not match CCGbank "
                f"item {identity.item_id!r}"
            )
            raise Conll2008AlignmentError(msg)
        aligned.append(AlignedConll2008Sentence(identity.item_id, sentence))

    return tuple(aligned)
