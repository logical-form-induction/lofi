"""CCGbank AUTO and PARG loading with conventional section splits.

Every file is read through the registered ``ccgbank-auto`` and
``ccgbank-parg`` codecs' own ``parse``, which returns the derivations and
dependencies directly; the codecs' ``decode`` into Layers records translates
the same parsed value.
"""

from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, cast

from lofi._types import ItemId
from lofi.data import (
    Derivation,
    DerivationNode,
    Example,
    GoldDerivation,
    Item,
    LoadReport,
)
from lofi.data.corpora.ccgbank._auto import CcgbankAutoCodec
from lofi.data.corpora.ccgbank._parg import PargCodec
from lofi.data.corpora.ccgbank.roundtrip import gold_sentence

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from lofi.data import SupervisionArm
    from lofi.data.corpora.ccgbank._auto_parse import AutoDerivation
    from lofi.data.corpora.ccgbank._parg_parse import PargSentence

__all__ = [
    "CCGBANK_SPLITS",
    "CcgbankCorpusReader",
    "CcgbankSplit",
    "UnknownWordPolicy",
]

type CcgbankSplit = Literal["train", "validation", "development", "test"]
"""A conventional CCGbank section split."""

type UnknownWordPolicy = Literal["skip", "assign"]
"""How a held-out sentence containing a training-unseen word is handled."""

CCGBANK_SPLITS: Final[tuple[CcgbankSplit, ...]] = (
    "train",
    "validation",
    "development",
    "test",
)
"""The conventional split order used by a full CCGbank load."""

_SECTION_SPLIT: Final[Mapping[str, CcgbankSplit]] = MappingProxyType(
    {
        **{f"{section:02d}": "train" for section in range(2, 22)},
        "00": "validation",
        "22": "development",
        "23": "test",
    }
)

_UNKNOWN_REASON: Final = "word absent from CCGbank training sections"


class _Source[T]:
    """Repeatable finite access to an immutable tuple."""

    __slots__ = ("_values",)

    def __init__(self, values: tuple[T, ...], /) -> None:
        self._values = values

    def __iter__(self) -> Iterator[T]:
        """Return an iterator at the start of the tuple."""
        return iter(self._values)

    def __len__(self) -> int:
        """Return the tuple's length."""
        return len(self._values)


class _LocatedAuto(NamedTuple):
    """One AUTO file beside the split its section assigns it to."""

    path: Path
    split: CcgbankSplit


def _section(path: Path, root: Path, /) -> str | None:
    """Return the conventional two-digit section containing a corpus file."""
    for part in reversed(path.relative_to(root).parts[:-1]):
        if len(part) == 2 and part.isascii() and part.isdecimal():
            return part

    stem = path.stem
    if stem.startswith("wsj_") and len(stem) >= 6 and stem[4:6].isdecimal():
        return stem[4:6]

    return None


def _located(root: Path, /) -> tuple[_LocatedAuto, ...]:
    """Return AUTO files in stable source order with their conventional splits."""
    if root.is_file():
        if root.suffix.casefold() != ".auto":
            msg = f"a CCGbank source file is not AUTO data: {root}"
            raise ValueError(msg)
        return (_LocatedAuto(root, "train"),)

    if not root.is_dir():
        msg = f"a CCGbank root is not a readable file or directory: {root}"
        raise ValueError(msg)

    located: list[_LocatedAuto] = []
    for path in sorted(root.rglob("*.auto")):
        section = _section(path, root)
        split = None if section is None else _SECTION_SPLIT.get(section)
        if split is not None:
            located.append(_LocatedAuto(path, split))

    if not located:
        msg = f"a CCGbank root contains no AUTO files in sections 00 or 02--23: {root}"
        raise ValueError(msg)

    return tuple(located)


def _identifier(header: tuple[tuple[str, str], ...], fallback: str, /) -> ItemId:
    """Return a derivation's released identifier, or its stable file position."""
    identifier = dict(header).get("ID", fallback)
    return ItemId(identifier)


def _parg_files(root: Path, /) -> Mapping[str, tuple[Path, ...]]:
    """Index optional PARG files by released file stem."""
    if root.is_file():
        candidate = root.with_suffix(".parg")
        return MappingProxyType(
            {root.stem: (candidate,)} if candidate.is_file() else {}
        )

    indexed: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*.parg")):
        indexed.setdefault(path.stem, []).append(path)
    return MappingProxyType({stem: tuple(paths) for stem, paths in indexed.items()})


def _lexicalized(
    derivation: Derivation[str],
    categories: tuple[str, ...],
    unseen_positions: frozenset[int],
    unknown_category: str | None,
    /,
) -> Derivation[str]:
    """Return a derivation whose lexical leaves bear declared category text."""
    nodes = {
        node_id: (
            DerivationNode(
                node.span,
                (
                    cast("str", unknown_category)
                    if node.span.start in unseen_positions
                    else categories[node.span.start]
                ),
            )
            if node.is_lexical
            else node
        )
        for node_id, node in derivation.nodes.items()
    }
    return Derivation(derivation.root, nodes)


class CcgbankCorpusReader:
    """Read CCGbank derivations and dependencies under conventional splits.

    PARG is optional, but every paired file must name exactly the AUTO
    sentences in its file. Training-section words define the held-out
    vocabulary. By default, a held-out sentence containing an unseen word is
    skipped and counted in the load report.
    """

    __slots__ = ("_examples", "_parg", "_report", "_splits")

    def __init__(
        self,
        root: Path,
        /,
        *,
        splits: tuple[CcgbankSplit, ...] = CCGBANK_SPLITS,
        unknown_words: UnknownWordPolicy = "skip",
        unknown_category: str | None = None,
    ) -> None:
        if not splits or len(set(splits)) != len(splits):
            msg = "a CCGbank load names a nonempty sequence of distinct splits"
            raise ValueError(msg)
        if any(split not in CCGBANK_SPLITS for split in splits):
            msg = f"a CCGbank load names an unknown split: {splits!r}"
            raise ValueError(msg)
        if unknown_words == "assign" and not unknown_category:
            msg = "the assign unknown-word policy needs a category"
            raise ValueError(msg)
        if unknown_words == "skip" and unknown_category is not None:
            msg = "the skip unknown-word policy takes no category"
            raise ValueError(msg)

        source = Path(root)
        auto = CcgbankAutoCodec()
        parg = PargCodec()
        located = _located(source)
        paired = _parg_files(source)
        parsed: list[tuple[CcgbankSplit, str, AutoDerivation]] = []
        dependencies: dict[ItemId, PargSentence] = {}
        training_forms: set[str] = set()

        for entry in located:
            derivations = auto.parse(entry.path.read_bytes())
            identifiers: set[str] = set()

            for position, derivation in enumerate(derivations):
                fallback = f"{entry.path.name}:{position}"
                item_id = _identifier(derivation.header, fallback)
                identifier = str(item_id)
                if identifier in identifiers:
                    msg = f"{entry.path} repeats AUTO identifier {identifier!r}"
                    raise ValueError(msg)
                identifiers.add(identifier)
                parsed.append((entry.split, identifier, derivation))
                if entry.split == "train" or source.is_file():
                    training_forms.update(
                        token.surface for token in gold_sentence(derivation.tree).tokens
                    )

            for parg_path in paired.get(entry.path.stem, ()):
                for sentence in parg.parse(parg_path.read_bytes()):
                    if sentence.identifier not in identifiers:
                        msg = (
                            f"{parg_path} names {sentence.identifier!r}, which "
                            f"{entry.path} does not"
                        )
                        raise ValueError(msg)
                    item_id = ItemId(sentence.identifier)
                    if item_id in dependencies:
                        msg = f"PARG data repeats sentence {sentence.identifier!r}"
                        raise ValueError(msg)
                    dependencies[item_id] = sentence

        accepted: dict[CcgbankSplit, list[Example[GoldDerivation[str]]]] = {
            split: [] for split in splits
        }
        skipped = 0
        seen = 0

        for split, identifier, derivation in parsed:
            if split not in accepted:
                continue
            prepared = gold_sentence(derivation.tree)
            seen += 1
            unseen = frozenset(
                token.index
                for token in prepared.tokens
                if token.surface not in training_forms
            )
            if unseen and unknown_words == "skip":
                skipped += 1
                continue

            item = Item(
                ItemId(identifier),
                prepared.tokens,
                text=" ".join(token.surface for token in prepared.tokens),
                languages=("en",),
                split=split,
            )
            accepted[split].append(
                Example(
                    item,
                    GoldDerivation(
                        _lexicalized(
                            prepared.derivation,
                            prepared.categories,
                            unseen,
                            unknown_category,
                        )
                    ),
                )
            )

        self._splits = splits
        self._examples = MappingProxyType(
            {split: tuple(accepted[split]) for split in splits}
        )
        self._parg = MappingProxyType(dependencies)
        self._report = LoadReport(
            seen,
            seen - skipped,
            {} if not skipped else {_UNKNOWN_REASON: skipped},
        )

    @property
    def name(self) -> str:
        """Return the reader's registry name."""
        return "ccgbank"

    @property
    def grammar(self) -> None:
        """Return no grammar because the dataset declaration supplies it."""
        return None

    @property
    def lexicon(self) -> None:
        """Return no lexical source because the dataset derives it by split."""
        return None

    @property
    def dependencies(self) -> Mapping[ItemId, PargSentence]:
        """Return paired PARG sentences by released sentence identifier."""
        return self._parg

    def splits(self) -> tuple[str, ...]:
        """Return selected splits in declaration order."""
        return self._splits

    def items(self, split: str, /) -> _Source[Item]:
        """Return repeatable items for one selected split."""
        held = self._examples.get(cast("CcgbankSplit", split), ())
        return _Source(tuple(example.item for example in held))

    def examples(self, split: str, /) -> _Source[Example[GoldDerivation[str]]]:
        """Return repeatable gold derivations for one selected split."""
        return _Source(self._examples.get(cast("CcgbankSplit", split), ()))

    def supervision_kinds(self) -> frozenset[SupervisionArm]:
        """Return the single supervision arm emitted by this reader."""
        return frozenset((GoldDerivation,))

    def report(self) -> LoadReport:
        """Return emitted and unknown-word skip counts."""
        return self._report
