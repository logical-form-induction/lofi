"""Static token embedding loaders."""

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, override

import torch
from torch import nn

from lofi._alignment import Alignment
from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.nn._values import Encoded, TokenEncoding
from lofi.nn.policy import place
from lofi.nn.sites import (
    Access,
    ComponentKind,
    DTypeClass,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)
from lofi.registry import ComponentPlugin, Registry, RegistryError
from lofi.spec.tokens.static import StaticTokenEncoderSettings, StaticTokenEncoderSpec

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from torch import Tensor

    from lofi.nn.build import BuildContext
    from lofi.nn.sites import ComponentId
    from lofi.nn.tokens.window import ContextWindow

__all__ = [
    "NORMAL_OOV",
    "ZERO_OOV",
    "LoadReport",
    "OovInitializationPlugin",
    "ReservedTokens",
    "StaticTokenEmbedder",
    "TokenVocabulary",
    "VectorFormatPlugin",
    "from_glove_text",
    "from_word2vec_text",
    "oov_initialization",
    "vector_format",
]

_ROW_AXES: Final[tuple[str, ...]] = ("item", "row", "width")
"""Token-encoder output axes."""

_TABLE_RANK: Final[int] = 2
"""The embedding-table rank."""

_HEADER_FIELDS: Final[int] = 2
"""The word2vec header width."""


class ReservedTokens:
    """Maps reserved tokens to embedding indices.

    Parameters
    ----------
    assignments : Mapping[str, int]
        Each reserved token under the index its owner gave it.

    Raises
    ------
    TypeError
        If a name is not a string or an index is not an integer.
    ValueError
        If a name is empty, an index is negative, or two tokens share an index.
    """

    __slots__ = ("_assignments",)

    def __init__(self, assignments: Mapping[str, int]) -> None:
        checked = {
            ensure_name("a reserved token", name): ensure_non_negative(
                f"the index of the reserved token {name!r}", index
            )
            for name, index in assignments.items()
        }
        taken: dict[int, str] = {}

        for name, index in checked.items():
            first = taken.setdefault(index, name)

            if first != name:
                msg = (
                    f"the reserved tokens {first!r} and {name!r} were both "
                    f"assigned index {index}"
                )
                raise ValueError(msg)

        self._assignments = checked

    @property
    def names(self) -> tuple[str, ...]:
        """Return reserved tokens in embedding-index order."""
        return tuple(sorted(self._assignments, key=self._assignments.__getitem__))

    @property
    def indices(self) -> frozenset[int]:
        """Return the indices these tokens occupy."""
        return frozenset(self._assignments.values())

    @property
    def count(self) -> int:
        """Return the number of reserved tokens."""
        return len(self._assignments)

    def index_of(self, name: str) -> int:
        """Return the index one reserved token was assigned.

        Parameters
        ----------
        name : str
            The reserved token.

        Returns
        -------
        int
            Its index.

        Raises
        ------
        KeyError
            If the token was not set aside.
        """
        try:
            return self._assignments[name]

        except KeyError as error:
            msg = f"{name!r} is not reserved here; {self.names} are"
            raise KeyError(msg) from error

    def holds(self, name: str) -> bool:
        """Return whether one token was set aside.

        Parameters
        ----------
        name : str
            The token to look for.

        Returns
        -------
        bool
            Whether it has a reserved index.
        """
        return name in self._assignments

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value assigns the same tokens the same indices.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not ReservedTokens:
            return NotImplemented

        return self._assignments == other._assignments

    @override
    def __hash__(self) -> int:
        """Return a hash over the token-to-index assignment."""
        return hash((ReservedTokens, frozenset(self._assignments.items())))

    @override
    def __repr__(self) -> str:
        """Return an evaluable representation."""
        return f"ReservedTokens({self._assignments!r})"


class TokenVocabulary:
    """Maps tokens to embedding rows.

    Parameters
    ----------
    tokens : Sequence[str]
        The vocabulary's own tokens, in the order they take the free indices.
    reserved : ReservedTokens
        The tokens set aside, each at the index its owner gave it.
    unknown : str
        Which reserved token a token outside the vocabulary is read as.

    Raises
    ------
    TypeError
        If a token is not a string.
    ValueError
        If tokens are empty, repeated, or reserved, or reserved indices and the unknown
        token are invalid.
    """

    __slots__ = ("_indices", "_reserved", "_size", "_unknown")

    def __init__(
        self,
        tokens: Sequence[str],
        *,
        reserved: ReservedTokens,
        unknown: str,
    ) -> None:
        if not reserved.holds(unknown):
            msg = (
                f"{unknown!r} is read as the unknown token and was not set "
                f"aside; {reserved.names} were"
            )
            raise ValueError(msg)

        named = tuple(ensure_name("a vocabulary token", token) for token in tokens)
        size = len(named) + reserved.count
        beyond = sorted(index for index in reserved.indices if index >= size)

        if beyond:
            msg = (
                f"a vocabulary of {size} rows was given the reserved indices "
                f"{beyond}, which no row has"
            )
            raise ValueError(msg)

        free = iter(sorted(set(range(size)) - reserved.indices))
        indices: dict[str, int] = {}

        for token in named:
            if reserved.holds(token):
                msg = f"{token!r} is both a vocabulary token and a reserved one"
                raise ValueError(msg)

            if token in indices:
                msg = f"{token!r} appears twice in the vocabulary"
                raise ValueError(msg)

            indices[token] = next(free)

        self._indices = indices
        self._reserved = reserved
        self._size = size
        self._unknown = reserved.index_of(unknown)

    @property
    def size(self) -> int:
        """Return the number of embedding rows."""
        return self._size

    @property
    def unknown_index(self) -> int:
        """Return the row a token outside the vocabulary is read as."""
        return self._unknown

    @property
    def reserved(self) -> ReservedTokens:
        """Return the reserved tokens and their assigned indices."""
        return self._reserved

    @property
    def tokens(self) -> tuple[str, ...]:
        """Return the vocabulary's own tokens, in the order they were given."""
        return tuple(self._indices)

    def holds(self, token: str) -> bool:
        """Return whether the vocabulary has a row of its own for one token.

        Parameters
        ----------
        token : str
            The token to look for.

        Returns
        -------
        bool
            Whether a loaded vector for it would land somewhere.
        """
        return token in self._indices

    def index_of(self, token: str) -> int:
        """Return the row one token is read as.

        Parameters
        ----------
        token : str
            The token to look up.

        Returns
        -------
        int
            Its own row where it has one, the reserved row where it was set aside, and
            the unknown row otherwise.
        """
        if token in self._indices:
            return self._indices[token]

        if self._reserved.holds(token):
            return self._reserved.index_of(token)

        return self._unknown

    @override
    def __repr__(self) -> str:
        """Return the size, the reserved tokens and the unknown row."""
        return (
            f"TokenVocabulary(size={self._size}, "
            f"reserved={self._reserved.names!r}, unknown_index={self._unknown})"
        )


class LoadReport:
    """Counts embedding rows read, loaded, and skipped.

    Parameters
    ----------
    rows_read : int
        How many vector rows the file held.
    rows_matched : int
        How many of them the vocabulary claimed a row for.
    rows_dropped : int
        How many of them named a token the vocabulary does not hold.

    Raises
    ------
    TypeError
        If a count is not an integer.
    ValueError
        If a count is negative or the matched and dropped rows do not partition the rows
        read.
    """

    __slots__ = ("_rows_dropped", "_rows_matched", "_rows_read")

    def __init__(self, *, rows_read: int, rows_matched: int, rows_dropped: int) -> None:
        read = ensure_non_negative("the rows read", rows_read)
        matched = ensure_non_negative("the rows matched", rows_matched)
        dropped = ensure_non_negative("the rows dropped", rows_dropped)

        if read != matched + dropped:
            msg = (
                f"{read} rows were read and {matched} matched beside "
                f"{dropped} dropped, which do not account for them"
            )
            raise ValueError(msg)

        self._rows_read = read
        self._rows_matched = matched
        self._rows_dropped = dropped

    @property
    def rows_read(self) -> int:
        """Return the number of vector rows read."""
        return self._rows_read

    @property
    def rows_matched(self) -> int:
        """Return the number of vector rows loaded."""
        return self._rows_matched

    @property
    def rows_dropped(self) -> int:
        """Return the number of rows absent from the vocabulary."""
        return self._rows_dropped

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report holds the same three counts.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not LoadReport:
            return NotImplemented

        return (
            self._rows_read == other._rows_read
            and self._rows_matched == other._rows_matched
            and self._rows_dropped == other._rows_dropped
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the three counts."""
        return hash(
            (LoadReport, self._rows_read, self._rows_matched, self._rows_dropped)
        )

    @override
    def __repr__(self) -> str:
        """Return an evaluable representation."""
        return (
            f"LoadReport(rows_read={self._rows_read}, "
            f"rows_matched={self._rows_matched}, "
            f"rows_dropped={self._rows_dropped})"
        )

    def measurements(self) -> Mapping[str, float]:
        """Return the counts under stable run-report metric names."""
        return {
            "static_vectors/rows_read": float(self._rows_read),
            "static_vectors/rows_matched": float(self._rows_matched),
            "static_vectors/rows_dropped": float(self._rows_dropped),
        }


class StaticTokenEmbedder(nn.Module):
    """A token embedder backed by fixed rows.

    Parameters
    ----------
    table : Tensor
        The vectors, rank 2 over the axes ``("index", "width")``, at floating precision,
        with one row per index the vocabulary describes.
    vocabulary : TokenVocabulary
        What the table's rows are indexed by.
    component : ComponentId
        The handle the run assigned this component, which its published point is named
        under.
    trainable : bool
        Whether the table takes a gradient.
    window : ContextWindow | None
        How many words one item may hold and what to do with a longer one, or None
        where an item of any length is embedded whole.

    Raises
    ------
    TypeError
        If the table is not of a floating point dtype.
    ValueError
        If the table is not rank 2, holds no columns, or holds a number of rows other
        than the vocabulary's size.
    """

    def __init__(
        self,
        table: Tensor,
        *,
        vocabulary: TokenVocabulary,
        component: ComponentId,
        trainable: bool,
        window: ContextWindow | None,
        load_report: LoadReport | None,
    ) -> None:
        super().__init__()

        if not table.is_floating_point():
            msg = f"an embedding table holds real numbers, not {table.dtype}"
            raise TypeError(msg)

        if table.ndim != _TABLE_RANK:
            msg = (
                f"an embedding table is one row per index and one column per "
                f"feature, so it is rank {_TABLE_RANK}, not {table.ndim}"
            )
            raise ValueError(msg)

        if table.shape[0] != vocabulary.size:
            msg = (
                f"a vocabulary of {vocabulary.size} rows was given a table of "
                f"{table.shape[0]}"
            )
            raise ValueError(msg)

        if table.shape[1] == 0:
            msg = "an embedding table holds at least one column"
            raise ValueError(msg)

        self._table = nn.Embedding.from_pretrained(table, freeze=not trainable)
        self._vocabulary = vocabulary
        self._component = component
        self._window = window
        self._load_report = load_report

    @property
    def output_width(self) -> int:
        """Return the size of the feature axis of the rows this embedder produces."""
        return int(self._table.weight.shape[1])

    @property
    def vocabulary(self) -> TokenVocabulary:
        """Return the vocabulary indexing the embedding rows."""
        return self._vocabulary

    @property
    def window(self) -> ContextWindow | None:
        """Return the bound on an item's words, or None if there is none."""
        return self._window

    @property
    def load_report(self) -> LoadReport | None:
        """Return the static-vector load counts, where rows came from a file."""
        return self._load_report

    def report(self) -> Mapping[str, float]:
        """Return vector-loading counts for the run report."""
        if self._load_report is None:
            return {}

        return self._load_report.measurements()

    def _kept(self, words: Sequence[str]) -> int:
        """Return how many leading words of an item the window admits.

        Parameters
        ----------
        words : Sequence[str]
            The item's words.

        Returns
        -------
        int
            The count, which is every word where there is no window.

        Raises
        ------
        ContextOverflowError
            If the item is longer than a window that refuses overflow.
        """
        if self._window is None:
            return len(words)

        return len(words) - len(self._window.fit(words))

    def encode_words(self, items: Sequence[Sequence[str]], /) -> TokenEncoding:
        """Return one embedding row per input word the window admits.

        Parameters
        ----------
        items : Sequence[Sequence[str]]
            The words of each item, in item order and then in word order.

        Returns
        -------
        TokenEncoding
            Rows over the axes ``("item", "row", "width")``, real valued and masked over
            the item and row axes, with one alignment per item that is total where the
            item fits the window and names the dropped suffix where it does not.

        Raises
        ------
        ContextOverflowError
            If an item is longer than a window that refuses overflow, in which case
            nothing is returned for the batch.
        """
        kept = tuple(self._kept(words) for words in items)
        longest = max(kept, default=0)
        device = self._table.weight.device
        indices = torch.full(
            (len(items), longest),
            self._vocabulary.unknown_index,
            dtype=torch.long,
            device=device,
        )
        live = torch.zeros((len(items), longest), dtype=torch.bool, device=device)

        for item, (words, count) in enumerate(zip(items, kept, strict=True)):
            for word, token in enumerate(words[:count]):
                indices[item, word] = self._vocabulary.index_of(token)
                live[item, word] = True

        rows = self._table(indices) * live.unsqueeze(-1).to(self._table.weight.dtype)
        alignments = tuple(
            Alignment(
                tuple((word, word + 1) for word in range(count))
                + (None,) * (len(words) - count)
            )
            for words, count in zip(items, kept, strict=True)
        )

        return TokenEncoding(Encoded(rows, _ROW_AXES, live), alignments, words=items)

    def site_map(self) -> SiteMap:
        """Return the rows this embedder produces for an item.

        Returns
        -------
        SiteMap
            The rows, under ``Role.TOKEN_EMBEDDING``.
        """
        return SiteMap(
            self._component,
            ComponentKind.TOKEN_ENCODER,
            {
                Site(self._component, Role.TOKEN_EMBEDDING): SiteSpec(
                    _ROW_AXES,
                    DTypeClass.REAL,
                    differentiable=True,
                    access=Access.OBSERVE,
                )
            },
        )


def _rows(lines: Iterator[str]) -> Iterator[tuple[str, tuple[float, ...]]]:
    """Yield the token and the vector of each non-empty line.

    Parameters
    ----------
    lines : Iterator[str]
        The lines to read, each a token followed by its numbers.

    Yields
    ------
    tuple of str and tuple of float
        The token and its vector.

    Raises
    ------
    ValueError
        If a line has no number, if a field is not a number, or if a line holds a
        different number of columns from the first one.
    """
    width: int | None = None

    for number, line in enumerate(lines):
        fields = line.split()

        if not fields:
            continue

        token, rest = fields[0], fields[1:]

        if not rest:
            msg = f"line {number + 1} names {token!r} and has no vector"
            raise ValueError(msg)

        if width is None:
            width = len(rest)

        if len(rest) != width:
            msg = (
                f"line {number + 1} has {len(rest)} columns where the "
                f"first row has {width}"
            )
            raise ValueError(msg)

        yield token, tuple(float(field) for field in rest)


class _OovInitializer(Protocol):
    """Initialize a complete static embedding table before file overlay."""

    def __call__(
        self, rows: int, width: int, *, std: float, generator: torch.Generator
    ) -> Tensor:
        """Return one initialized row per vocabulary index."""
        ...


class OovInitializationPlugin(FrozenValue):
    """Initialize rows not supplied by a static-vector file."""

    __slots__ = ("_initialize",)
    _noun = "OOV-initialization plugin"

    def __init__(self, initialize: _OovInitializer, /) -> None:
        object.__setattr__(self, "_initialize", initialize)

    def initialize(
        self, rows: int, width: int, *, std: float, generator: torch.Generator
    ) -> Tensor:
        """Return a complete initialized table."""
        return self._initialize(rows, width, std=std, generator=generator)


def _zero_oov(
    rows: int, width: int, *, std: float, generator: torch.Generator
) -> Tensor:
    """Return a zero-initialized table."""
    del std, generator
    return torch.zeros((rows, width), dtype=torch.float32)


def _normal_oov(
    rows: int, width: int, *, std: float, generator: torch.Generator
) -> Tensor:
    """Return a normally initialized table under a dedicated generator."""
    return torch.randn((rows, width), dtype=torch.float32, generator=generator) * std


ZERO_OOV: Final = OovInitializationPlugin(_zero_oov)
NORMAL_OOV: Final = OovInitializationPlugin(_normal_oov)


def oov_initialization(name: str, /) -> OovInitializationPlugin:
    """Return the OOV initializer installed under ``name``."""
    member = Registry[object]("oov_initialization").get(name)

    if not isinstance(member, OovInitializationPlugin):
        msg = (
            f"the oov_initialization entry {name!r} names {member!r}, "
            "not an OOV-initialization plugin"
        )
        raise RegistryError(msg, family="oov_initialization")

    return member


def _vocabulary_indices(
    vocabulary: TokenVocabulary, *, lowercase: bool
) -> dict[str, int]:
    """Return file-token spellings mapped to vocabulary rows."""
    indexed: dict[str, int] = {}

    for token in vocabulary.tokens:
        normalized = token.lower() if lowercase else token

        if normalized in indexed:
            msg = (
                f"lowercasing collides on vocabulary token {normalized!r}; "
                "static-vector rows would not have a unique destination"
            )
            raise ValueError(msg)

        indexed[normalized] = vocabulary.index_of(token)

    return indexed


def _loaded(
    rows: Iterator[tuple[str, tuple[float, ...]]],
    vocabulary: TokenVocabulary,
    *,
    lowercase: bool,
    initializer: OovInitializationPlugin,
    init_std: float,
    generator: torch.Generator,
) -> tuple[Tensor, LoadReport]:
    """Return the loaded table and load report.

    Parameters
    ----------
    rows : Iterator[tuple[str, tuple[float, ...]]]
        The token and vector of each row of the file.
    vocabulary : TokenVocabulary
        What the table is indexed by.

    Returns
    -------
    tuple[Tensor, LoadReport]
        The table, rank 2 over the axes ``("index", "width")`` at single precision, and
        the three counts.

    Raises
    ------
    ValueError
        If the file holds no row.
    """
    placed: dict[int, tuple[float, ...]] = {}
    spellings: dict[str, str] = {}
    read = 0
    matched = 0
    dropped = 0
    width: int | None = None

    indices = _vocabulary_indices(vocabulary, lowercase=lowercase)

    for token, vector in rows:
        read += 1
        width = len(vector)
        normalized = token.lower() if lowercase else token

        if normalized in indices:
            index = indices[normalized]
            first = spellings.setdefault(normalized, token)

            if lowercase and first != token:
                msg = f"lowercasing collides on vector rows {first!r} and {token!r}"
                raise ValueError(msg)

            placed[index] = vector
            matched += 1
        else:
            dropped += 1

    if width is None:
        msg = "the file holds no vector, so the table it describes has no width"
        raise ValueError(msg)

    table = initializer.initialize(
        vocabulary.size, width, std=init_std, generator=generator
    )

    for index, vector in placed.items():
        table[index] = torch.tensor(vector, dtype=torch.float32)

    return table, LoadReport(rows_read=read, rows_matched=matched, rows_dropped=dropped)


def from_glove_text(
    path: Path,
    *,
    vocabulary: TokenVocabulary,
    component: ComponentId,
    trainable: bool,
    lowercase: bool = False,
    oov_init: str = "zeros",
    oov_init_std: float = 0.02,
    seed: int = 0,
) -> tuple[StaticTokenEmbedder, LoadReport]:
    """Load a static embedder from GloVe text.

    Parameters
    ----------
    path : Path
        The file to read.
    vocabulary : TokenVocabulary
        What the table is indexed by.
    component : ComponentId
        The handle the run assigned the embedder.
    trainable : bool
        Whether the loaded rows take gradients.

    Returns
    -------
    tuple[StaticTokenEmbedder, LoadReport]
        The embedder and the rows read, matched and dropped.

    Raises
    ------
    ValueError
        If the file holds no vector, if a line has no number, or if two lines have
        different numbers of columns.
    """
    with path.open(encoding="utf-8") as handle:
        table, report = _loaded(
            _rows(iter(handle)),
            vocabulary,
            lowercase=lowercase,
            initializer=oov_initialization(oov_init),
            init_std=oov_init_std,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )

    return (
        StaticTokenEmbedder(
            table,
            vocabulary=vocabulary,
            component=component,
            trainable=trainable,
            window=None,
            load_report=report,
        ),
        report,
    )


def from_word2vec_text(
    path: Path,
    *,
    vocabulary: TokenVocabulary,
    component: ComponentId,
    trainable: bool,
    lowercase: bool = False,
    oov_init: str = "zeros",
    oov_init_std: float = 0.02,
    seed: int = 0,
) -> tuple[StaticTokenEmbedder, LoadReport]:
    """Load a static embedder from word2vec text.

    Parameters
    ----------
    path : Path
        The file to read.
    vocabulary : TokenVocabulary
        What the table is indexed by.
    component : ComponentId
        The handle the run assigned the embedder.
    trainable : bool
        Whether the loaded rows take gradients.

    Returns
    -------
    tuple[StaticTokenEmbedder, LoadReport]
        The embedder and the rows read, matched and dropped.

    Raises
    ------
    ValueError
        If the header is malformed, vectors are absent or ragged, a field is nonnumeric,
        or declared dimensions disagree with the rows.
    """
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().split()

        if len(header) != _HEADER_FIELDS:
            msg = (
                f"a word2vec text file opens with its row count and its width, "
                f"not with {header}"
            )
            raise ValueError(msg)

        declared, width = (int(field) for field in header)
        table, report = _loaded(
            _rows(iter(handle)),
            vocabulary,
            lowercase=lowercase,
            initializer=oov_initialization(oov_init),
            init_std=oov_init_std,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )

    if report.rows_read != declared:
        msg = (
            f"the header declares {declared} rows and the file holds {report.rows_read}"
        )
        raise ValueError(msg)

    if int(table.shape[1]) != width:
        msg = (
            f"the header declares a width of {width} and the rows have {table.shape[1]}"
        )
        raise ValueError(msg)

    return (
        StaticTokenEmbedder(
            table,
            vocabulary=vocabulary,
            component=component,
            trainable=trainable,
            window=None,
            load_report=report,
        ),
        report,
    )


class _VectorReader(Protocol):
    """Read a static embedding table under one installed text format."""

    def __call__(
        self,
        path: Path,
        *,
        vocabulary: TokenVocabulary,
        component: ComponentId,
        trainable: bool,
        lowercase: bool,
        oov_init: str,
        oov_init_std: float,
        seed: int,
    ) -> tuple[StaticTokenEmbedder, LoadReport]:
        """Return the loaded embedder and row report."""
        ...


class VectorFormatPlugin(FrozenValue):
    """Read one installed static-vector text format."""

    __slots__ = ("_read",)
    _noun = "vector-format plugin"

    def __init__(self, read: _VectorReader, /) -> None:
        object.__setattr__(self, "_read", read)

    def read(
        self,
        path: Path,
        *,
        vocabulary: TokenVocabulary,
        component: ComponentId,
        trainable: bool,
        lowercase: bool,
        oov_init: str,
        oov_init_std: float,
        seed: int,
    ) -> tuple[StaticTokenEmbedder, LoadReport]:
        """Return the embedder and report read from ``path``."""
        return self._read(
            path,
            vocabulary=vocabulary,
            component=component,
            trainable=trainable,
            lowercase=lowercase,
            oov_init=oov_init,
            oov_init_std=oov_init_std,
            seed=seed,
        )


def vector_format(name: str, /) -> VectorFormatPlugin:
    """Return the static-vector format installed under ``name``."""
    member = Registry[object]("vector_format").get(name)

    if not isinstance(member, VectorFormatPlugin):
        msg = (
            f"the vector_format entry {name!r} names {member!r}, "
            "not a vector-format plugin"
        )
        raise RegistryError(msg, family="vector_format")

    return member


GLOVE: Final = VectorFormatPlugin(from_glove_text)
WORD2VEC: Final = VectorFormatPlugin(from_word2vec_text)


def _build_static(
    settings: StaticTokenEncoderSettings, context: BuildContext, /
) -> StaticTokenEmbedder:
    """Build and place a static token embedder from a file of fixed vectors.

    Parameters
    ----------
    settings : StaticTokenEncoderSettings
        The file, its layout and whether its rows take gradients.
    context : BuildContext
        The vocabulary the loaded table is indexed by, the component handle and
        the device policy.

    Returns
    -------
    StaticTokenEmbedder
        The embedder holding the rows the file supplied.
    """
    embedder, _ = vector_format(settings.vectors_format).read(
        Path(settings.vectors),
        vocabulary=context.vocabulary,
        component=context.component,
        trainable=settings.trainable,
        lowercase=settings.lowercase,
        oov_init=settings.oov_init,
        oov_init_std=settings.oov_init_std,
        seed=settings.seed,
    )

    return place(embedder, context.policy)


PLUGIN: Final = ComponentPlugin(spec=StaticTokenEncoderSpec, build=_build_static)
"""The static token encoder's plugin."""
