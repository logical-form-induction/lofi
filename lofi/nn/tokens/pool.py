"""Word-level pooling of encoder rows."""

from typing import TYPE_CHECKING, Final, NamedTuple, cast

import torch
from torch import nn

from lofi.nn._layers import feedforward, recurrence
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
from lofi.registry import ComponentPlugin
from lofi.spec.poolers.attention import (
    AttentionSubwordPoolerSettings,
    AttentionSubwordPoolerSpec,
)
from lofi.spec.poolers.first import FirstSubwordPoolerSettings, FirstSubwordPoolerSpec
from lofi.spec.poolers.mean import MeanSubwordPoolerSettings, MeanSubwordPoolerSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.nn.build import BuildContext
    from lofi.nn.ports import TokenEncoder
    from lofi.nn.sites import ComponentId
    from lofi.spec.layers import FeedForwardLayer
    from lofi.spec.poolers.attention import SubwordContextSettings

__all__ = [
    "AttentionSubwordPooler",
    "FirstSubwordPooler",
    "MeanSubwordPooler",
]

_POOLED_AXES: Final[tuple[str, ...]] = ("item", "word", "width")
"""Pooled word-vector axes."""


class _WordLayout(NamedTuple):
    """Maps batch words to encoder-row indices.

    Parameters
    ----------
    items : int
        How many items the batch holds.
    words : int
        How many words the longest item of the batch holds.
    item_of : Tensor
        Which item each pair belongs to: rank 1 over the pair axis, integral.
    word_of : Tensor
        Which word of that item each pair is: rank 1 over the pair axis, integral.
    row_of : Tensor
        Which row each position of each pair reads: rank 2 over the axes ``("pair",
        "subword")``, integral, and zero where the position is beyond the word's rows.
    live : Tensor
        Whether each position of each pair holds one of the word's rows: rank 2 over the
        axes ``("pair", "subword")``, boolean.
    """

    items: int
    words: int
    item_of: Tensor
    word_of: Tensor
    row_of: Tensor
    live: Tensor


def _layout(encoding: TokenEncoding) -> _WordLayout:
    """Return the index tensors gathering each encoded word's rows.

    Parameters
    ----------
    encoding : TokenEncoding
        The rows to pool and the alignment saying which rows hold which word.

    Returns
    -------
    _WordLayout
        The pair axis and the index tensors over it, placed where the rows are.
    """
    alignments = encoding.alignments
    device = encoding.vectors.tensor.device
    spans = [
        (item, word, span)
        for item, alignment in enumerate(alignments)
        for word, span in enumerate(alignment.rows)
        if span is not None
    ]
    longest = max((end - start for _, _, (start, end) in spans), default=0)
    row_of = torch.zeros((len(spans), longest), dtype=torch.long, device=device)
    live = torch.zeros((len(spans), longest), dtype=torch.bool, device=device)

    for pair, (_, _, (start, end)) in enumerate(spans):
        row_of[pair, : end - start] = torch.arange(start, end, device=device)
        live[pair, : end - start] = True

    return _WordLayout(
        items=len(alignments),
        words=max((alignment.word_count for alignment in alignments), default=0),
        item_of=torch.tensor([item for item, _, _ in spans], dtype=torch.long).to(
            device
        ),
        word_of=torch.tensor([word for _, word, _ in spans], dtype=torch.long).to(
            device
        ),
        row_of=row_of,
        live=live,
    )


def _gathered(vectors: Encoded, layout: _WordLayout) -> Tensor:
    """Return each encoded word's rows, padded to the longest word of the batch.

    Parameters
    ----------
    vectors : Encoded
        The encoder's rows, over the axes ``("item", "row", "width")``.
    layout : _WordLayout
        Where each encoded word takes its rows from.

    Returns
    -------
    Tensor
        Rank 3 over the axes ``("pair", "subword", "width")``, at the rows' own
        precision, and zero at a position beyond a word's rows.
    """
    rows = vectors.tensor
    picked = rows[layout.item_of.unsqueeze(1), layout.row_of]

    return picked * layout.live.unsqueeze(-1).to(picked.dtype)


def _scattered(reduced: Tensor, layout: _WordLayout, width: int) -> Encoded:
    """Return one vector per word of each item, masked where none was pooled.

    Parameters
    ----------
    reduced : Tensor
        One vector per encoded word, over the axes ``("pair", "width")``.
    layout : _WordLayout
        Where each encoded word came from.
    width : int
        How wide each vector is.

    Returns
    -------
    Encoded
        Rank 3 over the axes ``("item", "word", "width")``, real valued, with a mask
        over the item and word axes true exactly where a row encoded the word.
    """
    pooled = torch.zeros(
        (layout.items, layout.words, width),
        dtype=reduced.dtype,
        device=reduced.device,
    )
    kept = torch.zeros(
        (layout.items, layout.words), dtype=torch.bool, device=reduced.device
    )
    pooled = pooled.index_put((layout.item_of, layout.word_of), reduced)
    kept = kept.index_put(
        (layout.item_of, layout.word_of),
        torch.ones_like(layout.item_of, dtype=torch.bool),
    )

    return Encoded(pooled, _POOLED_AXES, kept)


def _pooled(
    encoding: TokenEncoding,
    reduce: Callable[[Tensor, Tensor], Tensor],
    width: int,
) -> Encoded:
    """Return one vector per word, reducing each word's rows with one rule.

    Parameters
    ----------
    encoding : TokenEncoding
        The rows to pool and the alignment saying which rows hold which word.
    reduce : Callable[[Tensor, Tensor], Tensor]
        Turn a word's rows and the mask over them into one vector.
    width : int
        How wide the vectors that come back are.

    Returns
    -------
    Encoded
        Rank 3 over the axes ``("item", "word", "width")``, real valued, masked over the
        item and word axes.
    """
    layout = _layout(encoding)
    rows = _gathered(encoding.vectors, layout)
    reduced = (
        rows.new_zeros((0, width))
        if layout.row_of.numel() == 0
        else reduce(rows, layout.live)
    )

    return _scattered(reduced, layout, width)


def _word_site(component: ComponentId) -> SiteMap:
    """Return the pooler's published word-vector site.

    Parameters
    ----------
    component : ComponentId
        The component the point belongs to.

    Returns
    -------
    SiteMap
        The pooled vectors, under ``Role.WORD_VECTOR``.
    """
    return SiteMap(
        component,
        ComponentKind.SUBWORD_POOLER,
        {
            Site(component, Role.WORD_VECTOR): SiteSpec(
                _POOLED_AXES,
                DTypeClass.REAL,
                differentiable=True,
                access=Access.OBSERVE,
            )
        },
    )


class FirstSubwordPooler(nn.Module):
    """Pools each word from its first subword row.

    Parameters
    ----------
    width : int
        How wide the rows this pooler gathers are.
    component : ComponentId
        The handle the run assigned this component, which its published point is named
        under.
    """

    def __init__(self, width: int, *, component: ComponentId) -> None:
        super().__init__()
        self._width = width
        self._component = component

    @property
    def output_width(self) -> int:
        """Return the size of the feature axis of the vectors this pooler returns."""
        return self._width

    def pool(self, encoding: TokenEncoding, /) -> Encoded:
        """Return the first row of each word of each item.

        Parameters
        ----------
        encoding : TokenEncoding
            The rows to pool and the alignment saying which rows hold which word.

        Returns
        -------
        Encoded
            Rank 3 over the axes ``("item", "word", "width")``, real valued, masked over
            the item and word axes.
        """
        return _pooled(encoding, _first, self._width)

    def site_map(self) -> SiteMap:
        """Return the vectors this pooler gathers each word into.

        Returns
        -------
        SiteMap
            The pooled vectors, under ``Role.WORD_VECTOR``.
        """
        return _word_site(self._component)


class MeanSubwordPooler(nn.Module):
    """Pools each word by averaging its subword rows.

    Parameters
    ----------
    width : int
        How wide the rows this pooler gathers are.
    component : ComponentId
        The handle the run assigned this component, which its published point is named
        under.
    """

    def __init__(self, width: int, *, component: ComponentId) -> None:
        super().__init__()
        self._width = width
        self._component = component

    @property
    def output_width(self) -> int:
        """Return the size of the feature axis of the vectors this pooler returns."""
        return self._width

    def pool(self, encoding: TokenEncoding, /) -> Encoded:
        """Return the mean subword vector for each word.

        Parameters
        ----------
        encoding : TokenEncoding
            The rows to pool and the alignment saying which rows hold which word.

        Returns
        -------
        Encoded
            Rank 3 over the axes ``("item", "word", "width")``, real valued, masked over
            the item and word axes.
        """
        return _pooled(encoding, _mean, self._width)

    def site_map(self) -> SiteMap:
        """Return the vectors this pooler gathers each word into.

        Returns
        -------
        SiteMap
            The pooled vectors, under ``Role.WORD_VECTOR``.
        """
        return _word_site(self._component)


class AttentionSubwordPooler(nn.Module):
    """Pools each word with learned attention over subword rows.

    A scorer gives each of a word's rows one score, optionally after a recurrence has
    read the word's rows, and the word's vector is the sum of its rows under the
    softmax of their scores.

    Parameters
    ----------
    width : int
        How wide the rows this pooler gathers are.
    component : ComponentId
        The handle the run assigned this component, which its published point is named
        under.
    context : SubwordContextSettings | None
        The recurrence the scorer reads a word's rows with, or None to score the rows
        as they are.
    scorer : tuple[FeedForwardLayer, ...]
        The declared hidden layers of the scorer.
    """

    def __init__(
        self,
        width: int,
        *,
        component: ComponentId,
        context: SubwordContextSettings | None,
        scorer: tuple[FeedForwardLayer, ...],
    ) -> None:
        super().__init__()
        self._width = width
        self._component = component
        self._context: nn.RNNBase | None = None
        reads = width

        if context is not None:
            state = width if context.width is None else context.width
            self._context = recurrence(
                context.cell,
                inputs=width,
                width=state,
                layers=context.layers,
                bidirectional=context.bidirectional,
            )
            reads = state * (2 if context.bidirectional else 1)

        self._query = feedforward(
            scorer,
            inputs=reads,
            output=1,
            bias=False,
        )

    @property
    def output_width(self) -> int:
        """Return the size of the feature axis of the vectors this pooler returns."""
        return self._width

    def _scores(self, rows: Tensor, live: Tensor) -> Tensor:
        """Return one score per position of each word's rows.

        Parameters
        ----------
        rows : Tensor
            Rank 3 over the axes ``("pair", "subword", "width")``.
        live : Tensor
            Rank 2 over the axes ``("pair", "subword")``, boolean.

        Returns
        -------
        Tensor
            Rank 2 over the axes ``("pair", "subword")``, real valued.
        """
        context = self._context
        read = rows if context is None else _contextual(context, rows, live)

        return self._query(read).squeeze(-1).masked_fill(~live, -torch.inf)

    def pool(self, encoding: TokenEncoding, /) -> Encoded:
        """Return the attention-weighted subword vector for each word.

        Parameters
        ----------
        encoding : TokenEncoding
            The rows to pool and the alignment saying which rows hold which word.

        Returns
        -------
        Encoded
            Rank 3 over the axes ``("item", "word", "width")``, real valued, masked over
            the item and word axes.
        """
        return _pooled(encoding, self._weighted, self._width)

    def _weighted(self, rows: Tensor, live: Tensor) -> Tensor:
        """Return each word's rows combined under the weights they scored.

        Parameters
        ----------
        rows : Tensor
            Rank 3 over the axes ``("pair", "subword", "width")``.
        live : Tensor
            Rank 2 over the axes ``("pair", "subword")``, boolean.

        Returns
        -------
        Tensor
            Rank 2 over the axes ``("pair", "width")``, real valued.
        """
        weights = self._scores(rows, live).softmax(dim=1)

        return (weights.unsqueeze(-1) * rows).sum(dim=1)

    def site_map(self) -> SiteMap:
        """Return the vectors this pooler gathers each word into.

        Returns
        -------
        SiteMap
            The pooled vectors, under ``Role.WORD_VECTOR``.
        """
        return _word_site(self._component)


def _contextual(context: nn.RNNBase, rows: Tensor, live: Tensor) -> Tensor:
    """Return each word's rows read by a recurrence.

    Parameters
    ----------
    context : nn.RNNBase
        The recurrence to read the rows with.
    rows : Tensor
        Rank 3 over the axes ``("pair", "subword", "width")``.
    live : Tensor
        Rank 2 over the axes ``("pair", "subword")``, boolean.

    Returns
    -------
    Tensor
        Rank 3 over the axes ``("pair", "subword", "width")``, real valued, as wide as
        the recurrence's output and zero beyond a word's rows.
    """
    lengths = live.sum(dim=1).to(device="cpu", dtype=torch.int64)
    packed = nn.utils.rnn.pack_padded_sequence(
        rows, lengths, batch_first=True, enforce_sorted=False
    )
    read, _ = nn.utils.rnn.pad_packed_sequence(
        context(packed)[0], batch_first=True, total_length=rows.shape[1]
    )

    return read


def _first(rows: Tensor, live: Tensor) -> Tensor:
    """Return the row of each word's first subword.

    Parameters
    ----------
    rows : Tensor
        Rank 3 over the axes ``("pair", "subword", "width")``.
    live : Tensor
        Rank 2 over the axes ``("pair", "subword")``, boolean and unread.

    Returns
    -------
    Tensor
        Rank 2 over the axes ``("pair", "width")``, real valued.
    """
    del live

    return rows[:, 0]


def _mean(rows: Tensor, live: Tensor) -> Tensor:
    """Return the mean subword vector for each word.

    Parameters
    ----------
    rows : Tensor
        Rank 3 over the axes ``("pair", "subword", "width")``, zero beyond a word's
        rows.
    live : Tensor
        Rank 2 over the axes ``("pair", "subword")``, boolean.

    Returns
    -------
    Tensor
        Rank 2 over the axes ``("pair", "width")``, real valued.
    """
    counted = live.sum(dim=1, keepdim=True).to(rows.dtype)

    return rows.sum(dim=1) / counted


def _encoder_width(context: BuildContext, /) -> int:
    """Return the output width of the token encoder a pooler gathers rows from.

    Parameters
    ----------
    context : BuildContext
        The context wiring the pooler's encoder port.
    """
    return cast("TokenEncoder", context.producer("encoder")).output_width


def _build_first(
    settings: FirstSubwordPoolerSettings, context: BuildContext, /
) -> FirstSubwordPooler:
    """Build and place a pooler reading each word off its first subword row.

    Parameters
    ----------
    settings : FirstSubwordPoolerSettings
        The pooler's settings, which hold nothing.
    context : BuildContext
        The wired token encoder, the component handle and the device policy.
    """
    del settings

    return place(
        FirstSubwordPooler(_encoder_width(context), component=context.component),
        context.policy,
    )


def _build_mean(
    settings: MeanSubwordPoolerSettings, context: BuildContext, /
) -> MeanSubwordPooler:
    """Build and place a pooler averaging the rows each word reached.

    Parameters
    ----------
    settings : MeanSubwordPoolerSettings
        The pooler's settings, which hold nothing.
    context : BuildContext
        The wired token encoder, the component handle and the device policy.
    """
    del settings

    return place(
        MeanSubwordPooler(_encoder_width(context), component=context.component),
        context.policy,
    )


def _build_attention(
    settings: AttentionSubwordPoolerSettings, context: BuildContext, /
) -> AttentionSubwordPooler:
    """Build and place a pooler weighing each word's rows with a learned query.

    Parameters
    ----------
    settings : AttentionSubwordPoolerSettings
        The recurrence the scorer reads a word's rows with, if any, and the
        number, width and nonlinearities of the scorer's hidden layers.
    context : BuildContext
        The wired token encoder, the component handle and the device policy.
    """
    return place(
        AttentionSubwordPooler(
            _encoder_width(context),
            component=context.component,
            context=settings.context,
            scorer=settings.scorer,
        ),
        context.policy,
    )


FIRST_PLUGIN: Final = ComponentPlugin(spec=FirstSubwordPoolerSpec, build=_build_first)
"""The first-subword pooler's plugin."""

MEAN_PLUGIN: Final = ComponentPlugin(spec=MeanSubwordPoolerSpec, build=_build_mean)
"""The mean pooler's plugin."""

ATTENTION_PLUGIN: Final = ComponentPlugin(
    spec=AttentionSubwordPoolerSpec, build=_build_attention
)
"""The attention pooler's plugin."""
