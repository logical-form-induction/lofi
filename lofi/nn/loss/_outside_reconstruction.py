"""DIORA-style word reconstruction from contextual outside vectors."""

from typing import TYPE_CHECKING, Final, override

import torch

from lofi._checks import CHECKS, ensure_name
from lofi.nn._values import Encoded
from lofi.nn.loss._errors import MissingOutputError
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._ports import LossTerm
from lofi.registry import ComponentPlugin
from lofi.spec.losses.outside_reconstruction import (
    OutsideReconstructionLossSettings,
    OutsideReconstructionLossSpec,
)

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.deduce import SpanIndex
    from lofi.nn._values import LogDistribution
    from lofi.nn.loss._ports import (
        LossComponentBindings,
        PublishedOutputs,
        SpanTokenDecoder,
        SurfaceItems,
    )
    from lofi.nn.tokens.embedder import TokenVocabulary

__all__ = ["OutsideReconstructionObjective"]

_OUTSIDE_RANK: Final[int] = 3
"""The rank of contextual outside vectors over items, spans and features."""

_SCORE_AXES: Final[tuple[str, ...]] = ("span", "position", "token")
"""The axes over which a span decoder scores selected leaf spans."""


def _checked_outside(outside: Encoded, spans: SpanIndex, *, items: int) -> None:
    """Raise unless outside vectors use the batch's span addressing."""
    if outside.tensor.ndim != _OUTSIDE_RANK:
        msg = (
            f"the outside vectors are over the axes {outside.axes}, and one "
            "vector per span of each item is of rank three"
        )
        raise ValueError(msg)

    if outside.axes[: len(SPAN_AXES)] != SPAN_AXES:
        msg = (
            f"the outside vectors are over the axes {outside.axes}, and this "
            f"objective reads them over {SPAN_AXES} and a feature axis"
        )
        raise ValueError(msg)

    for axis, extent in zip(SPAN_AXES, (items, spans.num_spans), strict=True):
        if outside.size(axis) != extent:
            msg = (
                f"the outside vectors hold {outside.size(axis)} positions on the "
                f"axis {axis!r}, and the batch and its index name {extent}"
            )
            raise ValueError(msg)

    mask = outside.mask

    if mask is not None and mask.ndim > len(SPAN_AXES):
        msg = (
            f"the outside vectors are masked over {outside.mask_axes}, so the mask "
            "divides a vector's features rather than saying which vectors are real"
        )
        raise ValueError(msg)


def _live_leaves(outside: Encoded, spans: SpanIndex, *, items: int) -> Tensor:
    """Return which leaf outside vector is present for each batch item."""
    mask = outside.mask

    if mask is None:
        return torch.ones(
            (items, spans.n), dtype=torch.bool, device=outside.tensor.device
        )

    if mask.ndim == 1:
        return mask.unsqueeze(1).expand(items, spans.n)

    leaves = torch.tensor(
        [spans.span_id(position, position + 1) for position in range(spans.n)],
        dtype=torch.long,
        device=mask.device,
    )

    return mask.index_select(1, leaves)


def _checked_scores(scored: LogDistribution, targets: Tensor) -> None:
    """Raise unless the decoder scored the requested target rows."""
    if scored.axes != _SCORE_AXES:
        msg = (
            f"the decoder scored the axes {scored.axes}, and reconstructed words "
            f"are scored over {_SCORE_AXES}"
        )
        raise ValueError(msg)

    wanted = tuple(targets.shape)
    given = tuple(scored.tensor.shape[:2])

    if given != wanted:
        msg = (
            f"the decoder was handed {wanted[0]} words of {wanted[1]} positions "
            f"and scored {given[0]} of {given[1]}"
        )
        raise ValueError(msg)

    if scored.normalized_over != _SCORE_AXES[-1]:
        msg = (
            f"the decoder's scores are normalized over {scored.normalized_over!r}, "
            f"not {_SCORE_AXES[-1]!r}"
        )
        raise ValueError(msg)


class OutsideReconstructionObjective[BatchT: SurfaceItems]:
    """Reconstruct each word from its contextual outside vector.

    This is the DIORA-style leaf reconstruction objective. Each live word is
    decoded from the vector computed outside its own one-word span, so its memory
    contains sentence context but not that leaf's inside representation. This is
    not span reconstruction, which predicts every span's yield from its inside
    vector.

    Parameters
    ----------
    decoder : SpanTokenDecoder
        What scores each word and the end token after its outside vector.
    name : str
        The name the term is logged and namespaced under.

    Raises
    ------
    KeyError
        If the decoder's vocabulary sets no row aside for its end token.
    TypeError
        If a name is not a string.
    ValueError
        If a name is empty, or if the end token is also the vocabulary's unknown
        row.
    """

    __slots__ = ("_decoder", "_end", "_end_id", "_name")

    def __init__(self, decoder: SpanTokenDecoder, *, name: str) -> None:
        vocabulary = decoder.vocabulary
        self._decoder = decoder
        self._end = ensure_name("the end token", decoder.end)
        self._end_id = vocabulary.reserved.index_of(self._end)

        if self._end_id == vocabulary.unknown_index:
            msg = (
                f"the end token {self._end!r} is row {self._end_id}, which is the "
                "unknown row this vocabulary reads a word outside it as"
            )
            raise ValueError(msg)

        self._name = ensure_name("the objective's name", name)

    @property
    def name(self) -> str:
        """Return the name this objective is logged and namespaced under."""
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        """Return the contextual chart outputs this objective reads."""
        return frozenset({"spans", "outside_cells", "batch"})

    @property
    def end(self) -> str:
        """Return the reserved token each reconstructed word is followed by."""
        return self._end

    @property
    def vocabulary(self) -> TokenVocabulary:
        """Return the vocabulary whose rows the decoder scores."""
        return self._decoder.vocabulary

    def evaluate(self, outputs: PublishedOutputs[BatchT], /) -> LossTerm:
        """Return the mean negative log probability of context-reconstructed words.

        Padded words and leaves omitted by the published outside mask take no part
        in the term and receive no gradient. A batch with no selected word returns
        the ordinary inapplicable term.
        """
        outside = outputs.outside_cells

        if outside is None:
            raise MissingOutputError(
                objective=self._name,
                missing=frozenset({"outside_cells"}),
                published=frozenset(),
            )

        spans = outputs.spans
        words = outputs.batch.words

        _checked_outside(outside, spans, items=len(words))

        if not words or spans.n == 0:
            return LossTerm.inapplicable(outside.tensor)

        lengths = torch.tensor(
            [len(item) for item in words],
            dtype=torch.long,
            device=outside.tensor.device,
        )

        if bool((lengths > spans.n).any()):
            item = int((lengths > spans.n).nonzero()[0])
            msg = (
                f"item {item} holds {lengths[item].item()} words, and its chart "
                f"is indexed over {spans.n}"
            )
            raise ValueError(msg)

        live = _live_leaves(outside, spans, items=len(words))
        positions = torch.arange(spans.n, device=outside.tensor.device).unsqueeze(0)
        selected = live & (positions < lengths.unsqueeze(1))

        if not bool(selected.any()):
            return LossTerm.inapplicable(outside.tensor)

        leaf_ids = torch.tensor(
            [spans.span_id(position, position + 1) for position in range(spans.n)],
            dtype=torch.long,
            device=outside.tensor.device,
        )
        leaves = outside.tensor.index_select(1, leaf_ids)
        prefixes = leaves[selected]
        ids = torch.full(
            (len(words), spans.n),
            self._decoder.vocabulary.unknown_index,
            dtype=torch.long,
            device=outside.tensor.device,
        )

        for item, held in enumerate(words):
            ids[item, : len(held)] = torch.tensor(
                [self._decoder.vocabulary.index_of(word) for word in held],
                dtype=torch.long,
                device=outside.tensor.device,
            )

        rows = torch.stack(
            (
                ids[selected],
                ids.new_full((int(selected.sum()),), self._end_id),
            ),
            dim=1,
        )
        scored = self._decoder.score_tokens(
            Encoded(prefixes, ("span", "width")), targets=rows
        )

        if CHECKS:
            _checked_scores(scored, rows)

        nll = -scored.tensor.gather(2, rows.unsqueeze(2)).squeeze(2)

        return LossTerm(
            nll.mean(dim=1).mean(),
            int(selected.sum()),
            diagnostics={"per_token": nll.detach().mean()},
        )

    @override
    def __repr__(self) -> str:
        """Return the name and reserved end token."""
        return f"OutsideReconstructionObjective(name={self._name!r}, end={self._end!r})"


def _build[BatchT: SurfaceItems](
    settings: OutsideReconstructionLossSettings,
    context: LossComponentBindings,
    /,
) -> OutsideReconstructionObjective[BatchT]:
    """Build contextual outside reconstruction through its declared decoder."""
    return OutsideReconstructionObjective(
        context.span_decoder(settings.decoder),
        name=context.name,
    )


PLUGIN: Final = ComponentPlugin(spec=OutsideReconstructionLossSpec, build=_build)
"""The contextual outside-reconstruction objective's plugin."""
