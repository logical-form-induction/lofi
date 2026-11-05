"""Recurrent decoding of the words a chart span covers."""

from typing import TYPE_CHECKING, Final, Protocol, cast, override

import torch
from torch import nn

from lofi.nn._layers import feedforward, recurrence
from lofi.nn._values import LogDistribution
from lofi.nn.policy import place
from lofi.nn.tokens.embedder import ReservedTokens, TokenVocabulary
from lofi.registry import ComponentPlugin
from lofi.spec.span_decoders.rnn import RnnSpanDecoderSettings, RnnSpanDecoderSpec

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.nn._values import Encoded
    from lofi.nn.build import BuildContext
    from lofi.nn.sites import ComponentId
    from lofi.spec.layers import FeedForwardLayer, RecurrentCell

__all__ = ["RecurrentSpanDecoder"]

_PREFIX_AXES: Final = ("span", "width")
"""The axes of the vector each span is decoded from."""

_SCORE_AXES: Final = ("span", "position", "token")
"""The axes of the scored distribution."""


class _WidthProducer(Protocol):
    """A producer whose vectors determine a consumer's input width."""

    @property
    def output_width(self) -> int:
        """Return one produced vector's feature width."""


def _with_end(vocabulary: TokenVocabulary, end: str, /) -> TokenVocabulary:
    """Return a vocabulary that also reserves the end token, at its last row.

    Parameters
    ----------
    vocabulary : TokenVocabulary
        The run's token vocabulary, whose rows keep their indices.

    Raises
    ------
    ValueError
        If the vocabulary already holds the end token.
    """
    reserved = vocabulary.reserved

    if reserved.holds(end) or vocabulary.holds(end):
        msg = f"the token vocabulary already holds the end token {end!r}"
        raise ValueError(msg)

    assignments = {name: reserved.index_of(name) for name in reserved.names}
    unknown = next(
        name for name, index in assignments.items() if index == vocabulary.unknown_index
    )

    return TokenVocabulary(
        vocabulary.tokens,
        reserved=ReservedTokens({**assignments, end: vocabulary.size}),
        unknown=unknown,
    )


class RecurrentSpanDecoder(nn.Module):
    """Scores the words a span covers, left to right, from the span's vector.

    The span's vector is the input at the position before the span's first word, and
    each later position reads the embedding of the word before it. A stack of
    recurrent layers steps the state, and a head scores every row of the vocabulary
    from the top layer's state. The vocabulary is the run's token vocabulary with the
    end token reserved at one more row.

    Parameters
    ----------
    vocabulary : TokenVocabulary
        The run's token vocabulary.
    width : int
        The width of the span vectors, which is also the width of the word
        embeddings.
    cell : RecurrentCell
        The cell each recurrent layer steps its state with.
    layers : int
        How many recurrent layers are stacked.
    state_width : int | None
        The width of every recurrent layer's state, or None for the width of the
        span vectors.
    head : tuple[FeedForwardLayer, ...]
        The declared hidden layers of the scoring head.
    component : ComponentId
        The handle the run assigned this component.

    Raises
    ------
    ValueError
        If the vocabulary already holds the end token.
    """

    def __init__(
        self,
        vocabulary: TokenVocabulary,
        *,
        width: int,
        cell: RecurrentCell,
        layers: int,
        state_width: int | None,
        head: tuple[FeedForwardLayer, ...],
        end: str,
        component: ComponentId,
    ) -> None:
        super().__init__()
        self._vocabulary = _with_end(vocabulary, end)
        self._end = end
        self._width = width
        self._component = component
        rows = self._vocabulary.size
        self._embed = nn.Embedding(rows, width)
        state = width if state_width is None else state_width
        self._recur = recurrence(
            cell, inputs=width, width=state, layers=layers, bidirectional=False
        )
        self._score = feedforward(
            head,
            inputs=state,
            output=rows,
            bias=True,
        )

    @property
    def vocabulary(self) -> TokenVocabulary:
        """Return the vocabulary whose rows the scored distribution is over."""
        return self._vocabulary

    @property
    def end(self) -> str:
        """Return the reserved token each span's words are followed by."""
        return self._end

    @property
    def input_width(self) -> int:
        """Return the width of the span vectors this decoder reads."""
        return self._width

    def score_tokens(self, prefix: Encoded, /, *, targets: Tensor) -> LogDistribution:
        """Return the log probability of each token at each position of each span.

        Parameters
        ----------
        prefix : Encoded
            One vector per span, over the axes ``("span", "width")``.
        targets : Tensor
            The rows to score, of rank 2 over the spans and then their positions.

        Returns
        -------
        LogDistribution
            Rank 3 over the axes ``("span", "position", "token")``, normalized over
            the token axis.

        Raises
        ------
        ValueError
            If the prefix is not over the span axes or is not as wide as the decoder,
            or if the targets do not run over its spans.
        """
        vectors = prefix.tensor

        if prefix.axes != _PREFIX_AXES or vectors.shape[1] != self._width:
            msg = (
                f"a span decoder of width {self._width} reads vectors over "
                f"{_PREFIX_AXES}, not {prefix.axes} of shape {tuple(vectors.shape)}"
            )
            raise ValueError(msg)

        if targets.ndim != len(_PREFIX_AXES) or targets.shape[0] != vectors.shape[0]:
            msg = (
                f"the targets of shape {tuple(targets.shape)} do not give positions "
                f"for each of {vectors.shape[0]} spans"
            )
            raise ValueError(msg)

        previous = self._embed(targets[:, :-1]).to(vectors.dtype)
        inputs = torch.cat((vectors.unsqueeze(1), previous), dim=1)
        states, _ = self._recur(inputs)

        return LogDistribution(
            self._score(states).log_softmax(dim=-1),
            _SCORE_AXES,
            normalized_over=_SCORE_AXES[-1],
        )

    @override
    def extra_repr(self) -> str:
        """Return the component, the width and the vocabulary size."""
        return (
            f"component={self._component!r}, width={self._width}, "
            f"rows={self._vocabulary.size}"
        )


def _build(
    settings: RnnSpanDecoderSettings, context: BuildContext, /
) -> RecurrentSpanDecoder:
    """Build and place a recurrent span decoder.

    The decoder reads span vectors at the output width of the chart wired to its
    ``cells`` port.

    Parameters
    ----------
    settings : RnnSpanDecoderSettings
        The recurrent cell, the number and width of the recurrent layers and the
        number, width and nonlinearities of the head's hidden layers.
    context : BuildContext
        The run's vocabulary, the wired chart, the component handle and the
        device policy.
    """
    chart = cast("_WidthProducer", context.producer("cells"))

    return place(
        RecurrentSpanDecoder(
            context.vocabulary,
            width=chart.output_width,
            cell=settings.cell,
            layers=settings.layers,
            state_width=settings.state_width,
            head=settings.head,
            end=settings.end,
            component=context.component,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=RnnSpanDecoderSpec, build=_build)
"""The recurrent span decoder's plugin."""
