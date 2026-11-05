"""Token embeddings learned from random initialization."""

from typing import TYPE_CHECKING, Final

import torch

from lofi.nn.policy import place
from lofi.nn.tokens.embedder import StaticTokenEmbedder
from lofi.registry import ComponentPlugin
from lofi.spec.tokens.learned import (
    LearnedTokenEncoderSettings,
    LearnedTokenEncoderSpec,
)

if TYPE_CHECKING:
    from lofi.nn.build import BuildContext
    from lofi.nn.sites import ComponentId
    from lofi.nn.tokens.embedder import TokenVocabulary

__all__ = ["initialized_embedder"]


def initialized_embedder(
    width: int,
    *,
    init_std: float,
    vocabulary: TokenVocabulary,
    component: ComponentId,
) -> StaticTokenEmbedder:
    """Return a trainable embedder whose rows are drawn at random.

    Every row, reserved ones included, is drawn from a normal distribution with
    standard deviation ``1 / sqrt(width)``, so that a row's expected squared norm
    is one at any width. The draw is made on the CPU from torch's default
    generator, which a run seeds from its random streams before it builds its
    components, so one run seed gives one table on every device.

    Parameters
    ----------
    width : int
        The number of features of every row, positive by its declaration.
    vocabulary : TokenVocabulary
        What the table's rows are indexed by.
    component : ComponentId
        The handle the run assigned the embedder.

    Returns
    -------
    StaticTokenEmbedder
        The embedder, with its table taking gradients and embedding an item of
        any length whole.
    """
    table = (
        torch.randn(vocabulary.size, width, dtype=torch.float32, device="cpu")
        * init_std
    )

    return StaticTokenEmbedder(
        table,
        vocabulary=vocabulary,
        component=component,
        trainable=True,
        window=None,
        load_report=None,
    )


def _build_learned(
    settings: LearnedTokenEncoderSettings, context: BuildContext, /
) -> StaticTokenEmbedder:
    """Build and place a trainable embedder drawn at random at the declared width.

    Parameters
    ----------
    settings : LearnedTokenEncoderSettings
        The width of every row.
    context : BuildContext
        The vocabulary the drawn table is indexed by, the component handle and
        the device policy.

    Returns
    -------
    StaticTokenEmbedder
        The embedder holding one drawn row per index of the vocabulary.
    """
    return place(
        initialized_embedder(
            settings.width,
            init_std=settings.init_std,
            vocabulary=context.vocabulary,
            component=context.component,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=LearnedTokenEncoderSpec, build=_build_learned)
"""The learned token encoder's plugin."""
