"""Likelihood of graded judgments against a frozen encoder's pooled rows."""

from typing import TYPE_CHECKING, final

import torch
from torch import nn

from lofi._checks import ensure_name
from lofi.judgment.agreement import rank_correlation
from lofi.nn.loss._ports import LossTerm
from lofi.nn.sites import ComponentId
from lofi.nn.tokens.pretrained import EncoderArchitecturePlugin, load_pretrained
from lofi.spec.tokens.pretrained import (
    ContextWindowSettings,
    LastLayerReadoutSettings,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from lofi._artifacts import PinnedCheckpoint
    from lofi.judgment.grouped import _GroupedJudgmentBatch
    from lofi.judgment.ordinal import OrdinalHead
    from lofi.nn._values import TokenEncoding
    from lofi.nn.loss._ports import PublishedOutputs
    from lofi.nn.ports import TokenEncoder
    from lofi.nn.tokens.pretrained import PretrainedLoader

__all__ = ["PooledEncoderObjective", "frozen_encoder_objective", "pooled_rows"]


def pooled_rows(encoding: TokenEncoding) -> Tensor:
    """Return the mean over each item's live rows, one vector per item.

    Parameters
    ----------
    encoding : TokenEncoding
        Rows over the axes ``("item", "row", "width")`` with their mask.

    Raises
    ------
    ValueError
        If an item has no live row to pool.
    """
    rows = encoding.vectors.tensor
    mask = encoding.vectors.mask

    if mask is None:
        mask = torch.ones(rows.shape[:2], device=rows.device, dtype=torch.bool)

    counted = mask.sum(dim=1)

    if not bool((counted > 0).all()):
        msg = f"item {int(counted.eq(0).nonzero()[0])} has no row to pool"
        raise ValueError(msg)

    summed = rows.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1)

    return summed / counted.unsqueeze(-1).to(dtype=rows.dtype)


@final
class PooledEncoderObjective(nn.Module):
    """Score graded judgments from a frozen encoder's mean-pooled rows.

    Each item's words are encoded with no gradient tracked and averaged under
    their mask into one vector, a linear map reads a latent score from that
    vector, and the head gives the probability of each response level at that
    score. The value is the mean negative log probability of the recorded
    levels. The linear map and the head are the trainable parameters.

    Parameters
    ----------
    encoder : TokenEncoder
        The encoder whose rows are pooled.
    head : OrdinalHead
        The cutpoints between adjacent response levels.
    name : str
        The name the term is logged and namespaced under.
    """

    def __init__(
        self, encoder: TokenEncoder, head: OrdinalHead, *, name: str = "encoder"
    ) -> None:
        super().__init__()
        self._name = ensure_name("the objective's name", name)
        self.encoder = encoder
        self.head = head
        self.projection = nn.Linear(encoder.output_width, 1)

    @property
    def name(self) -> str:
        """Return the name this objective is logged and namespaced under."""
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        """Return the published outputs this objective reads."""
        return frozenset({"batch"})

    @property
    def levels(self) -> int:
        """Return the number of ordered response levels."""
        return self.head.levels

    def latent_scores(self, words: Sequence[Sequence[str]], /) -> Tensor:
        """Return the latent score of each item, in item order.

        Parameters
        ----------
        words : Sequence[Sequence[str]]
            The words of each item, in item order and then in word order.
        """
        with torch.no_grad():
            encoding = self.encoder.encode_words(words)

        return self.projection(pooled_rows(encoding)).squeeze(-1)

    def evaluate(self, outputs: PublishedOutputs[_GroupedJudgmentBatch], /) -> LossTerm:
        """Return the mean negative log probability of the batch's judgments.

        Parameters
        ----------
        outputs : PublishedOutputs[_GroupedJudgmentBatch]
            What the graph published for the step.

        Returns
        -------
        LossTerm
            The mean over the judgments, counted in them, reporting the mean
            log probability of the recorded levels as ``log_density``, the
            fraction of responses at the modal level as ``accuracy``, and the
            rank correlation of latent score with recorded level as
            ``rank_correlation`` where at least three responses vary on both
            sides.

        Raises
        ------
        IndexError
            If a judgment names an item outside the batch.
        ValueError
            If an item has no row to pool, or a judgment records a level the
            head does not have.
        """
        batch = outputs.batch
        words = batch.words
        design = batch.design
        latent = self.latent_scores(words)

        if design.response_levels != self.head.levels:
            msg = (
                f"a head over {self.head.levels} levels was given a design over "
                f"{design.response_levels}"
            )

            raise ValueError(msg)

        responses = design.responses.to(device=latent.device)
        response_latent = batch.response_rows(latent)
        log_probabilities = self.head.log_probabilities(response_latent)
        observed = log_probabilities.gather(1, responses.unsqueeze(1)).squeeze(1)
        modal = log_probabilities.detach().argmax(dim=1)
        diagnostics = {
            "log_density": observed.detach().mean(),
            "accuracy": modal.eq(responses).to(observed.dtype).mean(),
        }
        scored = response_latent.detach().tolist()
        levels = responses.tolist()

        if len(scored) >= 3 and len(set(scored)) > 1 and len(set(levels)) > 1:
            diagnostics["rank_correlation"] = observed.new_tensor(
                rank_correlation(scored, levels)
            )

        return LossTerm(-observed.mean(), design.n_rows, diagnostics=diagnostics)


def frozen_encoder_objective(
    head: OrdinalHead,
    *,
    checkpoint: PinnedCheckpoint,
    architecture: EncoderArchitecturePlugin,
    dtype: torch.dtype,
    loader: PretrainedLoader,
    name: str = "encoder",
) -> PooledEncoderObjective:
    """Return the objective over a pretrained encoder that takes no gradient.

    Parameters
    ----------
    head : OrdinalHead
        The cutpoints between adjacent response levels.
    checkpoint : PinnedCheckpoint
        The release to load.
    architecture : EncoderArchitecturePlugin
        Which stack of the release runs.
    dtype : torch.dtype
        The precision the encoder's parameters load in, which is the parameter
        precision of the device policy the objective runs under.
    loader : PretrainedLoader
        The loading backend.
    name : str
        The name the term is logged under and the encoder is handled as.
    """
    component = ComponentId(ensure_name("the objective's name", name))
    encoder = load_pretrained(
        checkpoint,
        architecture=architecture,
        component=component,
        trainable=False,
        window=ContextWindowSettings(None, "report"),
        readout=LastLayerReadoutSettings(),
        join=" ",
        loader=loader,
        dtype=dtype,
    )

    return PooledEncoderObjective(encoder, head, name=name)
