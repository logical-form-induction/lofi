"""Held-out unlabeled bracketing F1 over a trained parser's charts."""

import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, cast, override

import torch

from lofi.eval._bracketing import Bracketing, BracketingF1, Constituent, GoldBracketing
from lofi.eval._held_out import derivation_dataset
from lofi.eval._values import ConventionVector
from lofi.nn.parsers import k_best
from lofi.registry import ComponentPlugin
from lofi.spec.evaluators.bracketing import (
    BracketingEvaluatorSettings,
    BracketingEvaluatorSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import ItemId
    from lofi.data import Derivation, EvalDataset, Item
    from lofi.eval._held_out import HeldOutEvaluationContext
    from lofi.eval._ports import HeldOutData, HeldOutItems
    from lofi.nn.loss._values import ObjectiveOutputs

__all__ = ["PLUGIN", "BracketingEvaluator"]


class BracketingEvaluator:
    """Scores the minimum-risk bracketing of each held-out item.

    The predicted bracketing of an item is the binary tree whose spans hold the
    most constituent marginal mass, read off the chart's published marginals,
    together with its single-word spans. The gold bracketing is every node of the
    gold derivation, each at its depth below the root.

    Parameters
    ----------
    conventions : ConventionVector
        The conventions spans are scored and reported under.
    """

    __slots__ = (
        "_conventions",
        "_decoder",
        "_metrics",
        "_punctuation_forms",
        "_punctuation_source",
    )

    def __init__(
        self,
        conventions: ConventionVector,
        *,
        punctuation_source: Literal["none", "surface"],
        punctuation_forms: tuple[str, ...],
        decoder: Literal["minimum-risk", "viterbi"],
    ) -> None:
        self._punctuation_source = punctuation_source
        self._punctuation_forms = frozenset(punctuation_forms)
        self._decoder = decoder
        self._conventions = conventions.extended(
            {
                "punctuation_source": punctuation_source,
                "punctuation_forms": json.dumps(
                    punctuation_forms,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "decoder": decoder,
            }
        )
        self._metrics = (BracketingF1(self._conventions),)

    @property
    def conventions(self) -> ConventionVector:
        """Return the conventions spans are scored and reported under."""
        return self._conventions

    @property
    def metrics(self) -> tuple[BracketingF1, ...]:
        """Return the bracketing metric in report order."""
        return self._metrics

    def held_out(self, data: HeldOutData, split: str, /) -> EvalDataset[GoldBracketing]:
        """Return a retained partition's derivations as this evaluator reads them."""
        return derivation_dataset(self.annotate, data.derivations(split))

    def annotate(self, item: Item, derivation: Derivation[str], /) -> GoldBracketing:
        """Return every node of a gold derivation as a labeled constituent.

        Parameters
        ----------
        item : Item
            The held-out item.
        derivation : Derivation[str]
            Its gold derivation.

        Returns
        -------
        GoldBracketing
            One constituent per node, at its depth below the root.
        """
        depths = {derivation.root: 0}

        for node_id in derivation.preorder:
            for child in derivation.node(node_id).children:
                depths[child] = depths[node_id] + 1

        return GoldBracketing(
            len(item.tokens),
            (
                Constituent(
                    node.span.start,
                    node.span.end,
                    label=node.label,
                    depth=depths[node_id],
                )
                for node_id, node in derivation.nodes.items()
            ),
            punctuation=(
                ()
                if self._punctuation_source == "none"
                else (
                    token.index
                    for token in item.tokens
                    if token.surface in self._punctuation_forms
                )
            ),
        )

    def predict[BatchT: HeldOutItems](
        self, outputs: ObjectiveOutputs[BatchT], /
    ) -> Mapping[ItemId, Bracketing | None]:
        """Return the minimum-risk bracketing of every item of a batch.

        Parameters
        ----------
        outputs : ObjectiveOutputs[BatchT]
            The published forward pass over one held-out batch.

        Returns
        -------
        Mapping[ItemId, Bracketing | None]
            Each item's bracketing, or ``None`` where its chart admits no tree.

        Raises
        ------
        ValueError
            If the forward pass published no span marginals.
        """
        if self._decoder == "minimum-risk":
            marginals = outputs.marginals

            if marginals is None:
                msg = (
                    "minimum-risk bracketing reads span marginals, and none were "
                    "published"
                )
                raise ValueError(msg)

            gain = marginals.constituent.tensor.detach()
            cutpoints = cast("torch.Tensor", marginals.rule.mask)
            split_weights = gain.unsqueeze(-1).expand(cutpoints.shape)
            lengths = marginals.lengths
        else:
            splits = outputs.splits

            if splits is None:
                msg = (
                    "Viterbi bracketing reads split posteriors, and none were published"
                )
                raise ValueError(msg)

            gain = splits.log_probabilities.tensor.detach().logsumexp(dim=-1)
            cutpoints = splits.valid
            split_weights = splits.log_probabilities.tensor.detach()
            lengths = torch.tensor(
                [len(words) for words in outputs.batch.words],
                device=gain.device,
            )

        supported = torch.where(
            cutpoints,
            split_weights,
            gain.new_full((), -torch.inf),
        )
        leaves = gain.new_zeros((int(gain.shape[0]), outputs.spans.n))
        trees = k_best(leaves, supported, outputs.spans, k=1, lengths=lengths)
        words = outputs.batch.words

        return MappingProxyType(
            {
                item_id: (
                    Bracketing(
                        len(tokens),
                        (
                            *found[0].constituents,
                            *((start, start + 1) for start in range(len(tokens))),
                        ),
                    )
                    if found
                    else None
                )
                for item_id, tokens, found in zip(
                    outputs.batch.item_ids, words, trees, strict=True
                )
            }
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this evaluator."""
        return f"BracketingEvaluator({self._conventions!r}, decoder={self._decoder!r})"


def _build(
    settings: BracketingEvaluatorSettings,
    context: HeldOutEvaluationContext,
    /,
) -> BracketingEvaluator:
    """Build a held-out bracketing evaluator from its declared settings.

    Parameters
    ----------
    settings : BracketingEvaluatorSettings
        The declared conventions.
    context : HeldOutEvaluationContext
        The held-out resources, which bracketing does not read.
    """
    del context

    return BracketingEvaluator(
        ConventionVector.from_settings(settings.conventions),
        punctuation_source=settings.punctuation_source,
        punctuation_forms=settings.punctuation_forms,
        decoder=settings.decoder,
    )


PLUGIN: Final = ComponentPlugin(spec=BracketingEvaluatorSpec, build=_build)
