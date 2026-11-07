"""Binary logical form typing supervision."""

from typing import TYPE_CHECKING, Final, final

import torch

from lofi._checks import ensure_name
from lofi.nn._values import Encoded
from lofi.nn.loss._ports import LossTerm
from lofi.nn.loss._readout import CELL_AXES
from lofi.registry import ComponentPlugin
from lofi.spec.losses.form_typing import FormTypingLossSettings, FormTypingLossSpec

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.nn.loss._ports import (
        FormTypingItems,
        LossComponentBindings,
        PublishedOutputs,
        TermProbabilityReadout,
    )
    from lofi.terms import Term

__all__ = ["FormTypingObjective"]

_TERM_AXES: Final = ("term", "width")


def _mean_probability(probabilities: Tensor, selected: Tensor, /) -> Tensor:
    """Return the selected mean, or a graph-connected zero when none are selected."""
    values = probabilities[selected]
    return values.mean() if values.numel() else probabilities[:0].sum()


@final
class FormTypingObjective:
    """Train complete logical forms against observed typing labels."""

    __slots__ = ("_name", "_readout")

    def __init__(
        self,
        readout: TermProbabilityReadout,
        /,
        *,
        name: str,
    ) -> None:
        self._readout = readout
        self._name = ensure_name("the objective's name", name)

    @property
    def name(self) -> str:
        """Return the name this objective is logged under."""
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        """Return the published outputs this objective reads."""
        return frozenset({"batch", "cells", "spans"})

    def evaluate(self, outputs: PublishedOutputs[FormTypingItems], /) -> LossTerm:
        """Return binary negative log likelihood of the observed typing labels."""
        cells = outputs.cells

        if cells is None:
            msg = f"{self._name} reads chart cells, and the step published none"
            raise ValueError(msg)

        if cells.axes != CELL_AXES:
            msg = f"{self._name} reads cells over {CELL_AXES}, not {cells.axes}"
            raise ValueError(msg)

        batch = outputs.batch
        terms: list[Term] = []
        labels: list[str] = []
        item_positions: list[int] = []
        span_positions: list[int] = []

        if len(batch.form_typing) != len(batch.words):
            msg = (
                f"{self._name} received {len(batch.form_typing)} typing rows for "
                f"{len(batch.words)} surface items"
            )
            raise ValueError(msg)

        for item, (words, targets) in enumerate(
            zip(batch.words, batch.form_typing, strict=True)
        ):
            root = outputs.spans.span_id(0, len(words))

            for target in targets:
                terms.append(target.term)
                labels.append(target.observed_label)
                item_positions.append(item)
                span_positions.append(root)

        if not terms:
            return LossTerm.inapplicable(cells.tensor)

        items = torch.tensor(
            item_positions, dtype=torch.long, device=cells.tensor.device
        )
        spans = torch.tensor(
            span_positions, dtype=torch.long, device=cells.tensor.device
        )
        mask = cells.mask

        if mask is not None:
            live = mask[items] if mask.ndim == 1 else mask[items, spans]

            if not bool(live.all()):
                first = int((~live).nonzero(as_tuple=False)[0, 0])
                msg = (
                    f"{self._name} cannot score the uncomputed root of item "
                    f"{item_positions[first]}"
                )
                raise ValueError(msg)

        vectors = Encoded(cells.tensor[items, spans], _TERM_AXES)
        log_probabilities = self._readout.score_terms(vectors, tuple(terms)).tensor

        if bool((log_probabilities > 0.0).any()):
            msg = f"{self._name} received a positive complete-term log probability"
            raise ValueError(msg)

        ill_typed = torch.tensor(
            [label == "ill-typed" for label in labels],
            dtype=torch.bool,
            device=log_probabilities.device,
        )
        log_complements = torch.log(-torch.expm1(log_probabilities))
        losses = torch.where(ill_typed, -log_complements, -log_probabilities)
        probabilities = log_probabilities.exp()

        return LossTerm(
            losses.mean(),
            len(terms),
            diagnostics={
                "well_typed_probability": _mean_probability(probabilities, ~ill_typed),
                "ill_typed_probability": _mean_probability(probabilities, ill_typed),
            },
        )


def _build(
    settings: FormTypingLossSettings, context: LossComponentBindings, /
) -> FormTypingObjective:
    return FormTypingObjective(
        context.terms_readout(settings.readout),
        name=context.name,
    )


PLUGIN: Final = ComponentPlugin(spec=FormTypingLossSpec, build=_build)
"""The logical form typing objective plugin."""
