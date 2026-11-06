"""Direct negative log likelihood of retained gold logical forms."""

from typing import TYPE_CHECKING, Final, final

import torch

from lofi._checks import ensure_name
from lofi.nn._values import Encoded
from lofi.nn.loss._ports import LossTerm
from lofi.nn.loss._readout import CELL_AXES
from lofi.registry import ComponentPlugin
from lofi.spec.losses.form_supervision import (
    FormSupervisionLossSettings,
    FormSupervisionLossSpec,
)

if TYPE_CHECKING:
    from lofi.nn.loss._ports import (
        GoldFormItems,
        LossComponentBindings,
        PublishedOutputs,
        TermProbabilityReadout,
    )
    from lofi.terms import Term

__all__ = ["FormSupervisionObjective"]

_TERM_AXES: Final = ("term", "width")


@final
class FormSupervisionObjective:
    """Maximize each retained gold form's complete term probability."""

    __slots__ = ("_name", "_readout")

    def __init__(self, readout: TermProbabilityReadout, /, *, name: str) -> None:
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

    def evaluate(self, outputs: PublishedOutputs[GoldFormItems], /) -> LossTerm:
        """Return mean negative log likelihood of retained gold forms."""
        cells = outputs.cells

        if cells is None:
            msg = f"{self._name} reads chart cells, and the step published none"
            raise ValueError(msg)

        if cells.axes != CELL_AXES:
            msg = f"{self._name} reads cells over {CELL_AXES}, not {cells.axes}"
            raise ValueError(msg)

        batch = outputs.batch

        if len(batch.gold_forms) != len(batch.words):
            msg = (
                f"{self._name} received {len(batch.gold_forms)} gold-form rows for "
                f"{len(batch.words)} surface items"
            )
            raise ValueError(msg)

        terms: list[Term] = []
        item_positions: list[int] = []
        span_positions: list[int] = []

        for item, (words, term) in enumerate(
            zip(batch.words, batch.gold_forms, strict=True)
        ):
            if term is None:
                continue

            terms.append(term)
            item_positions.append(item)
            span_positions.append(outputs.spans.span_id(0, len(words)))

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
            msg = f"{self._name} received a positive gold-form log probability"
            raise ValueError(msg)

        return LossTerm(
            -log_probabilities.mean(),
            len(terms),
            diagnostics={"gold_form_probability": log_probabilities.exp().mean()},
        )


def _build(
    settings: FormSupervisionLossSettings, context: LossComponentBindings, /
) -> FormSupervisionObjective:
    return FormSupervisionObjective(
        context.terms_readout(settings.readout),
        name=context.name,
    )


PLUGIN: Final = ComponentPlugin(spec=FormSupervisionLossSpec, build=_build)
"""The direct gold-form supervision plugin."""
