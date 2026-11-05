"""The held-out evaluator scoring decoded logical forms by proof."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, override

from lofi._checks import ensure_name
from lofi._types import ReadoutId
from lofi.eval._equivalence import (
    ACCOUNTING_MEASURES,
    EQUIVALENCE_MEASURES,
    EquivalenceAccountingMetric,
    EquivalenceLedger,
    EquivalenceMetric,
)
from lofi.eval._held_out import derivation_dataset
from lofi.eval._values import ConventionVector
from lofi.registry import ComponentPlugin
from lofi.spec.evaluators.equivalence import (
    EquivalenceEvaluatorSettings,
    EquivalenceEvaluatorSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import ItemId
    from lofi.data import Derivation, EvalDataset, Item
    from lofi.eval import HeldOutEvaluationContext, HeldOutItems
    from lofi.eval._ports import HeldOutData
    from lofi.lf.provers import ProofBudget
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.terms import Term

__all__ = ["PLUGIN", "EquivalenceEvaluator"]


type _EquivalenceRow = EquivalenceMetric | EquivalenceAccountingMetric
"""One row of an equivalence evaluation."""


class EquivalenceEvaluator:
    """Score the most probable decoded form of each held-out item by proof.

    The evaluator reads a complete-term readout at an item's root cell, selects
    the form with greatest decoder weight, writes it in the run language's
    surface syntax, and decides its equivalence with the item's gold form
    through the interpretation's shared proof service. Every row reads one
    ledger, so each item is decided once.
    """

    __slots__ = ("_context", "_conventions", "_metrics", "_readout")

    def __init__(
        self,
        context: HeldOutEvaluationContext,
        /,
        *,
        readout: str,
        budget: ProofBudget,
        conventions: ConventionVector,
    ) -> None:
        interpretation = context.interpretation

        if interpretation is None:
            msg = "equivalence evaluation requires a declared interpretation"
            raise ValueError(msg)

        if not context.forms:
            msg = "equivalence evaluation requires held-out gold forms"
            raise ValueError(msg)

        self._context = context
        self._readout = ReadoutId(
            ensure_name("an equivalence term-readout role", readout)
        )
        self._conventions = conventions.extended(
            {
                "equivalence_readout": readout,
                "proof_prover": interpretation.service.prover,
                "proof_calls_per_step": budget.calls_per_step,
                "proof_wall_per_step": budget.wall_per_step,
                "proof_cpu_seconds": budget.cpu_seconds,
            }
        )
        ledger = EquivalenceLedger(
            interpretation.service, interpretation.language, budget=budget
        )
        self._metrics: tuple[_EquivalenceRow, ...] = (
            *(
                EquivalenceMetric(
                    ledger, measure=measure, conventions=self._conventions
                )
                for measure in EQUIVALENCE_MEASURES
            ),
            *(
                EquivalenceAccountingMetric(
                    ledger, measure=measure, conventions=self._conventions
                )
                for measure in ACCOUNTING_MEASURES
            ),
        )

    @property
    def conventions(self) -> ConventionVector:
        """Return the conventions every row is reported under."""
        return self._conventions

    @property
    def metrics(self) -> tuple[_EquivalenceRow, ...]:
        """Return the rates and the service counts in report order."""
        return self._metrics

    def held_out(self, data: HeldOutData, split: str, /) -> EvalDataset[Term]:
        """Return a retained partition's derivations as this evaluator reads them."""
        return derivation_dataset(self.annotate, data.derivations(split))

    def annotate(self, item: Item, derivation: Derivation[str], /) -> Term:
        """Return the corpus's gold form for one held-out item."""
        del derivation

        try:
            text = self._context.forms[item.item_id]
        except KeyError:
            msg = f"the held-out item {item.item_id!r} has no gold logical form"
            raise ValueError(msg) from None

        interpretation = self._context.interpretation

        if interpretation is None:
            msg = "equivalence evaluation requires a declared interpretation"
            raise ValueError(msg)

        return interpretation.language.read(text)

    def predict[BatchT: HeldOutItems](
        self, outputs: ObjectiveOutputs[BatchT], /
    ) -> Mapping[ItemId, str]:
        """Return the highest-weight root form of each item as surface text."""
        decoded = outputs.terms.get(self._readout)

        if decoded is None:
            msg = (
                f"equivalence reads the complete terms {self._readout!r}, and "
                f"the pass published {sorted(outputs.terms)}"
            )
            raise ValueError(msg)

        interpretation = self._context.interpretation

        if interpretation is None:
            msg = "equivalence evaluation requires a declared interpretation"
            raise ValueError(msg)

        predictions: dict[ItemId, str] = {}

        for row, (item_id, words) in enumerate(
            zip(outputs.batch.item_ids, outputs.batch.words, strict=True)
        ):
            candidates = decoded.at(row, decoded.spans.span_id(0, len(words)))

            if not candidates.terms:
                msg = (
                    f"the complete-term readout {self._readout!r} decoded no "
                    f"logical form for {item_id!r}"
                )
                raise ValueError(msg)

            selected = int(candidates.log_weights.detach().argmax())
            predictions[item_id] = interpretation.render(candidates.terms[selected])

        return MappingProxyType(predictions)

    @override
    def __repr__(self) -> str:
        """Return a summary naming the readout."""
        return f"EquivalenceEvaluator(readout={self._readout!r})"


def _build(
    settings: EquivalenceEvaluatorSettings,
    context: HeldOutEvaluationContext,
    /,
) -> EquivalenceEvaluator:
    """Build the declared equivalence evaluator in its held-out context."""
    return EquivalenceEvaluator(
        context,
        readout=settings.readout,
        budget=settings.budget,
        conventions=ConventionVector.from_settings(settings.conventions),
    )


PLUGIN: Final = ComponentPlugin(spec=EquivalenceEvaluatorSpec, build=_build)
