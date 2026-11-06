"""Inference label objectives over one published grounded batch."""

from typing import TYPE_CHECKING, Final, final, override

import torch

from lofi._checks import ensure_finite, ensure_name
from lofi.lf import InferenceCell
from lofi.nn._inference_response import InferenceResponse, resolve_response
from lofi.nn._values import InferenceSignal
from lofi.nn.grounding._values import GroundedBatch
from lofi.nn.loss._ports import LossTerm
from lofi.registry import ComponentPlugin
from lofi.spec.losses.inference import InferenceLossSettings, InferenceLossSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from lofi.nn._inference_ports import InferencePairs
    from lofi.nn.forms import WeightedTerms
    from lofi.nn.loss._ports import (
        FormCredit,
        LossComponentBindings,
        PublishedOutputs,
    )

__all__ = ["InferenceObjective", "InferenceSignal"]


def _expected_credit(credit: FormCredit, forms: WeightedTerms) -> Tensor:
    weights = forms.log_weights

    if not forms.terms:
        return weights.sum()

    paid = weights.new_tensor([credit.credit(form) for form in forms.terms])
    return (weights.exp() * paid).sum()


def _diagnostics(grounded: object, like: Tensor) -> dict[str, Tensor]:
    if not isinstance(grounded, GroundedBatch):
        msg = f"inference reads a GroundedBatch, not {grounded!r}"
        raise TypeError(msg)

    ledger = grounded.ledger
    pairs = grounded.decided.numel()
    decided = int(grounded.decided.sum())
    cells = grounded.cells.tensor
    degenerate = cells[:, tuple(InferenceCell).index(InferenceCell.DEGENERATE)]
    variance = grounded.variance
    diagnostics = {
        "abstention_rate": like.new_tensor((pairs - decided) / pairs if pairs else 0.0),
        "calls": like.new_tensor(ledger.calls),
        "questions": like.new_tensor(ledger.questions),
        "questions_per_second": like.new_tensor(ledger.questions_per_second),
        "memo_hits": like.new_tensor(ledger.memo_hits),
        "memo_hit_rate": like.new_tensor(ledger.memo_hit_rate),
        "cpu_seconds": like.new_tensor(ledger.cpu_seconds),
        "wall_seconds": like.new_tensor(ledger.wall_seconds),
        "process_starts": like.new_tensor(ledger.process_starts or 0),
        "gradient_path": like.new_tensor(
            tuple(type(grounded.gradient_path)).index(grounded.gradient_path)
        ),
        "degenerate_mass": (
            degenerate.mean() if degenerate.numel() else like.new_zeros(())
        ),
        "variance": (
            variance.mean()
            if variance is not None and variance.numel()
            else like.new_zeros(())
        ),
    }

    for cause, count in ledger.abstentions.items():
        diagnostics[f"abstention/{cause}"] = like.new_tensor(count)

    if ledger.verified_share is not None:
        diagnostics["verified_share"] = like.new_tensor(ledger.verified_share)

    return diagnostics


@final
class InferenceObjective:
    """Train the published grounded distribution against declared labels."""

    __slots__ = ("_auxiliary_weight", "_credit", "_name", "_response", "_signal")

    def __init__(
        self,
        signal: InferenceSignal,
        response: InferenceResponse,
        /,
        *,
        auxiliary_weight: float,
        credit: FormCredit | None,
        name: str,
    ) -> None:
        if not isinstance(signal, InferenceSignal):
            msg = f"an inference signal is InferenceSignal, not {signal!r}"
            raise TypeError(msg)

        weight = ensure_finite("an inference auxiliary weight", auxiliary_weight)

        if weight < 0.0:
            msg = f"an inference auxiliary weight is non-negative, not {weight!r}"
            raise ValueError(msg)

        self._signal = signal
        self._response = response
        self._auxiliary_weight = weight
        self._credit = credit
        self._name = ensure_name("the objective's name", name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def signal(self) -> InferenceSignal:
        return self._signal

    @property
    def auxiliary_weight(self) -> float:
        return self._auxiliary_weight

    @property
    def credit(self) -> FormCredit | None:
        return self._credit

    @property
    def response(self) -> InferenceResponse:
        """Return the declared observation model."""
        return self._response

    @property
    def requires(self) -> frozenset[str]:
        required = {"batch", "entailment"}

        if self._credit is not None:
            required.add("forms")

        return frozenset(required)

    def _credit_term(self, forms: Sequence[WeightedTerms]) -> Tensor | None:
        credit = self._credit

        if credit is None or not forms:
            return None

        return torch.stack([_expected_credit(credit, one) for one in forms]).mean()

    def evaluate(self, outputs: PublishedOutputs[InferencePairs], /) -> LossTerm:
        grounded = outputs.entailment

        if grounded is None:
            msg = f"{self._name} reads entailment, and the step published none"
            raise ValueError(msg)

        log_probabilities = grounded.log_probabilities.tensor
        pairs = int(log_probabilities.shape[0])
        like = log_probabilities
        response = self._response.evaluate(self._signal, grounded, outputs.batch)
        decided = response.observed
        grounded.observe_payments(response.payments[decided])
        kept = response.costs[decided]
        value = kept.mean() if kept.numel() else like.reshape(-1)[:0].sum()
        auxiliary = grounded.auxiliary

        if auxiliary is not None:
            value = value + self._auxiliary_weight * auxiliary

        credit = (
            None
            if self._credit is None
            else self._credit_term(tuple(outputs.forms.values()))
        )

        if credit is not None:
            value = value - credit

        residual = grounded.residual[response.pair_positions][decided]
        mean_residual = (
            residual.mean()
            if residual.numel()
            else like.new_tensor(1.0 if pairs else 0.0)
        )
        diagnostics = _diagnostics(grounded, like)

        if response.rewards is not None:
            diagnostics["reward_mean"] = (
                response.rewards[decided].mean()
                if bool(decided.any())
                else like.new_zeros(())
            )

        return LossTerm(
            value,
            int(decided.sum()),
            residual=mean_residual,
            diagnostics=diagnostics,
        )

    @override
    def __repr__(self) -> str:
        return (
            f"InferenceObjective(name={self._name!r}, signal={self._signal.value!r}, "
            f"response={self._response!r}, "
            f"auxiliary_weight={self._auxiliary_weight!r}, "
            f"credit={self._credit!r})"
        )


def _build(
    settings: InferenceLossSettings, context: LossComponentBindings, /
) -> InferenceObjective:
    return InferenceObjective(
        InferenceSignal(settings.signal),
        resolve_response(settings.response),
        auxiliary_weight=settings.auxiliary_weight,
        credit=None,
        name=context.name,
    )


PLUGIN: Final = ComponentPlugin(spec=InferenceLossSpec, build=_build)
"""The inference label objective plugin."""
