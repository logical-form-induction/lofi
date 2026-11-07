"""Declared response models for grounded inference training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, cast, final

import torch

from lofi._frozen import FrozenValue
from lofi.nn._inference_ports import InferencePairs, JudgmentItems
from lofi.nn._values import InferenceSignal
from lofi.registry import Registry
from lofi.spec.losses.inference import LabelResponseSettings
from lofi.spec.losses.judgment import JudgmentResponseSettings

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.lf import InferenceLabel
    from lofi.nn.grounding._values import GroundedBatch
    from lofi.spec.losses.inference import InferenceResponseDeclaration
__all__ = [
    "InferenceResponse",
    "InferenceResponsePlugin",
    "ResponseResult",
    "resolve_response",
]


class ResponseResult(NamedTuple):
    """Costs, control-variate payments and rewards aligned to pair positions."""

    costs: Tensor
    observed: Tensor
    pair_positions: Tensor
    payments: Tensor
    rewards: Tensor | None


class InferenceResponse(Protocol):
    """Scores one declared response kind against grounded inference mass."""

    def evaluate(
        self,
        signal: InferenceSignal,
        grounded: GroundedBatch,
        batch: object,
        /,
    ) -> ResponseResult:
        """Return response costs aligned to their grounded pair positions."""
        ...

    def response_rows(
        self, labels: tuple[InferenceLabel, ...], /
    ) -> tuple[tuple[float, ...], ...]:
        """Return the response distribution each grounded label predicts.

        Parameters
        ----------
        labels : tuple[InferenceLabel, ...]
            The grounded label inventory, in row order.
        """
        ...


class InferenceResponsePlugin[S](FrozenValue):
    """Build one response runtime from its checked declaration settings."""

    _noun = "inference response plugin"
    __slots__ = ("_build", "_settings")

    def __init__(
        self,
        *,
        settings: type[S],
        build: Callable[[S], InferenceResponse],
    ) -> None:
        object.__setattr__(self, "_settings", settings)
        object.__setattr__(self, "_build", build)

    def construct(self, settings: object, /) -> InferenceResponse:
        """Build the runtime after checking the registered settings type."""
        if not isinstance(settings, self._settings):
            msg = (
                f"an inference response plugin reads {self._settings.__name__}, "
                f"not {type(settings).__name__}"
            )
            raise TypeError(msg)

        return self._build(settings)


def resolve_response(declaration: InferenceResponseDeclaration, /) -> InferenceResponse:
    """Build the installed response a checked declaration names.

    Parameters
    ----------
    declaration : InferenceResponseDeclaration
        The response tag and its settings.

    Raises
    ------
    TypeError
        If the installed entry is not an inference response plugin.
    """
    member = Registry[object]("inference_response").get(declaration.kind)

    if not isinstance(member, InferenceResponsePlugin):
        msg = (
            f"the inference response entry {declaration.kind!r} names "
            f"{member!r}, not an inference response plugin"
        )
        raise TypeError(msg)

    return cast("InferenceResponsePlugin[object]", member).construct(
        declaration.settings
    )


def _pair_labels(batch: object, pairs: int, /) -> tuple[InferenceLabel, ...]:
    if not isinstance(batch, InferencePairs):
        msg = f"a label response reads inference pairs, not {batch!r}"
        raise TypeError(msg)

    labels = batch.labels

    if len(labels) != pairs:
        msg = (
            "a label response reads one label per grounded pair, and the batch has "
            f"{len(labels)} labels for {pairs} pairs"
        )
        raise ValueError(msg)

    return labels


def _label_payments(grounded: GroundedBatch, probabilities: Tensor, /) -> Tensor:
    reward = grounded.reward

    if reward is None:
        msg = "reward inference requires declared proof payoffs"
        raise ValueError(msg)

    return probabilities.mul(reward.proved - reward.refuted).add(reward.refuted)


@final
class _LabelResponse:
    """Score the observed categorical inference label."""

    __slots__ = ()

    def evaluate(
        self,
        signal: InferenceSignal,
        grounded: GroundedBatch,
        batch: object,
        /,
    ) -> ResponseResult:
        log_probabilities = grounded.log_probabilities.tensor
        pairs = int(log_probabilities.shape[0])
        labels = _pair_labels(batch, pairs)
        columns = torch.tensor(
            [grounded.labels.index(label) for label in labels],
            dtype=torch.long,
            device=log_probabilities.device,
        )
        positions = torch.arange(pairs, device=log_probabilities.device)
        selected = log_probabilities[positions, columns]
        probabilities = selected.exp()
        rewards = None

        if signal is InferenceSignal.REWARD:
            rewards = _label_payments(grounded, probabilities)

        payments = probabilities if rewards is None else rewards
        return ResponseResult(
            -selected if rewards is None else -rewards,
            grounded.decided,
            positions,
            payments,
            rewards,
        )

    def response_rows(
        self, labels: tuple[InferenceLabel, ...], /
    ) -> tuple[tuple[float, ...], ...]:
        """Return the certain prediction of each grounded label."""
        return tuple(
            tuple(float(column == row) for column in range(len(labels)))
            for row in range(len(labels))
        )


@final
class _JudgmentResponse:
    """Marginalize inference labels through a declared ordinal instrument."""

    __slots__ = ("_probabilities",)

    def __init__(
        self,
        probabilities: tuple[tuple[InferenceLabel, tuple[float, ...]], ...],
        /,
    ) -> None:
        self._probabilities = dict(probabilities)

    def evaluate(
        self,
        signal: InferenceSignal,
        grounded: GroundedBatch,
        batch: object,
        /,
    ) -> ResponseResult:
        if not isinstance(batch, JudgmentItems):
            msg = f"a judgment response reads ordinal judgments, not {batch!r}"
            raise TypeError(msg)

        judgments = batch.judgments
        like = grounded.log_probabilities.tensor
        positions = torch.tensor(
            [pair for pair, _ in judgments], dtype=torch.long, device=like.device
        )
        levels = torch.tensor(
            [level for _, level in judgments], dtype=torch.long, device=like.device
        )
        rows = self.response_rows(grounded.labels)
        width = len(rows[0])

        if any(not 0 <= level < width for _, level in judgments):
            msg = f"a judgment response level lies outside [0, {width}): {judgments!r}"
            raise ValueError(msg)

        if positions.numel() == 0:
            empty = like.reshape(-1)[:0]
            return ResponseResult(
                empty,
                torch.empty(0, dtype=torch.bool, device=like.device),
                positions,
                empty,
                empty if signal is InferenceSignal.REWARD else None,
            )

        instrument = like.new_tensor(rows)
        response_probabilities = like.exp().matmul(instrument)
        selected = response_probabilities[positions, levels].log()
        payments = selected.exp()
        rewards = payments if signal is InferenceSignal.REWARD else None
        return ResponseResult(
            -selected if rewards is None else -rewards,
            grounded.decided[positions],
            positions,
            payments,
            rewards,
        )

    def response_rows(
        self, labels: tuple[InferenceLabel, ...], /
    ) -> tuple[tuple[float, ...], ...]:
        """Return the declared instrument row of each grounded label.

        Raises
        ------
        ValueError
            If the instrument does not name the grounded inventory exactly.
        """
        if set(self._probabilities) != set(labels):
            msg = (
                "a judgment instrument names the grounded label inventory exactly; "
                f"instrument={tuple(self._probabilities)!r}, grounded={labels!r}"
            )
            raise ValueError(msg)

        return tuple(self._probabilities[label] for label in labels)


def _label(settings: LabelResponseSettings, /) -> _LabelResponse:
    del settings
    return _LabelResponse()


def _judgment(settings: JudgmentResponseSettings, /) -> _JudgmentResponse:
    return _JudgmentResponse(settings.probabilities)


LABEL_PLUGIN: Final = InferenceResponsePlugin(
    settings=LabelResponseSettings,
    build=_label,
)
JUDGMENT_PLUGIN: Final = InferenceResponsePlugin(
    settings=JudgmentResponseSettings,
    build=_judgment,
)
