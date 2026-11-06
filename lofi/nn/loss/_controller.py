"""Supervision of the controller on labeled rule instances."""

from typing import TYPE_CHECKING, Final, override

import torch

from lofi._checks import ensure_name
from lofi.nn._values import Encoded
from lofi.nn.loss._likelihood import live_rows
from lofi.nn.loss._ports import LossTerm

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.judge import Action, ActionSet
    from lofi.nn.loss._ports import PublishedOutputs, RuleInstances
    from lofi.nn.ports import Controller, TermEncoder
    from lofi.terms import TermCodec

__all__ = ["ControllerSupervisionObjective"]

_CELL_AXES: Final[tuple[str, ...]] = ("cell", "width")
"""The axes a controller reads one vector per pair over."""

_WEIGHT_AXES: Final[tuple[str, ...]] = ("cell", "action")
"""The axes a controller weighs the actions of each pair over."""


def _column(action: Action, action_set: ActionSet) -> int:
    """Return the column a controller weighs one action at.

    Parameters
    ----------
    action : Action
        The action a label names.
    action_set : ActionSet
        The actions the controller weighs, in its column order.

    Raises
    ------
    ValueError
        If the set holds no action of that name, or holds one applying another rule.
    """
    try:
        column = action_set.index(action.name)

    except KeyError as error:
        msg = f"a label names {action!r}, and the controller weighs no such action"
        raise ValueError(msg) from error

    held = action_set.actions[column]

    if held != action:
        msg = f"a label names {action!r}, and the controller weighs {held!r}"
        raise ValueError(msg)

    return column


def _checked_weights(
    weights: Tensor, axes: tuple[str, ...], rows: int, width: int
) -> None:
    """Raise unless a controller weighed every action of every instance.

    Parameters
    ----------
    weights : Tensor
        What the controller returned.
    axes : tuple[str, ...]
        The axes it declares the weights over.
    rows : int
        How many instances it was handed.
    width : int
        How many actions its set holds.

    Raises
    ------
    ValueError
        If the weights are not over the cell and then the action, or do not hold one
        row per instance and one column per action.
    """
    if axes != _WEIGHT_AXES or tuple(weights.shape) != (rows, width):
        msg = (
            f"the controller weighed {axes} of shape {tuple(weights.shape)}, and "
            f"{rows} instances are weighed over {width} actions along {_WEIGHT_AXES}"
        )
        raise ValueError(msg)


class ControllerSupervisionObjective[BatchT: RuleInstances]:
    """Scores how much weight the controller puts on the actions admitting a pair.

    Each instance contributes the negative log of the controller mass on its admitted
    actions. Where none is admitted it scores the mass on the null action, or on
    the identity action where the controller's set declares no null action. Rows
    missing either daughter are excluded.

    Parameters
    ----------
    encoder : TermEncoder
        What encodes each daughter to the vector the controller reads.
    codec : TermCodec
        What linearizes the binary premises the controller reads.
    controller : Controller
        What weighs the actions over a pair of encoded daughters.
    name : str
        The name the term is logged and namespaced under.

    Raises
    ------
    TypeError
        If the name is not a string.
    ValueError
        If the name is empty.
    """

    __slots__ = ("_codec", "_controller", "_encoder", "_name")

    def __init__(
        self,
        encoder: TermEncoder,
        codec: TermCodec,
        controller: Controller,
        *,
        name: str = "controller-supervision",
    ) -> None:
        self._encoder = encoder
        self._codec = codec
        self._controller = controller
        self._name = ensure_name("the objective's name", name)

    @property
    def name(self) -> str:
        """Return the name this objective is logged and namespaced under."""
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        """Return the published outputs this objective reads, which is the batch."""
        return frozenset({"batch"})

    @property
    def action_set(self) -> ActionSet:
        """Return the actions the controller weighs, in its column order."""
        return self._controller.action_set

    def _admitted(
        self, labels: tuple[tuple[Action, ...], ...], device: torch.device
    ) -> Tensor:
        """Return which column each instance's label takes mass from.

        Parameters
        ----------
        labels : tuple[tuple[Action, ...], ...]
            The actions admitting each instance, empty where none does.
        device : torch.device
            Where the controller's weights sit.

        Returns
        -------
        Tensor
            A boolean tensor over the instance and the column, true at the columns of
            the actions admitting the instance. Where no action does, it is true at
            the null action's column, or at the identity's where the set declares no
            null action.

        Raises
        ------
        ValueError
            If a label names an action the controller does not weigh.
        """
        action_set = self._controller.action_set
        null = action_set.null
        unadmitted = action_set.identity if null is None else null
        fallback = action_set.index(unadmitted.name)
        admitted = torch.zeros(
            len(labels), len(action_set.actions), dtype=torch.bool, device=device
        )

        for row, label in enumerate(labels):
            columns = [_column(action, action_set) for action in label] or [fallback]
            admitted[row, columns] = True

        return admitted

    def evaluate(self, outputs: PublishedOutputs[BatchT], /) -> LossTerm:
        """Return the mean negative log mass of the step's labels.

        Parameters
        ----------
        outputs : PublishedOutputs[BatchT]
            What the graph published for the step.

        Returns
        -------
        LossTerm
            The mean over the instances holding two terms of the negative log mass the
            controller puts on the instance's label, counted in those instances,
            reporting the fraction of them whose heaviest action is in the label as
            ``accuracy``.

        Raises
        ------
        ValueError
            If the two linearizations do not hold one row per label, if a label names
            an action the controller does not weigh, or if the controller does not
            weigh every action of every instance.
        """
        instances = tuple(
            instance
            for instance in outputs.batch.instances
            if len(instance.inputs) == 2
        )

        if not instances:
            return LossTerm.inapplicable(torch.zeros((), requires_grad=True))

        labels = tuple(instance.actions for instance in instances)
        left = self._encoder.encode(
            self._codec.encode_batch(
                tuple(instance.inputs[0] for instance in instances)
            )
        )
        right = self._encoder.encode(
            self._codec.encode_batch(
                tuple(instance.inputs[1] for instance in instances)
            )
        )
        weighed = self._controller.weigh(
            Encoded(left.tensor, _CELL_AXES, left.mask),
            right=Encoded(right.tensor, _CELL_AXES, right.mask),
        )
        scores = weighed.tensor
        _checked_weights(
            scores, weighed.axes, len(labels), len(self._controller.action_set.actions)
        )
        live = live_rows((left.mask, right.mask, weighed.mask))

        if live is not None:
            scores = torch.where(live.unsqueeze(-1), scores, scores.new_zeros(()))

        admitted = self._admitted(labels, scores.device)
        log_probabilities = scores.log_softmax(dim=-1)
        labeled = torch.where(
            admitted, log_probabilities, log_probabilities.new_full((), -torch.inf)
        )
        losses = -labeled.logsumexp(dim=-1)
        heaviest = log_probabilities.detach().argmax(dim=-1, keepdim=True)
        hits = admitted.gather(1, heaviest).squeeze(1)

        if live is None:
            count = int(losses.shape[0])
        else:
            count = int(live.sum())
            losses = torch.where(live, losses, losses.new_zeros(()))
            hits = hits & live

        if count == 0:
            return LossTerm.inapplicable(scores)

        return LossTerm(
            losses.sum() / count,
            count,
            diagnostics={"accuracy": hits.to(scores.dtype).sum() / count},
        )

    @override
    def __repr__(self) -> str:
        """Return the name and the two components' types."""
        return (
            f"ControllerSupervisionObjective(name={self._name!r}, "
            f"encoder={type(self._encoder).__name__}, "
            f"controller={type(self._controller).__name__})"
        )
