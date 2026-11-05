"""Supervision of combinator decoders on labeled rule instances."""

from types import MappingProxyType
from typing import TYPE_CHECKING, override

import torch

from lofi._checks import ensure_name
from lofi.nn.loss._likelihood import negative_log_likelihood
from lofi.nn.loss._ports import LossTerm
from lofi.nn.loss._schedule import TeacherForcingSchedule

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.judge import Action
    from lofi.nn.loss._ports import (
        CombinatorScorer,
        PublishedOutputs,
        ScheduledRuleInstances,
    )
    from lofi.nn.ports import ForwardOptions

__all__ = ["CombinatorSupervisionObjective"]


def _checked_decoders(
    decoders: Mapping[Action, CombinatorScorer], schedule: TeacherForcingSchedule
) -> Mapping[Action, CombinatorScorer]:
    """Return the decoders, keyed by the action each composes under.

    Parameters
    ----------
    decoders : Mapping[Action, CombinatorScorer]
        The decoder composing under each action.
    schedule : TeacherForcingSchedule
        The passes the decoders are to run.

    Returns
    -------
    Mapping[Action, CombinatorScorer]
        A read-only view of the decoders.

    Raises
    ------
    ValueError
        If there is no decoder, or if one does not run in a mode the schedule names.
    """
    if not decoders:
        msg = "combinator supervision trains at least one decoder"
        raise ValueError(msg)

    for action, decoder in decoders.items():
        missing = schedule.modes - decoder.modes

        if missing:
            names = ", ".join(sorted(mode.name for mode in missing))
            msg = (
                f"the schedule runs {names} passes, which the decoder composing "
                f"under {action.name!r} does not run"
            )
            raise ValueError(msg)

    return MappingProxyType(dict(decoders))


class CombinatorSupervisionObjective[BatchT: ScheduledRuleInstances]:
    """Scores how poorly each action's decoder reads a conclusion from its premises.

    Each admitted action scores the conclusion from the encoded daughters under the
    scheduled pass. The objective averages negative log weights over scored
    instance-action pairs and excludes instances admitted by no action.

    Parameters
    ----------
    decoders : Mapping[Action, CombinatorScorer]
        The scorer for each action, keyed by the action.
    schedule : TeacherForcingSchedule | None
        Which pass the decoders run at each step of the run.
    name : str
        The name the term is logged and namespaced under.

    Raises
    ------
    TypeError
        If the name is not a string.
    ValueError
        If the name is empty, if there is no decoder, or if one does not run in a
        mode the schedule names.
    """

    __slots__ = ("_decoders", "_name", "_schedule")

    def __init__(
        self,
        decoders: Mapping[Action, CombinatorScorer],
        *,
        schedule: TeacherForcingSchedule | None = None,
        name: str = "combinator-supervision",
    ) -> None:
        self._schedule = TeacherForcingSchedule() if schedule is None else schedule
        self._decoders = _checked_decoders(decoders, self._schedule)
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
    def decoders(self) -> Mapping[Action, CombinatorScorer]:
        """Return the decoder composing under each action, keyed by the action."""
        return self._decoders

    @property
    def schedule(self) -> TeacherForcingSchedule:
        """Return which pass the decoders run at each step of the run."""
        return self._schedule

    def _grouped(
        self, labels: tuple[tuple[Action, ...], ...]
    ) -> Mapping[Action, tuple[int, ...]]:
        """Return the instances each decoder scores.

        Parameters
        ----------
        labels : tuple[tuple[Action, ...], ...]
            The actions admitting each instance.

        Returns
        -------
        Mapping[Action, tuple[int, ...]]
            The rows admitted by each action, ascending, keyed by the action.

        Raises
        ------
        ValueError
            If an instance is admitted by an action no decoder composes under.
        """
        groups: dict[Action, list[int]] = {action: [] for action in self._decoders}

        for row, label in enumerate(labels):
            for action in label:
                rows = groups.get(action)

                if rows is None:
                    known = ", ".join(sorted(held.name for held in self._decoders))
                    msg = (
                        f"instance {row} is admitted by {action!r}, and {self._name} "
                        f"composes under {known}"
                    )
                    raise ValueError(msg)

                rows.append(row)

        return {action: tuple(rows) for action, rows in groups.items()}

    def evaluate(self, outputs: PublishedOutputs[BatchT], /) -> LossTerm:
        """Return the mean negative log probability of the step's conclusions.

        Parameters
        ----------
        outputs : PublishedOutputs[BatchT]
            What the graph published for the step.

        Returns
        -------
        LossTerm
            The mean over the instance and action pairs scored of the conclusion's
            negative log probability under the action's decoder, counted in those
            pairs, reporting the same mean per slot of the linearization as
            ``per_slot`` and the fraction of the batch the pass drew as ``drawn``.

        Raises
        ------
        ValueError
            If the three linearizations do not hold one row per label, if an instance
            is admitted by an action no decoder composes under, or if a decoder does
            not weigh one conclusion per instance it was handed.
        """
        batch = outputs.batch
        instances = tuple(batch.instances)
        labels = tuple(instance.actions for instance in instances)

        options = self._schedule.at(batch.step, horizon=batch.horizon)
        total: torch.Tensor | None = None
        correct: torch.Tensor | None = None
        count = 0

        for action, chosen in self._grouped(labels).items():
            if not chosen:
                continue

            scorer = self._decoders[action]
            selected = tuple(instances[row] for row in chosen)

            for row, instance in zip(chosen, selected, strict=True):
                if len(instance.inputs) != scorer.arity:
                    msg = (
                        f"instance {row} has arity {len(instance.inputs)}, but "
                        f"{action.name!r} scores arity {scorer.arity}"
                    )
                    raise ValueError(msg)

            weights = scorer.score(
                tuple(instance.inputs for instance in selected),
                conclusions=tuple(instance.output for instance in selected),
                options=options,
            )
            summed, live = negative_log_likelihood(
                weights,
                terms=len(chosen),
                what=f"the conclusions under {action.name!r}",
            )
            matches = scorer.exact_matches(
                tuple(instance.inputs for instance in selected),
                conclusions=tuple(instance.output for instance in selected),
                options=options,
            )

            if tuple(matches.shape) != (len(chosen),):
                msg = (
                    f"the scorer under {action.name!r} returned exact matches of "
                    f"shape {tuple(matches.shape)}, not {(len(chosen),)}"
                )
                raise ValueError(msg)

            live_mask = weights.mask

            if live_mask is not None:
                matches = matches & live_mask

            matched = matches.to(summed.dtype).sum()
            total = summed if total is None else total + summed
            correct = matched if correct is None else correct + matched
            count += live

        if count == 0 or total is None or correct is None:
            return LossTerm.inapplicable(torch.zeros((), requires_grad=True))

        value = total / count

        return LossTerm(
            value,
            count,
            diagnostics={
                "exact_match": correct.detach() / count,
                "per_slot": value.detach() / next(iter(self._decoders.values())).slots,
                "drawn": value.new_tensor(_drawn(options)),
            },
        )

    @override
    def __repr__(self) -> str:
        """Return the name, each action's decoder type and the schedule."""
        named = ", ".join(
            f"{action.name}: {type(decoder).__name__}"
            for action, decoder in self._decoders.items()
        )

        return (
            f"CombinatorSupervisionObjective(name={self._name!r}, "
            f"decoders={{{named}}}, schedule={self._schedule!r})"
        )


def _drawn(options: ForwardOptions) -> float:
    """Return the fraction of the batch a pass draws.

    Parameters
    ----------
    options : ForwardOptions
        The pass.

    Returns
    -------
    float
        The sampling rate of a sampling pass, and zero for any other.
    """
    if options.declares("sampling_rate"):
        return options.sampling_rate

    return 0.0
