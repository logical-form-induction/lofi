"""Regularization over a globally normalized tree distribution."""

from typing import TYPE_CHECKING, Final, override

from lofi._checks import ensure_name
from lofi.nn.loss._errors import MissingOutputError
from lofi.nn.loss._ports import LossTerm
from lofi.registry import ComponentPlugin
from lofi.spec.losses.derivation_entropy import (
    DerivationEntropyDirection,
    DerivationEntropyLossSettings,
    DerivationEntropyLossSpec,
)

if TYPE_CHECKING:
    from lofi.nn.loss._ports import LossComponentBindings, PublishedOutputs

__all__ = ["DerivationEntropyObjective"]


class DerivationEntropyObjective[BatchT]:
    """Minimize or maximize mean derivation entropy over a batch."""

    __slots__ = ("_direction", "_name")

    def __init__(
        self,
        direction: DerivationEntropyDirection,
        *,
        name: str,
    ) -> None:
        if direction not in {"minimize", "maximize"}:
            msg = f"an entropy direction is 'minimize' or 'maximize', not {direction!r}"
            raise ValueError(msg)

        self._direction = direction
        self._name = ensure_name("the objective's name", name)

    @property
    def name(self) -> str:
        """Return the name under which the term is logged."""
        return self._name

    @property
    def direction(self) -> DerivationEntropyDirection:
        """Return whether fitting concentrates or spreads posterior mass."""
        return self._direction

    @property
    def requires(self) -> frozenset[str]:
        """Return the parser publication consumed by this objective."""
        return frozenset({"entropy"})

    def evaluate(self, outputs: PublishedOutputs[BatchT], /) -> LossTerm:
        """Return signed mean entropy and its unsigned diagnostic."""
        entropy = outputs.entropy

        if entropy is None:
            raise MissingOutputError(
                objective=self._name,
                missing=self.requires,
                published=frozenset(),
            )

        if entropy.axes != ("item",):
            msg = f"derivation entropy runs over ('item',), not {entropy.axes}"
            raise ValueError(msg)

        values = entropy.tensor
        live = entropy.mask

        if live is not None:
            values = values[live]

        count = int(values.numel())

        if count == 0:
            return LossTerm.inapplicable(entropy.tensor)

        mean = values.mean()
        objective = mean if self._direction == "minimize" else -mean
        return LossTerm(
            objective,
            count,
            diagnostics={"entropy": mean.detach()},
        )

    @override
    def __repr__(self) -> str:
        """Return the objective name and direction."""
        return (
            f"DerivationEntropyObjective(name={self._name!r}, "
            f"direction={self._direction!r})"
        )


def _build[BatchT](
    settings: DerivationEntropyLossSettings,
    context: LossComponentBindings,
    /,
) -> DerivationEntropyObjective[BatchT]:
    """Build the declared entropy regularizer."""
    return DerivationEntropyObjective(settings.direction, name=context.name)


PLUGIN: Final = ComponentPlugin(spec=DerivationEntropyLossSpec, build=_build)
"""The derivation-entropy objective plugin."""
