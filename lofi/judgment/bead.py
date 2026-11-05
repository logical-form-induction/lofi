"""Logical-form prediction capabilities consumed by bead serving."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Protocol, final, override

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.judgment._spec import Choice, Cumulative, ResponseFamily

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    from bead.protocol import ResponseEncoding

    from lofi.lf import InferenceLabel

__all__ = [
    "InferenceText",
    "PointInputs",
    "RenderedItem",
    "TrainedPointPredictor",
    "response_family_for",
]


class RenderedItem(Protocol):
    """An identified item and the text it renders, as a bead item has them."""

    @property
    def id(self) -> UUID | str:
        """Return the item's identity."""
        ...

    @property
    def rendered_elements(self) -> Mapping[str, str]:
        """Return the item's text by element name."""
        ...


@final
class PointInputs(FrozenValue):
    """The rendered element each input role of a trained point reads.

    A trained point publishes the roles its input fills; a declaration maps
    each of them to the element of an item that fills it.

    Parameters
    ----------
    elements : Mapping[str, str]
        The element each role reads, keyed by role.
    roles : tuple[str, ...]
        The input roles the trained point publishes, in its order.

    Raises
    ------
    ValueError
        If the declaration maps other roles than the point publishes, or a
        role or element name is empty.
    """

    _noun = "trained-point inputs"
    __slots__ = ("_elements",)

    def __init__(
        self, elements: Mapping[str, str], /, *, roles: tuple[str, ...]
    ) -> None:
        held = {
            ensure_name("an input role", role): ensure_name(
                f"input role {role!r}'s element", element
            )
            for role, element in elements.items()
        }
        if set(held) != set(roles):
            msg = (
                f"the trained point reads the input roles {sorted(roles)!r}, and "
                f"the declaration maps {sorted(held)!r}"
            )
            raise ValueError(msg)
        object.__setattr__(
            self, "_elements", MappingProxyType({role: held[role] for role in roles})
        )

    @property
    def elements(self) -> Mapping[str, str]:
        """Return the element each role reads, in the point's role order."""
        return self._elements

    def texts(self, item: RenderedItem, /) -> dict[str, str]:
        """Return the text each role reads from one item.

        Raises
        ------
        ValueError
            If the item renders a declared element nowhere.
        """
        rendered = item.rendered_elements
        texts: dict[str, str] = {}
        for role, element in self._elements.items():
            text = rendered.get(element)
            if text is None:
                offered = ", ".join(sorted(rendered)) or "nothing"
                msg = (
                    f"item {item.id} renders no {element!r} element for the input "
                    f"role {role!r}; it renders {offered}"
                )
                raise ValueError(msg)
            texts[role] = text
        return texts

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, PointInputs) and dict(self._elements) == dict(
            other._elements
        )

    @override
    def __hash__(self) -> int:
        return hash((PointInputs, tuple(self._elements.items())))

    @override
    def __repr__(self) -> str:
        return f"PointInputs({dict(self._elements)!r})"


@final
class InferenceText(FrozenValue):
    """One identified premise and hypothesis accepted by a trained point.

    The point publishes its input roles as ``ROLES``.
    """

    ROLES: ClassVar[tuple[str, str]] = ("premise", "hypothesis")
    """The input roles an inference point reads, in its order."""

    _noun = "inference text pair"
    __slots__ = ("_hypothesis", "_item_id", "_premise")

    @classmethod
    def of(cls, item: RenderedItem, inputs: PointInputs, /) -> InferenceText:
        """Return one item's inference input through the declared role mapping.

        Raises
        ------
        ValueError
            If the item renders a mapped element nowhere.
        """
        texts = inputs.texts(item)
        premise, hypothesis = (texts[role] for role in cls.ROLES)
        return cls(str(item.id), premise, hypothesis)

    def __init__(self, item_id: str, premise: str, hypothesis: str) -> None:
        object.__setattr__(
            self, "_item_id", ensure_name("an inference item id", item_id)
        )
        object.__setattr__(
            self, "_premise", ensure_name("an inference premise", premise)
        )
        object.__setattr__(
            self, "_hypothesis", ensure_name("an inference hypothesis", hypothesis)
        )

    @property
    def item_id(self) -> str:
        """Return the source item's preserved identifier."""
        return self._item_id

    @property
    def premise(self) -> str:
        """Return the premise text."""
        return self._premise

    @property
    def hypothesis(self) -> str:
        """Return the hypothesis text."""
        return self._hypothesis

    @override
    def __eq__(self, other: object) -> bool:
        return (
            type(other) is InferenceText
            and self._item_id == other._item_id
            and self._premise == other._premise
            and self._hypothesis == other._hypothesis
        )

    @override
    def __hash__(self) -> int:
        return hash((InferenceText, self._item_id, self._premise, self._hypothesis))

    @override
    def __repr__(self) -> str:
        return (
            f"InferenceText({self._item_id!r}, {self._premise!r}, {self._hypothesis!r})"
        )


class TrainedPointPredictor(Protocol):
    """Predict through a restored trained point's published sites."""

    @property
    def response_levels(self) -> int:
        """Return the width of the declared response distribution."""
        ...

    @property
    def relation_labels(self) -> tuple[InferenceLabel, ...]:
        """Return the relation-probability columns in model order."""
        ...

    def predict_responses(
        self, pairs: Sequence[InferenceText], /
    ) -> tuple[tuple[float, ...], ...]:
        """Return the point's declared judgment probabilities."""
        ...

    def predict_relations(
        self, pairs: Sequence[InferenceText], /
    ) -> tuple[tuple[float, ...], ...]:
        """Return the point's inference-label probabilities."""
        ...


def response_family_for(encoding: ResponseEncoding, /) -> ResponseFamily:
    """Return LoFI's declared response family for a bead encoding."""
    if encoding.is_binary or encoding.is_ordinal:
        return Cumulative()
    return Choice()
