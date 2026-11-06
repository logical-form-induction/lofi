"""Registered inference label vocabularies over logical relation cells."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final, override

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.lf._relation import InferenceLabel, LabelMap
from lofi.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "GRADED",
    "RELATION_LABELS",
    "RTE",
    "THREE_WAY",
    "InferenceLabeling",
    "registered_labeling",
]


RELATION_LABELS: Final = (
    InferenceLabel.ENTAILS,
    InferenceLabel.CONTRADICTS,
    InferenceLabel.NEUTRAL,
)
"""The three logical relations established by the paired proof questions."""


@final
class InferenceLabeling(FrozenValue):
    """A declared output vocabulary and coarsening of logical relations."""

    _noun = "inference labeling"
    __slots__ = ("_exclusive", "_graded", "_labels", "_name", "_targets")

    def __init__(
        self,
        name: str,
        labels: Sequence[InferenceLabel],
        targets: Mapping[InferenceLabel, InferenceLabel],
        /,
        *,
        exclusive: bool,
        graded: bool,
    ) -> None:
        held_labels = tuple(labels)

        if any(not isinstance(label, InferenceLabel) for label in held_labels):
            msg = f"inference-label outputs are InferenceLabel values: {held_labels!r}"
            raise TypeError(msg)

        if any(
            not isinstance(source, InferenceLabel)
            or not isinstance(target, InferenceLabel)
            for source, target in targets.items()
        ):
            msg = "an inference labeling maps InferenceLabel values"
            raise TypeError(msg)

        if len(held_labels) < 2 or len(set(held_labels)) != len(held_labels):
            msg = "an inference labeling has at least two distinct output labels"
            raise ValueError(msg)

        if set(targets) != set(RELATION_LABELS):
            msg = (
                "an inference labeling maps each logical relation exactly once; "
                f"received {tuple(targets)!r}"
            )
            raise ValueError(msg)

        if set(targets.values()) != set(held_labels):
            msg = (
                "every inference-label output is reached by a logical relation; "
                f"labels={held_labels!r}, targets={tuple(targets.values())!r}"
            )
            raise ValueError(msg)

        for field, value in (("exclusive", exclusive), ("graded", graded)):
            if not isinstance(value, bool):
                msg = f"an inference labeling's {field} flag is a bool, not {value!r}"
                raise TypeError(msg)

        object.__setattr__(self, "_name", ensure_name("an inference labeling", name))
        object.__setattr__(self, "_labels", held_labels)
        object.__setattr__(self, "_targets", MappingProxyType(dict(targets)))
        object.__setattr__(self, "_exclusive", exclusive)
        object.__setattr__(self, "_graded", graded)

    @property
    def name(self) -> str:
        """Return the registered labeling name."""
        return self._name

    @property
    def labels(self) -> tuple[InferenceLabel, ...]:
        """Return output labels in model-column order."""
        return self._labels

    @property
    def graded(self) -> bool:
        """Return whether responses supervise an entailment degree."""
        return self._graded

    @property
    def exclusive(self) -> bool:
        """Return whether each relation cell belongs to one output label."""
        return self._exclusive

    def encode(self, relation: InferenceLabel, /) -> InferenceLabel:
        """Map one logical relation to this labeling's observed label."""
        if not isinstance(relation, InferenceLabel):
            msg = f"an inference relation is an InferenceLabel, not {relation!r}"
            raise TypeError(msg)

        try:
            return self._targets[relation]
        except KeyError:
            msg = f"{relation!r} is not a logical inference relation"
            raise ValueError(msg) from None

    def label_map(self, relation_map: LabelMap, /) -> LabelMap:
        """Coarsen a language's relation cell semantics into this vocabulary."""
        if tuple(relation_map.labels) != RELATION_LABELS:
            msg = (
                "an inference labeling coarsens the three logical relations, not "
                f"{relation_map.labels!r}"
            )
            raise ValueError(msg)

        if not self._exclusive and all(
            self._targets[relation] is relation for relation in RELATION_LABELS
        ):
            return relation_map

        cells = {
            label: {
                cell
                for relation in RELATION_LABELS
                if self._targets[relation] is label
                for cell in relation_map.cells[relation]
            }
            for label in self._labels
        }

        if self._exclusive:
            claimed = set()
            for label in self._labels:
                cells[label].difference_update(claimed)
                claimed.update(cells[label])

        return LabelMap(
            f"{relation_map.name}:{self._name}",
            {label: frozenset(cells[label]) for label in self._labels},
        )

    @override
    def __eq__(self, other: object) -> bool:
        return (
            type(other) is InferenceLabeling
            and self._name == other._name
            and self._labels == other._labels
            and self._targets == other._targets
            and self._exclusive == other._exclusive
            and self._graded == other._graded
        )

    @override
    def __hash__(self) -> int:
        return hash(
            (
                InferenceLabeling,
                self._name,
                self._labels,
                tuple(self._targets.items()),
                self._exclusive,
                self._graded,
            )
        )

    @override
    def __repr__(self) -> str:
        return (
            f"InferenceLabeling({self._name!r}, {self._labels!r}, "
            f"{dict(self._targets)!r}, exclusive={self._exclusive!r}, "
            f"graded={self._graded!r})"
        )


THREE_WAY: Final = InferenceLabeling(
    "three-way",
    RELATION_LABELS,
    {relation: relation for relation in RELATION_LABELS},
    exclusive=False,
    graded=False,
)
"""Entailment, contradiction, and neutral as distinct categorical labels."""

RTE: Final = InferenceLabeling(
    "rte",
    (InferenceLabel.ENTAILS, InferenceLabel.NON_ENTAILS),
    {
        InferenceLabel.ENTAILS: InferenceLabel.ENTAILS,
        InferenceLabel.CONTRADICTS: InferenceLabel.NON_ENTAILS,
        InferenceLabel.NEUTRAL: InferenceLabel.NON_ENTAILS,
    },
    exclusive=True,
    graded=False,
)
"""The RTE entailment versus nonentailment categorical labeling."""

GRADED: Final = InferenceLabeling(
    "graded",
    RTE.labels,
    {
        InferenceLabel.ENTAILS: InferenceLabel.ENTAILS,
        InferenceLabel.CONTRADICTS: InferenceLabel.NON_ENTAILS,
        InferenceLabel.NEUTRAL: InferenceLabel.NON_ENTAILS,
    },
    exclusive=True,
    graded=True,
)
"""An entailment degree grounded by the RTE probability scale."""


def registered_labeling(name: str, /) -> InferenceLabeling:
    """Return the installed inference label vocabulary of one name.

    Parameters
    ----------
    name : str
        The entry name in the ``lofi.inference_labeling`` group.

    Returns
    -------
    InferenceLabeling
        The installed label vocabulary.

    Raises
    ------
    TypeError
        If the entry names something other than an inference labeling.
    """
    member = Registry[object]("inference_labeling").get(name)

    if not isinstance(member, InferenceLabeling):
        msg = (
            f"the inference-labeling entry {name!r} names {member!r}, "
            "not an InferenceLabeling"
        )
        raise TypeError(msg)

    return member
