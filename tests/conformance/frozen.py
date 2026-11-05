"""The suite every frozen value in the tree answers to."""

from copy import copy, deepcopy
from importlib import import_module
from pathlib import Path
from pickle import dumps
from pkgutil import walk_packages
from typing import Final

import pytest

import lofi
from lofi._checks import article
from lofi._frozen import FrozenValue, Unpicklable

__all__ = [
    "FROZEN_VALUES",
    "SELF_COPYING",
    "UNPICKLABLE",
    "FrozenValueConformance",
    "bare",
]

_ENTRY_POINT: Final = "__main__"

_PROBE: Final = "_conformance_probe"


def _discovered() -> tuple[type[FrozenValue], ...]:
    """Return every frozen value class the package declares, in import order.

    Returns
    -------
    tuple[type[FrozenValue], ...]
        The classes, each declared by the module whose name it is found under.
    """
    found: dict[str, type[FrozenValue]] = {}

    for module in walk_packages(lofi.__path__, f"{lofi.__name__}."):
        if module.name.rsplit(".", maxsplit=1)[-1] == _ENTRY_POINT:
            continue

        imported = import_module(module.name)

        for name, member in vars(imported).items():
            if not isinstance(member, type) or not issubclass(member, FrozenValue):
                continue

            if member is FrozenValue or member.__module__ != module.name:
                continue

            found[f"{module.name}.{name}"] = member

    return tuple(found.values())


FROZEN_VALUES: Final = _discovered()
"""Every frozen value class declared below the package root."""


def _identifiers(values: tuple[type[FrozenValue], ...]) -> tuple[str, ...]:
    """Return the test identifier for each class.

    Parameters
    ----------
    values : tuple[type[FrozenValue], ...]
        The classes to name.

    Returns
    -------
    tuple[str, ...]
        The dotted path of each class.
    """
    return tuple(f"{value.__module__}.{value.__qualname__}" for value in values)


_IDENTIFIERS: Final = _identifiers(FROZEN_VALUES)


def bare[ValueT](value: type[ValueT]) -> ValueT:
    """Return an instance of a value class with none of its fields filled.

    Parameters
    ----------
    value : type[ValueT]
        The class to instantiate.

    Returns
    -------
    ValueT
        An instance built without running ``__init__``, so that the contract the
        base establishes can be checked on a class whose fields take arguments
        the suite does not have.
    """
    return value.__new__(value)


def _declares(value: type[FrozenValue], name: str) -> bool:
    """Return whether the class supplies a dunder rather than inheriting it.

    Parameters
    ----------
    value : type[FrozenValue]
        The class to inspect.
    name : str
        The attribute looked for.

    Returns
    -------
    bool
        Whether the attribute appears on the class itself or on one of its bases
        below the shared frozen base.
    """
    return any(
        name in ancestor.__dict__
        for ancestor in value.__mro__
        if ancestor not in {FrozenValue, Unpicklable}
    )


SELF_COPYING: Final = tuple(
    value
    for value in FROZEN_VALUES
    if not _declares(value, "__copy__") and not _declares(value, "__deepcopy__")
)
"""Every frozen value class that leaves copying to the shared base."""

UNPICKLABLE: Final = tuple(
    value for value in FROZEN_VALUES if issubclass(value, Unpicklable)
)
"""Every frozen value class declaring a cause for refusing serialization."""

_SELF_COPYING_IDENTIFIERS: Final = _identifiers(SELF_COPYING)

_UNPICKLABLE_IDENTIFIERS: Final = _identifiers(UNPICKLABLE)


class FrozenValueConformance:
    pytestmark = pytest.mark.conformance

    @pytest.mark.parametrize("value", FROZEN_VALUES, ids=_IDENTIFIERS)
    def test_the_value_names_the_kind_its_refusals_report(
        self, value: type[FrozenValue]
    ) -> None:
        """Check that the class declares a noun.

        Parameters
        ----------
        value : type[FrozenValue]
            The class under test.
        """
        assert value._noun
        assert value._noun == value._noun.strip()

    @pytest.mark.parametrize("value", FROZEN_VALUES, ids=_IDENTIFIERS)
    def test_the_value_refuses_field_reassignment(
        self, value: type[FrozenValue]
    ) -> None:
        """Check that assigning to a field raises and names the value kind.

        Parameters
        ----------
        value : type[FrozenValue]
            The class under test.
        """
        subject = bare(value)

        with pytest.raises(
            AttributeError, match=f"{article(value._noun)} {value._noun} is immutable"
        ):
            setattr(subject, _PROBE, 1)

    @pytest.mark.parametrize("value", FROZEN_VALUES, ids=_IDENTIFIERS)
    def test_the_value_refuses_field_deletion(self, value: type[FrozenValue]) -> None:
        """Check that deleting a field raises and names the value kind.

        Parameters
        ----------
        value : type[FrozenValue]
            The class under test.
        """
        subject = bare(value)

        with pytest.raises(
            AttributeError, match=f"{article(value._noun)} {value._noun} is immutable"
        ):
            delattr(subject, _PROBE)

    @pytest.mark.parametrize("value", SELF_COPYING, ids=_SELF_COPYING_IDENTIFIERS)
    def test_copying_the_value_returns_the_value(
        self, value: type[FrozenValue]
    ) -> None:
        """Check that both copies of a value are the value.

        Parameters
        ----------
        value : type[FrozenValue]
            The class under test.
        """
        subject = bare(value)

        assert copy(subject) is subject
        assert deepcopy(subject) is subject

    @pytest.mark.parametrize("value", UNPICKLABLE, ids=_UNPICKLABLE_IDENTIFIERS)
    def test_the_value_refuses_serialization_naming_its_cause(
        self, value: type[Unpicklable]
    ) -> None:
        """Check that a value declaring a cause refuses to be pickled.

        Parameters
        ----------
        value : type[Unpicklable]
            The class under test.
        """
        subject = bare(value)
        reported = (
            f"{article(value._noun)} {value._noun} "
            f"{value._because} and cannot be pickled"
        )

        with pytest.raises(TypeError, match=reported):
            dumps(subject)

    @pytest.mark.parametrize("value", FROZEN_VALUES, ids=_IDENTIFIERS)
    def test_the_value_keeps_its_fields_in_slots(
        self, value: type[FrozenValue]
    ) -> None:
        """Check that the class adds no instance dictionary.

        Parameters
        ----------
        value : type[FrozenValue]
            The class under test.
        """
        subject = bare(value)

        with pytest.raises(TypeError):
            vars(subject)

    def test_the_tree_declares_frozen_values_to_check(self) -> None:
        """Check that discovery found the value classes the suite runs against."""
        assert FROZEN_VALUES
        assert len(set(FROZEN_VALUES)) == len(FROZEN_VALUES)

    def test_no_module_declares_a_frozen_base_of_its_own(self) -> None:
        """Check that the shared base is the only one the tree declares."""
        root = Path(lofi.__file__).parent
        declared = [
            path.relative_to(root.parent)
            for path in sorted(root.rglob("*.py"))
            if "__pycache__" not in path.parts
            and path.name != "_frozen.py"
            and "class _Frozen" in path.read_text(encoding="utf-8")
        ]

        assert declared == []
