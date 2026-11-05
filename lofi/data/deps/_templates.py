"""Immutable declarations for nonlocal CCG dependencies."""

from typing import TYPE_CHECKING, Literal, final, override

from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable

__all__ = [
    "CCGDependencyConvention",
    "CCGDependencyTemplate",
    "CCGProjectionReport",
    "CategoryDirection",
    "CoindexedPath",
]

#: A result or argument step in a structural category path.
type CategoryDirection = Literal["result", "argument"]

_MIN_COINDEX_OCCURRENCES = 2


@final
class CoindexedPath(FrozenValue):
    """A structural category path assigned a nonlocal coindex.

    Parameters
    ----------
    path : Iterable[CategoryDirection]
        Result and argument steps from the lexical category root.
    index : int
        Positive coindex shared by two or more paths in a template.

    Raises
    ------
    TypeError
        If the coindex is not an integer.
    ValueError
        If the path is empty or invalid, or the coindex is not positive.
    """

    _noun = "coindexed path"

    __slots__ = ("_index", "_path")

    def __init__(self, path: Iterable[CategoryDirection], index: int) -> None:
        held_path = tuple(path)

        if not held_path:
            msg = "a coindexed category path is empty"
            raise ValueError(msg)

        if any(direction not in ("result", "argument") for direction in held_path):
            msg = f"a coindexed category path is invalid: {held_path!r}"
            raise ValueError(msg)

        if isinstance(index, bool) or not isinstance(index, int):
            msg = f"a category coindex is not an integer: {index!r}"
            raise TypeError(msg)

        if index < 1:
            msg = f"a category coindex is not positive: {index}"
            raise ValueError(msg)

        object.__setattr__(self, "_path", held_path)
        object.__setattr__(self, "_index", index)

    @property
    def path(self) -> tuple[CategoryDirection, ...]:
        """Return the structural path from the category root."""
        return self._path

    @property
    def index(self) -> int:
        """Return the positive coindex."""
        return self._index

    def _fields(self) -> tuple[tuple[CategoryDirection, ...], int]:
        """Return fields in declaration order."""
        return self._path, self._index

    @override
    def __reduce__(
        self,
    ) -> tuple[type[CoindexedPath], tuple[tuple[CategoryDirection, ...], int]]:
        """Return the reconstruction call."""
        return CoindexedPath, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another path has the same fields."""
        if type(other) is not CoindexedPath:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((CoindexedPath, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"CoindexedPath{self._fields()!r}"


@final
class CCGDependencyTemplate[C: Hashable](FrozenValue):
    """A category and its repeated nonlocal coindices.

    Parameters
    ----------
    name : str
        Nonempty template name.
    category : C
        Category matched through a category algebra.
    coindices : Iterable[CoindexedPath]
        Structural paths grouped by positive coindex.

    Raises
    ------
    ValueError
        If the name or coindices are empty, a path repeats, or a coindex occurs fewer
        than twice.
    """

    _noun = "CCG dependency template"

    __slots__ = ("_category", "_coindices", "_name")

    def __init__(
        self,
        name: str,
        category: C,
        coindices: Iterable[CoindexedPath],
    ) -> None:
        if not name:
            msg = "a CCG dependency template has no name"
            raise ValueError(msg)

        held_coindices = tuple(coindices)

        if not held_coindices:
            msg = f"CCG dependency template {name!r} has no coindices"
            raise ValueError(msg)

        paths = tuple(binding.path for binding in held_coindices)

        if len(frozenset(paths)) != len(paths):
            msg = f"CCG dependency template {name!r} repeats a category path"
            raise ValueError(msg)

        counts: dict[int, int] = {}

        for binding in held_coindices:
            counts[binding.index] = counts.get(binding.index, 0) + 1

        singleton = tuple(
            index for index, count in counts.items() if count < _MIN_COINDEX_OCCURRENCES
        )

        if singleton:
            msg = (
                f"CCG dependency template {name!r} has unshared coindices: "
                f"{singleton!r}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_category", category)
        object.__setattr__(self, "_coindices", held_coindices)

    @property
    def name(self) -> str:
        """Return the template name."""
        return self._name

    @property
    def category(self) -> C:
        """Return the category matched by this template."""
        return self._category

    @property
    def coindices(self) -> tuple[CoindexedPath, ...]:
        """Return the declared structural coindices."""
        return self._coindices

    def _fields(self) -> tuple[str, C, tuple[CoindexedPath, ...]]:
        """Return fields in declaration order."""
        return self._name, self._category, self._coindices

    @override
    def __reduce__(
        self,
    ) -> tuple[
        type[CCGDependencyTemplate[C]],
        tuple[str, C, tuple[CoindexedPath, ...]],
    ]:
        """Return the reconstruction call."""
        return CCGDependencyTemplate, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another template has the same fields."""
        if type(other) is not CCGDependencyTemplate:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((CCGDependencyTemplate, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"CCGDependencyTemplate{self._fields()!r}"


@final
class CCGDependencyConvention[C: Hashable](FrozenValue):
    """A named immutable set of nonlocal dependency templates.

    Parameters
    ----------
    name : str
        Nonempty convention name.
    templates : Iterable[CCGDependencyTemplate[C]]
        Declared templates. An empty sequence is valid.

    Raises
    ------
    ValueError
        If the name is empty.
    """

    _noun = "CCG dependency convention"

    __slots__ = ("_name", "_templates")

    def __init__(
        self,
        name: str,
        templates: Iterable[CCGDependencyTemplate[C]] = (),
    ) -> None:
        if not name:
            msg = "a CCG dependency convention has no name"
            raise ValueError(msg)

        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_templates", tuple(templates))

    @property
    def name(self) -> str:
        """Return the convention name."""
        return self._name

    @property
    def templates(self) -> tuple[CCGDependencyTemplate[C], ...]:
        """Return the declared templates."""
        return self._templates

    def _fields(self) -> tuple[str, tuple[CCGDependencyTemplate[C], ...]]:
        """Return fields in declaration order."""
        return self._name, self._templates

    @override
    def __reduce__(
        self,
    ) -> tuple[
        type[CCGDependencyConvention[C]],
        tuple[str, tuple[CCGDependencyTemplate[C], ...]],
    ]:
        """Return the reconstruction call."""
        return CCGDependencyConvention, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another convention has the same fields."""
        if type(other) is not CCGDependencyConvention:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((CCGDependencyConvention, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"CCGDependencyConvention{self._fields()!r}"


@final
class CCGProjectionReport(FrozenValue):
    """Template declarations and matches for one CCG projection.

    Parameters
    ----------
    convention_name : str | None
        Supplied convention name, or None when no convention was supplied.
    declared_template_count : int
        Number of templates in the supplied convention.
    matched_template_count : int
        Number of lexical template matches in the projection.

    Raises
    ------
    TypeError
        If either count is not an integer.
    ValueError
        If the convention name is empty, a count is negative, or an absent
        convention has nonzero counts.
    """

    _noun = "CCG projection report"

    __slots__ = ("_convention_name", "_declared", "_matched")

    def __init__(
        self,
        convention_name: str | None,
        declared_template_count: int,
        matched_template_count: int,
    ) -> None:
        if convention_name == "":
            msg = "a present CCG dependency convention has no name"
            raise ValueError(msg)

        for label, count in (
            ("declared template", declared_template_count),
            ("matched template", matched_template_count),
        ):
            if isinstance(count, bool) or not isinstance(count, int):
                msg = f"a {label} count is not an integer: {count!r}"
                raise TypeError(msg)

            if count < 0:
                msg = f"a {label} count is negative: {count}"
                raise ValueError(msg)

        if convention_name is None and (
            declared_template_count or matched_template_count
        ):
            msg = "an absent CCG dependency convention has template counts"
            raise ValueError(msg)

        object.__setattr__(self, "_convention_name", convention_name)
        object.__setattr__(self, "_declared", declared_template_count)
        object.__setattr__(self, "_matched", matched_template_count)

    @property
    def convention_present(self) -> bool:
        """Return whether the caller supplied a convention."""
        return self._convention_name is not None

    @property
    def convention_name(self) -> str | None:
        """Return the convention name, or None if none was supplied."""
        return self._convention_name

    @property
    def declared_template_count(self) -> int:
        """Return the number of declared templates."""
        return self._declared

    @property
    def matched_template_count(self) -> int:
        """Return the number of lexical template matches."""
        return self._matched

    def _fields(self) -> tuple[str | None, int, int]:
        """Return fields in declaration order."""
        return self._convention_name, self._declared, self._matched

    @override
    def __reduce__(
        self,
    ) -> tuple[type[CCGProjectionReport], tuple[str | None, int, int]]:
        """Return the reconstruction call."""
        return CCGProjectionReport, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report has the same fields."""
        if type(other) is not CCGProjectionReport:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((CCGProjectionReport, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"CCGProjectionReport{self._fields()!r}"
