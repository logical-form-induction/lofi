"""Report audits against preregistered analyses."""

import json
from types import MappingProxyType
from typing import TYPE_CHECKING, override

from lofi._checks import ensure_finite, ensure_name
from lofi.eval._boundary import ValidationError, decode_preregistration
from lofi.eval._values import Comparison

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi._types import JsonValue

__all__ = [
    "DeclaredHypothesis",
    "Preregistration",
    "PreregistrationAudit",
    "PreregistrationError",
    "UndeclaredFamilyError",
    "check_preregistration",
    "read_preregistration",
]


class PreregistrationError(ValueError):
    """Signals that a report and the plan it was preregistered under disagree."""


class UndeclaredFamilyError(PreregistrationError):
    """Signals a comparison corrected within a family the plan does not declare."""


class DeclaredHypothesis:
    """A preregistered hypothesis, family, and margin.

    Attributes
    ----------
    name : str
        The hypothesis name.
    left : str
        The left comparison arm.
    right : str
        The right comparison arm.
    family : str
        The multiple comparison family.
    margin : float | None
        The optional equivalence margin.
    pair : tuple[str, str]
        The two arms in canonical order.

    Parameters
    ----------
    name : str
        The hypothesis name.
    left : str
        The left comparison arm.
    right : str
        The right comparison arm.
    family : str
        The multiple comparison family.
    margin : float | None
        The optional equivalence margin.

    Raises
    ------
    TypeError
        If a text field or margin has an invalid type.
    ValueError
        If a text field, arm pair, or margin is invalid.
    """

    __slots__ = ("_family", "_left", "_margin", "_name", "_right")

    def __init__(
        self,
        name: str,
        left: str,
        right: str,
        *,
        family: str,
        margin: float | None = None,
    ) -> None:
        self._name = ensure_name("a declared hypothesis", name)
        self._left = ensure_name("a declared hypothesis's left arm", left)
        self._right = ensure_name("a declared hypothesis's right arm", right)
        self._family = ensure_name("a declared hypothesis's family", family)
        self._margin = (
            None
            if margin is None
            else ensure_finite("a declared equivalence margin", margin)
        )

        if self._left == self._right:
            msg = f"the hypothesis {name!r} names one arm on both sides: {left!r}"
            raise ValueError(msg)

        if self._margin is not None and self._margin <= 0.0:
            msg = (
                f"the hypothesis {name!r} declares an equivalence margin that is "
                f"not positive: {margin!r}"
            )
            raise ValueError(msg)

    @property
    def name(self) -> str:
        """Return the hypothesis name."""
        return self._name

    @property
    def left(self) -> str:
        """Return one of the two arms."""
        return self._left

    @property
    def right(self) -> str:
        """Return the other arm."""
        return self._right

    @property
    def family(self) -> str:
        """Return the multiple comparison family the hypothesis is corrected within."""
        return self._family

    @property
    def margin(self) -> float | None:
        """Return the equivalence margin declared for it, if one was."""
        return self._margin

    @property
    def pair(self) -> tuple[str, str]:
        """Return the two arms in a fixed order, which is the hypothesis's identity."""
        return min(self._left, self._right), max(self._left, self._right)

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the hypothesis as plan columns.

        Returns
        -------
        dict[str, JsonValue]
            The serialized hypothesis.
        """
        return {
            "name": self._name,
            "left": self._left,
            "right": self._right,
            "family": self._family,
            "margin": self._margin,
        }

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value declares the same hypothesis."""
        if type(other) is not DeclaredHypothesis:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the five fields the declaration is equal on."""
        return hash((DeclaredHypothesis, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this hypothesis."""
        return (
            f"DeclaredHypothesis({self._name!r}, {self._left!r}, {self._right!r}, "
            f"family={self._family!r}, margin={self._margin!r})"
        )

    def _identity(self) -> tuple[str, str, str, str, float | None]:
        """Return the five fields a declaration is equal on."""
        return (self._name, self._left, self._right, self._family, self._margin)


class Preregistration:
    """A preregistered analysis plan.

    Attributes
    ----------
    grid : Mapping[str, tuple[JsonValue, ...]]
        The declared levels for each factor.
    grid_size : int
        The number of cells in the factor crossing.
    hypotheses : tuple[DeclaredHypothesis, ...]
        The declared hypotheses.
    declared : int
        The number of declared hypotheses.
    families : frozenset[str]
        The declared multiple comparison families.

    Parameters
    ----------
    grid : Mapping[str, Sequence[JsonValue]]
        The declared levels for each factor.
    hypotheses : Sequence[DeclaredHypothesis]
        The declared hypotheses.

    Raises
    ------
    TypeError
        If a factor or hypothesis has an invalid type.
    ValueError
        If the grid or hypotheses contain invalid or duplicate entries.
    """

    __slots__ = ("_by_pair", "_grid", "_hypotheses")

    def __init__(
        self,
        grid: Mapping[str, Sequence[JsonValue]],
        hypotheses: Sequence[DeclaredHypothesis],
    ) -> None:
        self._grid: Mapping[str, tuple[JsonValue, ...]] = MappingProxyType(
            {
                ensure_name("a declared factor", factor): _levels(factor, levels)
                for factor, levels in grid.items()
            }
        )
        self._hypotheses = tuple(hypotheses)
        self._by_pair: dict[tuple[str, str], DeclaredHypothesis] = {}
        named: set[str] = set()

        for entry in self._hypotheses:
            if not isinstance(entry, DeclaredHypothesis):
                msg = f"a declared hypothesis is not one: {entry!r}"
                raise TypeError(msg)

            if entry.name in named:
                msg = f"the hypothesis {entry.name!r} is declared twice"
                raise ValueError(msg)

            held = self._by_pair.get(entry.pair)

            if held is not None:
                msg = (
                    f"the hypotheses {held.name!r} and {entry.name!r} both "
                    f"declare the comparison of {entry.pair[0]!r} with "
                    f"{entry.pair[1]!r}"
                )
                raise ValueError(msg)

            named.add(entry.name)
            self._by_pair[entry.pair] = entry

        if not self._hypotheses:
            msg = "a preregistration declares no hypotheses at all"
            raise ValueError(msg)

    @property
    def grid(self) -> Mapping[str, tuple[JsonValue, ...]]:
        """Return the declared levels of each factor, as a read-only view."""
        return self._grid

    @property
    def grid_size(self) -> int:
        """Return how many cells the declared crossing has."""
        cells = 1

        for levels in self._grid.values():
            cells *= len(levels)

        return cells

    @property
    def hypotheses(self) -> tuple[DeclaredHypothesis, ...]:
        """Return the declared hypotheses, in the order the plan lists them."""
        return self._hypotheses

    @property
    def declared(self) -> int:
        """Return how many hypotheses the plan declares."""
        return len(self._hypotheses)

    @property
    def families(self) -> frozenset[str]:
        """Return the multiple comparison families the plan declares."""
        return frozenset(entry.family for entry in self._hypotheses)

    def hypothesis(self, left: str, right: str) -> DeclaredHypothesis:
        """Return the hypothesis declared for two arms.

        Parameters
        ----------
        left : str
            One comparison arm.
        right : str
            The other comparison arm.

        Returns
        -------
        DeclaredHypothesis
            The matching declaration.

        Raises
        ------
        PreregistrationError
            If no hypothesis declares the arm pair.
        """
        held = self._by_pair.get((min(left, right), max(left, right)))

        if held is None:
            msg = (
                f"the preregistration declares no hypothesis comparing {left!r} "
                f"with {right!r}"
            )
            raise PreregistrationError(msg)

        return held

    def margin(self, left: str, right: str) -> float:
        """Return the equivalence margin declared for two arms.

        Parameters
        ----------
        left : str
            One comparison arm.
        right : str
            The other comparison arm.

        Returns
        -------
        float
            The declared positive margin.

        Raises
        ------
        PreregistrationError
            If the pair or its margin is not declared.
        """
        held = self.hypothesis(left, right)
        margin = held.margin

        if margin is None:
            msg = (
                f"the hypothesis {held.name!r} declares no equivalence margin, "
                f"so none can be read for {left!r} against {right!r}"
            )
            raise PreregistrationError(msg)

        return margin

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the grid and declared hypotheses.

        Returns
        -------
        dict[str, JsonValue]
            The serialized analysis plan.
        """
        return {
            "grid": {factor: list(levels) for factor, levels in self._grid.items()},
            "grid_size": self.grid_size,
            "hypotheses": [entry.to_spec() for entry in self._hypotheses],
        }

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value declares the same grid and hypotheses."""
        if type(other) is not Preregistration:
            return NotImplemented

        return (
            dict(self._grid) == dict(other._grid)
            and self._hypotheses == other._hypotheses
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the grid and the hypotheses."""
        return hash(
            (Preregistration, tuple(sorted(self._grid.items())), self._hypotheses)
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this plan."""
        return f"Preregistration({dict(self._grid)!r}, {list(self._hypotheses)!r})"


class PreregistrationAudit:
    """Stores plan coverage and undeclared tests.

    Attributes
    ----------
    preregistration : Preregistration
        The audited analysis plan.
    tested : tuple[str, ...]
        The tested declared hypotheses.
    untested : tuple[str, ...]
        The untested declared hypotheses.
    undeclared : tuple[tuple[str, str], ...]
        The tested arm pairs absent from the plan.
    ratio : float
        The share of declared hypotheses tested.

    Parameters
    ----------
    preregistration : Preregistration
        The audited analysis plan.
    tested : Sequence[str]
        The tested declared hypothesis names.
    undeclared : Sequence[tuple[str, str]]
        The tested arm pairs absent from the plan.

    Raises
    ------
    TypeError
        If the preregistration has an invalid type.
    ValueError
        If a tested name is undeclared or repeated.
    """

    __slots__ = ("_preregistration", "_tested", "_undeclared")

    def __init__(
        self,
        preregistration: Preregistration,
        tested: Sequence[str],
        undeclared: Sequence[tuple[str, str]],
    ) -> None:
        if not isinstance(preregistration, Preregistration):
            msg = f"an audit's preregistration is not one: {preregistration!r}"
            raise TypeError(msg)

        self._preregistration = preregistration
        self._tested = tuple(tested)
        self._undeclared = tuple(undeclared)
        declared = {entry.name for entry in preregistration.hypotheses}
        seen: set[str] = set()

        for name in self._tested:
            if name not in declared:
                msg = f"an audit counts {name!r}, which the plan does not declare"
                raise ValueError(msg)

            if name in seen:
                msg = f"an audit counts the hypothesis {name!r} twice"
                raise ValueError(msg)

            seen.add(name)

    @property
    def preregistration(self) -> Preregistration:
        """Return the report's preregistered plan."""
        return self._preregistration

    @property
    def tested(self) -> tuple[str, ...]:
        """Return the declared hypotheses a comparison tested, by name."""
        return self._tested

    @property
    def untested(self) -> tuple[str, ...]:
        """Return the declared hypotheses nothing in the report tested, by name."""
        return tuple(
            entry.name
            for entry in self._preregistration.hypotheses
            if entry.name not in set(self._tested)
        )

    @property
    def undeclared(self) -> tuple[tuple[str, str], ...]:
        """Return the pairs compared in the report that no hypothesis declared."""
        return self._undeclared

    @property
    def ratio(self) -> float:
        """Return the share of declared hypotheses tested."""
        return len(self._tested) / self._preregistration.declared

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the audit as report columns.

        Returns
        -------
        dict[str, JsonValue]
            The serialized audit.
        """
        return {
            "grid_size": self._preregistration.grid_size,
            "declared": self._preregistration.declared,
            "tested": len(self._tested),
            "ratio": self.ratio,
            "untested": list(self.untested),
            "undeclared": [[left, right] for left, right in self._undeclared],
        }

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value audits the same report the same way."""
        if type(other) is not PreregistrationAudit:
            return NotImplemented

        return (
            self._preregistration == other._preregistration
            and self._tested == other._tested
            and self._undeclared == other._undeclared
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the plan, what was tested and what was not declared."""
        return hash(
            (
                PreregistrationAudit,
                self._preregistration,
                self._tested,
                self._undeclared,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this audit."""
        return (
            f"PreregistrationAudit({self._preregistration!r}, "
            f"{list(self._tested)!r}, {list(self._undeclared)!r})"
        )


def check_preregistration(
    preregistration: Preregistration, comparisons: Iterable[Comparison]
) -> PreregistrationAudit:
    """Measure report coverage of a preregistered plan.

    Parameters
    ----------
    preregistration : Preregistration
        The declared analysis plan.
    comparisons : Iterable[Comparison]
        The completed comparisons.

    Returns
    -------
    PreregistrationAudit
        The plan coverage audit.

    Raises
    ------
    TypeError
        If the plan or a comparison has an invalid type.
    UndeclaredFamilyError
        If a comparison is corrected within a family the plan does not declare.
    PreregistrationError
        If a comparison's family differs from the one its hypothesis declares.
    """
    if not isinstance(preregistration, Preregistration):
        msg = f"a checked report's preregistration is not one: {preregistration!r}"
        raise TypeError(msg)

    declared = preregistration.families
    tested: list[str] = []
    undeclared: list[tuple[str, str]] = []

    for comparison in comparisons:
        if not isinstance(comparison, Comparison):
            msg = f"a checked comparison is not one: {comparison!r}"
            raise TypeError(msg)

        if comparison.family not in declared:
            msg = (
                f"the comparison of {comparison.left!r} with {comparison.right!r} "
                f"was corrected within {comparison.family!r}, which the "
                f"preregistration does not declare"
            )
            raise UndeclaredFamilyError(msg)

        pair = (
            min(comparison.left, comparison.right),
            max(comparison.left, comparison.right),
        )

        try:
            held = preregistration.hypothesis(comparison.left, comparison.right)
        except PreregistrationError:
            if pair not in undeclared:
                undeclared.append(pair)

            continue

        if held.family != comparison.family:
            msg = (
                f"the comparison of {comparison.left!r} with {comparison.right!r} "
                f"was corrected within {comparison.family!r} where the "
                f"preregistration declares {held.family!r}"
            )
            raise PreregistrationError(msg)

        if held.name not in tested:
            tested.append(held.name)

    return PreregistrationAudit(preregistration, tested, undeclared)


def read_preregistration(text: str) -> Preregistration:
    """Return the plan declared by preregistration JSON.

    Parameters
    ----------
    text : str
        The JSON document.

    Returns
    -------
    Preregistration
        The parsed analysis plan.

    Raises
    ------
    PreregistrationError
        If the document is invalid or cannot form a plan.
    """
    try:
        decoded = decode_preregistration(text)
    except json.JSONDecodeError as error:
        msg = f"a preregistration is not JSON: {error}"
        raise PreregistrationError(msg) from error
    except (ValidationError, TypeError) as error:
        msg = f"a preregistration does not validate: {error}"
        raise PreregistrationError(msg) from error

    try:
        return Preregistration(
            decoded.grid,
            [
                DeclaredHypothesis(
                    entry.name,
                    entry.left,
                    entry.right,
                    family=entry.family,
                    margin=entry.margin,
                )
                for entry in decoded.hypotheses
            ],
        )
    except (TypeError, ValueError) as refusal:
        msg = f"a preregistration was refused: {refusal}"
        raise PreregistrationError(msg) from refusal


def _levels(factor: str, levels: Sequence[JsonValue]) -> tuple[JsonValue, ...]:
    """Return one factor's declared levels.

    Parameters
    ----------
    factor : str
        The factor name.
    levels : Sequence[JsonValue]
        The declared levels.

    Returns
    -------
    tuple[JsonValue, ...]
        The unique levels in declaration order.

    Raises
    ------
    ValueError
        If no levels exist or a level is repeated.
    """
    held = tuple(levels)
    seen: list[JsonValue] = []

    for level in held:
        if level in seen:
            msg = f"the factor {factor!r} declares the level {level!r} twice"
            raise ValueError(msg)

        seen.append(level)

    if not held:
        msg = f"the factor {factor!r} declares no levels"
        raise ValueError(msg)

    return held
