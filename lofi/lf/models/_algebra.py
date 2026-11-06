"""Declared fuzzy truth algebras for finite-model evaluation."""

from typing import TYPE_CHECKING, Final, Protocol, final, runtime_checkable

from lofi._checks import ensure_finite
from lofi._frozen import FrozenValue
from lofi.registry import Registry, RegistryError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "CRISP",
    "EXTREME",
    "FLOAT_ARITHMETIC",
    "GOEDEL",
    "LUKASIEWICZ",
    "MARGIN",
    "MATERIAL",
    "PRODUCT",
    "RESIDUUM",
    "TNORM_QUANTIFIER",
    "Arithmetic",
    "FuzzyAlgebra",
    "FuzzyEquality",
    "FuzzyImplication",
    "FuzzyLogicFactory",
    "FuzzyQuantifier",
    "TNorm",
    "TruthAlgebra",
    "fuzzy_algebra",
]


def _unit(value: float, field: str = "a truth value") -> float:
    """Return a finite value in the unit interval."""
    held = ensure_finite(field, value)

    if not 0.0 <= held <= 1.0:
        msg = f"{field} lies in the unit interval, not {value!r}"
        raise ValueError(msg)

    return held


@runtime_checkable
class Arithmetic[T](Protocol):
    """Supply the pointwise operations a t-norm is written in for one carrier."""

    def constant(self, value: float, like: T, /) -> T: ...

    def add(self, left: T, right: T, /) -> T: ...

    def subtract(self, left: T, right: T, /) -> T: ...

    def multiply(self, left: T, right: T, /) -> T: ...

    def divide(self, left: T, right: T, /) -> T: ...

    def minimum(self, left: T, right: T, /) -> T: ...

    def maximum(self, left: T, right: T, /) -> T: ...

    def where_at_most(self, left: T, right: T, then: T, otherwise: T, /) -> T: ...


@final
class _FloatArithmetic(FrozenValue):
    """Pointwise operations on Python floats."""

    _noun = "float arithmetic"
    __slots__ = ()

    def constant(self, value: float, like: float, /) -> float:
        del like

        return value

    def add(self, left: float, right: float, /) -> float:
        return left + right

    def subtract(self, left: float, right: float, /) -> float:
        return left - right

    def multiply(self, left: float, right: float, /) -> float:
        return left * right

    def divide(self, left: float, right: float, /) -> float:
        return left / right

    def minimum(self, left: float, right: float, /) -> float:
        return min(left, right)

    def maximum(self, left: float, right: float, /) -> float:
        return max(left, right)

    def where_at_most(
        self, left: float, right: float, then: float, otherwise: float, /
    ) -> float:
        return then if left <= right else otherwise


FLOAT_ARITHMETIC: Arithmetic[float] = _FloatArithmetic()
"""The arithmetic the scalar truth operations evaluate in."""


@runtime_checkable
class TNorm(Protocol):
    """Define a t-norm, its dual t-conorm and its residuum over any arithmetic."""

    @property
    def name(self) -> str: ...

    def conjunction[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T: ...

    def disjunction[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T: ...

    def residuum[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T: ...


@runtime_checkable
class FuzzyImplication(Protocol):
    """Interpret a conditional over one t-norm in any arithmetic."""

    @property
    def name(self) -> str: ...

    def imply[T](
        self, tnorm: TNorm, arithmetic: Arithmetic[T], left: T, right: T, /
    ) -> T: ...


@runtime_checkable
class FuzzyQuantifier(Protocol):
    """Aggregate a nonempty finite quantifier over one t-norm in any arithmetic."""

    @property
    def name(self) -> str: ...

    def every[T](
        self, tnorm: TNorm, arithmetic: Arithmetic[T], values: Sequence[T], /
    ) -> T: ...

    def some[T](
        self, tnorm: TNorm, arithmetic: Arithmetic[T], values: Sequence[T], /
    ) -> T: ...


@runtime_checkable
class FuzzyEquality(Protocol):
    """Compare two model values to a truth degree."""

    @property
    def name(self) -> str: ...

    def equal(self, left: object, right: object, margin: float, /) -> float: ...


@runtime_checkable
class TruthAlgebra(Protocol):
    """Interpret Boolean functions, quantifiers and equality over unit values."""

    @property
    def name(self) -> str: ...

    @property
    def margin(self) -> float: ...

    def lift(self, *, value: bool) -> float: ...

    def negate(self, value: float, /) -> float: ...

    def conjunction(self, left: float, right: float, /) -> float: ...

    def disjunction(self, left: float, right: float, /) -> float: ...

    def conditional(self, left: float, right: float, /) -> float: ...

    def conjoin[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T: ...

    def disjoin[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T: ...

    def imply[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T: ...

    def every[T](self, arithmetic: Arithmetic[T], values: Sequence[T], /) -> T: ...

    def some[T](self, arithmetic: Arithmetic[T], values: Sequence[T], /) -> T: ...

    def connective(
        self, table: tuple[bool, ...], arguments: Sequence[float], /
    ) -> float: ...

    def universal(self, values: Sequence[float], /) -> float: ...

    def existential(self, values: Sequence[float], /) -> float: ...

    def equal(self, left: object, right: object, /) -> float: ...


@final
class FuzzyAlgebra(FrozenValue):
    """One t-norm algebra with separately declared logical operations."""

    _noun = "truth algebra"
    __slots__ = ("_equality", "_implication", "_margin", "_quantifier", "_tnorm")

    def __init__(
        self,
        tnorm: TNorm,
        /,
        *,
        implication: FuzzyImplication,
        quantifier: FuzzyQuantifier,
        equality: FuzzyEquality,
        margin: float,
    ) -> None:
        held_margin = ensure_finite("an equality margin", margin)

        if held_margin < 0.0:
            msg = f"an equality margin is nonnegative, not {margin!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_tnorm", tnorm)
        object.__setattr__(self, "_implication", implication)
        object.__setattr__(self, "_quantifier", quantifier)
        object.__setattr__(self, "_equality", equality)
        object.__setattr__(self, "_margin", held_margin)

    @property
    def name(self) -> str:
        """Return the registered t-norm name."""
        return self._tnorm.name

    @property
    def margin(self) -> float:
        """Return the declared equality margin."""
        return self._margin

    def lift(self, *, value: bool) -> float:
        """Embed a crisp truth value."""
        if not isinstance(value, bool):
            msg = f"a crisp truth value is bool, not {value!r}"
            raise TypeError(msg)

        return float(value)

    def negate(self, value: float, /) -> float:
        """Return standard fuzzy negation."""
        return 1.0 - _unit(value)

    def conjunction(self, left: float, right: float, /) -> float:
        """Apply the declared t-norm."""
        return self.conjoin(FLOAT_ARITHMETIC, _unit(left), _unit(right))

    def disjunction(self, left: float, right: float, /) -> float:
        """Apply the dual t-conorm."""
        return self.disjoin(FLOAT_ARITHMETIC, _unit(left), _unit(right))

    def conditional(self, left: float, right: float, /) -> float:
        """Apply the declared material or residuated implication."""
        return self.imply(FLOAT_ARITHMETIC, _unit(left), _unit(right))

    def conjoin[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        """Apply the declared t-norm in a supplied arithmetic."""
        return self._tnorm.conjunction(arithmetic, left, right)

    def disjoin[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        """Apply the dual t-conorm in a supplied arithmetic."""
        return self._tnorm.disjunction(arithmetic, left, right)

    def imply[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        """Apply the declared implication in a supplied arithmetic."""
        return self._implication.imply(self._tnorm, arithmetic, left, right)

    def every[T](self, arithmetic: Arithmetic[T], values: Sequence[T], /) -> T:
        """Aggregate a nonempty universal quantifier in a supplied arithmetic."""
        return self._quantifier.every(self._tnorm, arithmetic, values)

    def some[T](self, arithmetic: Arithmetic[T], values: Sequence[T], /) -> T:
        """Aggregate a nonempty existential quantifier in a supplied arithmetic."""
        return self._quantifier.some(self._tnorm, arithmetic, values)

    def connective(
        self, table: tuple[bool, ...], arguments: Sequence[float], /
    ) -> float:
        """Lift a declared finite Boolean table into this algebra."""
        values = tuple(_unit(value) for value in arguments)

        if len(table) != 2 ** len(values):
            msg = f"a {len(values)}-ary connective needs {2 ** len(values)} rows"
            raise ValueError(msg)

        if len(values) == 1 and table == (True, False):
            return self.negate(values[0])

        if len(values) == 2:
            left, right = values

            if table == (False, False, False, True):
                return self.conjunction(left, right)

            if table == (False, True, True, True):
                return self.disjunction(left, right)

            if table == (True, True, False, True):
                return self.conditional(left, right)

            if table == (True, False, False, True):
                return self.conjunction(
                    self.conditional(left, right),
                    self.conditional(right, left),
                )

        total = 0.0

        for row, admitted in enumerate(table):
            if not admitted:
                continue

            degree = 1.0

            for position, value in enumerate(values):
                bit = bool(row & (1 << (len(values) - position - 1)))
                degree = self.conjunction(degree, value if bit else self.negate(value))

            total = self.disjunction(total, degree)

        return total

    def universal(self, values: Sequence[float], /) -> float:
        """Aggregate a finite universal quantifier."""
        held = tuple(_unit(value) for value in values)

        if not held:
            return 1.0

        return self.every(FLOAT_ARITHMETIC, held)

    def existential(self, values: Sequence[float], /) -> float:
        """Aggregate a finite existential quantifier."""
        held = tuple(_unit(value) for value in values)

        if not held:
            return 0.0

        return self.some(FLOAT_ARITHMETIC, held)

    def equal(self, left: object, right: object, /) -> float:
        """Compare model values under the declared equality treatment."""
        return self._equality.equal(left, right, self._margin)


@runtime_checkable
class FuzzyLogicFactory(Protocol):
    """Build one registered t-norm with declared operation choices."""

    def build(
        self,
        *,
        implication: FuzzyImplication,
        quantifier: FuzzyQuantifier,
        equality: FuzzyEquality,
        margin: float,
    ) -> TruthAlgebra: ...


@final
class _Product(FrozenValue):
    """The product t-norm, the probabilistic sum and the Goguen residuum."""

    _noun = "product t-norm"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "product"

    def conjunction[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        return arithmetic.multiply(left, right)

    def disjunction[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        return arithmetic.subtract(
            arithmetic.add(left, right), arithmetic.multiply(left, right)
        )

    def residuum[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        one = arithmetic.constant(1.0, left)
        denominator = arithmetic.where_at_most(left, right, one, left)
        return arithmetic.where_at_most(
            left, right, one, arithmetic.divide(right, denominator)
        )


@final
class _Goedel(FrozenValue):
    """The minimum t-norm, the maximum t-conorm and the Goedel residuum."""

    _noun = "Goedel t-norm"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "goedel"

    def conjunction[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        return arithmetic.minimum(left, right)

    def disjunction[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        return arithmetic.maximum(left, right)

    def residuum[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        return arithmetic.where_at_most(
            left, right, arithmetic.constant(1.0, left), right
        )


@final
class _Lukasiewicz(FrozenValue):
    """The Lukasiewicz t-norm, its bounded sum and its residuum."""

    _noun = "Lukasiewicz t-norm"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "lukasiewicz"

    def conjunction[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        one = arithmetic.constant(1.0, left)
        return arithmetic.maximum(
            arithmetic.constant(0.0, left),
            arithmetic.subtract(arithmetic.add(left, right), one),
        )

    def disjunction[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        return arithmetic.minimum(
            arithmetic.constant(1.0, left), arithmetic.add(left, right)
        )

    def residuum[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        one = arithmetic.constant(1.0, left)
        return arithmetic.minimum(
            one, arithmetic.add(arithmetic.subtract(one, left), right)
        )


@final
class _Factory(FrozenValue):
    _noun = "fuzzy-logic factory"
    __slots__ = ("_tnorm",)

    def __init__(self, tnorm: TNorm) -> None:
        object.__setattr__(self, "_tnorm", tnorm)

    def build(
        self,
        *,
        implication: FuzzyImplication,
        quantifier: FuzzyQuantifier,
        equality: FuzzyEquality,
        margin: float,
    ) -> TruthAlgebra:
        return FuzzyAlgebra(
            self._tnorm,
            implication=implication,
            quantifier=quantifier,
            equality=equality,
            margin=margin,
        )


PRODUCT: FuzzyLogicFactory = _Factory(_Product())
GOEDEL: FuzzyLogicFactory = _Factory(_Goedel())
LUKASIEWICZ: FuzzyLogicFactory = _Factory(_Lukasiewicz())


@final
class _Residuum(FrozenValue):
    """The residuated implication of the declared t-norm."""

    _noun = "residuated implication"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "residuum"

    def imply[T](
        self, tnorm: TNorm, arithmetic: Arithmetic[T], left: T, right: T, /
    ) -> T:
        return tnorm.residuum(arithmetic, left, right)


@final
class _Material(FrozenValue):
    """The material implication, the t-conorm of the negated antecedent."""

    _noun = "material implication"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "material"

    def imply[T](
        self, tnorm: TNorm, arithmetic: Arithmetic[T], left: T, right: T, /
    ) -> T:
        negated = arithmetic.subtract(arithmetic.constant(1.0, left), left)
        return tnorm.disjunction(arithmetic, negated, right)


@final
class _TNormQuantifier(FrozenValue):
    """Fold the t-norm for every and its t-conorm for some."""

    _noun = "t-norm quantifier"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "tnorm"

    def every[T](
        self, tnorm: TNorm, arithmetic: Arithmetic[T], values: Sequence[T], /
    ) -> T:
        result = arithmetic.constant(1.0, values[0])

        for value in values:
            result = tnorm.conjunction(arithmetic, result, value)

        return result

    def some[T](
        self, tnorm: TNorm, arithmetic: Arithmetic[T], values: Sequence[T], /
    ) -> T:
        result = arithmetic.constant(0.0, values[0])

        for value in values:
            result = tnorm.disjunction(arithmetic, result, value)

        return result


@final
class _Extreme(FrozenValue):
    """Take the minimum for every and the maximum for some."""

    _noun = "extreme quantifier"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "extreme"

    def every[T](
        self, tnorm: TNorm, arithmetic: Arithmetic[T], values: Sequence[T], /
    ) -> T:
        del tnorm
        result = values[0]

        for value in values[1:]:
            result = arithmetic.minimum(result, value)

        return result

    def some[T](
        self, tnorm: TNorm, arithmetic: Arithmetic[T], values: Sequence[T], /
    ) -> T:
        del tnorm
        result = values[0]

        for value in values[1:]:
            result = arithmetic.maximum(result, value)

        return result


@final
class _Crisp(FrozenValue):
    """Compare model values by identity."""

    _noun = "crisp equality"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "crisp"

    def equal(self, left: object, right: object, margin: float, /) -> float:
        del margin

        return float(left == right)


@final
class _Margin(FrozenValue):
    """Equate two floats within the declared margin and anything else crisply."""

    _noun = "margin equality"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "margin"

    def equal(self, left: object, right: object, margin: float, /) -> float:
        if isinstance(left, float) and isinstance(right, float):
            return float(abs(left - right) <= margin)

        return float(left == right)


RESIDUUM: Final[FuzzyImplication] = _Residuum()
"""The installed residuated implication."""

MATERIAL: Final[FuzzyImplication] = _Material()
"""The installed material implication."""

TNORM_QUANTIFIER: Final[FuzzyQuantifier] = _TNormQuantifier()
"""The installed quantifier that folds the t-norm and its t-conorm."""

EXTREME: Final[FuzzyQuantifier] = _Extreme()
"""The installed quantifier that takes the minimum and the maximum."""

CRISP: Final[FuzzyEquality] = _Crisp()
"""The installed equality that compares values by identity."""

MARGIN: Final[FuzzyEquality] = _Margin()
"""The installed equality that equates floats within the declared margin."""


def _installed[P](family: str, tag: str, protocol: type[P], /) -> P:
    """Return the entry of one fuzzy family, refused unless it implements it."""
    member = Registry[object](family).get(tag)

    if not isinstance(member, protocol):
        msg = f"the {family} entry {tag!r} names {member!r}, not a {protocol.__name__}"
        raise RegistryError(msg, family=family)

    return member


def fuzzy_algebra(
    tnorm: str,
    implication: str,
    quantifier: str,
    equality: str,
    margin: float,
    /,
) -> TruthAlgebra:
    """Return the truth algebra the installed fuzzy families name.

    Parameters
    ----------
    tnorm : str
        The ``lofi.fuzzy_logic`` entry.
    implication : str
        The ``lofi.fuzzy_implication`` entry.
    quantifier : str
        The ``lofi.fuzzy_quantifier`` entry.
    equality : str
        The ``lofi.fuzzy_equality`` entry.
    margin : float
        The equality margin.

    Returns
    -------
    TruthAlgebra
        The algebra combining the four installed operations.

    Raises
    ------
    RegistryError
        If an entry is missing or names something other than its family's
        operation.
    """
    factory = _installed("fuzzy_logic", tnorm, FuzzyLogicFactory)

    return factory.build(
        implication=_installed("fuzzy_implication", implication, FuzzyImplication),
        quantifier=_installed("fuzzy_quantifier", quantifier, FuzzyQuantifier),
        equality=_installed("fuzzy_equality", equality, FuzzyEquality),
        margin=margin,
    )
