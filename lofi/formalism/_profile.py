"""Deduction profiles and consumer bounds."""

from typing import (
    TYPE_CHECKING,
    Literal,
    Never,
    Protocol,
    Self,
    final,
    override,
    runtime_checkable,
)

from lofi._checks import ensure_name, ensure_non_negative

if TYPE_CHECKING:
    from lofi.judge import DeductionRequirements

__all__ = [
    "BoundedConsumer",
    "CategorySet",
    "DeductionProfile",
    "ProfileBound",
    "ProfileUnsupportedError",
]


type CategorySet = Literal["finite", "infinite", "unknown"]
"""Classifies a reachable category set."""

_CATEGORY_SETS = frozenset(("finite", "infinite", "unknown"))


def _positive(field: str, value: int) -> int:
    """Return an integer of at least one.

    Parameters
    ----------
    field : str
        Field name used in error messages.
    value : int
        Integer to validate.

    Returns
    -------
    int
        The validated integer.

    Raises
    ------
    TypeError
        If ``value`` is not an integer or is a boolean.
    ValueError
        If ``value`` is less than one.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{field} is not an integer: {value!r}"
        raise TypeError(msg)

    if value < 1:
        msg = f"{field} is below one: {value!r}"
        raise ValueError(msg)

    return value


def _arities(field: str, values: frozenset[int]) -> frozenset[int]:
    """Return validated nonnegative rule arities.

    Parameters
    ----------
    field : str
        Field name used in error messages.
    values : frozenset[int]
        Arities to validate.

    Returns
    -------
    frozenset[int]
        The validated arities.

    Raises
    ------
    TypeError
        If an arity is not an integer or is a boolean.
    ValueError
        If an arity is negative.
    """
    return frozenset(ensure_non_negative(field, value) for value in values)


class ProfileUnsupportedError(ValueError):
    """Reports an unsupported deduction profile."""


@final
class DeductionProfile:
    """A formalism's deduction requirements.

    Parameters
    ----------
    fanout : int
        Number of contiguous input regions covered by one item.
    ranks : frozenset[int]
        Rule arities used by the formalism.
    unary_closure : bool
        Whether each chart cell requires unary closure.
    category_set : CategorySet
        Classification of the reachable category set.
    lexicalized : bool
        Whether every derivation is anchored by a lexical axiom.

    Raises
    ------
    TypeError
        If a field has the wrong type.
    ValueError
        If a numeric field is invalid, the category-set label is unknown, or
        unary closure is requested without unary rules.
    """

    __slots__ = (
        "_category_set",
        "_fanout",
        "_lexicalized",
        "_ranks",
        "_unary_closure",
    )

    def __init__(
        self,
        fanout: int,
        ranks: frozenset[int],
        *,
        unary_closure: bool,
        category_set: CategorySet,
        lexicalized: bool,
    ) -> None:
        if not isinstance(unary_closure, bool):
            msg = f"unary closure is not a boolean: {unary_closure!r}"
            raise TypeError(msg)

        if category_set not in _CATEGORY_SETS:
            msg = f"category set is not finite, infinite, or unknown: {category_set!r}"
            raise ValueError(msg)

        if not isinstance(lexicalized, bool):
            msg = f"lexicalization is not a boolean: {lexicalized!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_fanout", _positive("a profile's fan-out", fanout))
        object.__setattr__(
            self,
            "_ranks",
            _arities("a profile's rule arity", ranks),
        )
        object.__setattr__(self, "_unary_closure", unary_closure)
        object.__setattr__(self, "_category_set", category_set)
        object.__setattr__(self, "_lexicalized", lexicalized)

        if unary_closure and 1 not in self._ranks:
            msg = (
                "a profile declares a unary closure over arities "
                f"{sorted(self._ranks)}, which hold no unary rule to close over"
            )
            raise ValueError(msg)

    @classmethod
    def derive(
        cls,
        requirements: DeductionRequirements,
        *,
        fanout: int,
        lexicalized: bool,
    ) -> DeductionProfile:
        """Build a profile from judge and formalism declarations.

        Parameters
        ----------
        requirements : DeductionRequirements
            Requirements derived from the rule inventory.
        fanout : int
            Number of contiguous input regions covered by one item.
        lexicalized : bool
            Whether every derivation is anchored by a lexical axiom.

        Returns
        -------
        DeductionProfile
            The combined deduction profile.
        """
        return cls(
            fanout,
            requirements.ranks,
            unary_closure=requirements.unary_closure,
            category_set=requirements.category_set,
            lexicalized=lexicalized,
        )

    def __copy__(self) -> Self:
        """Return this immutable value."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable value."""
        del memo

        return self

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a deduction profile is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a deduction profile is immutable"

        raise AttributeError(msg)

    @property
    def fanout(self) -> int:
        """Return how many contiguous regions of the input one item covers."""
        return self._fanout

    @property
    def ranks(self) -> frozenset[int]:
        """Return the rule arities the rule set uses."""
        return self._ranks

    @property
    def unary_closure(self) -> bool:
        """Return whether the engine must run unary-closure rounds in each cell."""
        return self._unary_closure

    @property
    def category_set(self) -> CategorySet:
        """Return whether the reachable category set is finite, infinite or unknown."""
        return self._category_set

    @property
    def lexicalized(self) -> bool:
        """Return whether every derivation is anchored by a lexical axiom."""
        return self._lexicalized

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a profile with the same five fields."""
        if type(other) is not DeductionProfile:
            return NotImplemented

        return (
            self._fanout == other._fanout
            and self._ranks == other._ranks
            and self._unary_closure == other._unary_closure
            and self._category_set == other._category_set
            and self._lexicalized == other._lexicalized
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the five fields."""
        return hash(
            (
                DeductionProfile,
                self._fanout,
                self._ranks,
                self._unary_closure,
                self._category_set,
                self._lexicalized,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this profile."""
        return (
            f"DeductionProfile({self._fanout!r}, {self._ranks!r}, "
            f"unary_closure={self._unary_closure!r}, "
            f"category_set={self._category_set!r}, "
            f"lexicalized={self._lexicalized!r})"
        )


@final
class ProfileBound:
    """The deduction profiles a consumer supports.

    Parameters
    ----------
    consumer : str
        Consumer name used in error messages.
    max_fanout : int
        Largest fan-out accepted by the consumer.
    ranks : frozenset[int]
        Rule arities accepted by the consumer.
    extension_point : str | None
        Required extension for larger fan-out, if one is available.

    Raises
    ------
    TypeError
        If a field has the wrong type.
    ValueError
        If a name is empty or a numeric field is invalid.
    """

    __slots__ = ("_consumer", "_extension_point", "_max_fanout", "_ranks")

    def __init__(
        self,
        consumer: str,
        max_fanout: int,
        ranks: frozenset[int],
        extension_point: str | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "_consumer",
            ensure_name("a bound's consumer", consumer),
        )
        object.__setattr__(
            self,
            "_max_fanout",
            _positive("a bound's fan-out", max_fanout),
        )
        object.__setattr__(
            self,
            "_ranks",
            _arities("a bound's rule arity", ranks),
        )
        object.__setattr__(
            self,
            "_extension_point",
            None
            if extension_point is None
            else ensure_name("a bound's extension point", extension_point),
        )

    def __copy__(self) -> Self:
        """Return this immutable value."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable value."""
        del memo

        return self

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a profile bound is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a profile bound is immutable"

        raise AttributeError(msg)

    @property
    def consumer(self) -> str:
        """Return the consumer's name, reported in a refusal."""
        return self._consumer

    @property
    def max_fanout(self) -> int:
        """Return the largest fan-out this consumer serves."""
        return self._max_fanout

    @property
    def ranks(self) -> frozenset[int]:
        """Return the rule arities this consumer serves."""
        return self._ranks

    @property
    def extension_point(self) -> str | None:
        """Return what would have to exist for a larger fan-out to be served."""
        return self._extension_point

    def check(self, profile: DeductionProfile, formalism_name: str) -> None:
        """Raise for an unsupported deduction profile.

        Parameters
        ----------
        profile : DeductionProfile
            Profile to validate against this bound.
        formalism_name : str
            Formalism name used in error messages.

        Raises
        ------
        ProfileUnsupportedError
            If the profile exceeds the fan-out or rule-arity bound.
        """
        if profile.fanout > self._max_fanout:
            msg = (
                f"{self._consumer} serves fan-out at most {self._max_fanout}: "
                f"formalism {formalism_name!r} declares fan-out {profile.fanout}."
            )

            if self._extension_point is not None:
                msg = f"{msg} Extension point: {self._extension_point}."

            raise ProfileUnsupportedError(msg)

        self.check_ranks(profile.ranks, formalism_name)

    def check_ranks(
        self,
        ranks: frozenset[int],
        formalism_name: str,
        /,
    ) -> None:
        """Raise when a rule inventory exceeds this consumer's arity bound.

        Parameters
        ----------
        ranks : frozenset[int]
            Rule arities used by the inventory.
        formalism_name : str
            Inventory or formalism name used in error messages.

        Raises
        ------
        ProfileUnsupportedError
            If an arity is not served by the consumer.
        """
        unserved = _arities("a profile's rule arity", ranks) - self._ranks

        if unserved:
            msg = (
                f"{self._consumer} serves rule arities {sorted(self._ranks)}: "
                f"formalism {formalism_name!r} uses {sorted(unserved)}."
            )
            raise ProfileUnsupportedError(msg)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a bound with the same four fields."""
        if type(other) is not ProfileBound:
            return NotImplemented

        return (
            self._consumer == other._consumer
            and self._max_fanout == other._max_fanout
            and self._ranks == other._ranks
            and self._extension_point == other._extension_point
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the four fields."""
        return hash(
            (
                ProfileBound,
                self._consumer,
                self._max_fanout,
                self._ranks,
                self._extension_point,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this bound."""
        return (
            f"ProfileBound({self._consumer!r}, {self._max_fanout!r}, "
            f"{self._ranks!r}, {self._extension_point!r})"
        )


@runtime_checkable
class BoundedConsumer(Protocol):
    """A declaration of a consumer that serves only some deduction profiles."""

    def profile_bound(self) -> ProfileBound:
        """Return the fan-out and rule arities the declared consumer serves."""
        ...
