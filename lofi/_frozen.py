"""The base a frozen value inherits, and the refusal a serialized one adds."""

from typing import ClassVar, Never, Self, override

from lofi._checks import article, refuse_reassignment

__all__ = ["FrozenValue", "Unpicklable"]


class FrozenValue:
    """Base for a slotted value whose fields are set once and never change.

    A subclass declares ``_noun``, names its fields in ``__slots__`` and fills
    them through ``object.__setattr__`` in ``__init__``. Afterwards the fields
    refuse reassignment and deletion, and both copies of the value are the value
    itself, since nothing observable distinguishes them.
    """

    __slots__ = ()

    _noun: ClassVar[str]
    """The value kind the refusal messages name."""

    def __init_subclass__(cls) -> None:
        """Refuse a subclass with no noun to put in its refusal messages.

        Raises
        ------
        TypeError
            If the subclass neither declares ``_noun`` nor inherits one.
        """
        super().__init_subclass__()

        if not getattr(cls, "_noun", ""):
            msg = f"{cls.__name__} declares no noun for its refusal messages"
            raise TypeError(msg)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment(self._noun)

    @override
    def __delattr__(self, name: str) -> Never:
        """Raise AttributeError on field deletion."""
        del name

        refuse_reassignment(self._noun)

    def __copy__(self) -> Self:
        """Return this immutable value."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable value."""
        del memo

        return self


class Unpicklable:
    """Mixin for a frozen value holding identities that serialization loses.

    A subclass declares ``_because``, the clause naming what the value holds,
    and inherits :class:`FrozenValue` for the noun the clause is about.
    """

    __slots__ = ()

    _noun: ClassVar[str]
    """The value kind the refusal message names."""

    _because: ClassVar[str]
    """What the value holds that survives no round trip through a pickle."""

    def __init_subclass__(cls) -> None:
        """Refuse a subclass that gives no cause for refusing serialization.

        Raises
        ------
        TypeError
            If the subclass neither declares ``_because`` nor inherits one.
        """
        super().__init_subclass__()

        if not getattr(cls, "_because", ""):
            msg = f"{cls.__name__} declares no cause for refusing serialization"
            raise TypeError(msg)

    @override
    def __reduce__(self) -> Never:
        """Raise TypeError on serialization.

        Raises
        ------
        TypeError
            Always.
        """
        msg = (
            f"{article(self._noun)} {self._noun} {self._because} and cannot be pickled"
        )

        raise TypeError(msg)
