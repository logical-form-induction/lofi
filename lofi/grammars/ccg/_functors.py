"""The binary functor view of a declared pair of category constructors."""

from types import MappingProxyType
from typing import TYPE_CHECKING, final

from lofi._frozen import FrozenValue, Unpicklable

if TYPE_CHECKING:
    from lofi.terms import OpName, Path, Sort, Term, TermStore

__all__ = ["FunctorView"]

_BINARY_ARITY = 2


@final
class FunctorView(Unpicklable, FrozenValue):
    """The binary functor view of a forward and a backward constructor.

    A functor category is built by one of the two constructors over a result
    and an argument, which sit at the declared child slots of every functor.
    The forward constructor takes its argument on the right and the backward
    one on the left. Naming one constructor for both makes it self-opposite:
    a functor it builds takes its argument on either side, which is the
    undirected arrow.

    Parameters
    ----------
    store : TermStore
        The category store the constructors build in.
    forward : OpName
        The constructor whose functors take their argument on the right.
    backward : OpName
        The constructor whose functors take their argument on the left.
    result : int
        The child slot of a functor's result.
    argument : int
        The child slot of a functor's argument.

    Raises
    ------
    ValueError
        If a constructor is not a payload-free binary operator over one
        category sort, the two construct different sorts, or the slots are
        not the two child positions.
    """

    _noun = "functor view"
    _because = "retains live term identities"

    __slots__ = (
        "_argument",
        "_backward",
        "_forward",
        "_operators",
        "_result",
        "_store",
    )

    def __init__(
        self,
        store: TermStore,
        forward: OpName,
        backward: OpName,
        *,
        result: int,
        argument: int,
    ) -> None:
        signature = store.signature
        operators = {name: signature.by_name(name) for name in (forward, backward)}

        for operator in operators.values():
            if (
                operator.arity != _BINARY_ARITY
                or operator.arg_sorts != (operator.sort, operator.sort)
                or operator.has_payload
            ):
                msg = f"constructor {operator.name!r} is not a binary endofunctor"
                raise ValueError(msg)

        if operators[forward].sort != operators[backward].sort:
            msg = "the forward and backward constructors build different sorts"
            raise ValueError(msg)

        if {result, argument} != {0, 1}:
            msg = (
                f"a functor's result slot {result} and argument slot {argument} "
                "are not its two child positions"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_forward", forward)
        object.__setattr__(self, "_backward", backward)
        object.__setattr__(self, "_operators", MappingProxyType(operators))
        object.__setattr__(self, "_result", result)
        object.__setattr__(self, "_argument", argument)

    @property
    def slashes(self) -> frozenset[OpName]:
        """Return the functor constructors."""
        return frozenset(self._operators)

    @property
    def forward_slash(self) -> OpName:
        """Return the constructor that takes its argument on the right."""
        return self._forward

    @property
    def backward_slash(self) -> OpName:
        """Return the constructor that takes its argument on the left."""
        return self._backward

    @property
    def result_path(self) -> Path:
        """Return the one-child path to a functor's result."""
        return (self._result,)

    @property
    def argument_path(self) -> Path:
        """Return the one-child path to a functor's argument."""
        return (self._argument,)

    @property
    def category_sort(self) -> Sort:
        """Return the sort both constructors build."""
        return self._operators[self._forward].sort

    def is_functor(self, term: Term, /) -> bool:
        """Return whether a category is headed by a functor constructor."""
        return term.op.name in self._operators

    def slash(self, term: Term, /) -> OpName:
        """Return a functor category's constructor."""
        return term.op.name

    def make(
        self,
        slash: OpName,
        result: Term,
        argument: Term,
        /,
    ) -> Term:
        """Return the functor a constructor builds over a result and an argument.

        Raises
        ------
        KeyError
            If the constructor is not one of the view's two.
        """
        children = (result, argument) if self._result == 0 else (argument, result)

        return self._store.op(self._operators[slash], *children)

    def opposite(self, slash: OpName, /) -> OpName:
        """Return the constructor with the opposite orientation.

        Raises
        ------
        KeyError
            If the constructor is not one of the view's two.
        """
        if slash == self._forward:
            return self._backward

        if slash == self._backward:
            return self._forward

        raise KeyError(slash)
