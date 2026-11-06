"""Native execution of declared term rules."""

from typing import TYPE_CHECKING, Never, final, override

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator, Sequence

    from lofi._native import RuleKernel, Term

__all__ = ["TermRuleKernel", "WitnessRow"]


type WitnessRow = tuple[int, int, str, tuple[str, ...]]
"""Principal slot, container slot, principal constructor, carried spine."""


@final
class TermRuleKernel:
    """One compiled implementation of a declared term rule.

    A kernel comes from the native grammar compiler: each compiled rule
    recipe builds its own kernel, and this wrapper is the one Python face of
    it. The forward, checking and residual directions all run the native
    relation, so they agree on which steps a declaration licenses.

    Parameters
    ----------
    native : RuleKernel
        The kernel a compiled rule recipe built.
    """

    __slots__ = ("_native",)

    def __init__(self, native: RuleKernel, /) -> None:
        object.__setattr__(self, "_native", native)

    @property
    def native(self) -> RuleKernel:
        """Return the native chart boundary."""
        return self._native

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        """Enumerate conclusions for one exact premise tuple."""
        yield from self._native.apply(inputs)

    def admits(self, inputs: tuple[Term, ...], output: Term, /) -> bool:
        """Return whether the relation holds of the premises and conclusion."""
        return self._native.admits(inputs, output)

    def residuate(
        self, output: Term, known: Sequence[Term | None], /
    ) -> tuple[Term, ...]:
        """Return the fillers of the one missing premise, marked ``None``."""
        return tuple(self._native.residuate(output, tuple(known)))

    def join_keys(self, term: Term, slot: int, /) -> tuple[Hashable, ...]:
        """Return the exact join keys one premise offers in one slot."""
        return tuple(self._native.join_keys(term, slot))

    def witnesses(
        self, inputs: tuple[Term, ...], output: Term, /
    ) -> tuple[WitnessRow, ...]:
        """Return the orientation and carried spine of each composition witness."""
        return tuple(
            (principal, container, slash, tuple(spine))
            for principal, container, slash, spine in self._native.witnesses(
                inputs, output
            )
        )

    def applied_order(self, inputs: tuple[Term, ...], output: Term, /) -> int | None:
        """Return the composition order of one step, if the relation draws it."""
        return self._native.applied_order(inputs, output)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a term-rule kernel is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a term-rule kernel is immutable"
        raise AttributeError(msg)

    @override
    def __repr__(self) -> str:
        """Return the native kernel representation."""
        return repr(self._native)
