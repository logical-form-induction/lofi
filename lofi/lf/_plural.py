"""Finite join-semilattice semantics for plural individuals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, final, runtime_checkable

from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.lf._semantics import LogicalModel, ProverSyntax, TruthOperations
    from lofi.lf.models._frame import ModelValue

__all__ = ["ATOM", "PART_OF", "SUM"]


@runtime_checkable
class _PluralModel(Protocol):
    """Expose a model's validated plural operations."""

    def join(self, left: object, right: object, /) -> ModelValue: ...

    def part_of(self, left: object, right: object, /) -> bool: ...

    def atom(self, value: object, /) -> bool: ...


class _PluralOperation(Protocol):
    """Apply one plural operation without inspecting its registered name."""

    @property
    def name(self) -> str: ...

    @property
    def arity(self) -> int: ...

    def denote(
        self,
        arguments: tuple[object, ...],
        algebra: TruthOperations,
        model: _PluralModel,
        /,
    ) -> ModelValue: ...


@final
class _SumOperation(FrozenValue):
    """The nonempty join operation."""

    _noun = "plural operation"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "sum"

    @property
    def arity(self) -> int:
        return 2

    def denote(
        self,
        arguments: tuple[object, ...],
        algebra: TruthOperations,
        model: _PluralModel,
        /,
    ) -> ModelValue:
        del algebra
        return model.join(arguments[0], arguments[1])


@final
class _PartOfOperation(FrozenValue):
    """The mereological inclusion operation."""

    _noun = "plural operation"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "part-of"

    @property
    def arity(self) -> int:
        return 2

    def denote(
        self,
        arguments: tuple[object, ...],
        algebra: TruthOperations,
        model: _PluralModel,
        /,
    ) -> ModelValue:
        return algebra.lift(value=model.part_of(arguments[0], arguments[1]))


@final
class _AtomOperation(FrozenValue):
    """The atomic-individual test."""

    _noun = "plural operation"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "atom"

    @property
    def arity(self) -> int:
        return 1

    def denote(
        self,
        arguments: tuple[object, ...],
        algebra: TruthOperations,
        model: _PluralModel,
        /,
    ) -> ModelValue:
        return algebra.lift(value=model.atom(arguments[0]))


def _plural_model(model: LogicalModel, /) -> _PluralModel:
    """Return a model with the declared plural resource."""
    if not isinstance(model, _PluralModel):
        msg = "plural semantics needs a join-semilattice model resource"
        raise TypeError(msg)

    return model


def _values(
    arguments: Sequence[object], arity: int, name: str, /
) -> tuple[object, ...]:
    """Return one logical constant's checked argument tuple."""
    values = tuple(arguments)

    if len(values) != arity:
        msg = f"{name} needs {arity} arguments, not {len(values)}"
        raise ValueError(msg)

    return values


@final
class _PluralSemantics(FrozenValue):
    """One logical operation supplied by a finite join-semilattice."""

    _noun = "logical-constant semantics"
    __slots__ = ("_operation",)

    def __init__(self, operation: _PluralOperation, /) -> None:
        object.__setattr__(self, "_operation", operation)

    @property
    def name(self) -> str:
        return self._operation.name

    @property
    def arity(self) -> int:
        return self._operation.arity

    @property
    def connective(self) -> bool:
        return False

    @property
    def quantifier(self) -> bool:
        return False

    @property
    def rules(self) -> tuple[str, ...]:
        return ()

    @property
    def declares_symbol(self) -> bool:
        return True

    @property
    def model_resource(self) -> str:
        return "join-semilattice"

    def denote(
        self,
        arguments: Sequence[object],
        algebra: TruthOperations,
        model: LogicalModel,
        /,
    ) -> ModelValue:
        values = _values(arguments, self.arity, self.name)
        plural = _plural_model(model)
        return self._operation.denote(values, algebra, plural)

    def encode(
        self,
        symbol: str,
        arguments: Sequence[str],
        /,
        *,
        dialect: ProverSyntax,
    ) -> str:
        values = _values(arguments, self.arity, self.name)

        if not dialect.higher_order:
            msg = "plural semantics is emitted with its THF semilattice axioms"
            raise ValueError(msg)

        return dialect.application(symbol, tuple(str(value) for value in values))


SUM = _PluralSemantics(_SumOperation())
"""The installed nonempty-sum operation."""

PART_OF = _PluralSemantics(_PartOfOperation())
"""The installed mereological part relation."""

ATOM = _PluralSemantics(_AtomOperation())
"""The installed atomic-individual predicate."""
