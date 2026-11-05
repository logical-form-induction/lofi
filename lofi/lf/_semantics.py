"""Open semantics plugins for postulated logical constants."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, final, runtime_checkable

from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.lf.models._frame import ModelValue

__all__ = [
    "MOST",
    "SMT_SYNTAX",
    "THF_SYNTAX",
    "TPTP_SYNTAX",
    "FinitePredicate",
    "LogicalConstantSemantics",
    "LogicalModel",
    "ProverSyntax",
    "TruthOperations",
]


@runtime_checkable
class ProverSyntax(Protocol):
    """How one prover dialect writes a symbol applied to its arguments."""

    @property
    def name(self) -> str:
        """Return the dialect tag."""
        ...

    @property
    def higher_order(self) -> bool:
        """Return whether the dialect writes higher-order terms and binders."""
        ...

    def application(self, symbol: str, arguments: Sequence[str], /) -> str:
        """Return the symbol applied to the written arguments."""
        ...


@final
class _ThfSyntax(FrozenValue):
    """Curried THF application."""

    _noun = "THF syntax"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "thf"

    @property
    def higher_order(self) -> bool:
        return True

    def application(self, symbol: str, arguments: Sequence[str], /) -> str:
        return f"({' @ '.join((symbol, *arguments))})"


@final
class _SmtSyntax(FrozenValue):
    """Parenthesized SMT-LIB application."""

    _noun = "SMT-LIB syntax"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "smt"

    @property
    def higher_order(self) -> bool:
        return False

    def application(self, symbol: str, arguments: Sequence[str], /) -> str:
        return f"({' '.join((symbol, *arguments))})"


@final
class _TptpSyntax(FrozenValue):
    """First-order TPTP application."""

    _noun = "TPTP syntax"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "tptp"

    @property
    def higher_order(self) -> bool:
        return False

    def application(self, symbol: str, arguments: Sequence[str], /) -> str:
        return f"{symbol}({', '.join(arguments)})"


THF_SYNTAX: ProverSyntax = _ThfSyntax()
"""Curried higher-order TPTP application."""

SMT_SYNTAX: ProverSyntax = _SmtSyntax()
"""SMT-LIB application."""

TPTP_SYNTAX: ProverSyntax = _TptpSyntax()
"""First-order TPTP application."""


@runtime_checkable
class FinitePredicate(Protocol):
    """A unary denotation tabulated over one finite domain."""

    @property
    def domain(self) -> Sequence[ModelValue]: ...

    @property
    def outputs(self) -> Sequence[ModelValue]: ...

    @property
    def known(self) -> Sequence[bool]: ...


class TruthOperations(Protocol):
    """Truth-algebra operations a logical-constant plugin may use."""

    def lift(self, *, value: bool) -> float: ...


class LogicalModel(Protocol):
    """A model supplying each capability its semantics plugins require."""


@runtime_checkable
class LogicalConstantSemantics(Protocol):
    """Denotation, prover encoding and proof rules for one logical constant."""

    @property
    def name(self) -> str: ...

    @property
    def arity(self) -> int: ...

    @property
    def connective(self) -> bool: ...

    @property
    def quantifier(self) -> bool: ...

    @property
    def rules(self) -> tuple[str, ...]: ...

    @property
    def declares_symbol(self) -> bool:
        """Return whether the prover translation declares an ordinary symbol."""
        ...

    @property
    def model_resource(self) -> str | None:
        """Return the model resource required for finite denotation."""
        ...

    def denote(
        self,
        arguments: Sequence[object],
        algebra: TruthOperations,
        model: LogicalModel,
        /,
    ) -> ModelValue: ...

    def encode(
        self,
        symbol: str,
        arguments: Sequence[str],
        /,
        *,
        dialect: ProverSyntax,
    ) -> str: ...


def _truth(value: object, /) -> bool:
    """Read one crisp truth value from a model denotation."""
    if isinstance(value, bool):
        return value

    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(value)

    msg = f"counting semantics needs crisp truth values, not {value!r}"
    raise ValueError(msg)


@final
class _MostSemantics(FrozenValue):
    """Strict-majority generalized quantification over two predicates."""

    _noun = "logical-constant semantics"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "most"

    @property
    def arity(self) -> int:
        return 2

    @property
    def connective(self) -> bool:
        return False

    @property
    def quantifier(self) -> bool:
        return True

    @property
    def rules(self) -> tuple[str, ...]:
        return ("strict-majority",)

    @property
    def declares_symbol(self) -> bool:
        return True

    @property
    def model_resource(self) -> None:
        return None

    def denote(
        self,
        arguments: Sequence[object],
        algebra: TruthOperations,
        model: LogicalModel,
        /,
    ) -> float:
        del model
        values = tuple(arguments)

        if len(values) != 2:
            msg = "most needs a restrictor and scope over one finite domain"
            raise ValueError(msg)

        restrictor, scope = values

        if not isinstance(restrictor, FinitePredicate) or not isinstance(
            scope, FinitePredicate
        ):
            msg = "most needs a restrictor and scope over one finite domain"
            raise TypeError(msg)

        if tuple(restrictor.domain) != tuple(scope.domain):
            msg = "most needs a restrictor and scope over one finite domain"
            raise ValueError(msg)

        if not all(restrictor.known) or not all(scope.known):
            msg = "most needs known restrictor and scope facts over its finite domain"
            raise ValueError(msg)

        restricted = tuple(_truth(value) for value in restrictor.outputs)
        scoped = tuple(_truth(value) for value in scope.outputs)
        support = sum(
            left and right for left, right in zip(restricted, scoped, strict=True)
        )
        counterexamples = sum(
            left and not right for left, right in zip(restricted, scoped, strict=True)
        )
        return algebra.lift(value=support > counterexamples)

    def encode(
        self,
        symbol: str,
        arguments: Sequence[str],
        /,
        *,
        dialect: ProverSyntax,
    ) -> str:
        values = tuple(arguments)

        if len(values) != 2:
            msg = "most needs a restrictor and scope"
            raise ValueError(msg)

        return dialect.application(symbol, values)


MOST = _MostSemantics()
"""The installed strict-majority generalized-quantifier semantics."""
