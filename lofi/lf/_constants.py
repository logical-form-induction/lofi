"""Finite-model and prover semantics for non-truth-valued logical constants."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, final, override, runtime_checkable

from lofi._frozen import FrozenValue
from lofi.lf._semantics import FinitePredicate

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.lf._semantics import LogicalModel, ProverSyntax, TruthOperations
    from lofi.lf.models._frame import ModelValue

__all__ = ["CHOICE", "ChoiceFunction"]


@runtime_checkable
class _ChoiceModel(Protocol):
    """Expose a model's validated choice operation."""

    def choose(self, predicate: FinitePredicate, /) -> ModelValue: ...


def _truth(value: object, /) -> bool:
    """Read one crisp truth value from a finite predicate."""
    if isinstance(value, bool):
        return value

    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(value)

    msg = f"choice semantics needs crisp truth values, not {value!r}"
    raise ValueError(msg)


def _facts(predicate: FinitePredicate, /) -> tuple[bool, ...]:
    """Return one predicate's total crisp truth table."""
    domain = tuple(predicate.domain)
    outputs = tuple(predicate.outputs)
    known = tuple(predicate.known)

    if not domain or len(outputs) != len(domain) or len(known) != len(domain):
        msg = "a choice predicate has one known output per nonempty domain value"
        raise ValueError(msg)

    if not all(known):
        msg = "choice semantics needs known facts over its finite domain"
        raise ValueError(msg)

    return tuple(_truth(value) for value in outputs)


@final
class ChoiceFunction(FrozenValue):
    """A finite model's sound choice mapping with canonical total completion."""

    _noun = "finite choice function"
    __slots__ = ("_selections",)

    def __init__(
        self,
        selections: Mapping[FinitePredicate, ModelValue],
        /,
    ) -> None:
        held: dict[FinitePredicate, ModelValue] = {}

        for predicate, selected in selections.items():
            if not isinstance(predicate, FinitePredicate):
                msg = f"a choice-function key is a finite predicate, not {predicate!r}"
                raise TypeError(msg)

            domain = tuple(predicate.domain)
            facts = _facts(predicate)

            if selected not in domain:
                msg = (
                    f"choice value {selected!r} lies outside predicate domain "
                    f"{domain!r}"
                )
                raise ValueError(msg)

            position = domain.index(selected)

            if any(facts) and not facts[position]:
                msg = (
                    "a finite choice function violates the choice axiom: "
                    f"{selected!r} does not satisfy its predicate"
                )
                raise ValueError(msg)

            held[predicate] = selected

        object.__setattr__(self, "_selections", MappingProxyType(held))

    def choose(self, predicate: FinitePredicate, /) -> ModelValue:
        """Select the declared witness or the first witness in domain order."""
        if not isinstance(predicate, FinitePredicate):
            msg = f"a choice argument is a finite predicate, not {predicate!r}"
            raise TypeError(msg)

        facts = _facts(predicate)
        selected = self._selections.get(predicate)

        if selected is not None:
            return selected

        domain = tuple(predicate.domain)

        for value, true in zip(domain, facts, strict=True):
            if true:
                return value

        return domain[0]

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is ChoiceFunction and dict(self._selections) == dict(
            other._selections
        )

    @override
    def __hash__(self) -> int:
        return hash(
            (
                ChoiceFunction,
                frozenset(self._selections.items()),
            )
        )

    @override
    def __repr__(self) -> str:
        return f"ChoiceFunction({dict(self._selections)!r})"


def _choice_binder(argument: str, /) -> str:
    """Rewrite the generated THF abstraction as the built-in choice binder."""
    prefix = "(^ ["
    middle = "]: "

    if not argument.startswith(prefix) or not argument.endswith(")"):
        msg = "THF choice needs one abstraction argument"
        raise ValueError(msg)

    header, separator, body = argument[len(prefix) : -1].partition(middle)

    if not separator or not header or not body:
        msg = "THF choice needs one typed abstraction argument"
        raise ValueError(msg)

    return f"(@+ [{header}]: {body})"


@final
class _ChoiceSemantics(FrozenValue):
    """Hilbert choice over one finite predicate."""

    _noun = "logical-constant semantics"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "choice"

    @property
    def arity(self) -> int:
        return 1

    @property
    def connective(self) -> bool:
        return False

    @property
    def quantifier(self) -> bool:
        return True

    @property
    def rules(self) -> tuple[str, ...]:
        return ("choice-axiom",)

    @property
    def declares_symbol(self) -> bool:
        return False

    @property
    def model_resource(self) -> str:
        return "choice-function"

    def denote(
        self,
        arguments: Sequence[object],
        algebra: TruthOperations,
        model: LogicalModel,
        /,
    ) -> ModelValue:
        del algebra
        values = tuple(arguments)

        if len(values) != 1 or not isinstance(values[0], FinitePredicate):
            msg = "choice needs one finite predicate"
            raise TypeError(msg)

        if not isinstance(model, _ChoiceModel):
            msg = "choice semantics needs a finite choice-function model resource"
            raise TypeError(msg)

        return model.choose(values[0])

    def encode(
        self,
        symbol: str,
        arguments: Sequence[str],
        /,
        *,
        dialect: ProverSyntax,
    ) -> str:
        del symbol
        values = tuple(arguments)

        if not dialect.higher_order:
            msg = "choice is emitted only through THF's choice binder"
            raise ValueError(msg)

        if len(values) != 1:
            msg = "choice needs one abstraction"
            raise ValueError(msg)

        return _choice_binder(values[0])


CHOICE = _ChoiceSemantics()
"""The installed Hilbert-choice semantics."""
