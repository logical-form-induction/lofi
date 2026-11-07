"""Independent ground-type checker for generated logical forms."""

from __future__ import annotations

from typing import Literal, NamedTuple

__all__ = [
    "And",
    "Apply",
    "AtomicType",
    "Bound",
    "Constant",
    "ConstantDeclaration",
    "Exists",
    "Forall",
    "FunctionType",
    "GroundType",
    "IllTypedError",
    "Lambda",
    "Not",
    "Or",
    "SignatureDeclaration",
    "TermDeclaration",
    "UnknownConstantError",
    "type_of",
]


class AtomicType(NamedTuple):
    """Ground atomic type."""

    name: Literal["e", "t"]


class FunctionType(NamedTuple):
    """Ground function type."""

    domain: GroundType
    codomain: GroundType


type GroundType = AtomicType | FunctionType
"""Ground semantic type accepted by the naive checker."""


class Bound(NamedTuple):
    """Bound variable identified by a de Bruijn index."""

    index: int


class Constant(NamedTuple):
    """Declared nonlogical constant."""

    name: str


class Lambda(NamedTuple):
    """Typed lambda abstraction."""

    parameter: GroundType
    body: TermDeclaration


class Exists(NamedTuple):
    """Existential quantification over a typed variable."""

    parameter: GroundType
    body: TermDeclaration


class Forall(NamedTuple):
    """Universal quantification over a typed variable."""

    parameter: GroundType
    body: TermDeclaration


class And(NamedTuple):
    """Logical conjunction."""

    left: TermDeclaration
    right: TermDeclaration


class Or(NamedTuple):
    """Logical disjunction."""

    left: TermDeclaration
    right: TermDeclaration


class Not(NamedTuple):
    """Logical negation."""

    body: TermDeclaration


class Apply(NamedTuple):
    """Function application."""

    function: TermDeclaration
    argument: TermDeclaration


type TermDeclaration = (
    Bound | Constant | Lambda | Exists | Forall | And | Or | Not | Apply
)
"""First-order declaration generated for checker conformance."""

type _TruthDeclaration = Exists | Forall | And | Or | Not
"""Declaration whose typing rule requires truth-valued premises."""


class ConstantDeclaration(NamedTuple):
    """Constant name paired with its ground type."""

    name: str
    type: GroundType


class SignatureDeclaration(NamedTuple):
    """Constant declarations available to a generated term."""

    constants: tuple[ConstantDeclaration, ...]

    def type_of(self, name: str) -> GroundType:
        """Return the type declared for a constant."""
        for declaration in self.constants:
            if declaration.name == name:
                return declaration.type

        raise UnknownConstantError(name)


class IllTypedError(ValueError):
    """A declaration has no ground type under the supplied signature."""


class UnknownConstantError(IllTypedError):
    """A declaration names a constant absent from its signature."""


def _require(
    signature: SignatureDeclaration,
    term: TermDeclaration,
    expected: GroundType,
    environment: tuple[GroundType, ...],
) -> None:
    """Reject a term whose inferred type differs from an expected type."""
    received = type_of(signature, term, environment)

    if received != expected:
        msg = f"expected {expected!r}, received {received!r}"
        raise IllTypedError(msg)


def _truth_type(
    signature: SignatureDeclaration,
    term: _TruthDeclaration,
    environment: tuple[GroundType, ...],
) -> GroundType:
    """Check truth-valued premises and return the truth type."""
    truth = AtomicType("t")

    match term:
        case Exists(parameter, body) | Forall(parameter, body):
            requirements = ((body, (parameter, *environment)),)
        case And(left, right) | Or(left, right):
            requirements = ((left, environment), (right, environment))
        case Not(body):
            requirements = ((body, environment),)

    for body, held_environment in requirements:
        _require(signature, body, truth, held_environment)

    return truth


def _application_type(
    signature: SignatureDeclaration,
    function: TermDeclaration,
    argument: TermDeclaration,
    environment: tuple[GroundType, ...],
) -> GroundType:
    """Check an application and return its codomain."""
    function_type = type_of(signature, function, environment)

    if not isinstance(function_type, FunctionType):
        msg = f"application head has nonfunction type {function_type!r}"
        raise IllTypedError(msg)

    _require(signature, argument, function_type.domain, environment)

    return function_type.codomain


def type_of(
    signature: SignatureDeclaration,
    term: TermDeclaration,
    environment: tuple[GroundType, ...] = (),
) -> GroundType:
    """Return a declaration's ground type."""
    match term:
        case Bound(index):
            if index < 0 or index >= len(environment):
                size = len(environment)
                msg = f"bound index {index} is outside an environment of {size}"
                raise IllTypedError(msg)

            return environment[index]

        case Constant(name):
            return signature.type_of(name)

        case Lambda(parameter, body):
            return FunctionType(
                parameter,
                type_of(signature, body, (parameter, *environment)),
            )

        case Exists() | Forall() | And() | Or() | Not():
            return _truth_type(signature, term, environment)

        case Apply(function, argument):
            return _application_type(signature, function, argument, environment)

    msg = "unreachable declaration variant"
    raise AssertionError(msg)
