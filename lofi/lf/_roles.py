"""Open logical roles resolved from a language's constant classes."""

from typing import TYPE_CHECKING, Literal, Protocol, final, runtime_checkable

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "BICONDITIONAL",
    "CONDITIONAL",
    "CONJUNCTION",
    "DISJUNCTION",
    "EQUALITY",
    "EXCLUSIVE_OR",
    "EXISTENTIAL",
    "GENERALIZED_UNIVERSAL",
    "NEGATION",
    "UNIVERSAL",
    "FuzzyConnective",
    "LogicalRole",
    "LogicalRoleError",
    "ModelOperation",
]

type ModelOperation = Literal[
    "equality",
    "universal",
    "existential",
    "restricted-universal",
]
"""A model-theoretic operation supplied by a logical role."""


class LogicalRoleError(ValueError):
    """Report a logical role used outside the semantics it declares."""


class FuzzyConnective(Protocol):
    """The algebra operation used to lift one Boolean truth table."""

    def connective(
        self, table: tuple[bool, ...], arguments: Sequence[float], /
    ) -> float: ...


@runtime_checkable
class LogicalRole(Protocol):
    """The semantics and prover renderers of one open logical role."""

    @property
    def name(self) -> str:
        """Return the registry tag used in constant declarations."""
        ...

    @property
    def arity(self) -> int:
        """Return the number of term arguments the role consumes."""
        ...

    @property
    def connective(self) -> bool:
        """Return whether the role joins or modifies formulas."""
        ...

    @property
    def quantifier(self) -> bool:
        """Return whether the role binds one abstraction."""
        ...

    @property
    def truth_functional(self) -> bool:
        """Return whether a finite Boolean table interprets the role."""
        ...

    @property
    def truth_table(self) -> tuple[bool, ...] | None:
        """Return the Boolean table, or None for model-theoretic roles."""
        ...

    @property
    def model_operation(self) -> ModelOperation | None:
        """Return the role's model-theoretic operation, if it has one."""
        ...

    def truth(self, arguments: Sequence[bool], /) -> bool:
        """Evaluate the role's Boolean truth table."""
        ...

    def fuzzy(
        self,
        arguments: Sequence[float],
        algebra: FuzzyConnective,
        /,
    ) -> float:
        """Evaluate the role through its declared fuzzy reading."""
        ...

    def tptp(
        self,
        arguments: Sequence[str],
        /,
        *,
        variable: str | None = None,
        sort: str | None = None,
    ) -> str:
        """Render one application in TPTP syntax."""
        ...

    def smt(
        self,
        arguments: Sequence[str],
        /,
        *,
        variable: str | None = None,
        sort: str | None = None,
    ) -> str:
        """Render one application in SMT-LIB syntax."""
        ...

    def thf(
        self,
        arguments: Sequence[str],
        /,
        *,
        variable: str | None = None,
        sort: str | None = None,
    ) -> str:
        """Render one application in typed higher-order TPTP syntax."""
        ...


@final
class _TruthRole(FrozenValue):
    """A truth-functional role with declared prover operators."""

    _noun = "logical role"
    __slots__ = ("_arity", "_name", "_smt", "_table", "_tptp")

    def __init__(
        self,
        name: str,
        arity: int,
        table: tuple[bool, ...],
        *,
        tptp: str,
        smt: str,
    ) -> None:
        held_arity = ensure_non_negative("logical-role arity", arity)

        if len(table) != 2**held_arity:
            msg = (
                f"logical role {name!r} needs {2**held_arity} truth values, "
                f"not {len(table)}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_name", ensure_name("logical role", name))
        object.__setattr__(self, "_arity", held_arity)
        object.__setattr__(self, "_table", table)
        object.__setattr__(self, "_tptp", ensure_name("TPTP operator", tptp))
        object.__setattr__(self, "_smt", ensure_name("SMT operator", smt))

    @property
    def name(self) -> str:
        """Return the role tag."""
        return self._name

    @property
    def arity(self) -> int:
        """Return the truth-function arity."""
        return self._arity

    @property
    def connective(self) -> bool:
        """Return true for a truth-functional role."""
        return True

    @property
    def quantifier(self) -> bool:
        """Return false for a truth-functional role."""
        return False

    @property
    def truth_functional(self) -> bool:
        """Return true for a role with a Boolean table."""
        return True

    @property
    def truth_table(self) -> tuple[bool, ...]:
        """Return the declared Boolean table."""
        return self._table

    @property
    def model_operation(self) -> None:
        """Return no model-theoretic operation for a truth function."""
        return None

    def truth(self, arguments: Sequence[bool], /) -> bool:
        """Evaluate the declared truth table."""
        values = tuple(arguments)

        if len(values) != self._arity:
            msg = f"logical role {self._name!r} needs {self._arity} arguments"
            raise LogicalRoleError(msg)

        index = sum(
            int(value) << (self._arity - at - 1) for at, value in enumerate(values)
        )
        return self._table[index]

    def fuzzy(
        self,
        arguments: Sequence[float],
        algebra: FuzzyConnective,
        /,
    ) -> float:
        """Lift the declared Boolean table through a fuzzy algebra."""
        return algebra.connective(self._table, arguments)

    def tptp(
        self,
        arguments: Sequence[str],
        /,
        *,
        variable: str | None = None,
        sort: str | None = None,
    ) -> str:
        """Render prefix or infix TPTP syntax."""
        del variable, sort
        values = tuple(arguments)

        if len(values) != self._arity:
            msg = f"logical role {self._name!r} needs {self._arity} arguments"
            raise LogicalRoleError(msg)

        if self._arity == 1:
            return f"({self._tptp} {values[0]})"

        if self._arity == 2:
            return f"({values[0]} {self._tptp} {values[1]})"

        msg = f"TPTP rendering for role {self._name!r} needs a plugin implementation"
        raise LogicalRoleError(msg)

    def smt(
        self,
        arguments: Sequence[str],
        /,
        *,
        variable: str | None = None,
        sort: str | None = None,
    ) -> str:
        """Render prefix SMT-LIB syntax."""
        del variable, sort
        values = tuple(arguments)

        if len(values) != self._arity:
            msg = f"logical role {self._name!r} needs {self._arity} arguments"
            raise LogicalRoleError(msg)

        return f"({self._smt} {' '.join(values)})"

    def thf(
        self,
        arguments: Sequence[str],
        /,
        *,
        variable: str | None = None,
        sort: str | None = None,
    ) -> str:
        """Render THF with the role's typed TPTP operator."""
        return self.tptp(arguments, variable=variable, sort=sort)


@final
class _QuantifierRole(FrozenValue):
    """A one-body quantifier role."""

    _noun = "logical role"
    __slots__ = ("_model_operation", "_name", "_smt", "_tptp")

    def __init__(
        self,
        name: str,
        *,
        model_operation: Literal["universal", "existential"],
        tptp: str,
        smt: str,
    ) -> None:
        object.__setattr__(self, "_name", ensure_name("logical role", name))
        object.__setattr__(self, "_model_operation", model_operation)
        object.__setattr__(self, "_tptp", ensure_name("TPTP quantifier", tptp))
        object.__setattr__(self, "_smt", ensure_name("SMT quantifier", smt))

    @property
    def name(self) -> str:
        """Return the role tag."""
        return self._name

    @property
    def arity(self) -> int:
        """Return one abstraction argument."""
        return 1

    @property
    def connective(self) -> bool:
        """Return false for a quantifier."""
        return False

    @property
    def quantifier(self) -> bool:
        """Return true for a quantifier."""
        return True

    @property
    def truth_functional(self) -> bool:
        """Return false because a quantifier has no Boolean truth table."""
        return False

    @property
    def truth_table(self) -> None:
        """Return no Boolean table for a quantifier."""
        return None

    @property
    def model_operation(self) -> Literal["universal", "existential"]:
        """Return the quantifier's model-theoretic operation."""
        return self._model_operation

    def truth(self, arguments: Sequence[bool], /) -> bool:
        """Refuse finite Boolean evaluation of a quantifier."""
        del arguments
        msg = f"logical role {self._name!r} is not truth-functional"
        raise LogicalRoleError(msg)

    def fuzzy(
        self,
        arguments: Sequence[float],
        algebra: FuzzyConnective,
        /,
    ) -> float:
        """Refuse connective-level fuzzy evaluation of a quantifier."""
        del arguments, algebra
        msg = f"logical role {self._name!r} needs a quantified domain"
        raise LogicalRoleError(msg)

    def tptp(
        self,
        arguments: Sequence[str],
        /,
        *,
        variable: str | None = None,
        sort: str | None = None,
    ) -> str:
        """Render a TPTP binder."""
        values = tuple(arguments)

        if len(values) != 1 or variable is None:
            msg = f"logical role {self._name!r} needs one bound body"
            raise LogicalRoleError(msg)

        annotation = "" if sort is None else f": {sort}"
        return f"({self._tptp} [{variable}{annotation}] : {values[0]})"

    def smt(
        self,
        arguments: Sequence[str],
        /,
        *,
        variable: str | None = None,
        sort: str | None = None,
    ) -> str:
        """Render an SMT-LIB binder."""
        values = tuple(arguments)

        if len(values) != 1 or variable is None or sort is None:
            msg = f"logical role {self._name!r} needs one sorted bound body"
            raise LogicalRoleError(msg)

        return f"({self._smt} (({variable} {sort})) {values[0]})"

    def thf(
        self,
        arguments: Sequence[str],
        /,
        *,
        variable: str | None = None,
        sort: str | None = None,
    ) -> str:
        """Render a typed THF binder."""
        return self.tptp(arguments, variable=variable, sort=sort)


@final
class _EqualityRole(FrozenValue):
    """Polymorphic equality."""

    _noun = "logical role"
    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the role tag."""
        return "equality"

    @property
    def arity(self) -> int:
        """Return equality's arity."""
        return 2

    @property
    def connective(self) -> bool:
        """Return false because equality compares terms."""
        return False

    @property
    def quantifier(self) -> bool:
        """Return false for equality."""
        return False

    @property
    def truth_functional(self) -> bool:
        """Return false because equality depends on a term interpretation."""
        return False

    @property
    def truth_table(self) -> None:
        """Return no Boolean table for equality."""
        return None

    @property
    def model_operation(self) -> Literal["equality"]:
        """Return the equality operation."""
        return "equality"

    def truth(self, arguments: Sequence[bool], /) -> bool:
        """Refuse propositional evaluation of equality."""
        del arguments
        msg = "equality is not truth-functional"
        raise LogicalRoleError(msg)

    def fuzzy(
        self,
        arguments: Sequence[float],
        algebra: FuzzyConnective,
        /,
    ) -> float:
        """Refuse connective-level fuzzy evaluation of equality."""
        del arguments, algebra
        msg = "equality needs model values"
        raise LogicalRoleError(msg)

    def tptp(
        self,
        arguments: Sequence[str],
        /,
        *,
        variable: str | None = None,
        sort: str | None = None,
    ) -> str:
        """Render TPTP equality."""
        del variable, sort
        values = tuple(arguments)

        if len(values) != 2:
            msg = "equality needs two arguments"
            raise LogicalRoleError(msg)

        return f"({values[0]} = {values[1]})"

    def smt(
        self,
        arguments: Sequence[str],
        /,
        *,
        variable: str | None = None,
        sort: str | None = None,
    ) -> str:
        """Render SMT-LIB equality."""
        del variable, sort
        values = tuple(arguments)

        if len(values) != 2:
            msg = "equality needs two arguments"
            raise LogicalRoleError(msg)

        return f"(= {values[0]} {values[1]})"

    def thf(
        self,
        arguments: Sequence[str],
        /,
        *,
        variable: str | None = None,
        sort: str | None = None,
    ) -> str:
        """Render typed higher-order equality."""
        return self.tptp(arguments, variable=variable, sort=sort)


@final
class _RestrictedUniversalRole(FrozenValue):
    """A generalized universal over a restrictor and a scope."""

    _noun = "logical role"
    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the role tag."""
        return "generalized-universal"

    @property
    def arity(self) -> int:
        """Return the restrictor and scope arity."""
        return 2

    @property
    def connective(self) -> bool:
        """Return false for a quantifier."""
        return False

    @property
    def quantifier(self) -> bool:
        """Return true for a generalized quantifier."""
        return True

    @property
    def truth_functional(self) -> bool:
        """Return false for a generalized quantifier."""
        return False

    @property
    def truth_table(self) -> None:
        """Return no Boolean table for a restricted quantifier."""
        return None

    @property
    def model_operation(self) -> Literal["restricted-universal"]:
        """Return the restricted-universal operation."""
        return "restricted-universal"

    def truth(self, arguments: Sequence[bool], /) -> bool:
        """Refuse propositional evaluation."""
        del arguments
        msg = "a generalized quantifier is not truth-functional"
        raise LogicalRoleError(msg)

    def fuzzy(
        self,
        arguments: Sequence[float],
        algebra: FuzzyConnective,
        /,
    ) -> float:
        """Refuse connective-level fuzzy evaluation of a restricted quantifier."""
        del arguments, algebra
        msg = "a generalized quantifier needs finite predicate denotations"
        raise LogicalRoleError(msg)

    def tptp(
        self,
        arguments: Sequence[str],
        /,
        *,
        variable: str | None = None,
        sort: str | None = None,
    ) -> str:
        """Render restricted universal quantification in TPTP."""
        values = tuple(arguments)

        if len(values) != 2 or variable is None:
            msg = "a generalized universal needs a restrictor and a scope"
            raise LogicalRoleError(msg)

        annotation = "" if sort is None else f": {sort}"
        return f"(! [{variable}{annotation}] : ({values[0]} => {values[1]}))"

    def smt(
        self,
        arguments: Sequence[str],
        /,
        *,
        variable: str | None = None,
        sort: str | None = None,
    ) -> str:
        """Render restricted universal quantification in SMT-LIB."""
        values = tuple(arguments)

        if len(values) != 2 or variable is None or sort is None:
            msg = "a generalized universal needs two sorted bound bodies"
            raise LogicalRoleError(msg)

        return f"(forall (({variable} {sort})) (=> {values[0]} {values[1]}))"

    def thf(
        self,
        arguments: Sequence[str],
        /,
        *,
        variable: str | None = None,
        sort: str | None = None,
    ) -> str:
        """Render restricted universal quantification in THF."""
        return self.tptp(arguments, variable=variable, sort=sort)


NEGATION = _TruthRole("negation", 1, (True, False), tptp="~", smt="not")
CONJUNCTION = _TruthRole(
    "conjunction", 2, (False, False, False, True), tptp="&", smt="and"
)
DISJUNCTION = _TruthRole(
    "disjunction", 2, (False, True, True, True), tptp="|", smt="or"
)
CONDITIONAL = _TruthRole(
    "conditional", 2, (True, True, False, True), tptp="=>", smt="=>"
)
BICONDITIONAL = _TruthRole(
    "biconditional", 2, (True, False, False, True), tptp="<=>", smt="="
)
EXCLUSIVE_OR = _TruthRole(
    "exclusive-or", 2, (False, True, True, False), tptp="<~>", smt="xor"
)
EQUALITY = _EqualityRole()
UNIVERSAL = _QuantifierRole(
    "universal", model_operation="universal", tptp="!", smt="forall"
)
EXISTENTIAL = _QuantifierRole(
    "existential", model_operation="existential", tptp="?", smt="exists"
)
GENERALIZED_UNIVERSAL = _RestrictedUniversalRole()
