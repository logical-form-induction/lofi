"""Differential oracle for the generalized algebraic theory of a grammar.

Declared sorts and operators become closed sorts and constructors. Proof
schemas become constructors of a category-indexed derivation family, with
premise-determined variables inferred as implicit indices. Nondeterministic
schemas and undeclared free result variables are rejected.

A rule standing for a family of schemata, such as composition at every finite
order, has no finite deterministic presentation and so no constructor. Such a
rule elaborates the proof term of each step it licenses instead, and the
compiled theory names it among its elaborated rules.
"""

from json import dumps
from typing import TYPE_CHECKING, Final, NamedTuple, Never, Self, final, override

import didactic.api as dx

from lofi.judge import (
    ElaboratedSchema,
    ProofSchema,
)
from lofi.spec._errors import DerivationIndexError
from lofi.spec.grammar._identifiers import encode_identifier
from lofi.terms import MVar, Node, PVar

if TYPE_CHECKING:
    from collections.abc import Sequence

    import panproto

    from lofi._types import JsonValue
    from lofi.judge import Rule, RuleSet
    from lofi.terms import Operator, Pattern, PayloadSort, Signature, Sort, Term

__all__ = [
    "CheckedTheory",
    "DerivationConstructor",
    "mirror",
    "theory_bytes",
]

DERIVATION_FAMILY: Final = "derivation"
"""The indexed family whose inhabitants are derivations of a category."""

LEXICAL_OPERATION: Final = "lexical"
"""The operation that admits a lexical axiom at an explicit category."""

_NATURAL_SORT: Final = "natural"
_ZERO: Final = "zero"
_SUCCESSOR: Final = "successor"


def _scope_counts(operator: Operator) -> tuple[int, ...]:
    """Return how many variables scope over each argument of an operator.

    Parameters
    ----------
    operator : Operator
        The operator.
    """
    return tuple(len(operator.scope(position)) for position in range(operator.arity))


class DerivationConstructor(NamedTuple):
    """One rule schema as a constructor of the derivation family.

    Attributes
    ----------
    rule : str
        The runtime rule name.
    constructor : str
        The theory constructor name.
    inferred : tuple[str, ...]
        Index parameters the premises determine, in declaration order.
    declared : tuple[str, ...]
        Index parameters the constructor takes explicitly.
    premises : int
        The premise count.
    premise_indices : tuple[str, ...]
        Complete canonical premise-index expressions.
    conclusion_index : str
        The complete canonical conclusion-index expression.
    """

    rule: str
    constructor: str
    inferred: tuple[str, ...]
    declared: tuple[str, ...]
    premises: int
    premise_indices: tuple[str, ...]
    conclusion_index: str


@final
class CheckedTheory:
    """A grammar's compiled theory with its derivation-constructor records.

    Parameters
    ----------
    builder : dx.GADT
        The sealed declaration the theory was compiled from.
    theory : panproto.Theory
        The compiled theory.
    constructors : tuple[DerivationConstructor, ...]
        The derivation constructors in rule order.
    elaborated : tuple[str, ...]
        The names of the rules whose steps are elaborated rather than
        presented as constructors, in rule order.
    """

    __slots__ = (
        "_builder",
        "_constructors",
        "_deferred",
        "_elaborated",
        "_theory",
    )

    def __init__(
        self,
        builder: dx.GADT,
        theory: panproto.Theory,
        constructors: tuple[DerivationConstructor, ...],
        elaborated: tuple[str, ...],
    ) -> None:
        object.__setattr__(self, "_builder", builder)
        object.__setattr__(self, "_theory", theory)
        object.__setattr__(self, "_constructors", constructors)
        object.__setattr__(self, "_elaborated", elaborated)
        object.__setattr__(self, "_deferred", None)

    @classmethod
    def deferred(
        cls,
        name: str,
        signature: Signature,
        rules: RuleSet[Term],
        category_sort: Sort,
        /,
    ) -> Self:
        """Return a diagnostic theory view materialized only when inspected."""
        value = cls.__new__(cls)
        object.__setattr__(value, "_builder", None)
        object.__setattr__(value, "_theory", None)
        object.__setattr__(value, "_constructors", None)
        object.__setattr__(value, "_elaborated", None)
        object.__setattr__(
            value,
            "_deferred",
            (name, signature, rules, category_sort),
        )
        return value

    def _components(
        self,
    ) -> tuple[
        dx.GADT,
        panproto.Theory,
        tuple[DerivationConstructor, ...],
        tuple[str, ...],
    ]:
        """Return the eager components, materializing a diagnostic view if needed."""
        if self._deferred is None:
            return self._builder, self._theory, self._constructors, self._elaborated

        materialized = mirror(*self._deferred)
        return (
            materialized.builder,
            materialized.theory,
            materialized.constructors,
            materialized.elaborated,
        )

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a checked theory is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a checked theory is immutable"

        raise AttributeError(msg)

    def __copy__(self) -> Self:
        """Return this immutable value."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable value."""
        del memo

        return self

    @property
    def builder(self) -> dx.GADT:
        """Return the sealed declaration, which infers sorts of derivation terms."""
        return self._components()[0]

    @property
    def theory(self) -> panproto.Theory:
        """Return the compiled theory."""
        return self._components()[1]

    @property
    def constructors(self) -> tuple[DerivationConstructor, ...]:
        """Return the derivation constructors in rule order."""
        return self._components()[2]

    @property
    def elaborated(self) -> tuple[str, ...]:
        """Return the rules whose steps the constructor language does not present."""
        return self._components()[3]


def theory_bytes(
    theory: panproto.Theory,
    *,
    source_version: str,
    target_version: str,
) -> bytes:
    """Return the canonical bytes of a theory under one lowering route.

    Parameters
    ----------
    theory : panproto.Theory
        The compiled theory.
    source_version : str
        The declaration version the theory was checked from.
    target_version : str
        The native version the theory was lowered to.

    Returns
    -------
    bytes
        The route and the theory's dictionary form as JSON with keys sorted,
        no whitespace and every non-ASCII character escaped, encoded as UTF-8.
    """
    document = dumps(
        {
            "route": {"source": source_version, "target": target_version},
            "theory": theory.to_dict(),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    return document.encode("utf-8")


def _prefixed(role: str, name: str) -> str:
    """Return the theory name for a declared name in one role.

    Parameters
    ----------
    role : str
        The name's role.
    name : str
        The declared name.
    """
    return f"{role}_{encode_identifier(name)}"


class _Naturals(NamedTuple):
    """The closed sort of binding counts with its two constructors.

    Attributes
    ----------
    sort : dx.Family
        The sort of binding counts.
    zero : dx.Operation
        The constructor of the count zero.
    successor : dx.Operation
        The constructor of the count one above its input.
    """

    sort: dx.Family
    zero: dx.Operation
    successor: dx.Operation

    def of(self, count: int) -> dx.Term:
        """Return a binding count as a term of the natural sort.

        Parameters
        ----------
        count : int
            The number of binders.
        """
        term: dx.Term = self.zero()

        for _ in range(count):
            term = self.successor(term)

        return term


class _Scope(NamedTuple):
    """The family of bodies of one sort under a number of binders.

    Attributes
    ----------
    family : dx.Family
        The family, indexed by a binding count.
    bind : dx.Operation
        The constructor taking a binding count and a body of the sort.
    naturals : _Naturals
        The sort of binding counts the family is indexed by.
    """

    family: dx.Family
    bind: dx.Operation
    naturals: _Naturals


def _index_expression(
    pattern: Pattern,
    signature: Signature,
    sort: Sort,
    binders: int,
    /,
) -> str:
    """Return one canonical diagnostic index expression."""
    return dumps(
        _index_expression_value(pattern, signature, sort, binders),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _index_expression_value(
    pattern: Pattern,
    signature: Signature,
    sort: Sort,
    binders: int,
    /,
) -> JsonValue:
    """Return the JSON value of one nested diagnostic index expression."""
    if isinstance(pattern, MVar):
        return {
            "kind": "variable",
            "name": _prefixed("index", pattern.name),
            "sort": sort,
            "binders": binders,
        }
    if isinstance(pattern, Node) and isinstance(pattern.op, str):
        operator = signature.by_name(pattern.op)
        payload: JsonValue
        if pattern.payload is None:
            payload = None
        elif isinstance(pattern.payload, PVar):
            payload = {
                "kind": "variable",
                "name": _prefixed("payload", pattern.payload.name),
                "payload": operator.payload_sort,
            }
        else:
            payload = {"kind": "fixed", "value": pattern.payload}
        return {
            "kind": "node",
            "operator": operator.name,
            "sort": sort,
            "binders": binders,
            "payload": payload,
            "arguments": [
                _index_expression_value(argument, signature, child, bound)
                for argument, child, bound in zip(
                    pattern.args,
                    operator.arg_sorts,
                    _scope_counts(operator),
                    strict=True,
                )
            ],
        }
    return {"kind": "nondeterministic", "pattern": repr(pattern)}


@final
class _Mirror:
    """Builds the theory of one signature and one rule inventory.

    Parameters
    ----------
    name : str
        The theory name.
    signature : Signature
        The signature to mirror.
    category_sort : Sort
        The sort that indexes the derivation family.
    """

    def __init__(self, name: str, signature: Signature, category_sort: Sort) -> None:
        self._signature = signature
        self._category_sort = category_sort
        self._builder = dx.GADT(_prefixed("grammar", name))
        self._sorts = {
            sort: self._builder.sort(_prefixed("sort", sort), closed=True)
            for sort in signature.sorts
        }
        self._operators: dict[str, dx.Operation] = {}
        self._payloads: dict[PayloadSort, dx.Family] = {}
        self._literals: dict[tuple[PayloadSort, int], dx.Operation] = {}
        self._bound: dict[Sort, _Scope] = {}
        self._natural: _Naturals | None = None
        self._variables: dict[str, dx.SortExpr] = {}
        self._elaborated: list[str] = []
        self._rule = ""
        category = self._sorts[category_sort]
        derivation = self._builder.family(DERIVATION_FAMILY, category=category)
        self._derivation = derivation
        self._builder.operation(
            LEXICAL_OPERATION,
            category=category,
            returns=lambda category: derivation[category],
        )

    def _payload(self, payload_sort: PayloadSort) -> dx.Family:
        """Return the open sort of one payload kind.

        Parameters
        ----------
        payload_sort : PayloadSort
            The payload kind.
        """
        family = self._payloads.get(payload_sort)

        if family is None:
            family = self._builder.sort(f"payload_{payload_sort}")
            self._payloads[payload_sort] = family

        return family

    def _naturals(self) -> _Naturals:
        """Return the closed sort of binding counts with its constructors."""
        if self._natural is None:
            sort = self._builder.sort(_NATURAL_SORT, closed=True)
            self._natural = _Naturals(
                sort,
                self._builder.constructor(_ZERO, returns=sort),
                self._builder.constructor(_SUCCESSOR, predecessor=sort, returns=sort),
            )

        return self._natural

    def _scope(self, sort: Sort) -> _Scope:
        """Return the family of bodies of one sort under a number of binders.

        Parameters
        ----------
        sort : Sort
            The body sort.
        """
        scope = self._bound.get(sort)

        if scope is None:
            naturals = self._naturals()
            family = self._builder.family(
                _prefixed("bound", sort), binders=naturals.sort
            )
            bind = self._builder.constructor(
                _prefixed("bind", sort),
                binders=naturals.sort,
                body=self._sorts[sort],
                returns=lambda binders: family[binders],
            )
            scope = _Scope(family, bind, naturals)
            self._bound[sort] = scope

        return scope

    def _position(self, sort: Sort, binds: int) -> dx.SortExpr:
        """Return the sort expression of an argument position.

        Parameters
        ----------
        sort : Sort
            The argument sort.
        binds : int
            The number of binders at the position.
        """
        if binds == 0:
            return self._sorts[sort]()

        scope = self._scope(sort)

        return scope.family[scope.naturals.of(binds)]

    def operators(self) -> None:
        """Declare one constructor per operator in signature code order."""
        for operator in self._signature.operators:
            inputs: dict[str, dx.InputSpec] = {}

            if operator.has_payload:
                inputs["payload"] = self._payload(operator.payload_sort)

            for position, (sort, binds) in enumerate(
                zip(operator.arg_sorts, _scope_counts(operator), strict=True)
            ):
                inputs[f"argument_{position}"] = self._position(sort, binds)

            self._operators[operator.name] = self._builder.constructor(
                _prefixed("op", operator.name),
                returns=self._sorts[operator.sort],
                **inputs,
            )

    def _literal(self, payload_sort: PayloadSort, value: int) -> dx.Term:
        """Return the nullary operation standing for one fixed payload, applied.

        Parameters
        ----------
        payload_sort : PayloadSort
            The payload kind.
        value : int
            The fixed payload value.
        """
        literal = self._literals.get((payload_sort, value))

        if literal is None:
            literal = self._builder.operation(
                _prefixed(f"payload_{payload_sort}", str(value)),
                returns=self._payload(payload_sort),
            )
            self._literals[(payload_sort, value)] = literal

        return literal()

    def _variable(self, name: str, sort: dx.SortExpr) -> dx.Term:
        """Return the index variable for a pattern variable at one sort.

        Parameters
        ----------
        name : str
            The variable name.
        sort : dx.SortExpr
            The variable's sort expression.
        """
        held = self._variables.get(name)

        if held is None:
            self._variables[name] = sort
        elif held != sort:
            reason = f"variable {name!r} stands at sorts {held} and {sort}"
            raise DerivationIndexError(rule=self._rule, reason=reason)

        return dx.Var(name)

    def _payload_term(
        self, payload: int | PVar | None, operator: Operator
    ) -> dx.Term | None:
        """Return the payload input of a node pattern, if its operator takes one.

        Parameters
        ----------
        payload : int | PVar | None
            The payload pattern.
        operator : Operator
            The node operator.
        """
        if not operator.has_payload:
            if payload is not None:
                reason = f"operator {operator.name!r} takes no payload"
                raise DerivationIndexError(rule=self._rule, reason=reason)

            return None

        if payload is None:
            reason = f"operator {operator.name!r} requires a payload"
            raise DerivationIndexError(rule=self._rule, reason=reason)

        if isinstance(payload, PVar):
            return self._variable(
                _prefixed("payload", payload.name),
                self._payload(operator.payload_sort)(),
            )

        return self._literal(operator.payload_sort, payload)

    def _application(
        self,
        operator: Operator,
        payload: int | PVar | None,
        arguments: tuple[dx.Term, ...],
        binds: int,
    ) -> dx.Term:
        """Return an operator applied to translated arguments at one position.

        Parameters
        ----------
        operator : Operator
            The operator to apply.
        payload : int | PVar | None
            The payload pattern.
        arguments : tuple[dx.Term, ...]
            The translated arguments.
        binds : int
            The number of binders at the position.
        """
        held = self._payload_term(payload, operator)
        term: dx.Term = self._operators[operator.name](
            *(() if held is None else (held,)), *arguments
        )

        if binds == 0:
            return term

        scope = self._scope(operator.sort)

        return scope.bind(scope.naturals.of(binds), term)

    def _node(self, pattern: Node, sort: Sort, binds: int) -> dx.Term:
        """Return the index term of a node pattern at one argument position.

        Parameters
        ----------
        pattern : Node
            The node pattern.
        sort : Sort
            The required result sort.
        binds : int
            The number of binders at the position.
        """
        if isinstance(pattern.op, PVar):
            reason = (
                f"operator variable {pattern.op.name!r} makes the schema "
                "nondeterministic"
            )
            raise DerivationIndexError(rule=self._rule, reason=reason)

        operator = self._signature.by_name(pattern.op)

        if operator.sort != sort or len(pattern.args) != operator.arity:
            reason = (
                f"operator {operator.name!r} has sort {operator.sort!r} and "
                f"arity {operator.arity} where sort {sort!r} and arity "
                f"{len(pattern.args)} are required"
            )
            raise DerivationIndexError(rule=self._rule, reason=reason)

        arguments = tuple(
            self._term(argument, child, bound)
            for argument, child, bound in zip(
                pattern.args, operator.arg_sorts, _scope_counts(operator), strict=True
            )
        )

        return self._application(operator, pattern.payload, arguments, binds)

    def _term(self, pattern: Pattern, sort: Sort, binds: int) -> dx.Term:
        """Return the index term of a pattern at one argument position.

        Parameters
        ----------
        pattern : Pattern
            The pattern to translate.
        sort : Sort
            The required result sort.
        binds : int
            The number of binders at the position.
        """
        if isinstance(pattern, MVar):
            return self._variable(
                _prefixed("index", pattern.name),
                self._position(sort, binds),
            )

        if isinstance(pattern, Node):
            return self._node(pattern, sort, binds)

        reason = f"spine context {pattern.name!r} makes the schema nondeterministic"
        raise DerivationIndexError(rule=self._rule, reason=reason)

    def _constructor(
        self,
        rule: Rule[Term],
        serial: int,
        premises: Sequence[Pattern],
        conclusion: Pattern,
    ) -> DerivationConstructor:
        """Declare one derivation constructor for one schema of a rule.

        Parameters
        ----------
        rule : Rule[Term]
            The source rule.
        serial : int
            The schema's position within the rule.
        premises : Sequence[Pattern]
            The premise patterns.
        conclusion : Pattern
            The conclusion pattern.
        """
        derivation = self._derivation
        self._variables = {}
        self._rule = rule.name
        category = self._category_sort
        premise_terms = tuple(self._term(premise, category, 0) for premise in premises)
        inferred = tuple(self._variables)
        conclusion_term = self._term(conclusion, category, 0)

        declared = tuple(name for name in self._variables if name not in inferred)

        if declared and rule.forward_is_complete:
            names = ", ".join(declared)
            raise DerivationIndexError(
                rule=rule.name,
                reason=f"the premises leave the result index variables {names} free",
            )

        name = f"rule_{serial}_{encode_identifier(rule.name)}"
        inputs: dict[str, dx.InputSpec] = {
            variable: dx.Implicit(self._variables[variable]) for variable in inferred
        }
        inputs.update((variable, self._variables[variable]) for variable in declared)
        inputs.update(
            (f"premise_{position}", derivation[term])
            for position, term in enumerate(premise_terms)
        )
        self._builder.constructor(name, returns=derivation[conclusion_term], **inputs)

        return DerivationConstructor(
            rule.name,
            name,
            inferred,
            declared,
            len(premises),
            tuple(
                _index_expression(premise, self._signature, category, 0)
                for premise in premises
            ),
            _index_expression(conclusion, self._signature, category, 0),
        )

    def rules(self, rules: RuleSet[Term]) -> tuple[DerivationConstructor, ...]:
        """Declare the derivation constructors of every rule and return the records.

        Parameters
        ----------
        rules : RuleSet[Term]
            The rule set to compile.
        """
        records: list[DerivationConstructor] = []

        for rule in rules:
            schemata = rule.proof_schemata if isinstance(rule, ProofSchema) else ()

            if not schemata:
                if not isinstance(rule, ElaboratedSchema):
                    raise DerivationIndexError(
                        rule=rule.name,
                        reason="the rule offers no deterministic schema",
                    )

                self._elaborated.append(rule.name)
                continue

            records.extend(
                self._constructor(rule, serial, premises, conclusion)
                for serial, (premises, conclusion) in enumerate(schemata)
            )

        return tuple(records)

    def compile(self, constructors: tuple[DerivationConstructor, ...]) -> CheckedTheory:
        """Compile the declared theory.

        Parameters
        ----------
        constructors : tuple[DerivationConstructor, ...]
            The derivation constructors declared so far, in rule order.
        """
        return CheckedTheory(
            self._builder,
            self._builder.compile(),
            constructors,
            tuple(self._elaborated),
        )


def mirror(
    name: str,
    signature: Signature,
    rules: RuleSet[Term],
    category_sort: Sort,
) -> CheckedTheory:
    """Return the compiled theory of a signature and its rule inventory.

    Parameters
    ----------
    name : str
        The grammar name the theory is named after.
    signature : Signature
        The signature to mirror.
    rules : RuleSet[Term]
        The rules whose schemata become derivation constructors.
    category_sort : Sort
        The sort that indexes the derivation family.

    Returns
    -------
    CheckedTheory
        The compiled theory with its derivation-constructor records.

    Raises
    ------
    DerivationIndexError
        If a rule offers no deterministic schema and elaborates no step, or a
        schema has a conclusion outside the category sort, an operator applied
        at the wrong arity or payload, an operator variable or spine context, a
        variable at two sorts, or a result index the premises leave free while
        the rule claims complete forward enumeration.
    """
    builder = _Mirror(name, signature, category_sort)
    builder.operators()

    return builder.compile(builder.rules(rules))
