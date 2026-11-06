"""The homomorphic evaluator for declared higher-order languages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, final

from lofi.lf.models._frame import FrameTypeError, FunctionValue, ModelValue

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.lf import LogicalRole, LogicProfile
    from lofi.lf.models._algebra import TruthAlgebra
    from lofi.lf.models._structure import Structure
    from lofi.terms import Term, TermStore

__all__ = ["EvaluationError", "Evaluator", "ModelLanguage", "truth_type"]

_OUT_OF_FRAGMENT: Final = "out-of-fragment"


class EvaluationError(ValueError):
    """Report a term that lies outside the evaluator's declared fragment."""

    __slots__ = ("cause",)

    def __init__(self, cause: str, message: str) -> None:
        self.cause = cause
        super().__init__(message)


class _Scheme(Protocol):
    """Expose the type information finite-model evaluation needs."""

    @property
    def quantified(self) -> tuple[int, ...]: ...

    @property
    def body(self) -> Term: ...


class ModelLanguage(Protocol):
    """Expose the declared-language capabilities used by finite models."""

    @property
    def store(self) -> TermStore: ...

    @property
    def profile(self) -> LogicProfile: ...

    @property
    def function_type(self) -> str: ...

    @property
    def term_sort(self) -> str: ...

    def scheme(self, constant: str, /) -> _Scheme: ...

    def constant(self, name: str, /) -> Term: ...

    def apply(self, function: Term, *arguments: Term) -> Term: ...

    def spine(self, term: Term, /) -> tuple[Term, tuple[Term, ...]]: ...

    def type_of(self, term: Term, /) -> Term: ...

    def print_type(self, term: Term, /) -> str: ...


def _result_type(language: ModelLanguage, type_: Term, /) -> Term:
    """Return a type's final result after its declared function spines."""
    result = type_

    while result.op.name == language.function_type and len(result.args) == 2:
        result = result.args[1]

    return result


def truth_type(language: ModelLanguage, /) -> Term:
    """Derive the unique truth type from every declared logical constant."""
    results: set[Term] = set()

    for name in language.profile.logical_constants:
        scheme = language.scheme(name)
        result = _result_type(language, scheme.body)

        if not result.op.is_constant:
            if scheme.quantified:
                continue

            written = language.print_type(result)
            msg = f"logical constant {name!r} has no ground result type: {written!r}"
            raise EvaluationError(_OUT_OF_FRAGMENT, msg)

        result_name = language.store.payload_name(result)

        semantics = language.profile.logical_semantics.get(name)

        if (
            semantics is not None
            and result_name not in language.profile.proposition_types
        ):
            continue

        results.add(result)

    if len(results) != 1:
        written = tuple(language.print_type(result) for result in results)
        msg = f"logical constants do not determine one truth type: {written!r}"
        raise EvaluationError(_OUT_OF_FRAGMENT, msg)

    return next(iter(results))


def _as_truth(value: ModelValue, /) -> float:
    """Return a model value as a unit truth degree."""
    if isinstance(value, bool):
        return float(value)

    if isinstance(value, float) and 0.0 <= value <= 1.0:
        return value

    msg = f"a formula denotes a truth degree, not {value!r}"
    raise EvaluationError(_OUT_OF_FRAGMENT, msg)


def _function(value: ModelValue, /) -> FunctionValue:
    """Return a finite function value or refuse the occurrence."""
    if isinstance(value, FunctionValue):
        return value

    msg = f"an applied term denotes a finite function, not {value!r}"
    raise EvaluationError(_OUT_OF_FRAGMENT, msg)


def _truth_outputs(function: FunctionValue, /) -> tuple[float, ...]:
    """Return a finite predicate's known truth values or refuse unknown facts."""
    try:
        return tuple(_as_truth(function(value)) for value in function.domain)
    except ValueError as error:
        raise EvaluationError(_OUT_OF_FRAGMENT, str(error)) from error


@final
class Evaluator:
    """Evaluate closed terms in one finite structure and truth algebra."""

    __slots__ = ("_algebra", "_language", "_structure")

    def __init__(
        self,
        language: ModelLanguage,
        structure: Structure,
        algebra: TruthAlgebra,
        /,
    ) -> None:
        self._language = language
        self._structure = structure
        self._algebra = algebra

    def evaluate(self, term: Term, /) -> ModelValue:
        """Return the denotation of one closed, well-typed term."""
        if not self._language.store.contains(term):
            msg = "an evaluated term belongs to the language's store lineage"
            raise ValueError(msg)

        return self._visit(term, {})

    def truth(self, term: Term, /) -> float:
        """Return one formula's truth degree."""
        return _as_truth(self.evaluate(term))

    def theory_holds(self, *, floor: float) -> bool:
        """Return whether every declared postulate meets a truth floor."""
        return all(
            self.truth(sentence) >= floor for sentence in self._language.profile.theory
        )

    def _visit(
        self,
        term: Term,
        environment: Mapping[str, tuple[ModelValue, ...]],
    ) -> ModelValue:
        if term.op.is_bound_variable:
            return self._bound(term, environment)

        if term.op.is_free_variable:
            msg = "a finite structure does not assign an open variable"
            raise EvaluationError(_OUT_OF_FRAGMENT, msg)

        head, arguments = self._language.spine(term)

        if arguments:
            return self._applied(head, arguments, environment)

        if term.op.is_constant:
            return self._constant(term)

        if term.op.is_binder:
            return self._abstraction(term, environment)

        msg = f"operator {term.op.name!r} has no finite-model interpretation"
        raise EvaluationError(_OUT_OF_FRAGMENT, msg)

    def _applied(
        self,
        head: Term,
        arguments: tuple[Term, ...],
        environment: Mapping[str, tuple[ModelValue, ...]],
    ) -> ModelValue:
        """Evaluate one application spine."""
        if head.op.is_constant:
            name = self._language.store.payload_name(head)
            definition = self._language.profile.definitions.get(name)

            if definition is not None:
                function_or_value = self._visit(definition, environment)

                for argument in arguments:
                    try:
                        function_or_value = _function(function_or_value)(
                            self._visit(argument, environment)
                        )
                    except ValueError as error:
                        raise EvaluationError(_OUT_OF_FRAGMENT, str(error)) from error

                return function_or_value

            semantics = self._language.profile.logical_semantics.get(name)

            if semantics is not None:
                values = tuple(
                    self._visit(argument, environment) for argument in arguments
                )

                try:
                    return semantics.denote(values, self._algebra, self._structure)
                except (TypeError, ValueError) as error:
                    raise EvaluationError(_OUT_OF_FRAGMENT, str(error)) from error

            role = self._language.profile.logical_roles.get(name)

            if role is not None:
                return self._logical(role, arguments, environment)

        function_or_value = self._visit(head, environment)

        for argument in arguments:
            try:
                function_or_value = _function(function_or_value)(
                    self._visit(argument, environment)
                )
            except ValueError as error:
                raise EvaluationError(_OUT_OF_FRAGMENT, str(error)) from error

        return function_or_value

    def _constant(self, term: Term, /) -> ModelValue:
        """Evaluate one unapplied nonlogical constant."""
        if term.payload is None:
            msg = f"constant operator {term.op.name!r} has no identifier"
            raise EvaluationError(_OUT_OF_FRAGMENT, msg)

        name = self._language.store.payload_name(term)

        if name in self._language.profile.logical_constants:
            msg = f"logical constant {name!r} is not fully applied"
            raise EvaluationError(_OUT_OF_FRAGMENT, msg)

        try:
            return self._structure.denotation(term.payload)
        except ValueError as error:
            raise EvaluationError(_OUT_OF_FRAGMENT, str(error)) from error

    def _bound(
        self,
        term: Term,
        environment: Mapping[str, tuple[ModelValue, ...]],
    ) -> ModelValue:
        if term.payload is None:
            msg = "a bound-variable occurrence has no per-sort index"
            raise EvaluationError(_OUT_OF_FRAGMENT, msg)

        values = environment.get(term.op.sort, ())

        if term.payload >= len(values):
            msg = f"bound index {term.payload} is outside its {term.op.sort!r} scope"
            raise EvaluationError(_OUT_OF_FRAGMENT, msg)

        return values[term.payload]

    def _abstraction(
        self,
        term: Term,
        environment: Mapping[str, tuple[ModelValue, ...]],
    ) -> FunctionValue:
        if len(term.op.binding) != 1:
            msg = f"abstraction {term.op.name!r} binds {len(term.op.binding)} variables"
            raise EvaluationError(_OUT_OF_FRAGMENT, msg)

        site = term.op.binding[0]

        if len(site.arguments) != 1:
            msg = f"abstraction {term.op.name!r} has {len(site.arguments)} bodies"
            raise EvaluationError(_OUT_OF_FRAGMENT, msg)

        if site.annotation is None:
            wrapped = term
            wrappers = term.maxidx

            for at in range(wrappers):
                wrapped = self._language.store.bind(
                    term.op.name,
                    f"\N{NULL}lofi.model.scope:{at}",
                    wrapped,
                )

            abstraction_type = self._language.type_of(wrapped)

            for _ in range(wrappers):
                if (
                    abstraction_type.op.name != self._language.function_type
                    or len(abstraction_type.args) != 2
                ):
                    msg = "an enclosed abstraction has no function result type"
                    raise EvaluationError(_OUT_OF_FRAGMENT, msg)

                abstraction_type = abstraction_type.args[1]

            if (
                abstraction_type.op.name != self._language.function_type
                or len(abstraction_type.args) != 2
            ):
                msg = "an abstraction's principal type is not functional"
                raise EvaluationError(_OUT_OF_FRAGMENT, msg)

            variable_type = abstraction_type.args[0]
        else:
            variable_type = term.args[site.annotation]

        try:
            domain = self._structure.frame.domain(variable_type).values
        except FrameTypeError as error:
            raise EvaluationError(_OUT_OF_FRAGMENT, str(error)) from error
        body = term.args[site.arguments[0]]
        outer = environment.get(site.sort, ())
        outputs = tuple(
            self._visit(body, {**environment, site.sort: (value, *outer)})
            for value in domain
        )
        return FunctionValue(domain, outputs, (True,) * len(domain))

    def _logical(
        self,
        role: LogicalRole,
        arguments: tuple[Term, ...],
        environment: Mapping[str, tuple[ModelValue, ...]],
    ) -> ModelValue:
        if len(arguments) != role.arity:
            msg = f"logical role {role.name!r} needs {role.arity} arguments"
            raise EvaluationError(_OUT_OF_FRAGMENT, msg)

        values = tuple(self._visit(argument, environment) for argument in arguments)

        if role.truth_functional:
            return role.fuzzy(
                tuple(_as_truth(value) for value in values), self._algebra
            )

        match role.model_operation:
            case "equality":
                return self._algebra.equal(values[0], values[1])
            case "universal":
                function = _function(values[0])
                return self._algebra.universal(_truth_outputs(function))
            case "existential":
                function = _function(values[0])
                return self._algebra.existential(_truth_outputs(function))
            case "restricted-universal":
                restrictor = _function(values[0])
                scope = _function(values[1])

                if restrictor.domain != scope.domain:
                    msg = "a restricted quantifier's two predicates share a domain"
                    raise EvaluationError(_OUT_OF_FRAGMENT, msg)

                return self._algebra.universal(
                    tuple(
                        self._algebra.conditional(
                            _as_truth(restrictor(value)), _as_truth(scope(value))
                        )
                        for value in restrictor.domain
                    )
                )
            case None:
                msg = f"logical role {role.name!r} declares no model operation"
                raise EvaluationError(_OUT_OF_FRAGMENT, msg)
