"""Finite structures fitted only to a language's declared theory."""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING, Final, NamedTuple, assert_never, final, override

import torch
from torch import nn

from lofi._checks import ensure_finite
from lofi.lf.models import (
    Arithmetic,
    Element,
    Evaluator,
    Frame,
    FunctionValue,
    ModelLanguage,
    Structure,
    StructureSource,
    StructureSourceFactory,
    TruthAlgebra,
    truth_type,
)
from lofi.spec.structures import LearnedSettings

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.lf import LogicalRole
    from lofi.lf.models import ModelValue
    from lofi.terms import Term

__all__ = ["LEARNED", "LearnedStructureSource"]


def _seed(*parts: object) -> int:
    digest = sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


type _DomainValue = Element | FunctionValue


def _domain(frame: Frame, type_: Term, /) -> tuple[_DomainValue, ...]:
    values = frame.domain(type_).values

    if any(not isinstance(value, (Element, FunctionValue)) for value in values):
        msg = "a learned function domain contains entities or finite functions"
        raise ValueError(msg)

    return tuple(
        value for value in values if isinstance(value, (Element, FunctionValue))
    )


class _Distribution(NamedTuple):
    """A relaxed choice over one atomic carrier."""

    domain: tuple[_DomainValue, ...]
    probabilities: Tensor


class _Function(NamedTuple):
    """A differentiable function table over a finite domain."""

    domain: tuple[_DomainValue, ...]
    outputs: tuple[_Value, ...]


type _Value = Tensor | _DomainValue | _Distribution | _Function


def _truth(value: _Value, /) -> Tensor:
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return value

    msg = f"a learned formula denotes scalar truth, not {value!r}"
    raise ValueError(msg)


@final
class _TensorArithmetic:
    """Pointwise operations on scalar truth tensors."""

    def constant(self, value: float, like: Tensor, /) -> Tensor:
        return like.new_tensor(value)

    def add(self, left: Tensor, right: Tensor, /) -> Tensor:
        return left + right

    def subtract(self, left: Tensor, right: Tensor, /) -> Tensor:
        return left - right

    def multiply(self, left: Tensor, right: Tensor, /) -> Tensor:
        return left * right

    def divide(self, left: Tensor, right: Tensor, /) -> Tensor:
        return left / right

    def minimum(self, left: Tensor, right: Tensor, /) -> Tensor:
        return torch.minimum(left, right)

    def maximum(self, left: Tensor, right: Tensor, /) -> Tensor:
        return torch.maximum(left, right)

    def where_at_most(
        self, left: Tensor, right: Tensor, then: Tensor, otherwise: Tensor, /
    ) -> Tensor:
        return torch.where(left <= right, then, otherwise)


_TENSOR: Final[Arithmetic[Tensor]] = _TensorArithmetic()


def _and(algebra: TruthAlgebra, left: Tensor, right: Tensor, /) -> Tensor:
    return algebra.conjoin(_TENSOR, left, right)


def _or(algebra: TruthAlgebra, left: Tensor, right: Tensor, /) -> Tensor:
    return algebra.disjoin(_TENSOR, left, right)


def _conditional(algebra: TruthAlgebra, left: Tensor, right: Tensor, /) -> Tensor:
    return algebra.imply(_TENSOR, left, right)


def _connective(
    algebra: TruthAlgebra,
    table: tuple[bool, ...],
    arguments: tuple[Tensor, ...],
    /,
) -> Tensor:
    if len(table) != 2 ** len(arguments):
        msg = f"a {len(arguments)}-ary connective needs {2 ** len(arguments)} rows"
        raise ValueError(msg)

    if len(arguments) == 1 and table == (True, False):
        return 1.0 - arguments[0]

    if len(arguments) == 2:
        left, right = arguments

        if table == (False, False, False, True):
            return _and(algebra, left, right)

        if table == (False, True, True, True):
            return _or(algebra, left, right)

        if table == (True, True, False, True):
            return _conditional(algebra, left, right)

        if table == (True, False, False, True):
            return _and(
                algebra,
                _conditional(algebra, left, right),
                _conditional(algebra, right, left),
            )

    result = arguments[0].new_zeros(())

    for row, admitted in enumerate(table):
        if not admitted:
            continue

        degree = arguments[0].new_ones(())

        for position, value in enumerate(arguments):
            bit = bool(row & (1 << (len(arguments) - position - 1)))
            degree = _and(algebra, degree, value if bit else 1.0 - value)

        result = _or(algebra, result, degree)

    return result


def _aggregate(
    algebra: TruthAlgebra,
    values: tuple[Tensor, ...],
    /,
    *,
    universal: bool,
) -> Tensor:
    if not values:
        return torch.tensor(1.0 if universal else 0.0)

    if universal:
        return algebra.every(_TENSOR, values)

    return algebra.some(_TENSOR, values)


def _blend(values: tuple[_Value, ...], weights: Tensor, /) -> _Value:
    first = values[0]

    if isinstance(first, torch.Tensor):
        tensors = tuple(_truth(value) for value in values)
        return torch.sum(weights * torch.stack(tensors))

    if isinstance(first, _Distribution):
        if any(
            not isinstance(value, _Distribution) or value.domain != first.domain
            for value in values
        ):
            msg = "a relaxed function's atomic outputs share one carrier"
            raise ValueError(msg)

        probabilities = torch.stack(
            tuple(
                value.probabilities
                for value in values
                if isinstance(value, _Distribution)
            )
        )
        return _Distribution(
            first.domain, torch.sum(weights[:, None] * probabilities, dim=0)
        )

    if isinstance(first, _Function):
        if any(
            not isinstance(value, _Function) or value.domain != first.domain
            for value in values
        ):
            msg = "a relaxed function's functional outputs share one domain"
            raise ValueError(msg)

        functions = tuple(value for value in values if isinstance(value, _Function))
        return _Function(
            first.domain,
            tuple(
                _blend(
                    tuple(function.outputs[position] for function in functions), weights
                )
                for position in range(len(first.domain))
            ),
        )

    if all(type(value) is type(first) and value == first for value in values):
        return first

    msg = "a relaxed application cannot mix discrete functional outputs"
    raise ValueError(msg)


def _call(function: _Value, argument: _Value, /) -> _Value:
    if not isinstance(function, _Function):
        msg = f"an applied learned term denotes a finite function, not {function!r}"
        raise TypeError(msg)

    if isinstance(argument, _Distribution):
        if argument.domain != function.domain:
            msg = "a relaxed argument distribution matches its function domain"
            raise ValueError(msg)

        return _blend(function.outputs, argument.probabilities)

    try:
        position = function.domain.index(argument)
    except ValueError:
        msg = f"{argument!r} lies outside the learned function domain"
        raise ValueError(msg) from None

    return function.outputs[position]


@final
class _Evaluator:
    """Evaluate theory terms directly over relaxed denotations."""

    __slots__ = ("_algebra", "_frame", "_language", "_source", "_structure")

    def __init__(
        self,
        source: LearnedStructureSource,
        structure: int,
        language: ModelLanguage,
        frame: Frame,
        algebra: TruthAlgebra,
    ) -> None:
        self._source = source
        self._structure = structure
        self._language = language
        self._frame = frame
        self._algebra = algebra

    def truth(self, term: Term, /) -> Tensor:
        """Return relaxed truth for one closed theory term."""
        return _truth(self._visit(term, {}))

    def _visit(
        self,
        term: Term,
        environment: Mapping[str, tuple[_Value, ...]],
    ) -> _Value:
        if term.op.is_bound_variable:
            if term.payload is None:
                msg = "a bound variable has no per-sort index"
                raise ValueError(msg)

            values = environment.get(term.op.sort, ())

            if term.payload >= len(values):
                msg = f"bound index {term.payload} is outside its scope"
                raise ValueError(msg)

            return values[term.payload]

        if term.op.is_free_variable:
            msg = "a learned structure cannot fit an open theory term"
            raise ValueError(msg)

        head, arguments = self._language.spine(term)

        if arguments:
            return self._applied(head, arguments, environment)

        if term.op.is_constant:
            if term.payload is None:
                msg = "a constant has no vocabulary identifier"
                raise ValueError(msg)

            return self._source.relaxed_value(
                self._structure,
                term.payload,
                self._language.type_of(term),
                self._frame,
            )

        if term.op.is_binder:
            return self._abstraction(term, environment)

        msg = f"operator {term.op.name!r} has no learned-model interpretation"
        raise ValueError(msg)

    def _applied(
        self,
        head: Term,
        arguments: tuple[Term, ...],
        environment: Mapping[str, tuple[_Value, ...]],
    ) -> _Value:
        if head.op.is_constant:
            name = self._language.store.payload_name(head)
            definition = self._language.profile.definitions.get(name)

            if definition is not None:
                application = self._language.store.signature.by_name(
                    self._language.profile.application
                )
                expanded = definition

                for argument in arguments:
                    expanded = self._language.store.op(application, expanded, argument)

                return self._visit(expanded, environment)

            if name in self._language.profile.logical_semantics:
                msg = (
                    f"learned structures do not differentiate postulated semantics "
                    f"plugin {name!r}"
                )
                raise ValueError(msg)

            role = self._language.profile.logical_roles.get(name)

            if role is not None:
                return self._logical(role, arguments, environment)

        result = self._visit(head, environment)

        for argument in arguments:
            result = _call(result, self._visit(argument, environment))

        return result

    def _abstraction(
        self,
        term: Term,
        environment: Mapping[str, tuple[_Value, ...]],
    ) -> _Function:
        if len(term.op.binding) != 1:
            msg = "a learned abstraction binds exactly one variable"
            raise ValueError(msg)

        site = term.op.binding[0]

        if len(site.arguments) != 1:
            msg = "a learned abstraction has exactly one body"
            raise ValueError(msg)

        if site.annotation is None:
            abstraction_type = self._language.type_of(term)

            if (
                abstraction_type.op.name != self._language.function_type
                or len(abstraction_type.args) != 2
            ):
                msg = "a learned abstraction's type is functional"
                raise ValueError(msg)

            variable_type = abstraction_type.args[0]
        else:
            variable_type = term.args[site.annotation]

        domain = _domain(self._frame, variable_type)
        body = term.args[site.arguments[0]]
        outer = environment.get(site.sort, ())
        return _Function(
            domain,
            tuple(
                self._visit(body, {**environment, site.sort: (value, *outer)})
                for value in domain
            ),
        )

    def _logical(
        self,
        role: LogicalRole,
        arguments: tuple[Term, ...],
        environment: Mapping[str, tuple[_Value, ...]],
    ) -> _Value:
        if len(arguments) != role.arity:
            msg = f"logical role {role.name!r} needs {role.arity} arguments"
            raise ValueError(msg)

        values = tuple(self._visit(argument, environment) for argument in arguments)

        if role.truth_functional:
            table = role.truth_table

            if table is None:
                msg = f"truth-functional role {role.name!r} has no table"
                raise ValueError(msg)

            return _connective(
                self._algebra,
                table,
                tuple(_truth(value) for value in values),
            )

        match role.model_operation:
            case "equality":
                return self._equal(values[0], values[1])
            case "universal" | "existential" as operation:
                function = values[0]

                if not isinstance(function, _Function):
                    msg = f"{operation} reads a finite predicate"
                    raise TypeError(msg)

                return _aggregate(
                    self._algebra,
                    tuple(_truth(value) for value in function.outputs),
                    universal=operation == "universal",
                )
            case "restricted-universal":
                restrictor, scope = values

                if (
                    not isinstance(restrictor, _Function)
                    or not isinstance(scope, _Function)
                    or restrictor.domain != scope.domain
                ):
                    msg = "a restricted universal reads predicates on one domain"
                    raise ValueError(msg)

                return _aggregate(
                    self._algebra,
                    tuple(
                        _conditional(
                            self._algebra,
                            _truth(left),
                            _truth(right),
                        )
                        for left, right in zip(
                            restrictor.outputs, scope.outputs, strict=True
                        )
                    ),
                    universal=True,
                )
            case None:
                msg = f"logical role {role.name!r} declares no model operation"
                raise ValueError(msg)
            case unreachable:
                assert_never(unreachable)

    def _equal(self, left: _Value, right: _Value, /) -> Tensor:
        if isinstance(left, _Distribution) and isinstance(right, _Distribution):
            if left.domain != right.domain:
                msg = "equality compares choices over one carrier"
                raise ValueError(msg)

            return torch.dot(left.probabilities, right.probabilities)

        if isinstance(left, _Distribution):
            try:
                position = left.domain.index(right)
            except ValueError:
                return left.probabilities.new_zeros(())

            return left.probabilities[position]

        if isinstance(right, _Distribution):
            return self._equal(right, left)

        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return 1.0 - torch.abs(_truth(left) - _truth(right))

        like = next(iter(self._source.parameters()))
        return like.new_tensor(float(left == right))


@final
class LearnedStructureSource(nn.Module):
    """Fit relaxed denotations to theory, then score through detached structures."""

    def __init__(
        self,
        language: ModelLanguage,
        algebra: TruthAlgebra,
        /,
        *,
        structures: int,
        carriers: Mapping[str, int],
        full_function_space_cap: int,
        henkin_pool: int,
        denotations: str,
        seed: int,
        initialization_scale: float,
        theory_floor: float,
    ) -> None:
        super().__init__()
        floor = ensure_finite("a theory floor", theory_floor)

        if not 0.0 <= floor <= 1.0:
            msg = f"a theory floor lies in the unit interval, not {theory_floor!r}"
            raise ValueError(msg)

        self._theory_floor = floor
        self._language = language
        self._algebra = algebra
        self._denotations = denotations
        self._truth_type = truth_type(language)
        truth_name = language.store.payload_name(self._truth_type)
        self._frames = tuple(
            Frame(
                language.store,
                function_type=language.function_type,
                truth_type=truth_name,
                carriers=carriers,
                full_function_space_cap=full_function_space_cap,
                henkin_pool=henkin_pool,
                seed=_seed(seed, "learned-frame", position),
            )
            for position in range(structures)
        )
        self._logits = nn.ParameterDict()
        vocabulary = language.store.signature.vocabulary(language.term_sort)
        logical = language.profile.logical_constants

        for structure, frame in enumerate(self._frames):
            for identifier, name in enumerate(vocabulary.names):
                scheme = language.scheme(name)

                if name in logical or scheme.quantified:
                    continue

                width = self._width(scheme.body, frame)

                if width == 0:
                    continue

                generator = torch.Generator(device="cpu")
                generator.manual_seed(_seed(seed, structure, identifier))
                initialized = torch.empty(width, dtype=torch.float64)
                initialized.normal_(
                    mean=0.0, std=initialization_scale, generator=generator
                )
                self._logits[self._key(structure, identifier)] = nn.Parameter(
                    initialized
                )

        if not self._logits:
            msg = "a learned structure source has at least one learnable denotation"
            raise ValueError(msg)

        for structure, frame in enumerate(self._frames):
            evaluator = _Evaluator(self, structure, language, frame, algebra)

            for sentence in language.profile.theory:
                evaluator.truth(sentence)

    @property
    def name(self) -> str:
        """Return the registered source name."""
        return "learned"

    @property
    def evidence_label(self) -> str:
        """Name the relaxed evidence relation without claiming entailment."""
        return "truth in theory-fitted structures"

    @property
    def trainable(self) -> bool:
        """Return true because denotation logits are model parameters."""
        return True

    def structures(self) -> tuple[Structure, ...]:
        """Materialize detached finite structures for conditional scoring.

        A structure whose declared theory falls below the scorer's floor is not
        returned, as a sampled or given source refuses one, and the scorer's
        evidence counts the structures it read.
        """
        vocabulary = self._language.store.signature.vocabulary(self._language.term_sort)
        logical = self._language.profile.logical_constants
        held: list[Structure] = []

        with torch.no_grad():
            for position, frame in enumerate(self._frames):
                values: dict[int, ModelValue] = {}

                for identifier, name in enumerate(vocabulary.names):
                    scheme = self._language.scheme(name)

                    if name in logical or scheme.quantified:
                        continue

                    values[identifier] = self._model_value(
                        self.relaxed_value(position, identifier, scheme.body, frame)
                    )

                structure = Structure(f"learned-{position}", frame, values)

                if Evaluator(self._language, structure, self._algebra).theory_holds(
                    floor=self._theory_floor
                ):
                    held.append(structure)

        return tuple(held)

    def auxiliary(self) -> Tensor:
        """Return theory dissatisfaction with its sole path to structure logits."""
        truths = tuple(
            _Evaluator(self, position, self._language, frame, self._algebra).truth(
                sentence
            )
            for position, frame in enumerate(self._frames)
            for sentence in self._language.profile.theory
        )

        if truths:
            probabilities = torch.stack(truths).clamp_min(
                torch.finfo(truths[0].dtype).eps
            )
            return -probabilities.log().mean()

        return torch.stack(
            tuple(parameter.sum() * 0.0 for parameter in self.parameters())
        ).sum()

    def open_step(self) -> None:
        """Retain fitted parameters across optimization steps."""

    def _width(self, type_: Term, frame: Frame, /) -> int:
        if type_ == self._truth_type:
            return 1

        if type_.op.name == self._language.function_type and len(type_.args) == 2:
            return len(_domain(frame, type_.args[0])) * self._width(
                type_.args[1], frame
            )

        return len(frame.domain(type_).values) if self._denotations == "free" else 0

    def relaxed_value(
        self,
        structure: int,
        identifier: int,
        type_: Term,
        frame: Frame,
        /,
    ) -> _Value:
        key = self._key(structure, identifier)
        logits = self._logits.get(key)
        value, position = self._read(
            type_,
            frame,
            logits,
            0,
            identifier=identifier,
        )

        if logits is not None and position != logits.numel():
            msg = (
                f"learned denotation {key!r} consumed {position} of "
                f"{logits.numel()} logits"
            )
            raise RuntimeError(msg)

        return value

    def _read(
        self,
        type_: Term,
        frame: Frame,
        logits: Tensor | None,
        position: int,
        /,
        *,
        identifier: int,
    ) -> tuple[_Value, int]:
        if type_ == self._truth_type:
            if logits is None:
                msg = "a truth denotation has a learned logit"
                raise RuntimeError(msg)

            return torch.sigmoid(logits[position]), position + 1

        if type_.op.name == self._language.function_type and len(type_.args) == 2:
            domain = _domain(frame, type_.args[0])
            outputs: list[_Value] = []

            for _ in domain:
                output, position = self._read(
                    type_.args[1],
                    frame,
                    logits,
                    position,
                    identifier=identifier,
                )
                outputs.append(output)

            return _Function(domain, tuple(outputs)), position

        domain = _domain(frame, type_)

        if self._denotations == "free":
            if logits is None:
                msg = "a free atomic denotation has learned logits"
                raise RuntimeError(msg)

            width = len(domain)
            return (
                _Distribution(
                    domain, torch.softmax(logits[position : position + width], dim=0)
                ),
                position + width,
            )

        return domain[identifier % len(domain)], position

    def _model_value(self, value: _Value, /) -> ModelValue:
        if isinstance(value, torch.Tensor):
            return float(_truth(value).detach())

        if isinstance(value, _Distribution):
            return value.domain[int(value.probabilities.detach().argmax())]

        if isinstance(value, _Function):
            return FunctionValue(
                value.domain,
                tuple(self._model_value(output) for output in value.outputs),
                (True,) * len(value.domain),
            )

        return value

    @staticmethod
    def _key(structure: int, identifier: int, /) -> str:
        return f"structure_{structure}_constant_{identifier}"

    @override
    def extra_repr(self) -> str:
        return f"structures={len(self._frames)}, denotations={self._denotations!r}"


@final
class _LearnedFactory:
    """Build the learned source registered under its boundary tag."""

    def build(
        self,
        settings: object,
        language: ModelLanguage,
        algebra: TruthAlgebra,
        /,
        *,
        theory_floor: float,
    ) -> StructureSource:
        if not isinstance(settings, LearnedSettings):
            msg = f"the learned source reads LearnedSettings, not {settings!r}"
            raise TypeError(msg)

        return LearnedStructureSource(
            language,
            algebra,
            structures=settings.structures,
            carriers=settings.carriers,
            full_function_space_cap=settings.full_function_space_cap,
            henkin_pool=settings.henkin_pool,
            denotations=settings.denotations,
            seed=settings.seed,
            initialization_scale=settings.initialization_scale,
            theory_floor=theory_floor,
        )


LEARNED: Final[StructureSourceFactory] = _LearnedFactory()
