"""Crisp countermodel search over complete finite frames."""

from __future__ import annotations

from math import prod
from random import Random
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, final

from lofi._checks import ensure_name, ensure_non_negative, ensure_positive
from lofi._frozen import FrozenValue
from lofi.lf._verdict import Countermodel
from lofi.lf.models._algebra import CRISP, EXTREME, GOEDEL, MATERIAL, TruthAlgebra
from lofi.lf.models._evaluate import (
    EvaluationError,
    Evaluator,
    ModelLanguage,
    truth_type,
)
from lofi.lf.models._frame import Frame, ModelValue
from lofi.lf.models._structure import Structure

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.terms import Term

__all__ = ["CountermodelFilter", "CountermodelSettings"]


@final
class CountermodelSettings(FrozenValue):
    """Resolved settings for one bounded countermodel search."""

    _noun = "countermodel settings"
    __slots__ = ("_cap", "_carriers", "_method", "_seed", "_structures")

    def __init__(
        self,
        method: Literal["random", "exhaustive"],
        /,
        *,
        carriers: Mapping[str, int],
        structures: int,
        seed: int,
        full_function_space_cap: int,
    ) -> None:
        if method not in ("random", "exhaustive"):
            msg = f"unknown countermodel method {method!r}"
            raise ValueError(msg)

        held_carriers = {
            ensure_name("a countermodel carrier", name): ensure_positive(
                f"countermodel carrier {name!r} size", size
            )
            for name, size in carriers.items()
        }

        if not held_carriers:
            msg = "a countermodel search declares at least one atomic carrier"
            raise ValueError(msg)

        object.__setattr__(self, "_method", method)
        object.__setattr__(self, "_carriers", MappingProxyType(held_carriers))
        object.__setattr__(
            self, "_structures", ensure_positive("countermodel structures", structures)
        )
        object.__setattr__(
            self, "_seed", ensure_non_negative("countermodel seed", seed)
        )
        object.__setattr__(
            self,
            "_cap",
            ensure_positive("the full function-space cap", full_function_space_cap),
        )

    @property
    def method(self) -> Literal["random", "exhaustive"]:
        return self._method

    @property
    def carriers(self) -> Mapping[str, int]:
        return self._carriers

    @property
    def structures(self) -> int:
        return self._structures

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def full_function_space_cap(self) -> int:
        return self._cap


def _type_subterms(type_: Term, /) -> tuple[Term, ...]:
    return (type_, *(child for arg in type_.args for child in _type_subterms(arg)))


def _decode_assignment(
    index: int,
    axes: tuple[tuple[int, tuple[ModelValue, ...]], ...],
    /,
) -> dict[int, ModelValue]:
    denotations: dict[int, ModelValue] = {}

    for identifier, values in axes:
        denotations[identifier] = values[index % len(values)]
        index //= len(values)

    return denotations


@final
class _CrispStructure:
    """Expose checked Boolean satisfaction for one finite structure."""

    __slots__ = ("_evaluator", "_structure")

    def __init__(
        self,
        language: ModelLanguage,
        structure: Structure,
        algebra: TruthAlgebra,
        /,
    ) -> None:
        self._structure = structure
        self._evaluator = Evaluator(language, structure, algebra)

    @property
    def identity(self) -> str:
        return self._structure.name

    @property
    def rendered(self) -> str:
        values = tuple(sorted(self._structure.denotations.items()))
        return f"{self._structure.name}:{values!r}"

    def holds(self, formula: Term, /) -> bool:
        return self._evaluator.truth(formula) == 1.0


@final
class CountermodelFilter:
    """Find certified finite countermodels before invoking a prover."""

    __slots__ = (
        "_algebra",
        "_cap",
        "_frame",
        "_language",
        "_negation",
        "_structures",
    )

    def __init__(
        self,
        language: ModelLanguage,
        settings: CountermodelSettings,
        /,
    ) -> None:
        self._language = language
        self._algebra = GOEDEL.build(
            implication=MATERIAL,
            quantifier=EXTREME,
            equality=CRISP,
            margin=0.0,
        )
        self._cap = settings.full_function_space_cap
        self._negation = self._negation_name(language)
        self._frame, self._structures = self._build_structures(language, settings)

    @property
    def structures(self) -> tuple[Structure, ...]:
        """Return the complete-frame, profile-theory models searched."""
        return self._structures

    def refute(
        self,
        premise: Term,
        goal: Term,
        /,
        *,
        theory: tuple[Term, ...] = (),
    ) -> Countermodel | None:
        """Return the first checked countermodel, or none when none was found."""
        obligations = (*self._language.profile.theory, *theory)

        if not self._full_binder_domains((premise, goal, *obligations)):
            return None

        negated_goal = self._language.apply(
            self._language.constant(self._negation), goal
        )

        for structure in self._structures:
            checked = _CrispStructure(self._language, structure, self._algebra)

            try:
                if not all(checked.holds(sentence) for sentence in obligations):
                    continue

                if checked.holds(premise) and checked.holds(negated_goal):
                    return Countermodel(
                        checked,
                        (premise,),
                        negated_goal,
                        theory=obligations,
                    )
            except ValueError:
                continue

        return None

    def _full_binder_domains(self, formulas: tuple[Term, ...], /) -> bool:
        """Return whether every inferable bound domain is a complete frame."""
        pending = list(formulas)

        while pending:
            term = pending.pop()
            pending.extend(term.args)

            if not term.op.is_binder:
                continue

            try:
                type_ = self._language.type_of(term)
            except ValueError:
                return False

            try:
                domain = self._frame.domain(type_)
            except ValueError:
                return False

            if not domain.full:
                rendered = self._language.print_type(type_)
                msg = (
                    f"countermodel binder type {rendered!r} exceeds "
                    f"full_function_space_cap={self._cap}"
                )
                raise ValueError(msg)

        return True

    @staticmethod
    def _negation_name(language: ModelLanguage, /) -> str:
        name = language.profile.negation

        if name is None:
            msg = "countermodel search needs a designated negation"
            raise ValueError(msg)

        return name

    def _build_structures(
        self,
        language: ModelLanguage,
        settings: CountermodelSettings,
        /,
    ) -> tuple[Frame, tuple[Structure, ...]]:
        truth = truth_type(language)
        truth_name = language.store.payload_name(truth)
        frame = Frame(
            language.store,
            function_type=language.function_type,
            truth_type=truth_name,
            carriers=settings.carriers,
            full_function_space_cap=settings.full_function_space_cap,
            henkin_pool=settings.full_function_space_cap,
            seed=settings.seed,
        )
        vocabulary = language.store.signature.vocabulary(language.term_sort)
        logical = language.profile.logical_constants
        axes: list[tuple[int, tuple[ModelValue, ...]]] = []

        for identifier, name in enumerate(vocabulary.names):
            scheme = language.scheme(name)

            if scheme.quantified:
                if name not in logical:
                    msg = (
                        "countermodel search cannot assign polymorphic nonlogical "
                        f"constant {name!r}"
                    )
                    raise ValueError(msg)
                continue

            for type_ in _type_subterms(scheme.body):
                domain = frame.domain(type_)

                if not domain.full:
                    rendered = language.print_type(type_)
                    msg = (
                        f"countermodel type {rendered!r} exceeds "
                        f"full_function_space_cap={settings.full_function_space_cap}"
                    )
                    raise ValueError(msg)

            if name not in logical:
                axes.append((identifier, frame.domain(scheme.body).values))

        held_axes = tuple(axes)
        total = prod(len(values) for _, values in held_axes)
        count = min(settings.structures, total)

        if settings.method == "exhaustive":
            positions = tuple(range(count))
        else:
            positions = tuple(sorted(Random(settings.seed).sample(range(total), count)))

        accepted: list[Structure] = []

        for position in positions:
            structure = Structure(
                f"countermodel-{settings.method}-{position}",
                frame,
                _decode_assignment(position, held_axes),
            )
            evaluator = Evaluator(language, structure, self._algebra)

            try:
                if evaluator.theory_holds(floor=1.0):
                    accepted.append(structure)
            except EvaluationError:
                continue

        return frame, tuple(accepted)
