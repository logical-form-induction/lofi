"""Signature declarations and the native signatures they build.

A signature declaration lists the primitives of the category sort, any
further operators, and the category constructors. A constructor may declare
functor roles, and the binary functor view those roles describe is what the
functor-family rules are built over. Categories remain surface text until
the declaration builds native values.
"""

from typing import TYPE_CHECKING, Literal, Self

import didactic.api as dx

from lofi._frozen import FrozenValue
from lofi.grammars.ccg import CCGAssembly, FunctorView
from lofi.judge import Conjunctions
from lofi.spec._constructor import (
    ConstructorSpec,
)
from lofi.terms import (
    BindingSite,
    CategoryResolution,
    ConstructorSyntax,
    Operator,
    SExpSyntax,
    Signature,
    Term,
    Vocabulary,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.judge import NormalForm
    from lofi.terms import TermMatching, TermStore, TermSyntax

__all__ = [
    "BindingSiteSpec",
    "ConjunctionPairSpec",
    "ConjunctionsSpec",
    "FunctorSpec",
    "OperatorSpec",
    "PreparedSignatureRuntime",
    "SignatureSpec",
]


def _nonempty(field: str, message: str) -> dx.Axiom:
    """Return an axiom requiring a nonempty string field.

    Parameters
    ----------
    field : str
        The field name used in the expression.
    message : str
        The validation message.
    """
    return dx.axiom(f"len({field}) > 0", message=message)


def _all_named(field: str, message: str) -> dx.Axiom:
    """Return an axiom requiring every entry of a string container to be nonempty.

    Parameters
    ----------
    field : str
        The field name used in the expression.
    message : str
        The validation message.
    """
    return dx.axiom(f"all({field})", message=message)


def _first_repeated_name(names: tuple[str, ...], /) -> str | None:
    """Return the first name declared more than once, if any."""
    seen: set[str] = set()

    for name in names:
        if name in seen:
            return name

        seen.add(name)

    return None


class BindingSiteSpec(dx.Model, extra="forbid"):
    """One variable an operator binds.

    Attributes
    ----------
    arguments : tuple[int, ...]
        The argument positions the variable scopes over, strictly ascending.
    sort : str
        The sort of the bound variable.
    annotation : int | None
        The argument position holding the variable's classifier, or nothing.
    """

    arguments: tuple[int, ...]
    sort: str
    annotation: int | None = None

    __axioms__ = (
        dx.axiom(
            "len(arguments) > 0", message="a binding site scopes over an argument"
        ),
        _nonempty("sort", "a bound variable needs a sort"),
    )

    def build(self) -> BindingSite:
        """Build the declared binding site."""
        return BindingSite(self.arguments, self.sort, self.annotation)


class OperatorSpec(dx.Model, extra="forbid"):
    """A term-algebra operator declaration."""

    name: str
    sort: str
    arg_sorts: tuple[str, ...] = ()
    binding: tuple[BindingSiteSpec, ...] = ()
    family: str = "primitive"
    erases_to: str | None = None

    __axioms__ = (
        _nonempty("name", "an operator needs a name"),
        _nonempty("sort", "an operator needs a result sort"),
        _all_named("arg_sorts", "an argument sort needs a name"),
        _nonempty("family", "an operator needs a family"),
        dx.axiom(
            "isNone(erases_to) or len(erases_to) > 0",
            message="an erasure target needs a name",
        ),
    )

    @classmethod
    def from_operator(cls, operator: Operator, /) -> OperatorSpec:
        """Return the declaration of a declared native operator.

        Parameters
        ----------
        operator : Operator
            The native operator, which carries no payload.
        """
        return cls(
            name=operator.name,
            sort=operator.sort,
            arg_sorts=operator.arg_sorts,
            binding=tuple(
                BindingSiteSpec(
                    arguments=site.arguments,
                    sort=site.sort,
                    annotation=site.annotation,
                )
                for site in operator.binding
            ),
            family=operator.family,
            erases_to=operator.erases_to,
        )

    def build(self) -> Operator:
        """Build the declared operator."""
        return Operator(
            self.name,
            self.sort,
            arg_sorts=self.arg_sorts,
            binding=tuple(site.build() for site in self.binding),
            family=self.family,
            erases_to=self.erases_to,
        )


class ConjunctionPairSpec(dx.Model, extra="forbid"):
    """An unmarked category and its conjunction-marked counterpart."""

    unmarked: str
    marked: str

    __axioms__ = (
        _nonempty("unmarked", "a conjunction pair needs an unmarked category"),
        _nonempty("marked", "a conjunction pair needs a marked category"),
    )


class ConjunctionsSpec(dx.Model, extra="forbid"):
    """The conjunction inventory coordination runs over.

    ``pairs`` names finite unmarked and marked category pairs. ``marking``,
    when declared, names a unary category constructor whose application to
    any category is that category's marked counterpart, so coordination also
    marks and merges categories no pair names, such as raised and composed
    ones.
    """

    name: str = "conjunctions"
    markers: tuple[str, ...]
    pairs: tuple[ConjunctionPairSpec, ...]
    marking: str | None = None

    __axioms__ = (
        _nonempty("name", "a conjunction inventory needs a name"),
        dx.axiom("len(markers) > 0", message="a conjunction inventory needs a marker"),
        _all_named("markers", "a conjunction marker needs a name"),
        dx.axiom("len(pairs) > 0", message="a conjunction inventory needs a pair"),
    )


class FunctorSpec(dx.Model, extra="forbid"):
    """One named functor view over a declared binary constructor."""

    name: str
    constructor: str
    argument_side: Literal["left", "right", "either"]
    opposite: str
    result: int = 0
    argument: int = 1

    __axioms__ = (
        _nonempty("name", "a functor view needs a name"),
        _nonempty("constructor", "a functor view needs a constructor"),
        _nonempty("opposite", "a functor view needs an opposite"),
        dx.axiom("result >= 0 and result < 2", message="a result slot is 0 or 1"),
        dx.axiom(
            "argument >= 0 and argument < 2", message="an argument slot is 0 or 1"
        ),
        dx.axiom(
            "result != argument",
            message="a functor's result and argument sit at different slots",
        ),
    )


class PreparedSignatureRuntime(FrozenValue):
    """Category syntax and functor metadata fixed at validation."""

    _noun = "prepared signature runtime"
    __slots__ = (
        "_category_sort",
        "_conjunctions",
        "_feature_delimiter",
        "_functors",
        "_notations",
        "_syntax",
    )

    def __init__(self, source: SignatureSpec, /) -> None:
        notations = [
            constructor.build_notation() for constructor in source.constructors
        ]

        conjunctions = source.conjunctions
        object.__setattr__(self, "_feature_delimiter", source.feature_delimiter)
        object.__setattr__(self, "_syntax", source.syntax)
        object.__setattr__(self, "_notations", tuple(notations))
        object.__setattr__(
            self,
            "_functors",
            tuple(
                (
                    view.name,
                    view.constructor,
                    view.argument_side,
                    view.opposite,
                    view.result,
                    view.argument,
                )
                for view in source.functors
            ),
        )
        object.__setattr__(
            self,
            "_conjunctions",
            (
                None
                if conjunctions is None
                else (
                    conjunctions.name,
                    conjunctions.markers,
                    tuple((pair.unmarked, pair.marked) for pair in conjunctions.pairs),
                    conjunctions.marking,
                )
            ),
        )
        object.__setattr__(self, "_category_sort", source.category_sort)

    def syntax(self, store: TermStore, /) -> TermSyntax:
        """Build the declared category syntax over one term store."""
        return (
            ConstructorSyntax(
                store.signature,
                self._notations,
                feature_delimiter=self._feature_delimiter,
            )
            if self._syntax == "declared"
            else SExpSyntax(store.signature)
        )

    def build(
        self,
        store: TermStore,
        syntax: TermSyntax,
        resolve_category: CategoryResolution,
        matching: TermMatching,
        normal_form: NormalForm[Term],
        rule_matchings: Sequence[tuple[str, TermMatching]] = (),
        /,
        *,
        categories: Sequence[str] = (),
    ) -> CCGAssembly:
        """Build the rule context from one category-resolution capability."""
        roles = {name: row for name, *row in self._functors}
        built: dict[tuple[str, str, int, int], FunctorView] = {}
        selected: dict[str, tuple[FunctorView, str]] = {}

        for (
            name,
            constructor,
            argument_side,
            opposite_name,
            result,
            argument,
        ) in self._functors:
            opposite = roles[opposite_name]
            opposite_constructor = opposite[0]
            forward = constructor if argument_side != "left" else opposite_constructor
            backward = constructor if argument_side != "right" else opposite_constructor
            key = (forward, backward, result, argument)
            view = built.setdefault(
                key,
                FunctorView(
                    store,
                    forward,
                    backward,
                    result=result,
                    argument=argument,
                ),
            )
            selected[name] = (view, constructor)

        conjunctions = self._conjunctions
        inventory = (
            None
            if conjunctions is None
            else Conjunctions(
                conjunctions[0],
                tuple(resolve_category(marker) for marker in conjunctions[1]),
                tuple(
                    (resolve_category(unmarked), resolve_category(marked))
                    for unmarked, marked in conjunctions[2]
                ),
                marking=conjunctions[3],
            )
        )
        return CCGAssembly(
            store,
            syntax,
            resolve_category,
            selected,
            matching,
            inventory,
            normal_form,
            category_sort=self._category_sort,
            rule_matchings=rule_matchings,
            categories=categories,
        )


def _feature(
    operator: OperatorSpec,
    target: str,
    delimiter: Literal["brackets", "colon"],
) -> str | None:
    """Return the feature a featured operator's name adds to its target's.

    Parameters
    ----------
    operator : OperatorSpec
        The declared operator.
    target : str
        The operator it erases to.

    Returns
    -------
    str | None
        The selected suffix's content, or ``None`` when the name is not the
        target's name followed by one feature in that syntax.
    """
    name = operator.name
    if delimiter == "colon":
        prefix = f"{target}:"
        if not name.startswith(prefix) or ":" in name[len(prefix) :]:
            return None
        feature = name[len(prefix) :]
    else:
        prefix = f"{target}["
        if not name.startswith(prefix) or not name.endswith("]"):
            return None
        feature = name[len(prefix) : -1]

    return feature or None


class SignatureSpec(dx.Model, extra="forbid"):
    """A term-signature declaration.

    The named category sort is the sort of every primitive, every constructor,
    every lexical and goal category, and every derivation index. A featured
    operator is declared by its name, which is its target's name with one
    feature in the selected delimiter syntax, its sort and the operator it
    erases to; it takes every other field from that target, and the signature's
    features are read off these declarations.

    Categories are written in infix syntax, with each binary constructor's
    spelling between its arguments, or as s-expressions. The functor
    constructors are either one forward and one backward constructor or one
    self-opposite constructor, sharing their result and argument slots. The
    conjunction inventory is optional: a signature without one has no
    conjunction marks to preserve, and a grammar declaration over it refuses
    coordination rules.
    """

    name: str = "unnamed"
    sorts: tuple[str, ...] = ("Cat",)
    category_sort: str = "Cat"
    primitives: tuple[str, ...] = ()
    operators: tuple[OperatorSpec, ...] = ()
    constructors: tuple[ConstructorSpec, ...] = ()
    functors: tuple[FunctorSpec, ...] = ()
    syntax: Literal["declared", "s-expression"] = "declared"
    feature_delimiter: Literal["brackets", "colon"] = "brackets"
    conjunctions: ConjunctionsSpec | None = None
    vocabularies: dict[str, tuple[str, ...]] = dx.field(default_factory=dict)
    open_sorts: tuple[str, ...] = ()
    classifiers: dict[str, str] = dx.field(default_factory=dict)
    allows_vacuous_binding: bool = True
    holes: bool = False

    __axioms__ = (
        _nonempty("name", "a signature needs a name"),
        dx.axiom("len(sorts) > 0", message="a signature needs a sort"),
        _all_named("sorts", "a sort needs a name"),
        _nonempty("category_sort", "a signature needs a category sort"),
        _all_named("primitives", "a primitive needs a name"),
        _all_named("open_sorts", "an open sort needs a name"),
        dx.axiom(
            'syntax != "declared" or len(constructors) > 0',
            message="declared category text needs a constructor",
        ),
    )

    @dx.validates("sorts")
    @classmethod
    def _sort_names_are_distinct(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Reject duplicate sort declarations."""
        repeated = _first_repeated_name(values)

        if repeated is not None:
            msg = f"sort {repeated!r} is declared twice"
            raise ValueError(msg)

        return values

    @dx.validates("primitives")
    @classmethod
    def _primitive_names_are_distinct(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Reject duplicate category-primitive declarations."""
        repeated = _first_repeated_name(values)

        if repeated is not None:
            msg = f"category primitive {repeated!r} is declared twice"
            raise ValueError(msg)

        return values

    @dx.validates("operators")
    @classmethod
    def _operator_names_are_distinct(
        cls, values: tuple[OperatorSpec, ...]
    ) -> tuple[OperatorSpec, ...]:
        """Reject duplicate operator and feature declarations."""
        repeated = _first_repeated_name(tuple(value.name for value in values))

        if repeated is not None:
            msg = f"operator {repeated!r} is declared twice"
            raise ValueError(msg)

        return values

    @dx.validates("vocabularies")
    @classmethod
    def _vocabulary_names_are_distinct(
        cls, values: dict[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        """Reject empty or duplicate constants in an explicit vocabulary row."""
        for sort, names in values.items():
            if not all(names):
                msg = f"vocabulary {sort!r} declares a constant with an empty name"
                raise ValueError(msg)

            repeated = _first_repeated_name(names)

            if repeated is not None:
                msg = f"vocabulary {sort!r} declares {repeated!r} twice"
                raise ValueError(msg)

        return values

    @dx.model_validator()
    def _category_sort_is_declared(self) -> Self:
        """Require the named category sort to belong to the signature."""
        if self.category_sort not in self.sorts:
            msg = (
                f"category sort {self.category_sort!r} is not among "
                f"the declared sorts {self.sorts!r}"
            )
            raise ValueError(msg)

        return self

    @dx.model_validator()
    def _declarations_are_not_redundant(self) -> Self:
        """Return a signature whose operators each have one declaration.

        Raises
        ------
        ValueError
            If a listed operator is a bare category atom or a category
            constructor, which the primitives and the constructors declare,
            or a featured operator is malformed or erases to an operator the
            signature does not declare.
        """
        sort = self.category_sort
        sorts = {
            **dict.fromkeys(self.primitives, sort),
            **{
                operator.name: operator.sort
                for operator in self.operators
                if operator.erases_to is None
            },
            **{
                constructor.name: constructor.kind[-1]
                for constructor in self.constructors
            },
        }

        for operator in self.operators:
            target = operator.erases_to
            plain = not operator.binding
            categorial = operator.sort == sort and all(
                argument == sort for argument in operator.arg_sorts
            )

            if target is None and plain and categorial:
                field = "constructors" if operator.arg_sorts else "primitives"
                msg = f"operator {operator.name!r} belongs among the {field}"
                raise ValueError(msg)

            if target is None:
                continue

            if _feature(operator, target, self.feature_delimiter) is None:
                syntax = (
                    "bracketed feature"
                    if self.feature_delimiter == "brackets"
                    else "colon-delimited feature"
                )
                msg = (
                    f"featured operator {operator.name!r} is not named "
                    f"{target!r} with one {syntax}"
                )
                raise ValueError(msg)

            if target not in sorts:
                msg = (
                    f"featured operator {operator.name!r} erases to {target!r}, "
                    "which the signature does not declare"
                )
                raise ValueError(msg)

            if sorts[target] != operator.sort:
                msg = f"featured operator {operator.name!r} has another sort"
                raise ValueError(msg)

            if operator.arg_sorts or not plain or operator.family != "primitive":
                msg = (
                    f"featured operator {operator.name!r} takes its shape from "
                    f"{target!r} and declares none of its own"
                )
                raise ValueError(msg)

        return self

    @dx.model_validator()
    def _category_names_have_one_source(self) -> Self:
        """Require each category constant to have one declaration source."""
        declarations = (
            *(
                (name, f"primitives[{index}]")
                for index, name in enumerate(self.primitives)
            ),
            *(
                (operator.name, f"operators[{index}]")
                for index, operator in enumerate(self.operators)
                if operator.erases_to is not None
                and operator.erases_to in self.primitives
                and _feature(operator, operator.erases_to, self.feature_delimiter)
                is not None
            ),
            *(
                (name, f"vocabularies[{self.category_sort!r}][{index}]")
                for index, name in enumerate(
                    self.vocabularies.get(self.category_sort, ())
                )
            ),
        )
        owners: dict[str, str] = {}

        for name, source in declarations:
            previous = owners.get(name)

            if previous is not None:
                msg = (
                    f"category name {name!r} is declared by both "
                    f"{previous} and {source}"
                )
                raise ValueError(msg)

            owners[name] = source

        return self

    @dx.model_validator()
    def _constructors_and_functors_are_well_formed(self) -> Self:
        """Return a signature whose constructors and named views are coherent.

        Raises
        ------
        ValueError
            If a constructor is not an endofunctor over the category sort, or
            a functor view does not name a binary constructor and a mutually
            opposite view with compatible orientation and child slots.
        """
        sort = self.category_sort
        constructors = {
            constructor.name: constructor for constructor in self.constructors
        }
        if len(constructors) != len(self.constructors):
            msg = "a signature repeats a constructor name"
            raise ValueError(msg)

        for constructor in self.constructors:
            if any(held != sort for held in constructor.kind):
                msg = (
                    f"category constructor {constructor.name!r} has kind "
                    f"{constructor.kind!r}, not an endofunctor over {sort!r}"
                )
                raise ValueError(msg)

            constructor.refuse_notation_arity()

        views = {view.name: view for view in self.functors}
        if len(views) != len(self.functors):
            msg = "a signature repeats a functor view name"
            raise ValueError(msg)

        for view in self.functors:
            constructor = constructors.get(view.constructor)
            if constructor is None:
                msg = (
                    f"functor view {view.name!r} names undeclared constructor "
                    f"{view.constructor!r}"
                )
                raise ValueError(msg)
            if constructor.arity != 2:
                msg = f"functor view {view.name!r} needs a binary constructor"
                raise ValueError(msg)

            opposite = views.get(view.opposite)
            if opposite is None or opposite.opposite != view.name:
                msg = f"functor view {view.name!r} has no mutual opposite"
                raise ValueError(msg)
            if (view.result, view.argument) != (opposite.result, opposite.argument):
                msg = (
                    f"opposite functor views {view.name!r} and "
                    f"{view.opposite!r} disagree on slots"
                )
                raise ValueError(msg)
            sides = (view.argument_side, opposite.argument_side)
            if sides not in (
                ("right", "left"),
                ("left", "right"),
                ("either", "either"),
            ):
                msg = (
                    f"opposite functor views {view.name!r} and "
                    f"{view.opposite!r} have incompatible sides"
                )
                raise ValueError(msg)
            if (
                view.argument_side == "either"
                and view.constructor != opposite.constructor
            ):
                msg = "self-opposite functor views must name one constructor"
                raise ValueError(msg)

        return self

    @property
    def functor_names(self) -> frozenset[str]:
        """Return the names a rule may give a functor view."""
        return frozenset(view.name for view in self.functors)

    @property
    def functor_slots(self) -> dict[str, tuple[int, int]]:
        """Return the result and argument slot of each functor constructor.

        Returns
        -------
        dict[str, tuple[int, int]]
            The child slots of a functor's result and argument, keyed by the
            name of each constructor that declares functor roles.
        """
        return {
            view.constructor: (view.result, view.argument) for view in self.functors
        }

    def extend(self, language: Signature, /) -> Signature:
        """Extend one learner-language signature with its category algebra."""
        sort = self.category_sort

        if sort not in language.sorts:
            msg = (
                f"grammar category sort {sort!r} is not declared by language "
                f"{language.name!r}"
            )
            raise ValueError(msg)

        foreign_sorts = tuple(
            held for held in self.sorts if held != sort and held not in language.sorts
        )

        if foreign_sorts:
            msg = (
                "grammar signature declares sorts outside the language "
                f"{foreign_sorts!r}"
            )
            raise ValueError(msg)

        feature_atoms: dict[str, str] = {}
        operator_features: dict[str, list[str]] = {}
        primitive_names = set(self.primitives)

        for operator in self.operators:
            target = operator.erases_to
            feature = (
                None
                if target is None
                else _feature(operator, target, self.feature_delimiter)
            )

            if target is None or feature is None:
                continue

            if target in primitive_names:
                feature_atoms[operator.name] = target
            else:
                operator_features.setdefault(target, []).append(feature)

        grammar_vocabularies = dict(self.vocabularies)
        category_names = (
            *self.primitives,
            *feature_atoms,
            *grammar_vocabularies.pop(sort, ()),
        )
        vocabularies = {
            **language.vocabularies,
            **{
                held: Vocabulary(held, names)
                for held, names in grammar_vocabularies.items()
            },
            sort: Vocabulary(sort, category_names),
        }
        constant_cores = {
            held: dict(rows) for held, rows in language.constant_cores.items()
        }

        if feature_atoms:
            constant_cores[sort] = feature_atoms

        generated_features = (
            operator_features if self.feature_delimiter == "brackets" else {}
        )
        targets = {
            operator.name: operator
            for operator in (
                *language.declarations,
                *(
                    operator.build()
                    for operator in self.operators
                    if operator.erases_to is None
                ),
                *(constructor.build() for constructor in self.constructors),
            )
        }
        explicit_features = ()
        if self.feature_delimiter == "colon":
            explicit_features = tuple(
                Operator(
                    operator.name,
                    targets[target].sort,
                    arg_sorts=targets[target].arg_sorts,
                    binding=targets[target].binding,
                    family=targets[target].family,
                    erases_to=target,
                )
                for operator in self.operators
                if (target := operator.erases_to) is not None
                and target not in primitive_names
            )

        return Signature(
            f"{language.name}:{self.name}",
            language.sorts,
            (
                *language.declarations,
                *(
                    operator.build()
                    for operator in self.operators
                    if operator.erases_to is None
                ),
                *explicit_features,
                *(constructor.build() for constructor in self.constructors),
            ),
            vocabularies=tuple(vocabularies.values()),
            constant_cores=constant_cores,
            open_sorts=tuple(dict.fromkeys((*language.open_sorts, *self.open_sorts))),
            classifiers={**language.classifiers, **self.classifiers},
            allows_vacuous_binding=(
                language.allows_vacuous_binding and self.allows_vacuous_binding
            ),
            features={**language.features, **generated_features},
            holes=language.has_holes or self.holes,
            constant_family=language.constant_family,
            variable_family=language.variable_family,
        )

    def prepare_runtime(self) -> PreparedSignatureRuntime:
        """Freeze the category syntax and functor metadata."""
        return PreparedSignatureRuntime(self)
