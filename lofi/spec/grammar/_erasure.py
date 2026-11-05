"""Declared category-to-type signature morphisms."""

from typing import TYPE_CHECKING, Literal

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec._errors import DeclarationError
from lofi.terms import AngleSyntax, SExpSyntax
from lofi.typecheck import Erasure

if TYPE_CHECKING:
    from lofi.spec.grammar._values import GrammarAssembly
    from lofi.spec.language import LanguageAssembly
    from lofi.terms import Operator, Term

__all__ = ["ErasureSpec", "SignatureMorphismErasureSpec"]


def _images(field: str, declared: dict[str, str]) -> dict[str, str]:
    """Return a mapping whose names and image texts are nonempty."""
    held: dict[str, str] = {}

    for name, text in declared.items():
        key = ensure_name(f"{field} key", name)

        if not text:
            msg = f"{field}[{key!r}] needs a type term"
            raise ValueError(msg)

        held[key] = text

    return held


def _core_operators(assembly: GrammarAssembly) -> tuple[Operator, ...]:
    """Return the grammar's unique category-constructor cores."""
    signature = assembly.store.signature
    category_sort = assembly.language.category_sort
    seen: set[str] = set()
    cores: list[Operator] = []

    for operator in signature.operators:
        if (
            operator.sort != category_sort
            or operator.has_payload
            or not operator.arg_sorts
            or set(operator.arg_sorts) != {category_sort}
        ):
            continue

        core = signature.erase(operator)

        if core.name not in seen:
            seen.add(core.name)
            cores.append(core)

    return tuple(cores)


def _read_atom_image(
    language: LanguageAssembly,
    text: str,
    type_syntax: Literal["angle", "s-expression"],
) -> Term:
    """Read one closed atom image in the declared type notation."""
    syntax = (
        AngleSyntax(language.store.signature, language.function_type)
        if type_syntax == "angle"
        else SExpSyntax(language.store.signature)
    )

    return syntax.parse(language.store, text)


def _read_constructor_image(
    language: LanguageAssembly,
    text: str,
    type_syntax: Literal["angle", "s-expression"],
    parameters: tuple[str, ...],
) -> Term:
    """Read one constructor image over its positional metavariables."""
    if type_syntax == "angle":
        return language.syntax.read_type(text, quantified=parameters)

    return SExpSyntax(language.store.signature).parse(language.store, text)


class ErasureSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """A declared treatment for erasing categories to semantic types."""


class SignatureMorphismErasureSpec(ErasureSpec, extra="forbid"):
    """Images of source atoms and constructors in a target type signature."""

    kind: Literal["signature-morphism"] = "signature-morphism"
    atoms: dict[str, str]
    constructors: dict[str, str] = dx.field(default_factory=dict)

    @dx.validates("atoms")
    @classmethod
    def _atom_images_are_named(cls, declared: dict[str, str]) -> dict[str, str]:
        """Return named atom images with nonempty type terms."""
        return _images("atoms", declared)

    @dx.validates("constructors")
    @classmethod
    def _constructor_images_are_named(cls, declared: dict[str, str]) -> dict[str, str]:
        """Return named constructor images with nonempty type terms."""
        return _images("constructors", declared)

    def build(
        self,
        assembly: GrammarAssembly,
        /,
        *,
        type_syntax: Literal["angle", "s-expression"],
    ) -> Erasure:
        """Build one total morphism, deriving every functor image."""
        language = assembly.language
        cores = _core_operators(assembly)
        source_constructors = {
            operator.name: operator for operator in cores if operator.arity > 0
        }
        functors = dict(assembly.category_measure.constructors)
        declared = set(self.constructors)
        restated = tuple(sorted(declared & set(functors)))
        expected = set(source_constructors) - set(functors)
        missing = tuple(sorted(expected - declared))
        extra = tuple(sorted(declared - expected))

        if restated:
            msg = (
                f"the erasure restates the functor constructors {restated!r} of "
                f"{assembly.grammar.name!r}, whose images are derived from their "
                "functor roles"
            )
            raise DeclarationError(msg)

        if missing or extra:
            msg = (
                f"constructor image coverage has missing {missing!r} and "
                f"extra {extra!r}"
            )
            raise DeclarationError(msg)

        store = language.store
        function = store.signature.by_name(language.function_type)
        constructor_images: dict[str, Term] = {}

        for name, operator in source_constructors.items():
            parameters = tuple(f"a{position}" for position in range(operator.arity))

            if name in functors:
                result = functors[name]

                if operator.arity != 2 or result not in {0, 1}:
                    msg = (
                        f"functor constructor {name!r} needs one result and one "
                        "argument slot for erasure"
                    )
                    raise DeclarationError(msg)

                argument = 1 - result
                variables = tuple(
                    store.fvar(language.type_sort, parameter)
                    for parameter in parameters
                )
                constructor_images[name] = store.op(
                    function, variables[argument], variables[result]
                )
            else:
                constructor_images[name] = _read_constructor_image(
                    language,
                    self.constructors[name],
                    type_syntax,
                    parameters,
                )

        atom_images = {
            name: _read_atom_image(language, text, type_syntax)
            for name, text in self.atoms.items()
        }

        return Erasure(
            store,
            atom_images,
            constructor_images,
            category_sort=language.category_sort,
            function_type=language.function_type,
        )
