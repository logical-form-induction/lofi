"""Principal type inference for constants declared ``type: infer``.

Every inferred constant is given a provisional scheme that leaves each of its
occurrences an independent type variable. Each typed use is checked on its
own, its root unified with the type the use requires, and the type of every
occurrence of an inferred constant recorded with its variables renamed apart
from every other use. Unifying the recorded occurrence types of each constant
then solves the whole constraint system at once, so a constant receives the
most general type all of its uses agree on, and a constant two of whose uses
demand types that do not unify is refused naming both uses.
"""

from itertools import count
from typing import TYPE_CHECKING

from lofi.spec.language._decl import materialize_leaf_lexicon
from lofi.spec.language._errors import ConstantInferenceError, LanguageError
from lofi.typecheck import InfiniteTypeError, Unifier

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from lofi.spec.language._decl import LanguageDecl
    from lofi.spec.language._values import LanguageAssembly
    from lofi.terms import Path, Term, TermStore

__all__ = ["infer_declaration"]


def _type_variable_names(source: LanguageDecl, /) -> Iterator[str]:
    """Yield type-variable names no declared type atom or constructor uses."""
    reserved = {atom.name for atom in source.atomic_types} | {
        constructor.name for constructor in source.constructors
    }

    for serial in count():
        for letter in "abcdefghijklmnopqrstuvwxyz":
            name = letter if serial == 0 else f"{letter}{serial}"

            if name not in reserved:
                yield name


def _provisional(source: LanguageDecl, /) -> LanguageDecl:
    """Give every inferred constant a scheme each occurrence instantiates afresh."""
    variable = next(_type_variable_names(source))

    return source.with_(
        constants=tuple(
            constant.with_(type=variable, quantifies=(variable,))
            if constant.type == "infer"
            else constant
            for constant in source.constants
        )
    )


def _mentions(term: Term, store: TermStore, sort: str, names: frozenset[str]) -> bool:
    """Return whether a term contains one of the named term constants."""
    if (
        term.op.sort == sort
        and term.op.is_constant
        and store.payload_name(term) in names
    ):
        return True

    return any(_mentions(argument, store, sort, names) for argument in term.args)


def _occurrences(
    term: Term,
    store: TermStore,
    sort: str,
    names: frozenset[str],
    path: Path = (),
) -> Iterator[tuple[str, Path]]:
    """Yield each occurrence of a named term constant with its path."""
    if term.op.sort == sort and term.op.is_constant:
        name = store.payload_name(term)

        if name in names:
            yield name, path

    for position, argument in enumerate(term.args):
        yield from _occurrences(argument, store, sort, names, (*path, position))


def _renamed_apart(
    store: TermStore,
    sort: str,
    use: int,
    types: Sequence[Term],
) -> Mapping[int, Term]:
    """Map every free type atom of one use to a variable private to that use."""
    free = 0

    for held in types:
        free |= held.free

    return {
        atom: store.fvar(sort, f"$use{use}.{atom}")
        for atom in range(free.bit_length())
        if free & (1 << atom)
    }


def _observe(
    language: LanguageAssembly,
    names: frozenset[str],
    uses: Sequence[tuple[str, str]],
) -> dict[str, list[tuple[int, Term]]]:
    """Return each inferred constant's occurrence types, keyed by use."""
    store = language.store
    unifier = Unifier(store)
    observed: dict[str, list[tuple[int, Term]]] = {name: [] for name in names}

    for at, (text, required) in enumerate(uses):
        desugared = language.syntax.desugar(language.syntax.parse(text))

        if not _mentions(desugared.term, store, language.term_sort, names):
            continue

        result = language.checker.check(
            desugared.term,
            language.lexicon,
            language.rules,
            annotations=(
                desugared.annotations if language.syntax.style == "curry" else None
            ),
        )

        if result.residual or result.scheme is None:
            msg = f"language use {text!r} is not typable"
            raise LanguageError(msg)

        try:
            root = unifier.unify(result.scheme.body, language.read_type(required))
        except InfiniteTypeError as error:
            msg = f"language use {text!r} requires an infinite type"
            raise LanguageError(msg) from error

        if root is None:
            found = language.print_type(result.scheme.body)
            msg = f"language use {text!r} has type {found!r}, not {required!r}"
            raise LanguageError(msg)

        found_here = [
            (name, unifier.apply(root, result.assignment[path]))
            for name, path in _occurrences(
                desugared.term, store, language.term_sort, names
            )
        ]
        apart = _renamed_apart(
            store, language.type_sort, at, [held for _, held in found_here]
        )

        for name, held in found_here:
            observed[name].append((at, unifier.apply(apart, held)))

    return observed


def _compose(
    unifier: Unifier, standing: Mapping[int, Term], update: Mapping[int, Term]
) -> dict[int, Term]:
    """Return the substitution applying ``standing`` and then ``update``."""
    composed = {atom: unifier.apply(update, held) for atom, held in standing.items()}

    for atom, held in update.items():
        composed.setdefault(atom, held)

    return composed


def _solve(
    language: LanguageAssembly,
    names: Sequence[str],
    observed: Mapping[str, Sequence[tuple[int, Term]]],
    uses: Sequence[tuple[str, str]],
) -> dict[int, Term]:
    """Unify every inferred constant's occurrence types into one substitution."""
    unifier = Unifier(language.store)
    substitution: dict[int, Term] = {}

    for name in names:
        occurrences = observed[name]

        if not occurrences:
            raise ConstantInferenceError(name, "no declared use mentions it")

        first, representative = occurrences[0]

        for at, held in occurrences[1:]:
            left = unifier.apply(substitution, representative)
            right = unifier.apply(substitution, held)

            try:
                update = unifier.unify(left, right)
            except InfiniteTypeError:
                update = None

            if update is None:
                raise ConstantInferenceError(
                    name,
                    f"use {uses[first][0]!r} demands "
                    f"{language.print_type(left)!r} but use {uses[at][0]!r} "
                    f"demands {language.print_type(right)!r}",
                )

            substitution = _compose(unifier, substitution, update)

    return substitution


def infer_declaration(
    source: LanguageDecl,
    uses: Sequence[tuple[str, str]],
    lower: Callable[[LanguageDecl], LanguageAssembly],
    /,
) -> LanguageDecl:
    """Resolve every ``type: infer`` constant from typed logical form uses.

    Parameters
    ----------
    source : LanguageDecl
        The declaration whose inferred constants are resolved.
    uses : Sequence[tuple[str, str]]
        Logical form texts, each paired with the semantic type it must have.
    lower : Callable[[LanguageDecl], LanguageAssembly]
        Lowers a declaration with every constant typed.

    Returns
    -------
    LanguageDecl
        The declaration with each inferred constant given its principal type
        scheme, its remaining type variables quantified.

    Raises
    ------
    ConstantInferenceError
        If no use mentions an inferred constant, or two of its uses demand
        types that do not unify.
    LanguageError
        If a use is ill-typed regardless of the inferred constants or does
        not have its required type.
    """
    source = materialize_leaf_lexicon(source)
    names = tuple(
        constant.name for constant in source.constants if constant.type == "infer"
    )

    if not names:
        return source

    language = lower(_provisional(source))
    observed = _observe(language, frozenset(names), uses)
    substitution = _solve(language, names, observed, uses)
    unifier = Unifier(language.store)
    resolved: dict[str, tuple[str, tuple[str, ...]]] = {}

    for name in names:
        body = unifier.apply(substitution, observed[name][0][1])
        variables = _type_variable_names(source)
        renaming = {
            atom: language.store.fvar(language.type_sort, next(variables))
            for atom in range(body.free.bit_length())
            if body.free & (1 << atom)
        }
        resolved[name] = (
            language.print_type(unifier.apply(renaming, body)),
            tuple(language.store.payload_name(held) for held in renaming.values()),
        )

    return source.with_(
        constants=tuple(
            constant.with_(
                type=resolved[constant.name][0],
                quantifies=resolved[constant.name][1],
            )
            if constant.name in resolved
            else constant
            for constant in source.constants
        )
    )
