"""Checked lowering from a higher-order declaration to one language store."""

from typing import TYPE_CHECKING, Protocol, override

import didactic.api as dx

from lofi.formalism import TheoryArtifact
from lofi.lf import (
    CLASSICAL,
    STRICT,
    LogicalRole,
    LogicFragment,
    LogicProfile,
    LogicProfileError,
    SurfaceSyntax,
)
from lofi.spec._compile import (
    LANGUAGE_SOURCE_VERSION,
    LANGUAGE_TARGET_VERSION,
    canonical_declaration_bytes,
)
from lofi.spec._errors import DeclarationError
from lofi.spec.language._decl import (
    LanguageDecl,
    materialize_leaf_lexicon,
)
from lofi.spec.language._errors import ConstantInferenceError, LanguageError
from lofi.spec.language._infer import infer_declaration
from lofi.spec.language._values import LanguageAssembly
from lofi.terms import (
    BindingSite,
    Operator,
    Signature,
    Term,
    TermStore,
    Vocabulary,
)
from lofi.typecheck import LexicalSignature, Scheme, TypeChecker

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "LANGUAGE_SOURCE_VERSION",
    "LANGUAGE_TARGET_VERSION",
    "LanguageIso",
    "canonical_bytes",
    "lower_language",
    "lower_linked_language",
]


def canonical_bytes(declaration: LanguageDecl, /) -> bytes:
    """Return canonical JSON bytes for a language declaration."""
    return declaration.compile_envelope.canonical_declaration


class _SignatureExtension(Protocol):
    """A declaration that extends one language signature."""

    def __call__(self, signature: Signature, /) -> Signature:
        """Return the extended signature."""
        ...


def _signature(source: LanguageDecl) -> Signature:
    """Build the declaration's one many-sorted signature."""
    ty = source.type_sort
    tm = source.term_sort
    church = source.binders.style == "church"
    operators = tuple(
        Operator(
            constructor.name,
            ty,
            arg_sorts=(ty,) * constructor.arity,
            family="constructor",
        )
        for constructor in source.constructors
    )
    abstraction = Operator(
        source.abstraction,
        tm,
        arg_sorts=(ty, tm) if church else (tm,),
        binding=(
            BindingSite(
                (1,) if church else (0,),
                tm,
                annotation=0 if church else None,
            ),
        ),
        family=source.binder_family,
    )
    application = Operator(
        source.application,
        tm,
        arg_sorts=(tm, tm),
        family="application",
    )
    return Signature(
        source.name,
        (tm, ty, source.category_sort),
        (*operators, abstraction, application),
        vocabularies=(
            Vocabulary(ty, tuple(atom.name for atom in source.atomic_types)),
            Vocabulary(tm, tuple(constant.name for constant in source.constants)),
        ),
        open_sorts=(ty, tm),
        classifiers={tm: ty},
        constant_family=source.constant_family,
        variable_family=source.variable_family,
    )


def _subterms(term: Term, /) -> tuple[Term, ...]:
    """Return a term and every descendant in preorder."""
    return (term, *(held for argument in term.args for held in _subterms(argument)))


def _arity(term: Term, constructor: str, /) -> int:
    """Return the number of arguments along one declared function spine."""
    arity = 0
    result = term

    while result.op.name == constructor and len(result.args) == 2:
        arity += 1
        result = result.args[1]

    return arity


def _quantified_type_order(body: Term, function_type: str, /) -> int | None:
    """Return the order of the type quantified by one binder scheme."""
    if body.op.name != function_type or len(body.args) != 2:
        return None

    domain = body.args[0]
    if domain.op.name != function_type or len(domain.args) != 2:
        return None

    quantified = domain.args[0]

    def rank(term: Term) -> int:
        if term.op.is_free_variable:
            return 0

        if term.op.name != function_type or len(term.args) != 2:
            return 0

        return max(rank(term.args[0]) + 1, rank(term.args[1]))

    return rank(quantified)


def _check_declared_order(
    order: str,
    schemes: dict[int, Scheme],
    vocabulary: Vocabulary,
    quantifiers: frozenset[str],
    function_type: str,
) -> None:
    """Refuse a quantifier whose bound type exceeds the declared order."""
    allowed = {"first": 0, "second": 1}.get(order)

    if allowed is None:
        return

    for name in quantifiers:
        rank = _quantified_type_order(
            schemes[vocabulary.id(name)].body,
            function_type,
        )

        if rank is not None and rank > allowed:
            field = "order"
            msg = (
                f"logical order {order!r} cannot quantify over the declared "
                f"type of binder {name!r}"
            )
            raise LogicProfileError(field, msg)


def _result_type(term: Term, function_type: str, /) -> tuple[int, Term]:
    """Return a scheme body's argument count and final result type."""
    arguments = 0
    result = term

    while result.op.name == function_type and len(result.args) == 2:
        arguments += 1
        result = result.args[1]

    return arguments, result


def _check_declared_signature_resources(
    *,
    functions: bool,
    equality: bool,
    constants: tuple[tuple[str, str, tuple[str, ...], str | None, bool], ...],
    store: TermStore,
    schemes: dict[int, Scheme],
    vocabulary: Vocabulary,
    roles: dict[str, LogicalRole],
    proposition_types: frozenset[str],
    function_type: str,
) -> None:
    """Refuse operators excluded by the declared logic fragment."""
    equalities = sorted(
        name for name, role in roles.items() if role.model_operation == "equality"
    )

    if equalities and not equality:
        field = "equality"
        msg = f"logic with equality disabled declares equality operators {equalities!r}"
        raise LogicProfileError(field, msg)

    if functions:
        return

    for name, _, _, _, _ in constants:
        arity, result = _result_type(
            schemes[vocabulary.id(name)].body,
            function_type,
        )
        result_name = None if not result.op.is_constant else store.payload_name(result)

        if arity and result_name not in proposition_types:
            field = "functions"
            msg = f"logic with functions disabled declares function operator {name!r}"
            raise LogicProfileError(field, msg)


def _lower_language(
    source: LanguageDecl,
    /,
    *,
    signature: Signature | None = None,
    store: TermStore | None = None,
) -> LanguageAssembly:
    """Lower a declaration whose constant types are all resolved."""
    lowered_signature = _signature(source) if signature is None else signature
    lowered_store = (
        TermStore(lowered_signature, max_nodes=source.prepared_plugins.store_nodes)
        if store is None
        else store
    )
    if lowered_store.signature is not lowered_signature:
        msg = "a linked language store must own the materialized signature"
        raise ValueError(msg)
    store = lowered_store
    prepared = source.prepared_plugins
    constants = prepared.constants
    calculus = prepared.calculus
    (
        logic_order,
        functions,
        equality,
        domains,
        postulates,
        type_kinds,
        variables,
        semantics_name,
        negation,
        label_map_name,
    ) = prepared.logic
    syntax = SurfaceSyntax(
        store,
        tuple(name for name, _, _, _, _ in constants),
        prefix=prepared.prefix,
        infix=prepared.infix,
        binders=prepared.binders,
        type_sort=prepared.type_sort,
        term_sort=prepared.term_sort,
        abstraction=calculus.lam,
        application=calculus.app,
        style=prepared.binder_style,
        variables=variables,
        type_notations=prepared.type_notations,
        type_arities=prepared.type_arities,
    )
    vocabulary = lowered_signature.vocabulary(prepared.term_sort)
    schemes: dict[int, Scheme] = {}

    for name, declared_type, quantifies, _, _ in constants:
        body = syntax.read_type(declared_type, quantified=quantifies)
        quantified: list[int] = []

        for variable_name in quantifies:
            variable = store.fvar(prepared.type_sort, variable_name)

            if variable.payload is None:
                msg = f"quantified type variable {variable_name!r} has no atom"
                raise LanguageError(msg)

            quantified.append(variable.payload)

        schemes[vocabulary.id(name)] = Scheme(body, quantified)

    rules = prepared.rules
    declaration = canonical_bytes(source)
    route = f"{LANGUAGE_SOURCE_VERSION}->{LANGUAGE_TARGET_VERSION}"
    artifact = TheoryArtifact.from_specification(route, declaration)
    classes = prepared.classes
    roles = prepared.roles
    semantics = prepared.semantics
    defined = frozenset(
        name for name, _, _, definition, _ in constants if definition is not None
    )
    defined_quantifiers = frozenset(
        name
        for name, _, _, definition, binder_notation in constants
        if definition is not None and binder_notation
    )
    quantifiers = frozenset(
        (
            *(name for name, role in roles.items() if role.quantifier),
            *(name for name, plugin in semantics.items() if plugin.quantifier),
            *defined_quantifiers,
        )
    )

    if negation is not None:
        role = roles.get(negation)

        if role is None or role.truth_table != (True, False):
            field = "negation"
            msg = f"designated negation {negation!r} has no negation logical role"
            raise LogicProfileError(field, msg)

    label_map = CLASSICAL if label_map_name == "classical" else STRICT
    _check_declared_order(
        logic_order,
        schemes,
        vocabulary,
        quantifiers,
        prepared.function_type,
    )
    _check_declared_signature_resources(
        functions=functions,
        equality=equality,
        constants=constants,
        store=store,
        schemes=schemes,
        vocabulary=vocabulary,
        roles=dict(roles),
        proposition_types=frozenset(
            name for name, kind in type_kinds.items() if kind == "proposition"
        ),
        function_type=prepared.function_type,
    )

    for name, plugin in semantics.items():
        declared = schemes[vocabulary.id(name)].body
        arity = _arity(declared, prepared.function_type)

        if arity != plugin.arity:
            msg = (
                f"logical constant {name!r} has arity {arity}, but semantics "
                f"{plugin.name!r} needs {plugin.arity}"
            )
            raise DeclarationError(msg)

    if conflict := sorted((defined | frozenset(semantics)) & frozenset(roles)):
        msg = (
            "a defined or postulated logical constant cannot also select a logical "
            f"role: {conflict!r}"
        )
        raise DeclarationError(msg)

    fragment = LogicFragment(
        order=logic_order,
        quantifiers=quantifiers,
        connectives=(
            *(name for name, role in roles.items() if role.connective),
            *(name for name, plugin in semantics.items() if plugin.connective),
            *(defined - defined_quantifiers),
        ),
        functions=functions,
        equality=equality,
    )
    checker = TypeChecker(store)
    lexicon = LexicalSignature(schemes)
    bounds = prepared.bounds

    def assemble(profile: LogicProfile) -> LanguageAssembly:
        return LanguageAssembly(
            prepared.name,
            declaration,
            store,
            checker,
            lexicon,
            rules,
            calculus,
            syntax,
            classes,
            profile,
            artifact,
            bounds,
            type_sort=prepared.type_sort,
            term_sort=prepared.term_sort,
            category_sort=prepared.category_sort,
            function_type=prepared.function_type,
            source_version=LANGUAGE_SOURCE_VERSION,
            target_version=LANGUAGE_TARGET_VERSION,
        )

    profile = LogicProfile(
        store,
        formula_sort=prepared.term_sort,
        application=calculus.app,
        classes=classes,
        fragment=fragment,
        type_kinds=type_kinds,
        variables=variables,
        semantics_name=semantics_name,
        negation=negation,
        label_map=label_map,
        domains=domains,
        roles=roles,
        semantics=semantics,
    )
    provisional = assemble(profile)
    definitions = {
        name: provisional.read(definition)
        for name, _, _, definition, _ in constants
        if definition is not None
    }

    for name, definition in definitions.items():
        if any(
            term.op.is_constant and provisional.store.payload_name(term) in defined
            for term in _subterms(definition)
        ):
            msg = (
                f"definition for logical constant {name!r} must use primitives, "
                "not another defined logical constant"
            )
            raise DeclarationError(msg)

        if provisional.type_of(definition) != provisional.scheme(name).body:
            written = provisional.print_type(provisional.type_of(definition))
            expected = provisional.print_type(provisional.scheme(name).body)
            msg = (
                f"definition for logical constant {name!r} has type {written}, "
                f"not {expected}"
            )
            raise DeclarationError(msg)

    theory = tuple(provisional.read(postulate) for postulate in postulates)

    if not theory and not definitions:
        return provisional

    return assemble(
        LogicProfile(
            store,
            formula_sort=prepared.term_sort,
            application=calculus.app,
            classes=classes,
            fragment=fragment,
            type_kinds=type_kinds,
            variables=variables,
            semantics_name=semantics_name,
            negation=negation,
            label_map=label_map,
            theory=theory,
            definitions=definitions,
            domains=domains,
            roles=roles,
            semantics=semantics,
        )
    )


def _require_resolved(source: LanguageDecl, /) -> None:
    """Refuse a language whose constant types still require typed uses."""
    unresolved = tuple(
        name
        for name, declared_type, _, _, _ in source.prepared_plugins.constants
        if declared_type == "infer"
    )

    if unresolved:
        raise ConstantInferenceError(
            unresolved[0],
            "typed uses are required before the language can be finalized",
        )


def lower_language(source: LanguageDecl, /) -> LanguageAssembly:
    """Lower a validated, fully resolved declaration into one native assembly."""
    source = materialize_leaf_lexicon(source)
    _require_resolved(source)

    return _lower_language(source)


def lower_linked_language(
    source: LanguageDecl, signature: Signature, store: TermStore, /
) -> LanguageAssembly:
    """Lower a declaration through an already linked native signature and store."""
    source = materialize_leaf_lexicon(source)
    _require_resolved(source)

    return _lower_language(source, signature=signature, store=store)


class LanguageIso(dx.Iso[LanguageDecl, LanguageAssembly]):
    """Lower declarations and reconstruct them from canonical assembly bytes."""

    def __init__(self) -> None:
        self._lowered: dict[tuple[LanguageDecl, str], LanguageAssembly] = {}

    @override
    def forward(self, a: LanguageDecl, /) -> LanguageAssembly:
        """Return the cached native assembly for a declaration."""
        a = materialize_leaf_lexicon(a)
        _require_resolved(a)
        signature = _signature(a)
        key = (a, signature.digest)
        held = self._lowered.get(key)

        if held is not None:
            return held

        try:
            assembly = _lower_language(a, signature=signature)
        except ConstantInferenceError, LogicProfileError:
            raise
        except (KeyError, ValueError) as error:
            msg = f"language {a.name!r} does not lower: {error}"
            raise DeclarationError(msg) from error

        if assembly.declaration != canonical_declaration_bytes(a):
            msg = f"language {a.name!r} does not round-trip"
            raise DeclarationError(msg)

        self._lowered[key] = assembly
        return assembly

    def extend(
        self,
        source: LanguageDecl,
        extension: _SignatureExtension,
        /,
    ) -> LanguageAssembly:
        """Lower a language after extending its signature at a joint boundary."""
        source = materialize_leaf_lexicon(source)
        _require_resolved(source)
        signature = extension(_signature(source))
        key = (source, signature.digest)
        held = self._lowered.get(key)

        if held is not None:
            return held

        try:
            assembly = _lower_language(source, signature=signature)
        except ConstantInferenceError, LogicProfileError:
            raise
        except (KeyError, ValueError) as error:
            msg = f"language {source.name!r} does not lower: {error}"
            raise DeclarationError(msg) from error

        if assembly.declaration != canonical_declaration_bytes(source):
            msg = f"language {source.name!r} does not round-trip"
            raise DeclarationError(msg)

        self._lowered[key] = assembly
        return assembly

    @override
    def backward(self, b: LanguageAssembly, /) -> LanguageDecl:
        """Rebuild the declaration recorded by an assembly."""
        return LanguageDecl.model_validate_json(b.declaration)

    def infer(
        self,
        source: LanguageDecl,
        uses: Sequence[tuple[str, str]],
        /,
    ) -> LanguageAssembly:
        """Finalize inferred constant types from logical forms and required types."""
        source = materialize_leaf_lexicon(source)
        finalized = infer_declaration(
            source,
            tuple(uses),
            _lower_language,
        )
        signature = _signature(finalized)
        key = (finalized, signature.digest)
        held = self._lowered.get(key)

        if held is not None:
            return held

        assembly = _lower_language(finalized, signature=signature)

        if assembly.declaration != canonical_declaration_bytes(finalized):
            msg = f"language {source.name!r} does not round-trip after type inference"
            raise DeclarationError(msg)

        self._lowered[key] = assembly
        return assembly

    def infer_extended(
        self,
        source: LanguageDecl,
        uses: Sequence[tuple[str, str]],
        extension: _SignatureExtension,
        /,
    ) -> LanguageAssembly:
        """Infer constant types inside one extended language signature."""
        source = materialize_leaf_lexicon(source)
        finalized = infer_declaration(
            source,
            tuple(uses),
            lambda declared: _lower_language(
                declared,
                signature=extension(_signature(declared)),
            ),
        )
        signature = extension(_signature(finalized))
        key = (finalized, signature.digest)
        held = self._lowered.get(key)

        if held is not None:
            return held

        assembly = _lower_language(finalized, signature=signature)

        if assembly.declaration != canonical_declaration_bytes(finalized):
            msg = f"language {source.name!r} does not round-trip after type inference"
            raise DeclarationError(msg)

        self._lowered[key] = assembly
        return assembly
