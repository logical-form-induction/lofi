"""Compile grammar declarations into native assemblies."""

from json import dumps
from typing import TYPE_CHECKING, Final, cast, final

from lofi._frozen import FrozenValue, Unpicklable
from lofi.formalism import Formalism, FormalismDeclaration, ProfileBound, TheoryArtifact
from lofi.judge import capacity_report
from lofi.spec._compile import GRAMMAR_SOURCE_VERSION, GRAMMAR_TARGET_VERSION
from lofi.spec._errors import DeclarationError
from lofi.spec.grammar._grammar import GrammarDecl
from lofi.spec.grammar._report import CompileReport
from lofi.spec.grammar._values import CategoryMeasure, GrammarAssembly
from lofi.spec.language import (
    LanguageAssembly,
    LanguageDecl,
    LanguageIso,
    lower_linked_language,
)
from lofi.spec.language._decl import materialize_leaf_lexicon
from lofi.terms import (
    BindingSite,
    CategoryResolution,
    Operator,
    PayloadSort,
    RepresentativeUnavailableError,
    Signature,
    TermError,
    Vocabulary,
    _compile_grammar,
    _CompiledGrammar,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Protocol

    from lofi.grammars.ccg import CCGAssembly
    from lofi.judge import RuleSet
    from lofi.lexicon import StaticLexicon
    from lofi.spec._rule import BuiltRuleSet
    from lofi.terms import TargetSet, Term, TermStore, TermSyntax

    class _LexiconDeclaration(Protocol):
        """A declaration that builds the grammar's static lexicon."""

        def build(
            self,
            store: TermStore,
            syntax: TermSyntax,
            resolve_category: CategoryResolution,
            rules: RuleSet[Term],
            goal: TargetSet[Term],
            reserved: frozenset[Term],
            category_measure: CategoryMeasure,
            /,
        ) -> StaticLexicon[Term]:
            """Build the declared lexicon."""
            ...


class _CategoryResolver:
    """Resolve each category source once within one native crossing."""

    __slots__ = ("_category_sort", "_resolved", "_store", "_syntax")

    def __init__(
        self,
        store: TermStore,
        syntax: TermSyntax,
        category_sort: str,
        /,
    ) -> None:
        self._store = store
        self._syntax = syntax
        self._category_sort = category_sort
        self._resolved: dict[str, Term] = {}

    def __call__(self, source: str, /) -> Term:
        """Return one store-owned category for the source text."""
        held = self._resolved.get(source)
        if held is not None:
            return held

        category = self._syntax.parse(self._store, source)
        if not self._store.contains(category):
            msg = "a resolved category is outside the crossing store lineage"
            raise ValueError(msg)
        if category.op.sort != self._category_sort:
            msg = (
                f"a resolved category has sort {category.op.sort!r}, expected "
                f"category sort {self._category_sort!r}"
            )
            raise ValueError(msg)
        self._resolved[source] = category
        return category


__all__ = [
    "GRAMMAR_SOURCE_VERSION",
    "GRAMMAR_TARGET_VERSION",
    "AdoptedRules",
    "GrammarIso",
    "canonical_bytes",
]


def _symbolic_bound() -> ProfileBound:
    """Return the deduction profile served by the symbolic grammar consumer."""
    return ProfileBound(
        "lofi.deduce.parse",
        1,
        frozenset({1, 2, 3}),
        "an implementation of lofi.formalism.RegionIndex",
    )


_SYMBOLIC_BOUND: Final = (_symbolic_bound(),)
"""The immutable symbolic-consumer bound shared by grammar compilers."""


def canonical_bytes(declaration: GrammarDecl, /) -> bytes:
    """Return the one byte form of a grammar declaration."""
    return declaration.compile_envelope.canonical_declaration


def _is_lexicalized(rules: RuleSet[Term], /) -> bool:
    """Return whether every derivation step has a premise to anchor it."""
    return all(rule.arity != 0 for rule in rules)


def _linked_signature(plan: _CompiledGrammar, /) -> Signature:
    """Adapt one linked native signature to its public frozen wrapper."""
    sorts = tuple(plan.sorts)
    rows = tuple(plan.operators)
    operators_by_code = {code: name for name, code, *_ in rows}
    declarations = frozenset(plan.declarations)
    operators = tuple(
        Operator(
            name,
            sorts[sort],
            arg_sorts=tuple(sorts[argument] for argument in arguments),
            binding=tuple(
                BindingSite(
                    tuple(scopes),
                    sorts[bound_sort],
                    annotation=annotation,
                )
                for scopes, bound_sort, annotation in binding
            ),
            payload_sort=cast("PayloadSort", payload),
            family=family,
            erases_to=(None if erased == code else operators_by_code[erased]),
        )
        for name, code, sort, arguments, family, payload, erased, binding in rows
        if name in declarations
    )
    vocabularies: list[Vocabulary] = []
    constant_cores: dict[str, dict[str, str]] = {}

    for sort, names, cores in plan.vocabularies:
        sort_name = sorts[sort]
        vocabulary_names = tuple(names)
        vocabularies.append(Vocabulary(sort_name, vocabulary_names))
        mapped = {
            name: vocabulary_names[core]
            for name, core in zip(vocabulary_names, cores, strict=True)
            if vocabulary_names[core] != name
        }
        if mapped:
            constant_cores[sort_name] = mapped

    signature = Signature(
        plan.name,
        sorts,
        operators,
        vocabularies=tuple(vocabularies),
        constant_cores=constant_cores,
        open_sorts=tuple(sorts[sort] for sort in plan.open_sorts),
        classifiers={
            sorts[sort]: sorts[classifier] for sort, classifier in plan.classifiers
        },
        allows_vacuous_binding=plan.allows_vacuous_binding,
        features={name: tuple(values) for name, _, values in plan.features},
        holes=plan.has_holes,
        constant_family=plan.constant_family,
        variable_family=plan.variable_family,
    )
    if signature.digest != plan.signature_fingerprint:
        msg = (
            f"adapted signature fingerprint {signature.digest!r} does not "
            f"match native fingerprint {plan.signature_fingerprint!r}"
        )
        raise DeclarationError(msg, path="grammar.signature")
    return signature


def _native_plan(
    grammar: bytes,
    language: bytes,
    /,
    *,
    name: str,
) -> _CompiledGrammar:
    """Compile one immutable declaration envelope pair through native B-3."""
    try:
        return _compile_grammar(language, grammar)
    except (KeyError, TypeError, ValueError) as error:
        msg = f"grammar {name!r} does not compile natively: {error}"
        raise DeclarationError(msg) from error


def _native_theory_bytes(
    plan: _CompiledGrammar,
    *,
    source_version: str,
    target_version: str,
) -> bytes:
    """Return the canonical native theory record under one lowering route."""
    return dumps(
        {
            "route": {"source": source_version, "target": target_version},
            "theory": {
                "rule_fingerprint": plan.rule_fingerprint,
                "signature_fingerprint": plan.signature_fingerprint,
                "constructors": plan.theory_constructors,
                "elaborated": plan.elaborated_rules,
            },
        },
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def _linked_language(
    language: LanguageDecl, plan: _CompiledGrammar, /
) -> LanguageAssembly:
    """Adopt one compiled native core store into a language assembly."""
    try:
        signature = _linked_signature(plan)
        store = plan.term_store(signature)
        return lower_linked_language(language, signature, store)
    except DeclarationError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        msg = f"language {language.name!r} does not adopt its native plan: {error}"
        raise DeclarationError(msg) from error


@final
class AdoptedRules(Unpicklable, FrozenValue):
    """The runtime rules a grammar's native compile plan adopts.

    Parameters
    ----------
    context : CCGAssembly
        The rule context the plan's recipes were adopted into.
    built : BuiltRuleSet[RuleSet[Term]]
        The adopted rule set with its declared head selections.
    resolve_category : CategoryResolution
        The resolver of category texts into the context's store.
    """

    _noun = "adopted rule set"
    _because = "retains live term identities"
    __slots__ = ("_built", "_context", "_resolve_category")

    def __init__(
        self,
        context: CCGAssembly,
        built: BuiltRuleSet[RuleSet[Term]],
        resolve_category: CategoryResolution,
        /,
    ) -> None:
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_built", built)
        object.__setattr__(self, "_resolve_category", resolve_category)

    @property
    def context(self) -> CCGAssembly:
        """Return the rule context the recipes were adopted into."""
        return self._context

    @property
    def built(self) -> BuiltRuleSet[RuleSet[Term]]:
        """Return the adopted rules with their head selections."""
        return self._built

    @property
    def resolve_category(self) -> CategoryResolution:
        """Return the resolver of category texts into the context's store."""
        return self._resolve_category

    @property
    def rule_set(self) -> RuleSet[Term]:
        """Return the adopted runtime rule set."""
        return self._built.rule_set


def _adopt_rules(
    source: GrammarDecl,
    language: LanguageAssembly,
    plan: _CompiledGrammar,
    /,
) -> AdoptedRules:
    """Adopt a native compile plan's rule recipes into their runtime rules."""
    prepared = source.prepared_plugins
    store = language.store
    syntax = prepared.signature_runtime.syntax(store)
    resolve_category = _CategoryResolver(store, syntax, language.category_sort)
    context = prepared.signature_runtime.build(
        store,
        syntax,
        resolve_category,
        prepared.matching,
        prepared.normal_form,
        prepared.matching_rows,
        categories=plan.runtime_categories,
    )
    try:
        built = prepared.rules.adopt(tuple(plan.rules), context)
    except (RepresentativeUnavailableError, ValueError) as error:
        raise DeclarationError(str(error), path="grammar.rules") from error
    return AdoptedRules(context, built, resolve_category)


def _assemble(
    source: GrammarDecl,
    declaration: bytes,
    language: LanguageAssembly,
    plan: _CompiledGrammar,
    bounds: Sequence[ProfileBound],
    /,
) -> GrammarAssembly:
    """Adapt an authoritative native compile result into runtime containers."""
    prepared = source.prepared_plugins
    store = language.store
    signature = store.signature
    if signature.digest != plan.signature_fingerprint:
        msg = (
            f"grammar {prepared.name!r} has signature {signature.digest!r}, not "
            f"the native compile signature {plan.signature_fingerprint!r}"
        )
        raise DeclarationError(msg, path="grammar.signature")

    adopted = _adopt_rules(source, language, plan)
    context = adopted.context
    syntax = context.syntax
    resolve_category = adopted.resolve_category
    rules = adopted.built
    rule_set = rules.rule_set
    artifact = TheoryArtifact.from_specification(
        f"{GRAMMAR_SOURCE_VERSION}->{GRAMMAR_TARGET_VERSION}",
        _native_theory_bytes(
            plan,
            source_version=GRAMMAR_SOURCE_VERSION,
            target_version=GRAMMAR_TARGET_VERSION,
        ),
    )

    try:
        targets = prepared.goal_declaration.build(resolve_category, rule_set.matching)
    except (TermError, ValueError) as error:
        raise DeclarationError(str(error), path="grammar.goal.categories") from error
    goal = prepared.goal.construct(prepared.goal_declaration, targets)
    category_measure = prepared.category_measure
    try:
        conjunctions = context.conjunctions
        reserved = (
            frozenset() if conjunctions is None else frozenset(conjunctions.markers)
        )
        lexicon = cast("_LexiconDeclaration", prepared.lexicon).build(
            store,
            syntax,
            resolve_category,
            rule_set,
            targets,
            reserved,
            category_measure,
        )
    except (TermError, ValueError) as error:
        refusal = (
            error.prefixed("grammar.lexicon")
            if isinstance(error, DeclarationError)
            else DeclarationError(str(error), path="grammar.lexicon")
        )
        raise refusal from error
    seeds = tuple(
        entry.category for form in lexicon.forms for entry in lexicon.entries(form)
    )
    budget = prepared.budget
    closure = rule_set.closure(
        seeds,
        max_depth=prepared.closure_rounds,
        budget=prepared.closure_nodes,
        category_measure=category_measure,
        max_category_arity=budget.max_category_arity,
        max_category_size=budget.max_category_size,
    )
    unreachable = tuple(
        target
        for target in targets.categories
        if not any(
            rule_set.matching.equivalent(target, reached) for reached in closure.reached
        )
    )
    if unreachable:
        rendered = tuple(syntax.render(store, target) for target in unreachable)
        msg = f"targets {rendered!r} are unreachable in {closure!r}"
        raise DeclarationError(msg, path="grammar.goal.categories")

    category_sort = language.category_sort
    category_bound = language.bounds[category_sort]
    for category in (*seeds, *targets.categories):
        depth = 1 + category_measure.depth(category)
        if depth > category_bound.depth:
            rendered = syntax.render(store, category)
            msg = (
                f"category {rendered!r} occupies {depth} levels, exceeding "
                f"the language bound of {category_bound.depth}"
            )
            raise DeclarationError(
                msg,
                path=f"language.bounds.{category_sort}.depth",
            )

    formalism = Formalism(
        prepared.name,
        store,
        rule_set,
        targets,
        goal,
        FormalismDeclaration(
            closure,
            artifact,
            1 + prepared.max_hypotheses,
            budget,
            capacity_report(rule_set, prepared.capacity_evidence),
            lexicalized=_is_lexicalized(rule_set),
        ),
        interpretations=tuple(
            interpretation.construct(rules.head_selections)
            for interpretation in prepared.interpretations
        ),
    )
    compiled = formalism.compile(bounds)
    return GrammarAssembly(
        declaration,
        compiled,
        lexicon,
        syntax,
        language,
        CompileReport(
            compiled,
            closure,
            syntax,
            language.codec.at(category_sort),
            category_sort=category_sort,
        ),
        category_measure=category_measure,
        head_selections=rules.head_selections,
        max_category_arguments=prepared.max_category_arguments,
        max_category_order=prepared.max_category_order,
        source_version=GRAMMAR_SOURCE_VERSION,
        target_version=GRAMMAR_TARGET_VERSION,
    )


class GrammarIso:
    """Convert declaration envelopes through the one native B-3 compiler.

    A converter compiles each equal declaration pair once. A fresh converter
    still performs the complete native crossing on its first call.
    """

    def __init__(
        self,
        languages: LanguageIso | None = None,
        *,
        bounds: Sequence[ProfileBound] | None = None,
    ) -> None:
        self._languages = LanguageIso() if languages is None else languages
        self._bounds = _SYMBOLIC_BOUND if bounds is None else tuple(bounds)
        self._compiled: dict[tuple[bytes, bytes], GrammarAssembly] = {}
        self._linked: dict[tuple[bytes, str], LanguageAssembly] = {}

    @property
    def languages(self) -> LanguageIso:
        """Return the converter every learner language is lowered through."""
        return self._languages

    def forward(self, a: GrammarDecl, language: LanguageDecl, /) -> GrammarAssembly:
        """Compile one grammar and language declaration pair."""
        return self._forward_declaration(a, language)

    def forward_language(
        self, a: GrammarDecl, language: LanguageAssembly, /
    ) -> GrammarAssembly:
        """Compile a grammar against an already resolved language declaration."""
        declaration = LanguageDecl.model_validate_json(language.declaration)
        return self._forward_declaration(a, declaration)

    def rules(self, a: GrammarDecl, language: LanguageDecl, /) -> AdoptedRules:
        """Compile a declaration pair and adopt only its rule inventory.

        The native compiler validates and links the whole declaration, as
        :meth:`forward` does; the lexicon, goal and closure are not built, so
        a caller studying the rules alone pays for none of them.
        """
        language = materialize_leaf_lexicon(language)
        grammar_bytes = canonical_bytes(a)
        plan, lowered = self._plan(a, grammar_bytes, language)
        return _adopt_rules(a, lowered, plan)

    def _plan(
        self, grammar: GrammarDecl, grammar_bytes: bytes, language: LanguageDecl, /
    ) -> tuple[_CompiledGrammar, LanguageAssembly]:
        """Compile one declaration pair natively and link its language once."""
        language_bytes = language.compile_envelope.canonical_declaration
        plan = _native_plan(
            grammar_bytes,
            language_bytes,
            name=grammar.prepared_plugins.name,
        )
        linked_key = (language_bytes, plan.signature_fingerprint)
        lowered = self._linked.get(linked_key)
        if lowered is None:
            lowered = _linked_language(language, plan)
            self._linked[linked_key] = lowered
        return plan, lowered

    def _forward_declaration(
        self, grammar: GrammarDecl, language: LanguageDecl, /
    ) -> GrammarAssembly:
        """Compile one declaration pair through the sole native route."""
        language = materialize_leaf_lexicon(language)
        language_bytes = language.compile_envelope.canonical_declaration
        grammar_bytes = canonical_bytes(grammar)
        key = (grammar_bytes, language_bytes)
        held = self._compiled.get(key)
        if held is not None:
            return held

        plan, lowered = self._plan(grammar, grammar_bytes, language)
        assembly = _assemble(grammar, grammar_bytes, lowered, plan, self._bounds)
        self._compiled[key] = assembly
        return assembly

    def backward(self, b: GrammarAssembly, /) -> GrammarDecl:
        """Recover the declaration an assembly was built from."""
        return GrammarDecl.model_validate_json(b.declaration)
