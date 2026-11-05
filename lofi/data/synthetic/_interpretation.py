"""The declared interpretation of a grammar and the proof system that decides it.

An interpretation function assigns every lexical entry a meaning term, every
rule a semantic action, every primitive category a semantic type, and every
non-logical symbol a type scheme. A derivation is read bottom up: entries
contribute their meanings and rule applications run their actions, so the
root of a goal derivation is a closed formula. The interpretation holds the
shared proof service and the data-labeling budget that decide pairs of such
formulas.
"""

from json import dumps
from types import MappingProxyType
from typing import TYPE_CHECKING, final, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi.data.synthetic._entailment import EntailmentCertificate
from lofi.data.synthetic._illtyped import TypingConflict, TypingFailure
from lofi.data.synthetic._values import (
    IllTypedFormError,
    SymbolInventory,
    UninterpretableDerivationError,
)
from lofi.interpret import (
    AmbiguousMeaningError,
    BoundInterpretation,
    InterpretationDeclaration,
    MeaningReading,
    MeaningSelection,
    NormalizationPolicy,
    compile_meaning_actions,
    read,
)
from lofi.lexicon import Entry
from lofi.lf import Refuted, Verdict
from lofi.lf.provers import ProofBudget, ProofService, ProofSite, Questions
from lofi.terms import Normalization, SExpSyntax, Term, TermError, normalize
from lofi.typecheck import (
    CategoryTypeMismatchError,
    Erasure,
    UnknownSymbolError,
    check_logical_form_category,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi._types import ReportValue
    from lofi.data.synthetic._semantics import SemanticLexicon
    from lofi.deduce import Derivation
    from lofi.interpret import Lexicalization, MeaningActions
    from lofi.judge import ProofCalculus, ProofSearch, Rule
    from lofi.lf.models import CountermodelFilter
    from lofi.spec.grammar import GrammarAssembly
    from lofi.spec.language import LanguageAssembly
    from lofi.terms import Calculus, TermStore
    from lofi.typecheck import LexicalSignature, TypingResult

__all__ = [
    "InterpretationError",
    "InterpretationFunction",
    "LambdaCalculus",
    "interpretation_function",
]


class InterpretationError(ValueError):
    """Raised when a declared interpretation does not cover its grammar."""


@final
class LambdaCalculus:
    """The lambda calculus of one formula store.

    Parameters
    ----------
    store : TermStore
        The store proof terms are interned in.
    operators : Calculus
        The abstraction and application operators of the store's signature.
    """

    __slots__ = ("_operators", "_sort", "_store")

    def __init__(self, store: TermStore, operators: Calculus) -> None:
        self._store = store
        self._operators = operators
        self._sort = store.signature.by_name(operators.lam).binding[0].sort

    @property
    def store(self) -> TermStore:
        """Return the proof-term store."""
        return self._store

    @property
    def operators(self) -> Calculus:
        """Return the abstraction and application operators."""
        return self._operators

    def hypothesis(self, index: int, /) -> Term:
        """Return the stable free variable standing for one premise.

        Parameters
        ----------
        index : int
            The premise position.
        """
        return self._store.fvar(
            self._sort, f"p{ensure_non_negative('a premise index', index)}"
        )

    def free_variable(self, name: str, /) -> Term:
        """Return a named free variable.

        Parameters
        ----------
        name : str
            The variable name.
        """
        return self._store.fvar(self._sort, name)

    def bound_variable(self, index: int, /) -> Term:
        """Return a bound variable.

        Parameters
        ----------
        index : int
            The de Bruijn index.
        """
        return self._store.bvar(self._sort, index)

    def application(self, function: Term, argument: Term, /) -> Term:
        """Return an application.

        Parameters
        ----------
        function : Term
            The term applied.
        argument : Term
            The term it is applied to.
        """
        return self._store.op(
            self._store.signature.by_name(self._operators.app), function, argument
        )

    def abstraction(self, variable: str, body: Term, /) -> Term:
        """Return an abstraction over a named variable.

        Parameters
        ----------
        variable : str
            The variable abstracted over.
        body : Term
            The body the variable is free in.
        """
        return self._store.bind(self._operators.lam, variable, body)


@final
class InterpretationFunction:
    """A total interpretation of one grammar into one logical-form language.

    Parameters
    ----------
    language : LanguageAssembly
        The logical-form language, with its stores, checker and profile.
    erasure : Erasure
        The total morphism from the grammar's categories to semantic types.
    meanings : Mapping[tuple[str, Term], tuple[Term, ...]]
        The meanings of each lexical entry, keyed by form and category.
    actions : MeaningActions
        The semantic action of every rule, keyed by rule name.
    rules : Mapping[str, Rule[Term]]
        The grammar's rules, keyed by name.
    signature : LexicalSignature
        The type scheme of every non-logical symbol.
    truth : Term
        The type of a formula.
    inventory : SymbolInventory
        The non-logical symbols the forms are built from.
    service : ProofService
        The shared, memoized service that decides entailment between formulas.
    budget : ProofBudget
        The resource bounds of the data-labeling use site.
    """

    __slots__ = (
        "_actions",
        "_budget",
        "_countermodels",
        "_erasure",
        "_interpretations",
        "_inventory",
        "_language",
        "_meanings",
        "_normalization",
        "_reading",
        "_rules",
        "_semantic_lexicon",
        "_service",
        "_signature",
        "_site",
        "_truth",
        "_type_syntax",
    )

    def __init__(
        self,
        language: LanguageAssembly,
        *,
        erasure: Erasure,
        meanings: Mapping[tuple[str, Term], tuple[Term, ...]],
        interpretations: Mapping[str, InterpretationDeclaration],
        actions: MeaningActions,
        rules: Mapping[str, Rule[Term]],
        signature: LexicalSignature,
        truth: Term,
        inventory: SymbolInventory,
        service: ProofService,
        budget: ProofBudget,
        normalization: Normalization,
        normalization_policy: NormalizationPolicy,
        countermodels: CountermodelFilter | None = None,
        semantic_lexicon: SemanticLexicon | None = None,
    ) -> None:
        self._language = language
        self._erasure = erasure
        self._meanings = MappingProxyType(dict(meanings))
        self._actions = actions
        self._rules = MappingProxyType(dict(rules))
        self._signature = signature
        self._truth = truth
        self._inventory = inventory
        self._service = service
        self._site = ProofSite(f"data.{language.name}.labeling")
        self._budget = budget
        self._countermodels = countermodels
        self._normalization = normalization
        self._semantic_lexicon = semantic_lexicon
        self._reading = MeaningReading[Term](
            actions,
            normalization=normalization,
            policy=normalization_policy,
        )
        self._interpretations = MappingProxyType(
            {
                name: declaration.bind(self._reading)
                for name, declaration in interpretations.items()
            }
        )
        self._type_syntax = SExpSyntax(language.store.signature)

    @property
    def language(self) -> LanguageAssembly:
        """Return the logical-form language."""
        return self._language

    @property
    def store(self) -> TermStore:
        """Return the store every form is interned in."""
        return self._language.store

    @property
    def calculus(self) -> ProofCalculus:
        """Return the semantic actions' proof-term calculus."""
        return self._actions.calculus

    @property
    def erasure(self) -> Erasure:
        """Return the category-to-type morphism of this interpretation."""
        return self._erasure

    @property
    def actions(self) -> MeaningActions:
        """Return the semantic action of every rule."""
        return self._actions

    @property
    def signature(self) -> LexicalSignature:
        """Return the type scheme of every non-logical symbol."""
        return self._signature

    @property
    def truth(self) -> Term:
        """Return the type of a formula."""
        return self._truth

    @property
    def inventory(self) -> SymbolInventory:
        """Return the non-logical symbols the forms are built from."""
        return self._inventory

    @property
    def service(self) -> ProofService:
        """Return the shared proof service that decides entailment."""
        return self._service

    @property
    def budget(self) -> ProofBudget:
        """Return the resource bounds of the data-labeling use site."""
        return self._budget

    def open_proof_step(self) -> None:
        """Open the data-labeling site's next proof-service budget window."""
        self._service.open_step(self._site)

    @property
    def normalization(self) -> Normalization:
        """Return the compiled logical-form normalization settings."""
        return self._normalization

    @property
    def normalization_policy(self) -> NormalizationPolicy:
        """Return when derivation meanings normalize."""
        return self._reading.policy

    def report(self) -> Mapping[str, ReportValue]:
        """Return action coverage and proof-search exhaustion for persistence."""
        return MappingProxyType(
            {
                "actions": len(self._actions),
                "uncovered": dumps(
                    sorted(self._actions.uncovered), separators=(",", ":")
                ),
                "ambiguous": dumps(
                    sorted(self._actions.ambiguous), separators=(",", ":")
                ),
                "bound-exhausted": dumps(
                    {
                        name: sorted(bounds)
                        for name, bounds in sorted(self._actions.exhausted.items())
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            }
        )

    @property
    def interpretations(self) -> Mapping[str, BoundInterpretation]:
        """Return the executable derivation interpretations by declared name."""
        return self._interpretations

    def interpretation(self, name: str, /) -> BoundInterpretation:
        """Return the derivation interpretation selected by a document name.

        Parameters
        ----------
        name : str
            Document-level interpretation name.

        Raises
        ------
        InterpretationError
            If the grammar declares no interpretation under the name.
        """
        selected = self._interpretations.get(name)

        if selected is None:
            msg = f"the grammar declares no interpretation {name!r}"
            raise InterpretationError(msg)

        return selected

    @property
    def semantic_lexicon(self) -> SemanticLexicon | None:
        """Return the generated semantic lexicon, if the meanings were generated."""
        return self._semantic_lexicon

    @property
    def entries(self) -> Mapping[tuple[str, Term], tuple[Term, ...]]:
        """Return the meanings of each lexical entry, keyed by form and category."""
        return self._meanings

    def primary(self, form: str, category: Term) -> Entry[Term, Term]:
        """Return a lexical entry with its first declared meaning.

        Parameters
        ----------
        form : str
            The surface form.
        category : Term
            The category the form bears.

        Raises
        ------
        UninterpretableDerivationError
            If the interpretation assigns the entry no meaning.
        """
        return self.alternatives(form, category)[0]

    @property
    def primaries(self) -> Lexicalization[Term, Term]:
        """Return the lexicalization that reads each entry's first meaning."""
        return _PrimaryMeanings(self)

    def admitted(
        self, position: int, form: str, category: Term, /
    ) -> tuple[Entry[Term, Term], ...]:
        """Return one lexical entry per meaning, wherever the form stands.

        Parameters
        ----------
        position : int
            The axiom's token position, which no meaning depends on.
        form : str
            The surface form.
        category : Term
            The category the form bears.

        Raises
        ------
        UninterpretableDerivationError
            If the interpretation assigns the entry no meaning.
        """
        del position
        return self.alternatives(form, category)

    def alternatives(self, form: str, category: Term) -> tuple[Entry[Term, Term], ...]:
        """Return one lexical entry per meaning assigned to a surface entry."""
        meanings = self._meanings.get((form, category))

        if meanings is None:
            msg = f"the interpretation assigns {form!r} no meaning at {category!r}"
            raise UninterpretableDerivationError(msg)

        return tuple(Entry(form, category, meaning) for meaning in meanings)

    def rule(self, name: str) -> Rule[Term]:
        """Return the grammar rule with a name.

        Parameters
        ----------
        name : str
            The rule name.

        Raises
        ------
        UninterpretableDerivationError
            If the grammar has no rule of that name.
        """
        rule = self._rules.get(name)

        if rule is None:
            msg = f"the grammar has no rule {name!r}"
            raise UninterpretableDerivationError(msg)

        return rule

    def read(
        self,
        derivation: Derivation[Term, float],
        lexicalization: Lexicalization[Term, Term],
        /,
    ) -> tuple[Term, ...]:
        """Return the meaning of a derivation under each lexical choice.

        Parameters
        ----------
        derivation : Derivation[Term, float]
            The derivation, read off a parse forest.
        lexicalization : Lexicalization[Term, Term]
            The rules and meaning-bearing entries its edges name.

        Raises
        ------
        UninterpretableDerivationError
            If an entry has no meaning or a rule has no action.
        """
        try:
            meanings = read(derivation, self._reading, lexicalization)
        except AmbiguousMeaningError as error:
            msg = f"the interpretation assigns the derivation several meanings: {error}"
            raise UninterpretableDerivationError(msg) from error

        held = tuple(meaning for meaning in meanings if meaning is not None)

        if len(held) != len(meanings):
            msg = "the interpretation assigns the derivation no meaning"
            raise UninterpretableDerivationError(msg)

        return held

    def check(self, term: Term) -> TypingResult:
        """Type a term against the declared signature.

        Parameters
        ----------
        term : Term
            The term to type.

        Raises
        ------
        UnknownSymbolError
            If the term names a symbol outside the inventory.
        """
        language = self._language

        return language.checker.check(term, self._signature, language.rules)

    def formula(self, term: Term) -> Term:
        """Return a term once it types as a closed formula.

        Parameters
        ----------
        term : Term
            The term to certify.

        Raises
        ------
        IllTypedFormError
            If the term names a symbol outside the inventory, leaves a typing
            constraint unsatisfied, or types as something other than a
            formula.
        """
        if term.free:
            msg = f"{self.render(term)!r} names a symbol outside the inventory"
            raise IllTypedFormError(msg)

        try:
            result = self.check(term)
        except UnknownSymbolError as error:
            msg = f"{self.render(term)!r} names a symbol outside the inventory"
            raise IllTypedFormError(msg) from error

        if not result.is_well_typed:
            msg = f"{self.render(term)!r} leaves {len(result.residual)} constraints"
            raise IllTypedFormError(msg)

        scheme = result.scheme

        if scheme is None or scheme.body is not self._truth:
            msg = f"{self.render(term)!r} does not type as a formula"
            raise IllTypedFormError(msg)

        return term

    def render_type(self, semantic_type: Term) -> str:
        """Return a semantic type written in the s-expression syntax.

        Parameters
        ----------
        semantic_type : Term
            The type to write.
        """
        return self._type_syntax.render(self._language.store, semantic_type)

    def failure(self, term: Term) -> TypingFailure | None:
        """Return the refusal the checker returns on a term, or nothing when it types.

        Parameters
        ----------
        term : Term
            The term to check.

        Raises
        ------
        UnknownSymbolError
            If the term names a symbol outside the inventory.
        """
        result = self.check(term)

        if result.is_well_typed:
            return None

        return TypingFailure(
            tuple(
                TypingConflict(
                    constraint.kind,
                    self.render_type(constraint.left),
                    self.render_type(constraint.right),
                    constraint.path,
                )
                for constraint in result.residual
            ),
            language_digest=self._language.declaration_digest,
        )

    def verify_typing(self, term: Term, certificate: TypingFailure) -> bool:
        """Return whether checking a term again reproduces a refusal.

        Parameters
        ----------
        term : Term
            The term the certificate refused.
        certificate : TypingFailure
            The refusal to reproduce.
        """
        return certificate == self.failure(term)

    def render(self, term: Term) -> str:
        """Return a term written in the surface syntax that reads back to it.

        Parameters
        ----------
        term : Term
            The term to write.
        """
        return self._language.print(term)

    def read_text(self, text: str) -> Term:
        """Return the unchecked term a surface rendering reads back to.

        Parameters
        ----------
        text : str
            The surface text.
        """
        syntax = self._language.syntax
        return syntax.desugar(syntax.parse(text)).term

    def decide(self, premise: Term, hypothesis: Term, /) -> EntailmentCertificate:
        """Decide whether a premise entails a hypothesis and certify the answer.

        Parameters
        ----------
        premise : Term
            The formula assumed.
        hypothesis : Term
            The formula decided.
        """
        service = self._service
        verdict = self.verdicts(Questions((premise,), (hypothesis,)))[0]

        return EntailmentCertificate(
            verdict,
            prover=service.prover,
            profile_digest=self._language.profile.digest,
            timeout=service.timeout,
        )

    def verdicts(self, questions: Questions, /) -> tuple[Verdict, ...]:
        """Filter finite countermodels before sending undecided pairs to the prover."""
        filtered: list[Verdict | None] = []
        pending_premises = []
        pending_goals = []

        for premise, goal in zip(questions.premises, questions.goals, strict=True):
            countermodel = (
                None
                if self._countermodels is None
                else self._countermodels.refute(
                    premise,
                    goal,
                    theory=questions.theory,
                )
            )

            if countermodel is None:
                filtered.append(None)
                pending_premises.append(premise)
                pending_goals.append(goal)
            else:
                filtered.append(Refuted(countermodel))

        pending = Questions(
            tuple(pending_premises),
            tuple(pending_goals),
            theory=questions.theory,
        )
        proved = (
            ()
            if not pending.premises
            else self._service.decide(pending, budget=self._budget, site=self._site)
        )
        proof_at = iter(proved)
        return tuple(next(proof_at) if item is None else item for item in filtered)

    def verify(
        self,
        premise: Term,
        hypothesis: Term,
        certificate: EntailmentCertificate,
    ) -> bool:
        """Return whether deciding a pair again reproduces a certificate.

        Parameters
        ----------
        premise : Term
            The formula the certificate assumed.
        hypothesis : Term
            The formula the certificate decided.
        certificate : EntailmentCertificate
            The decision to reproduce under the declared labeling budget.
        """
        return certificate == self.decide(premise, hypothesis)

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the language, the entries and the actions."""
        return (
            f"InterpretationFunction({self._language.name!r}, "
            f"entries={len(self._meanings)}, actions={len(self._actions)}, "
            f"inventory={self._inventory!r})"
        )


def _mask(atoms: Iterable[Term]) -> int:
    """Return the free-variable mask of some variables.

    Parameters
    ----------
    atoms : Iterable[Term]
        The variables.
    """
    mask = 0

    for atom in atoms:
        mask |= atom.free

    return mask


def _spine(types: TermStore, arrow: str, names: Sequence[str]) -> Term:
    """Return the curried function type through some atomic types.

    Parameters
    ----------
    types : TermStore
        The semantic-type store.
    arrow : str
        The function-type constructor.
    names : Sequence[str]
        The atomic types, at least one, the last being the result.

    Raises
    ------
    InterpretationError
        If a name or the arrow is not declared by the type signature.
    """
    signature = types.signature

    try:
        type_sort = signature.classifiers[next(iter(signature.classifiers))]
        held = types.const(type_sort, names[-1])

        for name in reversed(names[:-1]):
            held = types.op(
                signature.by_name(arrow), types.const(type_sort, name), held
            )
    except TermError as error:
        msg = f"the type {' -> '.join(names)!r} is not over {signature.name!r}: {error}"
        raise InterpretationError(msg) from error

    return held


def _read(language: LanguageAssembly, field: str, text: str) -> Term:
    """Return a surface text read into a store, or refuse it by field.

    Parameters
    ----------
    language : LanguageAssembly
        The language whose generated syntax reads the text.
    field : str
        What the text is, for a refusal.
    text : str
        The surface text.

    Raises
    ------
    InterpretationError
        If the text does not parse or does not read into the store.
    """
    try:
        return language.read(ensure_name(field, text))
    except (TermError, ValueError) as error:
        msg = f"{field} {text!r} cannot be read: {error}"
        raise InterpretationError(msg) from error


def _supplied_action(
    language: LanguageAssembly,
    calculus: LambdaCalculus,
    name: str,
    arity: int,
    text: str,
    *,
    normalization: Normalization,
) -> Term:
    """Return a closed action text applied to a rule's hypotheses and normalized.

    Parameters
    ----------
    calculus : LambdaCalculus
        The lambda calculus of the formula store.
    name : str
        The rule the action belongs to.
    arity : int
        How many premises the rule takes.
    text : str
        The action as a closed term over as many arguments as premises.

    Raises
    ------
    InterpretationError
        If the text does not read, or the applied action does not leave
        exactly the rule's hypotheses open.
    """
    store = calculus.store
    applied = _read(language, f"the action of rule {name!r}", text)
    hypotheses = tuple(calculus.hypothesis(index) for index in range(arity))

    for hypothesis in hypotheses:
        applied = calculus.application(applied, hypothesis)

    action = normalize(
        store,
        applied,
        calculus.operators,
        normalization=normalization,
    )

    if action.free != _mask(hypotheses):
        msg = f"the action of rule {name!r} does not use each of its {arity} premises"
        raise InterpretationError(msg)

    return action


def _meanings(
    assembly: GrammarAssembly,
    language: LanguageAssembly,
    erasure: Erasure,
    declared: Mapping[tuple[str, str], tuple[str, ...]],
) -> dict[tuple[str, Term], tuple[Term, ...]]:
    """Read and type the declared meaning of every lexical entry.

    Parameters
    ----------
    assembly : GrammarAssembly
        The compiled grammar with its lexicon.
    language : LanguageAssembly
        The logical-form language.
    erasure : Erasure
        The homomorphism from categories to semantic types.
    declared : Mapping[tuple[str, str], tuple[str, ...]]
        The meaning texts of each entry, keyed by form and category text.

    Raises
    ------
    InterpretationError
        If an entry has no declared meaning, a declared meaning names no
        entry, a meaning does not read, names a symbol outside the
        inventory, or has no inferred type that instantiates to the erasure
        of its category.
    """
    categories = assembly.store
    syntax = assembly.syntax
    lexicon = assembly.lexicon
    meanings: dict[tuple[str, Term], tuple[Term, ...]] = {}
    pending = dict(declared)

    for form in lexicon.forms:
        for entry in lexicon.entries(form):
            category = syntax.render(categories, entry.category)
            texts = pending.pop((form, category), None)

            if texts is None:
                msg = f"the interpretation assigns {form!r} no meaning at {category!r}"
                raise InterpretationError(msg)

            field = f"the meaning of {form!r} at {category!r}"
            held: list[Term] = []

            for text in texts:
                meaning = _read(language, field, text)

                if meaning.free:
                    msg = f"{field} names a symbol outside the inventory"
                    raise InterpretationError(msg)

                try:
                    check_logical_form_category(
                        language,
                        erasure,
                        meaning,
                        entry.category,
                    )
                except UnknownSymbolError as error:
                    msg = f"{field} names a symbol outside the inventory"
                    raise InterpretationError(msg) from error
                except CategoryTypeMismatchError as error:
                    msg = (
                        f"{field} does not type as the erasure of its category: {error}"
                    )
                    raise InterpretationError(msg) from error

                held.append(meaning)

            meanings[(form, entry.category)] = tuple(held)

    if pending:
        extra = sorted(pending)
        msg = f"the interpretation names entries the lexicon lacks: {extra!r}"
        raise InterpretationError(msg)

    return meanings


def interpretation_function(
    assembly: GrammarAssembly,
    language: LanguageAssembly,
    *,
    erasure: Erasure,
    inventory: SymbolInventory,
    meanings: Mapping[tuple[str, str], tuple[str, ...]],
    supplied: Mapping[str, str],
    service: ProofService,
    budget: ProofBudget,
    truth: str,
    normalization: Normalization,
    normalization_policy: NormalizationPolicy,
    proof_search: ProofSearch,
    meaning_selection: MeaningSelection,
    countermodels: CountermodelFilter | None = None,
    semantic_lexicon: SemanticLexicon | None = None,
) -> InterpretationFunction:
    """Build a total interpretation of a grammar from its declared parts.

    Parameters
    ----------
    assembly : GrammarAssembly
        The compiled grammar with its lexicon.
    language : LanguageAssembly
        The logical-form language the forms are written in.
    erasure : Erasure
        The total signature morphism from categories to semantic types.
    inventory : SymbolInventory
        The non-logical symbols the forms are built from.
    meanings : Mapping[tuple[str, str], tuple[str, ...]]
        The meaning texts of each lexical entry, keyed by form and category
        text.
    supplied : Mapping[str, str]
        The action text of a rule, keyed by rule name, overriding or
        completing the actions derived from the rules.
    service : ProofService
        The shared proof service over the language's formula store.
    budget : ProofBudget
        The resource bounds of the data-labeling use site.
    truth : str
        The type of a formula.
    normalization : Normalization
        The declared normal form and grammar-owned reduction budget.
    normalization_policy : NormalizationPolicy
        Whether derivation actions normalize eagerly or lazily.
    proof_search : ProofSearch
        The structural logic and enumeration bounds for derived actions.
    meaning_selection : MeaningSelection
        How proof inhabitants become semantic actions.
    semantic_lexicon : SemanticLexicon | None
        The generated assignment the meanings came from, or nothing for a list.

    Raises
    ------
    InterpretationError
        If a meaning or an action does not read or type, an entry has no
        meaning, a meaning names no entry, or a rule ends up with no unique
        action.
    """
    calculus = LambdaCalculus(language.store, language.calculus)
    type_store = language.store

    rules = {rule.name: rule for rule in assembly.grammar.rules}
    overrides = {
        name: _supplied_action(
            language,
            calculus,
            name,
            rules[name].arity,
            text,
            normalization=normalization,
        )
        for name, text in supplied.items()
        if name in rules
    }
    unknown = sorted(set(supplied) - set(rules))

    if unknown:
        msg = (
            f"the interpretation supplies actions for rules the grammar lacks: "
            f"{unknown!r}"
        )
        raise InterpretationError(msg)

    actions = compile_meaning_actions(
        rules.values(),
        overrides,
        erasure=erasure,
        calculus=calculus,
        normalization=normalization,
        search=proof_search,
        selection=meaning_selection,
    )
    missing = sorted((set(rules) - set(actions.rules)) | set(actions.uncovered))

    if missing:
        exhausted = {
            name: sorted(bounds) for name, bounds in sorted(actions.exhausted.items())
        }
        msg = (
            f"no selected action covers every sequent of rules {missing!r}; "
            f"ambiguous: {sorted(actions.ambiguous)!r}, "
            f"uncovered: {sorted(actions.uncovered)!r}, "
            f"bound-exhausted: {exhausted!r}"
        )
        raise InterpretationError(msg)

    signature = language.lexicon

    return InterpretationFunction(
        language,
        erasure=erasure,
        meanings=_meanings(assembly, language, erasure, meanings),
        interpretations=assembly.grammar.interpretations,
        actions=actions,
        rules=rules,
        signature=signature,
        truth=_spine(type_store, erasure.function_type, (truth,)),
        inventory=inventory,
        service=service,
        budget=budget,
        normalization=normalization,
        normalization_policy=normalization_policy,
        countermodels=countermodels,
        semantic_lexicon=semantic_lexicon,
    )


@final
class _PrimaryMeanings:
    """The lexicalization that gives each entry its first declared meaning."""

    __slots__ = ("_function",)

    def __init__(self, function: InterpretationFunction, /) -> None:
        self._function = function

    def rule(self, name: str, /) -> Rule[Term]:
        """Return the grammar rule with a name."""
        return self._function.rule(name)

    def admitted(
        self, position: int, form: str, category: Term, /
    ) -> tuple[Entry[Term, Term], ...]:
        """Return the entry with the form's first meaning."""
        del position
        return (self._function.primary(form, category),)
