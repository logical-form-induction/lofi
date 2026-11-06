"""The declared interpretation of a grammar into a logical form language.

An interpretation declaration names the symbol inventory, the semantic types
of the primitive categories and the semantic action of each rule. Building
it checks every action against the compiled logical form language and
returns the interpretation function a draw applies.
"""

from typing import TYPE_CHECKING, Literal, Protocol, Self, cast, final

import didactic.api as dx

from lofi._frozen import FrozenValue
from lofi.data.synthetic._interpretation import (
    InterpretationError,
    interpretation_function,
)
from lofi.data.synthetic._values import SymbolInventory
from lofi.data.synthetic.spec._validation import nonempty as _nonempty
from lofi.data.synthetic.spec.semantic_lexicon import (
    ResemblingSemanticLexiconSpec,
    SemanticLexiconStructure,
    preset,
)
from lofi.interpret import InhabitantSelection, MeaningSelection
from lofi.judge import PROOF_LOGICS, ProofLogic, ProofSearch
from lofi.lf.models import CountermodelFilter
from lofi.lf.provers import ProofService
from lofi.registry import plugin_for
from lofi.spec import ConfiguredSpec, DeclarationError
from lofi.spec.countermodels import CountermodelSpec
from lofi.spec.grammar import ErasureSpec
from lofi.spec.language import (
    ConstantSpec,
    LanguageDecl,
    LanguageIso,
)
from lofi.spec.normalization import NormalizationSpec
from lofi.terms import TermError

if TYPE_CHECKING:
    from lofi.data.synthetic._interpretation import InterpretationFunction
    from lofi.data.synthetic._semantics import SemanticLexicon
    from lofi.lf import EntailmentOracle
    from lofi.spec.grammar import GrammarAssembly, GrammarDecl, GrammarIso
    from lofi.spec.language import LanguageAssembly, LogicSpec
    from lofi.typecheck import Erasure


__all__ = [
    "InterpretationSpec",
    "LexicalMeaningSourceSpec",
    "MeaningSourceSpec",
    "MeaningSpec",
    "ProofSearchSpec",
    "ResemblingMeaningSourceSpec",
    "StructuredMeaningSourceSpec",
    "lower_interpretation",
]

type MeaningEntry = tuple[str, str, str]
"""One surface form, category text and logical form text."""


@final
class MeaningAssignment(FrozenValue):
    """Meanings and generated constants resolved for one compiled grammar."""

    __slots__ = ("_constants", "_entries", "_semantic_lexicon")

    _noun = "meaning assignment"

    def __init__(
        self,
        entries: tuple[MeaningEntry, ...],
        constants: tuple[ConstantSpec, ...] = (),
        *,
        semantic_lexicon: SemanticLexicon | None = None,
    ) -> None:
        object.__setattr__(self, "_entries", entries)
        object.__setattr__(self, "_constants", constants)
        object.__setattr__(self, "_semantic_lexicon", semantic_lexicon)

    @property
    def entries(self) -> tuple[MeaningEntry, ...]:
        """Return every resolved lexical meaning."""
        return self._entries

    @property
    def constants(self) -> tuple[ConstantSpec, ...]:
        """Return the generated nonlogical constant declarations."""
        return self._constants

    @property
    def semantic_lexicon(self) -> SemanticLexicon | None:
        """Return the generated semantic lexicon, if meanings were generated."""
        return self._semantic_lexicon


def _generated_constants(
    lexicon: SemanticLexicon, *, entity: str, truth: str, arrow: str
) -> tuple[ConstantSpec, ...]:
    """Return typed declarations for the open-class symbols of a semantic lexicon."""
    declarations = [
        ConstantSpec(name=meaning.symbol, type=entity, classes=("individual",))
        for meaning in lexicon.inventory.get("constant", ())
    ]

    for meaning in lexicon.inventory.get("predicate", ()):
        arity = meaning.arity

        if arity is None:
            msg = f"generated predicate {meaning.symbol!r} has no arity"
            raise ValueError(msg)

        declarations.append(
            ConstantSpec(
                name=meaning.symbol,
                type=f" {arrow} ".join((*((entity,) * arity), truth)),
                classes=("predicate",),
            )
        )

    return tuple(declarations)


def _generated_assignment(
    lexicon: SemanticLexicon, *, entity: str, truth: str, arrow: str
) -> MeaningAssignment:
    """Return boundary meanings and constants from an interior semantic lexicon."""
    return MeaningAssignment(
        tuple(
            (form, category, meaning.text)
            for (category, form), meanings in lexicon.meanings.items()
            for meaning in meanings
        ),
        _generated_constants(lexicon, entity=entity, truth=truth, arrow=arrow),
        semantic_lexicon=lexicon,
    )


class _MeaningSourceDeclaration(Protocol):
    """A declaration that resolves lexical meanings against a grammar."""

    def resolve(
        self,
        grammar: GrammarDecl,
        assembly: GrammarAssembly,
        /,
        *,
        entity: str,
        truth: str,
        arrow: str,
    ) -> MeaningAssignment:
        """Resolve meanings and generated constants against a compiled grammar."""
        ...


class _ErasureDeclaration(Protocol):
    """A declaration that builds a category-to-type erasure."""

    def build(
        self,
        assembly: GrammarAssembly,
        /,
        *,
        type_syntax: Literal["angle", "s-expression"],
    ) -> Erasure:
        """Build the declared erasure against a compiled grammar."""
        ...


class MeaningSourceSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """The single declared source of an interpretation's lexical meanings."""


class _LexicalMeaningDeclaration(Protocol):
    """A lexicon declaration that carries meanings on its entries."""

    def meaning_entries(self) -> tuple[MeaningEntry, ...]:
        """Return each declared form, category and logical form."""


class LexicalMeaningSourceSpec(MeaningSourceSpec, extra="forbid"):
    """Meanings declared in place on the grammar's lexical entries."""

    kind: Literal["lexical"] = "lexical"

    def resolve(
        self,
        grammar: GrammarDecl,
        assembly: GrammarAssembly,
        /,
        *,
        entity: str,
        truth: str,
        arrow: str,
    ) -> MeaningAssignment:
        """Return the grammar's lexical meanings after checking coverage."""
        del assembly, entity, truth, arrow
        entries = cast("_LexicalMeaningDeclaration", grammar.lexicon).meaning_entries()

        if not entries:
            msg = "a lexical meaning source needs a meaning on a lexical entry"
            raise ValueError(msg)

        if len(set(entries)) != len(entries):
            msg = "a lexical entry lists one meaning twice"
            raise ValueError(msg)

        return MeaningAssignment(entries)


class StructuredMeaningSourceSpec(MeaningSourceSpec, extra="forbid"):
    """Meanings generated from an inline or packaged semantic structure."""

    kind: Literal["structured"] = "structured"
    structure: SemanticLexiconStructure | None = None
    preset: str | None = None

    @dx.validates("preset")
    @classmethod
    def _preset_is_named(cls, name: str | None) -> str | None:
        """Return a named configured preset or nothing."""
        if name == "":
            msg = "a semantic-lexicon preset needs a name"
            raise ValueError(msg)

        return name

    @dx.model_validator()
    def _one_structure_spelling(self) -> Self:
        """Require exactly one inline structure or configured preset."""
        if (self.structure is None) == (self.preset is None):
            msg = "structured meanings need exactly one structure or preset"
            raise ValueError(msg)

        return self

    def semantic_lexicon(self) -> SemanticLexiconStructure:
        """Return the inline structure or load the named configured preset."""
        structure = self.structure

        if structure is not None:
            return structure

        name = self.preset

        if name is None:
            msg = "structured meanings have neither a structure nor a preset"
            raise ValueError(msg)

        return preset(name)

    def resolve(
        self,
        grammar: GrammarDecl,
        assembly: GrammarAssembly,
        /,
        *,
        entity: str,
        truth: str,
        arrow: str,
    ) -> MeaningAssignment:
        """Generate meanings over the compiled surface lexicon."""
        del grammar
        return _generated_assignment(
            self.semantic_lexicon().build(assembly),
            entity=entity,
            truth=truth,
            arrow=arrow,
        )


class ResemblingMeaningSourceSpec(MeaningSourceSpec, extra="forbid"):
    """Meanings drawn at a declared size from a measured semantic lexicon."""

    kind: Literal["resembling"] = "resembling"
    source: ResemblingSemanticLexiconSpec

    def resolve(
        self,
        grammar: GrammarDecl,
        assembly: GrammarAssembly,
        /,
        *,
        entity: str,
        truth: str,
        arrow: str,
    ) -> MeaningAssignment:
        """Draw meanings resembling the declared source estimate."""
        del grammar
        return _generated_assignment(
            self.source.build(assembly), entity=entity, truth=truth, arrow=arrow
        )


class MeaningSpec(dx.Model, extra="forbid"):
    """Declare lexical meanings and proof-inhabitant selection."""

    source: MeaningSourceSpec
    inhabitant_selection: InhabitantSelection = "unique"
    search_budget: int = 64

    __axioms__ = (
        dx.axiom(
            "search_budget > 0",
            message="a meaning search budget must be positive",
        ),
    )

    def selection(self) -> MeaningSelection:
        """Build the immutable inhabitant-selection policy."""
        return MeaningSelection(self.inhabitant_selection)


class ProofSearchSpec(dx.Model, extra="forbid"):
    """Declare the structural logic and contraction bound of proof search."""

    logic: ProofLogic = "relevant"
    max_uses: int = 1

    __axioms__ = (dx.axiom("max_uses > 0", message="max_uses must be positive"),)

    @dx.model_validator()
    def _structural_use_bound(self) -> Self:
        """Require substructural logics to use each assumption at most once."""
        if not PROOF_LOGICS[self.logic].contraction and self.max_uses != 1:
            msg = "linear and affine proof search use each assumption at most once"
            raise ValueError(msg)

        return self

    def build(self, *, max_inhabitants: int) -> ProofSearch:
        """Build the immutable proof search policy."""
        rules = PROOF_LOGICS[self.logic]

        return ProofSearch(
            contraction=rules.contraction,
            weakening=rules.weakening,
            max_uses=self.max_uses,
            max_inhabitants=max_inhabitants,
        )


def _inventory(
    language: LanguageAssembly,
    constants: tuple[ConstantSpec, ...],
) -> SymbolInventory:
    """Derive role-indexed symbols from typed constant declarations."""
    roles: dict[str, dict[str, int]] = {}

    for constant in constants:
        body = language.scheme(constant.name).body
        arity = 0
        result = body

        while result.op.name == language.function_type and len(result.args) == 2:
            arity += 1
            result = result.args[1]

        if not constant.classes:
            msg = f"typed constant {constant.name!r} needs an inventory role"
            raise ValueError(msg)

        for role in constant.classes:
            roles.setdefault(role, {})[constant.name] = arity

    return SymbolInventory(roles)


def _language_uses(
    assembly: GrammarAssembly,
    meanings: tuple[MeaningEntry, ...],
    erasure: Erasure,
    language: LanguageAssembly,
) -> tuple[tuple[str, str], ...]:
    """Return logical form uses paired with their category-induced types."""
    return tuple(
        (
            meaning,
            language.print_type(
                erasure(assembly.syntax.parse(assembly.store, category))
            ),
        )
        for _, category, meaning in meanings
    )


def _meaning_texts(
    assembly: GrammarAssembly,
    meanings: tuple[MeaningEntry, ...],
    /,
) -> dict[tuple[str, str], tuple[str, ...]]:
    """Group logical form texts by form and canonical category spelling."""
    held: dict[tuple[str, str], list[str]] = {}

    for form, category_text, meaning in meanings:
        category = assembly.syntax.render(
            assembly.store,
            assembly.syntax.parse(assembly.store, category_text),
        )
        held.setdefault((form, category), []).append(meaning)

    return {item: tuple(texts) for item, texts in held.items()}


def _validate_associative_commutative(
    language: LanguageAssembly,
    names: tuple[str, ...],
) -> None:
    """Refuse an AC declaration for anything but a closed binary operation."""
    for name in names:
        language.constant(name)
        first = language.scheme(name).body

        if first.op.name != language.function_type or len(first.args) != 2:
            msg = f"the associative-commutative symbol {name!r} is not binary"
            raise ValueError(msg)

        left, remaining = first.args

        if remaining.op.name != language.function_type or len(remaining.args) != 2:
            msg = f"the associative-commutative symbol {name!r} is not binary"
            raise ValueError(msg)

        right, result = remaining.args

        if left is not right or right is not result:
            msg = (
                f"the associative-commutative symbol {name!r} does not have "
                "one shared argument and result type"
            )
            raise ValueError(msg)


class InterpretationSpec(dx.Model, extra="forbid"):
    """The declared interpretation of a grammar into a logical form language.

    Attributes
    ----------
    entity : str
        The semantic type of an individual.
    truth : str
        The semantic type of a formula.
    type_syntax : Literal["angle", "s-expression"]
        The syntax of closed atom images: angle brackets over the language's
        function type, so that ``⟨⟨e,t⟩,t⟩`` is a function from properties
        to truth values, or the canonical s-expression syntax of the
        language's signature. In angle mode, constructor images use the
        language's surface type notation so they can name positional
        metavariables; in s-expression mode they use canonical terms.
    erasure : ErasureSpec
        The images of category atoms and non-functor constructors. Functor
        images are derived from the grammar's roles and the language's
        function type constructor.
    constants : tuple[ConstantSpec, ...]
        The typed nonlogical constants the forms are built from.
    meanings : MeaningSpec
        The lexical source, inhabitant selection and search budget.
    actions : dict[str, str]
        The semantic action of a rule, keyed by rule name, as a closed term
        applied to the rule's premises in order. An action listed here
        overrides the one derived from the rule, and a rule with no unique
        derived action needs one.
    proof_search : ProofSearchSpec
        The structural logic and contraction bound for derived actions.
    normalization : NormalizationSpec
        Beta traversal, eager-or-lazy derivation policy and quotient equations.
    """

    entity: str = "e"
    truth: str = "t"
    type_syntax: Literal["angle", "s-expression"] = "angle"
    erasure: ErasureSpec
    constants: tuple[ConstantSpec, ...]
    meanings: MeaningSpec
    normalization: NormalizationSpec = dx.field(default_factory=NormalizationSpec)
    countermodels: CountermodelSpec | None = None
    actions: dict[str, str] = dx.field(default_factory=dict)
    proof_search: ProofSearchSpec = dx.field(default_factory=ProofSearchSpec)

    __axioms__ = (
        _nonempty("entity", "an interpretation needs an entity type"),
        _nonempty("truth", "an interpretation needs a truth type"),
        dx.axiom("entity != truth", message="entity and truth are two types"),
    )

    def build(
        self,
        assembly: GrammarAssembly,
        resolved: MeaningAssignment,
        logic: LogicSpec,
        /,
    ) -> InterpretationFunction:
        """Build the interpretation inside a compiled grammar's language.

        Parameters
        ----------
        assembly : GrammarAssembly
            The compiled grammar the interpretation covers.
        resolved : MeaningAssignment
            The meanings and generated constants resolved at the dataset join.

        Raises
        ------
        DeclarationError
            If the language does not lower, its profile lies outside the
            declared prover's bound, a type term does not read, or the erasure
            does not cover the grammar.
        """
        language = assembly.language

        try:
            _validate_associative_commutative(
                language,
                self.normalization.associative_commutative,
            )

            normalization = self.normalization.build(
                fuel=assembly.grammar.budget.beta_fuel
            )
            erasure = cast("_ErasureDeclaration", self.erasure).build(
                assembly, type_syntax=self.type_syntax
            )
            inventory = _inventory(
                language,
                (*self.constants, *resolved.constants),
            )
            configured = cast("ConfiguredSpec[object]", logic.prover)

            def oracle_factory() -> EntailmentOracle:
                return cast(
                    "EntailmentOracle",
                    plugin_for(configured).construct(configured, language),
                )

            service = ProofService(
                oracle_factory,
                profile=language.profile,
                workers=logic.prover.workers,
                batch_size=logic.prover.batch_size,
                timeout=logic.prover.timeout,
                cache_size=logic.prover.cache_size,
            )

            try:
                countermodels = (
                    None
                    if self.countermodels is None
                    else CountermodelFilter(
                        language,
                        self.countermodels.settings(),
                    )
                )
                return interpretation_function(
                    assembly,
                    language,
                    erasure=erasure,
                    inventory=inventory,
                    meanings=_meaning_texts(assembly, resolved.entries),
                    supplied=self.actions,
                    service=service,
                    budget=logic.labeling.budget.build(),
                    truth=self.truth,
                    normalization=normalization,
                    normalization_policy=self.normalization.policy,
                    proof_search=self.proof_search.build(
                        max_inhabitants=self.meanings.search_budget
                    ),
                    meaning_selection=self.meanings.selection(),
                    countermodels=countermodels,
                    semantic_lexicon=resolved.semantic_lexicon,
                )
            except Exception:
                service.close()
                raise
        except (
            InterpretationError,
            TermError,
            ValueError,
        ) as error:
            msg = (
                f"the interpretation does not lower over "
                f"{assembly.grammar.name!r}: {error}"
            )
            raise DeclarationError(msg) from error


def _complete_language(
    declared: LanguageDecl,
    interpretation: InterpretationSpec,
    generated: tuple[ConstantSpec, ...],
    /,
) -> LanguageDecl:
    """Return the dataset language with every interpretation constant declared."""
    additions = (*interpretation.constants, *generated)
    existing = {constant.name for constant in declared.constants}
    repeated = sorted(
        existing & {constant.name for constant in additions}
        | {
            name
            for name in (constant.name for constant in additions)
            if sum(constant.name == name for constant in additions) > 1
        }
    )

    if repeated:
        msg = f"the language already declares the constants {repeated!r}"
        raise DeclarationError(msg)

    return declared.with_(constants=(*declared.constants, *additions))


def lower_interpretation(
    grammar: GrammarDecl,
    declared: LanguageDecl,
    interpretation: InterpretationSpec,
    grammars: GrammarIso,
    languages: LanguageIso,
    /,
) -> tuple[GrammarAssembly, InterpretationFunction]:
    """Lower one grammar and interpretation into a single final language store."""
    preliminary = _complete_language(declared, interpretation, ())
    typed_preliminary = preliminary.with_(
        constants=tuple(
            constant for constant in preliminary.constants if constant.type != "infer"
        )
    )
    provisional_grammar = grammars.forward(grammar, typed_preliminary)
    try:
        resolved = cast(
            "_MeaningSourceDeclaration", interpretation.meanings.source
        ).resolve(
            grammar,
            provisional_grammar,
            entity=interpretation.entity,
            truth=interpretation.truth,
            arrow=declared.function_type,
        )
    except ValueError as error:
        msg = (
            f"the interpretation does not lower over "
            f"{provisional_grammar.grammar.name!r}: {error}"
        )
        raise DeclarationError(msg) from error
    complete = _complete_language(declared, interpretation, resolved.constants)

    if any(constant.type == "infer" for constant in complete.constants):
        typed = complete.with_(
            constants=tuple(
                constant for constant in complete.constants if constant.type != "infer"
            )
        )
        typed_grammar = grammars.forward(grammar, typed)

        try:
            erasure = cast("_ErasureDeclaration", interpretation.erasure).build(
                typed_grammar,
                type_syntax=interpretation.type_syntax,
            )
            uses = _language_uses(
                typed_grammar,
                resolved.entries,
                erasure,
                typed_grammar.language,
            )
        except (TermError, ValueError) as error:
            msg = (
                f"the interpretation does not lower over "
                f"{typed_grammar.grammar.name!r}: {error}"
            )
            raise DeclarationError(msg) from error

        language = languages.infer_extended(complete, uses, grammar.signature.extend)
        assembly = grammars.forward_language(grammar, language)
    else:
        assembly = grammars.forward(grammar, complete)

    return assembly, interpretation.build(assembly, resolved, complete.logic)
