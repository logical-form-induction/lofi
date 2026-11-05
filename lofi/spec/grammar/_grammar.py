"""Grammar declarations and their runtime values.

A grammar declaration composes a signature, rules, a matching congruence, a
normal form, a lexicon, goals and an approximation budget.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Final, Literal, Self, cast, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name
from lofi._frozen import FrozenValue
from lofi.deduce import CoverGoal, Goal, PenaltyForm, RootGoal, Tiling
from lofi.formalism import Budget
from lofi.judge import (
    AllOf,
    CapacityEvidence,
    EisnerNormalForm,
    HockenmaierBiskNormalForm,
    NoNormalForm,
    NormalForm,
)
from lofi.registry import Registry, resolve_plugin
from lofi.spec._compile import (
    GRAMMAR_SOURCE_VERSION,
    GRAMMAR_TARGET_VERSION,
    GrammarCompileEnvelope,
    canonical_declaration_bytes,
)
from lofi.spec._matching import (
    AlphaMatchingSpec,
    FeatureErasureMatchingSpec,
    IdentityMatchingSpec,
    MatchingSpec,
    ProductMatchingSpec,
    SubsumptionMatchingSpec,
    matching_plugin,
)
from lofi.spec._rule import (
    PreparedRuleSet,
    RuleSetSpec,
    prepare_rule_set,
)
from lofi.spec.grammar._interpretation import (
    NamedInterpretationSpec,
    PreparedInterpretation,
    default_interpretations,
    interpretation_plugin,
)
from lofi.spec.grammar._signature import (
    PreparedSignatureRuntime,
    SignatureSpec,
    _all_named,
    _nonempty,
)
from lofi.spec.grammar._spec import GrammarSpec
from lofi.spec.grammar._values import CategoryMeasure
from lofi.terms import TargetSet, Term, TermMatching

if TYPE_CHECKING:
    from collections import abc

    from didactic.types._typing import FieldValue

    from lofi.spec.grammar._lexicon import LexiconSpec
    from lofi.terms import CategoryResolution, Matching

__all__ = [
    "AllOfNormalFormSpec",
    "AlphaMatchingSpec",
    "BudgetSpec",
    "CapacityEvidenceSpec",
    "CoverGoalSpec",
    "EisnerNormalFormSpec",
    "FeatureErasureMatchingSpec",
    "GoalPlugin",
    "GoalSpec",
    "GrammarDecl",
    "HockenmaierBiskNormalFormSpec",
    "IdentityMatchingSpec",
    "MatchingSpec",
    "NoNormalFormSpec",
    "NormalFormPlugin",
    "NormalFormSpec",
    "PreparedGrammarPlugins",
    "ProductMatchingSpec",
    "RootGoalSpec",
    "SubsumptionMatchingSpec",
    "UndirectedEisnerNormalFormSpec",
    "goal_plugin",
    "normal_form_plugin",
]

_DEFAULT_INCLUDE_CROSSED: Final = True
_DEFAULT_MAX_HYPOTHESES: Final = 0


class NormalFormSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """A derivation normal-form declaration."""

    def build(self) -> NormalForm[Term]:
        """Build the selected installed normal form."""
        plugin = normal_form_plugin(self)
        return plugin.construct(self)


class NormalFormPlugin[Declaration](FrozenValue):
    """Pair one open normal-form declaration with its runtime builder."""

    __slots__ = ("_build", "_spec")

    _noun = "normal-form plugin"

    def __init__(
        self,
        *,
        spec: type[Declaration],
        build: abc.Callable[[Declaration], NormalForm[Term]],
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_build", build)

    @property
    def spec(self) -> type[Declaration]:
        """Return the declaration class this plugin builds."""
        return self._spec

    def construct(self, declaration: Declaration, /) -> NormalForm[Term]:
        """Build one normal form from a validated declaration."""
        return self._build(declaration)


def normal_form_plugin(
    declaration: NormalFormSpec, /
) -> NormalFormPlugin[NormalFormSpec]:
    """Return the installed plugin selected by a normal-form declaration."""
    plugin = resolve_plugin(
        "normal_form",
        cast("str", declaration.kind),
        NormalFormPlugin,
        type(declaration),
    )

    return cast("NormalFormPlugin[NormalFormSpec]", plugin)


class NoNormalFormSpec(NormalFormSpec, extra="forbid"):
    """Every derivation step is admitted."""

    kind: Literal["none"] = "none"


class EisnerNormalFormSpec(NormalFormSpec, extra="forbid"):
    """Eisner's composition normal form."""

    kind: Literal["eisner"] = "eisner"
    include_crossed: bool = _DEFAULT_INCLUDE_CROSSED


class UndirectedEisnerNormalFormSpec(NormalFormSpec, extra="forbid"):
    """Eisner's form with both composition directions identified."""

    kind: Literal["undirected-eisner"] = "undirected-eisner"
    include_crossed: bool = _DEFAULT_INCLUDE_CROSSED


class HockenmaierBiskNormalFormSpec(NormalFormSpec, extra="forbid"):
    """The Hockenmaier and Bisk normal form over every composition order."""

    kind: Literal["hockenmaier-bisk"] = "hockenmaier-bisk"
    order_bound: int = 1
    track_child_order: bool = False
    composition_primary: bool = True
    unit_primary: bool = True
    raised_secondary: bool = True
    raised_application: bool = True
    coordination_raising: bool = True

    __axioms__ = (
        dx.axiom(
            "order_bound > 0",
            message="a Hockenmaier-Bisk order bound must be positive",
        ),
    )


class AllOfNormalFormSpec(NormalFormSpec, extra="forbid"):
    """Conjunction of declared normal forms."""

    kind: Literal["all-of"] = "all-of"
    normal_forms: tuple[NormalFormSpec, ...]

    __axioms__ = (
        dx.axiom(
            "len(normal_forms) > 0",
            message="an all-of normal form needs at least one member",
        ),
    )


def _build_none(declaration: NoNormalFormSpec, /) -> NormalForm[Term]:
    """Build the unrestricted filter."""
    del declaration
    return NoNormalForm[Term]()


def _build_eisner(declaration: EisnerNormalFormSpec, /) -> NormalForm[Term]:
    """Build the directed Eisner filter."""
    return EisnerNormalForm[Term](include_crossed=declaration.include_crossed)


def _build_undirected_eisner(
    declaration: UndirectedEisnerNormalFormSpec, /
) -> NormalForm[Term]:
    """Build the undirected Eisner filter."""
    return EisnerNormalForm[Term](
        include_crossed=declaration.include_crossed,
        undirected=True,
    )


def _build_hockenmaier_bisk(
    declaration: HockenmaierBiskNormalFormSpec, /
) -> NormalForm[Term]:
    """Build the configured Hockenmaier-Bisk filter."""
    return HockenmaierBiskNormalForm[Term](
        order_bound=declaration.order_bound,
        track_child_order=declaration.track_child_order,
        composition_primary=declaration.composition_primary,
        unit_primary=declaration.unit_primary,
        raised_secondary=declaration.raised_secondary,
        raised_application=declaration.raised_application,
        coordination_raising=declaration.coordination_raising,
    )


def _build_all_of(declaration: AllOfNormalFormSpec, /) -> NormalForm[Term]:
    """Build the conjunction of the declared members."""
    return AllOf(tuple(member.build() for member in declaration.normal_forms))


NONE_NORMAL_FORM_PLUGIN = NormalFormPlugin(spec=NoNormalFormSpec, build=_build_none)
EISNER_NORMAL_FORM_PLUGIN = NormalFormPlugin(
    spec=EisnerNormalFormSpec,
    build=_build_eisner,
)
UNDIRECTED_EISNER_NORMAL_FORM_PLUGIN = NormalFormPlugin(
    spec=UndirectedEisnerNormalFormSpec,
    build=_build_undirected_eisner,
)
HOCKENMAIER_BISK_NORMAL_FORM_PLUGIN = NormalFormPlugin(
    spec=HockenmaierBiskNormalFormSpec,
    build=_build_hockenmaier_bisk,
)
ALL_OF_NORMAL_FORM_PLUGIN = NormalFormPlugin(
    spec=AllOfNormalFormSpec,
    build=_build_all_of,
)


class GoalSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """A declared parse goal and its weighted grammatical root inventory."""

    categories: tuple[str, ...]

    __axioms__ = (
        dx.axiom("len(categories) > 0", message="a grammar goal needs a category"),
        _all_named("categories", "a goal category needs a name"),
    )

    def build(
        self,
        resolve_category: CategoryResolution,
        matching: Matching[Term],
        /,
    ) -> TargetSet[Term]:
        """Build the target set over the interned goal categories.

        Parameters
        ----------
        resolve_category : CategoryResolution
            Resolve category text in the crossing's term store.
        matching : Matching[Term]
            Congruence the rule set runs under.
        """
        return TargetSet(
            (resolve_category(category) for category in self.categories),
            matching,
        )

    def build_goal(self, targets: TargetSet[Term], /) -> Goal[Term]:
        """Build the installed runtime goal over checked targets."""
        return goal_plugin(self).construct(self, targets)


class GoalPlugin[Declaration](FrozenValue):
    """Pair one open goal declaration with its runtime builder."""

    __slots__ = ("_build", "_spec")
    _noun = "goal plugin"

    def __init__(
        self,
        *,
        spec: type[Declaration],
        build: abc.Callable[[Declaration, TargetSet[Term]], Goal[Term]],
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_build", build)

    @property
    def spec(self) -> type[Declaration]:
        """Return the declaration class this plugin builds."""
        return self._spec

    def construct(
        self,
        declaration: Declaration,
        targets: TargetSet[Term],
        /,
    ) -> Goal[Term]:
        """Build one runtime goal from a checked declaration and target set."""
        return self._build(declaration, targets)


def goal_plugin(declaration: GoalSpec, /) -> GoalPlugin[GoalSpec]:
    """Return the installed plugin selected by a goal declaration."""
    plugin = resolve_plugin(
        "goal", cast("str", declaration.kind), GoalPlugin, type(declaration)
    )

    return cast("GoalPlugin[GoalSpec]", plugin)


class RootGoalSpec(GoalSpec, extra="forbid"):
    """A weighted whole-sentence root goal."""

    kind: Literal["root"] = "root"
    categories: tuple[str, ...] = ("S",)
    root_weights: tuple[float, ...] | None = None

    __axioms__ = (
        dx.axiom(
            "isNone(root_weights) or len(root_weights) == len(categories)",
            message="a grammar goal needs one root weight per category",
        ),
    )

    @dx.validates("root_weights")
    @classmethod
    def _root_weights_are_finite(
        cls,
        root_weights: tuple[float, ...] | None,
    ) -> tuple[float, ...] | None:
        """Return optional finite root log weights."""
        if root_weights is None:
            return None

        for weight in root_weights:
            ensure_finite("a goal root weight", weight)

        return root_weights


class CoverGoalSpec(GoalSpec, extra="forbid"):
    """A shallow goal selecting a declared tiling and fragment penalty."""

    kind: Literal["cover"] = "cover"
    max_fragments: int
    penalty: float = 0.0
    penalty_form: Literal["after-first", "each-fragment"] = "after-first"
    tiling: Literal["fewest-fragments", "lexical"] = "fewest-fragments"

    __axioms__ = (
        dx.axiom("max_fragments > 0", message="a fragment bound must be positive"),
        dx.axiom("penalty >= 0", message="a fragment penalty must be nonnegative"),
    )

    @dx.validates("penalty")
    @classmethod
    def _penalty_is_finite(cls, penalty: float) -> float:
        """Return a finite fragment penalty."""
        return ensure_finite("a fragment penalty", penalty)


def _build_root_goal(
    declaration: RootGoalSpec,
    targets: TargetSet[Term],
    /,
) -> Goal[Term]:
    """Build a weighted spanning-root goal."""
    return RootGoal(targets, declaration.root_weights)


def _build_cover_goal(
    declaration: CoverGoalSpec,
    targets: TargetSet[Term],
    /,
) -> Goal[Term]:
    """Build a declared fragment-cover goal."""
    del targets
    return CoverGoal(
        declaration.max_fragments,
        declaration.penalty,
        penalty_form=PenaltyForm(declaration.penalty_form),
        tiling=Tiling(declaration.tiling),
    )


ROOT_GOAL_PLUGIN = GoalPlugin(spec=RootGoalSpec, build=_build_root_goal)
COVER_GOAL_PLUGIN = GoalPlugin(spec=CoverGoalSpec, build=_build_cover_goal)


class BudgetSpec(dx.Model, extra="forbid"):
    """Approximation bounds and the structural-closure limits."""

    depth: int = 5
    mass: float = 1.0
    beam: int = 8
    unary_rounds: int = 1
    threads: int = 1
    beta_fuel: int = 1_000
    closure_rounds: int = 2
    closure_nodes: int = 2_048
    max_category_arity: int | None = None
    max_category_size: int | None = None

    __axioms__ = (
        dx.axiom("depth > 0", message="a term depth bound must be positive"),
        dx.axiom("mass > 0 and mass <= 1", message="a mass target lies in (0, 1]"),
        dx.axiom("beam > 0", message="a beam width must be positive"),
        dx.axiom("unary_rounds >= 0", message="unary rounds must be nonnegative"),
        dx.axiom("threads > 0", message="a thread count must be positive"),
        dx.axiom("beta_fuel > 0", message="a beta-reduction budget must be positive"),
        dx.axiom("closure_rounds >= 0", message="closure rounds must be nonnegative"),
        dx.axiom("closure_nodes >= 0", message="a closure node cap is nonnegative"),
        dx.axiom(
            "isNone(max_category_arity) or max_category_arity > 0",
            message="a maximum category arity must be positive",
        ),
        dx.axiom(
            "isNone(max_category_size) or max_category_size > 0",
            message="a maximum category size must be positive",
        ),
    )

    def build(self) -> Budget:
        """Build the runtime budget."""
        return Budget(
            depth=self.depth,
            mass=self.mass,
            beam=self.beam,
            unary_rounds=self.unary_rounds,
            threads=self.threads,
            beta_fuel=self.beta_fuel,
            max_category_arity=self.max_category_arity,
            max_category_size=self.max_category_size,
        )


class CapacityEvidenceSpec(dx.Model, extra="forbid"):
    """Declared hypotheses for one CCG language-capacity theorem."""

    rule_family: str
    bounded_degree: bool
    prefix_closed: bool
    application_target_restrictions: bool

    @dx.validates("rule_family")
    @classmethod
    def _rule_family_is_installed(cls, value: str) -> str:
        """Return a rule family an installed ``lofi.rule_family`` entry names."""
        name = ensure_name("a CCG rule family", value)
        installed = Registry[object]("rule_family").available()

        if name not in installed:
            available = ", ".join(installed) or "(none)"
            msg = (
                f"unknown CCG rule family {name!r}; installed rule families are "
                f"{available}"
            )
            raise ValueError(msg)

        return name

    def build(self) -> CapacityEvidence:
        """Build the immutable theorem-hypothesis evidence."""
        return CapacityEvidence(
            self.rule_family,
            bounded_degree=self.bounded_degree,
            prefix_closed=self.prefix_closed,
            application_target_restrictions=self.application_target_restrictions,
        )


_DEFAULT_GOAL: Final = dx.field(default_factory=RootGoalSpec)
"""The one default grammar root declaration."""

_DEFAULT_MATCHING: Final = dx.field(default_factory=IdentityMatchingSpec)
"""The one default grammar matching declaration."""

_DEFAULT_NORMAL_FORM: Final = dx.field(default_factory=NoNormalFormSpec)
"""The one default grammar normal-form declaration."""

_DEFAULT_BUDGET: Final = dx.field(default_factory=BudgetSpec)
"""The one default grammar approximation budget."""


class PreparedGrammarPlugins(FrozenValue):
    """Runtime plugin selections fixed at grammar-declaration validation."""

    _noun = "prepared grammar plugin selection"
    __slots__ = (
        "_budget",
        "_capacity_evidence",
        "_category_measure",
        "_closure_nodes",
        "_closure_rounds",
        "_goal",
        "_goal_declaration",
        "_interpretations",
        "_lexicon",
        "_matching",
        "_matching_rows",
        "_max_category_arguments",
        "_max_category_order",
        "_max_hypotheses",
        "_name",
        "_normal_form",
        "_rules",
        "_signature_runtime",
    )

    def __init__(
        self,
        matching: TermMatching,
        matching_rows: tuple[tuple[str, TermMatching], ...],
        normal_form: NormalForm[Term],
        goal: GoalPlugin[GoalSpec],
        interpretations: tuple[PreparedInterpretation, ...],
        rules: PreparedRuleSet,
        signature_runtime: PreparedSignatureRuntime,
        name: str,
        goal_declaration: GoalSpec,
        lexicon: LexiconSpec,
        budget: Budget,
        closure_rounds: int,
        closure_nodes: int,
        capacity_evidence: CapacityEvidence | None,
        category_measure: CategoryMeasure,
        max_hypotheses: int,
        max_category_arguments: int | None,
        max_category_order: int | None,
        /,
    ) -> None:
        object.__setattr__(self, "_matching", matching)
        object.__setattr__(self, "_matching_rows", matching_rows)
        object.__setattr__(self, "_normal_form", normal_form)
        object.__setattr__(self, "_goal", goal)
        object.__setattr__(self, "_interpretations", interpretations)
        object.__setattr__(self, "_rules", rules)
        object.__setattr__(self, "_signature_runtime", signature_runtime)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_goal_declaration", goal_declaration)
        object.__setattr__(self, "_lexicon", lexicon)
        object.__setattr__(self, "_budget", budget)
        object.__setattr__(self, "_closure_rounds", closure_rounds)
        object.__setattr__(self, "_closure_nodes", closure_nodes)
        object.__setattr__(self, "_capacity_evidence", capacity_evidence)
        object.__setattr__(self, "_category_measure", category_measure)
        object.__setattr__(self, "_max_hypotheses", max_hypotheses)
        object.__setattr__(self, "_max_category_arguments", max_category_arguments)
        object.__setattr__(self, "_max_category_order", max_category_order)

    @property
    def name(self) -> str:
        """Return the validated grammar name."""
        return self._name

    @property
    def goal_declaration(self) -> GoalSpec:
        """Return the validated goal declaration."""
        return self._goal_declaration

    @property
    def lexicon(self) -> LexiconSpec:
        """Return the validated lexicon declaration."""
        return self._lexicon

    @property
    def budget(self) -> Budget:
        """Return the built formalism budget."""
        return self._budget

    @property
    def closure_rounds(self) -> int:
        """Return the declared closure depth."""
        return self._closure_rounds

    @property
    def closure_nodes(self) -> int:
        """Return the declared closure node budget."""
        return self._closure_nodes

    @property
    def capacity_evidence(self) -> CapacityEvidence | None:
        """Return the built capacity evidence, if declared."""
        return self._capacity_evidence

    @property
    def category_measure(self) -> CategoryMeasure:
        """Return the category measure derived from the signature."""
        return self._category_measure

    @property
    def max_hypotheses(self) -> int:
        """Return the withdrawn-hypothesis bound."""
        return self._max_hypotheses

    @property
    def max_category_arguments(self) -> int | None:
        """Return the lexicon's category-arity bound, if any."""
        return self._max_category_arguments

    @property
    def max_category_order(self) -> int | None:
        """Return the lexicon's category-order bound, if any."""
        return self._max_category_order

    @property
    def matching(self) -> TermMatching:
        """Return the built grammar congruence."""
        return self._matching

    @property
    def matching_rows(self) -> tuple[tuple[str, TermMatching], ...]:
        """Return prepared rule-level congruences in declaration order."""
        return self._matching_rows

    @property
    def normal_form(self) -> NormalForm[Term]:
        """Return the built derivation normal form."""
        return self._normal_form

    @property
    def goal(self) -> GoalPlugin[GoalSpec]:
        """Return the selected goal builder."""
        return self._goal

    @property
    def interpretations(
        self,
    ) -> tuple[PreparedInterpretation, ...]:
        """Return prepared interpretation builders in declaration order."""
        return self._interpretations

    @property
    def rules(self) -> PreparedRuleSet:
        """Return decoded declarations and their selected rule builders."""
        return self._rules

    @property
    def signature_runtime(self) -> PreparedSignatureRuntime:
        """Return prepared category syntax and functor metadata."""
        return self._signature_runtime


class GrammarDecl(GrammarSpec, extra="forbid"):
    """The complete declaration of one grammar."""

    formalism: Literal["declared"] = "declared"
    name: str
    signature: SignatureSpec
    rules: RuleSetSpec
    interpretations: tuple[NamedInterpretationSpec, ...] = dx.field(
        default_factory=default_interpretations
    )
    goal: GoalSpec = _DEFAULT_GOAL
    matching: MatchingSpec = _DEFAULT_MATCHING
    normal_form: NormalFormSpec = _DEFAULT_NORMAL_FORM
    budget: BudgetSpec = _DEFAULT_BUDGET
    max_hypotheses: int = _DEFAULT_MAX_HYPOTHESES
    capacity_evidence: CapacityEvidenceSpec | None = None

    __axioms__ = (
        _nonempty("name", "a grammar needs a name"),
        dx.axiom(
            "max_hypotheses >= 0",
            message="a withdrawn-hypothesis bound must be nonnegative",
        ),
        dx.axiom(
            "not isNone(signature.conjunctions) or "
            'not any(map (\\r -> r.rule == "coordination") rules.rules)',
            message=(
                "a coordination rule needs a conjunction inventory, and this "
                "signature declares none"
            ),
        ),
    )

    @dx.model_validator()
    def _interpretation_names_are_distinct(self) -> Self:
        """Return a grammar with one interpretation per document-level name."""
        names = tuple(interpretation.name for interpretation in self.interpretations)

        if len(set(names)) != len(names):
            msg = "a grammar declares one interpretation per name"
            raise ValueError(msg)

        return self

    @dx.model_validator()
    def _functor_rules_name_functors(self) -> Self:
        """Return a grammar whose functor-family rules name functor constructors.

        Raises
        ------
        ValueError
            If a composition, substitution, type-raising or clausal-adjunct
            rule names a constructor the signature gives no functor roles.
        """
        functors = self.signature.functor_names

        for rule in self.rules.rules:
            for slash in rule.functor_views():
                if slash not in functors:
                    msg = (
                        f"a {rule.rule} rule names {slash!r}, which is no functor "
                        f"constructor of signature {self.signature.name!r}"
                    )
                    raise ValueError(msg)

        return self

    @dx.derived
    def compile_envelope(self) -> GrammarCompileEnvelope:
        """Return the eagerly prepared declaration and its lowering route."""
        return _compile_grammar_declaration(self)

    @property
    def prepared_plugins(self) -> PreparedGrammarPlugins:
        """Return the runtime plugins resolved with this declaration."""
        return cast("PreparedGrammarPlugins", self.compile_envelope.prepared)

    @dx.model_validator()
    def _prepare_compile_envelope(self) -> Self:
        """Prepare canonical bytes once at the declaration boundary."""
        _ = self.compile_envelope
        return self

    @override
    def model_dump(
        self,
        *,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        by_alias: bool = False,
        exclude_none: bool = False,
        exclude_defaults: bool = False,
    ) -> dict[str, FieldValue]:
        """Render only fields supplied by the grammar declaration."""
        hidden = set() if exclude is None else set(exclude)
        hidden.add("compile_envelope")
        return super().model_dump(
            include=include,
            exclude=hidden,
            by_alias=by_alias,
            exclude_none=exclude_none,
            exclude_defaults=exclude_defaults,
        )


def _construct_term_matching(declaration: MatchingSpec, /) -> TermMatching:
    """Build a matching relation with a native representative kernel."""
    relation = matching_plugin(declaration).construct(declaration)

    if not isinstance(relation, TermMatching) or not relation.has_key:
        msg = f"matching {relation.name!r} has no native representative kernel"
        raise ValueError(msg)

    return relation


def _prepare_grammar_plugins(source: GrammarDecl, /) -> PreparedGrammarPlugins:
    """Resolve and build every runtime plugin selected by one grammar."""
    rule_set = source.rules
    declarations = [source.matching]
    declarations.extend(
        rule.matching for rule in rule_set.rules if rule.matching is not None
    )
    unique: list[MatchingSpec] = []

    for declaration in declarations:
        if declaration not in unique:
            unique.append(declaration)

    matching_rows = [
        (
            canonical_declaration_bytes(declaration).decode(),
            _construct_term_matching(declaration),
        )
        for declaration in unique
    ]

    normal_form = source.normal_form.build()
    signature = source.signature
    signature_runtime = signature.prepare_runtime()
    goal = source.goal
    lexicon = source.lexicon
    budget = source.budget
    capacity = source.capacity_evidence
    max_category_arguments, max_category_order = lexicon.category_complexity_bounds()
    return PreparedGrammarPlugins(
        matching_rows[0][1],
        tuple(matching_rows),
        normal_form,
        goal_plugin(source.goal),
        tuple(
            PreparedInterpretation(
                interpretation.name,
                plugin,
                plugin.prepare(interpretation.reading),
            )
            for interpretation in source.interpretations
            for plugin in (interpretation_plugin(interpretation.reading),)
        ),
        prepare_rule_set(rule_set),
        signature_runtime,
        source.name,
        goal,
        lexicon,
        budget.build(),
        budget.closure_rounds,
        budget.closure_nodes,
        None if capacity is None else capacity.build(),
        CategoryMeasure(
            tuple((name, slots[0]) for name, slots in signature.functor_slots.items())
        ),
        source.max_hypotheses,
        max_category_arguments,
        max_category_order,
    )


@cache
def _compile_grammar_declaration(source: GrammarDecl, /) -> GrammarCompileEnvelope:
    """Compile one immutable grammar declaration once per process."""
    return GrammarCompileEnvelope(
        source_version=GRAMMAR_SOURCE_VERSION,
        target_version=GRAMMAR_TARGET_VERSION,
        canonical_declaration=canonical_declaration_bytes(source),
        runtime=_prepare_grammar_plugins(source),
    )
