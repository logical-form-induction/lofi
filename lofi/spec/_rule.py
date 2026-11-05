"""Serializable rule declarations and their construction boundary."""

from collections import abc
from types import MappingProxyType
from typing import (
    Final,
    Literal,
    Never,
    Protocol,
    Self,
    cast,
    override,
)

import didactic.api as dx

from lofi._frozen import FrozenValue
from lofi._types import CompositionRestriction, HeadPolicy, HeadSelection
from lofi.registry import Registry, RegistryError, resolve_plugin
from lofi.spec._matching import MatchingSpec
from lofi.terms import (
    MVar,
    Node,
    Pattern,
    PVar,
    Spine,
    _CompiledRule,
)

__all__ = [
    "AbsorptionSpec",
    "ApplicationSpec",
    "BuiltRuleSet",
    "ClausalAdjunctSpec",
    "CoercionSetSpec",
    "CoercionSpec",
    "CombinatorSetSpec",
    "CompositionSpec",
    "ConstantPatternSpec",
    "CoordinationSpec",
    "FixedOperatorSpec",
    "FixedPayloadSpec",
    "HeadPolicy",
    "HeadSelection",
    "NodePatternSpec",
    "NormalFormRoleSpec",
    "OperatorPatternSpec",
    "PatternContext",
    "PatternRuleSpec",
    "PatternSpec",
    "PayloadPatternSpec",
    "PreparedRuleSet",
    "RuleContext",
    "RulePlugin",
    "RuleRestrictionSpec",
    "RuleSetPlan",
    "RuleSetSpec",
    "RuleSpec",
    "SpinePatternSpec",
    "SubstitutionSpec",
    "TermPatternSpec",
    "TypeRaisingSpec",
    "VariableOperatorSpec",
    "VariablePayloadSpec",
    "prepare_rule_set",
    "resolve_rule_plugins",
    "rule_plugin",
]

_DEFAULT_CARRIED_HEAD: Final[HeadPolicy] = "pass-through"
_DEFAULT_CROSSED: Final = False


class PatternContext(Protocol):
    """Signature facts needed to build checked term patterns."""

    @property
    def default_sort(self) -> str:
        """Return the sort omitted term variables default to."""
        ...

    @property
    def sorts(self) -> abc.Collection[str]:
        """Return every sort declared by the signature."""
        ...

    def constant(self, name: str, sort: str, /) -> Node:
        """Return the fixed pattern for one vocabulary constant."""
        ...


class RuleSetPlan(Protocol):
    """Prepared rule-set fields consumed by a runtime rule context."""

    @property
    def name(self) -> str:
        """Return the rule-set name."""
        ...

    @property
    def restriction_mode(self) -> Literal["unrestricted", "grammar_specific"]:
        """Return the rule-set restriction mode."""
        ...


class RuleContext[RuntimeRule, RuntimeRuleSet, Category](PatternContext, Protocol):
    """Published resources shared by every installed rule builder."""

    def category(self, source: str, /) -> Category:
        """Resolve a category declaration."""
        ...

    def functor_names(self) -> tuple[str, ...]:
        """Return the published functor constructors in declaration order."""
        ...

    def rule_name(self, rule: RuntimeRule, /) -> str:
        """Return a runtime rule name."""
        ...

    def rule_set(
        self,
        declaration: RuleSetPlan,
        rules: tuple[RuntimeRule, ...],
        introduces: tuple[tuple[Category, ...] | None, ...],
        consumes: tuple[tuple[Category, ...] | None, ...],
        /,
    ) -> RuntimeRuleSet:
        """Build a rule set."""
        ...


class RulePlugin[Declaration, Context, RuntimeRule](FrozenValue):
    """Pair one open rule declaration with its runtime builder."""

    __slots__ = (
        "_adopt",
        "_spec",
    )

    _noun = "rule plugin"

    def __init__(
        self,
        *,
        spec: type[Declaration],
        adopt: abc.Callable[
            [Context, _CompiledRule],
            tuple[RuntimeRule, HeadSelection],
        ],
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_adopt", adopt)

    @property
    def spec(self) -> type[Declaration]:
        """Return the declaration class this plugin builds."""
        return self._spec

    def adopt_compiled(
        self,
        context: Context,
        compiled: _CompiledRule,
        /,
    ) -> tuple[RuntimeRule, HeadSelection]:
        """Adopt one rule recipe emitted by the native grammar compiler."""
        return self._adopt(context, compiled)

    @override
    def __repr__(self) -> str:
        """Return the declaration class the plugin builds."""
        return f"RulePlugin(spec={self._spec.__name__})"


class OperatorPatternSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """An operator position in a term pattern."""


class _OperatorPatternDeclaration(Protocol):
    """A declaration that builds an operator pattern."""

    def build(self) -> str | PVar:
        """Build the operator pattern."""
        ...


class FixedOperatorSpec(OperatorPatternSpec, extra="forbid"):
    """A fixed operator in a term pattern."""

    kind: Literal["fixed"] = "fixed"
    name: str

    __axioms__ = (dx.axiom("len(name) > 0", message="an operator needs a name"),)

    def build(self) -> str:
        """Build the fixed operator."""
        return self.name


class VariableOperatorSpec(OperatorPatternSpec, extra="forbid"):
    """A variable operator in a term pattern."""

    kind: Literal["variable"] = "variable"
    name: str

    __axioms__ = (
        dx.axiom("len(name) > 0", message="an operator variable needs a name"),
    )

    def build(self) -> PVar:
        """Build the operator variable."""
        return PVar(self.name)


class PayloadPatternSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """A payload position in a term pattern."""


class _PayloadPatternDeclaration(Protocol):
    """A declaration that builds a payload pattern."""

    def build(self) -> int | PVar:
        """Build the payload pattern."""
        ...


class FixedPayloadSpec(PayloadPatternSpec, extra="forbid"):
    """A fixed integer payload in a term pattern."""

    kind: Literal["fixed"] = "fixed"
    value: int

    def build(self) -> int:
        """Build the fixed payload."""
        return self.value


class VariablePayloadSpec(PayloadPatternSpec, extra="forbid"):
    """A variable payload in a term pattern."""

    kind: Literal["variable"] = "variable"
    name: str

    __axioms__ = (dx.axiom("len(name) > 0", message="a payload variable needs a name"),)

    def build(self) -> PVar:
        """Build the payload variable."""
        return PVar(self.name)


class PatternSpec(dx.TaggedUnion, discriminator="pattern", extra="forbid"):
    """A serializable term pattern."""


class _PatternDeclaration(Protocol):
    """A declaration that builds a term pattern."""

    def build(self, context: PatternContext, /) -> Pattern:
        """Build the term pattern."""
        ...


class TermPatternSpec(PatternSpec, extra="forbid"):
    """A whole-term variable pattern."""

    pattern: Literal["term"] = "term"
    name: str
    sort: str | None = None

    __axioms__ = (
        dx.axiom("len(name) > 0", message="a term variable needs a name"),
        dx.axiom(
            "isNone(sort) or len(sort) > 0", message="a term variable needs a sort"
        ),
    )

    def build(self, context: PatternContext, /) -> MVar:
        """Build the whole-term variable."""
        sort = context.default_sort if self.sort is None else self.sort

        if sort not in context.sorts:
            msg = f"pattern variable {self.name!r} names undeclared sort {sort!r}"
            raise ValueError(msg)

        return MVar(self.name, sort)


class ConstantPatternSpec(PatternSpec, extra="forbid"):
    """A vocabulary constant pattern resolved by its declared name."""

    pattern: Literal["constant"] = "constant"
    name: str
    sort: str | None = None

    __axioms__ = (
        dx.axiom("len(name) > 0", message="a constant pattern needs a name"),
        dx.axiom(
            "isNone(sort) or len(sort) > 0",
            message="a constant pattern sort needs a name",
        ),
    )

    def build(self, context: PatternContext, /) -> Node:
        """Resolve the named constant through the pattern context."""
        sort = context.default_sort if self.sort is None else self.sort

        if sort not in context.sorts:
            msg = f"constant pattern {self.name!r} names undeclared sort {sort!r}"
            raise ValueError(msg)

        return context.constant(self.name, sort)


class NodePatternSpec(PatternSpec, extra="forbid"):
    """An operator node pattern."""

    pattern: Literal["node"] = "node"
    operator: OperatorPatternSpec
    arguments: tuple[PatternSpec, ...] = ()
    payload: PayloadPatternSpec | None = None

    def build(self, context: PatternContext, /) -> Node:
        """Build the node pattern."""
        return Node(
            cast("_OperatorPatternDeclaration", self.operator).build(),
            tuple(
                cast("_PatternDeclaration", argument).build(context)
                for argument in self.arguments
            ),
            payload=(
                None
                if self.payload is None
                else cast("_PayloadPatternDeclaration", self.payload).build()
            ),
        )


class SpinePatternSpec(PatternSpec, extra="forbid"):
    """A bounded spine-context pattern."""

    pattern: Literal["spine"] = "spine"
    name: str
    core: PatternSpec
    operators: tuple[str, ...]
    slot: int = 0
    min_order: int = 0
    max_order: int | None

    __axioms__ = (
        dx.axiom("len(name) > 0", message="a spine variable needs a name"),
        dx.axiom("slot >= 0", message="a spine slot must be nonnegative"),
        dx.axiom("min_order >= 0", message="a spine minimum must be nonnegative"),
        dx.axiom(
            "isNone(max_order) or max_order >= min_order",
            message="a spine maximum is too small",
        ),
        dx.axiom("all(operators)", message="a spine operator needs a name"),
    )

    def build(self, context: PatternContext, /) -> Spine:
        """Build the spine pattern."""
        return Spine(
            self.name,
            cast("_PatternDeclaration", self.core).build(context),
            self.operators,
            slot=self.slot,
            min_order=self.min_order,
            max_order=self.max_order,
        )


class RuleRestrictionSpec(dx.Model, extra="forbid"):
    """A serializable rule-instance restriction the chart enforces.

    A restricted rule fires only when the target of its primary premise, the
    category reached by following functor results down to an atom, is one of
    ``primary_targets``, and its secondary premise is one of
    ``secondary_patterns``, each compared under the rule's matching. The
    primary premise of an application, composition or substitution rule is
    its functor and the secondary its other premise; a unary rule has only a
    primary premise; a binary pattern rule names its primary premise through
    its composition role. A restriction on a rule with no such premise is
    refused when the grammar compiles.
    """

    primary_targets: tuple[str, ...] = ()
    secondary_patterns: tuple[str, ...] = ()

    __axioms__ = (
        dx.axiom(
            "len(primary_targets) + len(secondary_patterns) > 0",
            message="a rule restriction needs a target or pattern",
        ),
        dx.axiom(
            "all(primary_targets) and all(secondary_patterns)",
            message="a rule restriction entry needs a name",
        ),
    )


class NormalFormRoleSpec(dx.Model, extra="forbid"):
    """A rule's structural role in derivation normal forms."""

    kind: Literal["composition", "raising", "coordination", "other"] = "other"
    direction: Literal["forward", "backward", "undirected"] | None = None
    principal_slot: int | None = None
    order: int | None = None
    crossed: bool = False

    __axioms__ = (
        dx.axiom(
            '(kind == "composition" and not isNone(direction) and '
            "not isNone(principal_slot)) or "
            '(kind == "raising" and not isNone(direction) and '
            "isNone(principal_slot) and isNone(order) and crossed == False) or "
            '(kind == "coordination" and isNone(direction) and '
            "isNone(principal_slot) and isNone(order) and crossed == False) or "
            '(kind == "other" and isNone(direction) and '
            "isNone(principal_slot) and isNone(order) and crossed == False)",
            message="normal-form role fields do not match its kind",
        ),
        dx.axiom(
            "isNone(principal_slot) or principal_slot >= 0",
            message="a normal-form principal slot must be nonnegative",
        ),
        dx.axiom(
            "isNone(order) or order >= 0",
            message="a normal-form order must be nonnegative",
        ),
    )


class RuleSpec(dx.TaggedUnion, discriminator="rule", extra="forbid"):
    """A serializable rule declaration.

    ``matching`` is the congruence the rule compares its premises under. A
    rule that declares none matches under the grammar's congruence, and one
    that declares identity matches exactly the categories its patterns name.
    A declaration that expands into several runtime rules, such as a
    combinator or coercion set, hands its ``restriction``, ``matching``,
    ``introduces`` and ``consumes`` to every rule it expands into.
    """

    restriction: RuleRestrictionSpec | None = None
    matching: MatchingSpec | None = None
    introduces: tuple[str, ...] | None = None
    consumes: tuple[str, ...] | None = None

    __axioms__ = (
        dx.axiom(
            "isNone(introduces) or (len(introduces) > 0 and all(introduces))",
            message="introduced categories must be nonempty names",
        ),
        dx.axiom(
            "isNone(consumes) or (len(consumes) > 0 and all(consumes))",
            message="consumed categories must be nonempty names",
        ),
    )

    def functor_views(self) -> tuple[str, ...]:
        """Return the functor constructors the rule names, in field order."""
        return ()

    def member_tags(self) -> tuple[str, ...]:
        """Return the plugin tags of the runtime rules the declaration yields.

        A declaration built by the plugin of its own tag yields that one tag;
        an expanding declaration yields the tags of the rules it expands into.
        """
        return (cast("str", self.rule),)

    def yields_only_application(self) -> bool:
        """Return whether every rule the declaration yields is functor application."""
        return False


def rule_plugin(declaration: RuleSpec, /) -> RulePlugin[RuleSpec, object, object]:
    """Return the installed plugin selected by a rule declaration's tag."""
    plugin = resolve_plugin(
        "rule", cast("str", declaration.rule), RulePlugin, type(declaration)
    )

    return cast("RulePlugin[RuleSpec, object, object]", plugin)


class ApplicationSpec(RuleSpec, extra="forbid"):
    """An application declaration distinct from positive-order composition."""

    rule: Literal["application"] = "application"
    view: str

    __axioms__ = (dx.axiom("len(view) > 0", message="application needs a view"),)

    @override
    def functor_views(self) -> tuple[str, ...]:
        """Return the functor constructor the rule names."""
        return (self.view,)

    @override
    def yields_only_application(self) -> bool:
        """Return that the declaration is one application rule."""
        return True


class CompositionSpec(RuleSpec, extra="forbid"):
    """A generalized-composition declaration."""

    rule: Literal["composition"] = "composition"
    direction: CompositionRestriction
    order: int | None
    view: str
    crossed: bool = _DEFAULT_CROSSED

    __axioms__ = (
        dx.axiom(
            "isNone(order) or order > 0",
            message="composition order must be positive",
        ),
        dx.axiom("len(view) > 0", message="composition needs a view"),
    )

    @override
    def functor_views(self) -> tuple[str, ...]:
        """Return the functor constructor the rule names."""
        return (self.view,)


class SubstitutionSpec(RuleSpec, extra="forbid"):
    """A generalized-substitution declaration."""

    rule: Literal["substitution"] = "substitution"
    order: int | None
    view: str
    crossed: bool = _DEFAULT_CROSSED

    __axioms__ = (
        dx.axiom(
            "isNone(order) or order > 0",
            message="substitution order must be positive",
        ),
        dx.axiom("len(view) > 0", message="substitution needs a view"),
    )

    @override
    def functor_views(self) -> tuple[str, ...]:
        """Return the functor constructor the rule names."""
        return (self.view,)


class TypeRaisingSpec(RuleSpec, extra="forbid"):
    r"""A type-raising declaration.

    ``targets`` are the categories ``T`` a premise raises to, and
    ``arguments`` the categories ``X`` that may raise to ``T/(T\X)`` or
    ``T\(T/X)``, each compared under the rule's matching. Either is ``None``
    for every category; the arguments default to the atoms ``N`` and ``S``.
    """

    rule: Literal["type-raising"] = "type-raising"
    view: str
    targets: tuple[str, ...] | None = None
    arguments: tuple[str, ...] | None = ("N", "S")

    __axioms__ = (
        dx.axiom("len(view) > 0", message="type raising needs a view"),
        dx.axiom(
            "isNone(targets) or len(targets) > 0",
            message="finite type-raising targets must be nonempty",
        ),
        dx.axiom(
            "isNone(targets) or all(targets)",
            message="a type-raising target needs a name",
        ),
        dx.axiom(
            "isNone(arguments) or (len(arguments) > 0 and all(arguments))",
            message="raisable categories must be nonempty names",
        ),
    )

    @override
    def functor_views(self) -> tuple[str, ...]:
        """Return the functor constructor the rule names."""
        return (self.view,)


class ClausalAdjunctSpec(RuleSpec, extra="forbid"):
    """A feature-bearing clausal-adjunct coercion declaration."""

    rule: Literal["clausal-adjunct"] = "clausal-adjunct"
    name: str
    source_view: str
    target_view: str
    source_core: str = "S"
    argument: str = "NP"

    __axioms__ = (
        dx.axiom("len(name) > 0", message="a clausal-adjunct rule needs a name"),
        dx.axiom(
            "len(source_view) > 0",
            message="a clausal-adjunct rule needs a source view",
        ),
        dx.axiom(
            "len(target_view) > 0",
            message="a clausal-adjunct rule needs a target view",
        ),
        dx.axiom(
            "len(source_core) > 0",
            message="a clausal-adjunct rule needs a source core",
        ),
        dx.axiom(
            "len(argument) > 0",
            message="a clausal-adjunct rule needs an argument",
        ),
    )

    @override
    def functor_views(self) -> tuple[str, ...]:
        """Return the source and target functor constructors."""
        return (self.source_view, self.target_view)


class PatternRuleSpec(RuleSpec, extra="forbid"):
    """A named general pattern-rule declaration."""

    rule: Literal["pattern"] = "pattern"
    head_policy: HeadPolicy = _DEFAULT_CARRIED_HEAD
    name: str
    premises: tuple[PatternSpec, ...]
    conclusion: PatternSpec
    targets: tuple[str, ...] | None = None
    normal_form_role: NormalFormRoleSpec = dx.field(default_factory=NormalFormRoleSpec)

    __axioms__ = (
        dx.axiom("len(name) > 0", message="a pattern rule needs a name"),
        dx.axiom(
            "isNone(targets) or len(targets) > 0",
            message="finite pattern-rule targets must be nonempty",
        ),
        dx.axiom(
            "isNone(targets) or all(targets)",
            message="a pattern-rule target needs a category",
        ),
    )

    @property
    def arity(self) -> int:
        """Return the premise count derived from the declaration."""
        return len(self.premises)


class AbsorptionSpec(RuleSpec, extra="forbid"):
    """A named absorption-rule declaration."""

    rule: Literal["absorption"] = "absorption"
    head_policy: HeadPolicy = "non-absorbing"
    name: str
    premises: tuple[PatternSpec, ...]
    conclusion: PatternSpec
    absorbing: tuple[str, ...]
    side: Literal["left", "right"]
    other_must_be_nonabsorbing: bool = True

    __axioms__ = (
        dx.axiom("len(name) > 0", message="an absorption rule needs a name"),
        dx.axiom(
            "len(absorbing) > 0",
            message="an absorption rule needs an absorbing category",
        ),
        dx.axiom(
            "other_must_be_nonabsorbing == True",
            message="an absorption rule must reject two absorbing premises",
        ),
        dx.axiom(
            "all(absorbing)",
            message="an absorbing category needs a name",
        ),
    )

    @property
    def arity(self) -> int:
        """Return the premise count derived from the declaration."""
        return len(self.premises)


class CoordinationSpec(RuleSpec, extra="forbid"):
    """A named coordination-rule declaration."""

    rule: Literal["coordination"] = "coordination"
    head_policy: HeadPolicy
    name: str
    premises: tuple[PatternSpec, ...]
    conclusion: PatternSpec
    phase: Literal["mark", "merge"]
    core_relation: Literal["same", "different"]
    conjunction_feature: str = "conj"
    targets: tuple[str, ...] | None = None

    __axioms__ = (
        dx.axiom("len(name) > 0", message="a coordination rule needs a name"),
        dx.axiom(
            "len(conjunction_feature) > 0",
            message="a coordination rule needs a conjunction feature",
        ),
        dx.axiom(
            "isNone(targets) or len(targets) > 0",
            message="finite coordination targets must be nonempty",
        ),
        dx.axiom(
            '(phase == "mark" and head_policy == "non-absorbing") or '
            '(phase == "merge" and head_policy == "left-conjunct")',
            message="a coordination phase requires its exact head policy",
        ),
        dx.axiom(
            'core_relation == "same" or phase == "mark"',
            message="different-core coordination is only a marking relation",
        ),
        dx.axiom(
            "isNone(targets) or all(targets)",
            message="a coordination target needs a name",
        ),
    )

    @property
    def arity(self) -> int:
        """Return the premise count derived from the declaration."""
        return len(self.premises)


class CoercionSpec(RuleSpec, extra="forbid"):
    """One ground unary member expanded from a shared coercion set."""

    rule: Literal["coercion"] = "coercion"
    label: str
    member: int
    source: str
    target: str

    __axioms__ = (
        dx.axiom("len(label) > 0", message="a coercion set needs a label"),
        dx.axiom("member >= 0", message="a coercion member index is nonnegative"),
        dx.axiom("len(source) > 0", message="a coercion needs a source category"),
        dx.axiom("len(target) > 0", message="a coercion needs a target category"),
    )


class CoercionSetSpec(RuleSpec, extra="forbid"):
    """Ground unary coercions sharing one scoring and grouping label.

    Each entry ``source => target`` expands into one ``coercion`` rule. Every
    member matches its premise under the set's ``matching``, or under the
    grammar's congruence when the set declares none.
    """

    rule: Literal["coercions"] = "coercions"
    label: str
    entries: tuple[str, ...]

    __axioms__ = (
        dx.axiom("len(label) > 0", message="a coercion set needs a label"),
        dx.axiom("len(entries) > 0", message="a coercion set needs an entry"),
        dx.axiom("all(entries)", message="a coercion entry must be nonempty"),
    )

    @override
    def member_tags(self) -> tuple[str, ...]:
        """Return the coercion tag every entry expands into."""
        return ("coercion",)


class CombinatorSetSpec(RuleSpec, extra="forbid"):
    """A concise inventory expanded over the published functor view.

    Every expanded rule matches its premises under the set's ``matching``, or
    under the grammar's congruence when the set declares none. Its type-raising
    rules raise ``raising_arguments`` to ``raising_targets``, as a
    :class:`TypeRaisingSpec` raises its ``arguments`` to its ``targets``.
    """

    rule: Literal["combinators"] = "combinators"
    application: bool = True
    composition: tuple[int | None, ...] = ()
    crossed_composition: tuple[int | None, ...] = ()
    substitution: tuple[int | None, ...] = ()
    crossed_substitution: tuple[int | None, ...] = ()
    direction: CompositionRestriction = "unrestricted"
    type_raising: bool = False
    raising_targets: tuple[str, ...] | None = None
    raising_arguments: tuple[str, ...] | None = ("N", "S")

    __axioms__ = (
        dx.axiom(
            "all(map (\\o -> isNone(o) or o > 0) composition)",
            message="composition orders must be positive or unbounded",
        ),
        dx.axiom(
            "all(map (\\o -> isNone(o) or o > 0) crossed_composition)",
            message="crossed composition orders must be positive or unbounded",
        ),
        dx.axiom(
            "all(map (\\o -> isNone(o) or o > 0) substitution)",
            message="substitution orders must be positive or unbounded",
        ),
        dx.axiom(
            "all(map (\\o -> isNone(o) or o > 0) crossed_substitution)",
            message="crossed substitution orders must be positive or unbounded",
        ),
        dx.axiom(
            "isNone(raising_targets) or len(raising_targets) > 0",
            message="finite raising targets must be nonempty",
        ),
        dx.axiom(
            "isNone(raising_targets) or type_raising == True",
            message="raising targets require type raising",
        ),
        dx.axiom(
            "isNone(raising_arguments) or "
            "(len(raising_arguments) > 0 and all(raising_arguments))",
            message="raisable categories must be nonempty names",
        ),
    )

    @override
    def member_tags(self) -> tuple[str, ...]:
        """Return the tag of every rule family the set expands into."""
        families = (
            ("application", self.application),
            ("composition", bool(self.composition or self.crossed_composition)),
            ("substitution", bool(self.substitution or self.crossed_substitution)),
            ("type-raising", self.type_raising),
        )
        return tuple(tag for tag, declared in families if declared)

    @override
    def yields_only_application(self) -> bool:
        """Return whether the set expands into application rules alone."""
        return not (
            self.composition
            or self.crossed_composition
            or self.substitution
            or self.crossed_substitution
            or self.type_raising
        )


class BuiltRuleSet[RuntimeRuleSet]:
    """A runtime rule set with its declared head selections."""

    __slots__ = ("_head_policies", "_head_selections", "_rule_set")

    def __init__(
        self,
        rule_set: RuntimeRuleSet,
        runtime_names: tuple[str, ...],
        head_selections: abc.Mapping[str, HeadSelection],
    ) -> None:
        if not isinstance(head_selections, abc.Mapping):
            msg = "head selections must be a name-to-selection mapping"
            raise TypeError(msg)

        declared = tuple(head_selections)

        if declared != runtime_names:
            msg = "head selections must name every runtime rule in declaration order"
            raise ValueError(msg)

        selections = dict(head_selections)

        if any(
            type(selection) is not HeadSelection for selection in selections.values()
        ):
            msg = "head selections must contain only HeadSelection values"
            raise TypeError(msg)

        object.__setattr__(self, "_rule_set", rule_set)
        object.__setattr__(
            self,
            "_head_selections",
            MappingProxyType(selections),
        )
        object.__setattr__(
            self,
            "_head_policies",
            MappingProxyType(
                {name: selection.policy for name, selection in selections.items()}
            ),
        )

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "a built rule set is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Raise AttributeError on field deletion."""
        del name
        msg = "a built rule set is immutable"

        raise AttributeError(msg)

    def __copy__(self) -> Self:
        """Return this immutable bundle."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable bundle."""
        del memo
        return self

    @override
    def __reduce__(self) -> Never:
        """Raise TypeError when pickling runtime rule identities."""
        msg = "a built rule set retains runtime rule identities and cannot be pickled"
        raise TypeError(msg)

    @property
    def rule_set(self) -> RuntimeRuleSet:
        """Return the runtime rule set."""
        return self._rule_set

    @property
    def head_policies(self) -> abc.Mapping[str, HeadPolicy]:
        """Return head policies by runtime rule name."""
        return self._head_policies

    @property
    def head_selections(self) -> abc.Mapping[str, HeadSelection]:
        """Return head selections by runtime rule name."""
        return self._head_selections


class RuleSetSpec(dx.Model, extra="forbid"):
    """A named runtime rule-set declaration."""

    name: str
    rules: tuple[RuleSpec, ...] = ()

    __axioms__ = (dx.axiom("len(name) > 0", message="a rule set needs a name"),)

    @property
    def restriction_mode(self) -> Literal["unrestricted", "grammar_specific"]:
        """Derive whether the set contains a grammar-specific restriction."""
        return (
            "grammar_specific"
            if any(rule.restriction is not None for rule in self.rules)
            else "unrestricted"
        )

    @property
    def plugin_tags(self) -> tuple[str, ...]:
        """Return every runtime rule plugin selected by this declaration."""
        return tuple(
            dict.fromkeys(tag for rule in self.rules for tag in rule.member_tags())
        )


class PreparedRuleSet(FrozenValue):
    """Decoded rule declarations and plugin selections fixed at validation."""

    _noun = "prepared rule set"
    __slots__ = (
        "_name",
        "_plugins",
        "_restriction_mode",
    )

    def __init__(
        self,
        declaration: RuleSetSpec,
        plugins: abc.Mapping[str, RulePlugin[RuleSpec, object, object]],
        /,
    ) -> None:
        object.__setattr__(self, "_name", declaration.name)
        object.__setattr__(self, "_restriction_mode", declaration.restriction_mode)
        object.__setattr__(self, "_plugins", MappingProxyType(dict(plugins)))
        for rule in declaration.rules:
            tag = cast("str", rule.rule)

            if tag not in rule.member_tags():
                continue

            plugin = plugins[tag]

            if plugin.spec is not type(rule):
                msg = (
                    f"the rule entry {rule.rule!r} builds {plugin.spec.__name__}, "
                    f"not {type(rule).__name__}"
                )
                raise TypeError(msg)

    @property
    def name(self) -> str:
        """Return the declared rule-set name."""
        return self._name

    @property
    def restriction_mode(self) -> Literal["unrestricted", "grammar_specific"]:
        """Return the validation-derived restriction mode."""
        return self._restriction_mode

    @property
    def plugins(
        self,
    ) -> abc.Mapping[str, RulePlugin[RuleSpec, object, object]]:
        """Return selected rule builders by plugin tag."""
        return self._plugins

    def adopt[RuntimeRule, RuntimeRuleSet, Category](
        self,
        plan: tuple[_CompiledRule, ...],
        context: RuleContext[RuntimeRule, RuntimeRuleSet, Category],
        /,
    ) -> BuiltRuleSet[RuntimeRuleSet]:
        """Adopt native-compiled recipes into the public runtime values."""
        adopted = tuple(
            self._plugins[row.tag].adopt_compiled(context, row) for row in plan
        )
        rules = tuple(cast("RuntimeRule", rule) for rule, _ in adopted)
        names = tuple(context.rule_name(rule) for rule in rules)
        _ensure_unique_names(names)
        rule_set = context.rule_set(
            self,
            rules,
            tuple(
                None
                if row.introduces is None
                else tuple(context.category(category) for category in row.introduces)
                for row in plan
            ),
            tuple(
                None
                if row.consumes is None
                else tuple(context.category(category) for category in row.consumes)
                for row in plan
            ),
        )
        return BuiltRuleSet(
            rule_set,
            names,
            {
                name: selection
                for name, (_, selection) in zip(names, adopted, strict=True)
            },
        )


def resolve_rule_plugins(
    tags: tuple[str, ...], /
) -> dict[str, RulePlugin[RuleSpec, object, object]]:
    """Resolve the installed rule plugin of every requested tag."""
    rules = Registry[object]("rule")
    resolved: dict[str, RulePlugin[RuleSpec, object, object]] = {}

    for tag in tags:
        member = rules.get(tag)

        if not isinstance(member, RulePlugin):
            msg = f"the rule entry {tag!r} names {member!r}, not a RulePlugin"
            raise RegistryError(msg, family="rule")

        resolved[tag] = cast("RulePlugin[RuleSpec, object, object]", member)

    return resolved


def prepare_rule_set(
    declaration: RuleSetSpec,
    /,
) -> PreparedRuleSet:
    """Decode and resolve one rule set at its validation boundary."""
    return PreparedRuleSet(
        declaration,
        resolve_rule_plugins(declaration.plugin_tags),
    )


def _ensure_unique_names(names: tuple[str, ...]) -> None:
    """Reject duplicate runtime rule names."""
    seen: dict[str, int] = {}

    for position, name in enumerate(names):
        if name in seen:
            msg = (
                f"a built rule set names {name!r} twice, "
                f"at positions {seen[name]} and {position}"
            )
            raise ValueError(msg)

        seen[name] = position
