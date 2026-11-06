"""Installed specification plugins for the CCG runtime assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, cast, final

from lofi._frozen import FrozenValue
from lofi.grammars.ccg._assembly import (
    CCGAssembly,
    _Absorption,
    _ClausalAdjunct,
    _CoordinationPhase,
    _fixed_pattern,
    _pattern_probe,
)
from lofi.judge import (
    CompositionRole,
    GeneralizedComposition,
    GeneralizedSubstitution,
    NormalFormRole,
    PatternRule,
    RuleRestriction,
    TypeRaising,
)
from lofi.spec._rule import (
    AbsorptionSpec,
    ApplicationSpec,
    ClausalAdjunctSpec,
    CoercionSpec,
    CompositionSpec,
    CoordinationSpec,
    HeadSelection,
    PatternRuleSpec,
    RulePlugin,
    SubstitutionSpec,
    TypeRaisingSpec,
)
from lofi.terms import MVar, Node, PVar, Spine, TermRuleKernel

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.terms import (
        Pattern,
        Term,
        TermMatching,
        _CompiledNormalFormRole,
        _CompiledPattern,
        _CompiledRule,
    )

__all__ = [
    "ABSORPTION_PLUGIN",
    "APPLICATION_PLUGIN",
    "CLAUSAL_ADJUNCT_PLUGIN",
    "COERCION_PLUGIN",
    "COMPOSITION_PLUGIN",
    "COORDINATION_PLUGIN",
    "PATTERN_PLUGIN",
    "SUBSTITUTION_PLUGIN",
    "TYPE_RAISING_PLUGIN",
]


def _pattern(source: _CompiledPattern, context: CCGAssembly, /) -> Pattern:
    """Adopt one native-compiled pattern into its frozen runtime value."""
    kind = source.kind
    if kind == "term":
        sort = source.sort
        return MVar(
            cast("str", source.name),
            context.default_sort if sort is None else sort,
        )
    if kind == "constant":
        sort = source.sort
        return context.constant(
            cast("str", source.name),
            context.default_sort if sort is None else sort,
        )
    if kind == "spine":
        return Spine(
            cast("str", source.name),
            _pattern(cast("_CompiledPattern", source.core), context),
            tuple(source.operators),
            slot=cast("int", source.slot),
            min_order=cast("int", source.min_order),
            max_order=source.max_order,
        )
    operator_name = cast("str", source.operator_name)
    held_operator = (
        operator_name if source.operator_kind == "fixed" else PVar(operator_name)
    )
    payload: int | PVar | None
    if source.payload_kind is None:
        payload = None
    elif source.payload_kind == "fixed":
        payload = cast("int", source.payload_value)
    else:
        payload = PVar(cast("str", source.payload_name))
    return Node(
        held_operator,
        tuple(_pattern(argument, context) for argument in source.arguments),
        payload=payload,
    )


def _restriction(compiled: _CompiledRule, /) -> RuleRestriction | None:
    """Adopt the native-compiled restriction."""
    row = compiled.restriction
    return None if row is None else RuleRestriction(*row)


type _Adopt = Callable[[CCGAssembly, _CompiledRule], tuple[object, HeadSelection]]
"""One plugin's adoption of a native-compiled recipe."""


def _enforced(adopt: _Adopt, /) -> _Adopt:
    """Bind a recipe's restricted categories before adopting it.

    The chart enforces a restriction through the kernels the bound recipe
    builds, so every installed plugin adopts its recipe through this binding.
    """

    def bound(
        context: CCGAssembly, compiled: _CompiledRule
    ) -> tuple[object, HeadSelection]:
        row = compiled.restriction
        if row is None:
            return adopt(context, compiled)
        targets, patterns = row
        descents = [
            (slash, list(view.result_path))
            for view, _ in context.functors.values()
            for slash in sorted(view.slashes)
        ]
        return adopt(
            context,
            compiled.restricted(
                [context.category(target) for target in targets],
                [context.category(pattern) for pattern in patterns],
                descents,
            ),
        )

    return bound


def _matching(compiled: _CompiledRule, context: CCGAssembly, /) -> TermMatching:
    """Adopt the native-selected premise congruence."""
    return context.compiled_matching(compiled.matching_recipe)


def _patterns(
    compiled: _CompiledRule, context: CCGAssembly, /
) -> tuple[tuple[Pattern, ...], Pattern]:
    """Adopt premise and conclusion patterns from one native recipe."""
    return (
        tuple(_pattern(premise, context) for premise in compiled.premises),
        _pattern(cast("_CompiledPattern", compiled.conclusion), context),
    )


def _targets(
    source: list[str] | None, context: CCGAssembly, /
) -> tuple[Term, ...] | None:
    """Resolve finite target texts from one native recipe."""
    if source is None:
        return None
    return tuple(context.category(target) for target in source)


def _normal_form_role(source: _CompiledNormalFormRole, /) -> NormalFormRole:
    """Adopt one native-compiled normal form role."""
    direction = cast("Literal['forward', 'backward', 'undirected']", source.direction)
    if source.kind == "composition":
        return NormalFormRole(
            composition=CompositionRole(
                direction,
                principal_slot=cast("int", source.principal_slot),
                order=source.order,
                crossed=source.crossed,
            )
        )
    if source.kind == "raising":
        return NormalFormRole(raising=direction)
    return NormalFormRole(coordination=source.kind == "coordination")


@final
class _ClausalRecipe(FrozenValue):
    """Frozen runtime fields adopted from a native clausal recipe."""

    _noun = "compiled clausal recipe"
    __slots__ = (
        "argument",
        "head_policy",
        "matching",
        "name",
        "restriction",
        "source_core",
        "source_view",
        "target_view",
    )

    def __init__(self, compiled: _CompiledRule, /) -> None:
        object.__setattr__(self, "name", cast("str", compiled.name))
        object.__setattr__(self, "source_view", cast("str", compiled.source_view))
        object.__setattr__(self, "target_view", cast("str", compiled.target_view))
        object.__setattr__(self, "source_core", cast("str", compiled.source_core))
        object.__setattr__(self, "argument", cast("str", compiled.argument))
        object.__setattr__(self, "head_policy", "pass-through")
        object.__setattr__(self, "matching", None)
        object.__setattr__(self, "restriction", _restriction(compiled))


@final
class _CoordinationRecipe(FrozenValue):
    """Frozen runtime fields adopted from a native coordination recipe."""

    _noun = "compiled coordination recipe"
    __slots__ = (
        "conjunction_feature",
        "core_relation",
        "head_policy",
        "name",
        "phase",
        "targets",
    )

    def __init__(self, compiled: _CompiledRule, /) -> None:
        phase = cast("Literal['mark', 'merge']", compiled.phase)
        object.__setattr__(self, "name", cast("str", compiled.name))
        object.__setattr__(self, "phase", phase)
        object.__setattr__(
            self,
            "core_relation",
            cast(
                "Literal['same', 'different']",
                compiled.core_relation,
            ),
        )
        object.__setattr__(
            self,
            "conjunction_feature",
            cast("str", compiled.conjunction_feature),
        )
        object.__setattr__(
            self,
            "targets",
            (None if compiled.targets is None else tuple(compiled.targets)),
        )
        object.__setattr__(
            self,
            "head_policy",
            "non-absorbing" if phase == "mark" else "left-conjunct",
        )


def _adopt_application(
    context: CCGAssembly,
    compiled: _CompiledRule,
) -> tuple[object, HeadSelection]:
    """Adopt a typed native application recipe."""
    functors, slash = context.resolve_functor(cast("str", compiled.view))
    slot = (
        None
        if functors.forward_slash == functors.backward_slash
        else (0 if slash == functors.forward_slash else 1)
    )
    return (
        GeneralizedComposition(
            compiled,
            functors,
            slash,
            _matching(compiled, context),
            restriction=_restriction(compiled),
        ),
        HeadSelection("functor", slot),
    )


def _adopt_composition(
    context: CCGAssembly,
    compiled: _CompiledRule,
) -> tuple[object, HeadSelection]:
    """Adopt a native generalized composition recipe."""
    functors, slash = context.resolve_functor(cast("str", compiled.view))
    slot = (
        None
        if functors.forward_slash == functors.backward_slash
        else (0 if slash == functors.forward_slash else 1)
    )
    return (
        GeneralizedComposition(
            compiled,
            functors,
            slash,
            _matching(compiled, context),
            restriction=_restriction(compiled),
        ),
        HeadSelection("functor", slot),
    )


def _adopt_substitution(
    context: CCGAssembly,
    compiled: _CompiledRule,
) -> tuple[object, HeadSelection]:
    """Adopt a native generalized-substitution recipe."""
    functors, slash = context.resolve_functor(cast("str", compiled.view))
    slot = (
        None
        if functors.forward_slash == functors.backward_slash
        else (0 if slash == functors.forward_slash else 1)
    )
    return (
        GeneralizedSubstitution(
            compiled,
            functors,
            slash,
            _matching(compiled, context),
            restriction=_restriction(compiled),
        ),
        HeadSelection("functor", slot),
    )


def _adopt_type_raising(
    context: CCGAssembly,
    compiled: _CompiledRule,
) -> tuple[object, HeadSelection]:
    """Adopt a native type raising recipe."""
    functors, slash = context.resolve_functor(cast("str", compiled.view))
    return (
        TypeRaising(
            compiled,
            functors,
            slash,
            _matching(compiled, context),
            _targets(compiled.targets, context),
            _targets(compiled.raising_arguments, context),
            restriction=_restriction(compiled),
        ),
        HeadSelection("pass-through", 0),
    )


def _adopt_clausal_adjunct(
    context: CCGAssembly,
    compiled: _CompiledRule,
) -> tuple[object, HeadSelection]:
    """Adopt a typed native clausal adjunct recipe."""
    declaration = _ClausalRecipe(compiled)
    _, source_slash = context.resolve_functor(declaration.source_view)
    _, target_slash = context.resolve_functor(declaration.target_view)
    matching = _matching(compiled, context)
    rule = _ClausalAdjunct(
        context.store,
        matching,
        declaration,
        (
            source_slash,
            target_slash,
            context.category(declaration.source_core),
            context.category(declaration.argument),
        ),
        TermRuleKernel(compiled.empty_kernel(matching)),
    )
    return rule, HeadSelection("pass-through", 0)


def _adopt_coercion(
    context: CCGAssembly,
    compiled: _CompiledRule,
) -> tuple[object, HeadSelection]:
    """Adopt a ground coercion selected by the native compiler."""
    source_text = cast("str", compiled.source_category)
    target_text = cast("str", compiled.target_category)
    source = _fixed_pattern(context.category(source_text))
    target = _fixed_pattern(context.category(target_text))
    name = f"{compiled.label}:{compiled.coercion_member}:{source_text}=>{target_text}"
    return PatternRule(
        compiled,
        name,
        (source,),
        target,
        context.store,
        _matching(compiled, context),
        restriction=_restriction(compiled),
        probe=_pattern_probe(
            (source,), _matching(compiled, context), context.store.signature
        ),
    ), HeadSelection("pass-through", 0)


def _adopt_pattern(
    context: CCGAssembly,
    compiled: _CompiledRule,
) -> tuple[object, HeadSelection]:
    """Adopt a typed native general-pattern recipe."""
    premises, conclusion = _patterns(compiled, context)
    policy = cast(
        "Literal['functor', 'pass-through', 'non-absorbing', 'left-conjunct']",
        compiled.head_policy,
    )
    rule = PatternRule(
        compiled,
        cast("str", compiled.name),
        premises,
        conclusion,
        context.store,
        _matching(compiled, context),
        normal_form_role=_normal_form_role(
            cast("_CompiledNormalFormRole", compiled.normal_form_role)
        ),
        restriction=_restriction(compiled),
        probe=_pattern_probe(
            premises, _matching(compiled, context), context.store.signature
        ),
        targets=_targets(compiled.targets, context),
    )
    return rule, HeadSelection(policy, compiled.head_slot)


def _adopt_absorption(
    context: CCGAssembly,
    compiled: _CompiledRule,
) -> tuple[object, HeadSelection]:
    """Adopt a typed native absorption recipe."""
    premises, conclusion = _patterns(compiled, context)
    sources = tuple(cast("list[str]", compiled.absorbing))
    side = cast("Literal['left', 'right']", compiled.side)
    absorbing = tuple(context.category(source) for source in sources)
    pattern = PatternRule(
        compiled,
        cast("str", compiled.name),
        premises,
        conclusion,
        context.store,
        _matching(compiled, context),
        restriction=_restriction(compiled),
        probe=_pattern_probe(
            premises, _matching(compiled, context), context.store.signature
        ),
    )
    rule = _Absorption(
        pattern,
        sources,
        absorbing,
        side,
        TermRuleKernel(
            compiled.absorption_kernel(pattern.lowering.native, absorbing, side)
        ),
    )
    return rule, HeadSelection("non-absorbing", 1 if side == "left" else 0)


def _adopt_coordination(
    context: CCGAssembly,
    compiled: _CompiledRule,
) -> tuple[object, HeadSelection]:
    """Adopt a typed native coordination recipe."""
    premises, conclusion = _patterns(compiled, context)
    declaration = _CoordinationRecipe(compiled)
    targets = _targets(compiled.targets, context)
    pattern = PatternRule(
        compiled,
        declaration.name,
        premises,
        conclusion,
        context.store,
        _matching(compiled, context),
        normal_form_role=NormalFormRole(coordination=True),
        restriction=_restriction(compiled),
        probe=_pattern_probe(
            premises, _matching(compiled, context), context.store.signature
        ),
    )
    pairs = context.coordination_pairs(declaration.conjunction_feature)
    marking = context.coordination_marking(declaration.conjunction_feature)
    rule = _CoordinationPhase(
        pattern,
        pairs,
        declaration,
        targets,
        TermRuleKernel(
            compiled.coordination_phase_kernel(
                pattern.lowering.native,
                tuple(pairs),
                targets,
                declaration.phase,
                declaration.core_relation,
                marking,
            )
        ),
        store=context.store,
        marking=marking,
    )
    slot = 1 if declaration.phase == "mark" else 0
    return rule, HeadSelection(declaration.head_policy, slot)


APPLICATION_PLUGIN: Final = RulePlugin(
    spec=ApplicationSpec,
    adopt=_enforced(_adopt_application),
)
COMPOSITION_PLUGIN: Final = RulePlugin(
    spec=CompositionSpec,
    adopt=_enforced(_adopt_composition),
)
SUBSTITUTION_PLUGIN: Final = RulePlugin(
    spec=SubstitutionSpec,
    adopt=_enforced(_adopt_substitution),
)
TYPE_RAISING_PLUGIN: Final = RulePlugin(
    spec=TypeRaisingSpec,
    adopt=_enforced(_adopt_type_raising),
)
CLAUSAL_ADJUNCT_PLUGIN: Final = RulePlugin(
    spec=ClausalAdjunctSpec,
    adopt=_enforced(_adopt_clausal_adjunct),
)
COERCION_PLUGIN: Final = RulePlugin(
    spec=CoercionSpec,
    adopt=_enforced(_adopt_coercion),
)
PATTERN_PLUGIN: Final = RulePlugin(
    spec=PatternRuleSpec,
    adopt=_enforced(_adopt_pattern),
)
ABSORPTION_PLUGIN: Final = RulePlugin(
    spec=AbsorptionSpec,
    adopt=_enforced(_adopt_absorption),
)
COORDINATION_PLUGIN: Final = RulePlugin(
    spec=CoordinationSpec,
    adopt=_enforced(_adopt_coordination),
)
