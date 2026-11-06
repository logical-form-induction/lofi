"""Installed concise declarations for finite and type-logical grammars."""

from itertools import product
from json import dumps
from typing import Final, Literal, Self

import didactic.api as dx

from lofi.grammars.tlg import ModalMatching
from lofi.spec._constructor import ConstructorSpec, PrefixConstructorNotationSpec
from lofi.spec._matching import MatchingPlugin, MatchingSpec
from lofi.spec._rule import (
    ConstantPatternSpec,
    FixedOperatorSpec,
    NodePatternSpec,
    PatternRuleSpec,
    RuleSetSpec,
    TermPatternSpec,
)
from lofi.spec.grammar._formalism import FormalismPlugin
from lofi.spec.grammar._grammar import (
    _DEFAULT_BUDGET,
    _DEFAULT_GOAL,
    _DEFAULT_MATCHING,
    _DEFAULT_MAX_HYPOTHESES,
    _DEFAULT_NORMAL_FORM,
    BudgetSpec,
    CapacityEvidenceSpec,
    GrammarDecl,
    NormalFormSpec,
    RootGoalSpec,
)
from lofi.spec.grammar._lexicon import (
    InlineLexiconSpec,
    LexicalEntrySpec,
    LexiconSpec,
)
from lofi.spec.grammar._signature import SignatureSpec
from lofi.spec.grammar._spec import GrammarSpec

__all__ = [
    "AB_PLUGIN",
    "CFG_PLUGIN",
    "MODAL_MATCHING_PLUGIN",
    "TLG_PLUGIN",
    "UNDIRECTED_CCG_PLUGIN",
    "ABFormalismSpec",
    "CFGFormalismSpec",
    "CFGProductionSpec",
    "ModalMatchingSpec",
    "ModeOrderSpec",
    "ModeRelationSpec",
    "TLGFormalismSpec",
    "UndirectedCCGFormalismSpec",
]


def _atom(name: str, sort: str = "Cat") -> str:
    """Return one vocabulary constant in canonical s-expression syntax."""
    return f'("#const[{sort}]" {dumps(name, ensure_ascii=False)})'


def _tlg_budget() -> BudgetSpec:
    """Return the NL-diamond fragment's structural closure budget."""
    return BudgetSpec(unary_rounds=3)


_TLG_BUDGET: Final = dx.field(default_factory=_tlg_budget)
"""The NL-diamond fragment's structural closure budget."""


class ABFormalismSpec(GrammarSpec, extra="forbid"):
    """An Ajdukiewicz-Bar-Hillel grammar with application as its sole rule."""

    formalism: Literal["ab"] = "ab"
    name: str
    signature: SignatureSpec
    rules: RuleSetSpec
    lexicon: LexiconSpec
    goal: RootGoalSpec = _DEFAULT_GOAL
    matching: MatchingSpec = _DEFAULT_MATCHING
    normal_form: NormalFormSpec = _DEFAULT_NORMAL_FORM
    budget: BudgetSpec = _DEFAULT_BUDGET
    max_hypotheses: int = _DEFAULT_MAX_HYPOTHESES
    capacity_evidence: CapacityEvidenceSpec | None = None

    __axioms__ = (
        dx.axiom("len(name) > 0", message="an AB grammar needs a name"),
        dx.axiom(
            "max_hypotheses >= 0",
            message="a withdrawn-hypothesis bound must be nonnegative",
        ),
    )

    @dx.model_validator()
    def _has_only_application(self) -> Self:
        """Return an application-only categorial grammar."""
        non_application = tuple(
            rule.rule for rule in self.rules.rules if not rule.yields_only_application()
        )

        if non_application:
            msg = f"an AB grammar declares non-application rules {non_application!r}"
            raise ValueError(msg)

        return self


def _declare_ab(source: ABFormalismSpec, /) -> GrammarDecl:
    """Expand an AB declaration to the generic grammar boundary."""
    return GrammarDecl(
        name=source.name,
        signature=source.signature,
        rules=source.rules,
        lexicon=source.lexicon,
        goal=source.goal,
        matching=source.matching,
        normal_form=source.normal_form,
        budget=source.budget,
        max_hypotheses=source.max_hypotheses,
        capacity_evidence=source.capacity_evidence,
    )


class UndirectedCCGFormalismSpec(GrammarSpec, extra="forbid"):
    """A CCG whose one functor constructor takes arguments on either side."""

    formalism: Literal["undirected-ccg"] = "undirected-ccg"
    name: str
    signature: SignatureSpec
    rules: RuleSetSpec
    lexicon: LexiconSpec
    goal: RootGoalSpec = _DEFAULT_GOAL
    matching: MatchingSpec = _DEFAULT_MATCHING
    normal_form: NormalFormSpec = _DEFAULT_NORMAL_FORM
    budget: BudgetSpec = _DEFAULT_BUDGET
    max_hypotheses: int = _DEFAULT_MAX_HYPOTHESES
    capacity_evidence: CapacityEvidenceSpec | None = None

    __axioms__ = (
        dx.axiom(
            "max_hypotheses >= 0",
            message="a withdrawn-hypothesis bound must be nonnegative",
        ),
    )

    @dx.model_validator()
    def _declares_one_undirected_functor(self) -> Self:
        """Return a signature with exactly one self-opposite functor."""
        roles = self.signature.functors

        if (
            len(roles) != 1
            or roles[0].argument_side != "either"
            or roles[0].opposite != roles[0].name
        ):
            msg = "an undirected CCG needs exactly one self-opposite functor"
            raise ValueError(msg)

        return self


def _declare_undirected_ccg(source: UndirectedCCGFormalismSpec, /) -> GrammarDecl:
    """Expand an undirected CCG to the generic grammar boundary."""
    return GrammarDecl(
        name=source.name,
        signature=source.signature,
        rules=source.rules,
        lexicon=source.lexicon,
        goal=source.goal,
        matching=source.matching,
        normal_form=source.normal_form,
        budget=source.budget,
        max_hypotheses=source.max_hypotheses,
        capacity_evidence=source.capacity_evidence,
    )


class CFGProductionSpec(dx.Model, extra="forbid"):
    """One named context-free production of arbitrary arity."""

    name: str
    parent: str
    children: tuple[str, ...]

    __axioms__ = (
        dx.axiom("len(name) > 0", message="a CFG production needs a name"),
        dx.axiom("len(parent) > 0", message="a CFG production needs a parent"),
        dx.axiom("all(children)", message="a CFG production child needs a name"),
    )


class CFGFormalismSpec(GrammarSpec, extra="forbid"):
    """A finite context-free grammar with declared matching and bounds."""

    formalism: Literal["cfg"] = "cfg"
    name: str = "cfg"
    nonterminals: tuple[str, ...]
    productions: tuple[CFGProductionSpec, ...]
    lexicon: LexiconSpec
    goal: str
    matching: MatchingSpec = _DEFAULT_MATCHING
    normal_form: NormalFormSpec = _DEFAULT_NORMAL_FORM
    budget: BudgetSpec = _DEFAULT_BUDGET

    __axioms__ = (
        dx.axiom("len(name) > 0", message="a CFG needs a name"),
        dx.axiom("len(nonterminals) > 0", message="a CFG needs a nonterminal"),
        dx.axiom("all(nonterminals)", message="a CFG nonterminal needs a name"),
        dx.axiom("len(productions) > 0", message="a CFG needs a production"),
        dx.axiom("len(goal) > 0", message="a CFG needs a goal"),
    )

    @dx.model_validator()
    def _references_are_declared(self) -> Self:
        """Return a CFG whose goals, productions and lexicon are in scope."""
        declared = frozenset(self.nonterminals)

        if len(declared) != len(self.nonterminals):
            msg = "a CFG repeats a nonterminal"
            raise ValueError(msg)

        if self.goal not in declared:
            msg = f"CFG goal {self.goal!r} is not a declared nonterminal"
            raise ValueError(msg)

        names = tuple(production.name for production in self.productions)
        if len(set(names)) != len(names):
            msg = "a CFG repeats a production name"
            raise ValueError(msg)

        structures = tuple(
            (production.parent, production.children) for production in self.productions
        )
        if len(set(structures)) != len(structures):
            msg = "a CFG repeats a production"
            raise ValueError(msg)

        for production in self.productions:
            for symbol in (production.parent, *production.children):
                if symbol not in declared:
                    msg = (
                        f"production {production.name!r} names undeclared "
                        f"nonterminal {symbol!r}"
                    )
                    raise ValueError(msg)

        if not isinstance(self.lexicon, InlineLexiconSpec):
            msg = "a concise CFG needs an inline lexicon"
            raise TypeError(msg)

        for entry in self.lexicon.entries:
            if entry.category not in declared:
                msg = (
                    f"lexical entry {entry.form!r} names undeclared nonterminal "
                    f"{entry.category!r}"
                )
                raise ValueError(msg)

        return self


def _declare_cfg(source: CFGFormalismSpec, /) -> GrammarDecl:
    """Expand a concise CFG to the generic grammar declaration."""
    if not isinstance(source.lexicon, InlineLexiconSpec):
        msg = "a concise CFG needs an inline lexicon"
        raise TypeError(msg)

    return GrammarDecl(
        name=source.name,
        signature=SignatureSpec(
            name=f"{source.name}-signature",
            primitives=source.nonterminals,
            syntax="s-expression",
        ),
        rules=RuleSetSpec(
            name=f"{source.name}-productions",
            rules=tuple(
                PatternRuleSpec(
                    name=production.name,
                    premises=tuple(
                        ConstantPatternSpec(name=child) for child in production.children
                    ),
                    conclusion=ConstantPatternSpec(name=production.parent),
                )
                for production in source.productions
            ),
        ),
        lexicon=InlineLexiconSpec(
            entries=tuple(
                LexicalEntrySpec(
                    form=entry.form,
                    category=_atom(entry.category),
                    weight=entry.weight,
                )
                for entry in source.lexicon.entries
            )
        ),
        goal=RootGoalSpec(categories=(_atom(source.goal),)),
        matching=source.matching,
        normal_form=source.normal_form,
        budget=source.budget,
    )


def _transitive_closure(
    modes: tuple[str, ...],
    relations: tuple[tuple[str, str], ...],
) -> frozenset[tuple[str, str]]:
    """Return the reflexive-transitive closure of a finite relation."""
    closure = set(relations)
    closure.update((mode, mode) for mode in modes)

    while True:
        added = {
            (lower, upper)
            for lower, middle in closure
            for other_middle, upper in closure
            if middle == other_middle and (lower, upper) not in closure
        }
        if not added:
            return frozenset(closure)
        closure.update(added)


def _maximal(
    values: tuple[str, ...], order: frozenset[tuple[str, str]]
) -> tuple[str, ...]:
    """Return the maximal values under a finite order."""
    return tuple(
        value
        for value in values
        if not any(value != other and (value, other) in order for other in values)
    )


def _minimal(
    values: tuple[str, ...], order: frozenset[tuple[str, str]]
) -> tuple[str, ...]:
    """Return the minimal values under a finite order."""
    return tuple(
        value
        for value in values
        if not any(value != other and (other, value) in order for other in values)
    )


class ModeRelationSpec(dx.Model, extra="forbid"):
    """One declared lower-to-upper relation between modes."""

    lower: str
    upper: str

    __axioms__ = (
        dx.axiom("len(lower) > 0", message="a lower mode needs a name"),
        dx.axiom("len(upper) > 0", message="an upper mode needs a name"),
    )


class ModeOrderSpec(dx.Model, extra="forbid"):
    """A finite lattice of type-logical modes."""

    modes: tuple[str, ...]
    order: tuple[ModeRelationSpec, ...] = ()

    __axioms__ = (
        dx.axiom("len(modes) > 0", message="a modal order needs a mode"),
        dx.axiom("all(modes)", message="a mode needs a name"),
    )

    @dx.model_validator()
    def _is_a_lattice(self) -> Self:
        """Return a finite antisymmetric order with unique meets and joins."""
        if len(set(self.modes)) != len(self.modes):
            msg = "a modal order repeats a mode"
            raise ValueError(msg)

        declared = frozenset(self.modes)
        for relation in self.order:
            lower = relation.lower
            upper = relation.upper
            if lower not in declared or upper not in declared:
                msg = f"modal order {lower!r} <= {upper!r} names an undeclared mode"
                raise ValueError(msg)

        closure = self.closure
        for left, right in closure:
            if left != right and (right, left) in closure:
                msg = f"modal order relates {left!r} and {right!r} both ways"
                raise ValueError(msg)

        for left, right in product(self.modes, repeat=2):
            lowers = tuple(
                mode
                for mode in self.modes
                if (mode, left) in closure and (mode, right) in closure
            )
            uppers = tuple(
                mode
                for mode in self.modes
                if (left, mode) in closure and (right, mode) in closure
            )
            if len(_maximal(lowers, closure)) != 1:
                msg = f"modes {left!r} and {right!r} have no unique meet"
                raise ValueError(msg)
            if len(_minimal(uppers, closure)) != 1:
                msg = f"modes {left!r} and {right!r} have no unique join"
                raise ValueError(msg)

        return self

    @property
    def closure(self) -> frozenset[tuple[str, str]]:
        """Return the reflexive-transitive closure."""
        return _transitive_closure(
            self.modes,
            tuple((relation.lower, relation.upper) for relation in self.order),
        )

    @property
    def canonical_order(self) -> tuple[tuple[str, str], ...]:
        """Return the closure in declaration order."""
        positions = {mode: at for at, mode in enumerate(self.modes)}
        return tuple(
            sorted(
                self.closure,
                key=lambda pair: (positions[pair[0]], positions[pair[1]]),
            )
        )


class ModalMatchingSpec(MatchingSpec, extra="forbid"):
    """Exact category matching qualified by a modal lattice."""

    kind: Literal["modal"] = "modal"
    modes: tuple[str, ...]
    order: tuple[ModeRelationSpec, ...]

    @classmethod
    def reify(cls, matching: ModalMatching, /) -> Self:
        """Recover the declaration from native modal matching."""
        return cls(
            modes=matching.modes,
            order=tuple(
                ModeRelationSpec(lower=lower, upper=upper)
                for lower, upper in matching.order
            ),
        )


def _modal_matching(declaration: ModalMatchingSpec, /) -> ModalMatching:
    """Build exact matching carrying the declared modal vocabulary."""
    return ModalMatching(
        declaration.modes,
        tuple((relation.lower, relation.upper) for relation in declaration.order),
    )


class TLGFormalismSpec(GrammarSpec, extra="forbid"):
    """A declared multimodal NL-diamond elimination fragment."""

    formalism: Literal["tlg"] = "tlg"
    name: str = "nl-diamond"
    primitives: tuple[str, ...]
    modes: ModeOrderSpec
    lexicon: LexiconSpec
    goal: str
    max_hypotheses: int = _DEFAULT_MAX_HYPOTHESES
    normal_form: NormalFormSpec = _DEFAULT_NORMAL_FORM
    budget: BudgetSpec = _TLG_BUDGET

    __axioms__ = (
        dx.axiom("len(name) > 0", message="a type-logical grammar needs a name"),
        dx.axiom(
            "len(primitives) > 0",
            message="a type-logical grammar needs a primitive",
        ),
        dx.axiom("all(primitives)", message="a type-logical primitive needs a name"),
        dx.axiom("len(goal) > 0", message="a type-logical grammar needs a goal"),
        dx.axiom(
            "max_hypotheses >= 0",
            message="a withdrawn-hypothesis bound must be nonnegative",
        ),
    )

    @dx.model_validator()
    def _goal_is_declared(self) -> Self:
        """Return a fragment whose goal is a declared primitive."""
        if len(set(self.primitives)) != len(self.primitives):
            msg = "a type-logical grammar repeats a primitive"
            raise ValueError(msg)
        if self.goal not in frozenset(self.primitives):
            msg = f"type-logical goal {self.goal!r} is not a declared primitive"
            raise ValueError(msg)
        return self


def _node(name: str, *arguments: TermPatternSpec) -> NodePatternSpec:
    """Return a node pattern over one fixed constructor."""
    return NodePatternSpec(
        operator=FixedOperatorSpec(name=name),
        arguments=arguments,
    )


def _mode_rules(mode: str) -> tuple[PatternRuleSpec, ...]:
    """Return the elimination and product/modality rules for one mode."""
    result = TermPatternSpec(name="A")
    argument = TermPatternSpec(name="B")
    forward = f"/_{mode}"
    backward = f"\\_{mode}"
    dot = f"dot_{mode}"
    diamond = f"diamond_{mode}"
    box_down = f"box-down_{mode}"
    return (
        PatternRuleSpec(
            name=f"/E_{mode}",
            premises=(_node(forward, result, argument), argument),
            conclusion=result,
        ),
        PatternRuleSpec(
            name=f"\\E_{mode}",
            premises=(argument, _node(backward, result, argument)),
            conclusion=result,
        ),
        PatternRuleSpec(
            name=f"dotI_{mode}",
            premises=(result, argument),
            conclusion=_node(dot, result, argument),
        ),
        PatternRuleSpec(
            name=f"diamondI_{mode}",
            premises=(result,),
            conclusion=_node(diamond, result),
        ),
        PatternRuleSpec(
            name=f"box-downE_{mode}",
            premises=(_node(box_down, result),),
            conclusion=result,
        ),
    )


def _declare_tlg(source: TLGFormalismSpec, /) -> GrammarDecl:
    """Expand an NL-diamond fragment to the generic grammar declaration."""
    constructors = tuple(
        ConstructorSpec(
            name=f"{base}_{mode}",
            kind=("Cat",) * (arity + 1),
            notation=PrefixConstructorNotationSpec(spelling=f"{base}_{mode}"),
        )
        for mode in source.modes.modes
        for base, arity in (
            ("/", 2),
            ("\\", 2),
            ("dot", 2),
            ("diamond", 1),
            ("box-down", 1),
        )
    )
    return GrammarDecl(
        name=source.name,
        signature=SignatureSpec(
            name=f"{source.name}-signature",
            primitives=source.primitives,
            constructors=constructors,
            syntax="s-expression",
        ),
        rules=RuleSetSpec(
            name=f"{source.name}-rules",
            rules=tuple(
                rule for mode in source.modes.modes for rule in _mode_rules(mode)
            ),
        ),
        lexicon=source.lexicon,
        goal=RootGoalSpec(categories=(_atom(source.goal),)),
        matching=ModalMatchingSpec(
            modes=source.modes.modes,
            order=tuple(
                ModeRelationSpec(lower=lower, upper=upper)
                for lower, upper in source.modes.canonical_order
            ),
        ),
        normal_form=source.normal_form,
        budget=source.budget,
        max_hypotheses=source.max_hypotheses,
    )


AB_PLUGIN = FormalismPlugin(spec=ABFormalismSpec, declare=_declare_ab)
CFG_PLUGIN = FormalismPlugin(spec=CFGFormalismSpec, declare=_declare_cfg)
MODAL_MATCHING_PLUGIN = MatchingPlugin(
    spec=ModalMatchingSpec,
    build=_modal_matching,
)
TLG_PLUGIN = FormalismPlugin(spec=TLGFormalismSpec, declare=_declare_tlg)
UNDIRECTED_CCG_PLUGIN = FormalismPlugin(
    spec=UndirectedCCGFormalismSpec,
    declare=_declare_undirected_ccg,
)
