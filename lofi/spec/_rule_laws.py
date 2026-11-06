"""Declared grammars that the built-in rule-law providers compile their cases from.

A law case checks one rule the native grammar compiler built, so each case
declares a grammar over a small signature and adopts its rules through
:meth:`~lofi.spec.grammar.GrammarIso.rules`, the route a practitioner's
grammar takes. One converter serves every case over a signature, and so every
rule the cases compile shares that signature's term store and its terms.

The law signatures declare the atoms ``x``, ``y``, ``z0``, ``z1``, ``t``,
``u`` and ``other`` with one functor arrangement each. The direct arrangement
has a forward ``fwd`` and a backward ``bwd`` constructor whose result is the
first child; the swapped one puts the result second; the undirected one has a
single self-opposite ``arrow``.
"""

from typing import TYPE_CHECKING, Final, Literal, final

from lofi._frozen import FrozenValue, Unpicklable
from lofi.spec._constructor import ConstructorSpec, InfixConstructorNotationSpec
from lofi.spec._rule import ApplicationSpec, RuleSetSpec
from lofi.spec.grammar import (
    FunctorSpec,
    GrammarDecl,
    GrammarIso,
    IdentityMatchingSpec,
    InlineLexiconSpec,
    LexicalEntrySpec,
    RootGoalSpec,
    SignatureSpec,
)
from lofi.spec.grammar._packs import _atom
from lofi.spec.language import registered_language

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.judge import Rule
    from lofi.spec._rule import RuleSpec
    from lofi.spec.grammar import AdoptedRules, MatchingSpec, NormalFormSpec
    from lofi.spec.language import LanguageDecl
    from lofi.terms import Functors, OpName, Term

__all__ = [
    "LAW_ATOMS",
    "LawGrammar",
    "LawView",
    "SlashRole",
    "declared_rules",
]

type LawView = Literal["direct", "swapped", "undirected"]
"""A functor arrangement of the law signatures."""

type SlashRole = Literal["forward", "backward"]
"""A distinguished slash of a functor view."""

LAW_ATOMS: Final = ("x", "y", "z0", "z1", "t", "u", "other")
"""The atomic categories every law signature declares."""

_LAW_LANGUAGE: Final = "propositional"
"""The registered learner language the law grammars compile against."""


def declared_rules(
    iso: GrammarIso,
    language: LanguageDecl,
    signature: SignatureSpec,
    rules: RuleSetSpec,
    /,
    *,
    matching: MatchingSpec,
    normal_form: NormalFormSpec | None = None,
) -> AdoptedRules:
    """Compile a rule inventory over a signature and adopt its rules.

    The grammar declares one lexical entry and goal, both the signature's
    first primitive, because a grammar declaration needs them; adopting its
    rules builds neither.

    Parameters
    ----------
    iso : GrammarIso
        The converter, whose linked language is shared by every grammar over
        one signature.
    language : LanguageDecl
        The learner language the grammar compiles against.
    signature : SignatureSpec
        The category signature, with at least one primitive.
    rules : RuleSetSpec
        The rule inventory, whose name the grammar takes.
    matching : MatchingSpec
        The grammar's premise congruence.
    normal_form : NormalFormSpec | None
        The grammar's normal form, or the declaration default.

    Returns
    -------
    AdoptedRules
        The context and the adopted rules.

    Raises
    ------
    ValueError
        If the signature declares no primitive.
    DeclarationError
        If the native compiler refuses the declaration.
    """
    if not signature.primitives:
        msg = "a rule grammar needs a primitive for its lexical entry and goal"
        raise ValueError(msg)

    primitive = signature.primitives[0]
    anchor = (
        _atom(primitive, signature.category_sort)
        if signature.syntax == "s-expression"
        else primitive
    )
    declaration = GrammarDecl(
        name=rules.name,
        signature=signature,
        rules=rules,
        lexicon=InlineLexiconSpec(
            entries=(LexicalEntrySpec(form=primitive, category=anchor),)
        ),
        goal=RootGoalSpec(categories=(anchor,)),
        matching=matching,
    )

    if normal_form is not None:
        declaration = declaration.with_(normal_form=normal_form)

    return iso.rules(declaration, language)


def _constructor(name: str, spelling: str) -> ConstructorSpec:
    """Declare one binary category constructor."""
    return ConstructorSpec(
        name=name,
        kind=("Cat", "Cat", "Cat"),
        notation=InfixConstructorNotationSpec(
            spelling=spelling, precedence=10, associativity="left"
        ),
    )


def _signature(view: LawView) -> SignatureSpec:
    """Declare the law signature of one functor arrangement."""
    if view == "undirected":
        return SignatureSpec(
            name="law-undirected",
            primitives=LAW_ATOMS,
            constructors=(_constructor("arrow", "->"),),
            functors=(
                FunctorSpec(
                    name="arrow",
                    constructor="arrow",
                    argument_side="either",
                    opposite="arrow",
                ),
            ),
        )

    result, argument = (1, 0) if view == "swapped" else (0, 1)

    return SignatureSpec(
        name=f"law-{view}",
        primitives=LAW_ATOMS,
        constructors=(_constructor("fwd", "/"), _constructor("bwd", "\\")),
        functors=(
            FunctorSpec(
                name="fwd",
                constructor="fwd",
                argument_side="right",
                opposite="bwd",
                result=result,
                argument=argument,
            ),
            FunctorSpec(
                name="bwd",
                constructor="bwd",
                argument_side="left",
                opposite="fwd",
                result=result,
                argument=argument,
            ),
        ),
    )


@final
class LawGrammar(Unpicklable, FrozenValue):
    """The compiled law signature of one functor arrangement.

    Every rule this grammar compiles shares one term store, so the terms a
    case draws can be given to the rule it checks and to that rule's rebuild.

    Parameters
    ----------
    view : LawView
        The functor arrangement.
    """

    _noun = "law grammar"
    _because = "retains live term identities"
    __slots__ = ("_atoms", "_functors", "_iso", "_language", "_signature", "_view")

    def __init__(self, view: LawView, /) -> None:
        iso = GrammarIso()
        language = registered_language(_LAW_LANGUAGE)
        signature = _signature(view)
        anchor = declared_rules(
            iso,
            language,
            signature,
            RuleSetSpec(
                name="laws", rules=(ApplicationSpec(view=signature.functors[0].name),)
            ),
            matching=IdentityMatchingSpec(),
        )
        context = anchor.context
        syntax = context.syntax
        object.__setattr__(self, "_view", view)
        object.__setattr__(self, "_iso", iso)
        object.__setattr__(self, "_language", language)
        object.__setattr__(self, "_signature", signature)
        object.__setattr__(
            self,
            "_atoms",
            {atom: syntax.parse(context.store, atom) for atom in LAW_ATOMS},
        )
        object.__setattr__(self, "_functors", context.functors)

    @property
    def view(self) -> LawView:
        """Return the functor arrangement."""
        return self._view

    def atom(self, name: str, /) -> Term:
        """Return one declared atomic category."""
        return self._atoms[name]

    def view_name(self, role: SlashRole, /) -> str:
        """Return the name of the functor view a slash role selects."""
        if self._view == "undirected":
            return "arrow"

        return "fwd" if role == "forward" else "bwd"

    def functor(self, role: SlashRole, /) -> tuple[Functors[Term], OpName]:
        """Return the functor view and slash a slash role selects."""
        return self._functors[self.view_name(role)]

    def view_of(self, slash: OpName, /) -> str:
        """Return the name of the functor view whose slash is the given one.

        Raises
        ------
        KeyError
            If no view of the arrangement has the slash.
        """
        for name, (_, held) in self._functors.items():
            if held == slash:
                return name

        raise KeyError(slash)

    def rules(
        self,
        rules: Sequence[RuleSpec],
        /,
        *,
        matching: MatchingSpec | None = None,
    ) -> tuple[Rule[Term], ...]:
        """Compile a rule inventory over the law signature.

        Parameters
        ----------
        rules : Sequence[RuleSpec]
            The rule declarations.
        matching : MatchingSpec | None
            The premise congruence, identity unless given.
        """
        adopted = declared_rules(
            self._iso,
            self._language,
            self._signature,
            RuleSetSpec(name="laws", rules=tuple(rules)),
            matching=IdentityMatchingSpec() if matching is None else matching,
        )

        return tuple(adopted.rule_set)

    def rule(self, rule: RuleSpec, /) -> Rule[Term]:
        """Compile one rule over the law signature under identity matching."""
        (compiled,) = self.rules((rule,))

        return compiled
