"""Runtime rules compiled from declared grammars for tests.

A test that studies rules declares its signature and rule inventory and
compiles them through the native grammar compiler, the route a
practitioner's grammar takes. One converter serves every test, so every
compile over one signature shares that signature's term store, and terms a
test builds from one compile are premises of the rules of another.
"""

from typing import TYPE_CHECKING, Final

from lofi.spec import ConstructorSpec, InfixConstructorNotationSpec
from lofi.spec._rule_laws import declared_rules as _declared_rules
from lofi.spec.grammar import (
    FunctorSpec,
    GrammarIso,
    IdentityMatchingSpec,
    SignatureSpec,
)
from lofi.spec.language import registered_language

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.spec import RuleSetSpec
    from lofi.spec.grammar import (
        AdoptedRules,
        ConjunctionsSpec,
        MatchingSpec,
        NormalFormSpec,
        OperatorSpec,
    )
    from lofi.spec.language import LanguageDecl

__all__ = ["ccg_signature", "declared_rules", "slash_constructors", "slash_views"]

_ISO: Final = GrammarIso()
"""The converter every test compile shares."""


def declared_rules(
    signature: SignatureSpec,
    rules: RuleSetSpec,
    /,
    *,
    matching: MatchingSpec | None = None,
    normal_form: NormalFormSpec | None = None,
    language: LanguageDecl | None = None,
) -> AdoptedRules:
    """Compile a rule inventory over a signature and adopt its rules.

    Parameters
    ----------
    signature : SignatureSpec
        The category signature, with at least one primitive.
    rules : RuleSetSpec
        The rule inventory.
    matching : MatchingSpec | None
        The premise congruence, identity unless given.
    normal_form : NormalFormSpec | None
        The normal form, the declaration default unless given.
    language : LanguageDecl | None
        The learner language, the registered propositional one unless given.

    Returns
    -------
    AdoptedRules
        The rule context, its store and syntax, and the adopted rules.
    """
    return _declared_rules(
        _ISO,
        registered_language("propositional") if language is None else language,
        signature,
        rules,
        matching=IdentityMatchingSpec() if matching is None else matching,
        normal_form=normal_form,
    )


def slash_constructors() -> tuple[ConstructorSpec, ...]:
    """Return the forward and backward slash category constructors."""
    return tuple(
        ConstructorSpec(
            name=name,
            kind=("Cat", "Cat", "Cat"),
            notation=InfixConstructorNotationSpec(
                spelling=name, precedence=10, associativity="left"
            ),
        )
        for name in ("/", "\\")
    )


def slash_views() -> tuple[FunctorSpec, ...]:
    """Return the directed functor views over the slash constructors."""
    return (
        FunctorSpec(name="/", constructor="/", argument_side="right", opposite="\\"),
        FunctorSpec(name="\\", constructor="\\", argument_side="left", opposite="/"),
    )


def ccg_signature(
    *primitives: str,
    operators: Sequence[OperatorSpec] = (),
    conjunctions: ConjunctionsSpec | None = None,
    name: str = "ccg",
) -> SignatureSpec:
    """Declare a CCG category signature over the slash constructors.

    Parameters
    ----------
    *primitives : str
        The atomic categories, the first of which anchors the lexicon.
    operators : Sequence[OperatorSpec]
        The featured operators.
    conjunctions : ConjunctionsSpec | None
        The conjunction inventory, if any.
    name : str
        The signature name.
    """
    return SignatureSpec(
        name=name,
        primitives=primitives,
        operators=tuple(operators),
        constructors=slash_constructors(),
        functors=slash_views(),
        conjunctions=conjunctions,
    )
