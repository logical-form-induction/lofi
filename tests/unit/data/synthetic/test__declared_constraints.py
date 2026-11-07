"""Declared span pins built from their declaration through one plugin."""

from typing import Final

from lofi._types import ItemId
from lofi.data import PerOccurrenceRule, PinnedConstraintSource, SelectionPolicy
from lofi.data.synthetic._declared_constraints import PLUGIN
from lofi.data.synthetic._gold_sources import ConstraintSourceContext
from lofi.lexicon import Entry, StaticLexicon
from lofi.spec.constraint_policy import PolicyDomain
from lofi.spec.constraint_sources import DeclaredConstraintSourceSpec
from lofi.terms import Operator, Signature, TermStore

SIGNATURE: Final = Signature(
    "declared-constraint-plugin-test",
    ("Cat",),
    (Operator("NP", "Cat", family="primitive"),),
)
STORE: Final = TermStore(SIGNATURE)
NOUN_PHRASE: Final = STORE.op(SIGNATURE.by_name("NP"))
POLICY: Final = SelectionPolicy[str, str](PerOccurrenceRule())


def test_the_declared_plugin_builds_every_pin_and_the_score_quantile_tail() -> None:
    declaration = DeclaredConstraintSourceSpec.model_validate(
        {
            "kind": "declared",
            "policy": "syntactic",
            "pins": [
                {"kind": "item-feature", "item": "break", "feature": "NP"},
                {
                    "kind": "span-pattern",
                    "pattern": ["which", None],
                    "permitted": ["NP"],
                },
                {"kind": "sentence-top", "permitted": ["NP"]},
            ],
            "score_quantile": {
                "quantile": 0.5,
                "scores": {"at://example.invalid/expression/a": 1.0},
            },
        }
    )
    context = ConstraintSourceContext(
        syntactic=POLICY,
        semantic=POLICY,
        lexicon=StaticLexicon((Entry("break", NOUN_PHRASE),)),
        render=lambda term: term.op.name,
        interpretation=None,
    )
    source: PinnedConstraintSource[str, str] = PLUGIN.construct(declaration, context)

    assert source.policy is POLICY
    assert [pattern.pattern for pattern in source.patterns] == [
        ("break",),
        ("which", None),
        None,
    ]
    assert source.score_quantile is not None
    assert source.score_quantile.quantile == 0.5
    assert source.score_quantile.scores == {
        ItemId("at://example.invalid/expression/a"): 1.0
    }
    assert declaration.policy is PolicyDomain.SYNTACTIC
