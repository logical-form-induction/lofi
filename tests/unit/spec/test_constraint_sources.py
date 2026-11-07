from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING, cast

import didactic.api as dx
import pytest

from lofi.lexicon import ClampMode, ClampScope, UnsatisfiableAction
from lofi.spec.components.data import ConstraintSourceSpec
from lofi.spec.constraint_policy import PolicyDomain
from lofi.spec.constraint_sources import (
    ClampSpec,
    DeclaredConstraintSourceSpec,
    GoldConstraintSourceSettings,
    GoldDerivationConstraintSourceSpec,
    GoldLexiconConstraintSourceSpec,
    GoldLogicalFormConstraintSourceSpec,
    ItemFeaturePinSpec,
    ScoreQuantileSpec,
    SentenceTopPinSpec,
    SpanPatternPinSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue

    type GoldSourceSpec = (
        GoldDerivationConstraintSourceSpec
        | GoldLexiconConstraintSourceSpec
        | GoldLogicalFormConstraintSourceSpec
    )

GOLD_DOMAINS = {
    "gold-derivation": PolicyDomain.SYNTACTIC,
    "gold-lexicon": PolicyDomain.SYNTACTIC,
    "gold-logical-form": PolicyDomain.SEMANTIC,
}


@pytest.mark.parametrize(
    ("kind", "domain", "expected"),
    [
        (
            "gold-derivation",
            PolicyDomain.SYNTACTIC,
            GoldDerivationConstraintSourceSpec,
        ),
        ("gold-lexicon", PolicyDomain.SYNTACTIC, GoldLexiconConstraintSourceSpec),
        (
            "gold-logical-form",
            PolicyDomain.SEMANTIC,
            GoldLogicalFormConstraintSourceSpec,
        ),
    ],
)
def test_each_gold_source_is_reached_through_the_family_root(
    kind: str, domain: PolicyDomain, expected: type[ConstraintSourceSpec]
) -> None:
    declared = ConstraintSourceSpec.model_validate({"kind": kind, "policy": domain})

    assert type(declared) is expected
    assert cast("GoldSourceSpec", declared).constraint_route() == "target"
    assert declared.model_dump() == {
        "kind": kind,
        "policy": domain,
        "clamp": {
            "mode": ClampMode.HARD,
            "penalty": None,
            "scope": ClampScope.CONSTITUENT,
            "matching": None,
            "meaning_matching": {"kind": "identity"},
            "require_constituent": True,
            "negate": False,
            "on_unsatisfiable": {"kind": "error"},
        },
        "declared_output": None,
    }


@pytest.mark.parametrize(
    ("kind", "domain"),
    [
        ("gold-derivation", PolicyDomain.SYNTACTIC),
        ("gold-lexicon", PolicyDomain.SYNTACTIC),
        ("gold-logical-form", PolicyDomain.SEMANTIC),
    ],
)
def test_a_gold_source_declares_the_domain_it_serves(
    kind: str, domain: PolicyDomain
) -> None:
    declared = ConstraintSourceSpec.model_validate({"kind": kind, "policy": domain})
    settings = cast("GoldSourceSpec", declared).settings()

    assert declared.policy is domain
    assert isinstance(settings, GoldConstraintSourceSettings)
    assert settings.policy is domain
    assert settings.clamp.mode is ClampMode.HARD
    assert settings.clamp.penalty is None
    assert settings.clamp.scope is ClampScope.CONSTITUENT
    assert settings.clamp.matching is None
    assert settings.clamp.meaning_matching.name == "identity"
    assert settings.clamp.require_constituent
    assert not settings.clamp.negate
    assert settings.clamp.on_unsatisfiable.action is UnsatisfiableAction.ERROR


def test_a_soft_clamp_lowers_its_penalty_scope_matching_and_policy() -> None:
    declared = ClampSpec.model_validate(
        {
            "mode": "soft",
            "penalty": 0.5,
            "scope": "token",
            "matching": {"kind": "feature-erasure"},
            "meaning_matching": {"kind": "alpha"},
            "negate": True,
            "on_unsatisfiable": {"kind": "drop_clamp"},
        }
    )
    settings = declared.settings()

    assert settings.mode is ClampMode.SOFT
    assert settings.penalty == 0.5
    assert settings.scope is ClampScope.OCCURRENCE
    assert settings.matching is not None
    assert settings.matching.name == "feature-erasure"
    assert settings.meaning_matching.name == "alpha"
    assert settings.negate
    assert settings.on_unsatisfiable.action is UnsatisfiableAction.DROP_CLAMP


@pytest.mark.parametrize(
    "declaration",
    [
        {"mode": "soft"},
        {"mode": "soft", "penalty": 0.0},
        {"mode": "hard", "penalty": 0.5},
        {"scope": "token", "require_constituent": False},
        {"mode": "soft", "penalty": 0.5, "require_constituent": False},
        {"require_constituent": False, "negate": True},
    ],
)
def test_an_incoherent_clamp_declaration_is_refused(
    declaration: dict[str, JsonValue],
) -> None:
    with pytest.raises(dx.ValidationError):
        ClampSpec.model_validate(declaration)


def test_a_soft_negative_bracket_is_refused_at_the_declaration_boundary() -> None:
    with pytest.raises(dx.ValidationError, match="negative-bracket clamp is hard"):
        ClampSpec.model_validate(
            {
                "mode": "soft",
                "penalty": 0.5,
                "require_constituent": False,
            }
        )


def test_the_skip_item_policy_is_selected_from_its_open_group() -> None:
    settings = ClampSpec.model_validate(
        {"on_unsatisfiable": {"kind": "skip_item"}}
    ).settings()

    assert settings.on_unsatisfiable.action is UnsatisfiableAction.SKIP_ITEM


def test_a_gold_derivation_source_refuses_the_semantic_domain() -> None:
    with pytest.raises(dx.ValidationError, match="syntactic categories"):
        ConstraintSourceSpec.model_validate(
            {"kind": "gold-derivation", "policy": "semantic"}
        )


@pytest.mark.parametrize(
    ("kind", "domain", "message"),
    [
        ("gold-lexicon", "semantic", "syntactic categories"),
        ("gold-logical-form", "syntactic", "label forms with meanings"),
    ],
)
def test_each_lexical_gold_source_refuses_the_other_domain(
    kind: str, domain: str, message: str
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        ConstraintSourceSpec.model_validate({"kind": kind, "policy": domain})


@pytest.mark.parametrize("kind", GOLD_DOMAINS)
def test_a_gold_source_states_its_domain_rather_than_defaulting_one(
    kind: str,
) -> None:
    with pytest.raises(dx.ValidationError, match="policy"):
        ConstraintSourceSpec.model_validate({"kind": kind})


def test_no_registered_constraint_source_fixes_its_domain() -> None:
    kinds = sorted(point.name for point in entry_points(group="lofi.constraint_source"))
    served: set[PolicyDomain] = set()

    assert kinds

    for kind in kinds:
        for domain in PolicyDomain:
            try:
                declared = ConstraintSourceSpec.model_validate(
                    {"kind": kind, "policy": domain}
                )

            except dx.ValidationError:
                continue

            assert declared.policy is domain

            served.add(declared.policy)

        with pytest.raises(dx.ValidationError, match="policy"):
            ConstraintSourceSpec.model_validate({"kind": kind})

    assert served == set(PolicyDomain)


@pytest.mark.parametrize(("kind", "domain"), GOLD_DOMAINS.items())
def test_a_gold_source_refuses_unknown_fields(kind: str, domain: PolicyDomain) -> None:
    with pytest.raises(dx.ValidationError, match="coverage"):
        ConstraintSourceSpec.model_validate(
            {"kind": kind, "policy": domain, "coverage": 1.0}
        )


@pytest.mark.parametrize("kind", GOLD_DOMAINS)
def test_a_gold_source_refuses_an_undeclared_domain(kind: str) -> None:
    with pytest.raises(dx.ValidationError, match="pragmatic"):
        ConstraintSourceSpec.model_validate({"kind": kind, "policy": "pragmatic"})


def test_the_declared_source_reaches_every_pin_kind_through_one_family_variant() -> (
    None
):
    declared = ConstraintSourceSpec.model_validate(
        {
            "kind": "declared",
            "policy": "semantic",
            "pins": [
                {"kind": "item-feature", "item": "break", "feature": "BECOME"},
                {
                    "kind": "span-pattern",
                    "pattern": ["which", None],
                    "permitted": ["question"],
                },
                {"kind": "sentence-top", "permitted": ["t"]},
            ],
            "score_quantile": {
                "quantile": 0.9,
                "scores": {
                    "at://example.invalid/expression/low": 0.1,
                    "at://example.invalid/expression/high": 0.9,
                },
            },
            "clamp": {"negate": True},
        }
    )

    assert type(declared) is DeclaredConstraintSourceSpec

    pinned = declared
    settings = pinned.settings()

    assert tuple(type(pin) for pin in pinned.pins) == (
        ItemFeaturePinSpec,
        SpanPatternPinSpec,
        SentenceTopPinSpec,
    )
    assert [pattern.pattern for pattern in settings.patterns] == [
        ("break",),
        ("which", None),
        None,
    ]
    assert [pattern.permitted for pattern in settings.patterns] == [
        frozenset({"BECOME"}),
        frozenset({"question"}),
        frozenset({"t"}),
    ]
    assert (
        settings.score_quantile
        == ScoreQuantileSpec(
            quantile=0.9,
            scores={
                "at://example.invalid/expression/low": 0.1,
                "at://example.invalid/expression/high": 0.9,
            },
        ).settings()
    )
    assert settings.clamp.negate


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        ({"pins": []}, "at least one pin"),
        (
            {"pins": [{"kind": "span-pattern", "pattern": [], "permitted": ["S"]}]},
            "at least one matcher",
        ),
        (
            {
                "pins": [
                    {"kind": "sentence-top", "permitted": []},
                ]
            },
            "at least one label",
        ),
        (
            {
                "pins": [
                    {"kind": "item-feature", "item": "break", "feature": "BECOME"}
                ],
                "score_quantile": {"quantile": 0.9, "scores": {}},
            },
            "at least one item score",
        ),
    ],
)
def test_an_empty_declared_constraint_boundary_is_refused(
    declaration: dict[str, JsonValue], message: str
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        ConstraintSourceSpec.model_validate(
            {"kind": "declared", "policy": "syntactic", **declaration}
        )
