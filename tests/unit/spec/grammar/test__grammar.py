import json
from functools import partial
from typing import Final, Literal, Self

import didactic.api as dx
import pytest
from tests.unit.spec.grammar.test__lexicon import (
    _generated,
    _lexicon,
    _parametric,
    _structure,
    _tabular,
)
from tests.unit.spec.grammar.test__signature import (
    X,
    _application,
    _ccg,
    _constructor,
    _explicit,
    _rules,
    _store,
    slash_views,
    slashes,
)
from tests.unit.spec.test__matching import _isolated_variants

from lofi.deduce import CoverGoal, PenaltyForm, RootGoal, Tiling
from lofi.formalism import Budget
from lofi.judge import AllOf, EisnerNormalForm, HockenmaierBiskNormalForm, NoNormalForm
from lofi.spec import (
    ApplicationSpec,
    ClausalAdjunctSpec,
    CoordinationSpec,
    RuleSetSpec,
    RuleSpec,
    SubstitutionSpec,
    TypeRaisingSpec,
)
from lofi.spec.grammar import (
    AllOfNormalFormSpec,
    AlphaMatchingSpec,
    BudgetSpec,
    CapacityEvidenceSpec,
    CoverGoalSpec,
    EisnerNormalFormSpec,
    FeatureErasureMatchingSpec,
    FunctorSpec,
    GoalSpec,
    GrammarDecl,
    HockenmaierBiskNormalFormSpec,
    IdentityMatchingSpec,
    JointSpec,
    LexiconSpec,
    MatchingSpec,
    NoNormalFormSpec,
    NormalFormSpec,
    PowerLawRankLawSpec,
    ProductMatchingSpec,
    RankLawSpec,
    RootGoalSpec,
    SignatureSpec,
    SubsumptionMatchingSpec,
    UndirectedEisnerNormalFormSpec,
    UniformRankLawSpec,
)
from lofi.spec.grammar import _grammar as declaration_module
from lofi.terms import Identity, SExpSyntax, Term

ROOTS: Final[tuple[type[dx.TaggedUnion], ...]] = (
    GoalSpec,
    MatchingSpec,
    NormalFormSpec,
    LexiconSpec,
    RankLawSpec,
    JointSpec,
)


def _declaration(*, name: str = "toy") -> GrammarDecl:
    return GrammarDecl(
        name=name,
        signature=_explicit(),
        rules=_rules(),
        lexicon=_lexicon(),
        goal=RootGoalSpec(categories=('("#const[Cat]" "S")',)),
    )


def _ccg_with_default_rules() -> SignatureSpec:
    """Return the CCG fixture with every atom named by the default rules."""
    signature = _ccg()

    return signature.with_(primitives=(*signature.primitives, "VP"))


def test_capacity_evidence_refuses_a_rule_family_no_plugin_installs() -> None:
    with pytest.raises(dx.ValidationError, match="installed rule families are"):
        CapacityEvidenceSpec(
            rule_family="mildly_ccg",
            bounded_degree=True,
            prefix_closed=True,
            application_target_restrictions=False,
        )


def test_capacity_evidence_builds_the_declared_hypotheses() -> None:
    declaration = CapacityEvidenceSpec(
        rule_family="prefix_closed_vw_ccg",
        bounded_degree=True,
        prefix_closed=True,
        application_target_restrictions=False,
    )

    evidence = declaration.build()

    assert evidence.rule_family == "prefix_closed_vw_ccg"
    assert evidence.bounded_degree
    assert evidence.prefix_closed
    assert not evidence.application_target_restrictions


def test_the_compile_envelope_is_derived_eagerly_once_per_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dx.Model] = []
    canonical = declaration_module.canonical_declaration_bytes

    def counted(source: dx.Model, /) -> bytes:
        calls.append(source)
        return canonical(source)

    monkeypatch.setattr(declaration_module, "canonical_declaration_bytes", counted)

    declaration = _declaration(name="compile-envelope-test")
    envelope = declaration.compile_envelope

    assert calls == [declaration, declaration.matching]
    assert declaration.compile_envelope is envelope

    reparsed = GrammarDecl.model_validate_json(declaration.model_dump_json())

    assert reparsed.compile_envelope is envelope
    assert calls == [declaration, declaration.matching]

    changed = declaration.with_(name="changed")

    assert calls == [
        declaration,
        declaration.matching,
        changed,
        changed.matching,
    ]
    assert changed.compile_envelope is changed.compile_envelope
    assert changed.compile_envelope is not envelope
    assert changed.compile_envelope.canonical_declaration != (
        envelope.canonical_declaration
    )


def test_the_grammar_compile_envelope_is_frozen_and_not_declared() -> None:
    declaration = _declaration()
    envelope = declaration.compile_envelope

    assert envelope.source_version == "lofi-grammar-decl"
    assert envelope.target_version == "lofi-grammar-native"
    assert set(envelope.model_dump()) == {
        "source_version",
        "target_version",
        "canonical_declaration",
    }
    assert "compile_envelope" not in declaration.model_dump()
    assert declaration.model_dump(include={"compile_envelope"}) == {}
    assert "compile_envelope" not in json.loads(declaration.model_dump_json())

    with pytest.raises(AttributeError, match="immutable"):
        envelope.source_version = "other"

    assert GrammarDecl.model_validate_json(envelope.canonical_declaration) == (
        declaration
    )


@pytest.mark.parametrize("root", ROOTS, ids=lambda root: root.__name__)
def test_a_root_refuses_a_kind_nothing_claims_and_a_payload_naming_none(
    root: type[dx.TaggedUnion],
) -> None:
    with pytest.raises(dx.ValidationError):
        root.model_validate({"kind": "unclaimed"})

    with pytest.raises(dx.ValidationError):
        root.model_validate({"depth": 3})


@pytest.mark.parametrize("root", ROOTS, ids=lambda root: root.__name__)
def test_every_registered_variant_rebuilds_from_its_json(
    root: type[dx.TaggedUnion],
) -> None:
    samples: dict[str, dx.Model] = {
        "identity": IdentityMatchingSpec(),
        "root": RootGoalSpec(),
        "cover": CoverGoalSpec(categories=("S",), max_fragments=3),
        "alpha": AlphaMatchingSpec(),
        "feature-erasure": FeatureErasureMatchingSpec(),
        "product": ProductMatchingSpec(
            left=IdentityMatchingSpec(), right=FeatureErasureMatchingSpec()
        ),
        "subsumption": SubsumptionMatchingSpec(),
        "none": NoNormalFormSpec(),
        "eisner": EisnerNormalFormSpec(),
        "hockenmaier-bisk": HockenmaierBiskNormalFormSpec(),
        "undirected-eisner": UndirectedEisnerNormalFormSpec(),
        "all-of": AllOfNormalFormSpec(
            normal_forms=(EisnerNormalFormSpec(), NoNormalFormSpec())
        ),
        "inline": _lexicon(),
        "structured": _structure(),
        "generated": _generated(),
        "uniform": UniformRankLawSpec(),
        "power-law": PowerLawRankLawSpec(exponent=1.5),
        "parametric": _parametric(),
        "tabular": _tabular(),
    }
    # A variant another package declares is that package's to test, and the
    # plugin case has its own test below; this one covers what _decl declares.
    variants = {
        tag: variant
        for tag, variant in root.__variants__.items()
        if variant.__module__ == root.__module__
    }

    assert variants

    for tag, variant in variants.items():
        assert isinstance(tag, str)

        sample = samples[tag]

        assert type(sample) is variant

        rebuilt = root.model_validate_json(sample.model_dump_json())

        assert type(rebuilt) is variant
        assert rebuilt == sample


def test_each_normal_form_variant_builds_its_filter() -> None:
    assert type(NoNormalFormSpec().build()) is NoNormalForm
    assert type(EisnerNormalFormSpec().build()) is EisnerNormalForm
    assert UndirectedEisnerNormalFormSpec().build().name == "undirected-eisner"
    assert (
        type(AllOfNormalFormSpec(normal_forms=(EisnerNormalFormSpec(),)).build())
        is AllOf
    )

    built = HockenmaierBiskNormalFormSpec().build()

    assert type(built) is HockenmaierBiskNormalForm
    assert built.alphabet == frozenset(
        {
            "OTHER",
            "FTR",
            "BTR",
            "UTR",
            "COORD",
            "FC",
            "FC1",
            "FCn",
            "BC",
            "BC1",
            "BCn",
            "UC",
            "UC1",
            "UCn",
        }
    )

    with pytest.raises(dx.ValidationError):
        HockenmaierBiskNormalFormSpec.model_validate(
            {"kind": "hockenmaier-bisk", "max_order": 1}
        )

    exact = HockenmaierBiskNormalFormSpec(
        order_bound=3,
        track_child_order=True,
        raised_secondary=False,
    ).build()

    assert isinstance(exact, HockenmaierBiskNormalForm)
    assert exact.order_bound == 3
    assert exact.track_child_order

    with pytest.raises(dx.ValidationError):
        HockenmaierBiskNormalFormSpec(order_bound=0)

    with pytest.raises(dx.ValidationError):
        AllOfNormalFormSpec(normal_forms=())


def test_reified_roots_forward_cooperative_class_keywords() -> None:
    received: list[tuple[str, bool]] = []

    class Cooperative:
        def __init_subclass__(cls, *, enabled: bool = False) -> None:
            received.append((cls.__name__, enabled))
            super().__init_subclass__()

    with _isolated_variants(MatchingSpec):

        class CooperativeMatchingSpec(MatchingSpec, Cooperative, enabled=True):
            kind: Literal["cooperative-matching"] = "cooperative-matching"

            @classmethod
            def reify(cls, matching: Identity[Term], /) -> Self:
                del matching

                return cls()

        assert received == [("CooperativeMatchingSpec", True)]
        assert (
            MatchingSpec.__variants__["cooperative-matching"] is CooperativeMatchingSpec
        )

    assert "cooperative-matching" not in MatchingSpec.__variants__


def test_a_grammar_refuses_coordination_over_a_signature_without_conjunctions() -> None:
    merge = CoordinationSpec(
        name="coordination-merge",
        premises=(X, X),
        conclusion=X,
        phase="merge",
        core_relation="same",
        head_policy="left-conjunct",
    )
    declaration = GrammarDecl(
        name="coordinating",
        signature=_ccg(),
        rules=RuleSetSpec(name="coordination", rules=(merge,)),
        lexicon=_lexicon(),
        matching=FeatureErasureMatchingSpec(),
    )
    bare = SignatureSpec(
        constructors=slashes(), functors=slash_views(), primitives=("S", "NP")
    )

    with pytest.raises(dx.ValidationError, match="needs a conjunction inventory"):
        declaration.with_(signature=bare)

    with pytest.raises(dx.ValidationError, match="needs a conjunction inventory"):
        GrammarDecl.model_validate(
            {
                **json.loads(declaration.model_dump_json()),
                "signature": json.loads(bare.model_dump_json()),
            }
        )

    assert declaration.with_(signature=bare, rules=_application()).rules == (
        _application()
    )

    with pytest.raises(dx.ValidationError, match="needs a conjunction inventory"):
        declaration.with_(signature=_explicit(), matching=IdentityMatchingSpec())


@pytest.mark.parametrize(
    "rule",
    [
        ApplicationSpec(view="/"),
        SubstitutionSpec(order=1, view="\\"),
        TypeRaisingSpec(view="/", arguments=None),
        ClausalAdjunctSpec(name="adjunct", source_view="\\", target_view="\\"),
    ],
    ids=lambda rule: rule.rule,
)
def test_a_grammar_refuses_a_functor_family_rule_naming_no_functor(
    rule: RuleSpec,
) -> None:
    declaration = _declaration().with_(
        signature=_ccg_with_default_rules(), matching=FeatureErasureMatchingSpec()
    )
    named = RuleSetSpec(name="functors", rules=(rule,))

    assert declaration.with_(rules=named).rules == named

    with pytest.raises(dx.ValidationError, match="no functor constructor"):
        _declaration().with_(rules=named)


@pytest.mark.parametrize(
    ("source", "target"), [("\\", "pair"), ("pair", "\\")], ids=["target", "source"]
)
def test_a_grammar_refuses_a_clausal_adjunct_with_either_slash_naming_no_functor(
    source: str, target: str
) -> None:
    declaration = _declaration().with_(
        signature=_ccg_with_default_rules(), matching=FeatureErasureMatchingSpec()
    )
    rule = ClausalAdjunctSpec(name="adjunct", source_view=source, target_view=target)

    with pytest.raises(dx.ValidationError, match="names 'pair', which is no functor"):
        declaration.with_(rules=RuleSetSpec(name="functors", rules=(rule,)))


def test_a_grammar_accepts_rules_naming_declared_functor_views() -> None:
    signature = SignatureSpec(
        primitives=("S", "NP"),
        constructors=(
            _constructor("fwd", spelling="/"),
            _constructor("bwd"),
        ),
        functors=(
            FunctorSpec(
                name="/", constructor="fwd", argument_side="right", opposite="bwd"
            ),
            FunctorSpec(
                name="bwd", constructor="bwd", argument_side="left", opposite="/"
            ),
        ),
    )
    rules = RuleSetSpec(
        name="application",
        rules=(
            ApplicationSpec(view="/"),
            ApplicationSpec(view="bwd"),
        ),
    )

    assert _declaration().with_(signature=signature, rules=rules).rules == rules


def test_a_root_goal_builds_targets_under_the_rule_congruence() -> None:
    store = _store(_explicit())
    signature = store.signature
    syntax = SExpSyntax(signature)
    matching = Identity[Term]()
    targets = RootGoalSpec(
        categories=('("#const[Cat]" "S")', '("#const[Cat]" "NP")')
    ).build(partial(syntax.parse, store), matching)

    assert targets.categories == (
        syntax.parse(store, '("#const[Cat]" "S")'),
        syntax.parse(store, '("#const[Cat]" "NP")'),
    )
    assert targets.matching is matching

    with pytest.raises(dx.ValidationError):
        RootGoalSpec(categories=())

    with pytest.raises(dx.ValidationError):
        RootGoalSpec(categories=("S", ""))


def test_goal_variants_build_weighted_roots_and_declared_fragment_covers() -> None:
    store = _store(_explicit())
    syntax = SExpSyntax(store.signature)
    matching = Identity[Term]()
    root_spec = RootGoalSpec(
        categories=('("#const[Cat]" "S")',),
        root_weights=(-0.75,),
    )
    targets = root_spec.build(partial(syntax.parse, store), matching)
    root = root_spec.build_goal(targets)
    cover = CoverGoalSpec(
        categories=('("#const[Cat]" "S")',),
        max_fragments=4,
        penalty=0.5,
        penalty_form="each-fragment",
        tiling="lexical",
    ).build_goal(targets)

    assert isinstance(root, RootGoal)
    assert root.weights == (-0.75,)
    assert isinstance(cover, CoverGoal)
    assert cover.max_fragments == 4
    assert cover.penalty_form is PenaltyForm.EACH
    assert cover.tiling is Tiling.LEXICAL


def test_a_budget_declaration_builds_the_budget_it_names() -> None:
    declared = BudgetSpec(
        depth=3,
        mass=0.5,
        beam=2,
        unary_rounds=0,
        threads=8,
        beta_fuel=10,
        max_category_arity=4,
        max_category_size=13,
    )

    assert declared.build() == Budget(
        depth=3,
        mass=0.5,
        beam=2,
        unary_rounds=0,
        threads=8,
        beta_fuel=10,
        max_category_arity=4,
        max_category_size=13,
    )
    assert BudgetSpec().build() == Budget(
        depth=5,
        mass=1.0,
        beam=8,
        unary_rounds=1,
        threads=1,
        beta_fuel=1_000,
        max_category_arity=None,
        max_category_size=None,
    )


def test_a_budget_declares_no_log_zero_floor() -> None:
    with pytest.raises(dx.ValidationError, match="log_zero"):
        BudgetSpec.model_validate({"log_zero": -5.0})


@pytest.mark.parametrize(
    "field",
    [
        {"depth": 0},
        {"mass": 0.0},
        {"mass": 1.5},
        {"beam": 0},
        {"unary_rounds": -1},
        {"threads": 0},
        {"beta_fuel": 0},
        {"closure_rounds": -1},
        {"closure_nodes": -1},
        {"max_category_arity": 0},
        {"max_category_size": 0},
    ],
)
def test_a_budget_declaration_refuses_an_out_of_range_bound(
    field: dict[str, float],
) -> None:
    with pytest.raises(dx.ValidationError):
        BudgetSpec.model_validate(field)


@pytest.mark.parametrize(
    "matching",
    [IdentityMatchingSpec(), AlphaMatchingSpec(), FeatureErasureMatchingSpec()],
    ids=lambda matching: matching.kind,
)
def test_a_grammar_with_conjunctions_keeps_its_declared_matching(
    matching: MatchingSpec,
) -> None:
    declaration = _declaration()
    ccg = declaration.with_(signature=_ccg_with_default_rules(), matching=matching)

    assert ccg.matching == matching
    assert declaration.with_(matching=AlphaMatchingSpec()).matching == (
        AlphaMatchingSpec()
    )


def test_a_grammar_declaration_defaults_and_refuses_an_unknown_field() -> None:
    declaration = _declaration()

    assert declaration.goal == RootGoalSpec(categories=('("#const[Cat]" "S")',))
    assert declaration.matching == IdentityMatchingSpec()
    assert declaration.normal_form == NoNormalFormSpec()
    assert declaration.budget == BudgetSpec()

    payload = json.loads(declaration.model_dump_json())

    for superseded in (
        {"fanout": 1},
        {"layout": {"kind": "complete-tree", "depth": 5}},
        {"lexicalized": True},
        {"plugins": []},
    ):
        with pytest.raises(dx.ValidationError):
            GrammarDecl.model_validate(payload | superseded)

    with pytest.raises(dx.ValidationError):
        GrammarDecl.model_validate(payload | {"name": ""})


def test_a_grammar_declaration_survives_its_own_json() -> None:
    declaration = _declaration()

    assert GrammarDecl.model_validate_json(declaration.model_dump_json()) == (
        declaration
    )
    assert (
        GrammarDecl.model_validate_json(
            declaration.with_(
                signature=_ccg_with_default_rules(),
                matching=FeatureErasureMatchingSpec(),
            ).model_dump_json()
        )
        != declaration
    )
