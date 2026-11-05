from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

from tests.language import language

from lofi.lf import (
    PostulatedRule,
    SequentCalculus,
    SequentLimits,
    TruthOperations,
    TruthTableRule,
    finite_frame_rule_sound,
)
from lofi.spec.language import (
    ConstantSpec,
    LanguageIso,
    LogicSpec,
    PrefixNotationSpec,
    registered_language,
)
from lofi.terms import Normalization

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.lf import ProverSyntax
    from lofi.spec.language import LanguageAssembly


class _NegationSemantics:
    @property
    def name(self) -> str:
        return "postulated-not"

    @property
    def arity(self) -> int:
        return 1

    @property
    def connective(self) -> bool:
        return True

    @property
    def quantifier(self) -> bool:
        return False

    @property
    def rules(self) -> tuple[str, ...]:
        return ("postulated-not",)

    @property
    def declares_symbol(self) -> bool:
        return True

    @property
    def model_resource(self) -> None:
        return None

    def denote(
        self,
        arguments: Sequence[object],
        algebra: TruthOperations,
        model: object,
        /,
    ) -> float:
        del model
        return algebra.lift(value=not bool(arguments[0]))

    def encode(
        self,
        symbol: str,
        arguments: Sequence[str],
        /,
        *,
        dialect: ProverSyntax,
    ) -> str:
        del dialect
        return f"{symbol}({arguments[0]})"


NORMALIZATION = Normalization(
    fuel=32,
    order="leftmost-outermost",
    eta="contract",
    associative_commutative=(),
)
LIMITS = SequentLimits(steps=64, depth=8, top_k=4, instantiations=8)


def _second_order() -> LanguageAssembly:
    source = registered_language("neo-davidsonian")
    return LanguageIso().forward(
        source.with_(
            name="second-order-sequent",
            constants=(
                *source.constants,
                ConstantSpec(name="dog", type="e -> t"),
                ConstantSpec(name="rex", type="e"),
                ConstantSpec(name="fido", type="e"),
            ),
            logic=LogicSpec(order="second"),
        )
    )


def test_truth_table_rules_are_sound_on_every_finite_boolean_frame() -> None:
    declared = language()

    for constant, role in declared.profile.logical_roles.items():
        if not role.truth_functional:
            continue

        for sign in (False, True):
            rule = TruthTableRule.from_role(constant, role, sign=sign)
            assignments = set(rule.assignments)
            expected = {
                values
                for values in product((False, True), repeat=role.arity)
                if role.truth(values) is sign
            }
            assert assignments == expected


def test_postulated_rules_must_match_every_finite_boolean_frame() -> None:
    semantics = _NegationSemantics()

    assert finite_frame_rule_sound(
        semantics,
        PostulatedRule("postulated-not", ((False,),), sign=True),
    )
    assert not finite_frame_rule_sound(
        semantics,
        PostulatedRule("postulated-not", ((True,),), sign=True),
    )


def test_a_defined_modal_operator_gains_delta_proof_steps_from_its_declaration() -> (
    None
):
    source = registered_language("neo-davidsonian")
    declared = LanguageIso().forward(
        source.with_(
            name="defined-modal",
            constants=(
                *source.constants,
                ConstantSpec(name="rain", type="t"),
                ConstantSpec(
                    name="box",
                    type="t -> t",
                    classes=("sentential",),
                    definition=r"\p:t. p & p",
                    notation=PrefixNotationSpec(ascii="box", unicode="□", precedence=6),
                ),
            ),
        )
    )
    calculus = SequentCalculus(declared, NORMALIZATION, order="first")

    result = calculus.search(
        declared.read("box rain"),
        declared.read("rain"),
        limits=LIMITS,
    )

    assert result.supported
    assert result.proofs
    assert any(
        step.rule == "delta:box" for proof in result.proofs for step in proof.steps
    )


def test_truth_table_branching_requires_every_counterexample_branch_to_close() -> None:
    source = registered_language("neo-davidsonian")
    declared = LanguageIso().forward(
        source.with_(
            name="open-conditional",
            constants=(
                *source.constants,
                ConstantSpec(name="rain", type="t"),
                ConstantSpec(name="wet", type="t"),
            ),
        )
    )
    calculus = SequentCalculus(declared, NORMALIZATION, order="first")

    result = calculus.search(
        declared.read("rain => wet"),
        declared.read("wet"),
        limits=LIMITS,
    )

    assert result.supported
    assert result.proofs == ()


def test_beta_normalization_is_an_executable_proof_step() -> None:
    declared = _second_order()
    calculus = SequentCalculus(declared, NORMALIZATION, order="second")

    result = calculus.search(
        declared.read(r"(\x. dog(x))(rex)"),
        declared.read("dog(rex)"),
        limits=LIMITS,
    )

    assert result.proofs
    assert result.proofs[0].steps[0].rule == "beta"


def test_second_order_predicate_instantiation_proves_the_declared_pattern() -> None:
    declared = _second_order()
    calculus = SequentCalculus(declared, NORMALIZATION, order="second")
    premise = declared.read("(forall P:(e -> t). P(rex) => P(fido)) & dog(rex)")
    goal = declared.read("dog(fido)")

    result = calculus.search(premise, goal, limits=LIMITS)

    assert result.supported
    assert result.proofs
    assert any(
        step.rule.startswith("instantiate:forall")
        for proof in result.proofs
        for step in proof.steps
    )


def test_first_order_search_refuses_a_predicate_binder_naming_the_order() -> None:
    declared = _second_order()
    calculus = SequentCalculus(declared, NORMALIZATION, order="first")
    premise = declared.read("forall P:(e -> t). P(rex) => P(fido)")

    result = calculus.search(
        premise,
        declared.read("dog(fido)"),
        limits=LIMITS,
    )

    assert not result.supported
    assert result.proofs == ()
    assert result.reason is not None
    assert "first" in result.reason
    assert "second" in result.reason
