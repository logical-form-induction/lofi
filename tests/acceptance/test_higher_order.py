"""Higher-order logics are declared in documents and reach synthesis and training.

A document declares a logical-form language: its atomic types, its type
constructors, and each symbol with a type, a polymorphic scheme, or a request
that its type be inferred from use. The cases check that such a language
types the meanings the synthesizer draws, that higher-order quantification is
written and trained, and that the logic, its theory, its provers and its
groundings are declared rather than fixed.
"""

import os
from statistics import fmean
from typing import TYPE_CHECKING

import pytest
from tests.acceptance.driver import RUN_SEED, interpretation, leaf, train
from tests.stand_ins import stand_in

from lofi.lf.models import (
    CRISP,
    EXTREME,
    PRODUCT,
    RESIDUUM,
    Element,
    Evaluator,
    Frame,
    FunctionValue,
    Structure,
)
from lofi.lf.provers import tptp_problem
from lofi.run import read_metrics
from lofi.spec import derive_seed

if TYPE_CHECKING:
    from pathlib import Path


def fake_leo3(root: Path) -> Path:
    """Write a deterministic THF prover that reports every goal as proved."""
    calls = root / "leo3.calls"

    return stand_in(
        root / "leo3",
        "import pathlib\n"
        f"with pathlib.Path({str(calls)!r}).open('a', encoding='utf-8') as calls:\n"
        "    calls.write('call\\n')\n"
        "print('% SZS status Theorem')\n",
    )


def test_a_higher_order_signature_trains_a_tiny_point(tmp_path: Path) -> None:
    point = train("higher-order-point.yaml", tmp_path)

    assert (
        leaf(point.resolved, "data.datasets.train.dataset.language.name")
        == "higher-order"
    )
    assert point.rows


def test_declared_and_inferred_symbols_receive_their_schemes() -> None:
    language = interpretation("higher-order-point.yaml").language
    snore = language.scheme("snore")
    every = language.scheme("every")

    assert language.print_type(snore.body) == "e -> t"
    assert not snore.quantified
    assert language.print_type(every.body) == "(a -> t) -> (a -> t) -> t"
    assert every.quantified


def test_one_polymorphic_constant_is_used_at_two_types() -> None:
    language = interpretation("higher-order-point.yaml").language

    assert language.print_type(language.type_of(language.read("sleep && snore"))) == (
        "e -> t"
    )
    assert (
        language.print_type(
            language.type_of(language.read("sleep(john) && snore(john)"))
        )
        == "t"
    )


def test_a_logical_connective_left_to_inference_is_typed_from_use() -> None:
    language = interpretation("inferred-connective.yaml").language
    conjunction = language.scheme("and")

    assert language.print_type(conjunction.body) == "t -> t -> t"
    assert not conjunction.quantified


def test_a_meaning_that_quantifies_over_predicates_trains(tmp_path: Path) -> None:
    language = interpretation("predicate-quantifier.yaml").language
    quantified = language.read("someprop P:(e -> t). P(john)")

    assert language.print_type(language.type_of(quantified)) == "t"
    assert train("predicate-quantifier.yaml", tmp_path).rows


def test_a_language_declared_higher_order_synthesizes_and_trains(
    tmp_path: Path,
) -> None:
    language = interpretation("higher-order-logic.yaml").language

    assert language.profile.fragment.order == "higher"
    assert train("higher-order-logic.yaml", tmp_path).rows


def test_intensional_meanings_over_worlds_synthesize_and_train(
    tmp_path: Path,
) -> None:
    built = interpretation("intensional-point.yaml")
    language = built.language
    formula = built.read_text(r"(box \w. sleep_at(w)(john))(now)")

    assert language.profile.world_types == {"s"}
    assert built.decide(formula, formula).label == "entailment"
    assert train("intensional-point.yaml", tmp_path).rows


def test_a_language_declares_meaning_postulates(tmp_path: Path) -> None:
    built = interpretation("meaning-postulates.yaml")
    premise = built.read_text("sleep(john)")
    hypothesis = built.read_text("sleep(mary)")
    point = train("meaning-postulates.yaml", tmp_path)

    assert built.decide(premise, hypothesis).label == "entailment"
    assert leaf(point.resolved, "data.datasets.train.dataset.language.logic.postulates")


def test_a_document_declares_its_own_constant_class(tmp_path: Path) -> None:
    point = train("open-constant-class.yaml", tmp_path)

    constants = leaf(point.resolved, "data.datasets.train.dataset.language.constants")

    assert isinstance(constants, list)
    declared = constants[-1]
    assert isinstance(declared, dict)
    assert declared["classes"] == ["evidential"]


def test_a_higher_order_prover_rewards_training_under_a_declared_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_leo3(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    point = train("prover-in-training.yaml", tmp_path)
    recorded = read_metrics(point.directory)

    assert recorded is not None
    assert leaf(point.resolved, "model.grounding.scorer.budget.calls_per_step") == 8
    assert (
        leaf(point.resolved, "data.datasets.train.dataset.source.language.logic.order")
        == "higher"
    )
    assert (
        leaf(
            point.resolved,
            "data.datasets.train.dataset.source.language.logic.prover.dialect",
        )
        == "thf"
    )
    assert (
        leaf(
            point.resolved,
            "data.datasets.train.dataset.source.language.logic.prover.engine",
        )
        == "leo3"
    )
    assert leaf(point.resolved, "loss.terms.0.objective.signal") == "reward"
    grounding = recorded.contributed["grounding"]
    calls = grounding["calls"].value
    reward_mean = grounding["reward_mean"].value
    variance = grounding["variance"].value
    memo_hit_rate = grounding["memo_hit_rate"].value
    cpu_seconds = grounding["cpu_seconds"].value
    abstentions = grounding["abstentions"].value
    assert isinstance(calls, int)
    assert calls >= 0
    assert recorded.fitting is not None
    assert any(
        step.metrics.get("grounded/questions", 0.0) > 0.0
        and step.metrics.get("grounded/calls", 0.0)
        + step.metrics.get("grounded/memo_hits", 0.0)
        > 0.0
        for epoch in recorded.fitting.epochs
        for step in epoch.steps
    )
    assert (tmp_path / "leo3.calls").read_text(encoding="utf-8").splitlines()
    assert isinstance(reward_mean, float)
    assert 0.0 <= reward_mean <= 1.0
    assert isinstance(variance, float)
    assert variance >= 0.0
    assert isinstance(memo_hit_rate, float)
    assert 0.0 <= memo_hit_rate <= 1.0
    assert isinstance(cpu_seconds, float)
    assert cpu_seconds >= 0.0
    assert isinstance(abstentions, dict)
    assert all(isinstance(cause, str) for cause in abstentions)
    assert all(isinstance(count, int) and count >= 0 for count in abstentions.values())


def test_a_simulated_judgment_response_trains_through_the_higher_order_prover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_leo3(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    point = train("prover-judgment.yaml", tmp_path)
    recorded = read_metrics(point.directory)

    assert recorded is not None
    assert leaf(point.resolved, "data.datasets.train.dataset.labeling") == "graded"
    assert leaf(point.resolved, "loss.terms.0.objective.response.kind") == "judgment"
    assert leaf(point.resolved, "data.datasets.train.dataset.judgments") == [2] * 4
    rewards = tuple(
        value
        for epoch in (() if recorded.fitting is None else recorded.fitting.epochs)
        for step in epoch.steps
        if (value := step.metrics.get("loss/inference/reward_mean")) is not None
    )
    assert rewards
    assert all(0.0 <= value <= 1.0 for value in rewards)


def test_joint_inconsistency_trains_with_two_negation_constants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_leo3(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    point = train("joint-inconsistency.yaml", tmp_path)
    recorded = read_metrics(point.directory)
    constants = leaf(
        point.resolved, "data.datasets.train.dataset.source.language.constants"
    )

    assert recorded is not None
    assert isinstance(constants, list)
    negations = 0
    for constant in constants:
        assert isinstance(constant, dict)
        classes = constant.get("classes")
        assert isinstance(classes, list)
        negations += "negation" in classes
    assert negations == 2
    assert (
        leaf(point.resolved, "model.grounding.reward.contradiction")
        == "joint-inconsistency"
    )
    assert leaf(point.resolved, "model.grounding.reward.refuted") == -0.5
    reward_mean = recorded.contributed["grounding"]["reward_mean"].value
    assert isinstance(reward_mean, float)
    assert reward_mean == pytest.approx(0.5)


def test_declared_form_supervision_recovers_certified_gold_forms(
    tmp_path: Path,
) -> None:
    point = train("form-supervision.yaml", tmp_path)
    recorded = read_metrics(point.directory)

    assert recorded is not None
    assert recorded.fitting is not None
    assert leaf(point.resolved, "loss.terms.0.objective.kind") == "form-supervision"
    epochs = tuple(
        tuple(
            value
            for step in epoch.steps
            if (
                value := step.metrics.get("loss/form-supervision/gold_form_probability")
            )
            is not None
        )
        for epoch in recorded.fitting.epochs
    )
    assert len(epochs) > 1
    assert all(epochs)
    assert all(0.0 < value <= 1.0 for epoch in epochs for value in epoch)
    assert fmean(epochs[-1]) > fmean(epochs[0])


def test_shaped_proof_trains_sampled_forms_against_certified_gold_forms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_leo3(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    point = train("shaped-proof.yaml", tmp_path)
    recorded = read_metrics(point.directory)

    assert recorded is not None
    assert recorded.fitting is not None
    assert leaf(point.resolved, "loss.terms.0.objective.kind") == "shaped-proof"
    assert leaf(point.resolved, "loss.terms.0.objective.readout") == "forms"
    assert leaf(point.resolved, "model.readouts.forms.decoding.strategy") == "sample"
    assert leaf(point.resolved, "loss.terms.0.objective.budget.calls_per_step") == 8
    assert (
        leaf(
            point.resolved,
            "data.datasets.train.dataset.language.logic.prover.engine",
        )
        == "leo3"
    )
    losses = tuple(
        value
        for epoch in recorded.fitting.epochs
        for step in epoch.steps
        if (value := step.metrics.get("loss/shaped-proof")) is not None
    )
    abstentions = tuple(
        value
        for epoch in recorded.fitting.epochs
        for step in epoch.steps
        if (value := step.metrics.get("loss/shaped-proof/abstention_rate")) is not None
    )
    assert losses
    assert all(-1.0 <= value < 0.0 for value in losses)
    assert abstentions
    assert all(value == 0.0 for value in abstentions)
    assert (tmp_path / "leo3.calls").read_text(encoding="utf-8").splitlines()


def test_a_declared_higher_order_prover_labels_a_pair() -> None:
    built = interpretation("higher-order-oracle.yaml")
    premise = built.read_text("every x.(sleep(x), sleep(x))")
    decision = built.decide(premise, premise)

    assert decision.label == "entailment"
    assert decision.prover == "tptp:vampire:thf"


def test_a_renamed_truth_type_and_xor_emit_decide_and_model_check() -> None:
    built = interpretation("xor-renamed.yaml")
    language = built.language
    formula = built.read_text("sleep(john) ^^ sleep(mary)")
    entity = (Element("entity", 0), Element("entity", 1))
    frame = Frame(
        language.store,
        function_type=language.function_type,
        truth_type="truth",
        carriers={"entity": 2},
        full_function_space_cap=16,
        henkin_pool=2,
        seed=3,
    )
    vocabulary = language.store.signature.vocabulary(language.term_sort)
    structure = Structure(
        "renamed-xor",
        frame,
        {
            vocabulary.id("john"): entity[0],
            vocabulary.id("mary"): entity[1],
            vocabulary.id("sleep"): FunctionValue(entity, (True, False), (True, True)),
        },
    )
    algebra = PRODUCT.build(
        implication=RESIDUUM,
        quantifier=EXTREME,
        equality=CRISP,
        margin=0.0,
    )

    assert built.decide(formula, formula).label == "entailment"
    assert "<~>" in tptp_problem(language, formula, formula, dialect="thf")
    assert Evaluator(language, structure, algebra).truth(formula) == 1.0


def test_definitions_and_generalized_quantifiers_emit_and_model_check() -> None:
    built = interpretation("defined-gq.yaml")
    language = built.language
    conjunction = built.read_text("sleep(john) & sleep(mary)")
    quantified = built.read_text("most x.(sleep(x), sleep(x))")
    entity = tuple(Element("e", index) for index in range(3))
    frame = Frame(
        language.store,
        function_type=language.function_type,
        truth_type="truth",
        carriers={"e": 3},
        full_function_space_cap=64,
        henkin_pool=2,
        seed=5,
    )
    vocabulary = language.store.signature.vocabulary(language.term_sort)
    structure = Structure(
        "defined-generalized-quantifier",
        frame,
        {
            vocabulary.id("john"): entity[0],
            vocabulary.id("mary"): entity[1],
            vocabulary.id("sleep"): FunctionValue(
                entity,
                (True, False, True),
                (True, True, True),
            ),
        },
    )
    algebra = PRODUCT.build(
        implication=RESIDUUM,
        quantifier=EXTREME,
        equality=CRISP,
        margin=0.0,
    )
    evaluator = Evaluator(language, structure, algebra)
    problem = tptp_problem(language, quantified, quantified, dialect="thf")

    assert evaluator.truth(conjunction) == 0.0
    assert evaluator.truth(quantified) == 1.0
    assert "lofi_most" in problem
    assert "lofi_and" not in problem
    assert built.decide(quantified, quantified).label == "entailment"


def test_model_checking_over_sampled_structures_grounds_inference(
    tmp_path: Path,
) -> None:
    point = train("model-checking.yaml", tmp_path)
    recorded = read_metrics(point.directory)

    assert recorded is not None
    assert leaf(point.resolved, "model.grounding.scorer.kind") == "model-checking"
    assert leaf(point.resolved, "model.grounding.scorer.source.seed") == derive_seed(
        RUN_SEED, "model-structures"
    )
    grounding = recorded.contributed["grounding"]
    calls = grounding["calls"].value
    malformed = grounding["malformed_mass"].value
    assert isinstance(calls, int)
    assert calls > 0
    assert isinstance(malformed, float)
    assert malformed == pytest.approx(0.0)


def test_model_checking_over_a_given_knowledge_base_grounds_inference(
    tmp_path: Path,
) -> None:
    point = train("model-checking-given.yaml", tmp_path)
    recorded = read_metrics(point.directory)

    assert recorded is not None
    assert leaf(point.resolved, "model.grounding.scorer.source.kind") == "given"
    calls = recorded.contributed["grounding"]["calls"].value
    assert isinstance(calls, int)
    assert calls > 0


def test_model_checking_over_learned_structures_fits_declared_theory(
    tmp_path: Path,
) -> None:
    point = train("model-checking-learned.yaml", tmp_path)
    recorded = read_metrics(point.directory)

    assert recorded is not None
    assert leaf(point.resolved, "model.grounding.scorer.source.kind") == "learned"
    calls = recorded.contributed["grounding"]["calls"].value
    assert isinstance(calls, int)
    assert calls > 0


@pytest.mark.parametrize(
    ("tnorm", "entities"),
    [("product", 2), ("goedel", 3), ("lukasiewicz", 4)],
)
def test_model_checking_trains_across_declared_logic_and_frame_points(
    tmp_path: Path,
    tnorm: str,
    entities: int,
) -> None:
    point = train(
        "model-checking.yaml",
        tmp_path,
        f"model.grounding.scorer.logic.tnorm={tnorm}",
        (f'model.grounding.scorer.source.carriers={{"e":{entities},"v":1}}'),
    )
    recorded = read_metrics(point.directory)

    assert recorded is not None
    calls = recorded.contributed["grounding"]["calls"].value
    assert isinstance(calls, int)
    assert calls > 0
