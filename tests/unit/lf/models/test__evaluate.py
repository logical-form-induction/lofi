import pytest
from tests.language import language

from lofi.lf.models import (
    CRISP,
    EXTREME,
    PRODUCT,
    RESIDUUM,
    Element,
    EvaluationError,
    Evaluator,
    Frame,
    FunctionValue,
    Structure,
    TruthAlgebra,
    truth_type,
)
from lofi.spec.language import (
    BinderSpec,
    ConstantSpec,
    LanguageIso,
    registered_language,
)


def algebra() -> TruthAlgebra:
    return PRODUCT.build(
        implication=RESIDUUM,
        quantifier=EXTREME,
        equality=CRISP,
        margin=0.0,
    )


def test_evaluation_uses_roles_and_vocabulary_keyed_denotations() -> None:
    declared = language(("p", "q"))
    frame = Frame(
        declared.store,
        function_type=declared.function_type,
        truth_type="t",
        carriers={"e": 2, "v": 1},
        full_function_space_cap=8,
        henkin_pool=2,
        seed=5,
    )
    vocabulary = declared.store.signature.vocabulary(declared.term_sort)
    structure = Structure(
        "propositional",
        frame,
        {vocabulary.id("p"): True, vocabulary.id("q"): False},
    )
    evaluator = Evaluator(declared, structure, algebra())

    assert evaluator.truth(declared.read("p & q")) == 0.0
    assert evaluator.truth(declared.read("p => q")) == 0.0
    assert evaluator.truth(declared.read("p | q")) == 1.0


def test_evaluation_unfolds_a_closed_logical_definition() -> None:
    source = registered_language("neo-davidsonian")
    constants = tuple(
        constant.with_(
            classes=("sentential",),
            definition=r"\p:t. \q:t. ~(~p | ~q)",
        )
        if constant.name == "and"
        else constant
        for constant in source.constants
    )
    declared = LanguageIso().forward(
        source.with_(
            constants=(
                *constants,
                ConstantSpec(name="p", type="t"),
                ConstantSpec(name="q", type="t"),
            )
        )
    )
    frame = Frame(
        declared.store,
        function_type=declared.function_type,
        truth_type="t",
        carriers={"e": 1, "v": 1},
        full_function_space_cap=8,
        henkin_pool=2,
        seed=5,
    )
    vocabulary = declared.store.signature.vocabulary(declared.term_sort)
    structure = Structure(
        "defined",
        frame,
        {vocabulary.id("p"): True, vocabulary.id("q"): False},
    )

    assert (
        Evaluator(declared, structure, algebra()).truth(declared.read("p & q")) == 0.0
    )


def test_a_polymorphic_logical_result_does_not_redefine_the_truth_type() -> None:
    source = registered_language("neo-davidsonian")
    constants = tuple(
        constant.with_(type="a -> a -> a", quantifies=("a",))
        if constant.name == "and"
        else constant
        for constant in source.constants
    )
    declared = LanguageIso().forward(source.with_(constants=constants))

    assert declared.print_type(truth_type(declared)) == "t"


def test_evaluation_reads_binding_positions_and_the_bound_variable_sort() -> None:
    declared = language(("dog",), types={"dog": "e -> t"})
    frame = Frame(
        declared.store,
        function_type=declared.function_type,
        truth_type="t",
        carriers={"e": 2, "v": 1},
        full_function_space_cap=8,
        henkin_pool=2,
        seed=5,
    )
    vocabulary = declared.store.signature.vocabulary(declared.term_sort)
    denotation = FunctionValue(
        (Element("e", 0), Element("e", 1)),
        (True, False),
        (True, True),
    )
    structure = Structure("quantified", frame, {vocabulary.id("dog"): denotation})
    evaluator = Evaluator(declared, structure, algebra())

    assert evaluator.truth(declared.read("exists x. dog(x)")) == 1.0
    assert evaluator.truth(declared.read("forall x. dog(x)")) == 0.0


def test_evaluation_refuses_a_quantifier_over_an_unknown_fact() -> None:
    declared = language(("dog",), types={"dog": "e -> t"})
    frame = Frame(
        declared.store,
        function_type=declared.function_type,
        truth_type="t",
        carriers={"e": 2, "v": 1},
        full_function_space_cap=8,
        henkin_pool=2,
        seed=5,
    )
    vocabulary = declared.store.signature.vocabulary(declared.term_sort)
    denotation = FunctionValue(
        (Element("e", 0), Element("e", 1)),
        (True, False),
        (True, False),
    )
    evaluator = Evaluator(
        declared,
        Structure("open", frame, {vocabulary.id("dog"): denotation}),
        algebra(),
    )

    with pytest.raises(EvaluationError, match="unknown"):
        evaluator.truth(declared.read("forall x. dog(x)"))


def test_evaluation_reads_a_church_binder_annotation_from_its_binding_site() -> None:
    source = registered_language("neo-davidsonian").with_(
        binders=BinderSpec(style="church")
    )
    declared = LanguageIso().forward(
        source.with_(
            constants=(
                *source.constants,
                ConstantSpec(name="dog", type="e -> t"),
            )
        )
    )
    frame = Frame(
        declared.store,
        function_type=declared.function_type,
        truth_type="t",
        carriers={"e": 2, "v": 1},
        full_function_space_cap=8,
        henkin_pool=2,
        seed=5,
    )
    vocabulary = declared.store.signature.vocabulary(declared.term_sort)
    denotation = FunctionValue(
        (Element("e", 0), Element("e", 1)),
        (True, True),
        (True, True),
    )
    structure = Structure("church", frame, {vocabulary.id("dog"): denotation})

    assert (
        Evaluator(declared, structure, algebra()).truth(
            declared.read("forall x:e. dog(x)")
        )
        == 1.0
    )


def test_evaluation_refuses_an_open_variable() -> None:
    declared = language()
    frame = Frame(
        declared.store,
        function_type=declared.function_type,
        truth_type="t",
        carriers={"e": 1, "v": 1},
        full_function_space_cap=4,
        henkin_pool=2,
        seed=5,
    )
    evaluator = Evaluator(declared, Structure("empty", frame, {}), algebra())

    with pytest.raises(EvaluationError, match="open variable"):
        evaluator.evaluate(declared.store.fvar(declared.term_sort, "x"))


def test_evaluation_classifies_a_polymorphic_abstraction_as_out_of_fragment() -> None:
    declared = LanguageIso().forward(
        registered_language("neo-davidsonian").with_(binders=BinderSpec(style="church"))
    )
    frame = Frame(
        declared.store,
        function_type=declared.function_type,
        truth_type="t",
        carriers={"e": 1, "v": 1},
        full_function_space_cap=4,
        henkin_pool=2,
        seed=5,
    )
    abstraction = declared.store.op(
        declared.store.signature.by_name(declared.calculus.lam),
        declared.store.fvar(declared.type_sort, "a"),
        declared.store.bvar(declared.term_sort, 0),
    )
    evaluator = Evaluator(declared, Structure("empty", frame, {}), algebra())

    with pytest.raises(EvaluationError, match="atomic or functional") as error:
        evaluator.evaluate(abstraction)

    assert error.value.cause == "out-of-fragment"
