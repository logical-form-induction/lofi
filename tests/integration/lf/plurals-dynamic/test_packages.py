"""Plural and dynamic declarations through THF and finite evaluation."""

from typing import TYPE_CHECKING, cast

from tests.stand_ins import stand_in

from lofi.lf import Proved, ProverBound, Refuted
from lofi.lf.models import (
    CRISP,
    EXTREME,
    PRODUCT,
    RESIDUUM,
    Element,
    Evaluator,
    Frame,
    FunctionValue,
    SampledStructureSource,
    Structure,
    TruthAlgebra,
    sampling,
)
from lofi.lf.provers import VAMPIRE, TptpEngine, TptpOracle, tptp_problem
from lofi.spec import PACKAGED_CONFIG_ROOT, RunSpecCodec
from lofi.spec.language import LanguageIso

if TYPE_CHECKING:
    from pathlib import Path

    from lofi.data.synthetic.spec.corpus import SyntheticCorpusDecl
    from lofi.spec.language import LanguageAssembly


RUN = PACKAGED_CONFIG_ROOT / "run" / "synthetic-parser.yaml"


def _language(name: str, /) -> LanguageAssembly:
    codec = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))
    run = codec(RUN, overrides=(f"data/datasets/train/dataset/language={name}",))
    dataset = cast("SyntheticCorpusDecl", run.data.datasets["train"].dataset)

    assert dataset.language.name == name

    return LanguageIso().forward(dataset.language)


def _algebra() -> TruthAlgebra:
    return PRODUCT.build(
        implication=RESIDUUM,
        quantifier=EXTREME,
        equality=CRISP,
        margin=0.0,
    )


def _oracle(language: LanguageAssembly) -> TptpOracle:
    return TptpOracle(
        language,
        engine=VAMPIRE,
        dialect="thf",
        executable=None,
        prefix=None,
        problem_file="problem.p",
        bound=ProverBound("vampire", language.profile.fragment),
        options=(),
        memory_mb=None,
        cpu_seconds=None,
        seed=None,
        polymorphism="monomorphize",
        certificates=frozenset(),
    )


def _counter_oracle(language: LanguageAssembly, directory: Path, /) -> TptpOracle:
    executable = stand_in(
        directory / "finite-countermodel",
        "print('% SZS status CounterSatisfiable')\n",
    )
    engine = TptpEngine("finite-countermodel", dialects=("thf",))
    return TptpOracle(
        language,
        engine=engine,
        dialect="thf",
        executable=str(executable),
        prefix=None,
        problem_file="problem.p",
        bound=ProverBound("finite-countermodel", language.profile.fragment),
        options=(),
        memory_mb=None,
        cpu_seconds=None,
        seed=None,
        polymorphism="monomorphize",
        certificates=frozenset(),
    )


def _predicate(domain: tuple[Element, ...], outputs: tuple[bool, ...]) -> FunctionValue:
    return FunctionValue(domain, outputs, (True,) * len(domain))


def test_plural_sums_emit_axioms_and_separate_distributive_from_collective(
    tmp_path: Path,
) -> None:
    language = _language("plurals")
    entity = tuple(Element("e", at) for at in range(3))
    frame = Frame(
        language.store,
        function_type=language.function_type,
        truth_type="t",
        carriers={"e": 3},
        full_function_space_cap=64,
        henkin_pool=8,
        seed=2,
    )
    vocabulary = language.store.signature.vocabulary(language.term_sort)
    relation_rows = (
        _predicate(entity, (False, True, False)),
        _predicate(entity, (True, False, False)),
        _predicate(entity, (False, False, False)),
    )
    structure = Structure(
        "plural-intended",
        frame,
        {
            vocabulary.id("john"): entity[0],
            vocabulary.id("mary"): entity[1],
            vocabulary.id("met"): FunctionValue(
                entity, relation_rows, (True,) * len(entity)
            ),
            vocabulary.id("laughed"): _predicate(entity, (True, True, True)),
            vocabulary.id("gathered"): _predicate(entity, (False, False, True)),
        },
        semilattices=("e",),
    )
    evaluator = Evaluator(language, structure, _algebra())
    pair_premise = language.read("met(john)(mary) & met(mary)(john)")
    pair_goal = language.read("exists x. met(john)(x)")
    distributive = language.read("laughed(sum(john)(mary))")
    distributed = language.read("laughed(john)")
    collective = language.read("gathered(sum(john)(mary))")
    singular_collective = language.read("gathered(john)")
    problem = tptp_problem(
        language,
        distributive,
        distributed,
        dialect="thf",
        polymorphism="monomorphize",
    )

    assert evaluator.theory_holds(floor=1.0)
    assert evaluator.truth(pair_premise) == evaluator.truth(pair_goal) == 1.0
    assert evaluator.truth(distributive) == evaluator.truth(distributed) == 1.0
    assert evaluator.truth(collective) == 1.0
    assert evaluator.truth(singular_collective) == 0.0
    assert "lofi_sum" in problem
    assert len(language.profile.theory) >= 5
    sampled = SampledStructureSource(
        language,
        _algebra(),
        sampling=sampling("uniform", "independent", "table", "bernoulli"),
        structures=1,
        carriers={"e": 3},
        density=1.0,
        full_function_space_cap=64,
        henkin_pool=8,
        seed=4,
        resample="once",
        theory_floor=1.0,
    ).structures()[0]
    assert sampled.join(entity[0], entity[1]) == entity[2]
    assert isinstance(
        _oracle(language).entails(distributive, distributed, timeout=10.0), Proved
    )
    assert isinstance(
        _counter_oracle(language, tmp_path).entails(
            collective, singular_collective, timeout=10.0
        ),
        Refuted,
    )


def _dynamic_structure(
    language: LanguageAssembly,
    /,
    *,
    sat: tuple[bool, bool],
) -> Structure:
    entity = tuple(Element("e", at) for at in range(2))
    states = tuple(Element("s", at) for at in range(2))
    frame = Frame(
        language.store,
        function_type=language.function_type,
        truth_type="t",
        carriers={"e": 2, "s": 2},
        full_function_space_cap=512,
        henkin_pool=8,
        seed=3,
    )
    known_states = (True,) * len(states)
    assign = FunctionValue(
        entity,
        (
            FunctionValue(states, (states[0], states[0]), known_states),
            FunctionValue(states, (states[1], states[1]), known_states),
        ),
        (True,) * len(entity),
    )
    vocabulary = language.store.signature.vocabulary(language.term_sort)
    return Structure(
        "dynamic-intended",
        frame,
        {
            vocabulary.id("assign"): assign,
            vocabulary.id("referent"): FunctionValue(
                states, entity, (True,) * len(states)
            ),
            vocabulary.id("man"): _predicate(entity, (True, False)),
            vocabulary.id("woman"): _predicate(entity, (False, True)),
            vocabulary.id("walked"): _predicate(entity, (True, True)),
            vocabulary.id("sat"): _predicate(entity, sat),
        },
    )


def test_dynamic_state_passing_resolves_and_distinguishes_antecedents(
    tmp_path: Path,
) -> None:
    language = _language("dynamic")
    correct = language.read("close(d_and(introduce(man)(walked))(pronoun(sat)))")
    dynamic_existential = language.read(
        r"close(d_exists(\x:e. \i:s. \o:s. man(x) & o = assign(x)(i)))"
    )
    man_sat = language.read("exists x. man(x) & sat(x)")
    swapped = language.read("close(d_and(introduce(woman)(walked))(pronoun(sat)))")
    correct_model = Evaluator(
        language, _dynamic_structure(language, sat=(True, False)), _algebra()
    )
    swapped_model = Evaluator(
        language, _dynamic_structure(language, sat=(False, True)), _algebra()
    )

    assert correct_model.theory_holds(floor=1.0)
    assert correct_model.truth(correct) == correct_model.truth(man_sat) == 1.0
    assert correct_model.truth(dynamic_existential) == 1.0
    assert swapped_model.theory_holds(floor=1.0)
    assert swapped_model.truth(swapped) == 1.0
    assert swapped_model.truth(man_sat) == 0.0
    assert isinstance(_oracle(language).entails(correct, man_sat, timeout=10.0), Proved)
    assert isinstance(
        _counter_oracle(language, tmp_path).entails(swapped, man_sat, timeout=10.0),
        Refuted,
    )
