"""Hilbert choice through declaration, proof translation and finite models."""

from typing import TYPE_CHECKING, cast

from tests.stand_ins import stand_in

from lofi.lf import Proved, ProverBound
from lofi.lf.models import (
    CRISP,
    EXTREME,
    PRODUCT,
    RESIDUUM,
    Evaluator,
    SampledStructureSource,
    TruthAlgebra,
    sampling,
)
from lofi.lf.provers import TptpEngine, TptpOracle, tptp_problem
from lofi.spec import PACKAGED_CONFIG_ROOT, RunSpecCodec
from lofi.spec.language import LanguageIso

if TYPE_CHECKING:
    from pathlib import Path

    from lofi.data.synthetic.spec.corpus import SyntheticCorpusDecl
    from lofi.spec.language import LanguageAssembly


RUN = PACKAGED_CONFIG_ROOT / "run" / "synthetic-parser.yaml"


def _language() -> LanguageAssembly:
    codec = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))
    run = codec(RUN, overrides=("data/datasets/train/dataset/language=choice",))
    dataset = cast("SyntheticCorpusDecl", run.data.datasets["train"].dataset)

    assert dataset.language.name == "choice"

    return LanguageIso().forward(dataset.language)


def _algebra() -> TruthAlgebra:
    return PRODUCT.build(
        implication=RESIDUUM,
        quantifier=EXTREME,
        equality=CRISP,
        margin=0.0,
    )


def test_choice_round_trips_and_emits_the_thf_choice_binder() -> None:
    language = _language()
    term = language.read("ε x. dog(x)")
    written = language.print(term)
    problem = tptp_problem(
        language,
        language.read("∃ x. dog(x)"),
        language.read("dog(ε x. dog(x))"),
        dialect="thf",
        polymorphism="monomorphize",
    )

    assert written == "ε x0. dog(x0)"
    assert language.read(written).key() == term.key()
    assert "@+ [" in problem
    assert "lofi_epsilon" not in problem


def test_the_thf_oracle_decides_the_choice_axiom(
    tmp_path: Path,
) -> None:
    executable = stand_in(
        tmp_path / "choice-prover",
        "import pathlib\n"
        "import sys\n"
        'if "@+ [" not in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"):\n'
        "    raise SystemExit(1)\n"
        "print('% SZS status Theorem')\n",
    )
    language = _language()
    engine = TptpEngine("choice-fixture", dialects=("thf",))
    oracle = TptpOracle(
        language,
        engine=engine,
        dialect="thf",
        executable=str(executable),
        prefix=None,
        problem_file="problem.p",
        bound=ProverBound("choice-fixture", language.profile.fragment),
        options=(),
        memory_mb=None,
        cpu_seconds=None,
        seed=None,
        polymorphism="monomorphize",
        certificates=frozenset(),
    )

    verdict = oracle.entails(
        language.read("∃ x. dog(x)"),
        language.read("dog(ε x. dog(x))"),
        timeout=5.0,
    )

    assert isinstance(verdict, Proved)


def test_every_sampled_model_satisfies_the_choice_axiom() -> None:
    language = _language()
    algebra = _algebra()
    source = SampledStructureSource(
        language,
        algebra,
        sampling=sampling("uniform", "independent", "table", "bernoulli"),
        structures=32,
        carriers={"e": 2},
        density=0.5,
        full_function_space_cap=64,
        henkin_pool=8,
        seed=7,
        resample="once",
        theory_floor=1.0,
    )
    premise = language.read("∃ x. dog(x)")
    hypothesis = language.read("dog(ε x. dog(x))")
    eligible = 0

    for structure in source.structures():
        evaluator = Evaluator(language, structure, algebra)

        if evaluator.truth(premise) == 1.0:
            eligible += 1
            assert evaluator.truth(hypothesis) == 1.0

    assert eligible > 0
