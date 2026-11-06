import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import Literal

import pytest
from tests.stand_ins import stand_in

from lofi.data._provers import SMT_PLUGIN, TPTP_PLUGIN, TRUTH_TABLE_PLUGIN
from lofi.lf import (
    Proved,
    ProverUnsupportedError,
    Refuted,
    TruthTableOracle,
)
from lofi.lf.provers import (
    SmtOracle,
    TptpOracle,
)
from lofi.spec import ConstructorSpec, InfixConstructorNotationSpec
from lofi.spec.language import (
    AtomicTypeSpec,
    BinderSpec,
    ConstantSpec,
    LanguageDecl,
    LanguageIso,
    LogicSpec,
    registered_language,
)
from lofi.spec.provers import SmtProverSpec, TptpProverSpec, TruthTableProverSpec


def test_prover_plugins_construct_their_declared_oracles() -> None:
    language = LanguageIso().forward(registered_language("neo-davidsonian"))
    smt = SMT_PLUGIN.construct(
        SmtProverSpec(executable=sys.executable),
        language,
    )
    tptp = TPTP_PLUGIN.construct(
        TptpProverSpec(executable=sys.executable),
        language,
    )
    truth_table = TRUTH_TABLE_PLUGIN.construct(
        TruthTableProverSpec(),
        LanguageIso().forward(registered_language("propositional")),
    )

    try:
        assert isinstance(smt, SmtOracle)
        assert isinstance(tptp, TptpOracle)
        assert isinstance(truth_table, TruthTableOracle)
    finally:
        smt.close()


def test_truth_tables_refuse_a_quantified_profile_at_construction() -> None:
    language = LanguageIso().forward(registered_language("neo-davidsonian"))

    with pytest.raises(ProverUnsupportedError) as refused:
        TRUTH_TABLE_PLUGIN.construct(TruthTableProverSpec(), language)

    assert refused.value.field == "quantifiers"


def test_truth_tables_refuse_a_profile_that_declares_equality() -> None:
    language = LanguageIso().forward(
        registered_language("propositional").with_(logic=LogicSpec(equality=True))
    )

    with pytest.raises(ProverUnsupportedError) as refused:
        TRUTH_TABLE_PLUGIN.construct(TruthTableProverSpec(), language)

    assert refused.value.field == "equality"


@pytest.mark.parametrize(("dialect", "decided"), [("tff", Proved), ("fof", Refuted)])
def test_the_declared_dialect_reaches_the_tptp_problem(
    tmp_path: Path, dialect: Literal["fof", "tff"], decided: type
) -> None:
    executable = stand_in(
        tmp_path / "prover",
        "import pathlib\n"
        "import sys\n"
        'problem = pathlib.Path(sys.argv[-1]).read_text(encoding="utf-8")\n'
        'typed = any(line.startswith("tff(") for line in problem.splitlines())\n'
        "status = 'Theorem' if typed else 'CounterSatisfiable'\n"
        "print(f'% SZS status {status}')\n",
    )
    language = LanguageIso().forward(
        LanguageDecl(
            name="dialects",
            atomic_types=(
                AtomicTypeSpec(name="e"),
                AtomicTypeSpec(name="t", kind="proposition"),
            ),
            constructors=(
                ConstructorSpec(
                    name="->",
                    kind=("Ty", "Ty", "Ty"),
                    notation=InfixConstructorNotationSpec(
                        spelling="->", precedence=1, associativity="right"
                    ),
                ),
            ),
            constants=(
                ConstantSpec(name="dog", type="e -> t"),
                ConstantSpec(name="rex", type="e"),
                ConstantSpec(name="both", type="t -> t -> t", classes=("conjunction",)),
            ),
            binders=BinderSpec(style="curry"),
        )
    )
    oracle = TPTP_PLUGIN.construct(
        TptpProverSpec(dialect=dialect, executable=str(executable)), language
    )
    formula = language.read("dog(rex)")

    assert isinstance(oracle.entails(formula, formula, timeout=5.0), decided)


def test_a_quantifier_free_registered_engine_refuses_quantifiers_at_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = f"quantifier_free_engine_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module}.py").write_text(
        "from lofi.lf.provers import QUANTIFIER_FREE_CAPABILITY, SmtEngine\n"
        'ENGINE = SmtEngine("quantifier-free", (), '
        "serves=QUANTIFIER_FREE_CAPABILITY)\n",
        encoding="utf-8",
        newline="\n",
    )
    metadata = tmp_path / f"{module}-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {module}\nVersion: 1.0\n",
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "entry_points.txt").write_text(
        f"[lofi.smt_engine]\nquantifier-free = {module}:ENGINE\n",
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    language = LanguageIso().forward(registered_language("neo-davidsonian"))

    with pytest.raises(ProverUnsupportedError) as refused:
        SMT_PLUGIN.construct(
            SmtProverSpec(engine="quantifier-free", executable=sys.executable),
            language,
        )

    assert refused.value.field == "quantifiers"


def test_a_test_package_tptp_engine_decides_a_fixture(
    tmp_path: Path,
) -> None:
    package = tmp_path / "fixture_engine.py"
    metadata = tmp_path / "fixture_engine-1.0.dist-info"
    executable = tmp_path / "fixture-prover"
    metadata.mkdir()
    package.write_text(
        "from lofi.lf.provers import TptpEngine\n"
        'ENGINE = TptpEngine("fixture", dialects=("fof",))\n',
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: fixture-engine\nVersion: 1.0\n",
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "entry_points.txt").write_text(
        "[lofi.tptp_engine]\nfixture = fixture_engine:ENGINE\n",
        encoding="utf-8",
        newline="\n",
    )
    executable = stand_in(executable, "print('% SZS status Theorem')\n")
    root = Path(__file__).parents[4]
    script = dedent(
        f"""
        from lofi.lf import Proved
        from lofi.data._provers import TPTP_PLUGIN
        from lofi.spec.language import ConstantSpec, LanguageIso, registered_language
        from lofi.spec.provers import TptpProverSpec

        source = registered_language("neo-davidsonian")
        language = LanguageIso().forward(source.with_(constants=(
            *source.constants,
            ConstantSpec(name="dog", type="e -> t"),
            ConstantSpec(name="rex", type="e"),
        )))
        oracle = TPTP_PLUGIN.construct(
            TptpProverSpec(engine="fixture", executable={str(executable)!r}),
            language,
        )
        formula = language.read("dog(rex)")
        assert isinstance(oracle.entails(formula, formula, timeout=5.0), Proved)
        """
    )
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(root)))
    }

    subprocess.run(
        (sys.executable, "-c", script),
        check=True,
        cwd=root,
        env=environment,
        capture_output=True,
        encoding="utf-8",
        timeout=30.0,
    )
