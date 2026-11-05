import subprocess
import sys
from pathlib import Path

import lofi.eval
import lofi.eval._crosslingual_alignment as crosslingual_alignment
import lofi.eval._dependencies as dependencies
import lofi.eval._transfer as transfer
from lofi.eval import corpora


def test_the_optional_surface_moves_every_corpus_dependent_export() -> None:
    modules = (crosslingual_alignment, dependencies, transfer)
    names = {name for module in modules for name in module.__all__}

    assert set(corpora.__all__) == names
    assert names.isdisjoint(lofi.eval.__all__)

    for module in modules:
        for name in module.__all__:
            assert getattr(corpora, name) is getattr(module, name)
            assert not hasattr(lofi.eval, name)


def test_importing_the_corpus_surface_does_not_load_adapter_dependencies() -> None:
    script = """
import sys

for name in ("lairs", "pyarrow", "pyarrow.parquet"):
    sys.modules[name] = None

import lofi.eval
import lofi.eval.corpora
"""
    finished = subprocess.run(
        [sys.executable, "-"],
        input=script,
        capture_output=True,
        encoding="utf-8",
        check=False,
        cwd=Path(__file__).parents[3],
    )

    assert finished.returncode == 0, finished.stderr


def test_the_optional_surface_retains_only_crosslingual_language_values() -> None:
    language = corpora.TransferLanguage("de", "gold")
    missing = corpora.UnavailableLanguage(
        corpora.TransferLanguage("zh", "silver"), "no converted treebank"
    )

    assert language.to_spec() == {"language": "de", "source": "gold"}
    assert missing.language.code == "zh"
