"""Pure CCGbank PARG parser behavior."""

import subprocess
import sys

from lofi.data.corpora.ccgbank._parg_parse import (
    PargDependency,
    PargSentence,
    parse_parg,
    write_parg,
)


def test_a_dependency_round_trips_without_loading_layers() -> None:
    source = '<s id="one"> 1\n0\t1\tN\t1\tdogs\tbark\n<\\s>\n'
    parsed = (
        PargSentence(
            "one",
            1,
            (PargDependency(0, 1, "N", 1, "dogs", "bark"),),
        ),
    )

    assert parse_parg(source) == parsed
    assert write_parg(parsed) == source


def test_importing_the_parser_does_not_import_layers() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; "
                "from lofi.data.corpora.ccgbank._parg_parse import parse_parg; "
                "parse_parg('<s id=\\\"one\\\"> 0\\n<\\\\s>\\n'); "
                "print(any(name == 'lairs' or name.startswith('lairs.') "
                "for name in sys.modules))"
            ),
        ],
        capture_output=True,
        check=False,
        encoding="utf-8",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
