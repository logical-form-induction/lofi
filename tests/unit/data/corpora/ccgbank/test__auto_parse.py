"""Pure CCGbank AUTO parser behavior."""

import subprocess
import sys

from lofi.data.corpora.ccgbank._auto_parse import (
    AutoDerivation,
    AutoLeaf,
    parse_auto,
    write_auto,
)


def test_a_leaf_round_trips_without_loading_layers() -> None:
    source = "ID=one\n(<L N NN NN dogs N>)\n"
    parsed = (
        AutoDerivation(
            (("ID", "one"),),
            AutoLeaf("N", "NN", "NN", "dogs", "N"),
        ),
    )

    assert parse_auto(source) == parsed
    assert write_auto(parsed) == source


def test_a_document_with_crlf_line_ends_parses_as_its_lf_spelling() -> None:
    source = "ID=one\n(<L N NN NN dogs N>)\nID=two\n(<L N NN NN cats N>)\n"
    crlf = source.replace("\n", "\r\n")

    assert parse_auto(crlf) == parse_auto(source)
    assert parse_auto(crlf.encode()) == parse_auto(source)


def test_importing_the_parser_does_not_import_layers() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; "
                "from lofi.data.corpora.ccgbank._auto_parse import parse_auto; "
                "parse_auto('ID=one\\n(<L N NN NN dogs N>)\\n'); "
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
