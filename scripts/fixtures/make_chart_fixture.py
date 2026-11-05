"""Generate the pinned chart fixture used by the installed pricing factory.

Run as a script::

    python scripts/fixtures/make_chart_fixture.py --grammar cfg-toy --seed 7

By default this writes ``chart_200.jsonl`` and ``chart_200.fingerprint`` under
``lofi/run/_fixtures``. The declaration, serializer, and integrity
checks live in :mod:`lofi.run._benchmark_fixture`, so generation, benchmarks,
and the installed factory all describe one workload.
"""

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from lofi.run._benchmark_fixture import (
    FIXTURE_DIRECTORY,
    FIXTURE_GRAMMARS,
    FingerprintMismatchError,
    FixtureGrammar,
    FixtureMismatchError,
    PinnedFingerprint,
    SentenceDigestMismatchError,
    ShapeMismatchError,
    check_pinned_fixture,
    fixture_grammar,
    read_sentences,
    sample_sentences,
    write_fixture,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "FIXTURE_DIRECTORY",
    "FIXTURE_GRAMMARS",
    "FingerprintMismatchError",
    "FixtureGrammar",
    "FixtureMismatchError",
    "PinnedFingerprint",
    "SentenceDigestMismatchError",
    "ShapeMismatchError",
    "check_pinned_fixture",
    "fixture_grammar",
    "main",
    "read_sentences",
    "sample_sentences",
    "write_fixture",
]


def _parser() -> argparse.ArgumentParser:
    """Return the fixture-generation command line."""
    parser = argparse.ArgumentParser(
        description="Draw a pinned chart fixture from a named grammar pack.",
    )
    parser.add_argument(
        "--grammar",
        required=True,
        help="the grammar pack to draw from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="the seed of the draw, recorded in the fingerprint file",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="how many sentences to draw",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=20,
        help="how many tokens each sentence has",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=FIXTURE_DIRECTORY,
        help="the directory the two files are written under",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Draw a fixture, write its files, and print their paths and fingerprint."""
    arguments = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    grammar = fixture_grammar(arguments.grammar)
    sentences = sample_sentences(
        grammar,
        arguments.count,
        arguments.length,
        arguments.seed,
    )
    sentences_path = arguments.out / f"chart_{arguments.count}.jsonl"
    fingerprint_path = arguments.out / f"chart_{arguments.count}.fingerprint"
    pinned = write_fixture(
        grammar,
        sentences,
        arguments.seed,
        sentences_path,
        fingerprint_path,
    )
    sys.stdout.write(
        f"{sentences_path}\n{fingerprint_path}\n{pinned.fingerprint}\n",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
