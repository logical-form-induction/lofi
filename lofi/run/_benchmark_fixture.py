"""The checked grammar and sentences the parser training-step benchmarks time.

The fixture is packaged data, read through the package's resources. Its record
pins both the compiled grammar and the sentence bytes, so packaging cannot
silently change the timed workload.
"""

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

from lofi.data.synthetic import SyntheticGenerator, UniformLength
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.spec import ConstantPatternSpec, PatternRuleSpec, RuleSetSpec
from lofi.spec.grammar import (
    BudgetSpec,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    RootGoalSpec,
    SignatureSpec,
)
from lofi.spec.language import registered_language

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi.formalism import Grammar
    from lofi.lexicon import StaticLexicon
    from lofi.spec.grammar import GrammarAssembly
    from lofi.terms import Term

__all__ = [
    "FIXTURE_DIRECTORY",
    "FIXTURE_GRAMMARS",
    "FingerprintMismatchError",
    "FixtureGrammar",
    "FixtureMismatchError",
    "PinnedChart",
    "PinnedFingerprint",
    "SentenceDigestMismatchError",
    "ShapeMismatchError",
    "check_pinned_fixture",
    "fixture_grammar",
    "load_pinned_chart",
    "packaged_pinned_chart",
    "read_sentences",
    "sample_sentences",
    "write_fixture",
]

FIXTURE_DIRECTORY: Final = Path(__file__).with_name("_fixtures")
"""Directory holding the immutable pricing fixture in source and wheels."""


class FixtureGrammar:
    """A grammar pack sentences are drawn from.

    Parameters
    ----------
    declaration : GrammarDecl
        The declaration, whose name selects the pack.
    """

    __slots__ = ("_declaration",)

    def __init__(self, declaration: GrammarDecl) -> None:
        self._declaration = declaration

    @property
    def name(self) -> str:
        """Return the pack name."""
        return self._declaration.name

    @property
    def declaration(self) -> GrammarDecl:
        """Return the declaration the pack lowers."""
        return self._declaration

    def assemble(self) -> GrammarAssembly:
        """Lower the declaration into one grammar and lexicon assembly."""
        return GrammarIso().forward(
            self._declaration, registered_language("propositional")
        )


def _atom(name: str) -> ConstantPatternSpec:
    """Return the pattern matching one declared category constant."""
    return ConstantPatternSpec(name=name, sort="Cat")


def _cfg_toy() -> FixtureGrammar:
    """Return the English-like binary grammar the pricing fixture pins."""
    productions = (
        ("s-np-vp", "S", "NP", "VP", 1),
        ("s-pp-s", "S", "PP", "S", 1),
        ("s-s-sc", "S", "S", "SC", 0),
        ("sc-c-s", "SC", "C", "S", 1),
        ("np-det-n", "NP", "Det", "N", 1),
        ("np-det-ap", "NP", "Det", "AP", 1),
        ("np-np-pp", "NP", "NP", "PP", 0),
        ("np-np-nc", "NP", "NP", "NC", 0),
        ("nc-c-np", "NC", "C", "NP", 1),
        ("ap-adj-n", "AP", "Adj", "N", 1),
        ("ap-adj-ap", "AP", "Adj", "AP", 1),
        ("vp-v-np", "VP", "V", "NP", 0),
        ("vp-v-pp", "VP", "V", "PP", 0),
        ("vp-v-s", "VP", "V", "S", 0),
        ("vp-vp-pp", "VP", "VP", "PP", 0),
        ("vp-adv-vp", "VP", "Adv", "VP", 1),
        ("vp-vp-adv", "VP", "VP", "Adv", 0),
        ("vp-vp-vc", "VP", "VP", "VC", 0),
        ("vc-c-vp", "VC", "C", "VP", 1),
        ("pp-p-np", "PP", "P", "NP", 0),
    )
    nonterminals = (
        "S",
        "SC",
        "NP",
        "NC",
        "AP",
        "VP",
        "VC",
        "PP",
        "Det",
        "N",
        "V",
        "P",
        "Adj",
        "Adv",
        "C",
    )
    lexicon = {
        "Det": ("the", "a", "every", "some"),
        "N": ("cat", "dog", "bird", "fish", "girl", "boy"),
        "V": ("sees", "likes", "chases", "thinks", "says"),
        "P": ("with", "near", "under", "behind"),
        "Adj": ("big", "small", "red", "old"),
        "Adv": ("quickly", "quietly", "often"),
        "C": ("and", "or"),
    }

    return FixtureGrammar(
        GrammarDecl(
            name="cfg-toy",
            signature=SignatureSpec(
                name="cfg-toy", primitives=nonterminals, syntax="s-expression"
            ),
            rules=RuleSetSpec(
                name="cfg-toy",
                rules=tuple(
                    PatternRuleSpec(
                        name=name,
                        premises=(_atom(left), _atom(right)),
                        conclusion=_atom(parent),
                        head_policy=("left-conjunct" if head == 0 else "non-absorbing"),
                    )
                    for name, parent, left, right, head in productions
                ),
            ),
            lexicon=InlineLexiconSpec(
                entries=tuple(
                    LexicalEntrySpec(
                        form=form,
                        category=f'("#const[Cat]" "{preterminal}")',
                    )
                    for preterminal, forms in lexicon.items()
                    for form in forms
                )
            ),
            goal=RootGoalSpec(categories=('("#const[Cat]" "S")',)),
            budget=BudgetSpec(unary_rounds=1, closure_rounds=4, closure_nodes=64),
        )
    )


FIXTURE_GRAMMARS: Final = (_cfg_toy(),)
"""The grammar packs a fixture can be drawn from."""


def fixture_grammar(name: str) -> FixtureGrammar:
    """Return the grammar pack with the requested name.

    Raises
    ------
    ValueError
        If no pack has that name.
    """
    for grammar in FIXTURE_GRAMMARS:
        if grammar.name == name:
            return grammar

    known = ", ".join(grammar.name for grammar in FIXTURE_GRAMMARS)
    msg = f"unknown grammar pack {name!r}; the known packs are {known}"
    raise ValueError(msg)


def sample_sentences(
    grammar: FixtureGrammar,
    count: int,
    length: int,
    seed: int,
) -> tuple[tuple[str, ...], ...]:
    """Draw reproducible sentences of one length from a grammar pack.

    Raises
    ------
    ValueError
        If ``count`` or ``length`` is not positive.
    UnproductiveGrammarError
        If the goal derives no sentence of that length.
    """
    if count <= 0:
        msg = f"a fixture holds a positive number of sentences, not {count}"
        raise ValueError(msg)

    if length <= 0:
        msg = f"a sentence has a positive number of tokens, not {length}"
        raise ValueError(msg)

    corpus = SyntheticGenerator(
        grammar.assemble(),
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    ).sample(count, seed=seed, length=UniformLength(length, length))

    return tuple(
        tuple(token.surface for token in example.item.tokens)
        for example in corpus.examples
    )


def _encode_sentences(sentences: Iterable[Sequence[str]]) -> bytes:
    """Return sentences as one compact JSON object per line."""
    lines = (
        json.dumps({"tokens": list(sentence)}, separators=(",", ":"))
        for sentence in sentences
    )

    return "".join(f"{line}\n" for line in lines).encode()


def read_sentences(path: Path) -> tuple[tuple[str, ...], ...]:
    """Read a sentence fixture in file order."""
    with path.open(encoding="utf-8") as handle:
        return tuple(tuple(json.loads(line)["tokens"]) for line in handle)


def _lengths(sentences: Iterable[Sequence[str]]) -> dict[int, int]:
    """Return the sentence length histogram, shortest first."""
    return dict(sorted(Counter(len(sentence) for sentence in sentences).items()))


class PinnedFingerprint(NamedTuple):
    """The recorded identity and shape of one sentence fixture."""

    grammar: str
    seed: int
    fingerprint: str
    digest: str
    count: int
    lengths: Mapping[int, int]

    def write(self, path: Path) -> None:
        """Write the record as stable JSON."""
        record = self._asdict()
        record["lengths"] = {str(length): held for length, held in self.lengths.items()}
        encoded = json.dumps(record, indent=2, sort_keys=True)
        path.write_text(f"{encoded}\n", encoding="utf-8", newline="\n")

    @classmethod
    def read(cls, path: Path) -> PinnedFingerprint:
        """Read a record, refusing malformed fields or a malformed histogram."""
        loaded = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(loaded, dict) or set(loaded) != set(cls._fields):
            msg = f"{path} does not hold a pinned fingerprint record"
            raise ValueError(msg)

        lengths = {int(length): held for length, held in loaded["lengths"].items()}

        if sum(lengths.values()) != loaded["count"]:
            msg = (
                f"{path} records {loaded['count']} sentences but its length "
                f"histogram sums to {sum(lengths.values())}"
            )
            raise ValueError(msg)

        return cls(
            loaded["grammar"],
            loaded["seed"],
            loaded["fingerprint"],
            loaded["digest"],
            loaded["count"],
            lengths,
        )


def write_fixture(
    grammar: FixtureGrammar,
    sentences: Sequence[Sequence[str]],
    seed: int,
    sentences_path: Path,
    fingerprint_path: Path,
) -> PinnedFingerprint:
    """Write sentences and the fingerprint that pins their timed workload."""
    encoded = _encode_sentences(sentences)
    sentences_path.parent.mkdir(parents=True, exist_ok=True)
    sentences_path.write_bytes(encoded)
    pinned = PinnedFingerprint(
        grammar.name,
        seed,
        grammar.assemble().grammar.fingerprint,
        sha256(encoded).hexdigest(),
        len(sentences),
        _lengths(sentences),
    )
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    pinned.write(fingerprint_path)

    return pinned


class FixtureMismatchError(Exception):
    """A pinned fixture disagrees with the workload being checked."""

    def __init__(self, what: str, recorded: str, found: str) -> None:
        super().__init__(
            f"the pinned chart fixture's {what} is recorded as {recorded!r} "
            f"but the tree gives {found!r}"
        )
        self.recorded = recorded
        self.found = found


class FingerprintMismatchError(FixtureMismatchError):
    """The compiled grammar's fingerprint is not the recorded one."""

    def __init__(self, recorded: str, found: str) -> None:
        super().__init__("grammar fingerprint", recorded, found)


class SentenceDigestMismatchError(FixtureMismatchError):
    """The sentence file's digest is not the recorded one."""

    def __init__(self, recorded: str, found: str) -> None:
        super().__init__("sentence digest", recorded, found)


class ShapeMismatchError(FixtureMismatchError):
    """The sentence file's count or length histogram is not recorded."""

    def __init__(
        self, recorded: PinnedFingerprint, count: int, lengths: Mapping[int, int]
    ) -> None:
        if count != recorded.count:
            super().__init__("sentence count", str(recorded.count), str(count))
        else:
            super().__init__(
                "length histogram", repr(dict(recorded.lengths)), repr(dict(lengths))
            )


def check_pinned_fixture(
    grammar: Grammar,
    sentences_path: Path,
    fingerprint_path: Path,
) -> PinnedFingerprint:
    """Check a grammar and sentence file against their pinned record.

    Raises
    ------
    FingerprintMismatchError
        If the compiled grammar changed.
    SentenceDigestMismatchError
        If the sentence bytes changed.
    ShapeMismatchError
        If the sentence count or lengths changed.
    """
    pinned = PinnedFingerprint.read(fingerprint_path)

    if grammar.fingerprint != pinned.fingerprint:
        raise FingerprintMismatchError(pinned.fingerprint, grammar.fingerprint)

    digest = sha256(sentences_path.read_bytes()).hexdigest()

    if digest != pinned.digest:
        raise SentenceDigestMismatchError(pinned.digest, digest)

    sentences = read_sentences(sentences_path)
    lengths = _lengths(sentences)

    if len(sentences) != pinned.count or lengths != dict(pinned.lengths):
        raise ShapeMismatchError(pinned, len(sentences), lengths)

    return pinned


class PinnedChart(NamedTuple):
    """The checked grammar, lexicon, sentences, and pin record."""

    grammar: Grammar
    lexicon: StaticLexicon[Term]
    sentences: tuple[tuple[str, ...], ...]
    pinned: PinnedFingerprint


def load_pinned_chart(sentences_path: Path, fingerprint_path: Path) -> PinnedChart:
    """Compile and check a pinned chart fixture from two paths."""
    recorded = PinnedFingerprint.read(fingerprint_path)
    assembly = fixture_grammar(recorded.grammar).assemble()
    grammar = assembly.grammar
    pinned = check_pinned_fixture(grammar, sentences_path, fingerprint_path)

    return PinnedChart(
        grammar,
        assembly.lexicon,
        read_sentences(sentences_path),
        pinned,
    )


def packaged_pinned_chart() -> PinnedChart:
    """Return the checked pricing fixture shipped inside ``lofi``."""
    return load_pinned_chart(
        FIXTURE_DIRECTORY / "chart_200.jsonl",
        FIXTURE_DIRECTORY / "chart_200.fingerprint",
    )
