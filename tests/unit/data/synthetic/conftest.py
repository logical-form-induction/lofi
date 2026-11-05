"""Grammars and interpretations whose strings differ in semantic ambiguity."""

import sys
from hashlib import sha256
from math import inf, log
from pathlib import Path
from typing import TYPE_CHECKING, Final, final

from tests.unit.spec.grammar.test__signature import slash_views, slashes

from lofi.data.synthetic import Meaning, SyntheticGenerator
from lofi.data.synthetic.spec.interpretation import (
    InterpretationSpec,
    LexicalMeaningSourceSpec,
    MeaningSpec,
    lower_interpretation,
)
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.spec import (
    ApplicationSpec,
    CompositionSpec,
    RuleSetSpec,
    RuleSpec,
    TypeRaisingSpec,
)
from lofi.spec.grammar import (
    BudgetSpec,
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FeatureErasureMatchingSpec,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    OperatorSpec,
    RootGoalSpec,
    SignatureMorphismErasureSpec,
    SignatureSpec,
)
from lofi.spec.language import (
    ConstantSpec,
    LanguageDecl,
    LanguageIso,
    registered_language,
)
from lofi.spec.provers import (
    LabelingProverSpec,
    ProofBudgetSpec,
    ProverSpec,
    TptpProverSpec,
    TruthTableProverSpec,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.data.synthetic import LogicalClass

type Entry = tuple[str, str, str]
"""A surface form, the category it bears and the meaning it carries there."""

type Generator = SyntheticGenerator
"""A generator over the stores declarations lower to."""

ABSTAINING_TPTP: Final = TptpProverSpec(
    engine="vampire",
    executable=sys.executable,
    dialect="tff",
)
"""A bounded test double that serves quantified profiles and reports no SZS status."""

APPLICATION: Final[tuple[RuleSpec, ...]] = (
    ApplicationSpec(view="/"),
    ApplicationSpec(view="\\"),
)
"""Forward and backward application."""

SCOPING: Final[tuple[RuleSpec, ...]] = (
    *APPLICATION,
    CompositionSpec(direction="unrestricted", order=1, view="/"),
)
"""Application and the forward composition a raised subject needs."""

SPURIOUS: Final[tuple[RuleSpec, ...]] = (
    *SCOPING,
    CompositionSpec(direction="unrestricted", order=1, view="\\"),
    TypeRaisingSpec(view="/", targets=("S",), arguments=None),
)
"""Rules that multiply a string's derivations without changing its form."""

SCOPED_ENTRIES: Final[tuple[Entry, ...]] = (
    ("every-student", "S/(S\\NP)", "\\P. ∀x.(student(x) => P(x))"),
    ("a-student", "S/(S\\NP)", "\\P. ∃x.(student(x) & P(x))"),
    ("a-book", "S\\(S/NP)", "\\P. ∃y.(book(y) & P(y))"),
    ("every-book", "S\\(S/NP)", "\\P. ∀y.(book(y) => P(y))"),
    ("reads", "(S\\NP)/NP", "\\y. \\x. read(x)(y)"),
    ("reads", "(S\\NP)/(S\\(S/NP))", "\\Q. \\x. Q(\\y. read(x)(y))"),
)
"""Two raised subjects and two raised objects over one transitive verb.

The verb bears two categories: at the first the object is left to scope over
the whole sentence, and at the second it is taken in place, so a subject and
an object quantifier are ordered one way by one derivation and the other way
by another.
"""

SCOPED_INVENTORY: Final = (
    ConstantSpec(name="student", type="e -> t", classes=("predicate",)),
    ConstantSpec(name="book", type="e -> t", classes=("predicate",)),
    ConstantSpec(name="read", type="e -> e -> t", classes=("predicate",)),
)
"""The symbols the scoped meanings are built from."""

JOINED_ENTRIES: Final[tuple[Entry, ...]] = (
    ("john-too", "S/(S\\NP)", "\\P. (P(john) & sleep(mary))"),
    ("mary-too", "S\\(S/NP)", "\\P. (P(mary) & sleep(john))"),
    ("reads", "(S\\NP)/NP", "\\y. \\x. read(x)(y)"),
    ("reads", "(S\\NP)/(S\\(S/NP))", "\\Q. \\x. Q(\\y. read(x)(y))"),
)
"""The two scope positions filled by operators that conjoin rather than bind.

The two scopings conjoin the same three propositions in two associations,
so they are quantifier-free and entail each other.
"""

JOINED_INVENTORY: Final = (
    ConstantSpec(name="john", type="e", classes=("individual",)),
    ConstantSpec(name="mary", type="e", classes=("individual",)),
    ConstantSpec(name="sleep", type="e -> t", classes=("predicate",)),
    ConstantSpec(name="read", type="e -> e -> t", classes=("predicate",)),
)
"""The symbols the joined meanings are built from."""

FLAT_ENTRIES: Final[tuple[Entry, ...]] = (
    ("john", "NP", "john"),
    ("mary", "NP", "mary"),
    ("sleeps", "S\\NP", "\\x. sleep(x)"),
    ("loves", "(S\\NP)/NP", "\\y. \\x. love(x)(y)"),
)
"""Names and verbs, with no operator that takes scope over another."""

SPURIOUS_ENTRIES: Final[tuple[Entry, ...]] = (
    *FLAT_ENTRIES,
    ("and", "(S\\S)/S", "\\q. \\p. (p & q)"),
)
"""The flat entries beside a sentential conjunction."""

FLAT_INVENTORY: Final = (
    ConstantSpec(name="john", type="e", classes=("individual",)),
    ConstantSpec(name="mary", type="e", classes=("individual",)),
    ConstantSpec(name="sleep", type="e -> t", classes=("predicate",)),
    ConstantSpec(name="love", type="e -> e -> t", classes=("predicate",)),
)
"""The symbols the flat meanings are built from."""


def language(*quantifiers: str) -> LanguageDecl:
    """Return the first-order language declaring some quantifiers.

    Parameters
    ----------
    *quantifiers : str
        The quantifiers the fragment declares, none for the quantifier-free
        fragment.
    """
    base = (
        registered_language("neo-davidsonian")
        if quantifiers
        else registered_language("propositional")
    )
    return base.with_(
        name="scoped" if quantifiers else "flat",
        constants=tuple(
            constant
            for constant in base.constants
            if "quantifier" not in constant.classes or constant.name in quantifiers
        ),
        logic=base.logic.with_(
            prover=TruthTableProverSpec(timeout=5.0),
            labeling=LabelingProverSpec(budget=ProofBudgetSpec()),
        ),
    )


def grammar(
    name: str, entries: Sequence[Entry], rules: Sequence[RuleSpec]
) -> GrammarDecl:
    """Return a CCG over some entries under some rules.

    Parameters
    ----------
    name : str
        The grammar's name.
    entries : Sequence[Entry]
        The lexical entries, each with the meaning it is interpreted by.
    rules : Sequence[RuleSpec]
        The rules the grammar combines with.
    """
    return GrammarDecl(
        name=name,
        signature=SignatureSpec(
            constructors=slashes(),
            functors=slash_views(),
            primitives=("S", "NP", "conj"),
            operators=(OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),),
            conjunctions=ConjunctionsSpec(
                markers=("conj",),
                pairs=(ConjunctionPairSpec(unmarked="NP", marked="NP[conj]"),),
            ),
        ),
        rules=RuleSetSpec(name=name, rules=tuple(rules)),
        lexicon=InlineLexiconSpec(
            entries=tuple(
                LexicalEntrySpec(form=form, category=category, meaning=meaning)
                for form, category, meaning in entries
            )
        ),
        goal=RootGoalSpec(categories=("S",)),
        matching=FeatureErasureMatchingSpec(),
        budget=BudgetSpec(unary_rounds=1),
    )


def interpretation(
    inventory: tuple[ConstantSpec, ...],
) -> InterpretationSpec:
    """Return the interpretation assigning each entry its meaning.

    Parameters
    ----------
    inventory : tuple[ConstantSpec, ...]
        The non-logical symbols the meanings are built from.
    """
    return InterpretationSpec(
        erasure=SignatureMorphismErasureSpec(atoms={"S": "t", "NP": "e", "conj": "e"}),
        constants=inventory,
        meanings=MeaningSpec(source=LexicalMeaningSourceSpec()),
    )


def built(
    name: str,
    entries: Sequence[Entry],
    inventory: tuple[ConstantSpec, ...],
    rules: Sequence[RuleSpec],
    *quantifiers: str,
    prover: ProverSpec | None = None,
) -> Generator:
    """Compile a grammar and its interpretation into a generator.

    Parameters
    ----------
    name : str
        The grammar's name.
    entries : Sequence[Entry]
        The lexical entries with their meanings.
    inventory : tuple[ConstantSpec, ...]
        The non-logical symbols the meanings are built from.
    rules : Sequence[RuleSpec]
        The rules the grammar combines with.
    *quantifiers : str
        The quantifiers the fragment declares.
    """
    languages = LanguageIso()
    declared_language = language(*quantifiers)
    declared_prover = declared_language.logic.prover if prover is None else prover
    declared_language = declared_language.with_(
        logic=declared_language.logic.with_(
            prover=declared_prover,
            labeling=LabelingProverSpec(budget=ProofBudgetSpec()),
        )
    )
    assembly, interpreted = lower_interpretation(
        grammar(name, entries, rules),
        declared_language,
        interpretation(inventory),
        GrammarIso(languages),
        languages,
    )

    return SyntheticGenerator(
        assembly,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        interpretation=interpreted,
    )


def scoped_generator() -> Generator:
    """Return the generator whose strings scope two quantifiers two ways."""
    return built(
        "scoped",
        SCOPED_ENTRIES,
        SCOPED_INVENTORY,
        SCOPING,
        "forall",
        "exists",
        prover=ABSTAINING_TPTP,
    )


def joined_generator() -> Generator:
    """Return the generator whose two scopings are propositionally equivalent."""
    return built("joined", JOINED_ENTRIES, JOINED_INVENTORY, SCOPING)


def flat_generator() -> Generator:
    """Return the generator whose strings have one derivation and one reading."""
    return built("flat", FLAT_ENTRIES, FLAT_INVENTORY, APPLICATION)


def spurious_generator() -> Generator:
    """Return the generator whose strings have many derivations and one reading."""
    return built("spurious", SPURIOUS_ENTRIES, FLAT_INVENTORY, SPURIOUS)


NEGATED_ENTRIES: Final[tuple[Entry, ...]] = (
    ("john", "NP", "john"),
    ("mary", "NP", "mary"),
    ("sleeps", "S\\NP", "\\x. sleep(x)"),
    ("snores", "S\\NP", "\\x. (sleep(x) & snore(x))"),
    ("wakes", "S\\NP", "\\x. ~sleep(x)"),
    ("loves", "(S\\NP)/NP", "\\y. \\x. love(x)(y)"),
)
"""Predicates that entail, that contradict and that decide each other neutral way.

Snoring entails sleeping, waking contradicts it, and loving someone else does
neutral, so strings over these entries stand in all three inference relations.
The transitive verb is the only entry needing three tokens, so a draw at that
length reaches a form no two-token draw can.
"""

NEGATED_INVENTORY: Final = (
    ConstantSpec(name="john", type="e", classes=("individual",)),
    ConstantSpec(name="mary", type="e", classes=("individual",)),
    ConstantSpec(name="sleep", type="e -> t", classes=("predicate",)),
    ConstantSpec(name="snore", type="e -> t", classes=("predicate",)),
    ConstantSpec(name="love", type="e -> e -> t", classes=("predicate",)),
)
"""The symbols the negated meanings are built from."""


def negated_generator() -> Generator:
    """Return the generator whose strings stand in all three inference relations."""
    return built("negated", NEGATED_ENTRIES, NEGATED_INVENTORY, APPLICATION)


SAMPLE_LEXICON: Final = Path(__file__).with_name("fixtures") / "sample_lexicon.tsv"
"""The committed lexicon sample a resembling declaration is seeded from."""


def sample_rows() -> tuple[tuple[str, str, int], ...]:
    """Return the committed sample as a form, a category and a token count per row.

    Returns
    -------
    tuple[tuple[str, str, int], ...]
        The rows below the header, in file order.
    """
    lines = SAMPLE_LEXICON.read_text(encoding="utf-8").splitlines()[1:]

    return tuple(
        (form, category, int(count))
        for form, category, count in (line.split("\t") for line in lines)
    )


def sample_digest() -> str:
    """Return the SHA-256 of the committed sample's bytes as lowercase hexadecimal."""
    return sha256(SAMPLE_LEXICON.read_bytes()).hexdigest()


SEMANTIC_SAMPLE: Final = Path(__file__).with_name("fixtures") / "sample_semantics.tsv"
"""The committed lexicon sample a resembling semantic declaration is seeded from."""

type SemanticRow = tuple[str, str, LogicalClass, str, int | None, int]
"""A form, its category, and the class, constant, arity and count of its meaning."""


def semantic_sample_rows() -> tuple[SemanticRow, ...]:
    """Return the committed semantic sample as one row per lexical item.

    Returns
    -------
    tuple[SemanticRow, ...]
        The rows below the header, in file order. An item of a class other
        than the predicate class records no arity.
    """
    lines = SEMANTIC_SAMPLE.read_text(encoding="utf-8").splitlines()[1:]

    return tuple(
        (
            form,
            category,
            held,
            symbol,
            int(arity) if arity else None,
            int(count),
        )
        for form, category, held, symbol, arity, count in (
            line.split("\t") for line in lines
        )
    )


def semantic_sample_digest() -> str:
    """Return the SHA-256 of the committed semantic sample as lowercase hexadecimal."""
    return sha256(SEMANTIC_SAMPLE.read_bytes()).hexdigest()


@final
class SampleSemanticSource:
    """The committed semantic sample read as a source of logical forms.

    Parameters
    ----------
    rows : Sequence[SemanticRow]
        One row per item, with the class, the logical constant and the arity
        its meaning is built from and the token count it was read with. An
        item's logical form is rendered as its constant, which an estimate
        does not read.
    """

    def __init__(self, rows: Sequence[SemanticRow]) -> None:
        self._meanings = {
            (category, form): Meaning(logical_class, symbol, arity=arity, text=symbol)
            for form, category, logical_class, symbol, arity, _ in rows
        }
        self._masses = {
            (category, form): log(count) for form, category, _, _, _, count in rows
        }

    @property
    def items(self) -> tuple[tuple[str, str], ...]:
        """Return the category text and surface form of every item, in read order."""
        return tuple(self._meanings)

    def meaning(self, item: tuple[str, str]) -> Meaning | None:
        """Return the logical form an item bears, or nothing for an item not held.

        Parameters
        ----------
        item : tuple[str, str]
            The category text and surface form to look up.
        """
        return self._meanings.get(item)

    def log_mass(self, item: tuple[str, str]) -> float:
        """Return the log token mass of an item, or negative infinity for one not held.

        Parameters
        ----------
        item : tuple[str, str]
            The category text and surface form to look up.
        """
        return self._masses.get(item, -inf)


def semantic_source() -> SampleSemanticSource:
    """Return the committed semantic sample as a source."""
    return SampleSemanticSource(semantic_sample_rows())
