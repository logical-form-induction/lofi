import json
from hashlib import sha256
from typing import TYPE_CHECKING, Final, Literal

from tests.unit.spec.grammar.test__signature import slash_views, slashes

from lofi.data.laws import ISO_LAWS, IsoFixture
from lofi.data.synthetic import SyntheticDataset
from lofi.data.synthetic.spec.corpus import (
    IllTypingSpec,
    LabelNoiseSpec,
    SyntheticCorpusDecl,
    UniformLengthSpec,
)
from lofi.data.synthetic.spec.interpretation import (
    InterpretationSpec,
    LexicalMeaningSourceSpec,
    MeaningSpec,
)
from lofi.spec import ApplicationSpec, RuleSetSpec
from lofi.spec.grammar import (
    BudgetSpec,
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FeatureErasureMatchingSpec,
    GrammarDecl,
    InlineLexiconSpec,
    LexicalEntrySpec,
    OperatorSpec,
    RootGoalSpec,
    SignatureMorphismErasureSpec,
    SignatureSpec,
)
from lofi.spec.language import ConstantSpec, LanguageDecl, registered_language

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi._types import JsonValue

type Dataset = SyntheticDataset
type IsoLaw = Callable[[IsoFixture[SyntheticCorpusDecl, Dataset]], None]

ISO_LAW_NAMES: Final = tuple(law.__name__ for law in ISO_LAWS)


def _grammar(name: str = "toy", *, forms: int = 4) -> GrammarDecl:
    categories = ("NP", "S\\NP", "(S\\NP)/NP", "NP/NP")

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
        rules=RuleSetSpec(
            name="application",
            rules=(
                ApplicationSpec(view="/"),
                ApplicationSpec(view="\\"),
            ),
        ),
        lexicon=InlineLexiconSpec(
            entries=tuple(
                LexicalEntrySpec(
                    form=f"w{index}", category=categories[index % len(categories)]
                )
                for index in range(forms)
            )
        ),
        matching=FeatureErasureMatchingSpec(),
        budget=BudgetSpec(unary_rounds=1),
    )


def _projection(dataset: Dataset) -> str:
    corpus = dataset.corpus
    document: JsonValue = {
        "declaration": dataset.declaration.decode(),
        "fingerprint": dataset.fingerprint,
        "grammar": corpus.declaration_digest,
        "theory": corpus.theory_digest,
        "examples": [
            [
                example.item.item_id,
                [token.surface for token in example.item.tokens],
                repr(example.supervision),
            ]
            for example in corpus.examples
        ],
        "corrupted": [
            [
                record.item.item_id,
                [token.surface for token in record.item.tokens],
                record.kind,
                list(record.source),
            ]
            for record in corpus.corrupted
        ],
        "labels": [
            [label.item_id, label.true_label, label.observed_label, label.flipped]
            for label in corpus.labels
        ],
        "reports": [
            repr(corpus.ambiguity),
            repr(corpus.length),
            repr(corpus.lexicon),
            repr(corpus.corruption),
            repr(corpus.noise),
        ],
    }

    return sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


INTERPRETED_ENTRIES: Final[tuple[tuple[str, str, str], ...]] = (
    ("john", "NP", "john"),
    ("mary", "NP", "mary"),
    ("sleeps", "S\\NP", "\\x. sleep(x)"),
    ("loves", "(S\\NP)/NP", "\\y. \\x. love(x)(y)"),
    ("and", "(S\\S)/S", "\\q. \\p. p & q"),
    ("not", "S/S", "\\p. ~p"),
)
INTERPRETED_TYPES: Final[dict[str, str]] = {"S": "t", "NP": "e", "conj": "e"}
INTERPRETED_CONSTANTS: Final = (
    ConstantSpec(name="john", type="e", classes=("individual",)),
    ConstantSpec(name="mary", type="e", classes=("individual",)),
    ConstantSpec(name="sleep", type="e -> t", classes=("predicate",)),
    ConstantSpec(name="love", type="e -> e -> t", classes=("predicate",)),
)


def _interpreted_grammar(name: str = "interpreted") -> GrammarDecl:
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
        rules=RuleSetSpec(
            name="application",
            rules=(
                ApplicationSpec(view="/"),
                ApplicationSpec(view="\\"),
            ),
        ),
        lexicon=InlineLexiconSpec(
            entries=tuple(
                LexicalEntrySpec(form=form, category=category, meaning=meaning)
                for form, category, meaning in INTERPRETED_ENTRIES
            )
        ),
        goal=RootGoalSpec(categories=("S",)),
        matching=FeatureErasureMatchingSpec(),
        budget=BudgetSpec(unary_rounds=1),
    )


def _interpretation(
    *,
    atoms: dict[str, str] | None = None,
    constructors: dict[str, str] | None = None,
    type_syntax: Literal["angle", "s-expression"] = "angle",
    constants: tuple[ConstantSpec, ...] = INTERPRETED_CONSTANTS,
    actions: dict[str, str] | None = None,
) -> InterpretationSpec:
    return InterpretationSpec(
        type_syntax=type_syntax,
        erasure=SignatureMorphismErasureSpec(
            atoms=INTERPRETED_TYPES if atoms is None else atoms,
            constructors={} if constructors is None else constructors,
        ),
        constants=constants,
        meanings=MeaningSpec(source=LexicalMeaningSourceSpec()),
        actions={} if actions is None else actions,
    )


def _interpreted(
    *,
    size: int = 10,
    seed: int = 4,
    grammar: GrammarDecl | None = None,
    language: LanguageDecl | None = None,
    interpretation: InterpretationSpec | None = None,
    noise: LabelNoiseSpec | None = None,
    ill_typing: IllTypingSpec | None = None,
) -> SyntheticCorpusDecl:
    return SyntheticCorpusDecl(
        retained={"validation": 0.25},
        grammar=_interpreted_grammar() if grammar is None else grammar,
        language=registered_language("propositional") if language is None else language,
        size=size,
        seed=seed,
        length=UniformLengthSpec(min_length=2, max_length=5),
        noise=noise,
        interpretation=_interpretation() if interpretation is None else interpretation,
        ill_typing=ill_typing,
    )


CHI_SQUARE_CRITICAL: Final = {2: 5.991, 4: 9.488, 5: 11.070, 14: 23.685}
"""The chi-square critical value at the 0.05 level, by degrees of freedom."""


def _chi_square(
    observed: dict[str, int], expected: dict[str, float]
) -> tuple[float, int]:
    """Return the goodness-of-fit statistic and its degrees of freedom."""
    statistic = sum(
        (observed[category] - count) ** 2 / count
        for category, count in expected.items()
    )

    return statistic, len(expected) - 1
