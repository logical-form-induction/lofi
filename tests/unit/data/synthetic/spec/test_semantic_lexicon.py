from collections import Counter
from functools import cache
from itertools import pairwise
from json import loads
from math import inf
from typing import TYPE_CHECKING, Final, cast

import didactic.api as dx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.unit.data.synthetic.conftest import (
    APPLICATION,
    SampleSemanticSource,
    semantic_sample_digest,
    semantic_sample_rows,
    semantic_source,
)
from tests.unit.data.synthetic.spec.conftest import (
    CHI_SQUARE_CRITICAL,
    _chi_square,
    _grammar,
)
from tests.unit.spec.grammar.test__signature import slash_views, slashes

import lofi.data.synthetic.spec.semantic_lexicon as semantics_module
from lofi.data.synthetic import (
    LexiconReport,
    SemanticReport,
    SyntheticCorpus,
    SyntheticGenerator,
    Target,
    UniformLength,
)
from lofi.data.synthetic.spec.semantic_lexicon import (
    ArityCountSpec,
    ArityLawSpec,
    AritySpec,
    AssignmentCountSpec,
    AssignmentRowSpec,
    AssignmentSpec,
    ClassWeightSpec,
    IndependentAssignmentSpec,
    ParametricSemanticJointSpec,
    QuantifierMeaningSpec,
    ResemblingSemanticLexiconSpec,
    SemanticClassSpec,
    SemanticFragmentSpec,
    SemanticLexiconEstimateSpec,
    SemanticLexiconStructure,
    TabularAssignmentSpec,
    TabularSemanticJointSpec,
    preset,
)
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.spec import DeclarationError, RuleSetSpec
from lofi.spec.grammar import (
    BudgetSpec,
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FeatureErasureMatchingSpec,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    LexiconStructure,
    OperatorSpec,
    ParametricAmbiguityLawSpec,
    ParametricJointSpec,
    PowerLawRankLawSpec,
    RootGoalSpec,
    SignatureSpec,
    UniformRankLawSpec,
)
from lofi.spec.language import registered_language

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path
    from typing import Protocol

    from lofi.data.synthetic import LogicalClass, SemanticLexicon
    from lofi.spec.grammar import GrammarAssembly

    class _SemanticJointDeclaration(Protocol):
        """A declaration that supplies logical-form classes and their mass."""

        def classes(self) -> tuple[tuple[LogicalClass, int, float], ...]:
            """Return each class with its size and token share."""
            ...


NATURAL_CATEGORIES: Final = (
    "N",
    "NP",
    "NP/N",
    "S\\NP",
    "(S\\NP)/NP",
    "((S\\NP)/NP)/NP",
    "(S\\NP)/(S\\NP)",
)
SEMANTIC_COUPLINGS: Final = (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0)
ARITY_SHARES: Final = {1: 0.5, 2: 0.3, 3: 0.2}
SEMANTIC_COUNT: Final = 3000


def _structured_grammar(seed: int = 5) -> GrammarDecl:
    """Declare the natural categories over a flat syntactic joint."""
    return _grammar("structured").with_(
        signature=SignatureSpec(
            constructors=slashes(),
            functors=slash_views(),
            primitives=("S", "NP", "N", "conj"),
            operators=(OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),),
            conjunctions=ConjunctionsSpec(
                markers=("conj",),
                pairs=(ConjunctionPairSpec(unmarked="NP", marked="NP[conj]"),),
            ),
        ),
        lexicon=LexiconStructure(
            forms=800,
            joint=ParametricJointSpec(
                categories=NATURAL_CATEGORIES,
                category_shares=PowerLawRankLawSpec(exponent=1.0),
            ),
            ambiguity=ParametricAmbiguityLawSpec(
                mean=2.0, dispersion=0.5, maximum=4, placement=0.0
            ),
            rank_law=UniformRankLawSpec(),
            seed=seed,
        ),
        budget=BudgetSpec(unary_rounds=0),
    )


@cache
def _structured_assembly() -> GrammarAssembly:
    return GrammarIso().forward(
        _structured_grammar(), registered_language("neo-davidsonian")
    )


def _canonical_category(text: str) -> str:
    """Return category text rendered by the structured grammar's syntax."""
    assembly = _structured_assembly()
    category = assembly.syntax.parse(assembly.store, text)

    return assembly.syntax.render(assembly.store, category)


@cache
def _structured_corpus() -> SyntheticCorpus:
    return SyntheticGenerator(
        _structured_assembly(),
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    ).sample(SEMANTIC_COUNT, seed=1, length=UniformLength(2, 4))


def _arity_law() -> ArityLawSpec:
    return ArityLawSpec(
        arities=tuple(
            AritySpec(arity=arity, share=share) for arity, share in ARITY_SHARES.items()
        )
    )


def _semantics(
    coupling: float,
    *,
    assignment: AssignmentSpec | None = None,
    seed: int = 5,
) -> SemanticLexiconStructure:
    return SemanticLexiconStructure(
        joint=ParametricSemanticJointSpec(
            members=("predicate", "constant", "connective", "quantifier"),
            largest_class=128,
            class_sizes=PowerLawRankLawSpec(exponent=3.0),
            coupling=coupling,
        ),
        arities=_arity_law(),
        assignment=IndependentAssignmentSpec() if assignment is None else assignment,
        seed=seed,
    )


@cache
def _measured(coupling: float, seed: int = 5) -> SemanticReport:
    """Measure the independent assignment at a coupling over the one drawn corpus."""
    return SemanticReport.of(
        _semantics(coupling, seed=seed).build(_structured_assembly()),
        (_structured_corpus(),),
    )


def _per_form(report: LexiconReport) -> dict[str, float]:
    return {
        logical_class: report.token_counts[logical_class] / size
        for logical_class, size in report.class_sizes.items()
    }


def _extreme_classes(report: LexiconReport) -> tuple[str, str]:
    ranked = sorted(report.class_sizes, key=lambda held: report.class_sizes[held])

    return ranked[0], ranked[-1]


def _binders(text: str) -> set[str]:
    """Return quantifier spellings occurring in generated meaning text."""
    spellings = {"forall": ("forall", "∀"), "exists": ("exists", "∃")}
    return {
        name
        for name, alternatives in spellings.items()
        if any(alternative in text for alternative in alternatives)
    }


def _one_place_predicate(text: str) -> bool:
    """Return whether a form abstracts one variable and applies a symbol to it."""
    compact = "".join(text.split())

    if not compact or compact[0] not in ("\\", "λ") or "." not in compact:
        return False

    binder, body = compact[1:].split(".", maxsplit=1)
    return body.endswith(f"({binder})") and body[: -len(binder) - 2].isidentifier()


def _reversed(natural: TabularAssignmentSpec) -> TabularAssignmentSpec:
    """Swap the classes the determiner and noun rows draw."""
    by_category = {row.category: row.classes for row in natural.rows}

    return natural.with_(
        rows=tuple(
            row.with_(
                classes=by_category[
                    {"NP/N": "N", "N": "NP/N"}.get(row.category, row.category)
                ]
            )
            for row in natural.rows
        )
    )


def test_the_natural_preset_gives_determiners_quantifiers_and_nouns_predicates() -> (
    None
):
    natural = preset("natural")
    assembly = _structured_assembly()
    lexicon = natural.build(assembly)
    report = SemanticReport.of(lexicon, (_structured_corpus(),))
    by_category: dict[str, set[str]] = {}

    for (category, _), meanings in lexicon.meanings.items():
        by_category.setdefault(category, set()).update(
            meaning.logical_class for meaning in meanings
        )

    assert isinstance(natural.assignment, TabularAssignmentSpec)
    assert by_category[_canonical_category("NP/N")] == {"quantifier"}
    assert by_category[_canonical_category("N")] == {"predicate"}
    assert by_category[_canonical_category("NP")] == {"constant"}
    assert by_category[_canonical_category("(S\\NP)/(S\\NP)")] == {"identity"}
    assert all(
        _binders(meaning.text) & {"forall", "exists"}
        for (category, _), meanings in lexicon.meanings.items()
        if category == _canonical_category("NP/N")
        for meaning in meanings
    )
    assert all(
        _one_place_predicate(meaning.text)
        for (category, _), meanings in lexicon.meanings.items()
        if category == _canonical_category("N")
        for meaning in meanings
    )
    assert {
        meaning.arity
        for (category, _), meanings in lexicon.meanings.items()
        if category == _canonical_category("((S\\NP)/NP)/NP")
        for meaning in meanings
    } == {3}
    assert report.concordance is not None
    assert Target(1.0, 0.01).admits(report.concordance)


def test_the_independence_assignment_has_near_zero_concordance() -> None:
    report = _measured(0.0)

    assert report.concordance is not None
    assert Target(0.0, 0.1).admits(report.concordance)
    assert set(report.cells) == {
        _canonical_category(category) for category in NATURAL_CATEGORIES
    }


def test_a_table_reversing_the_natural_one_gives_determiners_one_place_predicates() -> (
    None
):
    natural = preset("natural")

    assert isinstance(natural.assignment, TabularAssignmentSpec)

    lexicon = natural.with_(assignment=_reversed(natural.assignment)).build(
        _structured_assembly()
    )
    determiners = [
        meaning
        for (category, _), meanings in lexicon.meanings.items()
        if category == "NP/N"
        for meaning in meanings
    ]
    nouns = [
        meaning
        for (category, _), meanings in lexicon.meanings.items()
        if category == "N"
        for meaning in meanings
    ]

    assert determiners
    assert nouns
    assert all(_one_place_predicate(meaning.text) for meaning in determiners)
    assert all(_binders(meaning.text) & {"forall", "exists"} for meaning in nouns)
    assert SemanticReport.of(
        lexicon, (_structured_corpus(),)
    ).concordance == pytest.approx(1.0)


def test_a_quantifier_free_fragment_assigns_no_quantifier_form() -> None:
    fragment = SemanticFragmentSpec(order="first", quantifiers=())
    structure = SemanticLexiconStructure(
        joint=TabularSemanticJointSpec(
            rows=(
                SemanticClassSpec(logical_class="predicate", size=20, share=0.6),
                SemanticClassSpec(logical_class="constant", size=4, share=0.2),
                SemanticClassSpec(logical_class="connective", size=3, share=0.2),
            )
        ),
        arities=_arity_law(),
        assignment=IndependentAssignmentSpec(),
        fragment=fragment,
        seed=2,
    )
    lexicon = structure.build(_structured_assembly())

    assert len(lexicon) == len(_structured_assembly().lexicon)
    assert not any(
        _binders(meaning.text) & {"forall", "exists"}
        for meanings in lexicon.meanings.values()
        for meaning in meanings
    )
    assert "quantifier" not in lexicon.inventory

    with pytest.raises(DeclarationError, match="declares 0 quantifiers"):
        _semantics(0.0).with_(fragment=fragment).build(_structured_assembly())


def test_an_undeclared_fragment_inventory_comes_from_the_run_language() -> None:
    assembly = GrammarIso().forward(
        _structured_grammar(), registered_language("propositional")
    )
    fragment = SemanticFragmentSpec().build(assembly.language)

    assert fragment == assembly.language.profile.fragment


def test_a_positive_coupling_makes_the_smallest_classes_most_frequent_per_form() -> (
    None
):
    report = _measured(1.0).joint
    per_form = _per_form(report)
    smallest, largest = _extreme_classes(report)

    assert per_form[smallest] == max(per_form.values())
    assert per_form[largest] == min(per_form.values())
    assert report.coupling is not None
    assert Target(1.0, 0.25).admits(report.coupling)
    assert report.concordance is not None
    assert report.concordance > 0.5


def test_a_zero_coupling_leaves_per_form_frequency_uncorrelated_with_class_size() -> (
    None
):
    report = _measured(0.0).joint

    assert report.coupling is not None
    assert Target(0.0, 0.2).admits(report.coupling)
    assert report.concordance is not None
    assert Target(0.0, 0.1).admits(report.concordance)


def test_a_negative_coupling_makes_the_largest_classes_most_frequent_per_form() -> None:
    report = _measured(-1.0).joint
    per_form = _per_form(report)
    smallest, largest = _extreme_classes(report)

    assert per_form[largest] == max(per_form.values())
    assert per_form[smallest] == min(per_form.values())
    assert report.coupling is not None
    assert Target(-1.0, 0.25).admits(report.coupling)
    assert report.concordance is not None
    assert report.concordance < -0.5


@settings(deadline=None, max_examples=8)
@given(st.lists(st.sampled_from(SEMANTIC_COUPLINGS), min_size=2, unique=True))
def test_the_concordance_is_monotone_in_the_coupling_across_a_family(
    couplings: list[float],
) -> None:
    concordances = [
        _measured(coupling).joint.concordance for coupling in sorted(couplings)
    ]

    assert None not in concordances

    for earlier, later in pairwise(concordances):
        assert earlier is not None
        assert later is not None
        assert earlier <= later

    for coupling in couplings:
        report = _measured(coupling)

        assert report.joint.class_sizes == _measured(0.0).joint.class_sizes
        assert report.lexicon.inventory == _measured(0.0).lexicon.inventory

        if coupling == 0.0:
            assert Target(0.0, 0.1).admits(report.joint.concordance or 0.0)


def test_changing_only_the_coupling_changes_only_the_achieved_joint() -> None:
    base = _semantics(0.0)
    varied = _semantics(1.0)
    base_document = loads(base.model_dump_json())
    varied_document = loads(varied.model_dump_json())

    assert {
        key for key in base_document if base_document[key] != varied_document[key]
    } == {"joint"}
    assert {
        key
        for key in base_document["joint"]
        if base_document["joint"][key] != varied_document["joint"][key]
    } == {"coupling"}

    base_report = _measured(0.0)
    varied_report = _measured(1.0)

    assert base_report.lexicon.inventory == varied_report.lexicon.inventory
    assert base_report.joint.class_sizes == varied_report.joint.class_sizes
    assert base_report.joint.forms.total == varied_report.joint.forms.total
    assert base_report.lexicon.shares != varied_report.lexicon.shares
    assert base_report.joint.token_counts != varied_report.joint.token_counts
    assert base_report.joint.concordance != varied_report.joint.concordance
    assert (
        _structured_corpus().lexicon
        == SyntheticGenerator(
            _structured_assembly(),
            policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        )
        .sample(SEMANTIC_COUNT, seed=1, length=UniformLength(2, 4))
        .lexicon
    )


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_achieved_arity_histogram_is_within_tolerance_of_the_declared_law(
    seed: int,
) -> None:
    report = _measured(-1.0, seed)
    arities = report.arities
    total = arities.total

    assert total > 100

    for arity, share in ARITY_SHARES.items():
        assert Target(share, 0.08).admits(arities[arity] / total)

    assert report.lexicon.sizes["predicate"] == 128
    assert Counter(
        meaning.arity for meaning in report.lexicon.inventory["predicate"]
    ) == {1: 64, 2: 38, 3: 26}


@pytest.mark.parametrize(
    ("size", "shares", "expected"),
    [
        (10, (1.0, 1.0), (5, 5)),
        (3, (0.9, 0.05, 0.05), (1, 1, 1)),
        (7, (0.5, 0.3, 0.2), (3, 2, 2)),
        (128, (0.5, 0.3, 0.2), (64, 38, 26)),
    ],
)
def test_the_arity_law_apportions_forms_by_largest_remainder(
    size: int, shares: tuple[float, ...], expected: tuple[int, ...]
) -> None:
    law = ArityLawSpec(
        arities=tuple(
            AritySpec(arity=arity, share=share) for arity, share in enumerate(shares, 1)
        )
    )

    assert law.counts(size) == tuple(enumerate(expected, 1))
    assert len(law.weights(size)) == size

    with pytest.raises(ValueError, match="cannot bear"):
        law.counts(len(shares) - 1)


def test_the_tabular_and_parametric_spellings_lower_to_one_assignment() -> None:
    parametric = _semantics(1.0)
    tabular = parametric.with_(
        joint=TabularSemanticJointSpec(
            rows=tuple(
                SemanticClassSpec(logical_class=held, size=size, share=share)
                for held, size, share in cast(
                    "_SemanticJointDeclaration", parametric.joint
                ).classes()
            )
        )
    )

    for (held, size, share), (other, other_size, other_share) in zip(
        cast("_SemanticJointDeclaration", tabular.joint).classes(),
        cast("_SemanticJointDeclaration", parametric.joint).classes(),
        strict=True,
    ):
        assert (held, size) == (other, other_size)
        assert share == pytest.approx(other_share)

    assert tabular.build(_structured_assembly()) == parametric.build(
        _structured_assembly()
    )


def test_the_same_seed_draws_the_same_assignment_and_another_varies_it() -> None:
    first = _semantics(0.0, seed=3).build(_structured_assembly())

    assert first == _semantics(0.0, seed=3).build(_structured_assembly())
    assert first != _semantics(0.0, seed=4).build(_structured_assembly())


def test_the_semantic_structure_round_trips_through_its_bytes() -> None:
    for structure in (_semantics(0.5), preset("natural")):
        rebuilt = SemanticLexiconStructure.model_validate_json(
            structure.model_dump_json()
        )

        assert rebuilt == structure
        assert rebuilt.build(_structured_assembly()) == structure.build(
            _structured_assembly()
        )

    with pytest.raises(FileNotFoundError):
        preset("no-such-preset")


def test_a_semantic_preset_resolves_from_a_config_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An installed configuration root can supply a semantic preset."""
    directory = tmp_path / "semantic_lexicons"
    directory.mkdir()
    expected = preset("natural").with_(seed=19)
    (directory / "external.json").write_text(
        expected.model_dump_json(), encoding="utf-8", newline="\n"
    )
    monkeypatch.setattr(
        semantics_module,
        "config_search_path",
        lambda _root: (tmp_path,),
    )

    assert preset("external") == expected


SEMANTIC_REFUSALS: Final[
    dict[str, tuple[Callable[[], SemanticLexiconStructure], str]]
] = {
    "quantifiers-beyond-the-fragment": (
        lambda: _semantics(0.0).with_(
            fragment=SemanticFragmentSpec(
                order="first", quantifiers=(QuantifierMeaningSpec(name="forall"),)
            )
        ),
        "declares 1 quantifiers",
    ),
    "a-quantifier-without-its-connective": (
        lambda: SemanticLexiconStructure(
            joint=TabularSemanticJointSpec(
                rows=(
                    SemanticClassSpec(logical_class="quantifier", size=2, share=1.0),
                    SemanticClassSpec(logical_class="constant", size=2, share=1.0),
                )
            ),
            assignment=IndependentAssignmentSpec(),
            fragment=SemanticFragmentSpec(
                order="first", connectives=("and", "or", "iff", "not")
            ),
            seed=1,
        ),
        "restricts by the connective 'if'",
    ),
    "connectives-beyond-the-fragment": (
        lambda: _semantics(0.0).with_(
            fragment=SemanticFragmentSpec(order="first", connectives=("and",))
        ),
        "declares 1 connectives",
    ),
    "an-identity-class-of-two-forms": (
        lambda: SemanticLexiconStructure(
            joint=TabularSemanticJointSpec(
                rows=(SemanticClassSpec(logical_class="identity", size=2, share=1.0),)
            ),
            assignment=IndependentAssignmentSpec(),
            seed=1,
        ),
        "one form",
    ),
    "a-category-without-a-row": (
        lambda: _semantics(
            0.0,
            assignment=TabularAssignmentSpec(
                rows=(
                    AssignmentRowSpec(
                        category="N",
                        classes=(ClassWeightSpec(logical_class="predicate"),),
                    ),
                )
            ),
        ),
        "no row for the categories",
    ),
    "a-row-naming-an-undeclared-class": (
        lambda: _semantics(
            0.0,
            assignment=TabularAssignmentSpec(
                rows=tuple(
                    AssignmentRowSpec(
                        category=category,
                        classes=(ClassWeightSpec(logical_class="identity"),),
                    )
                    for category in NATURAL_CATEGORIES
                )
            ),
        ),
        "does not declare",
    ),
    "a-row-forcing-an-absent-arity": (
        lambda: _semantics(
            0.0,
            assignment=TabularAssignmentSpec(
                rows=tuple(
                    AssignmentRowSpec(
                        category=category,
                        classes=(ClassWeightSpec(logical_class="predicate", arity=4),),
                    )
                    for category in NATURAL_CATEGORIES
                )
            ),
        ),
        "forces arity 4",
    ),
    "a-row-the-syntax-cannot-read": (
        lambda: _semantics(
            0.0,
            assignment=TabularAssignmentSpec(
                rows=(
                    AssignmentRowSpec(
                        category="((S",
                        classes=(ClassWeightSpec(logical_class="predicate"),),
                    ),
                )
            ),
        ),
        "does not read",
    ),
}


@pytest.mark.parametrize(
    "case", SEMANTIC_REFUSALS.values(), ids=SEMANTIC_REFUSALS.keys()
)
def test_the_semantic_structure_refuses_what_the_fragment_or_joint_cannot_bear(
    case: tuple[Callable[[], SemanticLexiconStructure], str],
) -> None:
    build, message = case

    with pytest.raises(DeclarationError, match=message):
        build().build(_structured_assembly())


def test_a_category_assignment_can_condition_connectives_on_arity() -> None:
    row = ClassWeightSpec(logical_class="connective", arity=1)

    assert (row.logical_class, row.arity) == ("connective", 1)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: _semantics(0.0).with_(arities=None), "needs an arity law"),
        (
            lambda: SemanticLexiconStructure(
                joint=TabularSemanticJointSpec(
                    rows=(
                        SemanticClassSpec(logical_class="constant", size=2, share=1.0),
                    )
                ),
                arities=_arity_law(),
                assignment=IndependentAssignmentSpec(),
                seed=1,
            ),
            "needs a predicate class",
        ),
        (
            lambda: ClassWeightSpec(logical_class="constant", arity=1),
            "only a predicate or connective row",
        ),
        (lambda: ClassWeightSpec(logical_class="predicate", arity=0), "positive"),
        (lambda: ClassWeightSpec(logical_class="predicate", weight=0.0), "positive"),
        (
            lambda: SemanticClassSpec(logical_class="constant", size=0, share=1.0),
            "positive",
        ),
        (
            lambda: SemanticClassSpec(logical_class="constant", size=1, share=0.0),
            "positive",
        ),
        (
            lambda: TabularSemanticJointSpec(
                rows=(
                    SemanticClassSpec(logical_class="constant", size=1, share=1.0),
                    SemanticClassSpec(logical_class="constant", size=2, share=1.0),
                )
            ),
            "twice",
        ),
        (
            lambda: ParametricSemanticJointSpec(
                members=("constant", "constant"),
                largest_class=4,
                class_sizes=UniformRankLawSpec(),
                coupling=0.0,
            ),
            "twice",
        ),
        (
            lambda: ArityLawSpec(
                arities=(
                    AritySpec(arity=1, share=1.0),
                    AritySpec(arity=1, share=1.0),
                )
            ),
            "twice",
        ),
        (
            lambda: TabularAssignmentSpec(
                rows=(
                    AssignmentRowSpec(
                        category="N",
                        classes=(ClassWeightSpec(logical_class="constant"),),
                    ),
                    AssignmentRowSpec(
                        category="N",
                        classes=(ClassWeightSpec(logical_class="identity"),),
                    ),
                )
            ),
            "twice",
        ),
    ],
)
def test_the_semantic_declarations_refuse_a_setting_outside_their_range(
    build: Callable[[], dx.Model], message: str
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        build()


SEMANTIC_REPEATS: Final = 20
"""How many items the drawn grammar holds for each item of the committed sample."""


def _semantic_categories() -> list[str]:
    return [
        category
        for _, category, _, _, _, _ in semantic_sample_rows()
        for _ in range(SEMANTIC_REPEATS)
    ]


def _semantic_grammar(categories: Sequence[str] | None = None) -> GrammarDecl:
    """Declare a CCG over some categories, the sample's at their item shares."""
    held = _semantic_categories() if categories is None else categories

    return GrammarDecl(
        name="resembling-semantics",
        signature=SignatureSpec(
            constructors=slashes(),
            functors=slash_views(),
            primitives=("S", "NP", "N", "conj"),
            operators=(OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),),
            conjunctions=ConjunctionsSpec(
                markers=("conj",),
                pairs=(ConjunctionPairSpec(unmarked="NP", marked="NP[conj]"),),
            ),
        ),
        rules=RuleSetSpec(name="application", rules=APPLICATION),
        lexicon=InlineLexiconSpec(
            entries=tuple(
                LexicalEntrySpec(form=f"w{index}", category=category)
                for index, category in enumerate(held)
            )
        ),
        goal=RootGoalSpec(categories=("S",)),
        matching=FeatureErasureMatchingSpec(),
        budget=BudgetSpec(unary_rounds=0),
    )


@cache
def _semantic_assembly() -> GrammarAssembly:
    return GrammarIso().forward(
        _semantic_grammar(), registered_language("propositional")
    )


@cache
def _semantic_source_estimate() -> SemanticLexiconEstimateSpec:
    return SemanticLexiconEstimateSpec.measured(
        semantic_source(), semantic_sample_digest()
    )


def _resembling_semantics(
    constants: int,
    *,
    source: SemanticLexiconEstimateSpec | None = None,
    seed: int = 7,
) -> ResemblingSemanticLexiconSpec:
    """Declare a semantic lexicon drawn to resemble the committed sample."""
    return ResemblingSemanticLexiconSpec(
        constants=constants,
        estimate=_semantic_source_estimate() if source is None else source,
        seed=seed,
    )


@cache
def _drawn_semantics(constants: int, seed: int = 7) -> SemanticLexicon:
    """Draw the semantic lexicon a resembling declaration assigns the grammar."""
    return _resembling_semantics(constants, seed=seed).build(_semantic_assembly())


def _expected_cells() -> dict[str, float]:
    """Return how many items each category, class and arity is expected to draw."""
    assembly = _semantic_assembly()

    def canonical(text: str) -> str:
        category = assembly.syntax.parse(assembly.store, text)

        return assembly.syntax.render(assembly.store, category)

    return {
        f"{canonical(row.category)}|{row.logical_class}|{row.arity}": (
            row.items * SEMANTIC_REPEATS
        )
        for row in _semantic_source_estimate().assignment
    }


def _observed_cells(lexicon: SemanticLexicon) -> dict[str, int]:
    """Return source-estimate cells, whose arity axis belongs to predicates."""
    return Counter(
        f"{category}|{meaning.logical_class}|"
        f"{meaning.arity if meaning.logical_class == 'predicate' else None}"
        for (category, _), meanings in lexicon.meanings.items()
        for meaning in meanings
    )


def _totaled(counts: dict[str, float], at: int) -> dict[str, float]:
    """Return the counts summed over every cell sharing one field of its key."""
    held: dict[str, float] = {}

    for key, count in counts.items():
        held[key.split("|")[at]] = held.get(key.split("|")[at], 0.0) + count

    return held


@pytest.mark.parametrize("constants", [24, 72])
def test_a_resembling_semantic_lexicon_keeps_the_source_s_conditional(
    constants: int,
) -> None:
    expected = _expected_cells()
    observed = _observed_cells(_drawn_semantics(constants))
    statistic, freedom = _chi_square(observed, expected)

    assert set(observed) == set(expected)
    assert sum(observed.values()) == sum(expected.values())
    assert statistic < CHI_SQUARE_CRITICAL[freedom]


@pytest.mark.parametrize("constants", [24, 72])
def test_a_resembling_semantic_lexicon_keeps_the_source_s_per_class_distribution(
    constants: int,
) -> None:
    expected = _totaled(_expected_cells(), 1)
    observed = {
        logical_class: int(count)
        for logical_class, count in _totaled(
            dict(_observed_cells(_drawn_semantics(constants))), 1
        ).items()
    }
    statistic, freedom = _chi_square(observed, expected)

    assert set(observed) == set(expected)
    assert statistic < CHI_SQUARE_CRITICAL[freedom]


@pytest.mark.parametrize("constants", [24, 72])
def test_a_resembling_semantic_lexicon_keeps_the_source_s_per_arity_distribution(
    constants: int,
) -> None:
    expected = {
        arity: count
        for arity, count in _totaled(_expected_cells(), 2).items()
        if arity != "None"
    }
    observed = {
        arity: int(count)
        for arity, count in _totaled(
            dict(_observed_cells(_drawn_semantics(constants))), 2
        ).items()
        if arity != "None"
    }
    statistic, freedom = _chi_square(observed, expected)

    assert set(observed) == {"1", "2", "3"}
    assert statistic < CHI_SQUARE_CRITICAL[freedom]


@pytest.mark.parametrize("constants", [24, 72])
def test_a_resembling_semantic_lexicon_draws_the_declared_inventory(
    constants: int,
) -> None:
    scaled = _semantic_source_estimate().build().at(constants)

    assert dict(_drawn_semantics(constants).sizes) == {
        logical_class: size for logical_class, size, _ in scaled
    }
    assert sum(_drawn_semantics(constants).sizes.values()) == constants


def test_the_drawn_logical_constants_are_a_subset_of_the_source_s() -> None:
    source = _semantic_source_estimate()
    smaller = _drawn_semantics(24)
    larger = _drawn_semantics(72)

    def named(lexicon: SemanticLexicon, logical_class: str) -> set[str]:
        return {meaning.symbol for meaning in lexicon.inventory.get(logical_class, ())}

    assert named(smaller, "quantifier") < set(source.quantifiers)
    assert named(smaller, "connective") < set(source.connectives)
    assert named(larger, "quantifier") == set(source.quantifiers)
    assert named(larger, "connective") == set(source.connectives)


def test_the_semantic_structure_carries_the_scaled_table_and_the_source_fragment() -> (
    None
):
    source = _semantic_source_estimate()
    structure = _resembling_semantics(72).structure()
    scaled = source.build().at(72)
    shares = {logical_class: share for logical_class, _, share in scaled}
    weights = {
        (row.category, held.logical_class, held.arity): held.weight
        for row in cast("TabularAssignmentSpec", structure.assignment).rows
        for held in row.classes
    }

    assert structure.seed == 7
    assert structure.rank_law == PowerLawRankLawSpec(exponent=source.exponent)
    assert structure.fragment == SemanticFragmentSpec(
        order="first",
        quantifiers=tuple(
            QuantifierMeaningSpec(name=name) for name in source.quantifiers
        ),
        connectives=source.connectives,
    )
    classes = cast("_SemanticJointDeclaration", structure.joint).classes()

    assert [(held, size) for held, size, _ in classes] == [
        (held, size) for held, size, _ in scaled
    ]
    assert [share for _, _, share in classes] == pytest.approx(
        [share for _, _, share in scaled]
    )
    assert structure.arities == ArityLawSpec(
        arities=tuple(
            AritySpec(
                arity=row.arity,
                share=float(row.constants),
                rank_law=PowerLawRankLawSpec(exponent=source.exponent),
            )
            for row in source.arities
        )
    )
    assert weights == pytest.approx(
        {
            (row.category, row.logical_class, row.arity): row.items
            / shares[row.logical_class]
            for row in source.assignment
        }
    )


def test_equal_seeds_draw_equal_semantic_lexicons_and_other_seeds_do_not() -> None:
    assert _drawn_semantics(72, 7) == _drawn_semantics(72, 7)
    assert _drawn_semantics(72, 8) != _drawn_semantics(72, 7)


def test_the_semantic_estimate_bytes_change_when_the_source_digest_changes() -> None:
    seeded = _resembling_semantics(24)
    elsewhere = _resembling_semantics(
        24, source=_semantic_source_estimate().with_(digest="0" * 64)
    )

    assert semantic_sample_digest() in seeded.model_dump_json()
    assert seeded.model_dump_json() != elsewhere.model_dump_json()
    assert (
        ResemblingSemanticLexiconSpec.model_validate_json(seeded.model_dump_json())
        == seeded
    )


def test_an_inventory_dropping_every_class_a_category_bears_is_refused() -> None:
    with pytest.raises(DeclarationError, match="drops every class"):
        _resembling_semantics(2).structure()


def test_a_semantic_source_with_no_rank_frequency_curve_is_refused() -> None:
    lone = SampleSemanticSource([("john", "NP", "constant", "john", None, 3)])

    with pytest.raises(DeclarationError, match="cannot be estimated"):
        SemanticLexiconEstimateSpec.measured(lone, "one")


def test_a_grammar_category_the_source_never_held_is_refused() -> None:
    assembly = GrammarIso().forward(
        _semantic_grammar([*_semantic_categories(), "NP/NP"]),
        registered_language("propositional"),
    )

    with pytest.raises(
        DeclarationError, match=r"no row for the categories \['NP/NP'\]"
    ):
        _resembling_semantics(72).build(assembly)


def test_a_quantifier_restricting_by_an_unused_connective_is_refused() -> None:
    narrow = SampleSemanticSource(
        [
            ("every", "NP/N", "quantifier", "forall", None, 95),
            ("and", "S\\NP", "connective", "and", None, 130),
            ("dog", "N", "predicate", "dog", 1, 155),
        ]
    )
    declared = ResemblingSemanticLexiconSpec(
        constants=3,
        estimate=SemanticLexiconEstimateSpec.measured(narrow, "narrow"),
        seed=1,
    )

    assembly = GrammarIso().forward(
        _semantic_grammar(["NP/N", "S\\NP", "N"]),
        registered_language("propositional"),
    )

    with pytest.raises(DeclarationError, match="restricts by the connective 'if'"):
        declared.build(assembly)


def test_quantifier_names_do_not_encode_their_logical_order() -> None:
    generalized = SampleSemanticSource(
        [
            ("most", "NP/N", "quantifier", "most", None, 95),
            ("dog", "N", "predicate", "dog", 1, 155),
        ]
    )
    estimated = SemanticLexiconEstimateSpec.measured(generalized, "generalized")
    declared = ResemblingSemanticLexiconSpec(constants=2, estimate=estimated, seed=1)

    language = _structured_assembly().language

    assert declared.structure().fragment.build(language).quantifiers == frozenset(
        {"most"}
    )
    assert (
        declared.with_(order="second").structure().fragment.build(language).order
        == "second"
    )


def test_a_resembling_semantic_declaration_refuses_an_empty_inventory() -> None:
    with pytest.raises(dx.ValidationError, match="constant inventory must be positive"):
        ResemblingSemanticLexiconSpec(
            constants=0, estimate=_semantic_source_estimate(), seed=1
        )


@pytest.mark.parametrize(
    ("logical_class", "arity", "message"),
    [
        ("constant", 1, "only a predicate row records an arity"),
        ("predicate", None, "only a predicate row records an arity"),
        ("predicate", 0, "an arity must be positive"),
    ],
)
def test_an_assigned_count_refuses_an_arity_its_class_cannot_bear(
    logical_class: str, arity: int | None, message: str
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        AssignmentCountSpec(
            category="N",
            logical_class=logical_class,
            items=1,
            arity=arity,
        )


@pytest.mark.parametrize(
    ("arity", "constants", "message"),
    [
        (0, 1, "an arity must be positive"),
        (1, 0, "an arity needs a predicate constant"),
    ],
)
def test_an_arity_count_refuses_a_row_it_cannot_place(
    arity: int, constants: int, message: str
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        ArityCountSpec(arity=arity, constants=constants)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda source: source.with_(digest=""), "digest of its source"),
        (lambda source: source.with_(rows=()), "an estimate needs a class"),
        (lambda source: source.with_(assignment=()), "needs an assigned item"),
        (lambda source: source.with_(exponent=inf), "not finite"),
        (lambda source: source.with_(quantifiers=("",)), "a quantifier needs a name"),
        (lambda source: source.with_(connectives=("",)), "a connective needs a name"),
    ],
)
def test_the_semantic_estimate_refuses_a_joint_it_cannot_hold(
    change: Callable[[SemanticLexiconEstimateSpec], SemanticLexiconEstimateSpec],
    message: str,
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        change(_semantic_source_estimate())


def test_an_estimate_whose_rows_are_not_one_joint_is_refused() -> None:
    declared = _resembling_semantics(
        24, source=_semantic_source_estimate().with_(arities=())
    )

    with pytest.raises(DeclarationError, match="cannot be resembled"):
        declared.structure()
