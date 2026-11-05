import json
import math
from functools import partial
from random import Random
from typing import TYPE_CHECKING, cast

import didactic.api as dx
import pytest
from tests.unit.spec.grammar.test__signature import (
    _application,
    _ccg,
    _ccg_runtime,
    _compiled,
    _explicit_runtime,
    _primitives,
    slash_views,
    slashes,
)

from lofi._stats import spearman
from lofi.spec import DERIVED_SEED, InventoryExhaustedError, UnresolvedSeedError
from lofi.spec.grammar import (
    AmbiguityBucketSpec,
    CategoryComplexityBoundSpec,
    CategoryMeasure,
    ClassSpec,
    FeatureErasureMatchingSpec,
    GeneratedLexiconSpec,
    GrammarDecl,
    HistogramAmbiguityLawSpec,
    InlineLexiconSpec,
    JointSpec,
    LexicalEntrySpec,
    LexiconStructure,
    ParametricAmbiguityLawSpec,
    ParametricJointSpec,
    PowerLawRankLawSpec,
    RootGoalSpec,
    SignatureSpec,
    TabularJointSpec,
    UniformRankLawSpec,
    category_depth,
    generate_inventory,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Protocol

    from lofi.judge import RuleSet
    from lofi.lexicon import StaticLexicon
    from lofi.terms import CategoryResolution, TargetSet, Term, TermStore, TermSyntax

    class _LexiconDeclaration(Protocol):
        """A declaration that builds a static lexicon."""

        def build(
            self,
            store: TermStore,
            syntax: TermSyntax,
            resolve_category: CategoryResolution,
            rules: RuleSet[Term],
            goal: TargetSet[Term],
            reserved: frozenset[Term],
            category_measure: CategoryMeasure,
            /,
        ) -> StaticLexicon[Term]:
            """Build the declared lexicon."""
            ...


S_TEXT = '("#const[Cat]" "S")'
NP_TEXT = '("#const[Cat]" "NP")'
VP_TEXT = '("#const[Cat]" "VP")'
MEASURE = CategoryMeasure((("/", 0), ("\\", 0)))


def _lexicon() -> InlineLexiconSpec:
    return InlineLexiconSpec(
        entries=(
            LexicalEntrySpec(form="john", category=NP_TEXT),
            LexicalEntrySpec(form="sleeps", category=VP_TEXT),
        )
    )


def _parametric() -> ParametricJointSpec:
    return ParametricJointSpec(
        categories=(NP_TEXT, VP_TEXT, S_TEXT),
        category_shares=PowerLawRankLawSpec(exponent=1.0),
    )


def _tabular() -> TabularJointSpec:
    return TabularJointSpec(
        rows=(
            ClassSpec(category=NP_TEXT, share=2.0),
            ClassSpec(category=VP_TEXT, share=1.0),
            ClassSpec(category=S_TEXT, share=1.0),
        )
    )


def _structure(joint: JointSpec | None = None) -> LexiconStructure:
    return LexiconStructure(
        forms=6,
        joint=_parametric() if joint is None else joint,
        ambiguity=ParametricAmbiguityLawSpec(
            mean=2.0, dispersion=0.0, maximum=3, placement=0.0
        ),
        rank_law=UniformRankLawSpec(),
        seed=11,
    )


def _generated(**changes: float) -> GeneratedLexiconSpec:
    return GeneratedLexiconSpec(
        forms=40,
        categories=10,
        category_shares=PowerLawRankLawSpec(exponent=1.0),
        coupling=0.0,
        ambiguity=ParametricAmbiguityLawSpec(
            mean=2.0, dispersion=0.5, maximum=3, placement=0.0
        ),
        rank_law=UniformRankLawSpec(),
        seed=11,
        max_depth=2,
    ).with_(**changes)


def _fitted_coupling(
    lexicon: StaticLexicon[Term], store: TermStore, syntax: TermSyntax
) -> float | None:
    """Return the coupling fitted to a built lexicon's achieved joint."""
    classes = _classes(lexicon, store, syntax)
    sizes = tuple(float(len(entries)) for entries in classes.values())
    per_form = tuple(
        sum(mass for _, mass in entries) / len(entries) for entries in classes.values()
    )
    x = tuple(math.log(value) for value in sizes)
    y = tuple(math.log(value) for value in per_form)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)

    if denominator == 0.0:
        return None

    slope = (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x, y, strict=True)
        )
        / denominator
    )

    return -slope


def _classes(
    lexicon: StaticLexicon[Term],
    store: TermStore,
    syntax: TermSyntax,
) -> dict[str, list[tuple[str, float]]]:
    """Return each category's forms with their masses, by rendered category."""
    by_category: dict[str, list[tuple[str, float]]] = {}

    for form in lexicon.forms:
        for entry in lexicon.entries(form):
            mass = math.exp(lexicon.log_mass(form) + entry.weight)
            by_category.setdefault(syntax.render(store, entry.category), []).append(
                (form, mass)
            )

    return by_category


def _ambiguities(lexicon: StaticLexicon[Term]) -> list[int]:
    """Return the number of categories borne by every form."""
    return [len(lexicon.entries(form)) for form in lexicon.forms]


def _correlation(lexicon: StaticLexicon[Term]) -> float | None:
    """Return the ambiguity-frequency rank correlation of a built lexicon."""
    return spearman(
        [float(value) for value in _ambiguities(lexicon)],
        [lexicon.log_mass(form) for form in lexicon.forms],
    )


def test_ambiguity_mean_is_independent_of_the_form_inventory() -> None:
    store, syntax, rules, goal = _explicit_runtime()
    law = ParametricAmbiguityLawSpec(mean=2.0, dispersion=0.5, maximum=3, placement=0.0)

    means = []

    for forms in (60, 240):
        structure = LexiconStructure(
            forms=forms,
            joint=ParametricJointSpec(
                categories=(NP_TEXT, VP_TEXT, S_TEXT),
                category_shares=PowerLawRankLawSpec(exponent=1.0),
            ),
            ambiguity=law,
            rank_law=PowerLawRankLawSpec(exponent=1.0),
            seed=11,
        )
        lexicon = structure.build(
            store,
            syntax,
            partial(syntax.parse, store),
            rules,
            goal,
            frozenset(),
            MEASURE,
        )

        assert len(lexicon.forms) == forms
        means.append(sum(_ambiguities(lexicon)) / forms)

    assert means == pytest.approx([2.0, 2.0])


def test_ambiguity_dispersion_moves_variance_at_a_fixed_mean() -> None:
    store, syntax, rules, goal = _explicit_runtime()
    variances = []

    for dispersion in (0.0, 0.5, 1.0):
        law = ParametricAmbiguityLawSpec(
            mean=2.0, dispersion=dispersion, maximum=3, placement=0.0
        )
        lexicon = LexiconStructure(
            forms=240,
            joint=ParametricJointSpec(
                categories=(NP_TEXT, VP_TEXT, S_TEXT),
                category_shares=UniformRankLawSpec(),
            ),
            ambiguity=law,
            rank_law=PowerLawRankLawSpec(exponent=1.0),
            seed=11,
        ).build(
            store,
            syntax,
            partial(syntax.parse, store),
            rules,
            goal,
            frozenset(),
            MEASURE,
        )
        values = _ambiguities(lexicon)
        mean = sum(values) / len(values)

        assert mean == pytest.approx(2.0)
        variances.append(sum((value - mean) ** 2 for value in values) / len(values))

    assert variances[0] < variances[1] < variances[2]


def test_ambiguity_placement_moves_achieved_correlation_monotonically() -> None:
    store, syntax, rules, goal = _explicit_runtime()
    achieved = []

    for placement in (-0.5, 0.0, 0.5):
        lexicon = LexiconStructure(
            forms=240,
            joint=ParametricJointSpec(
                categories=(NP_TEXT, VP_TEXT, S_TEXT),
                category_shares=UniformRankLawSpec(),
            ),
            ambiguity=ParametricAmbiguityLawSpec(
                mean=2.0, dispersion=1.0, maximum=3, placement=placement
            ),
            rank_law=PowerLawRankLawSpec(exponent=1.0),
            seed=11,
        ).build(
            store,
            syntax,
            partial(syntax.parse, store),
            rules,
            goal,
            frozenset(),
            MEASURE,
        )
        correlation = _correlation(lexicon)

        assert correlation is not None
        achieved.append(correlation)

    assert achieved[0] < achieved[1] < achieved[2]


def test_mean_ambiguity_leaves_size_and_the_category_marginal_fixed() -> None:
    store, syntax, rules, goal = _explicit_runtime()
    expected = {NP_TEXT: 0.5, VP_TEXT: 0.25, S_TEXT: 0.25}

    for mean in (1.5, 2.0, 2.5):
        structure = _structure(_tabular()).with_(
            forms=240,
            ambiguity=ParametricAmbiguityLawSpec(
                mean=mean, dispersion=0.5, maximum=3, placement=0.0
            ),
            rank_law=PowerLawRankLawSpec(exponent=1.0),
        )
        lexicon = structure.build(
            store,
            syntax,
            partial(syntax.parse, store),
            rules,
            goal,
            frozenset(),
            MEASURE,
        )
        marginal = {
            category: sum(mass for _, mass in entries)
            for category, entries in _classes(lexicon, store, syntax).items()
        }

        assert len(lexicon.forms) == 240
        assert marginal == pytest.approx(expected)
        assert sum(_ambiguities(lexicon)) / len(lexicon.forms) == pytest.approx(mean)


@pytest.mark.parametrize(
    ("mean", "dispersion", "forms"),
    [(1.1, 0.5, 40), (1.1, 0.75, 13), (1.2, 0.25, 97), (2.7, 0.5, 11)],
)
def test_a_parametric_law_realizes_its_mean_at_any_inventory_size(
    mean: float, dispersion: float, forms: int
) -> None:
    law = ParametricAmbiguityLawSpec(
        mean=mean, dispersion=dispersion, maximum=4, placement=0.0
    )
    counts = law.counts(forms)

    assert len(counts) == forms
    assert sum(counts) == round(forms * mean)
    assert set(counts) <= {1, 2, 3, 4}


def test_a_parametric_law_spans_its_support_from_mild_to_heavy() -> None:
    mild = ParametricAmbiguityLawSpec(
        mean=2.5, dispersion=0.0, maximum=4, placement=0.0
    ).counts(40)
    heavy = ParametricAmbiguityLawSpec(
        mean=2.5, dispersion=1.0, maximum=4, placement=0.0
    ).counts(40)

    assert set(mild) == {2, 3}
    assert set(heavy) == {1, 4}
    assert sum(mild) == sum(heavy) == 100


def test_a_histogram_ambiguity_law_realizes_its_declared_buckets() -> None:
    law = HistogramAmbiguityLawSpec(
        maximum=3,
        placement=0.0,
        buckets=(
            AmbiguityBucketSpec(categories=1, share=3.0),
            AmbiguityBucketSpec(categories=3, share=1.0),
        ),
    )

    assert law.counts(40).count(1) == 30
    assert law.counts(40).count(3) == 10

    with pytest.raises(dx.ValidationError, match="repeats"):
        law.with_(buckets=(law.buckets[0], law.buckets[0]))

    with pytest.raises(dx.ValidationError, match="exceeds"):
        law.with_(maximum=2)


@pytest.mark.parametrize(
    "law",
    [
        lambda: ParametricAmbiguityLawSpec(
            mean=0.5, dispersion=0.0, maximum=3, placement=0.0
        ),
        lambda: ParametricAmbiguityLawSpec(
            mean=4.0, dispersion=0.0, maximum=3, placement=0.0
        ),
        lambda: ParametricAmbiguityLawSpec(
            mean=2.0, dispersion=1.1, maximum=3, placement=0.0
        ),
        lambda: ParametricAmbiguityLawSpec(
            mean=2.0, dispersion=0.0, maximum=3, placement=-1.1
        ),
    ],
)
def test_an_ambiguity_law_refuses_an_invalid_shape(law: Callable[[], object]) -> None:
    with pytest.raises(dx.ValidationError):
        law()


def test_an_inline_lexicon_interns_each_category_once_and_normalizes_weights() -> None:
    store, syntax, rules, goal = _explicit_runtime()
    lexicon = InlineLexiconSpec(
        entries=(
            LexicalEntrySpec(form="john", category=NP_TEXT, weight=0.0),
            LexicalEntrySpec(form="john", category=S_TEXT, weight=-1.0),
            LexicalEntrySpec(form="sleeps", category=VP_TEXT, weight=-2.0),
        )
    ).build(
        store,
        syntax,
        partial(syntax.parse, store),
        rules,
        goal,
        frozenset(),
        MEASURE,
    )

    assert lexicon.forms == ("john", "sleeps")
    assert len(lexicon) == 3

    (only,) = lexicon.entries("sleeps")

    assert only.category is syntax.parse(store, VP_TEXT)
    assert only.weight == pytest.approx(0.0)
    assert lexicon.log_probability("john", syntax.parse(store, NP_TEXT)) == (
        pytest.approx(-0.3132616875182228)
    )
    assert lexicon.log_probability("john", syntax.parse(store, S_TEXT)) == (
        pytest.approx(-1.3132616875182228)
    )

    with pytest.raises(dx.ValidationError):
        InlineLexiconSpec(entries=())

    with pytest.raises(dx.ValidationError):
        LexicalEntrySpec(form="", category=NP_TEXT)

    with pytest.raises(dx.ValidationError, match="not finite"):
        LexicalEntrySpec(form="john", category=NP_TEXT, weight=math.nan)

    with pytest.raises(dx.ValidationError, match="not finite"):
        LexicalEntrySpec(form="john", category=NP_TEXT, weight=-math.inf)


def test_an_inline_lexical_entry_declares_its_meaning_in_place() -> None:
    entry = LexicalEntrySpec(
        form="john",
        category=NP_TEXT,
        meaning="john",
    )

    assert entry.meaning == "john"

    with pytest.raises(dx.ValidationError, match="logical form"):
        LexicalEntrySpec(form="john", category=NP_TEXT, meaning="")


def test_a_rank_law_weighs_its_ranks_to_mean_one() -> None:
    assert UniformRankLawSpec().weights(4) == (1.0, 1.0, 1.0, 1.0)

    zipf = PowerLawRankLawSpec(exponent=1.0).weights(4)

    assert sum(zipf) == pytest.approx(4.0)
    assert zipf[0] / zipf[1] == pytest.approx(2.0)
    assert zipf[0] / zipf[3] == pytest.approx(4.0)
    assert PowerLawRankLawSpec(exponent=0.0).weights(3) == (1.0, 1.0, 1.0)

    rising = PowerLawRankLawSpec(exponent=-1.0).weights(3)

    assert rising[2] / rising[0] == pytest.approx(3.0)

    with pytest.raises(dx.ValidationError, match="not finite"):
        PowerLawRankLawSpec(exponent=math.nan)


def test_a_parametric_joint_assigns_category_shares_by_rank() -> None:
    marginal = _parametric().marginal()
    shares = [share for _, share in marginal]

    assert [category for category, _ in marginal] == [NP_TEXT, VP_TEXT, S_TEXT]
    assert sum(shares) == pytest.approx(1.0)
    assert shares[0] / shares[1] == pytest.approx(2.0)
    assert shares[0] / shares[2] == pytest.approx(3.0)

    uniform = ParametricJointSpec(
        categories=(NP_TEXT, VP_TEXT), category_shares=UniformRankLawSpec()
    )

    assert uniform.marginal() == ((NP_TEXT, 0.5), (VP_TEXT, 0.5))

    with pytest.raises(dx.ValidationError, match="twice"):
        _parametric().with_(categories=(NP_TEXT, NP_TEXT))

    with pytest.raises(dx.ValidationError):
        _parametric().with_(categories=())


def test_a_tabular_joint_scales_its_shares_to_one() -> None:
    assert _tabular().marginal() == (
        (NP_TEXT, 0.5),
        (VP_TEXT, 0.25),
        (S_TEXT, 0.25),
    )

    with pytest.raises(dx.ValidationError, match="twice"):
        _tabular().with_(rows=(_tabular().rows[0], _tabular().rows[0]))

    with pytest.raises(dx.ValidationError):
        TabularJointSpec(rows=())

    with pytest.raises(dx.ValidationError, match="not positive"):
        ClassSpec(category=NP_TEXT, share=0.0)

    with pytest.raises(dx.ValidationError, match="not finite"):
        ClassSpec(category=NP_TEXT, share=math.nan)


def test_a_lexicon_structure_builds_its_independent_marginals() -> None:
    store, syntax, rules, goal = _explicit_runtime()
    structure = _structure(_tabular()).with_(rank_law=PowerLawRankLawSpec(exponent=1.0))
    lexicon = structure.build(
        store,
        syntax,
        partial(syntax.parse, store),
        rules,
        goal,
        frozenset(),
        MEASURE,
    )
    by_category: dict[str, list[tuple[str, float]]] = {}

    for form in lexicon.forms:
        assert form.startswith("w")
        assert 0 <= int(form[1:]) < structure.forms

        for entry in lexicon.entries(form):
            mass = math.exp(lexicon.log_mass(form) + entry.weight)
            by_category.setdefault(syntax.render(store, entry.category), []).append(
                (form, mass)
            )

    assert {
        category: sum(mass for _, mass in held)
        for category, held in by_category.items()
    } == pytest.approx({NP_TEXT: 0.5, VP_TEXT: 0.25, S_TEXT: 0.25})
    assert _ambiguities(lexicon) == [2] * structure.forms
    assert (
        structure.build(
            store,
            syntax,
            partial(syntax.parse, store),
            rules,
            goal,
            frozenset(),
            MEASURE,
        ).forms
        == lexicon.forms
    )
    other = structure.with_(seed=12).build(
        store,
        syntax,
        partial(syntax.parse, store),
        rules,
        goal,
        frozenset(),
        MEASURE,
    )
    assert _classes(other, store, syntax) != _classes(lexicon, store, syntax)


def test_a_lexicon_structure_refuses_unrealizable_ambiguity() -> None:
    with pytest.raises(dx.ValidationError, match="exceeds the 3 declared categories"):
        _structure().with_(
            ambiguity=ParametricAmbiguityLawSpec(
                mean=2.0, dispersion=0.0, maximum=4, placement=0.0
            )
        )

    with pytest.raises(dx.ValidationError):
        _structure().with_(forms=0)

    with pytest.raises(dx.ValidationError):
        LexiconStructure.model_validate({"kind": "structured", "forms": 6, "seed": 1})


def test_a_generated_lexicon_draws_its_categories_over_its_forms() -> None:
    store, syntax, rules, goal = _ccg_runtime(4)
    declared = _generated()
    lexicon = declared.build(
        store,
        syntax,
        partial(syntax.parse, store),
        rules,
        goal,
        frozenset(),
        MEASURE,
    )
    classes = _classes(lexicon, store, syntax)
    inventory = generate_inventory(
        rules,
        goal.categories,
        tuple(syntax.parse(store, name) for name in _primitives(4)),
        category_measure=MEASURE,
        categories=10,
        max_depth=2,
        max_arguments=None,
        max_order=None,
        derivation_depth=4,
        expansion=0.5,
        patience=1_000,
        random=Random(11),
    )

    assert set(classes) == {
        syntax.render(store, category) for category in inventory.categories
    }
    assert sum(_ambiguities(lexicon)) / len(lexicon.forms) == pytest.approx(2.0)
    assert sum(mass for held in classes.values() for _, mass in held) == (
        pytest.approx(1.0)
    )
    assert all(
        category_depth(syntax.parse(store, category)) <= 2 for category in classes
    )
    assert all(
        form.startswith("w") and 0 <= int(form[1:]) < 40 for form in lexicon.forms
    )


def test_generated_coupling_moves_the_achieved_joint_at_fixed_marginals() -> None:
    store, syntax, rules, goal = _ccg_runtime(4)
    built = []
    declarations = tuple(
        _generated(coupling=coupling) for coupling in (-0.75, 0.0, 0.75)
    )

    for declaration in declarations:
        lexicon = declaration.build(
            store,
            syntax,
            partial(syntax.parse, store),
            rules,
            goal,
            frozenset(),
            MEASURE,
        )
        achieved = _fitted_coupling(lexicon, store, syntax)

        assert achieved is not None
        built.append((lexicon, achieved))

    assert [achieved for _, achieved in built] == sorted(
        achieved for _, achieved in built
    )
    expected_masses = tuple(built[0][0].log_mass(form) for form in built[0][0].forms)
    expected_shares = {
        category: sum(mass for _, mass in entries)
        for category, entries in _classes(built[0][0], store, syntax).items()
    }

    for lexicon, _ in built:
        assert tuple(lexicon.log_mass(form) for form in lexicon.forms) == (
            pytest.approx(expected_masses)
        )
        assert {
            category: sum(mass for _, mass in entries)
            for category, entries in _classes(lexicon, store, syntax).items()
        } == pytest.approx(expected_shares)
    assert [_ambiguities(lexicon) for lexicon, _ in built] == [
        _ambiguities(built[0][0])
    ] * len(built)
    documents = tuple(
        json.loads(declaration.model_dump_json()) for declaration in declarations
    )

    assert {key for key in documents[0] if documents[0][key] != documents[1][key]} == {
        "coupling"
    }
    assert {key for key in documents[1] if documents[1][key] != documents[2][key]} == {
        "coupling"
    }


def test_generated_coupling_is_required_and_finite() -> None:
    document = json.loads(_generated().model_dump_json())
    del document["coupling"]

    with pytest.raises(dx.ValidationError):
        GeneratedLexiconSpec.model_validate(document)

    with pytest.raises(dx.ValidationError, match="not finite"):
        _generated(coupling=math.nan)


def test_generated_coupling_projects_an_indivisible_finite_marginal() -> None:
    store, syntax, rules, goal = _ccg_runtime(4)
    lexicon = (
        _generated(coupling=0.0)
        .with_(
            forms=8,
            categories=6,
            category_shares=UniformRankLawSpec(),
            ambiguity=ParametricAmbiguityLawSpec(
                mean=1.0, dispersion=0.0, maximum=1, placement=0.0
            ),
        )
        .build(
            store,
            syntax,
            partial(syntax.parse, store),
            rules,
            goal,
            frozenset(),
            MEASURE,
        )
    )
    classes = _classes(lexicon, store, syntax)

    assert sorted(len(entries) for entries in classes.values()) == [1, 1, 1, 1, 2, 2]
    assert tuple(lexicon.log_mass(form) for form in lexicon.forms) == pytest.approx(
        (math.log(1.0 / 8.0),) * 8
    )


def test_projected_coupling_keeps_every_declared_support_mass_positive() -> None:
    store, syntax, rules, goal = _ccg_runtime(4)
    lexicon = (
        _generated(coupling=0.0)
        .with_(
            forms=16,
            categories=6,
            category_shares=UniformRankLawSpec(),
            ambiguity=ParametricAmbiguityLawSpec(
                mean=1.5, dispersion=0.0, maximum=4, placement=0.0
            ),
            rank_law=PowerLawRankLawSpec(exponent=1.0),
            seed=1,
        )
        .build(
            store,
            syntax,
            partial(syntax.parse, store),
            rules,
            goal,
            frozenset(),
            MEASURE,
        )
    )

    assert sum(_ambiguities(lexicon)) == 24
    assert all(
        math.isfinite(lexicon.log_mass(form) + entry.weight)
        for form in lexicon.forms
        for entry in lexicon.entries(form)
    )


def test_a_generated_lexicon_obeys_declared_argument_and_order_bounds() -> None:
    """Argument count and category order are independent sweepable leaves."""
    store, syntax, rules, goal = _ccg_runtime(8)
    declared = _generated(categories=8, max_depth=3).with_(
        max_arguments=CategoryComplexityBoundSpec(value=1),
        max_order=CategoryComplexityBoundSpec(value=1),
    )
    lexicon = declared.build(
        store,
        syntax,
        partial(syntax.parse, store),
        rules,
        goal,
        frozenset(),
        MEASURE,
    )
    categories = {
        entry.category for form in lexicon.forms for entry in lexicon.entries(form)
    }

    assert categories
    assert max(map(MEASURE.arity, categories)) <= 1
    assert max(map(MEASURE.order, categories)) <= 1


@pytest.mark.parametrize("field", ["max_arguments", "max_order"])
def test_a_generated_lexicon_refuses_a_negative_complexity_bound(field: str) -> None:
    with pytest.raises(dx.ValidationError, match="nonnegative"):
        _generated().with_(**{field: CategoryComplexityBoundSpec(value=-1)})


def test_a_generated_lexicon_is_deterministic_in_its_seed() -> None:
    store, syntax, rules, goal = _ccg_runtime(4)

    def drawn(seed: int) -> list[tuple[str, str, float]]:
        lexicon = _generated(seed=seed).build(
            store,
            syntax,
            partial(syntax.parse, store),
            rules,
            goal,
            frozenset(),
            MEASURE,
        )

        return [
            (form, syntax.render(store, entry.category), entry.weight)
            for form in lexicon.forms
            for entry in lexicon.entries(form)
        ]

    assert drawn(11) == drawn(11)
    assert drawn(11) != drawn(12)


@pytest.mark.parametrize("primitives", [2, 4, 8, 16])
def test_a_generated_lexicon_builds_over_any_number_of_primitives(
    primitives: int,
) -> None:
    declaration = GrammarDecl(
        name="generated",
        signature=SignatureSpec(
            constructors=slashes(),
            functors=slash_views(),
            primitives=_primitives(primitives),
        ),
        rules=_application(),
        lexicon=_generated(),
        matching=FeatureErasureMatchingSpec(),
    )
    payload = json.loads(declaration.model_dump_json())

    assert payload["signature"]["conjunctions"] is None
    assert GrammarDecl.model_validate(payload) == declaration

    store, syntax, rules, goal = _ccg_runtime(primitives)
    lexicon = cast("_LexiconDeclaration", declaration.lexicon).build(
        store,
        syntax,
        partial(syntax.parse, store),
        rules,
        goal,
        frozenset(),
        MEASURE,
    )
    categories = {
        entry.category for form in lexicon.forms for entry in lexicon.entries(form)
    }
    atoms: set[str] = set()
    pending = list(categories)

    while pending:
        term = pending.pop()

        if not term.args:
            atoms.add(syntax.render(store, term))

        pending.extend(term.args)

    assert len(categories) == 10
    assert atoms <= set(_primitives(primitives))


def test_a_generated_lexicon_refuses_bounds_its_rules_cannot_fill() -> None:
    store, syntax, rules, goal = _ccg_runtime(2)

    with pytest.raises(InventoryExhaustedError, match="reached 1 of 3"):
        _generated(categories=3, max_depth=0, patience=50).build(
            store,
            syntax,
            partial(syntax.parse, store),
            rules,
            goal,
            frozenset(),
            MEASURE,
        )

    with pytest.raises(InventoryExhaustedError, match="reached 6 of 7"):
        _generated(categories=7, max_depth=1, derivation_depth=1, patience=50).build(
            store,
            syntax,
            partial(syntax.parse, store),
            rules,
            goal,
            frozenset(),
            MEASURE,
        )


def _atoms(store: TermStore, categories: Iterable[Term]) -> set[str]:
    """Return the names of the primitives the categories are built from."""
    atoms: set[str] = set()
    pending = list(categories)

    while pending:
        term = pending.pop()

        if not term.args:
            atoms.add(store.payload_name(term))

        pending.extend(term.args)

    return atoms


def test_a_generated_lexicon_never_draws_a_conjunction_marker_as_a_premise() -> None:
    adopted = _compiled(_ccg(), _application())
    context = adopted.context
    store = context.store
    syntax = context.syntax
    rules = adopted.rule_set
    goal = RootGoalSpec().build(partial(syntax.parse, store), rules.matching)
    conjunctions = context.conjunctions
    assert conjunctions is not None
    reserved = frozenset(conjunctions.markers)

    assert reserved == frozenset({syntax.parse(store, "conj")})

    def drawn(excluded: frozenset[Term]) -> set[str]:
        lexicon = _generated(seed=3).build(
            store,
            syntax,
            partial(syntax.parse, store),
            rules,
            goal,
            excluded,
            MEASURE,
        )

        return _atoms(
            store,
            (
                entry.category
                for form in lexicon.forms
                for entry in lexicon.entries(form)
            ),
        )

    assert "conj" in drawn(frozenset())
    assert drawn(reserved) <= {"S", "NP"}


@pytest.mark.parametrize(
    "change",
    [
        {"forms": 0},
        {"categories": 0},
        {"max_depth": -1},
        {"derivation_depth": 0},
        {"expansion": 1.5},
        {"expansion": -0.1},
        {"expansion": math.nan},
        {"patience": 0},
    ],
)
def test_a_generated_lexicon_refuses_a_malformed_bound(
    change: dict[str, int | float],
) -> None:
    with pytest.raises(dx.ValidationError):
        _generated(**change)


def test_a_seeded_lexicon_takes_the_seed_derived_for_its_path() -> None:
    asked: list[str] = []

    def derive(path: str) -> int:
        asked.append(path)

        return 11

    for declared in (_structure(), _generated()):
        derived = declared.with_(seed=DERIVED_SEED)

        assert derived.seed == "derived"
        assert (
            derived.with_derived_seed(
                derive, "data.datasets.train.dataset.grammar.lexicon"
            )
            == declared
        )
        assert declared.with_derived_seed(derive, "elsewhere") == declared

    assert asked == ["data.datasets.train.dataset.grammar.lexicon.seed"] * 2


def test_an_unseeded_lexicon_is_kept_whatever_a_point_derives() -> None:
    inline = InlineLexiconSpec(
        entries=(LexicalEntrySpec(form="john", category=NP_TEXT),)
    )

    assert (
        inline.with_derived_seed(
            lambda _: 3, "data.datasets.train.dataset.grammar.lexicon"
        )
        == inline
    )


def test_a_lexicon_drawn_under_an_unresolved_derived_seed_is_refused() -> None:
    store, syntax, rules, goal = _explicit_runtime()

    with pytest.raises(UnresolvedSeedError, match="lexicon's seed is derived"):
        _structure().with_(seed=DERIVED_SEED).build(
            store,
            syntax,
            partial(syntax.parse, store),
            rules,
            goal,
            frozenset(),
            MEASURE,
        )


def test_a_lexicon_seed_is_an_integer_or_derived() -> None:
    with pytest.raises(dx.ValidationError, match="'drawn'"):
        _structure().with_(seed="drawn")

    with pytest.raises(dx.ValidationError, match="negative"):
        _generated().with_(seed=-1)
