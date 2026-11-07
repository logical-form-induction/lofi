"""Tests for registered complexity measures and the bands they are aimed at."""

from copy import copy
from importlib.metadata import entry_points
from typing import Final

import pytest
from tests.unit.data.synthetic.conftest import (
    NEGATED_INVENTORY,
    SCOPED_INVENTORY,
    language,
)

from lofi.data.synthetic import (
    ComplexityContext,
    ComplexityMeter,
    ComplexityReport,
    ComplexityTarget,
    FormComplexity,
    Histogram,
    PairComplexityTarget,
    Target,
)
from lofi.spec.language import (
    ConstantSpec,
    InfixNotationSpec,
    LanguageIso,
    registered_language,
)

QUANTIFIED: Final = "∀x. student(x) => ∃y. book(y) & read(x)(y)"
FLAT: Final = "sleep(john) & snore(john)"


def _measure(text: str, *, quantified: bool = True) -> FormComplexity:
    declared = language("forall", "exists") if quantified else language()
    inventory = SCOPED_INVENTORY if quantified else NEGATED_INVENTORY
    assembly = LanguageIso().forward(
        declared.with_(constants=(*declared.constants, *inventory))
    )
    meter = ComplexityMeter(
        assembly.store,
        assembly.profile,
        abstraction=assembly.calculus.lam,
    )

    return meter.measure(assembly.read(text))


def test_measures_a_quantified_form() -> None:
    """Nested binders, joined subformulas and predicate occurrences are counted."""
    measured = _measure(QUANTIFIED)

    assert measured.value("quantifier_depth") == 2
    assert measured.value("connective_count") == 2
    assert measured.value("predicate_count") == 3
    assert measured.value("variable_count") == 2
    assert measured.predicates == frozenset({"student", "book", "read"})
    assert measured.quantifiers == frozenset({"forall", "exists"})


def test_measures_a_quantifier_free_form() -> None:
    """A form with no binder has zero depth and introduces no variable."""
    measured = _measure(FLAT, quantified=False)

    assert measured.value("quantifier_depth") == 0
    assert measured.value("variable_count") == 0
    assert measured.quantifiers == frozenset()
    assert measured.value("connective_count") == 1
    assert measured.value("predicate_count") == 2


def test_application_and_equality_are_not_connectives() -> None:
    """A predication alone holds no connective, though it is an application."""
    assert _measure("sleep(john)", quantified=False).value("connective_count") == 0


def test_an_open_logical_role_is_counted_without_a_class_table() -> None:
    base = registered_language("propositional")
    language = LanguageIso().forward(
        base.with_(
            constants=(
                *base.constants,
                ConstantSpec(
                    name="xor",
                    type="t -> t -> t",
                    classes=("exclusive-or",),
                    notation=InfixNotationSpec(
                        ascii="xor",
                        unicode="xor",
                        precedence=3,
                        associativity="left",
                    ),
                ),
                ConstantSpec(name="p", type="t", classes=("predicate",)),
                ConstantSpec(name="q", type="t", classes=("predicate",)),
            )
        )
    )
    meter = ComplexityMeter(
        language.store,
        language.profile,
        abstraction=language.calculus.lam,
    )

    measured = meter.measure(language.read("p xor q"))

    assert measured.value("connective_count") == 1
    assert measured.value("predicate_count") == 2


def test_depth_counts_nesting_not_occurrences() -> None:
    """Two quantifiers side by side nest one deep, not two."""
    measured = _measure("(∀x. student(x)) & (∀y. book(y))")

    assert measured.value("quantifier_depth") == 1
    assert measured.value("variable_count") == 2


def test_overlap_is_the_shared_predicate_count() -> None:
    """The overlap of two forms is how many predicate symbols both use."""
    left = _measure("sleep(john) & snore(john)", quantified=False)
    right = _measure("sleep(mary)", quantified=False)

    assert left.overlap(right) == 1
    assert right.overlap(left) == 1
    assert left.overlap(left) == 2


@pytest.mark.parametrize(
    "control",
    [
        "connective_count",
        "meaning_depth",
        "operator_nesting",
        "predicate_count",
        "quantifier_depth",
        "variable_count",
    ],
)
def test_value_reads_each_control(control: str) -> None:
    """Every declared control names a measurement the form carries."""
    measured = _measure(QUANTIFIED)

    assert measured.value(control) == measured.controls[control]


def test_every_shipped_complexity_meter_is_registered() -> None:
    """The installed group is the one source of the available controls."""
    names = {point.name for point in entry_points(group="lofi.lf_complexity")}

    assert names == {
        "connective_count",
        "meaning_depth",
        "operator_nesting",
        "predicate_count",
        "quantifier_depth",
        "variable_count",
    }


def test_a_plugin_context_is_a_frozen_public_value() -> None:
    """Third-party meters receive named language resources they cannot replace."""
    declared = language()
    assembly = LanguageIso().forward(
        declared.with_(constants=(*declared.constants, *NEGATED_INVENTORY))
    )
    context = ComplexityContext(
        assembly.store,
        assembly.profile,
        abstraction=assembly.calculus.lam,
    )

    assert context.application == assembly.profile.application
    assert context.abstraction == assembly.calculus.lam
    assert copy(context) is context


def test_registered_meters_measure_every_declared_control() -> None:
    """One traversal interface reports the installed measurements by name."""
    measured = _measure(QUANTIFIED)

    assert set(measured.controls) == {
        "connective_count",
        "meaning_depth",
        "operator_nesting",
        "predicate_count",
        "quantifier_depth",
        "variable_count",
    }
    assert measured.value("meaning_depth") > 0
    assert measured.value("operator_nesting") == 2


def test_meaning_depth_increases_with_structural_embedding() -> None:
    """Adding an argument raises meaning depth at fixed PI-04 LF factors."""
    shallow = _measure("sleep(john)", quantified=False)
    deep = _measure("love(john)(mary)", quantified=False)

    assert shallow.value("meaning_depth") < deep.value("meaning_depth")
    for fixed in (
        "connective_count",
        "operator_nesting",
        "predicate_count",
        "quantifier_depth",
        "variable_count",
    ):
        assert shallow.value(fixed) == deep.value(fixed)


def test_operator_nesting_uses_the_language_roles_not_constant_spellings() -> None:
    """A renamed connective contributes because its declared role says it does."""
    base = registered_language("propositional")
    declared = base.with_(
        constants=(
            *base.constants,
            ConstantSpec(
                name="xor",
                type="t -> t -> t",
                classes=("exclusive-or",),
                notation=InfixNotationSpec(
                    ascii="xor",
                    unicode="xor",
                    precedence=3,
                    associativity="left",
                ),
            ),
            ConstantSpec(name="p", type="t", classes=("predicate",)),
            ConstantSpec(name="q", type="t", classes=("predicate",)),
            ConstantSpec(name="r", type="t", classes=("predicate",)),
            ConstantSpec(name="s", type="t", classes=("predicate",)),
            ConstantSpec(name="pair", type="t -> t -> t", classes=("predicate",)),
            ConstantSpec(name="wrap", type="t -> t", classes=("predicate",)),
        )
    )
    assembly = LanguageIso().forward(declared)
    meter = ComplexityMeter(
        assembly.store,
        assembly.profile,
        abstraction=assembly.calculus.lam,
    )

    flat = meter.measure(assembly.read("pair(p xor q)(wrap(wrap(wrap(r))) xor s)"))
    nested = meter.measure(assembly.read("pair((p xor q) xor r)(wrap(wrap(wrap(s))))"))

    assert flat.value("operator_nesting") == 1
    assert nested.value("operator_nesting") == 2
    for fixed in (
        "connective_count",
        "meaning_depth",
        "predicate_count",
        "quantifier_depth",
        "variable_count",
    ):
        assert flat.value(fixed) == nested.value(fixed)


def test_value_refuses_an_unknown_control() -> None:
    """A name that is not a control is refused rather than answered."""
    with pytest.raises(KeyError):
        _measure(QUANTIFIED).value("predicate_depth")


def test_a_negative_measurement_is_refused() -> None:
    """A complexity cannot be built from a negative count."""
    with pytest.raises(ValueError, match="quantifier depth"):
        FormComplexity(
            {
                "quantifier_depth": -1,
                "connective_count": 0,
                "predicate_count": 0,
                "variable_count": 0,
                "meaning_depth": 0,
                "operator_nesting": 0,
            }
        )


def test_a_target_admits_only_within_its_band() -> None:
    """A controlled dimension admits a measurement inside its tolerance alone."""
    target = ComplexityTarget({"connective_count": Target(2, 0)})

    assert target.admits(_measure(QUANTIFIED))
    assert not target.admits(_measure(FLAT, quantified=False))


def test_an_uncontrolled_dimension_admits_anything() -> None:
    """A target that sets no band admits every measurement."""
    assert ComplexityTarget({}).admits(_measure(QUANTIFIED))
    assert ComplexityTarget({}).band("variable_count") is None


def test_a_pair_target_checks_both_sides_and_the_overlap() -> None:
    """A candidate is admitted only when both forms and their overlap are."""
    left = _measure("sleep(john) & snore(john)", quantified=False)
    right = _measure("sleep(mary)", quantified=False)
    target = PairComplexityTarget(
        premise=ComplexityTarget({"connective_count": Target(1, 0)}),
        hypothesis=ComplexityTarget({"connective_count": Target(0, 0)}),
        overlap=Target(1, 0),
    )

    assert target.admits(left, right)
    assert not target.admits(right, left)


def test_a_pair_target_refuses_an_overlap_outside_its_band() -> None:
    """A pair inside both sides' bands is still refused on its overlap."""
    left = _measure("sleep(john) & snore(john)", quantified=False)
    target = PairComplexityTarget(overlap=Target(1, 0))

    assert not target.admits(left, left)


def _report(
    values: tuple[int, ...], target: PairComplexityTarget | None
) -> ComplexityReport:
    controls = (
        "connective_count",
        "meaning_depth",
        "operator_nesting",
        "predicate_count",
        "quantifier_depth",
        "variable_count",
    )
    return ComplexityReport(
        {control: Histogram.of(values) for control in controls},
        {control: Histogram.of(values) for control in controls},
        overlap=Histogram.of(values),
        rejected=0,
        target=target,
    )


def test_a_report_without_a_target_is_satisfied() -> None:
    """A draw that aimed at nothing left nothing unmet."""
    assert _report((1, 2, 3), None).satisfied()


def test_a_report_is_satisfied_when_every_bucket_lies_in_its_band() -> None:
    """Every achieved value of a controlled dimension is within tolerance."""
    target = PairComplexityTarget(
        premise=ComplexityTarget({"connective_count": Target(2, 1)}),
        overlap=Target(2, 1),
    )

    assert _report((1, 2, 3), target).satisfied()


def test_a_report_is_unsatisfied_when_a_bucket_escapes_its_band() -> None:
    """One achieved value outside the band leaves the report unsatisfied."""
    target = PairComplexityTarget(
        premise=ComplexityTarget({"connective_count": Target(2, 0)})
    )

    assert not _report((1, 2, 3), target).satisfied()


def test_a_report_checks_the_overlap_band() -> None:
    """An overlap outside its band leaves the report unsatisfied on its own."""
    target = PairComplexityTarget(overlap=Target(2, 0))

    assert not _report((1, 2, 3), target).satisfied()


def test_a_report_needs_every_control() -> None:
    """The premise and hypothesis must report the same installed controls."""
    partial = {"quantifier_depth": Histogram.of(())}

    with pytest.raises(ValueError, match="do not cover every control"):
        ComplexityReport(
            partial,
            {
                "quantifier_depth": Histogram.of(()),
                "meaning_depth": Histogram.of(()),
            },
            overlap=Histogram.of(()),
            rejected=0,
            target=None,
        )
