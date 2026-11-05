from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import pytest

from lofi.deduce import Derivation, Edge, ItemKey
from lofi.judge import EisnerNormalForm, HockenmaierBiskNormalForm, NoNormalForm
from lofi.probe._causal import CombinatorAt, CounterfactualPair, SplitAt, TypeAt
from lofi.probe._provenance import Provenance
from lofi.probe._variables import ARMS, Extraction, extract_variables

if TYPE_CHECKING:
    from lofi.probe._causal import HighLevelVariable
    from lofi.probe._variables import Arm

EISNER = EisnerNormalForm[str]()
NONE = NoNormalForm[str]()
DATASET = "synthetic-dev"


def a_leaf(cell: tuple[int, int], category: str, tag: str) -> Derivation[str, int]:
    return Derivation(
        ItemKey(cell, category, tag),
        Edge(f"lex:{category}", (), category, 1),
    )


def a_step(
    rule: str,
    category: str,
    tag: str,
    left: Derivation[str, int],
    right: Derivation[str, int],
) -> Derivation[str, int]:
    cell = (left.key.span[0], right.key.span[1])
    edge = Edge(
        rule,
        (left.key, right.key),
        category,
        1,
        split=left.key.span[1],
    )

    return Derivation(ItemKey(cell, category, tag), edge, (left, right))


def a_unary(
    rule: str, category: str, tag: str, child: Derivation[str, int]
) -> Derivation[str, int]:
    edge = Edge(rule, (child.key,), category, 1)

    return Derivation(ItemKey(child.key.span, category, tag), edge, (child,))


def a_sentence(tag: str) -> Derivation[str, int]:
    the = a_leaf((0, 1), "NP/N", tag)
    cat = a_leaf((1, 2), "N", tag)
    sleeps = a_leaf((2, 3), "S\\NP", tag)
    subject = a_step("forward-application", "NP", tag, the, cat)

    return a_step("backward-application", "S", tag, subject, sleeps)


def a_composed_sentence() -> Derivation[str, int]:
    she = a_leaf((0, 1), "S/(S\\NP)", "OTHER")
    might = a_leaf((1, 2), "(S\\NP)/(S\\NP)", "OTHER")
    sleep = a_leaf((2, 3), "S\\NP", "OTHER")
    she_might = a_step("forward-composition", "S/(S\\NP)", "FC", she, might)

    return a_step("forward-application", "S", "OTHER", she_might, sleep)


def a_render(term: str) -> str:
    return term


def test_a_normal_form_derivation_yields_a_type_a_split_and_a_combinator() -> None:
    found = extract_variables(
        a_composed_sentence(), EISNER, render=a_render, dataset=DATASET
    )

    assert found.constrained
    assert found.arm == "normal-form"
    assert dict(found.assignment) == {
        TypeAt((0, 1)): "S/(S\\NP)",
        TypeAt((1, 2)): "(S\\NP)/(S\\NP)",
        TypeAt((2, 3)): "S\\NP",
        TypeAt((0, 2)): "S/(S\\NP)",
        TypeAt((0, 3)): "S",
        SplitAt((0, 2)): 1,
        SplitAt((0, 3)): 2,
        CombinatorAt((0, 2), 1): "forward-composition",
        CombinatorAt((0, 3), 2): "forward-application",
    }


def test_the_same_tags_under_no_normal_form_refuse_the_split_and_the_combinator() -> (
    None
):
    constrained = extract_variables(
        a_sentence("OTHER"), EISNER, render=a_render, dataset=DATASET
    )
    unconstrained = extract_variables(
        a_sentence("OTHER"), NONE, render=a_render, dataset=DATASET
    )

    assert constrained.constrained
    assert not unconstrained.constrained
    assert {SplitAt((0, 2)), CombinatorAt((0, 3), 2)} <= set(constrained.assignment)
    assert set(unconstrained.assignment) == {
        TypeAt((0, 1)),
        TypeAt((1, 2)),
        TypeAt((2, 3)),
        TypeAt((0, 2)),
        TypeAt((0, 3)),
    }
    assert {
        variable: decided
        for variable, decided in constrained.assignment.items()
        if variable in unconstrained.assignment
    } == dict(unconstrained.assignment)


def test_an_item_no_normal_form_ran_on_makes_the_derivation_unconstrained() -> None:
    the = a_leaf((0, 1), "NP/N", "OTHER")
    cat = a_leaf((1, 2), "N", "")
    phrase = a_step("forward-application", "NP", "OTHER", the, cat)

    found = extract_variables(phrase, EISNER, render=a_render, dataset=DATASET)

    assert not found.constrained
    assert SplitAt((0, 2)) not in found.assignment
    assert CombinatorAt((0, 2), 1) not in found.assignment


def test_the_ablation_reads_the_split_and_the_combinator_regardless() -> None:
    found = extract_variables(
        a_sentence("OTHER"),
        NONE,
        render=a_render,
        dataset=DATASET,
        arm="unconstrained",
    )

    assert not found.constrained
    assert found.arm == "unconstrained"
    assert found.assignment[SplitAt((0, 3))] == 2
    assert found.assignment[CombinatorAt((0, 3), 2)] == "backward-application"
    assert found.assignment[SplitAt((0, 2))] == 1
    assert found.assignment[CombinatorAt((0, 2), 1)] == "forward-application"


@pytest.mark.parametrize("arm", ARMS)
def test_the_provenance_records_the_arm_and_the_dataset(arm: Arm) -> None:
    found = extract_variables(
        a_sentence("OTHER"), NONE, render=a_render, dataset=DATASET, arm=arm
    )

    assert found.provenance == Provenance(f"variables:{arm}", DATASET, None)
    assert (
        found.provenance
        != extract_variables(
            a_sentence("OTHER"),
            NONE,
            render=a_render,
            dataset=DATASET,
            arm=next(other for other in ARMS if other != arm),
        ).provenance
    )


def test_a_two_tag_alphabet_constrains_whatever_the_filter_is_called() -> None:
    found = extract_variables(
        a_sentence("OTHER"),
        HockenmaierBiskNormalForm[str](),
        render=a_render,
        dataset=DATASET,
    )

    assert found.constrained
    assert found.assignment[SplitAt((0, 3))] == 2


def test_a_tag_outside_the_alphabet_is_refused() -> None:
    the = a_leaf((0, 1), "NP/N", "OTHER")
    cat = a_leaf((1, 2), "N", "TR")
    phrase = a_step("forward-application", "NP", "OTHER", the, cat)

    with pytest.raises(ValueError, match=r"'TR'.*'eisner'"):
        extract_variables(phrase, EISNER, render=a_render, dataset=DATASET)


def test_an_undeclared_arm_is_refused() -> None:
    with pytest.raises(ValueError, match="'primary'"):
        extract_variables(
            a_sentence("OTHER"),
            EISNER,
            render=a_render,
            dataset=DATASET,
            arm=cast("Arm", "primary"),
        )


def test_an_empty_rendering_is_refused() -> None:
    with pytest.raises(ValueError, match="rendered"):
        extract_variables(
            a_sentence("OTHER"), EISNER, render=lambda _: "", dataset=DATASET
        )


def test_a_unary_step_exposes_its_own_type_over_the_split_beneath_it() -> None:
    she = a_leaf((0, 1), "NP", "OTHER")
    raised = a_unary("type-raising", "S/(S\\NP)", "OTHER", she)
    sleeps = a_leaf((1, 2), "S\\NP", "OTHER")
    clause = a_step("forward-application", "S", "OTHER", raised, sleeps)
    whole = a_unary("topicalize", "S'", "OTHER", clause)

    found = extract_variables(whole, EISNER, render=a_render, dataset=DATASET)

    assert found.assignment[TypeAt((0, 1))] == "S/(S\\NP)"
    assert found.assignment[TypeAt((0, 2))] == "S'"
    assert found.assignment[SplitAt((0, 2))] == 1
    assert found.assignment[CombinatorAt((0, 2), 1)] == "forward-application"
    assert len(found.assignment) == 5


def test_the_assignment_is_read_only() -> None:
    found = extract_variables(
        a_sentence("OTHER"), EISNER, render=a_render, dataset=DATASET
    )

    held = cast("dict[HighLevelVariable, str | int]", found.assignment)

    with pytest.raises(TypeError):
        held[TypeAt((0, 1))] = "N"


def test_two_extractions_differing_at_a_split_form_a_counterfactual_pair() -> None:
    the = a_leaf((0, 1), "NP/N", "OTHER")
    cat = a_leaf((1, 2), "N", "OTHER")
    sleeps = a_leaf((2, 3), "S\\NP", "OTHER")
    left_first = a_step(
        "backward-application",
        "S",
        "OTHER",
        a_step("forward-application", "NP", "OTHER", the, cat),
        sleeps,
    )
    right_first = a_step(
        "forward-application",
        "S",
        "OTHER",
        the,
        a_step("backward-application", "N", "OTHER", cat, sleeps),
    )
    base = extract_variables(left_first, EISNER, render=a_render, dataset=DATASET)
    variant = extract_variables(right_first, EISNER, render=a_render, dataset=DATASET)
    shared = set(base.assignment) & set(variant.assignment)
    differing = {
        variable
        for variable in shared
        if base.assignment[variable] != variant.assignment[variable]
    }

    assert shared == {
        TypeAt((0, 1)),
        TypeAt((1, 2)),
        TypeAt((2, 3)),
        TypeAt((0, 3)),
        SplitAt((0, 3)),
    }
    assert differing == {SplitAt((0, 3))}

    pair = CounterfactualPair(
        {variable: base.assignment[variable] for variable in shared},
        {variable: variant.assignment[variable] for variable in shared},
        SplitAt((0, 3)),
        (0, 3),
        base.provenance,
    )

    assert pair.varied == SplitAt((0, 3))


def test_an_extraction_reports_its_arm_its_outcome_and_its_provenance() -> None:
    found = Extraction(
        MappingProxyType({TypeAt((0, 1)): "NP"}),
        "normal-form",
        constrained=False,
        provenance=Provenance("variables:normal-form", DATASET, None),
    )

    assert repr(found) == (
        "Extraction(arm='normal-form', constrained=False, variables=1, "
        f"provenance=Provenance('variables:normal-form', {DATASET!r}, seed=None))"
    )
