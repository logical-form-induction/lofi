from math import inf, log
from typing import TYPE_CHECKING

from lofi.deduce import Derivation, Edge, ItemKey
from lofi.interpret._derive import BestOfRanked, LogProbability
from lofi.interpret._ports import DerivationReducer, DerivationStyle, Ordering
from lofi.interpret._readings import HeadDependencies, HeadLink
from lofi.interpret._render import BraceString, ProofFigure

if TYPE_CHECKING:
    from lofi.deduce import Ranked


def key(span: tuple[int, int], category: str) -> ItemKey[str]:
    return ItemKey(span, category)


def leaf(span: tuple[int, int], category: str) -> Derivation[str, float]:
    return Derivation(key(span, category), Edge("lex", (), category, log(0.5)))


def combined() -> Derivation[str, float]:
    children = (leaf((0, 1), "NP/N"), leaf((1, 2), "N"))

    return Derivation(
        key((0, 2), "NP"),
        Edge("fa", tuple(child.key for child in children), "NP", log(0.9)),
        children,
    )


def test_the_log_probability_ordering_is_an_ordering() -> None:
    order: Ordering[float] = LogProbability()

    assert isinstance(order, Ordering)
    assert order.name == "log-probability"
    assert order.rank(log(0.5)) == log(0.5)


def test_the_ranked_ordering_is_an_ordering() -> None:
    order: Ordering[Ranked] = BestOfRanked()

    assert isinstance(order, Ordering)
    assert order.name == "best-of-ranked"
    assert order.rank((-0.5, -2.0)) == -0.5
    assert order.rank(()) == -inf


def test_the_proof_figure_is_a_derivation_style() -> None:
    style: DerivationStyle[str] = ProofFigure(("the", "dog"), str, {"fa": ">"})

    assert isinstance(style, DerivationStyle)
    assert style.name == "proof-figure"
    assert style.render(combined()).split("\n")[-1].strip() == "NP"


def test_the_brace_string_is_a_derivation_style() -> None:
    style: DerivationStyle[str] = BraceString(
        ("the", "dog"), ("DT", "NN"), str, {"fa": 0}
    )

    assert isinstance(style, DerivationStyle)
    assert style.name == "brace-string"
    assert style.render(combined()).startswith("(<T NP 0 2>")


def test_the_dependency_reduction_is_a_derivation_reducer() -> None:
    reduction: DerivationReducer[str, tuple[HeadLink, ...]] = HeadDependencies(
        {"fa": 0}
    )

    assert isinstance(reduction, DerivationReducer)
    assert reduction.name == "head-dependencies"
    assert reduction.reduce(combined()) == (HeadLink(head=0, dependent=1, label="fa"),)


def test_a_value_missing_a_member_is_not_taken_for_a_style() -> None:
    assert not isinstance(LogProbability(), DerivationStyle)
    assert not isinstance(ProofFigure((), str), Ordering)
