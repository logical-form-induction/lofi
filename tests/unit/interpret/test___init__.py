from math import exp, fsum, isclose, log

from lofi import interpret
from lofi.deduce import (
    Derivation,
    Edge,
    ItemKey,
    KBest,
    LogInside,
    PackedForest,
    SpanIndex,
)

TOKENS = ("the", "dog", "barks")
TAGS = ("DT", "NN", "VBZ")


def key(span: tuple[int, int], category: str) -> ItemKey[str]:
    return ItemKey(span, category)


def score(derivation: Derivation[str, float]) -> float:
    return derivation.edge.weight + fsum(score(child) for child in derivation.children)


def forest() -> PackedForest[str, float]:
    chart = PackedForest[str, float](SpanIndex(3), LogInside())

    for span, category in (((0, 1), "NP/N"), ((1, 2), "N"), ((2, 3), "S\\NP")):
        chart.add(key(span, category), Edge("lex", (), category, log(0.5)))

    chart.add(
        key((0, 2), "NP"),
        Edge("fa", (key((0, 1), "NP/N"), key((1, 2), "N")), "NP", log(0.9), 1),
    )
    chart.add(
        key((1, 3), "S/N"),
        Edge("bx", (key((1, 2), "N"), key((2, 3), "S\\NP")), "S/N", log(0.1), 2),
    )
    chart.add(
        key((0, 3), "S"),
        Edge("ba", (key((0, 2), "NP"), key((2, 3), "S\\NP")), "S", log(0.7), 2),
    )
    chart.add(
        key((0, 3), "S"),
        Edge("fa", (key((0, 1), "NP/N"), key((1, 3), "S/N")), "S", log(0.4), 1),
    )

    return chart


def test_the_published_surface_unfolds_a_forest_under_a_bound() -> None:
    chart = forest()

    taken = list(interpret.unfold(chart, key((0, 3), "S"), limit=1))

    assert len(taken) == 1
    assert len(list(interpret.unfold(chart, key((0, 3), "S"), limit=5))) == 2


def test_the_published_surface_ranks_the_derivations_it_hands_over() -> None:
    chart = forest()
    order = interpret.LogProbability()

    best = next(interpret.unfold(chart, key((0, 3), "S"), limit=2, order=order))
    scores = interpret.k_best_scores(chart, key((0, 3), "S"), order=order, k=2)

    assert best.edge.rule_name == "ba"
    assert list(scores) == sorted(scores, reverse=True)
    assert isinstance(interpret.BestOfRanked(), interpret.Ordering)


def test_the_published_surface_agrees_with_a_sweep_in_the_k_best_algebra() -> None:
    chart = forest()
    ranked = PackedForest[str, tuple[float, ...]](SpanIndex(3), KBest(2))

    for item in (
        key((0, 1), "NP/N"),
        key((1, 2), "N"),
        key((2, 3), "S\\NP"),
    ):
        ranked.add(item, Edge("lex", (), item.reading, (log(0.5),)))

    ranked.add(
        key((0, 2), "NP"),
        Edge("fa", (key((0, 1), "NP/N"), key((1, 2), "N")), "NP", (log(0.9),), 1),
    )
    ranked.add(
        key((1, 3), "S/N"),
        Edge("bx", (key((1, 2), "N"), key((2, 3), "S\\NP")), "S/N", (log(0.1),), 2),
    )
    ranked.add(
        key((0, 3), "S"),
        Edge("ba", (key((0, 2), "NP"), key((2, 3), "S\\NP")), "S", (log(0.7),), 2),
    )
    ranked.add(
        key((0, 3), "S"),
        Edge("fa", (key((0, 1), "NP/N"), key((1, 3), "S/N")), "S", (log(0.4),), 1),
    )

    folded = interpret.k_best_scores(
        chart, key((0, 3), "S"), order=interpret.LogProbability(), k=2
    )

    assert all(
        isclose(a, b)
        for a, b in zip(folded, ranked.inside(key((0, 3), "S")), strict=True)
    )


def test_the_published_surface_reports_the_entropy_over_derivations() -> None:
    chart = forest()

    total = chart.inside(key((0, 3), "S"))
    shares = [
        exp(score(tree) - total)
        for tree in interpret.unfold(chart, key((0, 3), "S"), limit=5)
    ]
    enumerated = -fsum(share * log(share) for share in shares)

    assert isclose(fsum(shares), 1.0)
    assert isclose(interpret.derivation_entropy(chart, key((0, 3), "S")), enumerated)


def test_the_published_surface_decides_by_minimum_bayes_risk() -> None:
    chart = forest()
    marginals = {key((0, 2), "NP"): 1.0}

    chosen = interpret.mbr(
        chart,
        key((0, 3), "S"),
        marginals=marginals,
        utility=interpret.ConstituentUtility(),
        candidates=5,
        order=interpret.LogProbability(),
    )

    assert chosen is not None
    assert key((0, 2), "NP") in interpret.constituents(chosen)
    assert interpret.constituent_gain(chosen, marginals=marginals) == 1.0


def test_the_published_surface_renders_and_reduces_a_derivation() -> None:
    chart = forest()

    best = next(
        interpret.unfold(
            chart, key((0, 3), "S"), limit=1, order=interpret.LogProbability()
        )
    )
    figure = interpret.ProofFigure(TOKENS, str, {"fa": ">", "ba": "<"})
    braces = interpret.BraceString(TOKENS, TAGS, str, {"fa": 0, "ba": 1})
    reduction = interpret.HeadDependencies({"fa": 0, "ba": 1})

    assert figure.render(best).split("\n")[-1].strip() == "S"
    assert braces.render(best).startswith("(<T S 1 2>")
    assert reduction.reduce(best) == (
        interpret.HeadLink(head=2, dependent=0, label="ba"),
        interpret.HeadLink(head=0, dependent=1, label="fa"),
    )


def test_the_published_surface_includes_meaning_interpretation() -> None:
    assert interpret.CategoryReading
    assert interpret.MeaningActions
    assert interpret.MeaningReading
    assert interpret.StringReading
    assert interpret.StepContext
    assert interpret.WeightReading
    assert interpret.compile_meaning_actions
    assert interpret.run_action
