"""Constraint selection declared in run documents, through to the selected budget."""

from collections import Counter
from typing import TYPE_CHECKING, Final, cast

import pytest

from lofi.data import (
    CategoryRole,
    ClassRole,
    FrequencyCount,
    LogicalFormSource,
    RankBand,
    SelectionReport,
    UnionRule,
)
from lofi.data.synthetic.spec.corpus import SyntheticCorpusDecl, SyntheticCorpusLowerer
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.run import plan_training, resolve_constraint_policy
from lofi.spec import (
    PACKAGED_CONFIG_ROOT,
    ConfigValueError,
    PolicyDomain,
    RunSpecCodec,
)
from lofi.train import constraint_source, select_constraints

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.data.synthetic import SyntheticDataset
    from lofi.spec import RunSpec

SEARCH_PATH: Final = (PACKAGED_CONFIG_ROOT / "run", PACKAGED_CONFIG_ROOT)

SHARE_TOLERANCE: Final = 0.05
"""How far the share one policy selects may move between the two sizes."""

LEXICAL_MEANINGS: Final[tuple[JsonValue, ...]] = (
    {"form": "word-0", "category": "NP", "meaning": "john"},
    {"form": "word-4", "category": "NP", "meaning": "mary"},
    {"form": "word-1", "category": "S\\NP", "meaning": r"\x. sleep(x)"},
    {"form": "word-5", "category": "S\\NP", "meaning": r"\x. snore(x)"},
    {
        "form": "word-2",
        "category": "S\\NP/NP",
        "meaning": r"\y. \x. love(x)(y)",
    },
    {
        "form": "word-6",
        "category": "S\\NP/NP",
        "meaning": r"\y. \x. see(x)(y)",
    },
)


def generated_grammar(name: str, *, seed: int, categories: int) -> JsonValue:
    """Return a grammar whose unambiguous lexicon is generated from its rules.

    Parameters
    ----------
    name : str
        The grammar's name.
    seed : int
        The seed every generation choice is drawn under.
    categories : int
        How many lexical categories the inventory holds.
    """
    return {
        "name": name,
        "signature": {"primitives": ["S", "NP", "N", "PP", "conj"]},
        "rules": {
            "name": "application",
            "rules": [
                {"rule": "application", "view": "/"},
                {"rule": "application", "view": "\\"},
            ],
        },
        "lexicon": {
            "kind": "generated",
            "forms": 60,
            "categories": categories,
            "category_shares": {"kind": "power-law", "exponent": 1.5},
            "coupling": 0.0,
            "ambiguity": {
                "spelling": "parametric",
                "mean": 1.0,
                "dispersion": 0.0,
                "maximum": 1,
                "placement": 0.0,
            },
            "rank_law": {"kind": "power-law", "exponent": 1.1},
            "seed": seed,
            "max_depth": 2,
        },
        "goal": {"kind": "root", "categories": ["S"]},
        "matching": {"kind": "feature-erasure"},
        "budget": {"unary_rounds": 1},
    }


INTERPRETATION: Final[JsonValue] = {
    "erasure": {
        "kind": "signature-morphism",
        "atoms": {"S": "t", "NP": "e", "conj": "e"},
    },
    "constants": [
        {"name": "john", "type": "e", "classes": ["individual"]},
        {"name": "mary", "type": "e", "classes": ["individual"]},
        {"name": "sleep", "type": "e -> t", "classes": ["predicate"]},
        {"name": "snore", "type": "e -> t", "classes": ["predicate"]},
        {"name": "love", "type": "e -> e -> t", "classes": ["predicate"]},
        {"name": "see", "type": "e -> e -> t", "classes": ["predicate"]},
    ],
    "meanings": {"source": {"kind": "lexical"}},
}


def document(directory: Path, name: str, *defaults: str) -> Path:
    """Write a run document composing the parser run with constraint fragments.

    Parameters
    ----------
    directory : Path
        Where the document is written.
    name : str
        The document's file stem.
    *defaults : str
        The fragment selections after the parser run.
    """
    path = directory / f"{name}.yaml"
    lines = ["defaults:", "  - synthetic-parser", *(f"  - {d}" for d in defaults)]
    path.write_text(
        "\n".join([*lines, "  - loss: pinning", ""]), encoding="utf-8", newline="\n"
    )

    return path


def composed(path: Path, extra: dict[str, JsonValue]) -> RunSpec:
    """Return the run a document composes to under in-process settings.

    Parameters
    ----------
    path : Path
        The run document.
    extra : dict[str, JsonValue]
        Settings keyed by dotted path.
    """
    return RunSpecCodec(search_path=SEARCH_PATH)(path, extra=extra)


def drawn(run_spec: RunSpec) -> SyntheticDataset:
    """Return the synthetic corpus a run declares.

    Parameters
    ----------
    run_spec : RunSpec
        The validated run.
    """
    lowerer = SyntheticCorpusLowerer()

    return lowerer.lower(
        lowerer.check(
            cast("SyntheticCorpusDecl", run_spec.data.datasets["train"].dataset)
        ),
        ParsePolicy(Retain.FOREST, Reachability.ALL),
    )


def budget(run_spec: RunSpec) -> SelectionReport[str]:
    """Return the budget the run's declared source selects on its corpus.

    Parameters
    ----------
    run_spec : RunSpec
        The validated run.
    """
    selection = select_constraints(run_spec, "train", drawn(run_spec))

    assert selection is not None

    return selection.budget


@pytest.fixture
def lexical(tmp_path: Path) -> Path:
    return document(
        tmp_path,
        "lexical",
        "data.datasets.train.constraint_source: gold-lexicon",
        "data.datasets.train.constraint_policy.syntactic: all",
    )


def observed_types(dataset: SyntheticDataset) -> int:
    """Return how many distinct forms the observed items of a corpus hold.

    Parameters
    ----------
    dataset : SyntheticDataset
        The corpus.
    """
    return len(
        {
            token.surface
            for example in dataset.corpus.observed()
            for token in example.item.tokens
        }
    )


@pytest.mark.parametrize(
    ("selector", "scale"),
    [
        ({"kind": "rank", "first": 0.0, "last": 0.2}, "types"),
        ({"kind": "frequency", "at_least": 0.02}, "mass"),
        ({"kind": "quantile", "share": 0.5, "over": "tokens"}, "mass"),
        ({"kind": "quantile", "share": 0.3, "over": "types"}, "types"),
        ({"kind": "category-role", "role": "closed", "share": 0.4}, "mass"),
    ],
    ids=["rank", "frequency", "token-quantile", "type-quantile", "closed-role"],
)
def test_one_share_policy_selects_the_same_share_at_four_and_eight_hundred(
    lexical: Path, selector: dict[str, JsonValue], scale: str
) -> None:
    shares = []

    for size in (400, 800):
        run_spec = composed(
            lexical,
            {
                "data.datasets.train.dataset.size": size,
                "data.datasets.train.dataset.grammar": generated_grammar(
                    "share", seed=3, categories=5
                ),
                "data.datasets.train.constraint_policy.syntactic.selector": selector,
            },
        )
        dataset = drawn(run_spec)
        selection = select_constraints(run_spec, "train", dataset)

        assert selection is not None

        report = selection.budget

        assert report.size_invariant
        assert report.types_pinned > 0

        shares.append(
            report.mass_share
            if scale == "mass"
            else report.types_pinned / observed_types(dataset)
        )

    small, large = shares

    assert abs(small - large) <= SHARE_TOLERANCE


def test_the_absolute_count_arm_is_selectable_and_named_in_the_report(
    lexical: Path,
) -> None:
    run_spec = composed(
        lexical,
        {
            "data.datasets.train.constraint_policy.syntactic.selector": {
                "kind": "count",
                "at_least": 2,
            }
        },
    )
    report = budget(run_spec)

    assert report.selector == FrequencyCount(2, None)
    assert not report.size_invariant
    assert "FrequencyCount(2, None)" in repr(report)


@pytest.mark.parametrize(("seed", "categories"), [(3, 5), (11, 7)])
def test_a_closed_class_selector_targets_the_smallest_class_of_a_generated_grammar(
    lexical: Path, seed: int, categories: int
) -> None:
    run_spec = composed(
        lexical,
        {
            "data.datasets.train.dataset.size": 320,
            "data.datasets.train.dataset.grammar": generated_grammar(
                f"closed-{seed}", seed=seed, categories=categories
            ),
            "data.datasets.train.constraint_policy.syntactic.selector": {
                "kind": "category-role",
                "role": "closed",
                "share": 1 / categories,
            },
        },
    )
    dataset = drawn(run_spec)
    grammar = dataset.source_grammar
    observed = {
        token.surface
        for example in dataset.corpus.observed()
        for token in example.item.tokens
    }
    sizes = Counter(
        grammar.syntax.render(grammar.store, entry.category)
        for form in grammar.lexicon.forms
        if form in observed
        for entry in grammar.lexicon.entries(form)
    )
    smallest = min(sizes.values())
    source = constraint_source(run_spec, "train", dataset)
    selection = select_constraints(run_spec, "train", dataset)

    assert source is not None
    assert selection is not None

    report = selection.budget

    assert source.policy.selector == CategoryRole(ClassRole.CLOSED, 1 / categories)
    assert set(report.by_category) == {
        category for category, size in sizes.items() if size == smallest
    }
    assert report.types_pinned == sum(
        size for size in sizes.values() if size == smallest
    )


def interpreted(
    tmp_path: Path, *defaults: str, extra: dict[str, JsonValue] | None = None
) -> RunSpec:
    """Return the parser run over an interpreted corpus under constraint fragments.

    Parameters
    ----------
    tmp_path : Path
        Where the document is written.
    *defaults : str
        The constraint fragment selections.
    extra : dict[str, JsonValue] | None
        Further settings keyed by dotted path.
    """
    return composed(
        document(tmp_path, "interpreted", *defaults),
        {
            "data.datasets.train.dataset.interpretation": INTERPRETATION,
            "data.datasets.train.dataset.grammar.lexicon.entries": list(
                LEXICAL_MEANINGS
            ),
            "data.datasets.train.dataset.language.logic.equality": False,
            "data.datasets.train.dataset.size": 16,
            **({} if extra is None else extra),
        },
    )


def test_a_semantic_policy_reaches_a_constraint_source_from_a_document(
    tmp_path: Path,
) -> None:
    run_spec = interpreted(
        tmp_path,
        "data.datasets.train.constraint_source: gold-logical-form",
        "data.datasets.train.constraint_policy.semantic: all",
    )
    dataset = drawn(run_spec)
    source = cast(
        "LogicalFormSource[str, str]",
        constraint_source(run_spec, "train", dataset),
    )
    selection = select_constraints(run_spec, "train", dataset)
    interpretation = dataset.source_interpretation

    assert run_spec.data.datasets["train"].constraint_source is not None
    assert (
        run_spec.data.datasets["train"].constraint_source.policy
        is PolicyDomain.SEMANTIC
    )
    assert source.policy == resolve_constraint_policy(
        run_spec.data.datasets["train"].constraint_policy.semantic
    )
    assert selection is not None
    assert source.policy.coverage == 1.0
    assert source.policy.ambiguity == UnionRule()
    assert source.policy.selector == RankBand(0.0, 1.0)
    assert interpretation is not None
    assert {
        label
        for example in selection.examples
        for constraint in example.supervision.constraints
        for label in constraint.permitted
    } <= {
        interpretation.render(meaning)
        for meanings in interpretation.entries.values()
        for meaning in meanings
    }
    assert selection.budget.types_pinned > 0
    assert {"individual", "predicate"} <= set(selection.budget.by_category)


def test_a_semantic_policy_cannot_be_applied_to_syntactic_categories(
    tmp_path: Path,
) -> None:
    meanings = interpreted(
        tmp_path,
        "data.datasets.train.constraint_source: gold-logical-form",
        "data.datasets.train.constraint_policy.semantic: all",
    )

    with pytest.raises(
        ConfigValueError,
        match=r"data\.datasets\.train\.constraint_source.*syntactic categories",
    ):
        interpreted(
            tmp_path,
            "data.datasets.train.constraint_source: gold-derivation",
            "data.datasets.train.constraint_policy.semantic: all",
            extra={"data.datasets.train.constraint_source.policy": "semantic"},
        )

    with pytest.raises(
        ValueError, match=r"serves the semantic policy.*represents 'Cat'"
    ):
        plan_training(meanings)
