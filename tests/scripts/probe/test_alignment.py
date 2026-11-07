import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest
from scripts.probe.alignment import (
    DECLARED_FIELDS,
    AlignmentCoverage,
    DeclaredFilter,
    RefusalCount,
    coverage_json,
    declaration_of,
    load_aligners,
    main,
    measure_coverage,
)
from tests.support.rules import ccg_signature, declared_rules

from lofi._artifacts import PinnedCheckpoint
from lofi.deduce import ParsePolicy, Reachability, Retain, RootGoal
from lofi.lexicon import Entry, StaticLexicon
from lofi.nn.tokens.pretrained import LocalPretrainedLoader
from lofi.probe import (
    ALIGNMENT_EXCLUSION_REASONS,
    ALIGNMENT_REFUSAL_REASONS,
    AlignmentPolicy,
    CounterfactualGenerator,
    TokenizerAlignment,
)
from lofi.run.hub import restore_trained_point
from lofi.run.probing import aligned_population
from lofi.spec import (
    ConstantPatternSpec,
    PatternRuleSpec,
    RuleSetSpec,
    TermPatternSpec,
)
from lofi.spec.components.counterfactuals import CounterfactualFilterSettings
from lofi.spec.tokens.pretrained import ContextWindowSettings
from lofi.terms import (
    Identity,
    TargetSet,
    Term,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi._types import JsonValue
    from lofi.nn.tokens.pretrained import CheckpointAligner
    from lofi.train import ParserFittingPlan

RULES: Final = declared_rules(
    ccg_signature("S", "Adv", "Vdcl", "Vng", name="toy-alignment"),
    RuleSetSpec(
        name="toy",
        rules=(
            PatternRuleSpec(
                name="apply",
                premises=(
                    ConstantPatternSpec(name="Adv"),
                    TermPatternSpec(name="X", sort="Cat"),
                ),
                conclusion=ConstantPatternSpec(name="S"),
            ),
        ),
    ),
)
STORE: Final = RULES.context.store
WORDPIECE: Final = PinnedCheckpoint("an-org/a-wordpiece", "a" * 40)
WORDLEVEL: Final = PinnedCheckpoint("an-org/a-wordlevel", "b" * 40)
EXCLUDING: Final = AlignmentPolicy("all", "exclude", contiguous=True)
INCLUDING: Final = AlignmentPolicy("all", "include", contiguous=True)
SENTENCES: Final = (("really", "sleeps"), ("really", "really", "sleeps"))


def a_category(name: str) -> Term:
    return STORE.const("Cat", name)


def a_generator() -> CounterfactualGenerator[Term, float]:
    lexicon = StaticLexicon(
        [
            Entry("really", a_category("Adv")),
            Entry("sleeps", a_category("Vdcl")),
            Entry("sleeps", a_category("Vng")),
        ]
    )

    return CounterfactualGenerator(
        RULES.rule_set,
        lexicon,
        RootGoal(TargetSet((a_category("S"),), Identity())),
        STORE.payload_name,
        "toy:1",
        unary_depth=0,
        parse_policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )


def tokenizer_assets(directory: Path, *, pieces: bool) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    padding, unknown, start, end = "[PAD]", "[UNK]", "[CLS]", "[SEP]"
    words = (
        [padding, unknown, start, end, "really", "sleep", "##s"]
        if pieces
        else [padding, unknown, "really", "sleeps"]
    )
    vocabulary = {word: index for index, word in enumerate(words)}
    model = (
        tokenizers.models.WordPiece(vocabulary, unk_token=unknown)
        if pieces
        else tokenizers.models.WordLevel(vocabulary, unk_token=unknown)
    )
    backend = tokenizers.Tokenizer(model)
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()

    if pieces:
        backend.post_processor = tokenizers.processors.TemplateProcessing(
            single=f"{start} $A {end}",
            special_tokens=[(start, vocabulary[start]), (end, vocabulary[end])],
        )

    transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=unknown,
        pad_token=padding,
        model_max_length=16,
    ).save_pretrained(directory)


def local_aligners(
    root: Path, *, rows: int | None = None
) -> tuple[CheckpointAligner, ...]:
    aligners: list[CheckpointAligner] = []

    for checkpoint, pieces in ((WORDPIECE, True), (WORDLEVEL, False)):
        directory = root / ("pieces" if pieces else "words")
        directory.mkdir(exist_ok=True)
        tokenizer_assets(directory, pieces=pieces)
        (aligner,) = load_aligners(
            (checkpoint,),
            loader=LocalPretrainedLoader(directory, checkpoint),
            join=" ",
            window=ContextWindowSettings(rows, "refuse"),
        )
        aligners.append(aligner)

    return tuple(aligners)


def measured(
    root: Path,
    policy: AlignmentPolicy,
    sentences: Sequence[Sequence[str]] = SENTENCES,
) -> AlignmentCoverage:
    return measure_coverage(
        a_generator(),
        sentences,
        local_aligners(root),
        policy=policy,
        dataset="unit",
        seed=3,
    )


def test_each_sentence_carries_the_exact_rows_of_every_checkpoint(
    tmp_path: Path,
) -> None:
    (first, second), excluded = aligned_population(SENTENCES, local_aligners(tmp_path))

    assert excluded == ()

    assert first.tokenizers == (WORDPIECE.identity, WORDLEVEL.identity)
    assert first.alignments == (
        TokenizerAlignment(WORDPIECE.identity, ((1,), (2, 3)), (0, 4), 5),
        TokenizerAlignment(WORDLEVEL.identity, ((0,), (1,)), (), 2),
    )
    assert second.alignments[0].token_rows == ((1,), (2,), (3, 4))


def test_a_fully_aligned_population_reports_every_count_exactly(
    tmp_path: Path,
) -> None:
    coverage = measured(tmp_path, EXCLUDING)

    assert coverage.tokenizers == (WORDPIECE.identity, WORDLEVEL.identity)
    assert coverage.candidate_count == len(SENTENCES)
    assert coverage.accepted_count == coverage.candidate_count
    assert coverage.aligned_fraction == 1.0
    assert len(coverage.accepted) == coverage.accepted_count
    assert coverage.refusal_counts == tuple(
        RefusalCount(tokenizer, reason, 0)
        for tokenizer in coverage.tokenizers
        for reason in ALIGNMENT_REFUSAL_REASONS
    )


def test_a_declared_score_filter_keeps_each_sentences_pairs_by_its_score(
    tmp_path: Path,
) -> None:
    settings = CounterfactualFilterSettings(
        scores=Path("scores.jsonl"),
        source="public",
        public=True,
        fit_signature=None,
        primary=0.5,
        sweep=(0.1, 0.3, 0.5, 0.7, 0.9),
    )
    screen = DeclaredFilter(settings, dict(zip(SENTENCES, (0.8, 0.2), strict=True)))
    coverage = measure_coverage(
        a_generator(),
        SENTENCES,
        local_aligners(tmp_path),
        policy=EXCLUDING,
        dataset="unit",
        seed=3,
        screen=screen,
    )

    assert coverage.filtered is not None
    filtered_settings, report = coverage.filtered
    kept = tuple(
        pair.id for pair in report.unfiltered.pairs if pair.tokens == SENTENCES[0]
    )

    assert filtered_settings == settings
    assert report.unfiltered.ids == coverage.accepted
    assert report.primary.ids == kept
    assert 0 < len(kept) < len(coverage.accepted)
    assert coverage_json(coverage)["filter"] == {
        "source": {"name": "public", "public": True, "fit_signature": None},
        "arms": [
            {
                "kind": arm.kind,
                "threshold": arm.threshold,
                "count": arm.count,
                "dropped": arm.dropped,
                "kept": list(arm.ids),
            }
            for arm in report.arms
        ],
    }


def test_a_score_filter_must_score_exactly_the_candidate_sentences(
    tmp_path: Path,
) -> None:
    settings = CounterfactualFilterSettings(
        scores=Path("scores.jsonl"),
        source="public",
        public=True,
        fit_signature=None,
        primary=0.5,
        sweep=(0.1, 0.3, 0.5, 0.7, 0.9),
    )

    with pytest.raises(ValueError, match="miss 1"):
        measure_coverage(
            a_generator(),
            SENTENCES,
            local_aligners(tmp_path),
            policy=EXCLUDING,
            dataset="unit",
            seed=3,
            screen=DeclaredFilter(settings, {SENTENCES[0]: 1.0}),
        )


def test_a_pair_refused_by_one_checkpoint_is_counted_against_it_alone(
    tmp_path: Path,
) -> None:
    coverage = measured(tmp_path, INCLUDING)
    counts = {
        (count.tokenizer, count.reason): count.count
        for count in coverage.refusal_counts
    }

    assert coverage.accepted_count == 0
    assert coverage.accepted == ()
    assert coverage.aligned_fraction == 0.0
    assert counts[WORDPIECE.identity, "noncontiguous-model-row-span"] == len(SENTENCES)
    assert sum(counts.values()) == len(SENTENCES)
    assert all(
        counts[WORDLEVEL.identity, reason] == 0 for reason in ALIGNMENT_REFUSAL_REASONS
    )


def test_an_empty_population_names_its_checkpoints_and_no_fraction(
    tmp_path: Path,
) -> None:
    coverage = measured(tmp_path, EXCLUDING, ())

    assert coverage.tokenizers == (WORDPIECE.identity, WORDLEVEL.identity)
    assert coverage.candidate_count == 0
    assert coverage.aligned_fraction is None
    assert coverage_json(coverage)["aligned_fraction"] is None


def test_a_rerun_reproduces_the_report_and_the_candidates_are_addressed(
    tmp_path: Path,
) -> None:
    first = coverage_json(measured(tmp_path, EXCLUDING))
    second = coverage_json(measured(tmp_path, EXCLUDING))
    other = coverage_json(measured(tmp_path, EXCLUDING, SENTENCES[:1]))

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["candidates"] != other["candidates"]
    assert first["grammar"] == "toy:1"
    assert first["policy"] == {
        "pieces": "all",
        "specials": "exclude",
        "contiguous": True,
    }
    assert first["checkpoints"] == [WORDPIECE.identity, WORDLEVEL.identity]


def test_a_sentence_a_checkpoint_cannot_align_whole_is_counted_as_excluded(
    tmp_path: Path,
) -> None:
    coverage = measure_coverage(
        a_generator(),
        SENTENCES,
        local_aligners(tmp_path, rows=5),
        policy=EXCLUDING,
        dataset="unit",
        seed=3,
    )
    excluded = {
        (count.tokenizer, count.reason): count.count
        for count in coverage.exclusion_counts
    }

    assert coverage.sentence_count == len(SENTENCES)
    assert coverage.excluded_count == 1
    assert coverage.candidate_count == coverage.accepted_count == 1
    assert excluded[WORDPIECE.identity, "context-overflow"] == 1
    assert sum(excluded.values()) == 1
    assert set(excluded) == {
        (tokenizer, reason)
        for tokenizer in coverage.tokenizers
        for reason in ALIGNMENT_EXCLUSION_REASONS
    }
    report = coverage_json(coverage)

    assert report["sentence_count"] == len(SENTENCES)
    assert report["excluded_count"] == 1
    assert {
        "checkpoint": WORDPIECE.identity,
        "reason": "context-overflow",
        "count": 1,
    } in cast("list[JsonValue]", report["exclusions"])


def declared_checkpoints(root: Path) -> list[JsonValue]:
    checkpoints: list[JsonValue] = []

    for checkpoint, pieces in ((WORDPIECE, True), (WORDLEVEL, False)):
        directory = root / ("pieces" if pieces else "words")
        directory.mkdir(exist_ok=True)
        tokenizer_assets(directory, pieces=pieces)
        checkpoints.append(
            {
                "checkpoint": {
                    "kind": "hub",
                    "repository": checkpoint.repository,
                    "revision": checkpoint.revision,
                },
                "source": str(directory),
            }
        )

    return checkpoints


def declaration(root: Path) -> dict[str, JsonValue]:
    return {
        "checkpoints": declared_checkpoints(root),
        "join": " ",
        "window_rows": 4,
        "artifacts": {"root": str(root / "artifacts"), "offline": True},
    }


def test_every_declared_field_is_required(tmp_path: Path) -> None:
    declared = declaration(tmp_path)

    for field in DECLARED_FIELDS:
        with pytest.raises(ValueError, match=field):
            declaration_of({k: v for k, v in declared.items() if k != field})

    with pytest.raises(ValueError, match="unknown"):
        declaration_of({**declared, "seed": 1})

    with pytest.raises(ValueError, match="checkpoint"):
        declaration_of({**declared, "checkpoints": []})


def test_main_measures_the_declared_population_under_a_restored_point(
    trained_run: Path, tmp_path: Path
) -> None:
    sentences = tmp_path / "candidates.txt"
    sentences.write_text(
        "word-0 word-1\nword-3 word-0 word-1\n", encoding="utf-8", newline="\n"
    )
    overlay = tmp_path / "counterfactuals.yaml"
    overlay.write_text(
        json.dumps(
            {
                "counterfactuals": {
                    "sentences": str(sentences),
                    "unary_depth": 0,
                    "readings": 4,
                    "render": "declared",
                    "pieces": "all",
                    "specials": "exclude",
                    "contiguous": True,
                    "seed": 3,
                }
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    declared = tmp_path / "alignment.json"
    declared.write_text(
        json.dumps(declaration(tmp_path)), encoding="utf-8", newline="\n"
    )
    out = tmp_path / "coverage.json"

    status = main(
        [
            "--run",
            str(trained_run),
            "--extra",
            str(overlay),
            "--declaration",
            str(declared),
            "--out",
            str(out),
        ]
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    plan = cast("ParserFittingPlan", restore_trained_point(trained_run).plan)

    assert status == 0
    assert report["grammar"] == plan.grammar.grammar.fingerprint
    assert report["seed"] == 3
    assert report["checkpoints"] == [WORDPIECE.identity, WORDLEVEL.identity]
    assert report["sentence_count"] == 2
    assert report["excluded_count"] == 1
    assert {
        "checkpoint": WORDPIECE.identity,
        "reason": "context-overflow",
        "count": 1,
    } in report["exclusions"]

    with pytest.raises(FileExistsError):
        main(
            [
                "--run",
                str(trained_run),
                "--extra",
                str(overlay),
                "--declaration",
                str(declared),
                "--out",
                str(out),
            ]
        )


def test_main_refuses_a_document_declaring_no_counterfactual_population(
    trained_run: Path, tmp_path: Path
) -> None:
    declared = tmp_path / "alignment.json"
    declared.write_text(
        json.dumps(declaration(tmp_path)), encoding="utf-8", newline="\n"
    )

    with pytest.raises(ValueError, match="counterfactuals"):
        main(
            [
                "--run",
                str(trained_run),
                "--declaration",
                str(declared),
                "--out",
                str(tmp_path / "coverage.json"),
            ]
        )


def test_main_reports_a_declared_score_filter_over_the_accepted_pairs(
    trained_run: Path, tmp_path: Path
) -> None:
    sentences = tmp_path / "candidates.txt"
    sentences.write_text(
        "word-0 word-1\nword-3 word-0 word-1\nword-0 word-2 word-4\n",
        encoding="utf-8",
        newline="\n",
    )
    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        json.dumps({"tokens": ["word-0", "word-1"], "score": 0.8})
        + "\n"
        + json.dumps({"tokens": ["word-3", "word-0", "word-1"], "score": 0.2})
        + "\n"
        + json.dumps({"tokens": ["word-0", "word-2", "word-4"], "score": 0.4})
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    sweep = [0.1, 0.3, 0.5, 0.7, 0.9]
    overlay = tmp_path / "counterfactuals.yaml"
    overlay.write_text(
        json.dumps(
            {
                "counterfactuals": {
                    "sentences": str(sentences),
                    "unary_depth": 0,
                    "readings": 4,
                    "render": "declared",
                    "pieces": "all",
                    "specials": "exclude",
                    "contiguous": True,
                    "seed": 3,
                    "filter": {
                        "scores": str(scores),
                        "source": {
                            "name": "public-ratings",
                            "public": True,
                            "fit_signature": None,
                        },
                        "primary": 0.5,
                        "sweep": sweep,
                    },
                }
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    declared = tmp_path / "alignment.json"
    declared.write_text(
        json.dumps({**declaration(tmp_path), "window_rows": None}),
        encoding="utf-8",
        newline="\n",
    )
    out = tmp_path / "coverage.json"

    main(
        [
            "--run",
            str(trained_run),
            "--extra",
            str(overlay),
            "--declaration",
            str(declared),
            "--out",
            str(out),
        ]
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    accepted = report["accepted"]
    arms = report["filter"]["arms"]
    assert report["filter"]["source"] == {
        "name": "public-ratings",
        "public": True,
        "fit_signature": None,
    }
    assert [arm["kind"] for arm in arms] == ["primary", *["sweep"] * 5, "unfiltered"]
    assert [arm["threshold"] for arm in arms] == [0.5, *sweep, None]
    assert arms[0]["kept"] == accepted
    assert arms[-1]["kept"] == accepted
    assert all(arm["count"] + arm["dropped"] == len(accepted) for arm in arms)
    assert all(arm["dropped"] == 0 for arm in arms[1:4])


def test_main_refuses_scores_that_miss_a_candidate_sentence(
    trained_run: Path, tmp_path: Path
) -> None:
    sentences = tmp_path / "candidates.txt"
    sentences.write_text("word-0 word-1\n", encoding="utf-8", newline="\n")
    scores = tmp_path / "scores.jsonl"
    scores.write_text("", encoding="utf-8")
    overlay = tmp_path / "counterfactuals.yaml"
    overlay.write_text(
        json.dumps(
            {
                "counterfactuals": {
                    "sentences": str(sentences),
                    "unary_depth": 0,
                    "readings": 4,
                    "render": "declared",
                    "pieces": "all",
                    "specials": "exclude",
                    "contiguous": True,
                    "seed": 3,
                    "filter": {
                        "scores": str(scores),
                        "source": {
                            "name": "public-ratings",
                            "public": True,
                            "fit_signature": None,
                        },
                        "primary": 0.5,
                        "sweep": [0.1, 0.3, 0.5, 0.7, 0.9],
                    },
                }
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    declared = tmp_path / "alignment.json"
    declared.write_text(
        json.dumps(declaration(tmp_path)), encoding="utf-8", newline="\n"
    )

    with pytest.raises(ValueError, match="miss 1 candidate"):
        main(
            [
                "--run",
                str(trained_run),
                "--extra",
                str(overlay),
                "--declaration",
                str(declared),
                "--out",
                str(tmp_path / "coverage.json"),
            ]
        )
