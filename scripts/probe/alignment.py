"""Measures exact alignment coverage of grammar counterfactuals under checkpoints.

Each candidate sentence is tokenized under every declared released checkpoint,
and every row of every source token becomes that checkpoint's exact tokenizer
alignment. The counterfactual generator then enforces the alignment policy
under all of them at once, so a pair is accepted only where every checkpoint
admits its varied span. The report records the checkpoint identities in
evaluation order, the sentence and excluded-sentence counts, the exclusion count
of every checkpoint under every exclusion reason, the candidate and accepted
counts, the aligned fraction, the refusal count of every checkpoint under every
refusal reason, the accepted pair
identities, and a digest of the candidate sentences, so a rerun over the same
sentences, grammar, policy and seed reproduces it exactly. Where the
counterfactuals section declares a score filter, the report also holds the
filter's source and its primary, sweep and unfiltered arms over the accepted
pairs, each with its threshold, kept and dropped counts and kept identities.

``main`` reads the counterfactuals section of a probe document composed over a
completed run's recorded configuration, restores the run's trained point, builds
the declared generator under the point's own grammar, reads the declared
candidate sentences and measures them under the checkpoints a separate
declaration names. That declaration names every choice and none has a default:
the checkpoints, each a hub source with its local source directory or none to
fetch it through the declared artifacts cache, the string words are joined by,
the row bound or none for each tokenizer's published one, and the artifacts
cache: its root and whether an absent release is refused rather than fetched.
Run it as::

    python -m scripts.probe.alignment --run RUN --extra DOC --declaration DECL \
        --out REPORT

A sentence some checkpoint cannot align whole, because a word has no row, a row
crosses two words or the sentence overflows the checkpoint's window, yields no
candidate. It is excluded before generation, and the report counts it under
every such checkpoint and reason beside the refusals, so the sentence count, the
excluded count and the accepted population account for every candidate
sentence.
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

from lofi._checks import ensure_name, ensure_positive
from lofi.nn._snapshots import artifact_fetcher
from lofi.nn.tokens.pretrained import (
    HuggingFacePretrainedLoader,
    LocalPretrainedLoader,
    load_aligner,
)
from lofi.probe import (
    ALIGNMENT_EXCLUSION_REASONS,
    ALIGNMENT_REFUSAL_REASONS,
    AlignmentPolicy,
    filter_counterfactuals,
    read_sentence_scores,
    require_score_coverage,
)
from lofi.run import RESOLVED_NAME, Selection
from lofi.run.hub import restore_trained_point
from lofi.run.probing import (
    aligned_population,
    candidates_digest,
    counterfactual_generator,
    probe_document,
    read_candidate_sentences,
)
from lofi.spec import config_search_path
from lofi.spec.artifacts import (
    ArtifactCacheSettings,
    ArtifactCacheSpec,
    PretrainedSourceSpec,
    source_artifact,
)
from lofi.spec.tokens.pretrained import ContextWindowSettings

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._artifacts import PinnedCheckpoint
    from lofi._types import JsonValue
    from lofi.nn.tokens.pretrained import CheckpointAligner, PretrainedLoader
    from lofi.probe import (
        AlignmentExclusionReason,
        AlignmentRefusalReason,
        CounterfactualGenerator,
        FilterReport,
        GeneratedPairId,
        RowAligner,
    )
    from lofi.spec.components.counterfactuals import CounterfactualFilterSettings

__all__ = [
    "DECLARED_FIELDS",
    "AlignmentCoverage",
    "AlignmentDeclaration",
    "DeclaredCheckpoint",
    "DeclaredFilter",
    "ExclusionCount",
    "RefusalCount",
    "coverage_json",
    "declaration_of",
    "load_aligners",
    "main",
    "measure_coverage",
]

DECLARED_FIELDS: Final = ("checkpoints", "join", "window_rows", "artifacts")
"""Every field an alignment declaration names, each required."""

CHECKPOINT_FIELDS: Final = ("checkpoint", "source")
"""Every field a declared checkpoint names, each required."""


class RefusalCount(NamedTuple):
    """How many candidates one checkpoint refused for one reason."""

    tokenizer: str
    reason: AlignmentRefusalReason
    count: int


class ExclusionCount(NamedTuple):
    """How many candidate sentences one checkpoint excluded for one reason."""

    tokenizer: str
    reason: AlignmentExclusionReason
    count: int


class DeclaredFilter(NamedTuple):
    """A declared score filter and the sentence scores it reads."""

    settings: CounterfactualFilterSettings
    scores: Mapping[tuple[str, ...], float]


class AlignmentCoverage(NamedTuple):
    """The exact alignment coverage of one candidate population."""

    grammar: str
    dataset: str
    seed: int
    policy: AlignmentPolicy
    tokenizers: tuple[str, ...]
    candidates: str
    sentence_count: int
    excluded_count: int
    exclusion_counts: tuple[ExclusionCount, ...]
    candidate_count: int
    accepted_count: int
    aligned_fraction: float | None
    refusal_counts: tuple[RefusalCount, ...]
    accepted: tuple[GeneratedPairId, ...]
    filtered: tuple[CounterfactualFilterSettings, FilterReport] | None


def load_aligners(
    checkpoints: Sequence[PinnedCheckpoint],
    *,
    loader: PretrainedLoader,
    join: str,
    window: ContextWindowSettings,
) -> tuple[CheckpointAligner, ...]:
    """Return one word-row aligner per checkpoint, in declaration order.

    Parameters
    ----------
    checkpoints : Sequence[PinnedCheckpoint]
        The released checkpoints whose tokenizers are measured.
    loader : PretrainedLoader
        The backend the tokenizers are loaded through.
    join : str
        The string a sentence's words are joined by before tokenization.
    window : ContextWindowSettings
        The declared row bound, or none for each tokenizer's published one.
    """
    return tuple(
        load_aligner(checkpoint, loader=loader, join=join, window=window)
        for checkpoint in checkpoints
    )


def measure_coverage[T, W](
    generator: CounterfactualGenerator[T, W],
    sentences: Sequence[Sequence[str]],
    aligners: Sequence[RowAligner],
    *,
    policy: AlignmentPolicy,
    dataset: str,
    seed: int,
    screen: DeclaredFilter | None = None,
) -> AlignmentCoverage:
    """Return the exact alignment coverage of a candidate population.

    Parameters
    ----------
    generator : CounterfactualGenerator[T, W]
        The generator whose candidates are aligned.
    sentences : Sequence[Sequence[str]]
        The candidate sentences, as source tokens.
    aligners : Sequence[RowAligner]
        The tokenizers, in evaluation order.
    policy : AlignmentPolicy
        The alignment policy every tokenizer applies.
    dataset : str
        The handle of the data the sentences came from.
    seed : int
        The seed the generator orders each sentence's pairs under.
    screen : DeclaredFilter | None
        The declared score filter reported over the accepted pairs, or None.

    Raises
    ------
    ValueError
        If a tokenizer fails on a sentence for a reason other than a word with
        no row, a row crossing two words or a window overflow, or the declared
        scores do not cover exactly the candidate sentences.
    """
    if screen is not None:
        require_score_coverage(screen.scores, sentences)

    tokenizers = tuple(aligner.identity for aligner in aligners)
    inputs, exclusions = aligned_population(sentences, aligners)
    generation = generator.generate(
        inputs,
        policy=policy,
        dataset=dataset,
        seed=seed,
        exclusions=exclusions,
    )
    counted = {
        (tokenizer, reason): 0
        for tokenizer in tokenizers
        for reason in ALIGNMENT_REFUSAL_REASONS
    }
    excluded = {
        (tokenizer, reason): 0
        for tokenizer in tokenizers
        for reason in ALIGNMENT_EXCLUSION_REASONS
    }

    for refusal in generation.refusals:
        counted[refusal.tokenizer, refusal.reason] += 1

    for exclusion in generation.exclusions:
        excluded[exclusion.tokenizer, exclusion.reason] += 1

    manifest = generation.manifest

    return AlignmentCoverage(
        generator.grammar,
        dataset,
        seed,
        policy,
        tokenizers,
        candidates_digest(sentences),
        len(sentences),
        len(sentences) - len(inputs),
        tuple(
            ExclusionCount(tokenizer, reason, count)
            for (tokenizer, reason), count in excluded.items()
        ),
        manifest.candidate_count,
        manifest.aligned_count,
        manifest.aligned_fraction,
        tuple(
            RefusalCount(tokenizer, reason, count)
            for (tokenizer, reason), count in counted.items()
        ),
        tuple(pair.id for pair in generation.pairs),
        None
        if screen is None
        else (
            screen.settings,
            filter_counterfactuals(
                generation.pairs,
                screen.scores,
                primary=screen.settings.primary,
                sweep=screen.settings.sweep,
            ),
        ),
    )


def _filter_json(
    filtered: tuple[CounterfactualFilterSettings, FilterReport] | None, /
) -> JsonValue:
    """Return a declared filter's source and arms as JSON-ready data."""
    if filtered is None:
        return None

    settings, report = filtered

    return {
        "source": {
            "name": settings.source,
            "public": settings.public,
            "fit_signature": settings.fit_signature,
        },
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


def coverage_json(coverage: AlignmentCoverage, /) -> dict[str, JsonValue]:
    """Return a coverage report as canonical JSON-ready data."""
    return {
        "grammar": coverage.grammar,
        "dataset": coverage.dataset,
        "seed": coverage.seed,
        "policy": {
            "pieces": coverage.policy.pieces,
            "specials": coverage.policy.specials,
            "contiguous": coverage.policy.contiguous,
        },
        "checkpoints": list(coverage.tokenizers),
        "candidates": coverage.candidates,
        "sentence_count": coverage.sentence_count,
        "excluded_count": coverage.excluded_count,
        "exclusions": [
            {
                "checkpoint": count.tokenizer,
                "reason": count.reason,
                "count": count.count,
            }
            for count in coverage.exclusion_counts
        ],
        "candidate_count": coverage.candidate_count,
        "accepted_count": coverage.accepted_count,
        "aligned_fraction": coverage.aligned_fraction,
        "refusals": [
            {
                "checkpoint": count.tokenizer,
                "reason": count.reason,
                "count": count.count,
            }
            for count in coverage.refusal_counts
        ],
        "accepted": list(coverage.accepted),
        "filter": _filter_json(coverage.filtered),
    }


class DeclaredCheckpoint(NamedTuple):
    """One declared released checkpoint and where its tokenizer is loaded from."""

    #: The release at its pinned commit.
    checkpoint: PinnedCheckpoint

    #: A local directory holding the release's files, or None to fetch it
    #: through the declared artifacts cache.
    source: Path | None


class AlignmentDeclaration(NamedTuple):
    """The checkpoints and tokenization one coverage measurement declares."""

    #: The checkpoints, in evaluation order.
    checkpoints: tuple[DeclaredCheckpoint, ...]

    #: The string a sentence's words are joined by before tokenization.
    join: str

    #: The row bound a sentence must fit whole, or None for each tokenizer's own.
    window_rows: int | None

    #: The cache every checkpoint without a source is fetched into.
    artifacts: ArtifactCacheSettings


def _string(document: Mapping[str, JsonValue], field: str) -> str:
    """Return one declared string field."""
    value = document[field]

    if not isinstance(value, str):
        msg = f"the declared {field} is not a string: {value!r}"
        raise TypeError(msg)

    return value


def _object(document: Mapping[str, JsonValue], field: str) -> dict[str, JsonValue]:
    """Return one declared object field."""
    value = document[field]

    if not isinstance(value, dict):
        msg = f"the declared {field} is not an object: {value!r}"
        raise TypeError(msg)

    return value


def _checkpoint(value: JsonValue) -> DeclaredCheckpoint:
    """Return one validated declared checkpoint."""
    if not isinstance(value, dict):
        msg = f"a declared checkpoint is not an object: {value!r}"
        raise TypeError(msg)

    missing = sorted(set(CHECKPOINT_FIELDS).difference(value))
    unknown = sorted(set(value).difference(CHECKPOINT_FIELDS))

    if missing or unknown:
        msg = f"a declared checkpoint is missing {missing} and names unknown {unknown}"
        raise ValueError(msg)

    source = value["source"]

    if source is not None and not isinstance(source, str):
        msg = f"a declared checkpoint source is not a path or null: {source!r}"
        raise TypeError(msg)

    return DeclaredCheckpoint(
        source_artifact(
            PretrainedSourceSpec.model_validate(_object(value, "checkpoint"))
        ).hub_checkpoint("an alignment checkpoint"),
        None if source is None else Path(ensure_name("a checkpoint source", source)),
    )


def declaration_of(document: Mapping[str, JsonValue]) -> AlignmentDeclaration:
    """Return a validated alignment declaration from a JSON document.

    Parameters
    ----------
    document : Mapping[str, JsonValue]
        One object naming exactly the declared fields.

    Raises
    ------
    TypeError
        If a field has the wrong type.
    ValueError
        If a field is missing or unknown, no checkpoint is declared, or a value
        is out of range.
    """
    missing = sorted(set(DECLARED_FIELDS).difference(document))
    unknown = sorted(set(document).difference(DECLARED_FIELDS))

    if missing or unknown:
        msg = (
            f"an alignment declaration is missing {missing} and names unknown {unknown}"
        )
        raise ValueError(msg)

    checkpoints = document["checkpoints"]

    if not isinstance(checkpoints, list) or not checkpoints:
        msg = f"the declared checkpoints are not a nonempty list: {checkpoints!r}"
        raise ValueError(msg)

    rows = document["window_rows"]

    if rows is not None and (isinstance(rows, bool) or not isinstance(rows, int)):
        msg = f"the declared window_rows is not an integer or null: {rows!r}"
        raise TypeError(msg)

    artifacts = ArtifactCacheSpec.model_validate(
        _object(document, "artifacts")
    ).settings()

    declared = tuple(_checkpoint(checkpoint) for checkpoint in checkpoints)
    identities = [entry.checkpoint.identity for entry in declared]

    if len(set(identities)) != len(identities):
        msg = f"the declared checkpoints repeat one: {identities!r}"
        raise ValueError(msg)

    return AlignmentDeclaration(
        declared,
        _string(document, "join"),
        None if rows is None else ensure_positive("the declared window_rows", rows),
        artifacts,
    )


def _loader(
    declared: DeclaredCheckpoint, *, artifacts: ArtifactCacheSettings
) -> PretrainedLoader:
    """Return the backend a declared checkpoint's tokenizer is loaded through."""
    if declared.source is None:
        return HuggingFacePretrainedLoader(artifact_fetcher(artifacts))

    return LocalPretrainedLoader(declared.source, declared.checkpoint)


def main(argv: Sequence[str] | None = None) -> int:
    """Measure the declared counterfactual population of a trained point.

    Parameters
    ----------
    argv : Sequence[str] | None
        The arguments without the program name, or None to read them from the
        process.

    Returns
    -------
    int
        Zero once the report is written.

    Raises
    ------
    UnprobeableRunError
        If the run directory records no completed run or the composed document
        changes what the point was trained under.
    ValueError
        If the composed document declares no counterfactuals section, or a
        declaration is refused.
    FileExistsError
        If the report exists.
    """
    parser = ArgumentParser(
        prog="python -m scripts.probe.alignment",
        description=(
            "Measure the exact alignment coverage of a trained point's declared "
            "counterfactual population under released checkpoints."
        ),
    )
    parser.add_argument("--run", type=Path, required=True, metavar="RUN")
    parser.add_argument("--declaration", type=Path, required=True, metavar="DECL")
    parser.add_argument("--out", type=Path, required=True, metavar="REPORT")
    parser.add_argument(
        "--extra", dest="overlays", action="append", default=[], type=Path
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    namespace = parser.parse_args(argv)
    run: Path = namespace.run
    out: Path = namespace.out
    document = json.loads(namespace.declaration.read_text(encoding="utf-8"))

    if not isinstance(document, dict):
        msg = f"the alignment declaration is not an object: {document!r}"
        raise TypeError(msg)

    declaration = declaration_of(document)
    composed = probe_document(
        run,
        Selection(
            run / RESOLVED_NAME,
            None,
            tuple(namespace.overrides),
            tuple(namespace.overlays),
            config_search_path(Path.cwd()),
        ),
    )
    population = composed.counterfactuals

    if population is None:
        msg = f"the document composed over {run} declares no counterfactuals section"
        raise ValueError(msg)

    if out.exists():
        msg = f"the coverage report {out} exists"
        raise FileExistsError(msg)

    settings = population.settings()
    generator = counterfactual_generator(restore_trained_point(run), composed)
    sentences = read_candidate_sentences(settings.sentences)
    window = ContextWindowSettings(declaration.window_rows, "refuse")
    aligners = tuple(
        load_aligner(
            declared.checkpoint,
            loader=_loader(declared, artifacts=declaration.artifacts),
            join=declaration.join,
            window=window,
        )
        for declared in declaration.checkpoints
    )
    coverage = measure_coverage(
        generator,
        sentences,
        aligners,
        policy=AlignmentPolicy(
            settings.pieces, settings.specials, contiguous=settings.contiguous
        ),
        dataset=candidates_digest(sentences),
        seed=settings.seed,
        screen=None
        if settings.filter is None
        else DeclaredFilter(
            settings.filter, read_sentence_scores(settings.filter.scores)
        ),
    )

    with out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(coverage_json(coverage), handle, indent=2, sort_keys=True)
        handle.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
