"""Declared probes executed against a trained point.

A run document declares its probes under ``probes``, each keyed by the role its
rows are reported under, and each built by the plugin its kind names in the
``lofi.probe`` entry point group. The probes are composed against the prepared
model before it is fitted: every component that publishes sites and admits an
instrumented region is offered to them, the held-out batches are run once to
find which published points they reach, and a probe naming a role or a site no
such component publishes where a held-out pass reaches it is refused then,
before any training step. After
training the probes are executed over the trained parser's held-out batches,
and their rows are written beside the run's other records and sent to the run's
declared metric sinks as one table.

A completed run directory is probed again with :func:`probe_trained_point`,
which restores the run's declared release checkpoint and executes the probes a document
composed over the run's recorded configuration declares. Such a document may
change only the run's probes, its counterfactual population and its sinks,
since every other section is what the checkpoint was trained under;
:func:`probe_document` composes one for other measurements of the same point.

A document's ``counterfactuals`` section declares a counterfactual population.
:func:`counterfactual_generator` builds its generator under the trained point's
own rules, lexicon and goal, :func:`read_candidate_sentences` reads its
candidates, and :func:`aligned_population` aligns them under some tokenizers,
excluding by reason each sentence a tokenizer cannot align whole.
:func:`pooled_leaves` returns the leaves the point's chart reads for given
sentences.

Executing a probe leaves the trained model as it found it. Each held-out pass
runs in the probe's declared gradient mode and in evaluation mode, inside
instrumented regions that restore their components when they close, and every
module's mode is restored afterwards.
"""

import json
from contextlib import contextmanager
from functools import partial
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, cast, final

import torch

from lofi.deduce import ParsePolicy, Retain
from lofi.nn.ports import Instrumentable, Observable, SubwordPooler, TokenEncoder
from lofi.nn.tokens.pretrained import CrossingPieceError, ZeroPieceWordError
from lofi.nn.tokens.window import ContextOverflowError
from lofi.probe import (
    AlignedInput,
    AlignmentExclusion,
    CounterfactualGenerator,
    PassLayout,
    ProbeRow,
    reached_sites,
    site_label,
    tokenizer_alignment,
)
from lofi.probe._probing import (
    PROBE_RECORDS_NAME,
    GradientMode,
    ProbeBuildContext,
    RecordingProbe,
    write_probe_records,
)
from lofi.registry import plugin_for
from lofi.run._boundary.manifest import ManifestStatus
from lofi.run._compose import Selection, compose, resolved_document
from lofi.run._device import resolve_device_policy
from lofi.run._executor import CHECKPOINTS_NAME
from lofi.run._manifest import read_manifest
from lofi.run._seed import RandomStreams
from lofi.run._sinks import SinkContext
from lofi.run._training import plan_training
from lofi.run._wiring import resolve_metric_sinks
from lofi.run.hub import restore_checkpoint
from lofi.stats import Dependence, NullProvenance
from lofi.terms import SExpSyntax
from lofi.train import prepare_training

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from pathlib import Path

    from torch import nn

    from lofi._types import ItemId, JsonValue
    from lofi.eval import HeldOutItems
    from lofi.nn._values import Encoded
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.nn.ports import Reading
    from lofi.probe import (
        AlignmentExclusionReason,
        DeclaredProbe,
        HeldOutPass,
        Probed,
        RowAligner,
        TokenizerAlignment,
    )
    from lofi.registry import ComponentPlugin
    from lofi.run._boundary.manifest import ExperimentManifest
    from lofi.run._ports import MetricSink
    from lofi.spec import (
        ConfiguredSpec,
        CounterfactualRendering,
        ProbeSpec,
        RunSpec,
    )
    from lofi.spec.grammar import GrammarAssembly
    from lofi.stats import NullFamily, Tail
    from lofi.terms import Term
    from lofi.train import (
        ParserFittingPlan,
        PreparedTraining,
    )

__all__ = [
    "PROBES_NAME",
    "PROBE_TABLE",
    "PointProbe",
    "ProbeExecution",
    "ProbeOutcome",
    "TrainedParser",
    "UnprobeableRunError",
    "aligned_population",
    "candidates_digest",
    "counterfactual_generator",
    "pooled_leaves",
    "probe_document",
    "probe_trained_point",
    "read_candidate_sentences",
    "read_probe_rows",
    "report_probe_rows",
    "write_probe_rows",
]

PROBES_NAME: Final = "probes.json"
"""The file in a run directory holding the rows of the run's declared probes."""

PROBE_TABLE: Final = "probes"
"""The table name under which probe rows reach a metric sink."""

type _ProbePlugin = ComponentPlugin[object, ProbeBuildContext, DeclaredProbe]
"""The plugin a probe kind lists, built with current pass capabilities."""


def aligned_population(
    sentences: Sequence[Sequence[str]], aligners: Sequence[RowAligner], /
) -> tuple[tuple[AlignedInput, ...], tuple[AlignmentExclusion, ...]]:
    """Return the sentences every tokenizer aligns whole and the exclusions.

    Every sentence is aligned under every tokenizer. A sentence some tokenizer
    cannot align whole, because a word has no row, a row crosses two words or
    the sentence overflows the tokenizer's window, is excluded, and each such
    tokenizer records one exclusion under its reason.

    Parameters
    ----------
    sentences : Sequence[Sequence[str]]
        The candidate sentences, as source tokens.
    aligners : Sequence[RowAligner]
        The tokenizers, in evaluation order.

    Returns
    -------
    tuple[tuple[AlignedInput, ...], tuple[AlignmentExclusion, ...]]
        The aligned sentences in sentence order, and every exclusion in sentence
        order and then tokenizer order.

    Raises
    ------
    ValueError
        If a tokenizer fails on a sentence for any other reason.
    """
    inputs: list[AlignedInput] = []
    exclusions: list[AlignmentExclusion] = []

    for sentence in sentences:
        tokens = tuple(sentence)
        alignments: list[TokenizerAlignment] = []
        reasons: list[tuple[str, AlignmentExclusionReason]] = []

        for aligner in aligners:
            try:
                alignments.append(tokenizer_alignment(aligner, tokens))
            except ZeroPieceWordError:
                reasons.append((aligner.identity, "zero-piece-word"))
            except CrossingPieceError:
                reasons.append((aligner.identity, "crossing-piece"))
            except ContextOverflowError:
                reasons.append((aligner.identity, "context-overflow"))

        if reasons:
            exclusions.extend(
                AlignmentExclusion(tokens, identity, reason)
                for identity, reason in reasons
            )
        else:
            inputs.append(AlignedInput(tokens, tuple(alignments)))

    return tuple(inputs), tuple(exclusions)


def read_candidate_sentences(path: Path, /) -> tuple[tuple[str, ...], ...]:
    """Return the candidate sentences of a file, one per line.

    Parameters
    ----------
    path : Path
        A text file holding one sentence per line, its tokens separated by
        whitespace.

    Raises
    ------
    ValueError
        If a line holds no token.
    """
    sentences: list[tuple[str, ...]] = []

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        tokens = tuple(line.split())

        if not tokens:
            msg = f"candidate sentence line {number + 1} of {path} holds no token"
            raise ValueError(msg)

        sentences.append(tokens)

    return tuple(sentences)


def candidates_digest(sentences: Sequence[Sequence[str]], /) -> str:
    """Return the SHA-256 digest of candidate sentences in order.

    The digest is the handle every pair generated from the sentences records as
    its dataset, so a pair's provenance does not depend on where the file lies.
    """
    payload = json.dumps(
        [list(sentence) for sentence in sentences],
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return sha256(payload.encode()).hexdigest()


def _rendering(
    grammar: GrammarAssembly, render: CounterfactualRendering, /
) -> Callable[[Term], str]:
    """Return how the declared rendering writes a category of the grammar."""
    syntaxes = {
        "declared": grammar.syntax,
        "s-expression": SExpSyntax(grammar.grammar.signature),
    }

    return partial(syntaxes[render].render, grammar.store)


def _trained_grammar(prepared: PreparedTraining, /) -> GrammarAssembly:
    """Return the learner grammar a prepared parser was fitted under."""
    return prepared.plan.parser_plan().grammar


def counterfactual_generator(
    prepared: PreparedTraining, run_spec: RunSpec, /
) -> CounterfactualGenerator[Term, float]:
    """Return the generator a run's counterfactuals section declares.

    The rules, the lexicon, the goal, the grammar fingerprint and the parse's
    thread count are the prepared parser's own, and the packed forest is read
    under the run's declared reachability.

    Parameters
    ----------
    prepared : PreparedTraining
        The prepared parser whose grammar the pairs are derived under.
    run_spec : RunSpec
        The document declaring the counterfactual population.

    Raises
    ------
    ValueError
        If the document declares no counterfactuals section, or the plan holds
        out no batch.
    """
    declared = run_spec.counterfactuals

    if declared is None:
        msg = "the document declares no counterfactuals section to generate from"
        raise ValueError(msg)

    settings = declared.settings()
    grammar = _trained_grammar(prepared)
    declared_parse = run_spec.parse.policy()

    return CounterfactualGenerator(
        grammar.grammar.rules,
        grammar.lexicon,
        grammar.grammar.goal,
        _rendering(grammar, settings.render),
        grammar.grammar.fingerprint,
        settings.unary_depth,
        ParsePolicy(
            Retain.FOREST,
            declared_parse.reachability,
            refinement_pairs=declared_parse.refinement_pairs,
        ),
        settings.readings,
        grammar.grammar.budget.threads,
    )


@contextmanager
def _evaluating(prepared: PreparedTraining, /) -> Iterator[None]:
    """Hold every module of a prepared model in evaluation mode, then restore it.

    Every module and submodule is restored to its own mode afterwards, each
    parent before its children, so a submodule whose mode differs from its
    parent's keeps it.
    """
    modules: tuple[nn.Module, ...] = tuple(
        module for root in prepared.model.modules.values() for module in root.modules()
    )
    training = tuple(module.training for module in modules)

    try:
        for module in modules:
            module.eval()

        yield

    finally:
        for module, mode in zip(modules, training, strict=True):
            module.train(mode)


def pooled_leaves(
    prepared: PreparedTraining, sentences: Sequence[Sequence[str]], /
) -> Encoded:
    """Return the pooled leaves a trained point's chart reads for some sentences.

    The sentences are encoded by the point's token encoder and pooled by its
    subword pooler, in evaluation mode and without an autograd graph, so the
    leaves are exactly what its chart scorer receives for them.

    Parameters
    ----------
    prepared : PreparedTraining
        The trained point.
    sentences : Sequence[Sequence[str]]
        The sentences, as words.

    Raises
    ------
    TypeError
        If the point's ``token_encoder`` or ``subword_pooler`` role is not a
        token encoder or a subword pooler.
    """
    modules = prepared.model.modules
    tokens = modules["token_encoder"]
    pooler = modules["subword_pooler"]

    if not isinstance(tokens, TokenEncoder):
        msg = "component role 'token_encoder' is not a token encoder"
        raise TypeError(msg)

    if not isinstance(pooler, SubwordPooler):
        msg = "component role 'subword_pooler' is not a subword pooler"
        raise TypeError(msg)

    with _evaluating(prepared), torch.no_grad():
        return pooler.pool(tokens.encode_words(sentences))


class UnprobeableRunError(ValueError):
    """A run directory cannot be probed as asked."""


class PointProbe(Protocol):
    """Probes the trained point a run directory holds."""

    def __call__(self, run: Path, selection: Selection, destination: Path, /) -> int:
        """Execute the probes a selection declares and write their rows.

        Parameters
        ----------
        run : Path
            The run directory of a completed point.
        selection : Selection
            The document composed over the run's recorded configuration.
        destination : Path
            The rows file to create, which must not exist.

        Returns
        -------
        int
            How many rows were written.
        """
        ...


def _gold(plan: ParserFittingPlan, /) -> Mapping[ItemId, Mapping[tuple[int, int], str]]:
    """Return each gold-analyzed item's constituents, each span under its label.

    A span covered by a unary chain is labeled by the chain's highest node.
    """
    analyzed: dict[ItemId, Mapping[tuple[int, int], str]] = {}
    corpus = plan.corpus

    if corpus is None:
        msg = "inference-pair training holds out no gold derivation corpus"
        raise ValueError(msg)

    for example in corpus.examples:
        derivation = example.supervision.derivation
        labels: dict[tuple[int, int], str] = {}

        for node_id in derivation.preorder:
            node = derivation.node(node_id)
            labels.setdefault((node.span.start, node.span.end), node.label)

        analyzed[example.item.item_id] = MappingProxyType(labels)

    return MappingProxyType(analyzed)


@final
class _HeldOutPass:
    """One held-out batch, run through the trained forward pass on demand."""

    __slots__ = ("_gold", "_plan", "_position", "_prepared")

    def __init__(
        self,
        prepared: PreparedTraining,
        plan: ParserFittingPlan,
        position: int,
        gold: Mapping[ItemId, Mapping[tuple[int, int], str]],
    ) -> None:
        self._prepared = prepared
        self._plan = plan
        self._position = position
        self._gold = gold

    def __call__(self, gradients: GradientMode, /) -> PassLayout:
        """Run the trained forward pass once in evaluation mode.

        Every module and submodule is restored to its own mode afterwards.
        """
        with (
            _evaluating(self._prepared),
            torch.set_grad_enabled(gradients is GradientMode.ENABLED),
        ):
            publish = cast(
                "Callable[[HeldOutItems], ObjectiveOutputs[HeldOutItems]]",
                self._plan.forward.publish,
            )
            outputs = publish(self._plan.fit_batches[self._position])

        marginals = outputs.marginals

        return PassLayout(
            outputs.batch.words,
            None if marginals is None else marginals.spans,
            tuple(self._gold.get(item_id) for item_id in outputs.batch.item_ids),
        )


@final
class TrainedParser:
    """A prepared parser offered to declared probes through its published sites.

    Parameters
    ----------
    prepared : PreparedTraining
        The prepared model and its fitting plan.

    Raises
    ------
    ValueError
        If the plan holds out no batch, as distance pretraining does not.
    """

    __slots__ = ("_components", "_passes")

    def __init__(self, prepared: PreparedTraining) -> None:
        plan = prepared.plan.parser_plan()
        gold = _gold(plan)
        self._components: Mapping[str, Probed[Reading]] = MappingProxyType(
            {
                role: cast("Probed[Reading]", module)
                for role, module in prepared.model.modules.items()
                if isinstance(module, Observable) and isinstance(module, Instrumentable)
            }
        )
        self._passes: tuple[HeldOutPass, ...] = tuple(
            _HeldOutPass(prepared, plan, position, gold)
            for position in range(len(plan.fit_batches))
        )

    @property
    def components(self) -> Mapping[str, Probed[Reading]]:
        """Return every component that publishes sites and admits a region."""
        return self._components

    @property
    def passes(self) -> tuple[HeldOutPass, ...]:
        """Return the held-out passes, in held-out order."""
        return self._passes


def _built(
    role: str, declaration: ProbeSpec, point: TrainedParser, /
) -> tuple[str, str, DeclaredProbe]:
    """Return a probe's role, its kind and the probe its declaration describes."""
    configured = cast("ConfiguredSpec[object]", declaration)
    plugin = cast("_ProbePlugin", plugin_for(configured))

    return (
        role,
        configured.kind,
        plugin.construct(configured, ProbeBuildContext(role, point)),
    )


class ProbeOutcome(NamedTuple):
    """The rows a run's probes measured and the records they persist.

    Attributes
    ----------
    rows : tuple[ProbeRow, ...]
        Each probe's values under its role and kind, in role order.
    records : Mapping[str, Mapping[str, JsonValue]]
        Each recording probe's named records under its role, holding only the
        roles that recorded something.
    """

    rows: tuple[ProbeRow, ...]
    records: Mapping[str, Mapping[str, JsonValue]]


@final
class ProbeExecution:
    """The probes a run declares, composed against a prepared parser.

    Parameters
    ----------
    run_spec : RunSpec
        The validated run document whose probes are composed.
    prepared : PreparedTraining
        The prepared model, which need not have been fitted yet.

    Raises
    ------
    UnpublishedSiteError
        If a probe names a role or a site no component of the model publishes
        where a held-out pass reaches it.
    ValueError
        If probes are declared and the plan holds out no batch.
    UnknownTagError
        If no installed probe is registered under a declared kind.
    """

    __slots__ = ("_point", "_probes")

    def __init__(self, run_spec: RunSpec, prepared: PreparedTraining) -> None:
        if not run_spec.probes:
            self._point: TrainedParser | None = None
            self._probes: tuple[tuple[str, str, DeclaredProbe], ...] = ()
            return

        point = TrainedParser(prepared)

        if not point.passes:
            msg = "the parser holds out no batch for its probes to read"
            raise ValueError(msg)

        probes = tuple(
            _built(role, declaration, point)
            for role, declaration in run_spec.probes.items()
        )
        reached = reached_sites(point)

        for _, _, probe in probes:
            probe.admit(reached)

        self._point = point
        self._probes = probes

    @property
    def roles(self) -> tuple[str, ...]:
        """Return the roles the composed probes are declared under, in order."""
        return tuple(role for role, _, _ in self._probes)

    def execute(self) -> ProbeOutcome:
        """Execute every composed probe and return its rows and records.

        Returns
        -------
        ProbeOutcome
            Each probe's values under its role and kind, in role order, and the
            records each recording probe persists.

        Raises
        ------
        ValueError
            If a probe cannot read what its passes publish.
        """
        point = self._point

        if point is None:
            return ProbeOutcome((), MappingProxyType({}))

        rows: list[ProbeRow] = []
        records: dict[str, Mapping[str, JsonValue]] = {}

        for role, kind, probe in self._probes:
            if isinstance(probe, RecordingProbe):
                measured = probe.recorded(point)
                values = measured.values

                if measured.records:
                    records[role] = measured.records
            else:
                values = probe.run(point)

            rows.extend(
                ProbeRow(
                    role,
                    kind,
                    value.measure,
                    None if value.site is None else site_label(value.site),
                    value.value,
                    value.count,
                    value.null,
                    value.raw_p,
                    value.adjusted_p,
                    value.rejected,
                    value.family,
                    value.dependence,
                )
                for value in values
            )

        return ProbeOutcome(tuple(rows), MappingProxyType(records))


def _null_record(null: NullProvenance | None, /) -> JsonValue:
    """Return null provenance as JSON, or the explicit absence of a null."""
    return None if null is None else null.to_json()


def _read_null(value: JsonValue, /) -> NullProvenance | None:
    """Return the real null provenance one JSON row records."""
    if value is None:
        return None
    if not isinstance(value, dict):
        msg = f"probe-row null provenance is not an object: {value!r}"
        raise TypeError(msg)

    exact = value["exact"]
    seed = value["seed"]
    matching = value["matching"]
    if type(exact) is not bool:
        msg = f"probe-row null exactness is not boolean: {exact!r}"
        raise TypeError(msg)
    if seed is not None and (type(seed) is not int or seed < 0):
        msg = f"probe-row null seed is not a non-negative integer: {seed!r}"
        raise TypeError(msg)
    if not isinstance(matching, dict):
        msg = f"probe-row null matching criteria are not an object: {matching!r}"
        raise TypeError(msg)

    return NullProvenance(
        str(value["name"]),
        cast("NullFamily", str(value["family"])),
        int(cast("int", value["replicates"])),
        tail=cast("Tail", str(value["tail"])),
        exact=exact,
        seed=seed,
        matching=cast("dict[str, str | int]", matching),
    )


def _record(row: ProbeRow, /) -> dict[str, JsonValue]:
    """Return one row as a JSON record."""
    return {
        "probe": row.probe,
        "kind": row.kind,
        "measure": row.measure,
        "site": row.site,
        "value": row.value,
        "count": row.count,
        "null": _null_record(row.null),
        "raw_p": row.raw_p,
        "adjusted_p": row.adjusted_p,
        "rejected": row.rejected,
        "family": row.family,
        "dependence": None if row.dependence is None else row.dependence.value,
    }


def write_probe_rows(path: Path, rows: Sequence[ProbeRow], /) -> None:
    """Write probe rows as a JSON list, replacing any file at the path.

    Parameters
    ----------
    path : Path
        The rows file.
    rows : Sequence[ProbeRow]
        The rows, in order.
    """
    path.write_text(
        json.dumps([_record(row) for row in rows], indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_probe_rows(path: Path, /) -> tuple[ProbeRow, ...]:
    """Return the probe rows a rows file holds, in order.

    Parameters
    ----------
    path : Path
        The rows file :func:`write_probe_rows` wrote.
    """
    records = cast("list[dict[str, JsonValue]]", json.loads(path.read_text("utf-8")))

    return tuple(
        ProbeRow(
            str(record["probe"]),
            str(record["kind"]),
            str(record["measure"]),
            None if record["site"] is None else str(record["site"]),
            float(cast("float", record["value"])),
            int(cast("int", record["count"])),
            _read_null(record["null"]),
            None if record["raw_p"] is None else float(cast("float", record["raw_p"])),
            (
                None
                if record["adjusted_p"] is None
                else float(cast("float", record["adjusted_p"]))
            ),
            (
                None
                if record["rejected"] is None
                else bool(cast("bool", record["rejected"]))
            ),
            None if record["family"] is None else str(record["family"]),
            (
                None
                if record["dependence"] is None
                else Dependence(str(record["dependence"]))
            ),
        )
        for record in records
    )


def report_probe_rows(sinks: Sequence[MetricSink], rows: Sequence[ProbeRow], /) -> None:
    """Send probe rows to every metric sink as one table.

    Parameters
    ----------
    sinks : Sequence[MetricSink]
        The run's metric sinks.
    rows : Sequence[ProbeRow]
        The rows, in order.
    """
    records = [_record(row) for row in rows]

    for sink in sinks:
        sink.log_table(PROBE_TABLE, records)


def _refuse_trained_changes(recorded: RunSpec, composed: RunSpec, /) -> None:
    """Refuse a composed document that changes what the checkpoint was trained under.

    The probes, the counterfactual population and the sections that only
    observe a run may change.

    Raises
    ------
    UnprobeableRunError
        If any other section differs from the recording.
    """
    free = recorded.observation_sections() | {"probes", "counterfactuals"}
    before = resolved_document(recorded)
    after = resolved_document(composed)
    changed = sorted(
        section
        for section in (before.keys() | after.keys()) - free
        if before.get(section) != after.get(section)
    )

    if changed:
        msg = (
            f"probing a trained run may change only {sorted(free)}, "
            f"and the composed document changes {changed}"
        )
        raise UnprobeableRunError(msg)


def _completed_composition(
    run: Path, selection: Selection, /
) -> tuple[ExperimentManifest, RunSpec]:
    """Return a completed run's manifest and the document composed over it.

    Raises
    ------
    UnprobeableRunError
        If the directory records no completed run, or the composed document
        changes more than the probes, the counterfactual population and the
        sinks.
    """
    composed = compose(selection)
    manifest = read_manifest(run)

    if manifest is None or manifest.status is not ManifestStatus.COMPLETED:
        msg = f"the run directory {run} records no completed run"
        raise UnprobeableRunError(msg)

    _refuse_trained_changes(manifest.run_spec, composed)

    return manifest, composed


def probe_document(run: Path, selection: Selection, /) -> RunSpec:
    """Return the probe document a selection composes over a completed run.

    Parameters
    ----------
    run : Path
        The run directory of a completed point.
    selection : Selection
        The overlays and overrides composed over the recorded configuration;
        its configuration path names the run's recorded configuration.

    Raises
    ------
    UnprobeableRunError
        If the directory records no completed run, or the composed document
        changes more than the probes, the counterfactual population and the
        sinks.
    """
    _, composed = _completed_composition(run, selection)

    return composed


def probe_trained_point(run: Path, selection: Selection, destination: Path, /) -> int:
    """Probe the trained point a run directory holds and write the rows.

    The run's recorded document is planned again and its model restored from
    the run's declared release checkpoint. The probes and sinks are those of
    the document the selection composes, which is composed over the run's
    recorded configuration. The run directory is left as it was.

    Parameters
    ----------
    run : Path
        The run directory of a completed point.
    selection : Selection
        The overlays and overrides composed over the recorded configuration;
        its configuration path names the run's recorded configuration.
    destination : Path
        The rows file to create, which must not exist. Records a recording
        probe persists are written beside it under ``probe-records.json``.

    Returns
    -------
    int
        How many rows were written.

    Raises
    ------
    UnprobeableRunError
        If the directory records no completed run, the composed document changes
        more than the probes and sinks, or declares no probe.
    FileExistsError
        If the destination exists.
    UnpublishedSiteError
        If a probe names a role or a site the trained model does not publish.
    """
    manifest, composed = _completed_composition(run, selection)

    if destination.exists():
        msg = f"the probe rows file {destination} exists"
        raise FileExistsError(msg)

    recorded = manifest.run_spec

    if not composed.probes:
        msg = f"the document composed over {run} declares no probe"
        raise UnprobeableRunError(msg)

    blueprint = plan_training(recorded)
    streams = RandomStreams(manifest.seed_policy.policy())
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(recorded))
    restore_checkpoint(
        run / CHECKPOINTS_NAME,
        blueprint,
        prepared,
        recorded.release.checkpoint,
    )
    rows, records = ProbeExecution(composed, prepared).execute()
    write_probe_rows(destination, rows)

    if records:
        write_probe_records(destination.with_name(PROBE_RECORDS_NAME), records)
    sinks = resolve_metric_sinks(
        composed.sinks, SinkContext(manifest.seed_derivation.point_id)
    )

    try:
        report_probe_rows(sinks, rows)

    finally:
        for sink in sinks:
            sink.close()

    return len(rows)
