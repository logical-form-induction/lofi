"""Layers corpus reading into shared data-layer values."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple

from lairs import Corpus
from lairs.data import JudgmentStudy, load_judgment_study
from lairs.data.dataset import Dataset
from lairs.records import annotation as annotation_records
from lairs.records import corpus as corpus_records
from lairs.records import expression as expression_records
from lairs.records import graph as graph_records
from lairs.records import judgment as judgment_records
from lairs.records import segmentation as segmentation_records
from lairs.store.repository import Repository

from lofi.data import (
    AnalysisCounts,
    Example,
    GoldDerivation,
    Item,
    LoadReport,
    Unlabeled,
    reachability_report,
)
from lofi.data.corpora.layers._translate import (
    EMITTED_ARMS,
    GRAPH_LIFT,
    RELATION_LIFT,
    RELATION_NAMES,
    GraphContext,
    LayersSkip,
    RecordSkippedError,
    RelationPair,
    TokenReading,
    annotator_of,
    graph_start,
    item_id_of,
    pair_response_of,
    provenance_of,
    relation_pair_of,
    response_of,
    supervision_of,
    tokenization_of,
    tokens_of,
)
from lofi.data.corpora.layers.spec import records_at

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping
    from pathlib import Path

    from lairs.atproto.pds import PdsClient

    from lofi._types import ItemId
    from lofi.data import (
        Derivation,
        DerivationOutcome,
        LabelMembership,
        Provenance,
        StepRecovery,
        Supervision,
        SupervisionArm,
        Token,
    )

__all__ = [
    "READER_NAME",
    "ComparisonInventory",
    "JudgmentStudyOptions",
    "LayersCorpusReader",
    "corpus_at",
    "judgment_study_at",
    "reader_at",
]


READER_NAME: Final = "layers"
"""The registered reader name."""

_JUDGMENT_SPLIT: Final = "train"
"""The split for a standalone judgment study."""


def _arm_of(supervision: Supervision[str]) -> SupervisionArm | None:
    """Return the arm of the supervision union one value belongs to.

    Parameters
    ----------
    supervision : Supervision[str]
        The value to place.

    Returns
    -------
    SupervisionArm | None
        The matching arm, or None if the reader does not emit it.
    """
    for arm in EMITTED_ARMS:
        if isinstance(supervision, arm):
            return arm

    return None


def _sole_corpus(uris: Iterable[str]) -> str | None:
    """Return the repository's sole corpus-record AT-URI.

    Parameters
    ----------
    uris : Iterable[str]
        Every AT-URI the repository stages.

    Returns
    -------
    str | None
        The corpus record's URI, or None if the repository holds none,
        in which case every membership it holds is read.

    Raises
    ------
    ValueError
        If more than one corpus record is staged, since which of them is meant
        is then undecided.
    """
    found = list(uris)

    if not found:
        return None

    if len(found) > 1:
        msg = (
            f"the repository holds {len(found)} corpus records, so which of "
            "them is to be read has to be named"
        )
        raise ValueError(msg)

    return found[0]


def _repository_at(
    directory: Path,
    corpus_uri: str | None,
) -> tuple[Corpus, tuple[str, ...]]:
    """Load known repository records and their skip reasons."""
    repository = Repository.open(directory)
    reading = records_at(repository)
    corpus = Corpus(uri=corpus_uri or _sole_corpus(reading.corpus_uris))

    for uri, record in reading.records:
        corpus.add_record(uri, record)

    return corpus, reading.skips


def corpus_at(directory: Path, *, corpus_uri: str | None = None) -> Corpus:
    """Read a valid committed Layers repository into a corpus.

    Parameters
    ----------
    directory : Path
        The repository directory.
    corpus_uri : str | None
        The AT-URI of the corpus record whose members are to be read. Absent
        where the repository holds exactly one corpus record, which is then
        the one read.

    Returns
    -------
    Corpus
        The pooled record graph.

    Raises
    ------
    ValueError
        If no corpus URI is given and the repository stages more than one
        corpus record.
    """
    corpus, skips = _repository_at(directory, corpus_uri)

    if skips:
        msg = f"the repository holds {len(skips)} invalid Layers records"
        raise ValueError(msg)

    return corpus


class JudgmentStudyOptions(NamedTuple):
    """Options for loading a Layers judgment study."""

    source: str = "auto"
    cache_dir: str | None = None
    revision: str | None = None
    pds_client: PdsClient | None = None
    follow_refs: bool = True


def judgment_study_at(
    uri: str,
    *,
    options: JudgmentStudyOptions | None = None,
) -> JudgmentStudy:
    """Load a Layers judgment study."""
    selected = JudgmentStudyOptions() if options is None else options

    return load_judgment_study(
        uri,
        source=selected.source,
        cache_dir=selected.cache_dir,
        revision=selected.revision,
        pds_client=selected.pds_client,
        follow_refs=selected.follow_refs,
    )


class ComparisonInventory[Recovered](NamedTuple):
    """The label inventory and rule set decoded derivations are classified against.

    Parameters
    ----------
    labels : LabelMembership[str]
        The labels the comparison admits.
    rules : StepRecovery[str, Recovered]
        The rule set that recovers a nonlexical step.
    """

    labels: LabelMembership[str]
    rules: StepRecovery[str, Recovered]


class _Response(NamedTuple):
    """One response with its experiment and annotator.

    Parameters
    ----------
    item_id : ItemId
        The expression the response is anchored on: the premise for a pair
        response, or the target expression for a unary response.
    judgment : judgment_records.Judgment
        The response itself.
    experiment : judgment_records.ExperimentDef | None
        The experiment it was collected under, or None if the corpus
        does not hold the experiment the set names.
    annotator : str | None
        Whoever responded, where the corpus records it.
    """

    item_id: ItemId
    judgment: judgment_records.Judgment
    experiment: judgment_records.ExperimentDef | None
    annotator: str | None
    pair: RelationPair | None


class _ResponseIndex(NamedTuple):
    """Responses by item and failures found while resolving pairs."""

    grouped: Mapping[str, tuple[_Response, ...]]
    skips: tuple[str, ...]


class _Candidate(NamedTuple):
    """A decoded item or its skip reason.

    Parameters
    ----------
    example : Example[Supervision[str]] | None
        The example the candidate became.
    reason : str | None
        Why the candidate was set aside.
    """

    example: Example[Supervision[str]] | None
    reason: str | None


class _Split[T]:
    """Repeatable access to one corpus split.

    Parameters
    ----------
    read : Callable[[], Iterator[T]]
        Return a fresh iterator over the values.
    count : int
        How many values a reading yields, settled at load.
    """

    __slots__ = ("_count", "_read")

    def __init__(self, read: Callable[[], Iterator[T]], count: int) -> None:
        self._read = read
        self._count = count

    def __iter__(self) -> Iterator[T]:
        """Return a fresh iterator over the values."""
        return self._read()

    def __len__(self) -> int:
        """Return how many values a reading yields."""
        return self._count


def _item(
    item_id: ItemId,
    expression: expression_records.Expression,
    tokens: tuple[Token, ...],
    split: str,
    provenance: Provenance | None,
) -> Item:
    """Return one datum of a Layers corpus.

    Parameters
    ----------
    item_id : ItemId
        The handle the datum is joined on.
    expression : expression_records.Expression
        The expression the datum reads.
    tokens : tuple[Token, ...]
        The expression's tokens.
    split : str
        The split the datum belongs to.
    provenance : Provenance | None
        Where the annotation came from, where the corpus says.

    Returns
    -------
    Item
        The datum.
    """
    return Item(
        item_id,
        tokens,
        text=expression.text,
        languages=tuple(expression.languages or ()),
        split=split,
        provenance=provenance,
    )


def _models_of[M](
    corpus: Corpus | JudgmentStudy, wanted: type[M]
) -> Iterator[tuple[str, M]]:
    """Yield every record of one type the pool holds, by AT-URI.

    Parameters
    ----------
    corpus : Corpus | JudgmentStudy
        The pooled record graph.
    wanted : type[M]
        The model type to select.

    Yields
    ------
    tuple of (str, M)
        The URI and the record, in URI order.
    """
    for uri in sorted(corpus.pool.uris()):
        record = corpus.pool.get(uri)

        if isinstance(record, wanted):
            yield uri, record


def _segmentations_of(
    corpus: Corpus,
) -> Mapping[str, tuple[segmentation_records.Segmentation, ...]]:
    """Return segmentations by expression URI.

    Parameters
    ----------
    corpus : Corpus
        The pooled record graph.

    Returns
    -------
    Mapping[str, tuple[segmentation_records.Segmentation, ...]]
        The segmentations, in URI order.
    """
    grouped: dict[str, list[segmentation_records.Segmentation]] = {}

    for _, record in _models_of(corpus, segmentation_records.Segmentation):
        grouped.setdefault(record.expression, []).append(record)

    return {uri: tuple(held) for uri, held in grouped.items()}


def _layers_of(
    corpus: Corpus,
) -> Mapping[str, tuple[tuple[str, annotation_records.AnnotationLayer], ...]]:
    """Return the annotation layers targeting each expression, by that URI.

    Parameters
    ----------
    corpus : Corpus
        The pooled record graph.

    Returns
    -------
    Mapping[str, tuple[tuple[str, annotation_records.AnnotationLayer], ...]]
        Each layer beside its own AT-URI, in URI order, which is what makes
        the id an item gets the same on a second load.
    """
    grouped: dict[str, list[tuple[str, annotation_records.AnnotationLayer]]] = {}

    for uri, record in _models_of(corpus, annotation_records.AnnotationLayer):
        grouped.setdefault(record.expression, []).append((uri, record))

    return {uri: tuple(held) for uri, held in grouped.items()}


def _responses_of(
    corpus: Corpus | JudgmentStudy,
) -> _ResponseIndex:
    """Return the responses recorded against each expression, by that URI.

    Parameters
    ----------
    corpus : Corpus | JudgmentStudy
        The pooled record graph.

    Returns
    -------
    _ResponseIndex
        Responses grouped by expression URI and reasons for skipped pair responses.
    """
    grouped: dict[str, list[_Response]] = {}
    skips: list[str] = []

    for _, record in _models_of(corpus, judgment_records.JudgmentSet):
        experiment = corpus.pool.get(record.experimentRef)
        held = (
            experiment
            if isinstance(experiment, judgment_records.ExperimentDef)
            else None
        )
        agent = None if record.agent is None else record.agent.id

        for judgment in record.judgments:
            target = judgment.item.recordRef

            if _is_pair_response(corpus, judgment, held):
                try:
                    pair = _response_pair(corpus, judgment)
                except RecordSkippedError as error:
                    skips.append(error.reason)

                    continue

                grouped.setdefault(pair.premise, []).append(
                    _Response(pair.premise, judgment, held, agent, pair)
                )

                continue

            if target is not None:
                grouped.setdefault(target, []).append(
                    _Response(item_id_of(target), judgment, held, agent, None)
                )

    return _ResponseIndex(
        {uri: tuple(responses) for uri, responses in grouped.items()},
        tuple(skips),
    )


def _is_pair_response(
    corpus: Corpus | JudgmentStudy,
    judgment: judgment_records.Judgment,
    experiment: judgment_records.ExperimentDef | None,
) -> bool:
    """Return whether a judgment names relational supervision."""
    target = judgment.item.recordRef
    record = None if target is None else corpus.pool.get(target)

    return (experiment is not None and experiment.measureType == "inference") or (
        isinstance(record, annotation_records.AnnotationLayer)
        and record.kind == "relation"
        and record.subkind is None
    )


def _response_pair(
    corpus: Corpus | JudgmentStudy,
    judgment: judgment_records.Judgment,
) -> RelationPair:
    """Return the pair annotation one elicited response names."""
    reference = judgment.item
    layer_uri = reference.recordRef

    if layer_uri is None:
        reason = "an elicited relation naming no pair layer"

        raise RecordSkippedError(reason)

    if reference.localId is not None or reference.knowledgeRef is not None:
        reason = "an elicited relation naming no exact pair reference"

        raise RecordSkippedError(reason)

    layer = corpus.pool.get(layer_uri)

    if not isinstance(layer, annotation_records.AnnotationLayer):
        reason = "an elicited relation naming no relation pair layer"

        raise RecordSkippedError(reason)

    object_id = reference.objectId

    if object_id is None:
        reason = "an elicited relation naming no pair annotation"

        raise RecordSkippedError(reason)

    annotations = tuple(
        annotation
        for annotation in layer.annotations
        if annotation.uuid.value == object_id.value
    )

    if len(annotations) != 1:
        reason = "an elicited relation naming no unique pair annotation"

        raise RecordSkippedError(reason)

    return relation_pair_of(layer, annotations[0])


def _membership_order(membership: corpus_records.Membership) -> tuple[bool, int, str]:
    """Return the key one split's memberships are read in.

    Parameters
    ----------
    membership : corpus_records.Membership
        The record to key.

    Returns
    -------
    tuple[bool, int, str]
        The missing-ordinal flag, ordinal, and expression URI used for sorting.
    """
    stated = membership.ordinal

    return (stated is None, 0 if stated is None else stated, membership.expressionRef)


def _memberships_of(
    corpus: Corpus,
    splits: tuple[str, ...],
) -> Mapping[str, tuple[corpus_records.Membership, ...]]:
    """Return the memberships of each split, in the order they are read.

    Parameters
    ----------
    corpus : Corpus
        The pooled record graph.
    splits : tuple[str, ...]
        The splits to group by.

    Returns
    -------
    Mapping[str, tuple[corpus_records.Membership, ...]]
        The memberships of each split, ordered by the ordinal the corpus
        states and then by the expression's URI.
    """
    grouped: dict[str, list[corpus_records.Membership]] = {
        split: [] for split in splits
    }

    for membership in corpus.memberships():
        held = None if membership.split is None else grouped.get(membership.split)

        if held is not None:
            held.append(membership)

    return {
        split: tuple(sorted(held, key=_membership_order))
        for split, held in grouped.items()
    }


def _unsplit_count(corpus: Corpus, splits: tuple[str, ...]) -> int:
    """Return how many memberships place their expression in no split.

    Parameters
    ----------
    corpus : Corpus
        The pooled record graph.
    splits : tuple[str, ...]
        The splits the reader hands out.

    Returns
    -------
    int
        Number of memberships outside ``splits``.
    """
    return sum(
        1 for membership in corpus.memberships() if membership.split not in splits
    )


def _unattached_relation_skips(
    corpus: Corpus,
    edges: Mapping[str, graph_records.GraphEdge],
    scheduled: frozenset[str],
) -> tuple[str, ...]:
    """Return failures for relation edges no expression item can visit."""
    skipped: list[str] = []

    for edge in edges.values():
        if edge.edgeType not in RELATION_NAMES:
            continue

        source = edge.source.recordRef
        source_record = None if source is None else corpus.pool.get(source)

        if not isinstance(source_record, expression_records.Expression):
            skipped.append("a relation whose endpoint is not an expression record")
        elif source not in scheduled:
            skipped.append("a relation whose endpoints are not members of one split")

    return tuple(skipped)


def _incident_edges(
    edges: Mapping[str, graph_records.GraphEdge],
) -> Mapping[str, tuple[tuple[str, graph_records.GraphEdge], ...]]:
    """Index graph edges by the records at either endpoint."""
    grouped: dict[str, list[tuple[str, graph_records.GraphEdge]]] = {}

    for edge_uri in sorted(edges):
        edge = edges[edge_uri]
        source = edge.source.recordRef
        target = edge.target.recordRef

        if source is not None:
            grouped.setdefault(source, []).append((edge_uri, edge))

        if target is not None and target != source:
            grouped.setdefault(target, []).append((edge_uri, edge))

    return MappingProxyType(
        {record_uri: tuple(held) for record_uri, held in grouped.items()}
    )


class LayersCorpusReader:
    """A Layers corpus backed by pooled records.

    Parameters
    ----------
    corpus : Corpus | JudgmentStudy
        The corpus or judgment study to read.
    labels : LabelMembership[str] | None
        The category labels admitted by the run. A tree layer's ontology type
        references are resolved against them, and a layer naming one under a
        reader given no inventory is set aside.
    comparison : ComparisonInventory[Recovered] | None
        The labels and rules every decoded derivation is classified against,
        once, at load. Absent where the load classifies nothing.
    _source_skips : tuple[str, ...]
        Skip reasons recorded while opening the source.
    """

    __slots__ = (
        "_accepted",
        "_arms",
        "_corpus",
        "_counts",
        "_edges",
        "_incident_edges",
        "_labels",
        "_layers",
        "_member_uris",
        "_memberships",
        "_nodes",
        "_outcomes",
        "_report",
        "_responses",
        "_segmentations",
        "_splits",
        "_study_members",
    )

    def __init__[Recovered](
        self,
        corpus: Corpus | JudgmentStudy,
        labels: LabelMembership[str] | None = None,
        *,
        comparison: ComparisonInventory[Recovered] | None = None,
        _source_skips: tuple[str, ...] = (),
    ) -> None:
        self._labels = labels

        if isinstance(corpus, JudgmentStudy):
            self._corpus = Corpus(corpus.pool)
            self._splits = (_JUDGMENT_SPLIT,)
            self._memberships = {}
            indexed = _responses_of(corpus)
            self._responses = indexed.grouped
            self._study_members = tuple(sorted(self._responses))
            self._member_uris = {
                _JUDGMENT_SPLIT: frozenset(
                    uri
                    for uri, _ in _models_of(
                        self._corpus, expression_records.Expression
                    )
                )
            }
            unsplit = 0
        else:
            self._corpus = corpus
            self._splits = corpus.splits()
            self._memberships = _memberships_of(corpus, self._splits)
            indexed = _responses_of(corpus)
            self._responses = indexed.grouped
            self._study_members = ()
            self._member_uris = {
                split: frozenset(
                    membership.expressionRef
                    for membership in self._memberships.get(split, ())
                )
                for split in self._splits
            }
            unsplit = _unsplit_count(corpus, self._splits)

        self._segmentations = _segmentations_of(self._corpus)
        self._layers = _layers_of(self._corpus)
        self._nodes = dict(_models_of(self._corpus, graph_records.GraphNode))
        self._edges = dict(_models_of(self._corpus, graph_records.GraphEdge))
        self._incident_edges = _incident_edges(self._edges)
        scheduled = (
            frozenset(self._study_members)
            if self._study_members
            else frozenset().union(*self._member_uris.values())
        )
        edge_skips = _unattached_relation_skips(
            self._corpus,
            self._edges,
            scheduled,
        )
        self._counts: dict[str, int] = {}
        arms: set[SupervisionArm] = set()
        skips: dict[str, int] = {}
        decoded: list[tuple[ItemId, Derivation[str]]] = []
        accepted_by_split: dict[str, tuple[Example[Supervision[str]], ...]] = {}

        for reason in (*_source_skips, *indexed.skips, *edge_skips):
            skips[reason] = skips.get(reason, 0) + 1

        seen = len(_source_skips) + len(indexed.skips) + len(edge_skips)

        for split in self._splits:
            emitted = 0
            accepted: list[Example[Supervision[str]]] = []

            for candidate in self._candidates(split):
                seen += 1

                if candidate.example is None:
                    reason = candidate.reason or LayersSkip.UNREADABLE_LAYER
                    skips[reason] = skips.get(reason, 0) + 1
                else:
                    emitted += 1
                    example = candidate.example
                    accepted.append(example)
                    supervision = example.supervision
                    arm = _arm_of(supervision)

                    if arm is not None:
                        arms.add(arm)

                    if isinstance(supervision, GoldDerivation):
                        decoded.append((example.item.item_id, supervision.derivation))

            self._counts[split] = emitted
            accepted_by_split[split] = tuple(accepted)

        self._accepted = MappingProxyType(accepted_by_split)

        if unsplit:
            seen += unsplit
            skips[LayersSkip.UNSPLIT] = unsplit

        self._arms = frozenset(arms)
        emitted_total = sum(self._counts.values())

        if comparison is None:
            self._outcomes = None
            self._report = LoadReport(seen, emitted_total, skips)
        else:
            reachability = reachability_report(
                comparison.labels,
                comparison.rules,
                [derivation for _, derivation in decoded],
            )
            self._outcomes = tuple(
                zip(
                    (item_id for item_id, _ in decoded),
                    reachability.outcomes,
                    strict=True,
                )
            )
            self._report = LoadReport(
                seen,
                emitted_total,
                skips,
                AnalysisCounts(
                    reachability.checked_derivations,
                    reachability.reachable_derivations,
                ),
                reachability,
            )

    @property
    def name(self) -> str:
        """Return the name this reader is registered and reported under."""
        return READER_NAME

    @property
    def grammar(self) -> None:
        """Return the grammar, which Layers does not declare."""
        return None

    @property
    def lexicon(self) -> None:
        """Return None because Layers does not supply lexical axioms."""
        return None

    def splits(self) -> tuple[str, ...]:
        """Return the splits the corpus's memberships place its expressions in.

        Returns
        -------
        tuple[str, ...]
            Each split's slug, in the order the corpus lists them.
        """
        return self._splits

    def items(self, split: str, /) -> _Split[Item]:
        """Return the items of one split.

        Parameters
        ----------
        split : str
            The split's slug.

        Returns
        -------
        _Split[Item]
            The re-readable items in example order.
        """
        return _Split(lambda: self._items(split), self._counts.get(split, 0))

    def examples(self, split: str, /) -> _Split[Example[Supervision[str]]]:
        """Return the items of one split beside what the corpus says about them.

        Parameters
        ----------
        split : str
            The split's slug.

        Returns
        -------
        _Split[Example[Supervision[str]]]
            The re-readable examples in membership order.
        """
        return _Split(lambda: self._examples(split), self._counts.get(split, 0))

    def supervision_kinds(self) -> frozenset[SupervisionArm]:
        """Return the arms of the supervision union this load emitted.

        Returns
        -------
        frozenset[SupervisionArm]
            The arms, named by their own types. A corpus that records nothing
            reports the unlabeled arm alone, which is what refuses a featurizer
            with nothing to learn from it.
        """
        return self._arms

    def report(self) -> LoadReport:
        """Return the corpus load report.

        Returns
        -------
        LoadReport
            The report of the load this reader was built by.
        """
        return self._report

    def reachability(self) -> tuple[tuple[ItemId, DerivationOutcome[str]], ...] | None:
        """Return each decoded derivation's outcome beside its item.

        Returns
        -------
        tuple[tuple[ItemId, DerivationOutcome[str]], ...] | None
            One pair per decoded derivation, in the order the report holds
            its outcomes, or None where the reader was given no comparison
            inventory.
        """
        return self._outcomes

    def _items(self, split: str) -> Iterator[Item]:
        """Yield the items of one split.

        Parameters
        ----------
        split : str
            The split's slug.

        Yields
        ------
        Item
            Each item, in the order the examples run.
        """
        for example in self._examples(split):
            yield example.item

    def _examples(self, split: str) -> Iterator[Example[Supervision[str]]]:
        """Yield the examples of one split, passing over what was set aside.

        Parameters
        ----------
        split : str
            The split's slug.

        Yields
        ------
        Example
            Each example, in membership order.
        """
        yield from self._accepted.get(split, ())

    def _candidates(self, split: str) -> Iterator[_Candidate]:
        """Yield every record of one split the reader considered.

        Parameters
        ----------
        split : str
            The split's slug.

        Yields
        ------
        _Candidate
            Each candidate, in membership order.
        """
        if self._study_members:
            for uri in self._study_members:
                yield from self._for_member(uri, split)

            return

        memberships = self._memberships.get(split, ())
        dataset = Dataset.streaming(
            source=lambda: iter(memberships), model=corpus_records.Membership
        )

        for membership in dataset:
            yield from self._for_member(membership.expressionRef, split)

    def _for_member(self, uri: str, split: str) -> Iterator[_Candidate]:
        """Yield every candidate one membership contributes.

        Parameters
        ----------
        uri : str
            The expression record's AT-URI.
        split : str
            The split's slug, which the items record.

        Yields
        ------
        _Candidate
            One per thing the corpus says about the expression, or one under
            the unlabeled arm where it says nothing.
        """
        expression = self._corpus.pool.get(uri)

        if not isinstance(expression, expression_records.Expression):
            yield _Candidate(None, LayersSkip.UNHELD_EXPRESSION)

            return

        try:
            tokens = tokens_of(expression.text, self._segmentations.get(uri, ()))
        except RecordSkippedError as error:
            yield _Candidate(None, error.reason)

            return

        reading = TokenReading(
            tokenization_of(self._segmentations.get(uri, ())), len(tokens)
        )
        said = False

        for candidate in self._analyzed(uri, expression, tokens, split, reading):
            said = True

            yield candidate

        for candidate in self._graphed(uri, expression, tokens, split):
            said = True

            yield candidate

        for candidate in self._responded(uri, expression, tokens, split):
            said = True

            yield candidate

        if not said:
            item = _item(item_id_of(uri), expression, tokens, split, None)

            yield _Candidate(Example(item, Unlabeled()), None)

    def _graphed(
        self,
        uri: str,
        expression: expression_records.Expression,
        tokens: tuple[Token, ...],
        split: str,
    ) -> Iterator[_Candidate]:
        """Yield one candidate per graph edge attaching a graph to an expression.

        Parameters
        ----------
        uri : str
            The expression's AT-URI.
        expression : expression_records.Expression
            The expression record.
        tokens : tuple[Token, ...]
            The expression's tokens.
        split : str
            The split's slug.

        Yields
        ------
        _Candidate
            One per attaching edge, in the order of the edges' own URIs.
        """
        for edge_uri, edge in self._incident_edges.get(uri, ()):
            source = edge.source.recordRef
            target = edge.target.recordRef
            source_record = None if source is None else self._corpus.pool.get(source)
            target_record = None if target is None else self._corpus.pool.get(target)
            is_expression_pair = isinstance(
                source_record, expression_records.Expression
            ) and isinstance(target_record, expression_records.Expression)

            if edge.edgeType in RELATION_NAMES or is_expression_pair:
                if source != uri:
                    continue

                try:
                    reading = RELATION_LIFT.edge(edge)
                except RecordSkippedError as error:
                    yield _Candidate(None, error.reason)

                    continue

                reason = self._pair_reason(reading.pair, split)

                if reason is not None:
                    yield _Candidate(None, reason)

                    continue

                item = _item(
                    reading.pair.premise,
                    expression,
                    tokens,
                    split,
                    None,
                )

                yield _Candidate(Example(item, reading.supervision), None)

                continue

            start = graph_start(edge, uri)

            if start is None:
                continue

            try:
                supervision = GRAPH_LIFT.read(
                    GraphContext(start, self._nodes, self._graph_edges(start))
                )
            except RecordSkippedError as error:
                yield _Candidate(None, error.reason)

                continue
            except ValueError:
                yield _Candidate(None, LayersSkip.UNREADABLE_GRAPH)

                continue

            item = _item(item_id_of(edge_uri), expression, tokens, split, None)

            yield _Candidate(Example(item, supervision), None)

    def _graph_edges(self, start: str) -> Mapping[str, graph_records.GraphEdge]:
        """Return the edges in the graph-node component reached from ``start``."""
        found = {start}
        pending = [start]
        held: dict[str, graph_records.GraphEdge] = {}

        while pending:
            here = pending.pop()

            for edge_uri, edge in self._incident_edges.get(here, ()):
                held[edge_uri] = edge

                for endpoint in (edge.source.recordRef, edge.target.recordRef):
                    if endpoint in self._nodes and endpoint not in found:
                        found.add(endpoint)
                        pending.append(endpoint)

        return held

    def _analyzed(
        self,
        uri: str,
        expression: expression_records.Expression,
        tokens: tuple[Token, ...],
        split: str,
        reading: TokenReading,
    ) -> Iterator[_Candidate]:
        """Yield one candidate per annotation layer targeting an expression.

        Parameters
        ----------
        uri : str
            The expression's AT-URI.
        expression : expression_records.Expression
            The expression record.
        tokens : tuple[Token, ...]
            The expression's tokens.
        split : str
            The split name.
        reading : TokenReading
            The tokenization used to read the expression.

        Yields
        ------
        _Candidate
            Candidates in annotation layer URI order.
        """
        for layer_uri, layer in self._layers.get(uri, ()):
            if layer.kind == "relation" and layer.subkind is None:
                yield from self._related(layer, expression, tokens, split)

                continue

            try:
                supervision = supervision_of(layer, reading, self._labels)
            except RecordSkippedError as error:
                yield _Candidate(None, error.reason)

                continue
            except ValueError:
                yield _Candidate(None, LayersSkip.UNREADABLE_LAYER)

                continue

            provenance = provenance_of(
                layer.sourceMethod,
                layer.licensing,
                layer.eprintRefs,
                layer.reproducibility,
            )
            item = _item(item_id_of(layer_uri), expression, tokens, split, provenance)

            yield _Candidate(Example(item, supervision), None)

    def _related(
        self,
        layer: annotation_records.AnnotationLayer,
        expression: expression_records.Expression,
        tokens: tuple[Token, ...],
        split: str,
    ) -> Iterator[_Candidate]:
        """Yield each annotated relation from one pair layer."""
        for annotation in layer.annotations:
            try:
                reading = RELATION_LIFT.layer(layer, annotation)
            except RecordSkippedError as error:
                yield _Candidate(None, error.reason)

                continue

            if reading is None:
                continue

            reason = self._pair_reason(reading.pair, split)

            if reason is not None:
                yield _Candidate(None, reason)

                continue

            provenance = provenance_of(
                layer.sourceMethod,
                layer.licensing,
                layer.eprintRefs,
                layer.reproducibility,
            )
            item = _item(reading.pair.premise, expression, tokens, split, provenance)

            yield _Candidate(Example(item, reading.supervision), None)

    def _responded(
        self,
        uri: str,
        expression: expression_records.Expression,
        tokens: tuple[Token, ...],
        split: str,
    ) -> Iterator[_Candidate]:
        """Yield one candidate per response recorded against an expression.

        Parameters
        ----------
        uri : str
            The expression's AT-URI.
        expression : expression_records.Expression
            The expression record.
        tokens : tuple[Token, ...]
            The expression's tokens.
        split : str
            The split's slug.

        Yields
        ------
        _Candidate
            One per response, in the order of the judgment sets' own URIs.
        """
        for response in self._responses.get(uri, ()):
            if response.experiment is None:
                yield _Candidate(None, LayersSkip.UNHELD_EXPERIMENT)

                continue

            try:
                if response.pair is None:
                    supervision = response_of(
                        response.judgment,
                        response.experiment,
                        annotator_of(response.annotator),
                    )
                else:
                    reason = self._pair_reason(response.pair, split)

                    if reason is not None:
                        yield _Candidate(None, reason)

                        continue

                    supervision = pair_response_of(
                        response.judgment,
                        response.experiment,
                        response.pair,
                        annotator_of(response.annotator),
                    )
            except RecordSkippedError as error:
                yield _Candidate(None, error.reason)

                continue

            item = _item(response.item_id, expression, tokens, split, None)

            yield _Candidate(Example(item, supervision), None)

    def _pair_reason(self, pair: RelationPair, split: str) -> str | None:
        """Return why a pair is not two expression items in one split."""
        premise = self._corpus.pool.get(pair.premise)
        hypothesis = self._corpus.pool.get(pair.hypothesis)

        if not isinstance(premise, expression_records.Expression) or not isinstance(
            hypothesis, expression_records.Expression
        ):
            return "a relation whose endpoint is not an expression record"

        members = self._member_uris.get(split, frozenset())

        if pair.premise not in members or pair.hypothesis not in members:
            return "a relation whose endpoints are not members of one split"

        return None


def reader_at[Recovered](
    directory: Path,
    *,
    corpus_uri: str | None = None,
    labels: LabelMembership[str] | None = None,
    comparison: ComparisonInventory[Recovered] | None = None,
) -> LayersCorpusReader:
    """Read a Layers repository into a corpus reader."""
    corpus, skips = _repository_at(directory, corpus_uri)

    return LayersCorpusReader(
        corpus,
        labels,
        comparison=comparison,
        _source_skips=skips,
    )
