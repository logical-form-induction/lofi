"""Builds the committed Layers sample corpus."""

import json
import tomllib
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING

from lairs.records import annotation, corpus, defs, expression, graph, judgment
from lairs.records import segmentation as segmentation_records
from lairs.store.repository import Repository

if TYPE_CHECKING:
    import didactic.api as dx

AUTHORITY = "did:plc:lofisample"
CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)

REPOSITORY_DIR = "repository"
MANIFEST_FILE = "MANIFEST.json"

LICENSING = defs.Licensing(
    expression="CC0-1.0",
    licenses=(
        defs.LicenseRef(
            spdx="CC0-1.0",
            name="Creative Commons Zero v1.0 Universal",
        ),
    ),
)


def uri_of(collection: str, rkey: str) -> str:
    """Return the AT-URI for a sample record.

    Parameters
    ----------
    collection
        The record type's namespace identifier.
    rkey
        The record's key within that collection.

    Returns
    -------
    str
        The AT-URI, which is the key the repository stores the record under.
    """
    return f"at://{AUTHORITY}/{collection}/{rkey}"


def tokens_of(
    text: str,
    words: tuple[str, ...],
) -> tuple[segmentation_records.Token, ...]:
    """Return whitespace-delimited tokens with UTF-8 byte spans.

    Parameters
    ----------
    text
        The sentence the words were taken from.
    words
        The words, in the order they appear.

    Returns
    -------
    tuple of pub.layers.segmentation.Token
        One token per word, with its position, form and byte span.
    """
    encoded = text.encode()
    found: list[segmentation_records.Token] = []
    offset = 0

    for index, word in enumerate(words):
        start = encoded.index(word.encode(), offset)
        end = start + len(word.encode())
        found.append(
            segmentation_records.Token(
                tokenIndex=index,
                text=word,
                textSpan=defs.Span(byteStart=start, byteEnd=end),
            )
        )
        offset = end

    return tuple(found)


def sentence(rkey: str, text: str, language: str) -> expression.Expression:
    """Return a sentence record for the sample corpus.

    Parameters
    ----------
    rkey
        The record key, which is also the expression's own id.
    text
        The sentence.
    language
        The BCP 47 tag the sentence is written in.

    Returns
    -------
    pub.layers.expression.Expression
        The expression record.
    """
    return expression.Expression(
        id=rkey,
        kind="sentence",
        createdAt=CREATED_AT,
        text=text,
        languages=(language,),
    )


def segmentation_of(
    expression_rkey: str,
    text: str,
    words: tuple[str, ...],
) -> segmentation_records.Segmentation:
    """Return a sentence's whitespace segmentation.

    Parameters
    ----------
    expression_rkey
        The record key of the sentence being segmented.
    text
        The sentence.
    words
        Its words, in order.

    Returns
    -------
    pub.layers.segmentation.Segmentation
        A segmentation record holding one tokenization.
    """
    target = uri_of("pub.layers.expression.expression", expression_rkey)

    return segmentation_records.Segmentation(
        createdAt=CREATED_AT,
        expression=target,
        licensing=LICENSING,
        tokenizations=(
            segmentation_records.Tokenization(
                uuid=defs.Uuid(value=f"tokenization-{expression_rkey}"),
                kind="whitespace",
                expressionRef=target,
                tokens=tokens_of(text, words),
            ),
        ),
    )


def membership_of(
    expression_rkey: str,
    split: str,
    ordinal: int,
) -> corpus.Membership:
    """Return a corpus membership record for a sentence.

    Parameters
    ----------
    expression_rkey
        The record key of the sentence.
    split
        The portion of the corpus the sentence belongs to.
    ordinal
        Its position within that portion.

    Returns
    -------
    pub.layers.corpus.Membership
        The membership record.
    """
    return corpus.Membership(
        corpusRef=uri_of("pub.layers.corpus.corpus", "sample"),
        expressionRef=uri_of("pub.layers.expression.expression", expression_rkey),
        createdAt=CREATED_AT,
        split=split,
        ordinal=ordinal,
    )


def _node(
    uuid: str,
    label: str,
    *,
    parent: str | None = None,
    children: tuple[str, ...] = (),
    token_index: int | None = None,
) -> annotation.Annotation:
    """Return a node in the sample categorial analysis.

    Parameters
    ----------
    uuid
        The node's handle within its layer.
    label
        Its category.
    parent
        The handle of the node it was combined into, absent for the root.
    children
        The handles of the nodes it was built from, empty for a leaf.
    token_index
        The token a leaf covers, absent for an internal node.

    Returns
    -------
    pub.layers.annotation.Annotation
        The node record.
    """
    return annotation.Annotation(
        uuid=defs.Uuid(value=uuid),
        label=label,
        parentId=None if parent is None else defs.Uuid(value=parent),
        childIds=tuple(defs.Uuid(value=child) for child in children),
        tokenIndex=token_index,
    )


def tree_layer(expression_rkey: str) -> annotation.AnnotationLayer:
    """Return the sample sentence's categorial derivation.

    Parameters
    ----------
    expression_rkey
        The record key of the sentence analyzed.

    Returns
    -------
    pub.layers.annotation.AnnotationLayer
        A layer of kind ``tree`` and subkind ``ccg``.
    """
    return annotation.AnnotationLayer(
        createdAt=CREATED_AT,
        expression=uri_of("pub.layers.expression.expression", expression_rkey),
        kind="tree",
        subkind="ccg",
        sourceMethod="manual-native",
        licensing=LICENSING,
        tokenizationId=defs.Uuid(value=f"tokenization-{expression_rkey}"),
        annotations=(
            _node("tree-0", "S", children=("tree-1", "tree-4")),
            _node("tree-1", "NP", parent="tree-0", children=("tree-2", "tree-3")),
            _node("tree-2", "NP/N", parent="tree-1", token_index=0),
            _node("tree-3", "N", parent="tree-1", token_index=1),
            _node("tree-4", "S\\NP", parent="tree-0", token_index=2),
        ),
    )


def dependency_layer(expression_rkey: str) -> annotation.AnnotationLayer:
    """Return the sample sentence's dependency analysis.

    Parameters
    ----------
    expression_rkey
        The record key of the sentence analyzed.

    Returns
    -------
    pub.layers.annotation.AnnotationLayer
        A layer of kind ``relation`` and subkind ``dependency``.
    """
    return annotation.AnnotationLayer(
        createdAt=CREATED_AT,
        expression=uri_of("pub.layers.expression.expression", expression_rkey),
        kind="relation",
        subkind="dependency",
        sourceMethod="manual-native",
        licensing=LICENSING,
        tokenizationId=defs.Uuid(value=f"tokenization-{expression_rkey}"),
        annotations=(
            annotation.Annotation(
                uuid=defs.Uuid(value="dep-0"),
                label="nsubj",
                headIndex=2,
                targetIndex=1,
            ),
            annotation.Annotation(
                uuid=defs.Uuid(value="dep-1"),
                label="det",
                headIndex=1,
                targetIndex=0,
            ),
        ),
    )


def token_tag_layer(expression_rkey: str) -> annotation.AnnotationLayer:
    """Return the sample sentence's token categories.

    Parameters
    ----------
    expression_rkey
        The record key of the sentence tagged.

    Returns
    -------
    pub.layers.annotation.AnnotationLayer
        A layer of kind ``token-tag``.
    """
    return annotation.AnnotationLayer(
        createdAt=CREATED_AT,
        expression=uri_of("pub.layers.expression.expression", expression_rkey),
        kind="token-tag",
        subkind="ccg",
        sourceMethod="manual-native",
        licensing=LICENSING,
        tokenizationId=defs.Uuid(value=f"tokenization-{expression_rkey}"),
        annotations=(
            annotation.Annotation(
                uuid=defs.Uuid(value="tag-0"),
                label="NP/N",
                tokenIndex=0,
            ),
            annotation.Annotation(
                uuid=defs.Uuid(value="tag-1"),
                label="N",
                tokenIndex=1,
            ),
            annotation.Annotation(
                uuid=defs.Uuid(value="tag-2"),
                label="S\\NP",
                tokenIndex=2,
            ),
        ),
    )


def records() -> dict[str, dx.Model]:
    """Return all sample records keyed by AT-URI.

    Returns
    -------
    dict
        Twenty records: one corpus, four sentences and their memberships,
        three segmentations, three annotation layers, a graph node and the edge
        that ties it to a sentence, and an experiment with the two annotators'
        response sets.
    """
    english = "the child sleeps"
    spanish = "el niño duerme"
    mandarin = "孩子 睡觉"
    repeated = "boats boats boats"

    english_uri = uri_of("pub.layers.expression.expression", "en-sleeps")
    graph_node_uri = uri_of("pub.layers.graph.graphNode", "sleep-situation")
    experiment_uri = uri_of("pub.layers.judgment.experimentDef", "acceptability")
    repeated_uri = uri_of("pub.layers.expression.expression", "en-repeated")

    built: dict[str, dx.Model] = {
        uri_of("pub.layers.corpus.corpus", "sample"): corpus.Corpus(
            name="LoFI sample",
            createdAt=CREATED_AT,
            description=(
                "Four sentences written for the test suite, in three languages, "
                "with a categorial analysis, a dependency analysis, per-token "
                "categories, a meaning graph and an acceptability study."
            ),
            languages=("en", "es", "zh"),
            licensing=LICENSING,
        ),
        uri_of("pub.layers.expression.expression", "en-sleeps"): sentence(
            "en-sleeps", english, "en"
        ),
        uri_of("pub.layers.expression.expression", "es-duerme"): sentence(
            "es-duerme", spanish, "es"
        ),
        uri_of("pub.layers.expression.expression", "zh-shuijiao"): sentence(
            "zh-shuijiao", mandarin, "zh"
        ),
        repeated_uri: sentence("en-repeated", repeated, "en"),
        uri_of("pub.layers.corpus.membership", "en-sleeps"): membership_of(
            "en-sleeps", "train", 0
        ),
        uri_of("pub.layers.corpus.membership", "es-duerme"): membership_of(
            "es-duerme", "train", 1
        ),
        uri_of("pub.layers.corpus.membership", "zh-shuijiao"): membership_of(
            "zh-shuijiao", "dev", 0
        ),
        uri_of("pub.layers.corpus.membership", "en-repeated"): membership_of(
            "en-repeated", "test", 0
        ),
        uri_of("pub.layers.segmentation.segmentation", "en-sleeps"): segmentation_of(
            "en-sleeps", english, ("the", "child", "sleeps")
        ),
        uri_of("pub.layers.segmentation.segmentation", "es-duerme"): segmentation_of(
            "es-duerme", spanish, ("el", "niño", "duerme")
        ),
        uri_of("pub.layers.segmentation.segmentation", "zh-shuijiao"): segmentation_of(
            "zh-shuijiao", mandarin, ("孩子", "睡觉")
        ),
        uri_of("pub.layers.annotation.annotationLayer", "en-sleeps-tree"): tree_layer(
            "en-sleeps"
        ),
        uri_of(
            "pub.layers.annotation.annotationLayer", "en-sleeps-dependency"
        ): dependency_layer("en-sleeps"),
        uri_of(
            "pub.layers.annotation.annotationLayer", "en-sleeps-token-tags"
        ): token_tag_layer("en-sleeps"),
        graph_node_uri: graph.GraphNode(
            nodeType="situation",
            createdAt=CREATED_AT,
            label="sleep",
            properties=defs.FeatureMap(
                entries=(
                    defs.Feature(key="tense", value="present"),
                    defs.Feature(key="polarity", value="positive"),
                )
            ),
        ),
        uri_of("pub.layers.graph.graphEdge", "en-sleeps-denotes"): graph.GraphEdge(
            source=defs.ObjectRef(recordRef=english_uri),
            target=defs.ObjectRef(recordRef=graph_node_uri),
            edgeType="denotes",
            createdAt=CREATED_AT,
            label="the situation the sentence describes",
        ),
        experiment_uri: judgment.ExperimentDef(
            name="LoFI sample acceptability rating",
            createdAt=CREATED_AT,
            description="A seven-point rating of how natural a sentence sounds.",
            measureType="acceptability",
            taskType="ordinal-scale",
            scaleMin=1,
            scaleMax=7,
            stimulusExpressionRefs=(repeated_uri,),
            languages=("en",),
            licensing=LICENSING,
        ),
    }

    for annotator, rating, response_time in (("a", 2, 5400), ("b", 6, 3100)):
        built[uri_of("pub.layers.judgment.judgmentSet", f"annotator-{annotator}")] = (
            judgment.JudgmentSet(
                experimentRef=experiment_uri,
                createdAt=CREATED_AT,
                agent=defs.AgentRef(
                    id=f"sample-annotator-{annotator}",
                    name=f"sample annotator {annotator.upper()}",
                ),
                judgments=(
                    judgment.Judgment(
                        item=defs.ObjectRef(recordRef=repeated_uri),
                        scalarValue=rating,
                        responseTimeMs=response_time,
                    ),
                ),
            )
        )

    return built


def lexicon_tree_hash() -> str:
    """Return the vendored Layers lexicon tree hash.

    Returns
    -------
    str
        The hex digest recorded in the vendoring manifest.

    Raises
    ------
    TypeError
        If the manifest records no tree hash.
    """
    manifest = files("lairs.lexicons").joinpath("MANIFEST.toml").read_bytes()
    provenance = tomllib.loads(manifest.decode())["provenance"]
    recorded = provenance.get("lexicon_tree_hash")

    if not isinstance(recorded, str):
        msg = "the vendored Layers lexicon manifest records no tree hash"
        raise TypeError(msg)

    return recorded


def write(directory: Path) -> Path:
    """Write the sample repository and manifest to a directory.

    Parameters
    ----------
    directory
        Where the repository and the manifest go. It is created if absent, and
        any repository already there is replaced.

    Returns
    -------
    pathlib.Path
        The repository directory.
    """
    repository_path = directory / REPOSITORY_DIR
    rmtree(repository_path, ignore_errors=True)
    repository = Repository.init(repository_path)

    for uri, record in sorted(records().items()):
        repository.save(uri, record)

    repository.commit("Write the LoFI sample corpus")
    (directory / MANIFEST_FILE).write_text(
        json.dumps(
            {
                "authority": AUTHORITY,
                "lexicon_tree_hash": lexicon_tree_hash(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return repository_path


if __name__ == "__main__":
    write(Path(__file__).parent)
