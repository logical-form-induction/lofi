"""An active-learning experiment runs from its documents.

A trained grammaticality point and a pinned masked language model are the only
artifacts the case builds; everything else is declared. The item-construction
document fills bead templates from a bead lexicon and a lexicon the masked
model proposes, the judgment-fit document
declares the response model refitted by MAP each round, and the active-learning
document names both beside the trained point whose root masses are the
predictors and the bead collector that fields each batch. Simulated annotators
answer every batch the run writes, rating each item by the tercile of the
trained point's root mass, so their judgments agree with the point. The run is
driven through the ``lofi active-learn`` command and, as a pilot experiment,
through the launcher; each stops awaiting a batch, resumes once its responses
are ingested, and records its rounds as rows. The same document, its collector
replaced by each simulated collector, completes in one command with no batch
fielded, recording whether the declared agreement, the round bound or the
exhausted pool ended it. Started instead from a collected per-stratum draw, a
simulated run reads no bead response file at all.
"""

import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Final
from urllib.parse import quote

import pyarrow.parquet as pq
import pytest
import torch
from tests.acceptance.driver import lofi, train
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast

from lofi.eval import UNFROZEN
from lofi.run import main
from lofi.run._active_inputs import pool_readouts, read_pool
from lofi.run.launching import ExperimentLaunching
from lofi.run.operations import TRAINING_OPERATIONS
from lofi.spec.acquisition import PointReadoutSettings, TokenizerSettings

if TYPE_CHECKING:
    from lofi._types import JsonValue

LABELS: Final = ("low", "mid", "high")
ANNOTATORS: Final = ("annotator-a", "annotator-b", "annotator-c")
PREDICATES: Final = ("word-2", "word-6", "word-1", "word-5", "word-3", "word-7")
PAD: Final = "[PAD]"
UNKNOWN: Final = "[UNK]"
CLASSIFIER: Final = "[CLS]"
SEPARATOR: Final = "[SEP]"
MASK: Final = "[MASK]"
GIT: Final = shutil.which("git") or "git"
MAX_ROUNDS: Final = 2
"""The round bound the documented example and the pilot experiment declare."""
AGREEMENT: Final = 0.5
"""The agreement a simulated run stops at, compared with the estimate."""
LATENT_RANGE: Final[dict[str, JsonValue]] = {"lower": -3.0, "upper": 3.0}
"""The latent interval a drawn design's cutpoints divide evenly."""
EXAMPLE: Final = (
    Path(__file__).resolve().parents[2]
    / "docs/examples/models-training/active-learning"
)
"""The documented active-learning example, whose documents every case runs."""


def _masked_model(directory: Path, /) -> None:
    """Write a local masked model that prefers the fixture predicates in order."""
    vocabulary = {
        token: position
        for position, token in enumerate(
            (
                PAD,
                UNKNOWN,
                CLASSIFIER,
                SEPARATOR,
                MASK,
                *(f"word-{index}" for index in range(8)),
            )
        )
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token=UNKNOWN))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=PAD,
        unk_token=UNKNOWN,
        cls_token=CLASSIFIER,
        sep_token=SEPARATOR,
        mask_token=MASK,
    ).save_pretrained(directory)
    model = BertForMaskedLM(
        BertConfig(
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=8,
            vocab_size=len(vocabulary),
            max_position_embeddings=16,
            type_vocab_size=2,
            pad_token_id=vocabulary[PAD],
        )
    )
    with torch.no_grad():
        model.cls.predictions.decoder.weight.zero_()
        bias = torch.full((len(vocabulary),), -50.0)
        for rank, predicate in enumerate(PREDICATES):
            bias[vocabulary[predicate]] = 10.0 - rank
        model.cls.predictions.bias.copy_(bias)
    model.save_pretrained(directory)


def _documents(folder: Path, trained: Path, cache: Path, /) -> None:
    """Copy the documented example documents, caching their pinned masked model.

    The masked model is written where the declared offline cache holds the
    pinned hub revision the item-construction document names, so the generator
    reads it from the cache without fetching.
    """
    shutil.copytree(EXAMPLE, folder)
    shutil.copytree(trained, folder / "trained")
    items = json.loads((folder / "items.json").read_text(encoding="utf-8"))
    generator = items["generators"][0]
    checkpoint = generator["checkpoint"]
    _masked_model(
        cache
        / "snapshots"
        / quote(checkpoint["repository"], safe="")
        / checkpoint["revision"]
    )
    generator["artifacts"]["root"] = str(cache)
    (folder / "items.json").write_text(
        json.dumps(items), encoding="utf-8", newline="\n"
    )


def _annotators(folder: Path, /) -> dict[str, int]:
    """Return the simulated annotators' response to every pool item."""
    pool = read_pool(folder / "items.json").items
    columns = pool_readouts(
        folder / "trained",
        pool,
        {"words": "sentence"},
        TokenizerSettings("whitespace", "en", None),
        {"p_gram": PointReadoutSettings("sentences", "log-mass")},
    )
    ranked = sorted(pool, key=lambda item: columns[str(item.id)]["p_gram"])
    responses = {
        str(item.id): position * len(LABELS) // len(ranked)
        for position, item in enumerate(ranked)
    }

    def members(structure: str, /) -> tuple[str, ...]:
        return tuple(
            str(held.id)
            for held in ranked
            if held.item_metadata["structure"] == structure
        )

    initial = [
        item
        for structure in ("transitive", "intransitive")
        for item in (*members(structure)[:2], *members(structure)[-2:])
    ]
    (folder / "initial.jsonl").write_text(
        _records({item: responses[item] for item in initial}),
        encoding="utf-8",
        newline="\n",
    )
    return responses


def _records(responses: dict[str, int], /) -> str:
    """Return every annotator's bead record for each answered item."""
    return "".join(
        json.dumps(
            {
                "annotator_id": annotator,
                "item_id": item,
                "question_name": "acceptability",
                "response_label": LABELS[response],
            }
        )
        + "\n"
        for item, response in responses.items()
        for annotator in ANNOTATORS
    )


def _answer(batch: Path, responses: dict[str, int], /) -> None:
    """Ingest the simulated annotators' responses beside a written batch."""
    items = tuple(
        json.loads(line)["id"]
        for line in batch.read_text(encoding="utf-8").splitlines()
    )
    (batch.parent / "responses.jsonl").write_text(
        _records({item: responses[item] for item in items}),
        encoding="utf-8",
        newline="\n",
    )


REGISTERED_DESIGN: Final = "megaacceptability"
"""An installed design tag whose document the registered-design case writes."""


def _registered(
    folder: Path, responses: dict[str, int], monkeypatch: pytest.MonkeyPatch, /
) -> None:
    """Declare the simulated annotators' answers to the pool as a bead design."""
    (folder / "designs").mkdir()
    labels = json.loads((folder / "encoding.json").read_text(encoding="utf-8"))[
        "labels"
    ]
    (folder / "answers.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "annotator_id": annotator,
                    "item_id": item,
                    "question_name": "acceptability",
                    "response_label": labels[response],
                }
            )
            + "\n"
            for item, response in responses.items()
            for annotator in ANNOTATORS
        ),
        encoding="utf-8",
        newline="\n",
    )
    design: dict[str, JsonValue] = {
        "source": "bead",
        "items": "items.json",
        "judgments": "answers.jsonl",
        "encoding": "encoding.json",
        "factors": {},
    }
    (folder / "designs" / f"{REGISTERED_DESIGN}.json").write_text(
        json.dumps(design), encoding="utf-8", newline="\n"
    )
    monkeypatch.setenv("LOFI_DATA_ROOT", str(folder))


def _inputs(tmp_path: Path, folder: Path, /) -> dict[str, int]:
    """Train the point, write the masked model and documents, and answer."""
    trained = train("grammaticality-point.yaml", tmp_path / "store")
    _documents(folder, trained.directory, tmp_path / "artifacts")
    return _annotators(folder)


def _collected_rounds(statuses: list[str], max_rounds: int, /) -> int:
    """Return how many rounds a run awaited before it completed.

    A run awaits the responses to each round's batch and then completes, after
    at least one round and at most the declared bound.
    """
    rounds = len(statuses) - 1
    assert 1 <= rounds <= max_rounds
    assert statuses == ["awaiting-responses"] * rounds + ["completed"]
    return rounds


def _stops(rounds: int, max_rounds: int, /) -> set[str]:
    """Return why a run that collected ``rounds`` rounds may have stopped.

    The declared asymptote needs more rounds than the bound allows, so a run
    that stops short of the bound stops because its remaining pool can no
    longer fill a batch; one that reaches the bound may also have exhausted it.
    """
    if rounds < max_rounds:
        return {"pool-exhausted"}
    return {"max-rounds", "pool-exhausted"}


def test_the_command_collects_every_round_a_document_declares(tmp_path: Path) -> None:
    folder = tmp_path / "documents"
    responses = _inputs(tmp_path, folder)
    out = tmp_path / "out"
    argv = ["active-learn", str(folder / "active-learning.yaml"), "--out", str(out)]
    reported: list[str] = []

    for _ in range(3):
        status = lofi()(argv)
        assert status == 0
        manifest = json.loads(
            (out / "active-learning.json").read_text(encoding="utf-8")
        )
        reported.append(manifest["status"])
        if manifest["awaiting"] is None:
            break
        _answer(out / "rounds" / str(manifest["awaiting"]) / "batch.jsonl", responses)

    rows = json.loads((out / "rows.json").read_text(encoding="utf-8"))["rows"]
    rounds = _collected_rounds(reported, manifest["document"]["max_rounds"])

    assert {row["round"] for row in rows} == set(range(1, rounds + 1))
    assert manifest["stop_reason"] in _stops(rounds, manifest["document"]["max_rounds"])
    assert all(row["task"] == "active-learning" for row in rows)
    ranks = [row for row in rows if row["metric"] == "rank-correlation"]
    assert all(row["value"] > 0.0 for row in ranks)
    assert all(row["lower"] <= row["value"] <= row["upper"] for row in ranks)
    agreement = [row for row in rows if row["metric"] == "agreement"]
    assert all(row["value"] == 1.0 for row in agreement)
    shares = [row for row in rows if row["metric"] == "stratum-share"]
    assert {row["stratum"] for row in shares} == {"transitive", "intransitive"}
    assert all(row["performance"] is not None for row in shares)


def _simulate(folder: Path, collector: dict[str, JsonValue], levels: int, /) -> None:
    """Declare a simulated collector, agreement stopping and the response levels.

    The initial judgments are rewritten on the declared levels, each tercile of
    the trained point's root mass at an evenly spaced level.
    """
    document = folder / "active-learning.yaml"
    text = document.read_text(encoding="utf-8")
    for declared, simulated in (
        (
            "collector: {kind: bead}",
            f"collector: {json.dumps(collector)}",
        ),
        (
            "stopping: {kind: asymptote, window: 3, maximum_gain: 0.0}",
            f"stopping: {{kind: agreement, threshold: {AGREEMENT}, basis: estimate}}",
        ),
    ):
        assert declared in text
        text = text.replace(declared, simulated)
    document.write_text(text, encoding="utf-8", newline="\n")

    labels = tuple(f"level-{level}" for level in range(levels))
    fit = folder / "fit.yaml"
    fit.write_text(
        fit.read_text(encoding="utf-8").replace(
            f"levels: {len(LABELS)}", f"levels: {levels}"
        ),
        encoding="utf-8",
        newline="\n",
    )
    encoding = json.loads((folder / "encoding.json").read_text(encoding="utf-8"))
    encoding.update(n_levels=levels, labels=list(labels))
    (folder / "encoding.json").write_text(
        json.dumps(encoding), encoding="utf-8", newline="\n"
    )
    spacing = (levels - 1) // (len(LABELS) - 1)
    initial = [
        {
            **record,
            "response_label": labels[LABELS.index(record["response_label"]) * spacing],
        }
        for record in map(
            json.loads,
            (folder / "initial.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    ]
    (folder / "initial.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in initial),
        encoding="utf-8",
        newline="\n",
    )


@pytest.mark.parametrize(
    ("collector", "levels"),
    [
        (
            {
                "kind": "predictor-truth",
                "column": "p_gram",
                "participants": 4,
                "scale": "standardized",
                "latent_range": LATENT_RANGE,
            },
            len(LABELS),
        ),
        (
            {"kind": "drawn-effects", "participants": 4, "latent_range": LATENT_RANGE},
            len(LABELS),
        ),
        (
            {"kind": "registered-design", "design": REGISTERED_DESIGN, "join": None},
            len(LABELS),
        ),
    ],
)
def test_a_simulated_collector_completes_a_run_without_fielding_a_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collector: dict[str, JsonValue],
    levels: int,
) -> None:
    folder = tmp_path / "documents"
    responses = _inputs(tmp_path, folder)
    _simulate(folder, collector, levels)
    if collector["kind"] == "registered-design":
        _registered(folder, responses, monkeypatch)
    out = tmp_path / "out"

    status = lofi()(
        ["active-learn", str(folder / "active-learning.yaml"), "--out", str(out)]
    )
    manifest = json.loads((out / "active-learning.json").read_text(encoding="utf-8"))
    rows = json.loads((out / "rows.json").read_text(encoding="utf-8"))["rows"]

    assert status == 0
    assert manifest["status"] == "completed"
    assert manifest["awaiting"] is None
    assert not (out / "rounds").exists()
    agreement = _stopped_on_record(
        manifest["stop_reason"], manifest["document"]["max_rounds"], rows
    )
    if collector["kind"] == "predictor-truth":
        assert agreement[-1] >= AGREEMENT


def test_a_simulated_run_from_a_collected_start_reads_no_bead_file(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "documents"
    _inputs(tmp_path, folder)
    for bead in ("initial.jsonl", "encoding.json"):
        (folder / bead).unlink()
    collector: dict[str, JsonValue] = {
        "kind": "predictor-truth",
        "column": "p_gram",
        "participants": 8,
        "scale": "standardized",
        "latent_range": LATENT_RANGE,
    }
    document = folder / "active-learning.yaml"
    text = document.read_text(encoding="utf-8")
    for declared, simulated in (
        (
            "initial: {kind: bead, responses: initial.jsonl}",
            (
                "initial: {kind: collected, stratify_by: \"item['structure']\", "
                "items_per_stratum: 3}"
            ),
        ),
        ("encoding: encoding.json\n", ""),
        (
            "collector: {kind: bead}",
            f"collector: {json.dumps(collector)}",
        ),
        (
            "stopping: {kind: asymptote, window: 3, maximum_gain: 0.0}",
            f"stopping: {{kind: agreement, threshold: {AGREEMENT}, basis: estimate}}",
        ),
        ("max_rounds: 2", "max_rounds: 5"),
    ):
        assert declared in text
        text = text.replace(declared, simulated)
    document.write_text(text, encoding="utf-8", newline="\n")
    out = tmp_path / "out"

    status = lofi()(["active-learn", str(document), "--out", str(out)])
    manifest = json.loads((out / "active-learning.json").read_text(encoding="utf-8"))
    rows = json.loads((out / "rows.json").read_text(encoding="utf-8"))["rows"]

    assert status == 0
    assert not (folder / "initial.jsonl").exists()
    assert not any(folder.glob("rounds/**/responses.jsonl"))
    assert manifest["status"] == "completed"
    assert manifest["initial_judgments"] == 2 * 3 * 8
    assert set(manifest["source_digests"]) == {
        "arguments.jsonl",
        "fit.yaml",
        "item-template.json",
        "items.json",
        "templates.jsonl",
        "trained/manifest.json",
    }
    assert not (out / "rounds").exists()
    assert manifest["rounds"] == max(row["round"] for row in rows)
    _stopped_on_record(manifest["stop_reason"], 5, rows)


def _value(row: dict[str, JsonValue], column: str, /) -> float:
    """Return one numeric column of a round row."""
    value = row[column]
    assert isinstance(value, float | int)
    return float(value)


def _stopped_on_record(
    final: JsonValue, max_rounds: int, rows: list[dict[str, JsonValue]], /
) -> list[float]:
    """Check each round's stop record against the manifest, returning agreement.

    Only the last round stops, under the reason the manifest records: the
    declared agreement if it was reached, else the round bound or the
    exhausted pool.
    """
    agreement = [row for row in rows if row["metric"] == "agreement"]
    values = [_value(row, "value") for row in agreement]
    assert [row["round"] for row in agreement] == list(range(1, len(agreement) + 1))
    assert all(value < AGREEMENT for value in values[:-1])
    assert [row["stopped"] for row in agreement] == [
        row is agreement[-1] for row in agreement
    ]
    assert [row["stop_reason"] for row in agreement] == [
        final if row is agreement[-1] else None for row in agreement
    ]
    if values[-1] >= AGREEMENT:
        assert final == "agreement"
    else:
        assert final in {"max-rounds", "pool-exhausted"}
    if final == "max-rounds":
        assert len(agreement) == max_rounds
    for row in rows:
        if "lower" in row:
            assert _value(row, "lower") <= _value(row, "value") <= _value(row, "upper")
    return values


def _git(root: Path, *arguments: str) -> None:
    subprocess.run([GIT, *arguments], cwd=root, check=True, capture_output=True)


def _experiment(root: Path, /) -> Path:
    """Write and commit a one-point experiment over the active-learning document."""
    folder = root / "experiments"
    agreement: dict[str, JsonValue] = {
        "evaluator": "active-learning",
        "metric": "agreement",
    }
    document: dict[str, JsonValue] = {
        "role": "collection",
        "freeze_id": "freeze/1",
        "config": {"kind": "active-learning", "path": "active-learning.yaml"},
        "launch": {"kind": "serial"},
        "failure": {"numerical": "final", "out_of_memory": "retry"},
        "sweep": {
            "grid": [{"path": "max_rounds", "values": [2]}],
            "seeds": [0],
            "expected_points": 1,
        },
        "seed_policy": {"root": 7},
        "rows": [
            agreement,
            {"evaluator": "active-learning", "metric": "rank-correlation"},
        ],
        "primary": agreement,
        "preregistration": "preregistration.json",
        "compute": {"kind": "unpriced"},
        "contraction": {"rungs": [{"kind": "fallback"}], "factorial": "one bound"},
        "stop_rules": [],
        "decision_point": {
            "date": "2026-12-01",
            "signal": {
                "kind": "metric",
                "row": agreement,
                "statistic": "mean",
                "direction": "above",
                "threshold": 0.5,
            },
            "fallback": "Collected judgments agree across annotators.",
        },
        "evaluation": {
            "alpha": 0.05,
            "correction": {"kind": "benjamini-hochberg"},
            "dependence": "prds",
            "minimum_seeds": 1,
            "seed_budget": {
                "metric": "agreement",
                "target": "mean",
                "precision": 0.05,
                "level": 0.95,
            },
        },
    }
    preregistration: dict[str, JsonValue] = {
        "grid": {"max_rounds": [MAX_ROUNDS]},
        "hypotheses": [
            {
                "name": "agreement",
                "left": "collected",
                "right": "chance",
                "family": "primary",
            }
        ],
    }
    (folder / "preregistration.json").write_text(
        json.dumps(preregistration), encoding="utf-8", newline="\n"
    )
    path = folder / "collection.json"
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")
    _git(root, "init", "--quiet", "--initial-branch", "main")
    _git(root, "config", "user.name", "Active Test")
    _git(root, "config", "user.email", "active@example.org")
    _git(root, "config", "commit.gpgSign", "false")
    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "--message", "Declare a collection")
    return path


def test_a_pilot_experiment_resumes_its_point_once_each_batch_is_answered(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    folder = root / "experiments"
    responses = _inputs(tmp_path, folder)
    experiment = _experiment(root)
    statuses: list[str] = []

    for _ in range(3):
        status = main(
            [
                "launch",
                "--experiment",
                str(experiment),
                "--pilot",
                "--out",
                str(tmp_path / "launch.json"),
            ],
            training=TRAINING_OPERATIONS,
            launching=ExperimentLaunching(checkout=root),
            stdout=io.StringIO(),
        )
        assert status == 0
        record = json.loads(
            (root / "pilot" / "results" / "collection" / "STATUS.json").read_text(
                encoding="utf-8"
            )
        )
        (entry,) = record["points"].values()
        statuses.append(entry["status"])
        if entry["status"] == "completed":
            break
        (batch,) = sorted((root / "pilot").glob("**/rounds/*/batch.jsonl"))[-1:]
        _answer(batch, responses)

    rows = pq.read_table(
        root / "pilot" / "results" / "collection" / "rows.parquet"
    ).to_pylist()

    rounds = _collected_rounds(statuses, MAX_ROUNDS)

    assert {row["round"] for row in rows} == set(range(1, rounds + 1))
    (stop,) = {row["stop_reason"] for row in rows if row["stopped"]}
    assert stop in _stops(rounds, MAX_ROUNDS)
    assert {row["metric"] for row in rows} == {
        "agreement",
        "rank-correlation",
        "batch-variance",
        "stratum-share",
    }
    assert all(row["freeze_id"] == UNFROZEN for row in rows)
