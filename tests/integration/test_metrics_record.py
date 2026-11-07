"""The metrics record a trained point leaves in its run directory, end to end.

The P1 point is the documentation run over a generated grammar with a richer
data section: a 40-form, 8-category generated lexicon with power-law category
shares and rank frequencies, 400 sentences of negative-binomial length, a
lexical ambiguity target, a quarter of the items corrupted by swaps and
deletions, and a tenth of the labels flipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest
import yaml
from tests.integration.test_experiment_execution import (
    Frozen,
    a_repository,
    evaluated_run,
    freeze,
)
from tests.unit.run.test__experiment import an_experiment_document
from tests.unit.run.test_freezing import record_price

from lofi.eval import config_digest
from lofi.run import (
    MetricsRecord,
    expand_experiment,
    plan_training,
    read_experiment,
    read_metrics,
)
from lofi.run._boundary.metrics import corpus_record
from lofi.run._entrypoint import main
from lofi.spec import RunSpec, RunSpecCodec, config_search_path

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.data.synthetic import SyntheticDataset

TREE: Final = Path(__file__).resolve().parents[2]
EXAMPLE: Final = TREE / "docs/examples/synthetic-generated.yaml"


def p1_document(*, selected: bool, seed: int = 5) -> dict[str, JsonValue]:
    """Return the P1 run document, selecting lexical constraints if asked."""
    document = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    dataset = document["data"]["datasets"]["train"]["dataset"]
    dataset["grammar"]["lexicon"].update(
        forms=40,
        categories=8,
        category_shares={"kind": "power-law", "exponent": 1.0},
        rank_law={"kind": "power-law", "exponent": 1.1},
        seed=3,
        max_depth=2,
    )
    dataset.update(
        size=400,
        seed=seed,
        length={
            "kind": "negative-binomial",
            "min_length": 2,
            "max_length": 9,
            "r": 3.0,
            "p": 0.6,
        },
        ambiguity={"lexical": {"value": 1.5, "tolerance": 0.5}, "attempts": 200},
        corruption={
            "proportion": {"value": 0.25, "draw": "exact"},
            "operators": {
                "entries": [
                    {
                        "operator": {"kind": "swap", "edits": 1},
                        "weight": 1.0,
                    },
                    {
                        "operator": {"kind": "deletion", "edits": 1},
                        "weight": 1.0,
                    },
                ]
            },
            "attempts": 50,
        },
        noise={
            "grammatical": {"rate": 0.1},
            "ungrammatical": {"rate": 0.1},
        },
    )
    document["trainer"].update(batch_size=16, accumulation_steps=1)

    if selected:
        document["data"]["datasets"]["train"]["constraint_source"] = {
            "kind": "gold-lexicon"
        }
        document["data"]["datasets"]["train"]["constraint_policy"] = {
            "syntactic": {"ambiguity": "union", "coverage": 1.0}
        }
        document["loss"]["terms"].append(
            {"objective": {"kind": "pinning", "readout": "types"}}
        )

    return document


def trained(directory: Path, document: dict[str, JsonValue]) -> tuple[Path, Path]:
    """Train a document through the run command and return it and its run directory."""
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / "p1.yaml"
    config.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8", newline="\n"
    )

    assert (
        main(
            [
                "run",
                "--config",
                str(config),
                "--root",
                str(directory / "root"),
                "--experiment-seed",
                "5",
            ]
        )
        == 0
    )

    (run,) = (directory / "root" / "runs").glob("[0-9a-f]*")

    return config, run


def recorded(run: Path) -> MetricsRecord:
    """Return the closed metrics record a run directory holds."""
    record = read_metrics(run)

    assert record is not None
    assert record.stage == "closed"

    return record


@pytest.mark.integration
def test_the_p1_point_reads_back_every_report_it_measured_and_achieved(
    tmp_path: Path,
) -> None:
    config, run = trained(tmp_path / "plain", p1_document(selected=False))
    record = recorded(run)
    blueprint = plan_training(
        RunSpecCodec(search_path=config_search_path(config.parent))(config)
    )
    drawn = cast("SyntheticDataset", blueprint.lower_dataset("train")).corpus
    corpus = record.corpus
    bucketing = record.bucketing
    fitting = record.fitting

    assert corpus == corpus_record(drawn)
    assert corpus is not None
    assert corpus.length.achieved.total == len(drawn.examples)
    assert corpus.lexicon.realized_size > 0
    assert corpus.lexicon.realized_vocabulary_size == len(
        {token.surface for example in drawn.examples for token in example.item.tokens}
    )
    assert corpus.lexicon.rank_frequency_exponent is not None
    assert corpus.lexicon.coupling is not None
    assert corpus.lexicon.concordance is not None
    assert set(corpus.inventory.primitives) <= {"S", "P1", "P2", "P3"}
    assert corpus.inventory.realized_primitive_count == len(corpus.inventory.primitives)
    assert corpus.ambiguity.lexical.total > 0
    assert corpus.corruption is not None
    assert corpus.corruption.achieved_proportion == 0.25
    assert set(corpus.corruption.accepted) == {"deletion", "swap"}
    assert corpus.noise is not None
    assert corpus.noise.sample_size == 400
    assert corpus.scope is None
    assert record.inference is None
    assert record.selection is None
    assert bucketing is not None
    assert bucketing.training.items + bucketing.validation.items == 400
    assert record.point.corpus_fingerprint == blueprint.corpus_revision
    assert record.point.roles["terms"].kind == "tree-rnn"
    assert record.point.roles["terms"].trainable == 0
    assert record.point.roles["parser"].trainable == (
        record.point.roles["parser"].parameters
    )
    assert record.point.trainable_count < record.point.parameter_count
    assert set(record.phases) == {"preparation", "recovery", "fitting"}
    assert record.peak_memory_bytes is None
    assert record.residuals == {}
    assert fitting is not None
    assert [epoch.epoch for epoch in fitting.epochs] == [0]
    assert len(fitting.epochs[0].steps) == bucketing.training.batches
    validation = fitting.epochs[0].validation
    assert validation is not None
    assert "loss/grammaticality" in validation
    assert fitting.stop == "horizon"
    assert fitting.best_checkpoint is not None
    assert fitting.optimizer_steps == bucketing.training.batches


@pytest.mark.integration
def test_the_p1_point_under_constraint_selection_records_the_budget_it_achieved(
    tmp_path: Path,
) -> None:
    _, run = trained(tmp_path / "selected", p1_document(selected=True))
    record = recorded(run)
    selection = record.selection
    corpus = record.corpus

    assert selection is not None
    assert corpus is not None
    assert selection.types_pinned == corpus.lexicon.realized_size
    assert selection.mass_share == 1.0
    assert set(selection.by_category) <= set(corpus.inventory.categories)


@pytest.mark.integration
def test_equal_declarations_write_byte_equal_report_blocks(tmp_path: Path) -> None:
    first = recorded(trained(tmp_path / "first", p1_document(selected=False))[1])
    second = recorded(trained(tmp_path / "second", p1_document(selected=False))[1])
    other = recorded(
        trained(tmp_path / "other", p1_document(selected=False, seed=6))[1]
    )
    first_corpus = first.corpus
    second_corpus = second.corpus
    other_corpus = other.corpus

    assert first.point.model_dump_json() == second.point.model_dump_json()
    assert first_corpus is not None
    assert second_corpus is not None
    assert other_corpus is not None
    assert first_corpus.model_dump_json() == second_corpus.model_dump_json()
    assert first.bucketing is not None
    assert second.bucketing is not None
    assert first.bucketing.model_dump_json() == second.bucketing.model_dump_json()
    assert first_corpus.model_dump_json() != other_corpus.model_dump_json()


@pytest.fixture
def frozen_widths(tmp_path: Path) -> Frozen:
    root = a_repository(tmp_path)
    folder = root / "experiments" / "A"
    (folder / "run.yaml").write_text(
        evaluated_run(size=4), encoding="utf-8", newline="\n"
    )
    experiment_document = an_experiment_document()
    experiment_document["config"] = {"kind": "run", "path": "run.yaml"}
    experiment_document["sweep"] = {
        "grid": [
            {"group": "model.encoders.terms", "values": ["rnn", "hrr"]},
            {"path": "model.encoders.terms.width", "values": [8, 16]},
        ],
        "seeds": [0, 1, 2],
        "expected_points": 12,
    }
    experiment = folder / "widths.yaml"
    experiment.write_text(
        json.dumps(experiment_document), encoding="utf-8", newline="\n"
    )
    record_price(root, "widths")

    return freeze(root, experiment, tmp_path)


@pytest.mark.integration
def test_a_width_sweep_ingests_parameter_counts_that_differ_by_width_and_kind(
    frozen_widths: Frozen,
) -> None:
    report = frozen_widths.execute()
    accepted = frozen_widths.ingest()
    declared: dict[str, int] = {}

    for point in expand_experiment(
        read_experiment(frozen_widths.experiment), frozen_widths.root
    ):
        encoder = RunSpec.model_validate(point.config).model.encoders["terms"]

        declared[config_digest(point.config)] = cast(
            "int", encoder.model_dump()["width"]
        )

    counted: dict[tuple[str, int], set[int]] = {}

    for row in accepted.points:
        key = (
            cast("str", row["point/roles/terms/kind"]),
            declared[cast("str", row["config_hash"])],
        )
        counted.setdefault(key, set()).add(cast("int", row["point/parameter_count"]))

    assert all(len(counts) == 1 for counts in counted.values())

    sizes = {key: min(counts) for key, counts in counted.items()}

    assert report.launched == 12
    assert len(accepted.points) == 12
    assert {row["status"] for row in accepted.points} == {"completed"}
    assert set(sizes) == {
        (kind, width) for kind in ("tree-rnn", "hrr") for width in (8, 16)
    }

    for kind in ("tree-rnn", "hrr"):
        assert sizes[kind, 8] < sizes[kind, 16]

    for width in (8, 16):
        assert sizes["tree-rnn", width] != sizes["hrr", width]
