from __future__ import annotations

import io
import sys
from json import loads
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from tests.unit.run.test_operations import (
    CONFIG,
    assert_one_entry_per_optimizer_step,
    histories,
    mlflow_overlay,
    offline,
)

from lofi.run import _entrypoint
from lofi.run._timing import TrainingStepTimer
from lofi.run.active import ACTIVE_LEARNING_COMMAND
from lofi.run.bead import predict_bead_items
from lofi.run.freezing import mint_freeze
from lofi.run.hub import HubRelease, export_trained_point
from lofi.run.launching import ExperimentLaunching
from lofi.run.operations import TRAINING_OPERATIONS
from lofi.run.probing import probe_trained_point
from lofi.run.publishing import publish_trained_point
from lofi.run.results import ingest_experiment, write_seed_budget

if TYPE_CHECKING:
    from collections.abc import Sequence


def test_main_supplies_the_concrete_training_operations_and_step_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def run_cli(argv: Sequence[str] | None, **operations: object) -> int:
        captured["argv"] = argv
        captured.update(operations)
        return 17

    monkeypatch.setattr(_entrypoint, "run_cli", run_cli)

    argv = ["configs"]
    assert _entrypoint.main(argv) == 17
    launching = captured.pop("launching")
    release = captured.pop("release")
    assert isinstance(launching, ExperimentLaunching)
    assert isinstance(release, HubRelease)
    assert captured == {
        "argv": argv,
        "training": TRAINING_OPERATIONS,
        "freeze": mint_freeze,
        "seed_budget": write_seed_budget,
        "timer": _entrypoint.STEP_TIMER,
        "export": export_trained_point,
        "bead_predict": predict_bead_items,
        "probe": probe_trained_point,
        "publish": publish_trained_point,
        "ingest": ingest_experiment,
        "reports": _entrypoint.render_experiment_reports,
        "active_learn": ACTIVE_LEARNING_COMMAND,
    }
    assert isinstance(_entrypoint.STEP_TIMER, TrainingStepTimer)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_main_writes_utf_8_whatever_the_locale_of_its_streams(
    newline: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = io.BytesIO()
    err = io.BytesIO()
    monkeypatch.setattr(
        sys, "stdout", io.TextIOWrapper(out, encoding="cp1252", newline=newline)
    )
    monkeypatch.setattr(
        sys, "stderr", io.TextIOWrapper(err, encoding="cp1252", newline=newline)
    )

    def run_cli(argv: Sequence[str] | None, **_: object) -> int:
        del argv
        sys.stdout.write("\N{GREEK SMALL LETTER LAMDA}x. dog(x)\n")
        sys.stderr.write("\N{RIGHTWARDS ARROW}\n")
        sys.stdout.flush()
        sys.stderr.flush()
        return 0

    monkeypatch.setattr(_entrypoint, "run_cli", run_cli)

    assert _entrypoint.main([]) == 0
    assert (
        out.getvalue().decode("utf-8")
        == f"\N{GREEK SMALL LETTER LAMDA}x. dog(x){newline}"
    )
    assert err.getvalue().decode("utf-8") == f"\N{RIGHTWARDS ARROW}{newline}"


TELEMETRY_BACKENDS = ("mlflow", "wandb", "trackio")


def test_lofi_run_logs_every_optimizer_step_to_a_declared_mlflow_sink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    attempts = offline(monkeypatch)
    store = tmp_path / "mlflow.db"
    argv = [
        "run",
        "--config",
        str(CONFIG),
        "--experiment-seed",
        "5",
        "--root",
        str(tmp_path / "runs"),
    ]

    overlay = mlflow_overlay(tmp_path, store)

    assert _entrypoint.main([*argv, "--extra", str(overlay)]) == 0
    tracked = capsys.readouterr().out.split()
    logged = histories(store)

    assert _entrypoint.main(argv) == 0
    untracked = capsys.readouterr().out.split()

    assert list(logged) == [tracked[0]]
    assert_one_entry_per_optimizer_step(logged[tracked[0]])
    assert tracked[:2] == untracked[:2] == [tracked[0], "completed"]
    assert histories(store) == logged
    assert attempts == []


def test_a_run_that_declares_no_sink_loads_and_contacts_no_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = offline(monkeypatch)

    for backend in TELEMETRY_BACKENDS:
        monkeypatch.setitem(sys.modules, backend, None)

    status = _entrypoint.main(
        [
            "run",
            "--config",
            str(CONFIG),
            "--experiment-seed",
            "5",
            "--root",
            str(tmp_path),
        ]
    )

    assert status == 0
    assert attempts == []


TWO_DECODERS = """model:
  decoders:
    secondary:
      kind: tree-rnn
      width: 8
  spaces:
    categories:
      combinators:
        decoders:
          "application:\\\\": secondary
"""


def test_two_roles_of_one_decoder_kind_train_and_both_reach_the_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    overlay = tmp_path / "two-decoders.yaml"
    overlay.write_text(TWO_DECODERS, encoding="utf-8", newline="\n")

    status = _entrypoint.main(
        [
            "run",
            "--config",
            str(CONFIG),
            "--extra",
            str(overlay),
            "--experiment-seed",
            "5",
            "--root",
            str(tmp_path / "runs"),
        ]
    )
    _, outcome, directory = capsys.readouterr().out.split()
    manifest = loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    fingerprints = manifest["component_fingerprints"]
    widths = {
        role: manifest["run_spec"]["model"]["decoders"][role]["width"]
        for role in ("application", "secondary")
    }

    assert (status, outcome) == (0, "completed")
    assert widths == {"application": 4, "secondary": 8}
    assert (
        fingerprints["model.decoders.application"]
        != (fingerprints["model.decoders.secondary"])
    )


TWO_COHERENCE_TERMS = """model:
  readouts:
    alternate:
      kind: terms
      space: categories
      reads: inside
      decoding: {strategy: exhaustive, residual_mass: 0.5, cap: 32}
loss:
  terms:
    - objective:
        kind: grammaticality
        readout: sentences
    - objective:
        kind: coherence
        readout: category-terms
        space: categories
        estimator: truncated
        warmup: null
      weight: 0.1
    - objective:
        kind: coherence
        readout: alternate
        space: categories
        estimator: truncated
        warmup: null
      weight: 0.2
      name: coherence-alternate
trainer:
  active_groups:
    [token_encoder, terms, identity, application, controller, parser]
"""


def test_two_coherence_terms_over_different_readouts_train_and_log_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    attempts = offline(monkeypatch)
    store = tmp_path / "mlflow.db"
    overlay = tmp_path / "two-coherence-terms.yaml"
    overlay.write_text(TWO_COHERENCE_TERMS, encoding="utf-8", newline="\n")

    status = _entrypoint.main(
        [
            "run",
            "--config",
            str(CONFIG),
            "--extra",
            str(overlay),
            "--extra",
            str(mlflow_overlay(tmp_path, store)),
            "--experiment-seed",
            "5",
            "--root",
            str(tmp_path / "runs"),
        ]
    )
    run, outcome, directory = capsys.readouterr().out.split()
    manifest = loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    ((_, logged),) = histories(store).items()
    terms = manifest["run_spec"]["loss"]["terms"]

    assert (status, outcome) == (0, "completed")
    assert [(term["objective"]["kind"], term["name"]) for term in terms] == [
        ("grammaticality", None),
        ("coherence", None),
        ("coherence", "coherence-alternate"),
    ]
    assert {
        "loss/coherence",
        "loss/coherence/controller_entropy",
        "loss/coherence-alternate",
        "loss/coherence-alternate/controller_entropy",
    } <= set(logged)
    assert logged["loss/coherence"] != logged["loss/coherence-alternate"]
    assert run in histories(store)
    assert attempts == []
