"""Process composition for commands that train or time the neural stack."""

import sys
from io import TextIOWrapper
from typing import TYPE_CHECKING

from lofi.run._cli import main as run_cli
from lofi.run._timing import TrainingStepTimer
from lofi.run.active import ACTIVE_LEARNING_COMMAND
from lofi.run.bead import predict_bead_items
from lofi.run.freezing import mint_freeze
from lofi.run.hub import HubRelease, export_trained_point
from lofi.run.launching import ExperimentLaunching
from lofi.run.operations import TRAINING_OPERATIONS
from lofi.run.probing import probe_trained_point
from lofi.run.publishing import publish_trained_point
from lofi.run.reporting import render_experiment_reports
from lofi.run.results import ingest_experiment, write_seed_budget

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["STEP_TIMER", "main"]

STEP_TIMER = TrainingStepTimer()
"""The timer a price command builds and times an experiment's stacks with."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line with its concrete operations.

    The command line writes UTF-8 to its standard output and error whatever
    the locale encoding of the console or pipe they are attached to, so a
    logical form reads the same on every platform.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, TextIOWrapper):
            stream.reconfigure(encoding="utf-8")

    return run_cli(
        argv,
        training=TRAINING_OPERATIONS,
        launching=ExperimentLaunching(),
        freeze=mint_freeze,
        seed_budget=write_seed_budget,
        release=HubRelease(),
        timer=STEP_TIMER,
        export=export_trained_point,
        bead_predict=predict_bead_items,
        probe=probe_trained_point,
        publish=publish_trained_point,
        ingest=ingest_experiment,
        reports=render_experiment_reports,
        active_learn=ACTIVE_LEARNING_COMMAND,
    )
