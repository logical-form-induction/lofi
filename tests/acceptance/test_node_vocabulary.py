"""A node-vocabulary comparison runs from an experiment document.

Each of four causal probes declares the representation it measures as its
node-vocabulary arm and persists that arm's complete sweep beside its rows.
An experiment document declares the node-vocabulary reducer over the four
persisted records, and reducing the experiment compares the arms over one
evaluated pair population.
"""

from typing import TYPE_CHECKING

from tests.unit.probe.test__causal_probe import PUBLISHED, _declaration, _Point
from tests.unit.run.test__experiment import an_experiment_document, write_experiment

from lofi.eval._reducers import ReductionContext, reduce_experiment_rows
from lofi.probe import (
    PROBE_RECORDS_NAME,
    NodeVocabularyArm,
    NodeVocabularyArmDecl,
    ProbeBuildContext,
    RecordingProbe,
    write_probe_records,
)
from lofi.probe._causal_probe import PLUGIN
from lofi.run import read_experiment

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


def _probed_arm(directory: Path, arm: NodeVocabularyArm, /) -> None:
    """Run one causal probe declaring an arm and persist its records."""
    point = _Point()
    declaration = _declaration().with_(
        arm=NodeVocabularyArmDecl(representation=arm.value, search="fixed-rank")
    )
    probe = PLUGIN.construct(declaration, ProbeBuildContext("causal", point))
    probe.admit(PUBLISHED)

    assert isinstance(probe, RecordingProbe)
    directory.mkdir(parents=True)
    write_probe_records(
        directory / PROBE_RECORDS_NAME, {"causal": probe.recorded(point).records}
    )


def test_an_experiment_document_compares_four_persisted_arms(tmp_path: Path) -> None:
    arms: list[JsonValue] = []

    for arm in NodeVocabularyArm:
        _probed_arm(tmp_path / "experiments" / "arms" / arm.value, arm)
        arms.append(
            {
                "representation": arm.value,
                "source": f"arms/{arm.value}/{PROBE_RECORDS_NAME}",
                "probe": "causal",
                "record": "pass-0",
            }
        )

    document = an_experiment_document()
    document["reducers"] = [{"kind": "node-vocabulary", "arms": arms}]
    path = write_experiment(tmp_path, document)
    experiment = read_experiment(path)

    reduced = reduce_experiment_rows(
        (),
        experiment.reducers,
        ReductionContext(path.parent, seed=experiment.seed_policy.root),
    )

    assert [row["arm"] for row in reduced.rows] == [
        arm.value for arm in NodeVocabularyArm
    ]
    assert {row["reducer"] for row in reduced.rows} == {"node-vocabulary"}
    assert len({row["pair_digest"] for row in reduced.rows}) == 1
    assert len({row["cell_count"] for row in reduced.rows}) == 1
    means = [row["mean_iia"] for row in reduced.rows]
    assert all(isinstance(mean, float) and 0.0 <= mean <= 1.0 for mean in means)
