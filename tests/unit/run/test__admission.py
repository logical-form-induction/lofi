from typing import TYPE_CHECKING

import pytest
from tests.unit.run.test__experiment import an_experiment_document, write_experiment

from lofi.run import Admission, AdmissionError, admit, read_experiment

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.run import Experiment


def experiment(
    root: Path,
    name: str,
    *,
    role: str = "headline",
    p90: float = 20.0,
    freeze_id: str = "freeze/1",
    priced: bool = True,
) -> Experiment:
    document = an_experiment_document(freeze_id=freeze_id)
    document["role"] = role
    compute: JsonValue = (
        {"kind": "estimate", "p50_hours": 1.0, "p90_hours": p90, "cost_model": "v1"}
        if priced
        else {"kind": "unpriced"}
    )
    document["compute"] = compute

    return read_experiment(write_experiment(root, document, name=name))


def test_a_headline_and_its_analysis_are_admitted_together_within_the_share(
    tmp_path: Path,
) -> None:
    headline = experiment(tmp_path, "headline", p90=30.0)
    analysis = experiment(tmp_path, "analysis", role="analysis", p90=25.0)

    admitted = admit(headline, (analysis,), admitted_hours=60.0)

    assert admitted == Admission(("headline", "analysis"), 55.0, 60.0)


def test_an_experiment_whose_p90_exceeds_its_admission_is_refused(
    tmp_path: Path,
) -> None:
    headline = experiment(tmp_path, "headline", p90=40.0)
    analysis = experiment(tmp_path, "analysis", role="analysis", p90=25.0)

    with pytest.raises(AdmissionError, match=r"65\.0 accelerator-hours at P90"):
        admit(headline, (analysis,), admitted_hours=60.0)


def test_a_headline_without_its_analysis_is_refused(tmp_path: Path) -> None:
    headline = experiment(tmp_path, "headline")

    with pytest.raises(AdmissionError, match="no analysis experiment"):
        admit(headline, (), admitted_hours=1000.0)

    analysis = experiment(tmp_path, "analysis", role="analysis")

    with pytest.raises(AdmissionError, match="no headline experiment"):
        admit(analysis, (), admitted_hours=1000.0)


def test_a_preflight_is_admitted_alone(tmp_path: Path) -> None:
    preflight = experiment(tmp_path, "preflight", role="preflight", p90=2.0)

    assert admit(preflight, (), admitted_hours=2.0).p90_hours == 2.0


@pytest.mark.parametrize(
    ("companion", "reason"),
    [
        ({"freeze_id": "freeze/2"}, "do not run against"),
        ({"priced": False}, "unpriced"),
    ],
)
def test_a_companion_off_the_freeze_or_unpriced_is_refused(
    tmp_path: Path, companion: dict[str, str | bool], reason: str
) -> None:
    headline = experiment(tmp_path, "headline")
    analysis = experiment(
        tmp_path,
        "analysis",
        role="analysis",
        freeze_id=str(companion.get("freeze_id", "freeze/1")),
        priced=bool(companion.get("priced", True)),
    )

    with pytest.raises(AdmissionError, match=reason):
        admit(headline, (analysis,), admitted_hours=1000.0)


def test_a_group_naming_one_experiment_twice_is_refused(tmp_path: Path) -> None:
    preflight = experiment(tmp_path, "preflight", role="preflight")

    with pytest.raises(AdmissionError, match="twice"):
        admit(preflight, (preflight,), admitted_hours=1000.0)
