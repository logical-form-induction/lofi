"""Generated declarations compose, build their data and train from a document.

Each example is a run document over a drawn grammar and a drawn data section,
written to disk and composed as any document is. The small profile runs on
every suite; the large profile is the opt-in generality run, which draws
declarations several times bigger and must finish within the profile's time
and memory limits.
"""

import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest
from hypothesis import HealthCheck, Phase, find, given, settings
from hypothesis import strategies as st
from tests.declarations import (
    LARGE,
    SMALL,
    DataKind,
    DataSection,
    Profile,
    ProfileExceededError,
    data_sections,
    exceeds_shipped_ranks,
    grammars,
    packaged_names,
    plan,
    run_document,
    train_one_step,
    within,
)

from lofi.data.synthetic import PairRelationUnreachableError
from lofi.data.synthetic.spec.inference import (
    SyntheticInferenceDecl,
    SyntheticInferenceLowerer,
)
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.lexicon import UnsatisfiableClampError
from lofi.nn.arity import ArityUnsupportedError
from lofi.spec import RunSpecCodec

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.spec.grammar import InlineLexiconSpec

TRAINED = (DataKind.STRINGS, DataKind.ACCEPTABILITY, DataKind.PARTIAL_SUPERVISION)
FOREST_POLICY: Final = ParsePolicy(Retain.FOREST, Reachability.ALL)


def _trains(section: DataSection) -> None:
    """Train one step of a drawn run, or see a declared capability refusal."""
    run = run_document(section)
    grammar = section.grammar

    with tempfile.TemporaryDirectory() as directory:
        try:
            metrics = train_one_step(run, Path(directory))
        except ArityUnsupportedError:
            if not exceeds_shipped_ranks(grammar):
                raise
            return
        except UnsatisfiableClampError:
            if section.kind is not DataKind.PARTIAL_SUPERVISION:
                raise
            return

    assert metrics["optimizer/loss_finite"] == 1.0
    assert all(math.isfinite(value) for value in metrics.values())


def _builds_inference(section: DataSection) -> None:
    """Compose an inference run document and draw its labeled pairs from it."""
    run = run_document(section)
    declared = SyntheticInferenceDecl.model_validate_json(json.dumps(section.dataset))

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "generated.json"
        path.write_text(json.dumps(run), encoding="utf-8", newline="\n")
        composed = RunSpecCodec()(path).data.datasets["train"].dataset

    assert isinstance(composed, SyntheticInferenceDecl)
    assert composed.grammar == declared.grammar
    assert composed.source == declared.source

    try:
        drawn = SyntheticInferenceLowerer().lower(composed, FOREST_POLICY)
    except PairRelationUnreachableError:
        return

    assert len(drawn.pairs) == declared.size
    assert {pair.label for pair in drawn.pairs} <= {
        "entails",
        "contradicts",
        "neutral",
    }


def _property(profile: Profile, kind: DataKind) -> Callable[[], None]:
    """Return the property one profile and data kind are checked under."""
    check = _builds_inference if kind is DataKind.INFERENCE else _trains

    @settings(
        max_examples=profile.examples,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    @given(st.data())
    def holds(data: st.DataObject) -> None:
        check(data.draw(data_sections(profile, kind)))

    return holds


@pytest.mark.parametrize("kind", TRAINED, ids=lambda kind: kind.value)
def test_every_small_generated_run_composes_builds_its_data_and_trains(
    kind: DataKind,
) -> None:
    with within(SMALL):
        _property(SMALL, kind)()


def test_every_small_generated_inference_run_composes_and_draws_its_pairs() -> None:
    with within(SMALL):
        _property(SMALL, DataKind.INFERENCE)()


@pytest.mark.generality
@pytest.mark.parametrize("kind", [*TRAINED, DataKind.INFERENCE], ids=lambda k: k.value)
def test_large_generated_runs_finish_within_their_profile(kind: DataKind) -> None:
    with within(LARGE):
        _property(LARGE, kind)()


def test_a_drawn_run_is_planned_from_its_document() -> None:
    @settings(max_examples=2, deadline=None)
    @given(data_sections(SMALL, DataKind.STRINGS))
    def planned(section: DataSection) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec, blueprint = plan(run_document(section), Path(directory))

        assert spec.data.datasets["train"].dataset.grammar == section.grammar
        assert blueprint.sampler_streams

    planned()


def test_a_drawn_grammar_declares_no_packaged_name() -> None:
    @settings(max_examples=4 * SMALL.examples, deadline=None)
    @given(st.data())
    def avoids(data: st.DataObject) -> None:
        grammar = data.draw(grammars(SMALL))
        signature = grammar.signature
        lexicon = cast("InlineLexiconSpec", grammar.lexicon)
        drawn = {
            *signature.sorts,
            *signature.primitives,
            *signature.constructors,
            *(operator.name for operator in signature.operators),
            *(entry.form for entry in lexicon.entries),
        }

        assert not drawn & packaged_names()

    avoids()


def test_acceptability_is_drawn_with_and_without_negatives() -> None:
    sections = data_sections(SMALL, DataKind.ACCEPTABILITY)
    quick = settings(
        max_examples=200, deadline=None, database=None, phases=(Phase.generate,)
    )

    with_negatives = find(
        sections, lambda section: "corruption" in section.dataset, settings=quick
    )
    without_negatives = find(
        sections, lambda section: "corruption" not in section.dataset, settings=quick
    )

    noise = cast("dict[str, dict[str, float]]", without_negatives.dataset["noise"])

    assert "noise" in with_negatives.dataset
    assert noise["ungrammatical"]["rate"] == 0.0


def test_a_run_past_its_profile_is_stopped_and_reported() -> None:
    @settings(max_examples=50, deadline=None, database=None)
    @given(st.integers())
    def endless(value: int) -> None:
        del value
        time.sleep(0.05)

    with (
        pytest.raises(ProfileExceededError, match=r"longer than its 0\.2 seconds"),
        within(SMALL._replace(seconds=0.2)),
    ):
        endless()

    with within(SMALL._replace(memory=1)):
        pass


def test_a_run_past_its_memory_is_stopped_while_it_runs() -> None:
    script = (
        "import time\n"
        "from tests.declarations import SMALL, peak_resident_bytes, within\n"
        "held = []\n"
        "with within(SMALL._replace(memory=peak_resident_bytes() + 2**25)):\n"
        "    held.append(b'x' * 2**27)\n"
        "    while True:\n"
        "        time.sleep(0)\n"
    )
    ran = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        encoding="utf-8",
        timeout=50,
        check=False,
    )

    assert ran.returncode != 0
    assert "ProfileExceededError" in ran.stderr
    assert "resident bytes" in ran.stderr


def test_a_peak_reached_and_released_between_samples_is_still_reported() -> None:
    script = (
        "from tests.declarations import SMALL, peak_resident_bytes, within\n"
        "with within(SMALL._replace(memory=peak_resident_bytes() + 2**25)):\n"
        "    held = b'x' * 2**27\n"
        "    del held\n"
    )
    ran = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        encoding="utf-8",
        timeout=50,
        check=False,
    )

    assert ran.returncode != 0
    assert "ProfileExceededError" in ran.stderr
    assert "resident bytes" in ran.stderr


def test_a_child_process_is_not_charged_its_parents_peak() -> None:
    held = b"x" * 2**31
    script = (
        "from tests.declarations import peak_resident_bytes\n"
        "print(peak_resident_bytes())\n"
    )
    ran = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        encoding="utf-8",
        timeout=50,
        check=True,
    )
    del held

    assert int(ran.stdout) < 2**31
