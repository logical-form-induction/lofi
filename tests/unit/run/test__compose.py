import json
import subprocess
import sys
from typing import TYPE_CHECKING

from tests.unit.run.conftest import a_document

from lofi.run import Selection, compose, composer, resolved_document, trace
from lofi.run import _compose as compose_module
from lofi.spec import config_search_path, derive_seed

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from lofi.spec import Composition, RunSpec


def a_selection(root: Path, *overrides: str) -> Selection:
    root.mkdir(parents=True, exist_ok=True)
    document = a_document()
    document["defaults"] = [{"trainer": "long"}]
    del document["trainer"]
    (root / "trainer").mkdir()
    (root / "trainer" / "long.json").write_text(
        json.dumps({"kind": "launcher-unit-trainer", "epochs": 5}),
        encoding="utf-8",
        newline="\n",
    )
    (root / "profiles").mkdir()
    (root / "profiles" / "longer.json").write_text(
        json.dumps({"trainer": {"epochs": 7}}), encoding="utf-8", newline="\n"
    )
    config = root / "run.json"
    config.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    return Selection(
        config=config,
        profile="longer",
        overrides=overrides,
        overlays=(),
        search_path=config_search_path(root),
    )


def test_a_selection_composes_its_group_profile_and_overrides_in_order(
    tmp_path: Path,
) -> None:
    profiled = resolved_document(compose(a_selection(tmp_path)))["trainer"]
    overridden = resolved_document(
        compose(a_selection(tmp_path / "overridden", "trainer.epochs=3"))
    )["trainer"]

    assert isinstance(profiled, dict)
    assert isinstance(overridden, dict)
    assert profiled["epochs"] == 7
    assert overridden["epochs"] == 3


def test_a_trace_names_the_override_that_set_a_leaf(tmp_path: Path) -> None:
    composition = trace(a_selection(tmp_path, "trainer.epochs=3"))

    origin = composition.provenance.origin_of("trainer.epochs")

    assert origin.source.value == "override"


def test_a_trace_validates_through_one_composition_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def duplicate(_selection: Selection, /) -> None:
        msg = "trace invoked the separate untraced composition route"
        raise AssertionError(msg)

    monkeypatch.setattr(compose_module, "compose", duplicate)

    assert trace(a_selection(tmp_path)).spec.trainer.kind == "launcher-unit-trainer"


def test_a_composer_applies_a_points_overrides_after_the_selections(
    tmp_path: Path,
) -> None:
    compose_point = composer(a_selection(tmp_path, "trainer.epochs=3"))

    point = compose_point(("trainer.epochs=9",), 0)

    trainer = point["trainer"]
    origin = compose_point.provenance(("trainer.epochs=9",), 0).origin_of(
        "trainer.epochs"
    )

    assert isinstance(trainer, dict)
    assert trainer["epochs"] == 9
    assert origin.source.value == "override"


def test_a_point_composer_reuses_a_seed_independent_compilation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = a_selection(tmp_path)
    traced = compose_module.trace
    selections: list[Selection] = []

    def recording(selection: Selection, /) -> Composition[RunSpec]:
        selections.append(selection)
        return traced(selection)

    monkeypatch.setattr(compose_module, "trace", recording)
    compose_point = composer(selected)

    first = compose_point((), 3)
    repeated = compose_point((), 3)
    replicate = compose_point((), 4)

    assert first == repeated == replicate
    assert len(selections) == 1
    assert compose_point.provenance((), 4) == compose_point.provenance((), 3)


def test_a_point_composer_derives_named_streams_from_each_run_seed(
    tmp_path: Path,
) -> None:
    selected = a_selection(tmp_path)
    group = tmp_path / "trainer" / "long.json"
    declaration = json.loads(group.read_text(encoding="utf-8"))
    declaration["epochs"] = "${lofi.seed:model-structures}"
    group.write_text(json.dumps(declaration), encoding="utf-8", newline="\n")
    compose_point = composer(selected._replace(profile=None))

    first = compose_point((), 3)
    second = compose_point((), 4)
    first_trainer = first["trainer"]
    second_trainer = second["trainer"]

    assert isinstance(first_trainer, dict)
    assert isinstance(second_trainer, dict)
    assert first_trainer["epochs"] == derive_seed(3, "model-structures")
    assert second_trainer["epochs"] == derive_seed(4, "model-structures")


def test_a_fresh_process_composes_named_streams_from_each_run_seed(
    tmp_path: Path,
) -> None:
    selected = a_selection(tmp_path)
    group = tmp_path / "trainer" / "long.json"
    declaration = json.loads(group.read_text(encoding="utf-8"))
    declaration["epochs"] = "${lofi.seed:model-structures}"
    group.write_text(json.dumps(declaration), encoding="utf-8", newline="\n")
    program = """
import json
import sys
from pathlib import Path

import tests.unit.run.conftest

from lofi.run import Selection, composer
from lofi.spec import config_search_path

root = Path(sys.argv[1])
selection = Selection(
    config=root / "run.json",
    profile=None,
    overrides=(),
    overlays=(),
    search_path=config_search_path(root),
)
compose_point = composer(selection)
values = [compose_point((), seed)["trainer"]["epochs"] for seed in (3, 4)]
print(json.dumps(values))
"""

    result = subprocess.run(
        [sys.executable, "-c", program, str(selected.config.parent)],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert json.loads(result.stdout) == [
        derive_seed(3, "model-structures"),
        derive_seed(4, "model-structures"),
    ]
