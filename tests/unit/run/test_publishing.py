import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest
from lairs.records import annotation as annotation_records

import lofi.eval._publish as publish_module
from lofi._types import ItemId
from lofi.data import LicenseRef, Licensing, Provenance
from lofi.eval import UNFROZEN, InducedBracketing
from lofi.run import _entrypoint
from lofi.run._boundary.manifest import (
    PointBinding,
    decode_seed_derivation,
)
from lofi.run._manifest import (
    read_manifest,
    write_binding,
)
from lofi.run.hub import restore_trained_point
from lofi.run.publishing import (
    LAYERS_TARGET,
    LayersPredictionLens,
    PredictedAnalysis,
    PublishLens,
    PublishTarget,
    UnlabeledDerivationError,
    publish_trained_point,
)
from lofi.run.results import PilotResultsError

if TYPE_CHECKING:
    from lofi.train import ParserFittingPlan


def test_the_run_package_publishes_the_publish_surface() -> None:
    assert PredictedAnalysis is publish_module.PredictedAnalysis
    assert LayersPredictionLens is publish_module.LayersPredictionLens
    assert PublishLens is publish_module.PublishLens
    assert PublishTarget is publish_module.PublishTarget
    assert PublishTarget("layers-records", "application/json") == LAYERS_TARGET


def test_a_public_analysis_renders_through_the_installed_lens() -> None:
    publishers = {point.name: point for point in entry_points(group="lofi.publisher")}
    lens = publishers["layers-records"].load()("lofi", "ccg")
    analysis = PredictedAnalysis(
        ItemId("expression-1"),
        "tokenization-1",
        InducedBracketing(2, {(0, 2): "S", (0, 1): "NP", (1, 2): "S\\NP"}),
        Provenance(
            "converted-corrected",
            Licensing("CC-BY-4.0", (LicenseRef("CC-BY-4.0", "annotations"),)),
            (),
            None,
        ),
        datetime(2026, 9, 22, 14, 30, tzinfo=UTC),
    )

    assert lens.target == LAYERS_TARGET
    assert json.loads(lens.render(analysis))["expression"] == "expression-1"


def test_importing_the_run_package_does_not_import_the_publish_surface() -> None:
    probe = (
        "import sys; import lofi.run; "
        "print(any(name in sys.modules for name in "
        "('lofi.run.publishing', 'lofi.eval._publish', 'lairs')))"
    )
    finished = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, encoding="utf-8", check=True
    )

    assert finished.stdout.strip() == "False"


CONFIG: Final = (
    Path(__file__).resolve().parents[3]
    / "lofi/configs/run/synthetic-decode-supertag.yaml"
)
CONVENTIONS: Final = (
    "{punctuation: keep, unary_chains: keep, trivial_spans: keep, "
    "minimum_length: 0, aggregation: sentence, coverage: zero, "
    "self_arcs: allow, gold_derivation: canonical}"
)
EVALUATORS: Final = f"""\
model:
  readouts:
    lexical-beams:
      kind: terms
      space: categories
      reads: inside
      decoding: {{strategy: beam, width: 100}}
evaluators:
  decoded-types:
    readout: lexical-beams
  beams:
    kind: decode
    interval: {{replicates: 50, level: 0.95}}
    readout: lexical-beams
    output: lexical-types
    conventions: {CONVENTIONS}
  brackets:
    kind: bracketing
    interval: {{replicates: 50, level: 0.95}}
    punctuation: {{source: none, forms: []}}
    decoder: minimum-risk
    conventions: {CONVENTIONS}
"""


def lofi(*argv: str) -> int:
    return _entrypoint.main(list(argv))


def digests(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def validation_items(run: Path) -> tuple[ItemId, ...]:
    plan = cast("ParserFittingPlan", restore_trained_point(run).plan)
    corpus = plan.corpus

    assert corpus is not None

    return tuple(
        example.item.item_id
        for example in corpus.examples
        if example.item.split == "validation"
    )


def selection(directory: Path, *roles: str, split: str = "validation") -> Path:
    path = directory / "selection.yaml"
    path.write_text(
        json.dumps({"evaluators": list(roles), "split": split, "formalism": "ccg"}),
        encoding="utf-8",
        newline="\n",
    )

    return path


def tokenizations(directory: Path, items: tuple[ItemId, ...]) -> Path:
    path = directory / "tokenizations.yaml"
    path.write_text(
        json.dumps(
            {
                "tokenizations": [
                    {"item": item_id, "tokenization": f"tokenization-{position}"}
                    for position, item_id in enumerate(items)
                ]
            }
        ),
        encoding="utf-8",
        newline="\n",
    )

    return path


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("published")
    overlay = root / "evaluators.yaml"
    overlay.write_text(EVALUATORS, encoding="utf-8", newline="\n")
    status = lofi(
        "run",
        "--config",
        str(CONFIG),
        "--extra",
        str(overlay),
        "--experiment-seed",
        "5",
        "--root",
        str(root / "store"),
    )

    assert status == 0

    (run,) = (
        path
        for path in (root / "store" / "runs").iterdir()
        if not path.name.startswith(".")
    )

    return run


def test_lofi_publish_writes_one_layers_record_per_item_of_the_split(
    trained_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    items = validation_items(trained_run)
    before = digests(trained_run)
    out = tmp_path / "records"
    capsys.readouterr()

    status = lofi(
        "publish",
        "--run",
        str(trained_run),
        "--tokenizations",
        str(tokenizations(tmp_path, items)),
        "--selection",
        str(selection(tmp_path, "decoded-types")),
        "--out",
        str(out),
    )

    assert status == 0
    assert items
    assert digests(trained_run) == before
    assert capsys.readouterr().out == (
        f"decoded-types: {len(items)} records, 0 unlabeled, "
        f"{out / 'decoded-types.jsonl'}\n"
    )

    lines = (out / "decoded-types.jsonl").read_text(encoding="utf-8").splitlines()

    for line in lines:
        annotation_records.AnnotationLayer.model_validate_json(line)

    layers = tuple(json.loads(line) for line in lines)

    assert tuple(layer["expression"] for layer in layers) == items
    assert tuple(layer["tokenizationId"]["value"] for layer in layers) == tuple(
        f"tokenization-{position}" for position in range(len(items))
    )

    for layer in layers:
        assert layer["formalism"] == "ccg"
        assert layer["sourceMethod"] == "automatic"
        assert layer["licensing"]["expression"] == "CC0-1.0"
        assert layer["metadata"]["tool"].startswith("lofi ")
        spans = {
            annotation["uuid"]["value"]: annotation["label"]
            for annotation in layer["annotations"]
        }
        length = max(int(span.split("-")[1]) for span in spans)

        assert spans[f"0-{length}"] == "S"
        assert all(f"{position}-{position + 1}" in spans for position in range(length))


@pytest.mark.parametrize("role", ["beams", "brackets"])
def test_a_role_that_yields_no_labeled_derivation_is_refused(
    trained_run: Path, tmp_path: Path, role: str
) -> None:
    items = validation_items(trained_run)
    out = tmp_path / "records"

    with pytest.raises(UnlabeledDerivationError, match=f"'{role}'"):
        publish_trained_point(
            trained_run,
            tokenizations(tmp_path, items),
            selection(tmp_path, role),
            out,
        )

    assert not out.exists()


def test_a_pilot_point_is_refused_before_any_record_is_written(
    trained_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pilot = tmp_path / "pilot-point"
    shutil.copytree(trained_run, pilot)
    manifest = read_manifest(pilot)

    assert manifest is not None

    write_binding(
        pilot, PointBinding(UNFROZEN, decode_seed_derivation(manifest.seed_derivation))
    )
    out = tmp_path / "records"

    with pytest.raises(PilotResultsError, match="pilot"):
        publish_trained_point(
            pilot,
            tokenizations(tmp_path, ()),
            selection(tmp_path, "decoded-types"),
            out,
        )

    capsys.readouterr()
    status = lofi(
        "publish",
        "--run",
        str(pilot),
        "--tokenizations",
        str(tokenizations(tmp_path, ())),
        "--selection",
        str(selection(tmp_path, "decoded-types")),
        "--out",
        str(out),
    )

    assert status == 2
    assert "pilot" in capsys.readouterr().err
    assert not out.exists()


def test_an_item_of_the_split_without_a_tokenization_is_refused(
    trained_run: Path, tmp_path: Path
) -> None:
    items = validation_items(trained_run)

    with pytest.raises(ValueError, match=repr(items[0])):
        publish_trained_point(
            trained_run,
            tokenizations(tmp_path, items[1:]),
            selection(tmp_path, "decoded-types"),
            tmp_path / "records",
        )


@pytest.mark.parametrize(
    ("roles", "split", "named"),
    [(("absent",), "validation", "'absent'"), (("decoded-types",), "test", "'test'")],
)
def test_an_undeclared_role_or_split_is_refused(
    trained_run: Path,
    tmp_path: Path,
    roles: tuple[str, ...],
    split: str,
    named: str,
) -> None:
    with pytest.raises(ValueError, match=named):
        publish_trained_point(
            trained_run,
            tokenizations(tmp_path, validation_items(trained_run)),
            selection(tmp_path, *roles, split=split),
            tmp_path / "records",
        )


def test_an_existing_destination_is_refused(trained_run: Path, tmp_path: Path) -> None:
    out = tmp_path / "records"
    out.mkdir()

    with pytest.raises(FileExistsError, match="records"):
        publish_trained_point(
            trained_run,
            tokenizations(tmp_path, validation_items(trained_run)),
            selection(tmp_path, "decoded-types"),
            out,
        )
