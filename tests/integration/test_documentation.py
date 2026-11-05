from pathlib import Path
from typing import Final

import pytest
import torch
from examples.external_site_map import run_example

from lofi import _checks
from lofi.run import MANIFEST_NAME, PRICE_NAME, read_price
from lofi.run._entrypoint import main

TREE: Final = Path(__file__).resolve().parents[2]
CONFIG: Final = TREE / "docs/examples/synthetic-parser.yaml"
EXPERIMENT: Final = TREE / "docs/examples/experiments/synthetic.yaml"
POINT_DIRECTORY: Final = "[0-9a-f]" * 16


@pytest.mark.integration
def test_the_external_site_map_example_executes_the_documented_public_route() -> None:
    result = run_example()

    assert torch.equal(result.observed, result.given)
    assert result.observed_axes == ("batch", "feature")
    assert torch.equal(result.baseline, result.given * 2.0)
    assert torch.equal(result.patched, torch.zeros_like(result.baseline))
    assert not torch.equal(result.patched, result.baseline)
    assert torch.equal(result.after_patch, result.baseline)
    assert result.failure == "the tiny model refused this input"
    assert torch.equal(result.after_failure, result.baseline)
    assert "is not published by this component" in result.unknown_site_refusal
    assert torch.equal(result.after_refusal, result.baseline)


@pytest.mark.integration
def test_the_checked_in_configuration_runs_through_the_production_executor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"

    status = main(
        [
            "run",
            "--config",
            str(CONFIG),
            "--root",
            str(root),
            "--seed",
            "11",
            "--experiment-seed",
            "5",
        ]
    )
    directories = tuple((root / "runs").glob(POINT_DIRECTORY))

    assert status == 0
    assert len(directories) == 1
    assert (directories[0] / MANIFEST_NAME).is_file()
    assert tuple((directories[0] / "checkpoints").iterdir())


@pytest.mark.integration
def test_the_documented_configuration_commands_use_public_workflows() -> None:
    assert main(["configs"]) == 0
    assert (
        main(
            [
                "show-config",
                "--config",
                str(CONFIG),
                "--set",
                "trainer.epochs=2",
            ]
        )
        == 0
    )


@pytest.mark.integration
def test_the_documented_price_command_writes_its_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "price"
    monkeypatch.setattr(_checks, "CHECKS", False)

    status = main(
        [
            "price",
            "--experiment",
            str(EXPERIMENT),
            "--device",
            "cpu",
            "--warmup",
            "1",
            "--repeats",
            "5",
            "--root",
            str(root),
        ]
    )
    (written,) = (root / "sweeps").glob(f"*/{PRICE_NAME}")
    price = read_price(written.parent)

    assert status == 0
    assert price is not None
    assert len(price.stacks) == 2
    assert price.p50_hours <= price.p90_hours
