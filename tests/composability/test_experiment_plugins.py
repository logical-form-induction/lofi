"""A distribution can extend every experiment-execution plugin family."""

import os
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _write_plugin(root: Path, module: str) -> None:
    """Install four experiment plugins through distribution metadata."""
    source = """from pathlib import Path
from typing import Literal

from lofi.run import (
    CorrectionPlugin,
    CorrectionSpec,
    ExperimentConfigPlugin,
    ExperimentConfigSpec,
    LaunchContext,
    Launcher,
    LauncherPlugin,
    LaunchSpec,
    RunHandle,
    RunStatus,
    SweepAssignment,
    SweepDesignPlan,
    SweepDesignPlugin,
    SweepDesignSpec,
    SweepPoint,
)
from lofi.stats import Correction, Holm


class ExternalCorrectionSpec(CorrectionSpec):
    kind: Literal["third-party-correction"] = "third-party-correction"


def build_correction(declaration: ExternalCorrectionSpec) -> Correction:
    del declaration
    return Holm()


CORRECTION = CorrectionPlugin(
    spec=ExternalCorrectionSpec,
    build=build_correction,
)


class ExternalConfigSpec(ExperimentConfigSpec):
    kind: Literal["third-party-config"] = "third-party-config"


def build_config(declaration: ExternalConfigSpec, path: Path) -> str:
    return f"{declaration.kind}:{path.name}"


EXPERIMENT_CONFIG = ExperimentConfigPlugin(
    spec=ExternalConfigSpec,
    build=build_config,
)


class ExternalLaunchSpec(LaunchSpec):
    kind: Literal["third-party-launcher"] = "third-party-launcher"


class ExternalLauncher:
    def admitted(self, points):
        return tuple(points)

    def rehearse(self, points):
        return f"{len(points)} points would go to the external service"

    def submit(self, points):
        return [RunHandle(point.point_id, "external") for point in points]

    def status(self, handle):
        del handle
        return RunStatus.COMPLETED


def build_launcher(
    declaration: ExternalLaunchSpec, context: LaunchContext
) -> Launcher:
    del declaration, context
    return ExternalLauncher()


LAUNCHER = LauncherPlugin(
    spec=ExternalLaunchSpec,
    build=build_launcher,
    requires_site=False,
)


class ExternalSweepDesignSpec(SweepDesignSpec):
    kind: Literal["third-party-design"] = "third-party-design"


def assignments(
    declaration: ExternalSweepDesignSpec, sweep: SweepDesignPlan
) -> tuple[SweepAssignment, ...]:
    del declaration
    axis = sweep.grid[0]
    return (((axis, axis.points[-1]),),)


SWEEP_DESIGN = SweepDesignPlugin(
    spec=ExternalSweepDesignSpec,
    assign=assignments,
)
"""
    (root / f"{module}.py").write_text(source, encoding="utf-8", newline="\n")
    metadata = root / "lofi_experiment_plugins-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: lofi-experiment-plugins\nVersion: 1.0\n",
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "entry_points.txt").write_text(
        f"[lofi.correction]\nthird-party-correction = {module}:CORRECTION\n"
        f"[lofi.experiment_config]\nthird-party-config = "
        f"{module}:EXPERIMENT_CONFIG\n"
        f"[lofi.launcher]\nthird-party-launcher = {module}:LAUNCHER\n"
        f"[lofi.sweep_design]\nthird-party-design = {module}:SWEEP_DESIGN\n",
        encoding="utf-8",
        newline="\n",
    )


def test_installed_experiment_plugins_validate_and_execute_from_documents(
    tmp_path: Path,
) -> None:
    module = f"lofi_experiment_plugins_{tmp_path.name.replace('-', '_')}"
    _write_plugin(tmp_path, module)
    script = r"""from pathlib import Path

from lofi.run import (
    Axis,
    CorrectionSpec,
    ExperimentConfigSpec,
    LaunchSpec,
    RunStatus,
    Sweep,
    SweepDesignSpec,
    SweepPoint,
    correction,
    design_assignments,
    experiment_config,
    launcher_plugin,
)
from lofi.spec import load_plugins


class Context:
    def __getattr__(self, name):
        raise AssertionError(f"the external launcher requested {name}")


loaded = load_plugins()
assert "third-party-correction" in loaded
assert "third-party-config" in loaded
assert "third-party-launcher" in loaded
assert "third-party-design" in loaded

correction_spec = CorrectionSpec.model_validate({"kind": "third-party-correction"})
assert type(correction_spec).__name__ == "ExternalCorrectionSpec"
assert correction(correction_spec).name == "holm"

config_spec = ExperimentConfigSpec.model_validate({
    "kind": "third-party-config",
    "path": "external.yaml",
})
assert type(config_spec).__name__ == "ExternalConfigSpec"
assert experiment_config(config_spec, Path("external.yaml")) == (
    "third-party-config:external.yaml"
)

launch_spec = LaunchSpec.model_validate({"kind": "third-party-launcher"})
external = launcher_plugin(launch_spec).construct(launch_spec, Context())
assert external.rehearse(()) == "0 points would go to the external service"
point = SweepPoint({"model": {"width": 8}})
(handle,) = external.submit((point,))
assert handle.point_id == point.point_id
assert external.status(handle) is RunStatus.COMPLETED

design_spec = SweepDesignSpec.model_validate({"kind": "third-party-design"})
sweep = Sweep(
    design=design_spec,
    grid=(Axis(path="model.width", values=(8, 16)),),
)
(((_, width),),) = design_assignments(sweep.design, sweep)
assert width == 16
print("experiment plugins executed")
"""
    python_path = os.pathsep.join((str(tmp_path), os.environ.get("PYTHONPATH", "")))
    completed = subprocess.run(
        [sys.executable, "-"],
        input=script,
        check=False,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONPATH": python_path},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "experiment plugins executed\n"
    assert module not in sys.modules
