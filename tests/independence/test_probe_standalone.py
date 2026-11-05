import subprocess
import sys
from pathlib import Path
from typing import Final

TREE: Final = Path(__file__).resolve().parents[2]
TIMEOUT_SECONDS: Final = 10.0

PROBE_SMOKE: Final = """
import lofi.probe as probe

declaration = probe.SiteDecl(
    role="CHART_CELL",
    axes=("item", "span", "width"),
    dtype_class="real",
    differentiable=True,
    access=("OBSERVE",),
)
published = probe.SiteMapIso().forward(
    probe.SiteMapSpec(
        component="standalone-probe", kind="custom", sites=(declaration,)
    )
)

class Published:
    def site_map(self):
        return published

context = probe.DiagnosticContext.observing(Published())

if len(context.published) != 1:
    raise RuntimeError("the public context lost the published site")
"""


def run_probe_smoke() -> subprocess.CompletedProcess[str]:
    """Run the public probe smoke test in a fresh interpreter.

    Returns
    -------
    subprocess.CompletedProcess[str]
        The completed interpreter process.
    """
    return subprocess.run(
        [sys.executable, "-c", PROBE_SMOKE],
        capture_output=True,
        encoding="utf-8",
        check=False,
        cwd=TREE,
        timeout=TIMEOUT_SECONDS,
    )


def test_the_probe_package_constructs_and_observes_standalone() -> None:
    result = run_probe_smoke()

    assert result.returncode == 0, result.stderr
