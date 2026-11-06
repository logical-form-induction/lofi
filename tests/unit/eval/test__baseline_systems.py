import json
import subprocess
import sys

from lofi.eval._baseline_systems import BISK_HOCKENMAIER_PLUGIN
from lofi.spec.baselines.bisk_hockenmaier import BiskHockenmaierBaselineSpec

PROBE = """\
import json
import sys

import lofi.eval._baseline_systems

print(json.dumps(sorted(
    name for name in ("lofi.eval._bisk_hockenmaier_baseline", "lairs", "panproto")
    if name in sys.modules
)))
"""
"""Report which implementations importing the baseline plugins loads."""


def test_importing_the_plugins_loads_no_baseline_implementation() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert json.loads(completed.stdout) == []


def test_the_bisk_hockenmaier_plugin_declares_its_baseline() -> None:
    assert BISK_HOCKENMAIER_PLUGIN.spec is BiskHockenmaierBaselineSpec
