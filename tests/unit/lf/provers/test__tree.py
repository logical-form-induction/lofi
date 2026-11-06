import subprocess
import sys
from typing import Final

from lofi.lf.provers._tree import ProcessTree

PARENT: Final = """
import subprocess
import sys
import time

subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"], stdout=sys.stdout
)
print("ready", flush=True)
time.sleep(60)
"""
"""A process that starts a child holding the same standard output, then waits.

The output pipe reaches its end only once both have exited.
"""


def _started() -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", PARENT],
        stdout=subprocess.PIPE,
        encoding="utf-8",
        start_new_session=True,
    )


def _ready(process: subprocess.Popen[str]) -> None:
    stdout = process.stdout

    assert stdout is not None
    assert stdout.readline() == "ready\n"


def test_killing_a_tree_ends_the_process_and_its_child() -> None:
    process = _started()

    try:
        with ProcessTree(process.pid) as tree:
            _ready(process)
            tree.kill()
            remaining, _ = process.communicate(timeout=30)

    finally:
        process.kill()
        process.wait()

    assert remaining == ""
    assert process.returncode != 0


def test_leaving_a_tree_kills_whatever_still_runs() -> None:
    process = _started()

    try:
        with ProcessTree(process.pid):
            _ready(process)

        remaining, _ = process.communicate(timeout=30)

    finally:
        process.kill()
        process.wait()

    assert remaining == ""
    assert process.returncode != 0
