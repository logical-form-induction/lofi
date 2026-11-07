import subprocess
import sys


def test_importing_the_package_does_not_import_component_declarations() -> None:
    source = """import sys
import lofi.spec.featurizers
loaded = tuple(
    name for name in sys.modules
    if name.startswith('lofi.spec.featurizers.')
)
assert loaded == (), loaded
"""
    subprocess.run((sys.executable, "-c", source), check=True)
