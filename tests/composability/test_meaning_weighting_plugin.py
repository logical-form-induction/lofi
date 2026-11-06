"""An installed package can supply a logical-form meaning weighting."""

from typing import TYPE_CHECKING

from tests.language import language

from lofi.spec.readouts import (
    ProofTermMeaningSettings,
    ProofTermMeaningSpec,
    TermsReadoutSpec,
)
from lofi.typecheck import meaning_shares, meaning_weighting

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _install_weighting(root: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Install one weighting through distribution metadata."""
    module = f"lofi_weighting_plugin_{root.name.replace('-', '_')}"
    (root / f"{module}.py").write_text(
        """from collections.abc import Mapping, Sequence
from typing import Final

from lofi._frozen import FrozenValue
from lofi.terms import Term
from lofi.typecheck import MeaningWeightingPlugin


class FinalWeighting(FrozenValue):
    _noun = "third-party meaning weighting"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "test-final"

    def shares(self, reached: Sequence[Term], /) -> Mapping[Term, float]:
        distinct = tuple(dict.fromkeys(reached))
        return {meaning: float(meaning is distinct[-1]) for meaning in distinct}


PLUGIN: Final = MeaningWeightingPlugin(FinalWeighting)
""",
        encoding="utf-8",
        newline="\n",
    )
    metadata = root / "lofi_weighting_fixture-1.0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: lofi-weighting-fixture\nVersion: 1.0.0\n",
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "entry_points.txt").write_text(
        f"[lofi.meaning_weighting]\ntest-final = {module}:PLUGIN\n",
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.syspath_prepend(str(root))

    return "test-final"


def test_external_meaning_weighting_is_selected_from_a_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = _install_weighting(tmp_path, monkeypatch)
    declared = TermsReadoutSpec.model_validate(
        {
            "kind": "terms",
            "space": "forms",
            "reads": "inside",
            "decoding": {"strategy": "greedy"},
            "meanings": {
                "kind": "proof-term",
                "weighting": name,
                "residual_mass": 0.999,
            },
        }
    )
    settings = declared.settings().meanings
    assert isinstance(settings, ProofTermMeaningSettings)
    selected = meaning_weighting(settings.weighting)
    assembly = language(("first", "second"))
    first = assembly.constant("first")
    second = assembly.constant("second")

    assert selected.name == name
    assert meaning_shares(selected, (first, second)) == {second: 1.0}
    assert isinstance(declared.meanings, ProofTermMeaningSpec)
    assert declared.meanings.weighting == name
