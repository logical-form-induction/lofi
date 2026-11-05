"""The installed parser-price fixture is pinned to the benchmark workload."""

from pathlib import Path
from typing import Final

from lofi.run._benchmark_fixture import packaged_pinned_chart

TREE: Final = Path(__file__).resolve().parents[3]


def test_the_packaged_fixture_passes_its_fingerprint_and_digest_checks() -> None:
    pinned = packaged_pinned_chart()
    packaged = TREE / "lofi" / "run" / "_fixtures"
    benchmark = TREE / "tests" / "benchmarks" / "fixtures"

    assert pinned.pinned.grammar == "cfg-toy"
    assert pinned.pinned.count == 200
    assert len(pinned.sentences) == pinned.pinned.count
    assert {len(sentence) for sentence in pinned.sentences} == {20}
    for name in ("chart_200.jsonl", "chart_200.fingerprint"):
        assert (packaged / name).read_bytes() == (benchmark / name).read_bytes()
