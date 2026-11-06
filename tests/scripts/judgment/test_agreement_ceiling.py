import json
from typing import TYPE_CHECKING

import pytest
from scripts.judgment.agreement_ceiling import grouped, main
from tests.unit.run.test_corpus_designs import (
    ACCEPTABILITY_PATH,
    ACCEPTABILITY_RELEASE,
    ACCEPTABILITY_SPEC,
    install,
    use,
)

from lofi.judgment import ReadDesign, split_half_ceiling
from lofi.run.fits import RegisteredDesigns

if TYPE_CHECKING:
    from pathlib import Path

    from lofi.judgment import JudgmentDesign


def _design() -> JudgmentDesign:
    read = RegisteredDesigns().read("megaacceptability")
    assert isinstance(read, ReadDesign)
    return read.design


def test_responses_are_grouped_by_the_declared_factors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(install(tmp_path / "data"), monkeypatch)
    design = _design()

    by_verb = grouped(design, ("verb",))
    by_both = grouped(design, ("verb", "frame"))

    assert set(by_verb) == {("think",), ("try",), ("want",)}
    assert sorted(len(held) for held in by_verb.values()) == [1, 1, 2]
    assert set(by_both) == {
        ("think", "NP _ that S"),
        ("try", "NP _ to VP"),
        ("want", "NP _ to VP"),
    }
    with pytest.raises(ValueError, match="no factor"):
        grouped(design, ("tense",))
    with pytest.raises(ValueError, match="at least one factor"):
        grouped(design, ())


HEADER = (
    "participant\tlist\tpresentationorder\tverb\tframe\ttense\tresponse\t"
    "nativeenglish\tsentence\tversion\n"
)
SENTENCES = {
    "think": ("NP _ that S", "Dana thinks that Lee left."),
    "want": ("NP _ to VP", "Dana wants to leave."),
    "try": ("NP _ to VP", "Dana tries to leave."),
    "hope": ("NP _ that S", "Dana hopes that Lee left."),
}


def test_the_ceiling_record_names_the_design_and_its_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = install(tmp_path / "data")
    rows = "".join(
        f"p{participant}\t1\t{order}\t{verb}\t{frame}\tpresent\t"
        f"{(order + participant * order % 5) % 7 + 1}.0\tyes\t{sentence}\t1\n"
        for participant in range(6)
        for order, (verb, (frame, sentence)) in enumerate(SENTENCES.items(), start=1)
    )
    (root / ACCEPTABILITY_PATH).write_text(HEADER + rows, encoding="utf-8")
    release = dict(ACCEPTABILITY_SPEC)
    release["verb_refs"] = {
        verb: f"at://did:plc:test/pub.layers.resource.entry/{verb}"
        for verb in SENTENCES
    }
    (root / ACCEPTABILITY_RELEASE).write_text(json.dumps(release), encoding="utf-8")
    use(root, monkeypatch)
    out = tmp_path / "ceiling.json"
    design = _design()
    expected = split_half_ceiling(
        grouped(design, ("verb",)).values(), splits=2, level=0.9
    )

    status = main(
        [
            "megaacceptability",
            "--group",
            "verb",
            "--splits",
            "2",
            "--level",
            "0.9",
            "--out",
            str(out),
        ]
    )
    written = json.loads(out.read_text(encoding="utf-8"))

    assert status == 0
    assert written["dataset"] == "megaacceptability"
    assert written["group_by"] == ["verb"]
    assert written["items"] == expected.items
    assert "designs/megaacceptability.json" in written["source_digests"]
