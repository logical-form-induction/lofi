from __future__ import annotations

import json
import math
from hashlib import sha256
from typing import TYPE_CHECKING, Final, NamedTuple, override

import pytest
import torch
from torch import nn

from lofi._artifacts import PinnedCheckpoint
from lofi.eval._boundary import (
    RootMassDocument,
    RootMassItemDocument,
    encode_root_mass,
)
from lofi.judgment import ReadBoundedDesign, ReadDesign
from lofi.run.corpus_designs import (
    DATA_ROOT_VARIABLE,
    MEGAACCEPTABILITY,
    MEGAINTENSIONALITY,
    MEGANEGRAISING,
    MEGAVERIDICALITY,
    AbsentCorpusError,
    DeclaredDesign,
    parse_unigram_table,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from torch import Tensor

    from lofi._types import JsonValue

ACCEPTABILITY_PATH: Final = "releases/mega-acceptability-v2/mega-acceptability-v2.tsv"
ACCEPTABILITY_RELEASE: Final = "releases/mega-acceptability-v2/release.json"
VERIDICALITY_PATH: Final = "releases/mega-veridicality-v2.1/mega-veridicality-v2.tsv"
VERIDICALITY_RELEASE: Final = "releases/mega-veridicality-v2.1/release.json"
NEGRAISING_PATH: Final = "releases/mega-negraising-v1/mega-negraising-v1.tsv"
NEGRAISING_RELEASE: Final = "releases/mega-negraising-v1/release.json"
INTENSIONALITY_PATH: Final = (
    "releases/mega-intensionality-v1/mega-intensionality-v1.tsv"
)
INTENSIONALITY_RELEASE: Final = "releases/mega-intensionality-v1/release.json"
UNIGRAMS_PATH: Final = "releases/morcela/unigrams.tsv"
ROOT_MASS_PATH: Final = "releases/lofi/root-mass.json"
MEGAACCEPTABILITY_TAG: Final = "megaacceptability"
MEGAACCEPTABILITY_MORCELA_TAG: Final = "megaacceptability-morcela"
MEGAVERIDICALITY_TAG: Final = "megaveridicality"
MEGANEGRAISING_TAG: Final = "meganegraising"
MEGAINTENSIONALITY_TAG: Final = "megaintensionality"

ACCEPTABILITY: Final = (
    "participant\tlist\tpresentationorder\tverb\tframe\ttense\tresponse\t"
    "nativeenglish\tsentence\tversion\n"
    "p01\t1\t1\tthink\tNP _ that S\tpresent\t7.0\tyes\t"
    "Dana thinks that Lee left.\t1\n"
    "p01\t1\t2\twant\tNP _ to VP\tpresent\t2.0\tyes\t"
    "Dana wants to leave.\t1\n"
    "p01\t1\t5\ttry\tNP _ to VP\tpresent\t6.0\tyes\t"
    "Dana tries to leave.\t1\n"
    "p02\t1\t3\tthink\tNP _ that S\tpresent\t6.0\tyes\t"
    "Dana thinks that Lee left.\t1\n"
    "p02\t1\t4\twant\tNP _ to VP\tpresent\t\tyes\t"
    "Dana wants to leave.\t1\n"
)
VERIDICALITY: Final = (
    "participant\tlist\tpresentationorder\tverb\tframe\tpolarity\tconditional\t"
    "sentence\tveridicality\tacceptability\tnativeenglish\n"
    "487\t0\t1\tsurmise\tNP Ved that S\tpositive\tFalse\t"
    "Someone surmised that a particular thing happened\tmaybe\t6\tTrue\n"
    "585\t32\t1\twant\tNP Ved to VP[+eventive]\tpositive\tFalse\t"
    "A particular person wanted to do a particular thing.\tyes\t7\tTrue\n"
    "585\t32\t2\topt\tNP Ved to VP[-eventive]\tnegative\tFalse\t"
    "A particular person didn't opt to have a particular thing.\t\t7\tTrue\n"
)
NEGRAISING: Final = (
    "participant\tlist\tpresentationorder\tverb\tframe\ttense\tsubject\t"
    "sentence1\tsentence2\tnegraising\tacceptability\tnativeenglish\n"
    "229\t81\t8\tabhor\tNP V that S\tpast\tfirst\t"
    "I didn't abhor that a particular thing happened.\t"
    "I abhorred that that thing didn't happen.\t0.30\t0.97\tTrue\n"
)
INTENSIONALITY: Final = (
    "participant\tantecedent_verb\tantecedent_polarity\ttarget\t"
    "antecedent_frame\tconsequent_verb\tconsequent_embedded_tense\t"
    "antecedent_text\tconsequent_text\tresponse\tsanity\n"
    "70\tconsider\tpositive\tsubject\tNP V that S\twant\tpast\t"
    "A considered that C happened\tA wanted C to have happened\t0.780\t4\n"
)
ACCEPTABILITY_SPEC: Final[dict[str, JsonValue]] = {
    "source_id": "synthetic-megaacceptability",
    "release_id": "v2",
    "spdx": "CC-BY-SA-4.0",
    "scale_min": 1,
    "scale_max": 7,
    "frame_refs": {
        "NP _ that S": "at://did:plc:test/pub.layers.resource.template/that-s",
        "NP _ to VP": "at://did:plc:test/pub.layers.resource.template/to-vp",
    },
    "verb_refs": {
        verb: f"at://did:plc:test/pub.layers.resource.entry/{verb}"
        for verb in ("think", "want", "try")
    },
}
ATTITUDE_SPEC: Final[dict[str, JsonValue]] = {
    "source_id": "synthetic-megaattitude",
    "release_id": "v1",
    "language": "en",
    "source_method": "elicited",
    "spdx": "CC-BY-SA-4.0",
}
SENTENCES: Final = (
    "Dana thinks that Lee left.",
    "Dana wants to leave.",
    "Dana tries to leave.",
)
WORDS: Final = ("<s>", *dict.fromkeys(word for s in SENTENCES for word in s.split()))
UNIGRAMS: Final = "token\tcount\n" + "".join(
    f"{word}\t{index}\n" for index, word in enumerate(WORDS[1:], start=1)
)
FIXTURE_CHECKPOINT: Final = PinnedCheckpoint("lofi-test/a-word-model", "1" * 40)


def _write(root: Path, relative: str, data: str | Mapping[str, JsonValue]) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        data if isinstance(data, str) else json.dumps(data),
        encoding="utf-8",
        newline="\n",
    )
    return path


def fragment(
    codec: str,
    declaration: str,
    responses: str,
    reading: dict[str, JsonValue],
    factors: tuple[str, ...],
) -> dict[str, JsonValue]:
    return {
        "source": "fragment",
        "fragment": {
            "kind": "codec",
            "codec": codec,
            "declaration": declaration,
            "responses": responses,
        },
        "responses": reading,
        "factors": {factor: f"item['{factor}']" for factor in factors},
        "readouts": None,
        "keep": None,
    }


def install(root: Path) -> Path:
    _write(root, ACCEPTABILITY_PATH, ACCEPTABILITY)
    _write(root, ACCEPTABILITY_RELEASE, ACCEPTABILITY_SPEC)
    _write(root, VERIDICALITY_PATH, VERIDICALITY)
    _write(root, VERIDICALITY_RELEASE, {**ATTITUDE_SPEC, "release_id": "v2.1"})
    _write(root, NEGRAISING_PATH, NEGRAISING)
    _write(root, NEGRAISING_RELEASE, {**ATTITUDE_SPEC, "release_id": "v1.1"})
    _write(root, INTENSIONALITY_PATH, INTENSIONALITY)
    _write(root, INTENSIONALITY_RELEASE, ATTITUDE_SPEC)
    _write(root, UNIGRAMS_PATH, UNIGRAMS)
    documents: dict[str, dict[str, JsonValue]] = {
        MEGAACCEPTABILITY_TAG: fragment(
            "megaacceptability",
            ACCEPTABILITY_RELEASE,
            ACCEPTABILITY_PATH,
            {"kind": "ordinal"},
            ("verb", "frame"),
        ),
        MEGAACCEPTABILITY_MORCELA_TAG: {
            **fragment(
                "megaacceptability",
                ACCEPTABILITY_RELEASE,
                ACCEPTABILITY_PATH,
                {"kind": "ordinal"},
                ("verb", "frame"),
            ),
            "readouts": {
                "kind": "causal-lm",
                "element": "stimulus",
                "unigrams": UNIGRAMS_PATH,
                "columns": {
                    "log_probability": "lm_log_probability",
                    "unigram_log_probability": "unigram_log_probability",
                    "length": "length",
                },
                "artifacts": {"root": "/var/cache/lofi/artifacts", "offline": True},
                "checkpoint": {
                    "kind": "hub",
                    "repository": FIXTURE_CHECKPOINT.repository,
                    "revision": FIXTURE_CHECKPOINT.revision,
                },
            },
        },
        MEGAVERIDICALITY_TAG: fragment(
            "megaveridicality",
            VERIDICALITY_RELEASE,
            VERIDICALITY_PATH,
            {"kind": "relation", "inventory": ["contradicts", "neutral", "entails"]},
            ("verb", "frame", "polarity"),
        ),
        MEGANEGRAISING_TAG: fragment(
            "meganegraising",
            NEGRAISING_RELEASE,
            NEGRAISING_PATH,
            {"kind": "bounded"},
            ("verb", "frame", "tense"),
        ),
        MEGAINTENSIONALITY_TAG: fragment(
            "megaintensionality",
            INTENSIONALITY_RELEASE,
            INTENSIONALITY_PATH,
            {"kind": "bounded"},
            ("verb", "frame", "target"),
        ),
    }
    for tag, document in documents.items():
        _write(root, f"designs/{tag}.json", document)
    return root


def use(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_ROOT_VARIABLE, str(root))


def test_megaacceptability_reads_the_published_v2_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(install(tmp_path / "data"), monkeypatch)
    read = MEGAACCEPTABILITY.read()

    assert isinstance(read, ReadDesign)
    assert read.design.n_rows == 4
    assert tuple(read.design.factors) == ("verb", "frame")
    assert sorted(read.design.responses.tolist()) == [1, 5, 5, 6]


def test_megaacceptability_joins_declared_published_root_masses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = install(tmp_path / "data")
    use(root, monkeypatch)
    items = MEGAACCEPTABILITY.read().design.item_levels
    artifact = RootMassDocument(
        readout="root-mass",
        items=tuple(
            RootMassItemDocument(
                item_id=item,
                log_mass=-(position + 1.0),
                log_complement=-(position + 4.0),
            )
            for position, item in enumerate(items)
        ),
    )
    path = root / ROOT_MASS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_root_mass(artifact))
    document = json.loads(
        (root / f"designs/{MEGAACCEPTABILITY_TAG}.json").read_text(encoding="utf-8")
    )
    document["readouts"] = {
        "kind": "root-mass",
        "path": ROOT_MASS_PATH,
        "mass": "p_gram",
        "complement": "p_gram_complement",
    }
    _write(root, f"designs/{MEGAACCEPTABILITY_TAG}.json", document)

    read = MEGAACCEPTABILITY.read()

    assert tuple(read.design.readouts) == ("p_gram", "p_gram_complement")
    assert read.design.readouts["p_gram"].tolist() == [
        -(index + 1.0) for index in read.design.items.tolist()
    ]
    assert read.design.readouts["p_gram_complement"].tolist() == [
        -(index + 4.0) for index in read.design.items.tolist()
    ]
    assert read.provenance.corpus_digests[ROOT_MASS_PATH] == _digest(path)


def test_megaveridicality_reads_responded_items_with_participant_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(install(tmp_path / "data"), monkeypatch)
    read = MEGAVERIDICALITY.read()

    assert isinstance(read, ReadDesign)
    assert read.design.n_rows == 2
    assert read.design.response_levels == 3
    assert sorted(read.design.participant_levels) == ["487", "585"]
    rows = sorted(
        (read.design.participant_levels[participant], response)
        for participant, response in zip(
            read.design.participants.tolist(),
            read.design.responses.tolist(),
            strict=True,
        )
    )
    assert rows == [("487", 1), ("585", 2)]
    assert tuple(read.design.factors) == (
        "verb",
        "frame",
        "polarity",
    )
    assert len(read.design.item_levels) == 2


@pytest.mark.parametrize(
    ("reader", "participant", "value"),
    [(MEGANEGRAISING, "229", 0.3), (MEGAINTENSIONALITY, "70", 0.78)],
)
def test_likelihood_releases_read_as_bounded_designs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader: DeclaredDesign,
    participant: str,
    value: float,
) -> None:
    use(install(tmp_path / "data"), monkeypatch)
    read = reader.read()

    assert isinstance(read, ReadBoundedDesign)
    assert read.design.n_rows == 1
    assert read.design.participant_levels == (participant,)
    assert read.design.responses.tolist() == pytest.approx([value])


def test_one_data_root_owns_every_release_and_design_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = install(tmp_path / "data")
    use(root, monkeypatch)
    provenance = MEGAVERIDICALITY.read().provenance

    assert provenance.corpus_digests == {
        f"designs/{MEGAVERIDICALITY_TAG}.json": _digest(
            root / f"designs/{MEGAVERIDICALITY_TAG}.json"
        ),
        VERIDICALITY_PATH: _digest(root / VERIDICALITY_PATH),
        VERIDICALITY_RELEASE: _digest(root / VERIDICALITY_RELEASE),
    }


def test_a_factor_reading_an_attribute_the_items_do_not_carry_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = install(tmp_path / "data")
    use(root, monkeypatch)
    path = root / f"designs/{MEGAACCEPTABILITY_TAG}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["factors"] = {"structure": "item['structure']"}
    _write(root, f"designs/{MEGAACCEPTABILITY_TAG}.json", document)

    with pytest.raises(ValueError, match="factor 'structure'"):
        MEGAACCEPTABILITY.read()


def test_a_judgment_filter_keeps_the_judgments_it_holds_of(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = install(tmp_path / "data")
    use(root, monkeypatch)
    path = f"designs/{MEGAACCEPTABILITY_TAG}.json"
    document = json.loads((root / path).read_text(encoding="utf-8"))
    document["keep"] = "judgment['participant'] == 'p01'"
    _write(root, path, document)

    with pytest.raises(ValueError, match="judgment filter"):
        MEGAACCEPTABILITY.read()

    document["keep"] = "judgment['presentationorder'] != '1'"
    _write(root, path, document)

    assert MEGAACCEPTABILITY.read().design.n_rows == 3


def test_a_feature_judgments_of_one_item_state_differently_is_no_attribute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = install(tmp_path / "data")
    use(root, monkeypatch)
    path = f"designs/{MEGAACCEPTABILITY_TAG}.json"
    document = json.loads((root / path).read_text(encoding="utf-8"))
    document["factors"] = {"order": "item['presentationorder']"}
    _write(root, path, document)

    with pytest.raises(ValueError, match="factor 'order'"):
        MEGAACCEPTABILITY.read()


def test_an_unset_data_root_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DATA_ROOT_VARIABLE, raising=False)

    with pytest.raises(AbsentCorpusError, match=DATA_ROOT_VARIABLE):
        MEGAACCEPTABILITY.read()


@pytest.mark.parametrize(
    "missing",
    [
        f"designs/{MEGAVERIDICALITY_TAG}.json",
        VERIDICALITY_PATH,
        VERIDICALITY_RELEASE,
    ],
)
def test_every_declared_input_must_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    root = install(tmp_path / "data")
    (root / missing).unlink()
    use(root, monkeypatch)

    with pytest.raises(AbsentCorpusError, match=missing):
        MEGAVERIDICALITY.read()


class Output(NamedTuple):
    logits: Tensor


class WordModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(17)
        self.weights = nn.Parameter(
            torch.randn(len(WORDS), len(WORDS), generator=generator)
        )

    @override
    def forward(self, *, input_ids: Tensor) -> Output:
        return Output(self.weights[input_ids])


class WordTokenizer:
    bos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [WORDS.index(word) for word in text.split()]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [WORDS[index] for index in ids]


class WordLoader:
    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def load_causal_model(self, checkpoint: PinnedCheckpoint) -> WordModel:
        if checkpoint != FIXTURE_CHECKPOINT:
            raise ValueError(checkpoint)
        return WordModel()

    def load_sentence_tokenizer(self, checkpoint: PinnedCheckpoint) -> WordTokenizer:
        if checkpoint != FIXTURE_CHECKPOINT:
            raise ValueError(checkpoint)
        return WordTokenizer()


def test_a_causal_model_scores_the_declared_element_of_every_rated_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = install(tmp_path / "data")
    use(root, monkeypatch)
    read = DeclaredDesign(MEGAACCEPTABILITY_MORCELA_TAG, lambda _: WordLoader()).read()

    assert tuple(read.design.readouts) == (
        "lm_log_probability",
        "unigram_log_probability",
        "length",
    )
    assert read.provenance.checkpoint == FIXTURE_CHECKPOINT.identity
    assert read.provenance.checkpoint_dtype == "float32"
    assert read.provenance.unigram_digest == _digest(root / UNIGRAMS_PATH)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_a_unigram_table_is_read_as_counts_per_token() -> None:
    table = parse_unigram_table(
        b'token\tcount\n\xc4\xa0the\t3\n"\t1\n', source="unigrams.tsv"
    )

    assert len(table) == 2
    assert table.log_frequencies(("Ġthe", '"')).tolist() == pytest.approx(
        [math.log(0.75), math.log(0.25)]
    )


def test_a_unigram_table_with_crlf_line_ends_reads_as_its_lf_spelling() -> None:
    lf = b"token\tcount\nthe\t3\ncat\t1\n"
    table = parse_unigram_table(lf.replace(b"\n", b"\r\n"), source="unigrams.tsv")

    assert len(table) == 2
    assert table.log_frequencies(("the", "cat")).tolist() == pytest.approx(
        parse_unigram_table(lf, source="unigrams.tsv")
        .log_frequencies(("the", "cat"))
        .tolist()
    )


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("word\tcount\nthe\t3\n", "header"),
        ("token\tcount\nthe\n", "line 2"),
        ("token\tcount\nthe\tmany\n", "line 2"),
        ("token\tcount\nthe\t3\nthe\t4\n", "repeats"),
        ("token\tcount\nthe\t0\n", "count"),
        ("token\tcount\n", "no tokens"),
    ],
)
def test_a_malformed_unigram_table_is_refused(text: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        parse_unigram_table(text.encode(), source="unigrams.tsv")
