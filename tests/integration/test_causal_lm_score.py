"""A pinned public causal model scores the MegaAcceptability fixture design."""

import csv
import io
import json
from collections import Counter
from pathlib import Path
from shutil import copytree
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch
from tests.conformance.lm_score import LmScoreSubject, LmScoreSuite
from tests.unit.run.test_corpus_designs import (
    ACCEPTABILITY,
    MEGAACCEPTABILITY_MORCELA_TAG,
    UNIGRAMS_PATH,
    install,
)

from lofi._artifacts import ArtifactFetcher, HttpArchiveTransport, PinnedCheckpoint
from lofi.judgment.language_model import (
    CausalLmScore,
    UnigramTable,
)
from lofi.nn._snapshots import HubSnapshotTransport
from lofi.nn.tokens.pretrained import HuggingFaceCausalLoader
from lofi.run.corpus_designs import (
    DATA_ROOT_VARIABLE,
    DESIGNS_DIRECTORY,
    DeclaredDesign,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

LM_LOG_PROBABILITY: Final = "lm_log_probability"
UNIGRAM_LOG_PROBABILITY: Final = "unigram_log_probability"
SENTENCE_LENGTH: Final = "length"


pytestmark = pytest.mark.integration

transformers = pytest.importorskip(
    "transformers", reason="the causal model is read through Transformers"
)

TINY: Final = PinnedCheckpoint(
    "hf-internal-testing/tiny-random-gpt2", "71034c5d8bde858ff824298bdedc65515b97d2b9"
)
ROWS: Final = tuple(csv.DictReader(io.StringIO(ACCEPTABILITY), delimiter="\t"))
SENTENCES: Final = tuple(dict.fromkeys(row["sentence"] for row in ROWS))
SEED: Final = 3


class _Snapshots:
    """Copy a local release directory, or download a hub revision."""

    def snapshot(self, repository: str, revision: str, target: Path, /) -> None:
        if Path(repository).is_dir():
            copytree(repository, target, dirs_exist_ok=True)
            return

        HubSnapshotTransport().snapshot(repository, revision, target)


@pytest.fixture(scope="module")
def fetcher(tmp_path_factory: pytest.TempPathFactory) -> ArtifactFetcher:
    """Return one fetcher every release of this module is fetched through."""
    return ArtifactFetcher(
        tmp_path_factory.mktemp("artifacts"),
        archives=HttpArchiveTransport(),
        snapshots=_Snapshots(),
        offline=False,
    )


@pytest.fixture(scope="module")
def reference() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Return the pinned model and tokenizer, fetched into the cache at most once."""
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            TINY.repository, revision=TINY.revision
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            TINY.repository, revision=TINY.revision
        )

    except OSError as error:
        pytest.skip(f"the pinned model {TINY.repository} is unavailable: {error}")

    return model.eval(), tokenizer


@pytest.fixture(scope="module")
def counts(
    reference: tuple[PreTrainedModel, PreTrainedTokenizerBase],
) -> Counter[str]:
    """Return the token counts of the fixture's own sentences."""
    _, tokenizer = reference

    return Counter(
        token for sentence in SENTENCES for token in tokenizer.tokenize(sentence)
    )


@pytest.fixture(scope="module")
def table(counts: Counter[str]) -> UnigramTable:
    return UnigramTable(counts)


@pytest.fixture
def installed(
    counts: Counter[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fetcher: ArtifactFetcher,
) -> DeclaredDesign:
    """Return the scored fixture design over an installed data root."""
    root = install(tmp_path / "data")
    (root / UNIGRAMS_PATH).write_text(
        "token\tcount\n"
        + "".join(f"{token}\t{count}\n" for token, count in counts.items()),
        encoding="utf-8",
        newline="\n",
    )
    document = root / DESIGNS_DIRECTORY / f"{MEGAACCEPTABILITY_MORCELA_TAG}.json"
    declaration = json.loads(document.read_text(encoding="utf-8"))
    declaration["readouts"]["checkpoint"] = {
        "kind": "hub",
        "repository": TINY.repository,
        "revision": TINY.revision,
    }
    document.write_text(json.dumps(declaration), encoding="utf-8", newline="\n")
    monkeypatch.setenv(DATA_ROOT_VARIABLE, str(root))

    return DeclaredDesign(
        MEGAACCEPTABILITY_MORCELA_TAG, lambda _: HuggingFaceCausalLoader(fetcher)
    )


@pytest.fixture(scope="module")
def scorer(table: UnigramTable, fetcher: ArtifactFetcher) -> CausalLmScore:
    return CausalLmScore(HuggingFaceCausalLoader(fetcher), table, checkpoint=TINY)


def recomputed(
    reference: tuple[PreTrainedModel, PreTrainedTokenizerBase], sentence: str
) -> float:
    """Return a sentence's log probability from the reference model's logits."""
    model, tokenizer = reference
    ids = torch.tensor([[tokenizer.bos_token_id, *tokenizer.encode(sentence)]])

    with torch.no_grad():
        logits = model(ids).logits[0, :-1].double()

    return -float(
        torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum")
    )


class TestTinyCausalLmScore(LmScoreSuite):
    @pytest.fixture
    def lm_score_subject(self, scorer: CausalLmScore) -> LmScoreSubject:
        return LmScoreSubject(scorer, SENTENCES, SEED)


def test_the_scorer_records_the_pinned_checkpoint(scorer: CausalLmScore) -> None:
    assert scorer.checkpoint == TINY


def test_the_fixture_design_carries_readouts_recomputed_from_the_logits(
    reference: tuple[PreTrainedModel, PreTrainedTokenizerBase],
    table: UnigramTable,
    installed: DeclaredDesign,
) -> None:
    _, tokenizer = reference
    sentence_of = {row["verb"]: row["sentence"] for row in ROWS}

    design = installed.read().design
    sentences = [
        sentence_of[design.factor_levels["verb"][int(index)]]
        for index in design.factors["verb"]
    ]

    assert design.readouts[LM_LOG_PROBABILITY].tolist() == pytest.approx(
        [recomputed(reference, sentence) for sentence in sentences], abs=1e-6
    )
    assert design.readouts[UNIGRAM_LOG_PROBABILITY].tolist() == pytest.approx(
        [
            float(table.log_frequencies(tokenizer.tokenize(sentence)).sum())
            for sentence in sentences
        ]
    )
    assert design.readouts[SENTENCE_LENGTH].tolist() == [
        float(len(tokenizer.encode(sentence))) for sentence in sentences
    ]


def test_the_fixture_design_is_read_identically_twice(
    installed: DeclaredDesign,
) -> None:
    first = installed.read().design
    second = installed.read().design

    for name, values in first.readouts.items():
        assert torch.equal(values, second.readouts[name])


def test_the_fixture_design_records_the_pinned_checkpoint(
    installed: DeclaredDesign,
) -> None:
    assert installed.read().provenance.checkpoint == TINY.identity


@pytest.fixture
def half_precision(
    reference: tuple[PreTrainedModel, PreTrainedTokenizerBase], tmp_path: Path
) -> PinnedCheckpoint:
    """Return a local checkpoint whose configuration declares half precision."""
    model, tokenizer = reference
    configuration = model.config.to_dict()
    configuration["dtype"] = "float16"
    half = transformers.AutoModelForCausalLM.from_config(
        type(model.config).from_dict(configuration), dtype=torch.float16
    )
    half.load_state_dict(model.state_dict())
    directory = tmp_path / "half"
    half.save_pretrained(directory)
    tokenizer.save_pretrained(directory)

    return PinnedCheckpoint(str(directory), TINY.revision)


def test_a_checkpoint_declaring_half_precision_is_loaded_in_single_precision(
    half_precision: PinnedCheckpoint, fetcher: ArtifactFetcher
) -> None:
    declared = json.loads(
        (Path(half_precision.repository) / "config.json").read_text(encoding="utf-8")
    )
    loaded = cast(
        "torch.nn.Module",
        HuggingFaceCausalLoader(fetcher).load_causal_model(half_precision),
    )

    assert declared["dtype"] == "float16"
    assert {parameter.dtype for parameter in loaded.parameters()} == {torch.float32}


def test_a_causal_score_over_half_precision_weights_runs_in_single_precision(
    half_precision: PinnedCheckpoint, table: UnigramTable, fetcher: ArtifactFetcher
) -> None:

    scorer = CausalLmScore(
        HuggingFaceCausalLoader(fetcher),
        table,
        checkpoint=half_precision,
    )
    single = transformers.AutoModelForCausalLM.from_pretrained(
        half_precision.repository, dtype=torch.float32
    ).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(half_precision.repository)

    assert scorer.dtype is torch.float32
    assert [
        float(score.log_probabilities.sum()) for score in scorer.score(SENTENCES)
    ] == pytest.approx(
        [recomputed((single, tokenizer), sentence) for sentence in SENTENCES],
        abs=1e-9,
    )
