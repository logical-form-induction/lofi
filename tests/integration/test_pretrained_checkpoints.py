"""Each packaged pretrained configuration loads its pinned checkpoint and encodes."""

import json
from typing import TYPE_CHECKING, Final, cast
from urllib.parse import quote

import pytest
import torch
from didactic.settings import load_document
from tests.conformance.component_plugin import plugin_context

from lofi.nn._snapshots import artifact_fetcher
from lofi.nn.build import build_component
from lofi.nn.sites import ComponentId
from lofi.nn.tokens.pool import MeanSubwordPooler
from lofi.nn.tokens.pretrained import AbsentModelError, HuggingFacePretrainedLoader
from lofi.run import section_root
from lofi.spec import TokenEncoderSpec
from lofi.spec._plugins import load_plugins
from lofi.spec.artifacts import ArtifactCacheSettings

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from lofi.nn.tokens.pretrained import PretrainedTokenEncoder
    from lofi.spec.tokens.pretrained import PretrainedTokenEncoderSpec

pytestmark = pytest.mark.integration

transformers = pytest.importorskip(
    "transformers", reason="the pinned encoders are read through Transformers"
)

TOKENS_SLOT: Final = "model.token_encoder"
WORDS: Final = ("The", "linguists", "reconsidered", "unbelievably", "odd", "data")


def _declared(option: str) -> PretrainedTokenEncoderSpec:
    """Return one packaged token-encoder option as its validated declaration."""
    load_plugins()
    document = load_document(section_root(TOKENS_SLOT) / f"{option}.yaml")

    return cast(
        "PretrainedTokenEncoderSpec",
        TokenEncoderSpec.model_validate_json(json.dumps(document)),
    )


def _loaded(
    option: str, root: Path
) -> tuple[PretrainedTokenEncoder, PretrainedTokenEncoderSpec]:
    """Return the encoder one packaged option builds, skipping if it cannot load."""
    declared = _declared(option)
    loader = HuggingFacePretrainedLoader(
        artifact_fetcher(ArtifactCacheSettings(root, offline=False))
    )

    try:
        encoder = cast(
            "PretrainedTokenEncoder",
            build_component(
                declared,
                plugin_context(component="token_encoder", loader=loader),
            ),
        )

    except AbsentModelError as error:
        pytest.skip(f"the pinned checkpoint of {option!r} is unavailable: {error}")

    return encoder, declared


def _expected_rows(
    natural: Sequence[int], pieces: Sequence[Sequence[int]]
) -> tuple[tuple[int, int], ...]:
    """Return each word's row span, read off where its own pieces sit in the text.

    Parameters
    ----------
    natural : Sequence[int]
        The token ids of the whole sentence, special tokens included.
    pieces : Sequence[Sequence[int]]
        The token ids of each word tokenized alone, after its leading space.
    """
    joined = [token for piece in pieces for token in piece]
    starts = [
        start
        for start in range(len(natural) - len(joined) + 1)
        if list(natural[start : start + len(joined)]) == joined
    ]

    assert len(starts) == 1, "the words' own pieces do not tile the sentence once"

    spans: list[tuple[int, int]] = []
    at = starts[0]

    for piece in pieces:
        spans.append((at, at + len(piece)))
        at += len(piece)

    return tuple(spans)


@pytest.mark.parametrize(
    "option", ["pretrained", "modernbert", "mmbert", "xlm-roberta", "mt5", "eurobert"]
)
def test_a_packaged_checkpoint_encodes_a_sentence_one_pooled_row_per_word(
    option: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    root = tmp_path_factory.getbasetemp() / "artifacts"
    encoder, declared = _loaded(option, root)
    checkpoint = declared.settings().checkpoint
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(
            root
            / "snapshots"
            / quote(checkpoint.repository, safe="")
            / checkpoint.revision
        )
    )
    natural = tokenizer(" ".join(WORDS)).input_ids
    pieces = [
        tokenizer(
            word if position == 0 else f" {word}", add_special_tokens=False
        ).input_ids
        for position, word in enumerate(WORDS)
    ]

    with torch.no_grad():
        encoded = encoder.encode_words((WORDS,))

    pooled = MeanSubwordPooler(
        encoder.output_width, component=ComponentId("pooler")
    ).pool(encoded)
    alignment = encoded.alignments[0]

    assert encoder.checkpoint.revision == declared.revision
    assert not encoder.trainable
    assert encoded.vectors.size("row") == len(natural)
    assert alignment.rows == _expected_rows(natural, pieces)
    assert pooled.tensor.shape == (1, len(WORDS), encoder.output_width)
    assert bool(torch.isfinite(pooled.tensor).all())
