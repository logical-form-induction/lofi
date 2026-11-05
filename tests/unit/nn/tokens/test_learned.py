from typing import Final

import pytest
import torch

from lofi.nn.sites import ComponentId
from lofi.nn.tokens.embedder import ReservedTokens, StaticTokenEmbedder, TokenVocabulary
from lofi.nn.tokens.learned import initialized_embedder

EMBEDDER: Final = ComponentId("embedder")
VOCABULARY: Final = TokenVocabulary(
    ("word-0", "word-1", "word-2"),
    reserved=ReservedTokens({"<pad>": 0, "<unk>": 1}),
    unknown="<unk>",
)
BATCH: Final = (("word-0", "word-1", "word-2"), ("word-1",))


def drawn(width: int, seed: int) -> StaticTokenEmbedder:
    with torch.random.fork_rng():
        torch.manual_seed(seed)

        return initialized_embedder(
            width, init_std=width**-0.5, vocabulary=VOCABULARY, component=EMBEDDER
        )


@pytest.mark.parametrize("width", [1, 16, 64, 256])
def test_every_row_has_the_declared_width_and_takes_a_gradient(width: int) -> None:
    embedder = drawn(width, 0)
    rows = embedder.encode_words(BATCH).vectors

    assert embedder.output_width == width
    assert rows.tensor.shape == (2, 3, width)
    assert all(parameter.requires_grad for parameter in embedder.parameters())

    rows.tensor.sum().backward()

    assert all(parameter.grad is not None for parameter in embedder.parameters())


def test_one_seed_draws_one_table_and_another_seed_draws_another() -> None:
    first = drawn(8, 3).encode_words(BATCH).vectors.tensor
    again = drawn(8, 3).encode_words(BATCH).vectors.tensor
    other = drawn(8, 4).encode_words(BATCH).vectors.tensor

    assert torch.equal(first, again)
    assert not torch.equal(first, other)


def placements() -> tuple[str, ...]:
    """Return the meta device and every accelerator this machine has."""
    accelerator = torch.accelerator.current_accelerator()

    if accelerator is None or not torch.accelerator.is_available():
        return ("meta",)

    return ("meta", accelerator.type)


def table_of(embedder: StaticTokenEmbedder) -> torch.Tensor:
    """Return an embedder's table, copied to the processor."""
    return next(embedder.parameters()).detach().to("cpu")


@pytest.mark.parametrize("device", placements())
def test_one_seed_draws_one_table_whatever_device_is_the_default(device: str) -> None:
    on_cpu = table_of(drawn(8, 3))

    with torch.device(device):
        under = drawn(8, 3)

    placed = table_of(drawn(8, 3).to(device)) if device != "meta" else on_cpu

    assert next(under.parameters()).device == torch.device("cpu")
    assert torch.equal(table_of(under), on_cpu)
    assert torch.equal(placed, on_cpu)


def test_a_token_reads_its_own_row_and_an_unknown_one_the_unknown_row() -> None:
    embedder = drawn(8, 0)
    rows = embedder.encode_words((("word-0", "word-1", "zzz", "<unk>"),)).vectors.tensor

    assert not torch.equal(rows[0, 0], rows[0, 1])
    assert torch.equal(rows[0, 2], rows[0, 3])


def test_a_row_has_unit_expected_squared_norm_at_any_width() -> None:
    wide = TokenVocabulary(
        tuple(f"word-{index}" for index in range(4096)),
        reserved=ReservedTokens({"<unk>": 0}),
        unknown="<unk>",
    )

    for width in (16, 256):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            embedder = initialized_embedder(
                width, init_std=width**-0.5, vocabulary=wide, component=EMBEDDER
            )

        table = next(embedder.parameters()).detach()
        mean_square_norm = float(table.pow(2).sum(dim=1).mean())

        assert mean_square_norm == pytest.approx(1.0, rel=0.05)


def test_without_a_window_an_item_of_any_length_is_embedded_whole() -> None:
    embedder = drawn(2, 0)
    long_item = tuple("word-0" for _ in range(64))

    assert embedder.window is None
    assert embedder.encode_words((long_item,)).vectors.tensor.shape == (1, 64, 2)
