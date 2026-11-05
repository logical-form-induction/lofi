"""Decoder language models converted into bidirectional token encoders.

The conversion follows LLM2Vec (BehnamGhader et al. 2024): the decoder's causal
mask is lifted so that every position attends to every unpadded position, the
model is optionally adapted by masked next-token prediction, and its rows are
then read as a pretrained encoder's are, pooled into words by the run's subword
pooler.
"""

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    NamedTuple,
    Protocol,
    cast,
    override,
    runtime_checkable,
)

import torch
from torch import nn
from torch.nn import functional
from torch.nn.utils import parametrize

from lofi._frozen import FrozenValue
from lofi.nn.policy import place
from lofi.nn.tokens.pretrained import (
    ENCODER_ONLY,
    DecoderLoader,
    PretrainedTokenEncoder,
    declared_window,
)
from lofi.registry import ComponentPlugin, Registry, RegistryError, plugin_for
from lofi.spec.tokens.llm2vec import (
    LLM2VecTokenEncoderSettings,
    LLM2VecTokenEncoderSpec,
    MaskedNextTokenSettings,
)
from lofi.spec.tokens.pretrained import LastLayerReadoutSettings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor
    from torch.optim import Optimizer

    from lofi.nn.build import BuildContext
    from lofi.nn.tokens.pretrained import DecoderModel, DecoderTokenizer
    from lofi.spec import ConfiguredSpec

__all__ = [
    "AdaptationPlugin",
    "BidirectionalDecoder",
    "MaskReplacementPlugin",
    "adapt_masked_next_token",
    "read_corpus",
]


class _Replacement(Protocol):
    """Replace positions chosen for masked next-token adaptation."""

    def __call__(
        self,
        ids: Tensor,
        chosen: Tensor,
        *,
        mask: int,
        vocabulary: int,
        mask_fraction: float,
        random_fraction: float,
        generator: torch.Generator,
    ) -> Tensor:
        """Return a corrupted copy of ``ids``."""
        ...


class MaskReplacementPlugin(FrozenValue):
    """Apply one installed masked-token replacement mechanism."""

    __slots__ = ("_replace",)
    _noun = "mask-replacement plugin"

    def __init__(self, replace: _Replacement, /) -> None:
        object.__setattr__(self, "_replace", replace)

    def replace(
        self,
        ids: Tensor,
        chosen: Tensor,
        *,
        mask: int,
        vocabulary: int,
        mask_fraction: float,
        random_fraction: float,
        generator: torch.Generator,
    ) -> Tensor:
        """Return a corrupted copy of ``ids``."""
        return self._replace(
            ids,
            chosen,
            mask=mask,
            vocabulary=vocabulary,
            mask_fraction=mask_fraction,
            random_fraction=random_fraction,
            generator=generator,
        )


def mask_replacement(name: str, /) -> MaskReplacementPlugin:
    """Return the replacement mechanism installed under ``name``."""
    member = Registry[object]("mask_replacement").get(name)

    if not isinstance(member, MaskReplacementPlugin):
        msg = (
            f"the mask_replacement entry {name!r} names {member!r}, "
            "not a mask-replacement plugin"
        )
        raise RegistryError(msg, family="mask_replacement")

    return member


def _mixed_replacement(
    ids: Tensor,
    chosen: Tensor,
    *,
    mask: int,
    vocabulary: int,
    mask_fraction: float,
    random_fraction: float,
    generator: torch.Generator,
) -> Tensor:
    """Replace chosen rows by the declared mask, random and unchanged mix."""
    corrupted = ids.clone()
    draw = torch.rand(ids.shape, generator=generator)
    corrupted[chosen & (draw < mask_fraction)] = mask
    random = chosen & (draw >= mask_fraction) & (draw < mask_fraction + random_fraction)
    corrupted[random] = torch.randint(
        vocabulary, (int(random.sum()),), generator=generator
    )
    return corrupted


def _mask_replacement(
    ids: Tensor,
    chosen: Tensor,
    *,
    mask: int,
    vocabulary: int,
    mask_fraction: float,
    random_fraction: float,
    generator: torch.Generator,
) -> Tensor:
    """Replace every chosen row by the mask token."""
    del vocabulary, mask_fraction, random_fraction, generator
    corrupted = ids.clone()
    corrupted[chosen] = mask
    return corrupted


MIXED: Final = MaskReplacementPlugin(_mixed_replacement)
MASK: Final = MaskReplacementPlugin(_mask_replacement)


class _Parameterize(Protocol):
    """Select and initialize the parameters an adaptation optimizer updates."""

    def __call__(
        self,
        model: nn.Module,
        *,
        rank: int,
        alpha: float,
        generator: torch.Generator,
    ) -> tuple[nn.Parameter, ...]:
        """Return exactly the parameters adaptation optimizes."""
        ...


class AdaptationPlugin(FrozenValue):
    """Apply one installed adaptation parameterization to a decoder."""

    __slots__ = ("_parameterize",)
    _noun = "adaptation plugin"

    def __init__(self, parameterize: _Parameterize, /) -> None:
        object.__setattr__(self, "_parameterize", parameterize)

    def parameters(
        self,
        model: nn.Module,
        *,
        rank: int,
        alpha: float,
        generator: torch.Generator,
    ) -> tuple[nn.Parameter, ...]:
        """Return exactly the parameters adaptation optimizes."""
        return self._parameterize(model, rank=rank, alpha=alpha, generator=generator)


def adaptation(name: str, /) -> AdaptationPlugin:
    """Return the parameterization installed under ``name``."""
    member = Registry[object]("adaptation").get(name)

    if not isinstance(member, AdaptationPlugin):
        msg = (
            f"the adaptation entry {name!r} names {member!r}, not an adaptation plugin"
        )
        raise RegistryError(msg, family="adaptation")

    return member


def _full_parameters(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    generator: torch.Generator,
) -> tuple[nn.Parameter, ...]:
    """Train every parameter of a decoder."""
    del rank, alpha, generator
    parameters = tuple(model.parameters())

    for parameter in parameters:
        parameter.requires_grad_()

    return parameters


class _LowRankWeight(nn.Module):
    """Add a trainable low-rank update to one public linear weight."""

    def __init__(
        self,
        rows: int,
        columns: int,
        *,
        rank: int,
        alpha: float,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.down = nn.Parameter(
            torch.randn((rank, columns), generator=generator) * columns**-0.5
        )
        self.up = nn.Parameter(torch.zeros((rows, rank)))
        self._scale = alpha / rank

    @override
    def forward(self, weight: Tensor) -> Tensor:
        """Return the frozen weight plus its low-rank update."""
        update = (self.up @ self.down).to(dtype=weight.dtype)
        return weight + update * self._scale


def _lora_parameters(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    generator: torch.Generator,
) -> tuple[nn.Parameter, ...]:
    """Parameterize every public linear projection with a low-rank update."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    adapters: list[nn.Parameter] = []

    for module in tuple(model.modules()):
        if not isinstance(module, nn.Linear):
            continue

        rows, columns = module.weight.shape
        update = _LowRankWeight(
            rows, columns, rank=rank, alpha=alpha, generator=generator
        )
        parametrize.register_parametrization(module, "weight", update)
        adapters.extend((update.down, update.up))

    if not adapters:
        msg = "LoRA adaptation requires at least one public linear projection"
        raise ValueError(msg)

    return tuple(adapters)


FULL: Final = AdaptationPlugin(_full_parameters)
LORA: Final = AdaptationPlugin(_lora_parameters)


@runtime_checkable
class _CausalFlag(Protocol):
    """An attention module that records whether it masks later positions."""

    is_causal: bool


class _Width(Protocol):
    """A configuration naming its model width ``hidden_size``."""

    hidden_size: int
    num_hidden_layers: int


class _Rows(NamedTuple):
    """The rows a bidirectional decoder returns, under an encoder's field name.

    Attributes
    ----------
    last_hidden_state : Tensor
        Rank 3 over the axes ``("item", "row", "width")``.
    """

    last_hidden_state: Tensor
    hidden_states: tuple[Tensor, ...] | None = None


def _bidirectional_mask(attention_mask: Tensor, dtype: torch.dtype) -> Tensor:
    """Return an additive mask letting every position attend to every unpadded one.

    Parameters
    ----------
    attention_mask : Tensor
        Rank 2 over the axes ``("item", "row")``, nonzero at unpadded rows.
    dtype : torch.dtype
        The precision of the attention scores the mask is added to.

    Returns
    -------
    Tensor
        Rank 4 over the axes ``("item", "head", "query", "key")``, with one head
        broadcast over all, zero where the key is unpadded and the least finite
        value of the precision where it is padding.
    """
    items, rows = attention_mask.shape
    keep = attention_mask.to(dtype=torch.bool)[:, None, None, :].expand(
        items, 1, rows, rows
    )
    mask = torch.zeros(keep.shape, dtype=dtype, device=attention_mask.device)

    return mask.masked_fill(~keep, torch.finfo(dtype).min)


class BidirectionalDecoder(nn.Module):
    """A decoder language model whose every position attends to every other.

    The model's decoder stack is run under an attention mask that hides only
    padding, and every attention module recording that it masks later positions
    is told it does not. The rows it returns are the decoder stack's final rows,
    and its output head scores them as next-token logits.

    Parameters
    ----------
    model : DecoderModel
        The causal language model, with its output head.

    Raises
    ------
    TypeError
        If the model is not a torch module.
    """

    def __init__(self, model: DecoderModel) -> None:
        super().__init__()

        if not isinstance(model, nn.Module):
            msg = f"a decoder language model is a torch module, not {model!r}"
            raise TypeError(msg)

        for module in model.modules():
            if isinstance(module, _CausalFlag):
                module.is_causal = False

        self.model = model

    @property
    def config(self) -> _Width:
        """Return the model configuration, which names its width ``hidden_size``."""
        return self.model.config

    def _rows(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return the decoder stack's final rows under the bidirectional mask."""
        dtype = next(self.model.parameters()).dtype
        stack = self.model.get_decoder()

        return stack(
            input_ids=input_ids,
            attention_mask=_bidirectional_mask(attention_mask, dtype),
            return_dict=True,
            output_hidden_states=False,
        ).last_hidden_state

    @override
    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        return_dict: bool,
        output_hidden_states: bool = False,
    ) -> _Rows:
        """Return the bidirectional rows of a batch of token indices.

        Parameters
        ----------
        input_ids : Tensor
            Rank 2 over the axes ``("item", "row")``, integral.
        attention_mask : Tensor
            Rank 2 over the axes ``("item", "row")``, nonzero at unpadded rows.
        return_dict : bool
            Accepted for the encoder-stack interface; the rows always come back
            under ``last_hidden_state``.

        Returns
        -------
        _Rows
            Rows over the axes ``("item", "row", "width")``.
        """
        del return_dict, output_hidden_states

        return _Rows(self._rows(input_ids, attention_mask), None)

    def next_token_logits(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return the output head's logits at every position of a batch.

        Parameters
        ----------
        input_ids : Tensor
            Rank 2 over the axes ``("item", "row")``, integral.
        attention_mask : Tensor
            Rank 2 over the axes ``("item", "row")``, nonzero at unpadded rows.

        Returns
        -------
        Tensor
            Rank 3 over the axes ``("item", "row", "vocabulary")``, real valued:
            the logits each row assigns the token after it.
        """
        head = self.model.get_output_embeddings()

        return head(self._rows(input_ids, attention_mask))


def read_corpus(path: Path, /) -> tuple[str, ...]:
    """Return the items of a text corpus, one per nonblank line.

    Parameters
    ----------
    path : Path
        The file, read as UTF-8.

    Returns
    -------
    tuple[str, ...]
        Each nonblank line's words joined by single spaces, in file order.

    Raises
    ------
    ValueError
        If the file holds no nonblank line.
    """
    lines = tuple(
        " ".join(line.split())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

    if not lines:
        msg = f"the adaptation corpus {str(path)!r} holds no line"
        raise ValueError(msg)

    return lines


def _mask_index(tokenizer: DecoderTokenizer, token: str) -> int:
    """Return the vocabulary index of the declared mask token.

    Raises
    ------
    ValueError
        If the tokenizer's vocabulary does not hold the token.
    """
    vocabulary = tokenizer.get_vocab()

    if token not in vocabulary:
        msg = f"the tokenizer's vocabulary holds no mask token {token!r}"
        raise ValueError(msg)

    return vocabulary[token]


def _batches(
    count: int, *, size: int, steps: int, generator: torch.Generator
) -> list[list[int]]:
    """Return the line indices each step reads, reshuffled on every pass.

    Parameters
    ----------
    count : int
        How many lines the corpus holds.
    size : int
        How many lines a step reads.
    steps : int
        How many steps there are.
    """
    order: list[int] = []

    while len(order) < size * steps:
        order.extend(torch.randperm(count, generator=generator).tolist())

    return [order[step * size : (step + 1) * size] for step in range(steps)]


def _packed(
    ids: Tensor, attention: Tensor, /, *, pad: int, rows: int | None
) -> tuple[Tensor, Tensor]:
    """Return each item's unpadded rows moved to its front and cut at a bound.

    A tokenizer may pad on either side; reading the rows from the front keeps
    each item's positions starting at zero and makes the bound cut its tail.

    Parameters
    ----------
    ids : Tensor
        Rank 2 over the axes ``("item", "row")``, integral.
    attention : Tensor
        Rank 2 over the axes ``("item", "row")``, true at unpadded rows.
    pad : int
        The index the vacated rows are filled with.
    rows : int | None
        The most rows of an item kept, or None to keep every one.

    Returns
    -------
    tuple[Tensor, Tensor]
        The packed indices and the mask true at their unpadded rows.
    """
    kept = [ids[item][attention[item]][:rows] for item in range(ids.shape[0])]
    longest = max(len(row) for row in kept)
    packed = ids.new_full((len(kept), longest), pad)
    live = attention.new_zeros((len(kept), longest))

    for item, row in enumerate(kept):
        packed[item, : len(row)] = row
        live[item, : len(row)] = True

    return packed, live


def adapt_masked_next_token(
    decoder: BidirectionalDecoder,
    tokenizer: DecoderTokenizer,
    lines: Sequence[str],
    /,
    *,
    settings: MaskedNextTokenSettings,
    rows: int | None,
) -> tuple[float, ...]:
    """Adapt a bidirectional decoder by masked next-token prediction.

    Each step tokenizes a batch of lines, moves each line's unpadded rows to its
    front, keeps at most the bounded number of them, and chooses each
    non-special token after the first row with the declared probability. The chosen
    tokens are replaced as declared, and the token originally at each chosen
    position is predicted from the row one position before it through the
    model's output head, under cross-entropy. Every parameter of the decoder is
    updated by AdamW at the declared learning rate. A step choosing no token
    takes no update.

    Parameters
    ----------
    decoder : BidirectionalDecoder
        The decoder adapted in place, on whatever device it is placed.
    tokenizer : DecoderTokenizer
        The decoder's tokenizer.
    lines : Sequence[str]
        The corpus items, each one line of text.
    settings : MaskedNextTokenSettings
        The mask token, the mask probability, the replacement, and the number,
        size and step size of the optimizer steps.
    rows : int | None
        The most token rows of a line read, or None to read every line whole.

    Returns
    -------
    tuple[float, ...]
        The loss of each step that took an update, in step order.

    Raises
    ------
    ValueError
        If the vocabulary holds no mask token or the corpus holds no line.
    """
    if not lines:
        msg = "an adaptation corpus holds at least one line"
        raise ValueError(msg)

    mask_index = _mask_index(tokenizer, settings.mask)
    replacement = mask_replacement(settings.replacement)
    generator = torch.Generator(device="cpu").manual_seed(settings.seed)
    parameter = next(decoder.parameters())
    device = parameter.device
    specials = torch.tensor(tokenizer.all_special_ids, dtype=torch.long)
    trained = decoder.training

    parameters = adaptation(settings.parameterization).parameters(
        decoder,
        rank=settings.rank,
        alpha=settings.alpha,
        generator=generator,
    )
    configured = cast("ConfiguredSpec[object]", settings.optimizer)
    optimizer = cast(
        "Optimizer",
        plugin_for(configured).construct(configured, parameters),
    )
    losses: list[float] = []
    decoder.train()

    for batch in _batches(
        len(lines),
        size=settings.batch_size,
        steps=settings.steps,
        generator=generator,
    ):
        tokenized = tokenizer(
            [lines[index] for index in batch],
            padding=True,
            return_tensors="pt",
            truncation=False,
            return_offsets_mapping=True,
        )
        ids, attention = _packed(
            tokenized.input_ids.to(dtype=torch.long),
            tokenized.attention_mask.to(dtype=torch.bool),
            pad=tokenizer.pad_token_id,
            rows=rows,
        )
        eligible = attention & ~torch.isin(ids, specials)
        eligible[:, 0] = False
        chosen = eligible & (
            torch.rand(ids.shape, generator=generator) < settings.mask_probability
        )

        if not bool(chosen.any()):
            continue

        corrupted = replacement.replace(
            ids,
            chosen,
            mask=mask_index,
            vocabulary=len(tokenizer),
            mask_fraction=settings.mask_replacement_fraction,
            random_fraction=settings.random_replacement_fraction,
            generator=generator,
        )

        logits = decoder.next_token_logits(
            corrupted.to(device=device), attention.to(device=device, dtype=torch.long)
        )
        targets = chosen[:, 1:].to(device=device)
        loss = functional.cross_entropy(
            logits[:, :-1][targets].float(), ids[:, 1:].to(device=device)[targets]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    optimizer.zero_grad(set_to_none=True)
    decoder.train(trained)

    return tuple(losses)


def _build_llm2vec(
    settings: LLM2VecTokenEncoderSettings, context: BuildContext, /
) -> PretrainedTokenEncoder:
    """Build and place a decoder converted into a bidirectional token encoder.

    The decoder and its tokenizer are loaded through the run's loader in the
    policy's precision, the causal mask is lifted, the declared adaptation runs
    on the policy's device, and the rows are read as a pretrained encoder's,
    whose width the model's configuration names.

    Parameters
    ----------
    settings : LLM2VecTokenEncoderSettings
        The pinned checkpoint, whether the decoder trains, its context window
        and the adaptation run before use.
    context : BuildContext
        The loader the weights come through, the component handle and the
        device policy.

    Returns
    -------
    PretrainedTokenEncoder
        The encoder over the bidirectional decoder.

    Raises
    ------
    TypeError
        If the run's loader cannot load a decoder language model.
    """
    loader = context.loader

    if not isinstance(loader, DecoderLoader):
        msg = (
            f"an LLM2Vec encoder loads a decoder language model, and the loader "
            f"{loader!r} loads none"
        )
        raise TypeError(msg)

    checkpoint = settings.checkpoint
    tokenizer = loader.load_tokenizer(checkpoint)
    decoder = place(
        BidirectionalDecoder(
            loader.load_decoder(checkpoint, dtype=context.policy.dtype)
        ),
        context.policy,
    )
    window = declared_window(settings.window, tokenizer)

    if settings.adaptation is not None:
        adapt_masked_next_token(
            decoder,
            tokenizer,
            read_corpus(Path(settings.adaptation.corpus)),
            settings=settings.adaptation,
            rows=window.rows,
        )

    return place(
        PretrainedTokenEncoder(
            decoder,
            tokenizer,
            architecture=ENCODER_ONLY,
            checkpoint=checkpoint,
            component=context.component,
            trainable=settings.trainable,
            window=window,
            readout=LastLayerReadoutSettings(),
            join=settings.join,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=LLM2VecTokenEncoderSpec, build=_build_llm2vec)
"""The LLM2Vec token encoder's plugin."""
