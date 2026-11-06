"""Sentence scores read directly from a pinned causal language model.

A sentence is tokenized without special tokens and read after the model's
beginning-of-sequence token, so its first token is conditioned on that token
alone and every later token on the tokens before it. Each sentence is read in
its own forward pass, so its score depends on nothing else in a batch. A
token's unigram log probability is its log relative frequency in a declared
count table over the model's tokens.
"""

import math
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, final, override

import torch

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.judgment._design import DesignCodec, DesignError, JudgmentRow, RowColumns
from lofi.judgment._values import SentenceScore

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi._artifacts import PinnedCheckpoint
    from lofi.judgment._design import JudgmentDesign
    from lofi.judgment._ports import JudgmentSource, LmScore
    from lofi.nn.tokens.pretrained import CausalLoader

__all__ = [
    "CausalLmScore",
    "LmReadoutColumns",
    "UnattestedTokenError",
    "UnigramTable",
    "lm_readout_design",
]


@final
class LmReadoutColumns(FrozenValue):
    """The readout column each language model measure of a sentence fills.

    Parameters
    ----------
    log_probability : str
        Column holding the sentence's log probability under the model.
    unigram_log_probability : str
        Column holding the sentence's unigram log probability.
    length : str
        Column holding the sentence's length in model tokens.
    """

    _noun = "language-model readout columns"
    __slots__ = ("_columns",)

    def __init__(
        self, log_probability: str, unigram_log_probability: str, length: str
    ) -> None:
        columns = tuple(
            ensure_name("a language-model readout column", column)
            for column in (log_probability, unigram_log_probability, length)
        )
        if len(set(columns)) != len(columns):
            msg = f"the language-model readouts repeat a column: {columns!r}"
            raise ValueError(msg)
        object.__setattr__(self, "_columns", columns)

    @property
    def columns(self) -> tuple[str, str, str]:
        """Return the log probability, unigram and length columns, in that order."""
        log_probability, unigram, length = self._columns
        return log_probability, unigram, length

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, LmReadoutColumns) and self._columns == other._columns

    @override
    def __hash__(self) -> int:
        return hash((LmReadoutColumns, self._columns))


class UnattestedTokenError(ValueError):
    """Reports a scored token that the unigram table does not count.

    Parameters
    ----------
    unattested : str
        The token without a count.
    """

    __slots__ = ("unattested",)

    def __init__(self, unattested: str) -> None:
        self.unattested = unattested

        super().__init__(
            f"the unigram table holds no count of the token {unattested!r}, so it has "
            f"no unigram log probability"
        )


@final
class UnigramTable(FrozenValue):
    """Token counts read as log relative frequencies.

    A token's unigram log probability is the log of its count over the sum of
    every count in the table.

    Parameters
    ----------
    counts : Mapping[str, int]
        Count of each token, every count at least one.

    Raises
    ------
    TypeError
        If a token is not a string or a count is not an integer.
    ValueError
        If the table counts no tokens, a token is empty, or a count is below
        one.
    """

    __slots__ = ("_log_frequencies",)

    _noun: ClassVar[str] = "a unigram table"

    def __init__(self, counts: Mapping[str, int]) -> None:
        if not counts:
            msg = "a unigram table counts no tokens"

            raise ValueError(msg)

        checked: dict[str, int] = {}

        for token, count in counts.items():
            name = ensure_name("a counted token", token)

            if ensure_non_negative(f"the count of {name!r}", count) < 1:
                msg = (
                    f"the count of {name!r} is {count}, and every count is at least one"
                )

                raise ValueError(msg)

            checked[name] = count

        total = math.log(sum(checked.values()))
        object.__setattr__(
            self,
            "_log_frequencies",
            MappingProxyType(
                {token: math.log(count) - total for token, count in checked.items()}
            ),
        )

    def log_frequencies(self, tokens: Sequence[str], /) -> Tensor:
        """Return each token's log relative frequency, in order.

        Parameters
        ----------
        tokens : Sequence[str]
            Tokens to look up.

        Raises
        ------
        UnattestedTokenError
            If the table holds no count of a token.
        """
        values: list[float] = []

        for token in tokens:
            if token not in self._log_frequencies:
                raise UnattestedTokenError(token)

            values.append(self._log_frequencies[token])

        return torch.tensor(values, dtype=torch.float64)

    def __len__(self) -> int:
        """Return the number of counted tokens."""
        return len(self._log_frequencies)

    @override
    def __repr__(self) -> str:
        """Return the number of counted tokens."""
        return f"UnigramTable(tokens={len(self._log_frequencies)})"


@final
class CausalLmScore:
    """Per-token log probabilities read from a causal language model.

    Parameters
    ----------
    loader : CausalLoader
        Loads the model, in the precision the loader declares, and its
        tokenizer; the model is put in evaluation mode.
    table : UnigramTable
        Counts over the tokenizer's tokens.
    checkpoint : PinnedCheckpoint
        The release the model and tokenizer are loaded from.

    Raises
    ------
    AbsentModelError
        If the loader cannot load the release.
    ValueError
        If the tokenizer declares no beginning-of-sequence token.
    """

    __slots__ = ("_bos", "_checkpoint", "_dtype", "_model", "_table", "_tokenizer")

    def __init__(
        self,
        loader: CausalLoader,
        table: UnigramTable,
        *,
        checkpoint: PinnedCheckpoint,
    ) -> None:
        tokenizer = loader.load_sentence_tokenizer(checkpoint)
        bos = tokenizer.bos_token_id

        if bos is None:
            msg = (
                f"the tokenizer of {checkpoint.identity} declares no "
                f"beginning-of-sequence token, so a sentence's first token has no "
                f"context to be scored in"
            )

            raise ValueError(msg)

        model = loader.load_causal_model(checkpoint)
        model.eval()
        self._bos = bos
        self._model = model
        self._tokenizer = tokenizer
        self._table = table
        self._checkpoint = checkpoint
        self._dtype = loader.dtype

    @property
    def checkpoint(self) -> PinnedCheckpoint:
        """Return the release the scores are read from."""
        return self._checkpoint

    @property
    def dtype(self) -> torch.dtype:
        """Return the precision the model's parameters were loaded in."""
        return self._dtype

    def score(self, sentences: Sequence[str], /) -> tuple[SentenceScore, ...]:
        """Return one score per sentence, in the sentences' order.

        Parameters
        ----------
        sentences : Sequence[str]
            Sentences to score.

        Raises
        ------
        ValueError
            If a sentence has no tokens.
        UnattestedTokenError
            If the unigram table holds no count of a scored token.
        """
        return tuple(self._scored(sentence) for sentence in sentences)

    def _scored(self, sentence: str) -> SentenceScore:
        """Return the score of one sentence from its own forward pass."""
        ids = self._tokenizer.encode(sentence, add_special_tokens=False)

        if not ids:
            msg = f"the sentence {sentence!r} has no tokens"

            raise ValueError(msg)

        context = torch.tensor([[self._bos, *ids]])

        with torch.no_grad():
            logits = self._model(input_ids=context).logits[0, :-1]

        log_probabilities = torch.log_softmax(logits.double(), dim=-1)
        targets = context[0, 1:].unsqueeze(1)
        tokens = tuple(self._tokenizer.convert_ids_to_tokens(ids))

        return SentenceScore(
            tokens,
            log_probabilities.gather(1, targets).squeeze(1),
            self._table.log_frequencies(tokens),
        )

    @override
    def __repr__(self) -> str:
        """Return the checkpoint and the size of the unigram table."""
        return f"CausalLmScore(checkpoint={self._checkpoint!r}, table={self._table!r})"


def lm_readout_design(
    source: JudgmentSource,
    sentences: Mapping[str, str],
    scorer: LmScore,
    readouts: LmReadoutColumns,
) -> JudgmentDesign:
    """Return a source's design with each row's language model readouts.

    Each rated item's sentence is scored once, and every row of the item
    carries the sentence's log probability, unigram log probability and
    length in tokens under the declared readout columns.

    Parameters
    ----------
    source : JudgmentSource
        Judgments whose rows carry no readouts.
    sentences : Mapping[str, str]
        The sentence of each rated item, by item identifier.
    scorer : LmScore
        Scores each sentence.
    readouts : LmReadoutColumns
        The readout column each measure fills.

    Raises
    ------
    DesignError
        If a rated item has no sentence, or the rows do not form a valid
        design.
    """
    rows = tuple(source.rows())
    items = tuple(dict.fromkeys(row.item for row in rows))
    missing = [item for item in items if item not in sentences]

    if missing:
        msg = f"the rated items {missing!r} have no sentence to score"

        raise DesignError(msg)

    scores = scorer.score([sentences[item] for item in items])
    log_probability, unigram, length = readouts.columns
    values = {
        item: {
            log_probability: float(score.log_probability),
            unigram: float(score.unigram_log_probability),
            length: float(score.length),
        }
        for item, score in zip(items, scores, strict=True)
    }
    codec = DesignCodec(source.response_levels, source.factors, readouts.columns)

    return codec.encode(
        JudgmentRow(
            row.participant,
            row.item,
            row.response,
            RowColumns(dict(row.factors), values[row.item]),
        )
        for row in rows
    )
