"""Prices one training step at the sentence lengths a real CCGbank holds.

Derivations are read from a licensed CCGbank root, a typical length is taken for
each decile of the sample, and one full training step of the real stack is
priced at each of them. The sample's length histogram is reported beside the
timings, so what the step costs at the working point and what it costs out in
the corpus's tail are read off one measured curve.

The corpus's longest sentence is priced on its own rather than as the last point
of the series. A batch of the tail costs memory quadratic in its length, and a
tail that does not fit the device should cost its own point rather than the nine
below it.

The corpus supplies the lengths that are priced. The stack, its grammar, its
lexicon and the sentences its steps read are the pricing reference run
document's, composed at each priced length: what is priced is the corpus's
length distribution through the real stack, not a grammar induced from
CCGbank.

The run is gated twice. It is marked ``integration``, so it needs
``--run-integration``, and it reads its corpus through ``ccgbank_root``, which
skips unless ``LOFI_CCGBANK`` names a directory. Nothing here downloads a
corpus, and nothing it reads is written back into the tree.
"""

import statistics
import textwrap
from collections import Counter
from itertools import cycle, islice, pairwise
from typing import TYPE_CHECKING, Final, NamedTuple, cast

import pytest
import torch
from tests.benchmarks.nn.test_training_step import (
    LENGTHS,
    POINT,
    SIZES,
    Timing,
    run,
    time_once,
)

from lofi.data import BatchShape
from lofi.data.corpora.ccgbank import AutoLeaf, parse_auto
from lofi.run import LengthBatchPricing
from lofi.run._cost_model_builtins import reference_stacks
from lofi.run._timing import ReferenceStep, prepare_reference

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from pytest_benchmark.fixture import BenchmarkFixture

    from lofi.data.corpora.ccgbank import AutoNode

#: The sentence length the activation budget's working point names, in tokens.
WORKING_LENGTH: Final = LENGTHS[-1]

#: The batch size the working point names, in sentences.
BATCH: Final = SIZES[-1]

#: How many sentences the priced sample holds at least.
SAMPLE: Final = 1_000

#: How many quantiles of the sample's lengths are priced.
DECILES: Final = 10

#: How many timed runs each priced length takes.
TIMED: Final = 5

CONFTEST: Final = textwrap.dedent(
    """
    from tests.conftest import ccgbank_root

    __all__ = ["ccgbank_root"]
    """
)

GATED: Final = textwrap.dedent(
    """
    def test_gated(ccgbank_root):
        assert ccgbank_root.is_dir()
    """
)


def auto_file(sentences: Sequence[tuple[str, ...]], /) -> str:
    """Return the AUTO text of one derivation per sentence, left-branching.

    Every leaf is a common noun and every internal node a binary sentence node,
    which is a derivation the AUTO grammar admits and holds no licensed content.

    Parameters
    ----------
    sentences : Sequence[tuple[str, ...]]
        The words of each derivation, in sentence order.
    """
    written: list[str] = []

    for index, words in enumerate(sentences):
        tree = f"(<L N NN NN {words[0]} N>)"

        for word in words[1:]:
            tree = f"(<T S 0 2> {tree} (<L N NN NN {word} N>) )"

        written.append(f"ID=s{index}\n{tree}\n")

    return "".join(written)


def leaf_words(node: AutoNode, /) -> tuple[str, ...]:
    """Return the words at a derivation node's leaves, in sentence order.

    Parameters
    ----------
    node : AutoNode
        The derivation node to read.
    """
    if isinstance(node, AutoLeaf):
        return (node.word,)

    return tuple(word for child in node.children for word in leaf_words(child))


def corpus_sentences(root: Path, *, at_least: int) -> tuple[tuple[str, ...], ...]:
    """Read derivation files under a root until enough sentences have been read.

    Files are read in path order and whole, so the sample is a prefix of the
    corpus in file order rather than a subsample drawn from all of it.

    Parameters
    ----------
    root : Path
        The CCGbank root to walk, whose ``.auto`` files hold the derivations.
    at_least : int
        How many sentences to read before stopping. The file that reaches the
        count is read to its end.
    """
    read: list[tuple[str, ...]] = []

    for path in sorted(root.rglob("*.auto")):
        derivations = parse_auto(path.read_bytes())
        read.extend(leaf_words(derivation.tree) for derivation in derivations)

        if len(read) >= at_least:
            break

    return tuple(read)


def length_histogram(sentences: Sequence[tuple[str, ...]], /) -> dict[int, int]:
    """Return how many sentences the sample holds at each length, shortest first.

    Parameters
    ----------
    sentences : Sequence[tuple[str, ...]]
        The sample, one tuple of words per sentence.
    """
    counted = Counter(len(sentence) for sentence in sentences)

    return {length: counted[length] for length in sorted(counted)}


def decile_lengths(
    sentences: Sequence[tuple[str, ...]], /, *, deciles: int
) -> tuple[int, ...]:
    """Return a typical length for each quantile of the sample, shortest first.

    The k-th value is the length of the sentence in the middle of the k-th
    quantile, at rank ``ceil((2k - 1) * n / (2 * deciles))`` in length order, so
    each priced point is a length that quantile's sentences actually have rather
    than the boundary they end at.

    Parameters
    ----------
    sentences : Sequence[tuple[str, ...]]
        The sample, one tuple of words per sentence.
    deciles : int
        How many quantiles to cut the sample into.

    Raises
    ------
    ValueError
        If the sample is empty, or fewer than one quantile was asked for.
    """
    if deciles < 1:
        msg = f"a sample is cut into at least one quantile, not {deciles}"

        raise ValueError(msg)

    ordered = sorted(len(sentence) for sentence in sentences)

    if not ordered:
        msg = "an empty sample has no quantile lengths"

        raise ValueError(msg)

    total = len(ordered)

    return tuple(
        ordered[((2 * k - 1) * total - 1) // (2 * deciles)]
        for k in range(1, deciles + 1)
    )


def tail_length(sentences: Sequence[tuple[str, ...]], /) -> int:
    """Return the length of the longest sentence in the sample.

    Parameters
    ----------
    sentences : Sequence[tuple[str, ...]]
        The sample, one tuple of words per sentence.

    Raises
    ------
    ValueError
        If the sample is empty.
    """
    if not sentences:
        msg = "an empty sample has no longest sentence"

        raise ValueError(msg)

    return max(len(sentence) for sentence in sentences)


def cut_to(
    sentences: Sequence[tuple[str, ...]], length: int, count: int
) -> tuple[tuple[str, ...], ...]:
    """Return sentences of the sample cut to one length, enough to fill a batch.

    A sentence shorter than the length is left out rather than padded, and the
    ones that remain repeat in sample order when the sample holds fewer of them
    than the batch takes.

    Parameters
    ----------
    sentences : Sequence[tuple[str, ...]]
        The sample, one tuple of words per sentence.
    length : int
        Tokens each returned sentence has.
    count : int
        How many sentences to return.

    Raises
    ------
    ValueError
        If no sentence in the sample is that long.
    """
    long_enough = tuple(
        sentence[:length] for sentence in sentences if len(sentence) >= length
    )

    if not long_enough:
        msg = f"no sentence in the sample reaches {length} tokens"

        raise ValueError(msg)

    return tuple(islice(cycle(long_enough), count))


class LengthCost(NamedTuple):
    """What one training step cost at one of the corpus's lengths.

    Attributes
    ----------
    sentence_length
        Tokens in each sentence of the priced batch.
    available
        How many sentences of the sample reach that length.
    seconds
        The median of the timed runs.
    seconds_fastest
        The fastest timed run, which is the one least disturbed by other work.
    peak_bytes
        The allocator's peak over the timed runs.
    """

    sentence_length: int
    available: int
    seconds: float
    seconds_fastest: float
    peak_bytes: int


def priced_lengths(lengths: Sequence[int], /) -> tuple[int, ...]:
    """Return the lengths to price, each once, in the order they first appear.

    Two quantiles of a corpus with a mode fall at the same length, and pricing
    that length twice would time one batch under two names.

    Parameters
    ----------
    lengths : Sequence[int]
        The lengths asked for, in report order.
    """
    return tuple(dict.fromkeys(lengths))


def summarize(
    sentences: Sequence[tuple[str, ...]], timed: Mapping[int, Sequence[Timing]]
) -> tuple[LengthCost, ...]:
    """Return what each priced length cost, read off its timed runs.

    Parameters
    ----------
    sentences : Sequence[tuple[str, ...]]
        The sample the priced batches were drawn from.
    timed : Mapping[int, Sequence[Timing]]
        The runs timed at each priced length, in the order they are reported.
    """
    return tuple(
        LengthCost(
            sentence_length=length,
            available=sum(1 for sentence in sentences if len(sentence) >= length),
            seconds=statistics.median(timing.seconds for timing in runs),
            seconds_fastest=min(timing.seconds for timing in runs),
            peak_bytes=max(timing.peak_bytes for timing in runs),
        )
        for length, runs in timed.items()
    )


def step_at(device: torch.device, length: int, /) -> ReferenceStep:
    """Return the pricing reference step at one length and the working batch size.

    Parameters
    ----------
    device : torch.device
        Where the step executes.
    length : int
        Tokens in each sentence of the step's batch.
    """
    pricing = LengthBatchPricing(lengths=(length, length + 1), batches=(1, BATCH))

    return prepare_reference(
        reference_stacks(pricing)[length, BATCH], POINT, device=str(device)
    )


def reference_width() -> int:
    """Return the parser width the pricing reference document declares."""
    pricing = LengthBatchPricing(lengths=LENGTHS, batches=SIZES)
    parser = next(iter(reference_stacks(pricing).values())).model.parser

    return 0 if parser is None else cast("int", parser.model_dump()["width"])


def series(
    device: torch.device, sentences: Sequence[tuple[str, ...]], lengths: Sequence[int]
) -> tuple[LengthCost, ...]:
    """Warm every priced length, then time a step at each of them in turn.

    Each pass round the series times one run at every length, so a stretch of
    the machine busy with other work is spread over the priced points rather
    than charged to whichever one it coincides with.

    Parameters
    ----------
    device : torch.device
        Where every step executes.
    sentences : Sequence[tuple[str, ...]]
        The sample whose lengths are priced.
    lengths : Sequence[int]
        The lengths to price, in the order they are reported.
    """
    priced = priced_lengths(lengths)
    steps = {length: step_at(device, length) for length in priced}

    for step in steps.values():
        run(step)

    timed: dict[int, list[Timing]] = {length: [] for length in priced}

    for _ in range(TIMED):
        for length in priced:
            timed[length].append(time_once(steps[length]))

    return summarize(sentences, timed)


def seconds_per_token(costs: Sequence[LengthCost], /) -> float:
    """Return the least-squares slope of the fastest step time against length.

    Parameters
    ----------
    costs : Sequence[LengthCost]
        The priced points, at two or more distinct lengths.
    """
    fit = statistics.linear_regression(
        [float(cost.sentence_length) for cost in costs],
        [cost.seconds_fastest for cost in costs],
    )

    return fit.slope


@pytest.fixture(scope="module")
def corpus_sample(ccgbank_root: Path) -> tuple[tuple[str, ...], ...]:
    """Return the sentences the priced sample is drawn from, read once.

    Returns
    -------
    tuple[tuple[str, ...], ...]
        The words of each sentence read, in corpus file order.
    """
    return corpus_sentences(ccgbank_root, at_least=SAMPLE)


@pytest.fixture(scope="module")
def corpus_device() -> torch.device:
    """Return the accelerator, or skip when the machine has none.

    Returns
    -------
    torch.device
        The device every priced step executes on.
    """
    device = torch.accelerator.current_accelerator()

    if device is None or not torch.accelerator.is_available():
        pytest.skip(
            "the hot path is priced on one accelerator and this machine has "
            "none, so nothing here would measure what it asserts"
        )

    return device


@pytest.mark.corpus
@pytest.mark.integration
def test_the_sample_covers_a_thousand_sentences_and_the_tail_beyond_the_working_point(
    corpus_sample: tuple[tuple[str, ...], ...],
) -> None:
    histogram = length_histogram(corpus_sample)

    assert len(corpus_sample) >= SAMPLE
    assert sum(histogram.values()) == len(corpus_sample)
    assert max(histogram) > WORKING_LENGTH
    assert min(histogram) > 0


@pytest.mark.corpus
@pytest.mark.integration
def test_the_step_series_covers_priced_lengths_with_positive_costs(
    corpus_sample: tuple[tuple[str, ...], ...],
    corpus_device: torch.device,
) -> None:
    lengths = decile_lengths(corpus_sample, deciles=DECILES)

    costs = series(corpus_device, corpus_sample, lengths)

    assert costs
    assert tuple(cost.sentence_length for cost in costs) == priced_lengths(lengths)
    assert costs[-1].sentence_length > costs[0].sentence_length
    assert costs[-1].seconds_fastest > costs[0].seconds_fastest
    assert costs[-1].peak_bytes > costs[0].peak_bytes
    assert seconds_per_token(costs) > 0
    assert all(cost.available > 0 for cost in costs)
    assert all(cost.seconds_fastest > 0 for cost in costs)


@pytest.mark.corpus
@pytest.mark.integration
def test_the_step_cost_rises_across_the_corpus_decile_lengths(
    benchmark: BenchmarkFixture,
    corpus_sample: tuple[tuple[str, ...], ...],
    corpus_device: torch.device,
) -> None:
    histogram = length_histogram(corpus_sample)
    lengths = decile_lengths(corpus_sample, deciles=DECILES)
    costs: tuple[LengthCost, ...] = benchmark.pedantic(
        series, args=(corpus_device, corpus_sample, lengths), rounds=1, iterations=1
    )
    benchmark.extra_info["device"] = str(corpus_device)
    benchmark.extra_info["width"] = reference_width()
    benchmark.extra_info["batch_size"] = BATCH
    benchmark.extra_info["sentences"] = len(corpus_sample)
    benchmark.extra_info["histogram"] = [
        [length, count] for length, count in histogram.items()
    ]
    benchmark.extra_info["deciles"] = list(lengths)
    benchmark.extra_info["series"] = [cost._asdict() for cost in costs]
    benchmark.extra_info["seconds_per_token"] = seconds_per_token(costs)


@pytest.mark.corpus
@pytest.mark.integration
def test_real_training_steps_allocate_memory_at_the_working_point_and_tail(
    corpus_sample: tuple[tuple[str, ...], ...],
    corpus_device: torch.device,
) -> None:
    tail = tail_length(corpus_sample)

    for length in (WORKING_LENGTH, tail):
        step = step_at(corpus_device, length)
        torch.accelerator.reset_peak_memory_stats(corpus_device)

        run(step)
        peak = int(torch.accelerator.max_memory_allocated(corpus_device))

        assert step.shape == BatchShape(float(length), float(BATCH))
        assert peak > 0

    assert tail > WORKING_LENGTH


@pytest.mark.corpus
@pytest.mark.integration
def test_one_training_step_at_the_corpus_working_point(
    benchmark: BenchmarkFixture,
    corpus_sample: tuple[tuple[str, ...], ...],
    corpus_device: torch.device,
) -> None:
    step = step_at(corpus_device, WORKING_LENGTH)
    run(step)
    torch.accelerator.reset_peak_memory_stats(corpus_device)
    benchmark.pedantic(run, args=(step,), rounds=TIMED, iterations=1)
    peak = int(torch.accelerator.max_memory_allocated(corpus_device))
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the step ran but pytest-benchmark timed nothing")

    beyond = sum(1 for sentence in corpus_sample if len(sentence) > WORKING_LENGTH)
    benchmark.extra_info["tokens"] = WORKING_LENGTH
    benchmark.extra_info["items"] = BATCH
    benchmark.extra_info["width"] = reference_width()
    benchmark.extra_info["device"] = str(corpus_device)
    benchmark.extra_info["peak_bytes"] = peak
    benchmark.extra_info["ms_per_step"] = measured.stats.median * 1e3
    benchmark.extra_info["sentences"] = len(corpus_sample)
    benchmark.extra_info["sentences_beyond_the_working_point"] = beyond


@pytest.mark.corpus
@pytest.mark.integration
def test_one_training_step_at_the_corpus_longest_sentence(
    benchmark: BenchmarkFixture,
    corpus_sample: tuple[tuple[str, ...], ...],
    corpus_device: torch.device,
) -> None:
    length = tail_length(corpus_sample)
    step = step_at(corpus_device, length)
    run(step)
    torch.accelerator.reset_peak_memory_stats(corpus_device)
    benchmark.pedantic(run, args=(step,), rounds=TIMED, iterations=1)
    peak = int(torch.accelerator.max_memory_allocated(corpus_device))
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the step ran but pytest-benchmark timed nothing")

    benchmark.extra_info["tokens"] = length
    benchmark.extra_info["items"] = BATCH
    benchmark.extra_info["width"] = reference_width()
    benchmark.extra_info["device"] = str(corpus_device)
    benchmark.extra_info["peak_bytes"] = peak
    benchmark.extra_info["ms_per_step"] = measured.stats.median * 1e3
    benchmark.extra_info["sentences"] = len(corpus_sample)


def test_a_derivations_leaves_are_read_in_sentence_order(tmp_path: Path) -> None:
    written = tmp_path / "wsj_0000.auto"
    written.write_text(
        auto_file((("the", "cat", "sat"),)), encoding="utf-8", newline="\n"
    )

    assert corpus_sentences(tmp_path, at_least=1) == (("the", "cat", "sat"),)


def test_a_one_word_derivation_reads_back_as_one_word(tmp_path: Path) -> None:
    written = tmp_path / "wsj_0000.auto"
    written.write_text(auto_file((("dogs",),)), encoding="utf-8", newline="\n")

    assert corpus_sentences(tmp_path, at_least=1) == (("dogs",),)


def test_derivation_files_are_read_in_path_order_through_subdirectories(
    tmp_path: Path,
) -> None:
    for name, words in (("01", ("b",)), ("00", ("a",))):
        section = tmp_path / name
        section.mkdir()
        (section / f"wsj_{name}.auto").write_text(
            auto_file((words,)), encoding="utf-8", newline="\n"
        )

    assert corpus_sentences(tmp_path, at_least=2) == (("a",), ("b",))


def test_reading_stops_at_the_file_that_reaches_the_count(tmp_path: Path) -> None:
    for name in ("wsj_0000.auto", "wsj_0001.auto"):
        (tmp_path / name).write_text(
            auto_file((("a", "b"), ("c", "d"))), encoding="utf-8", newline="\n"
        )

    assert corpus_sentences(tmp_path, at_least=1) == (("a", "b"), ("c", "d"))
    assert len(corpus_sentences(tmp_path, at_least=3)) == 4


def test_a_root_holding_no_derivation_file_reads_no_sentence(tmp_path: Path) -> None:
    (tmp_path / "wsj_0000.parg").write_text("", encoding="utf-8", newline="\n")

    assert corpus_sentences(tmp_path, at_least=1) == ()


def test_the_histogram_counts_every_sentence_at_its_own_length() -> None:
    sample = (("a",), ("a", "b"), ("c", "d"), ("a", "b", "c"))

    assert length_histogram(sample) == {1: 1, 2: 2, 3: 1}


def test_the_histogram_is_ordered_from_the_shortest_length_up() -> None:
    sample = (("a", "b", "c"), ("a",), ("a", "b"))

    assert tuple(length_histogram(sample)) == (1, 2, 3)


def test_each_decile_length_is_the_middle_of_its_tenth_of_the_sample() -> None:
    sample = tuple(("w",) * length for length in range(1, 101))

    assert decile_lengths(sample, deciles=DECILES) == (
        5,
        15,
        25,
        35,
        45,
        55,
        65,
        75,
        85,
        95,
    )


def test_a_sample_massed_at_one_length_prices_that_length_and_tails_beyond_it() -> None:
    sample = (*(("w",) * 5 for _ in range(95)), *(("w",) * 80 for _ in range(5)))

    assert set(decile_lengths(sample, deciles=DECILES)) == {5}
    assert tail_length(sample) == 80


def test_the_decile_lengths_never_decrease_on_a_lopsided_sample() -> None:
    sample = (*(("w",) * 5 for _ in range(90)), *(("w",) * 44 for _ in range(10)))
    lengths = decile_lengths(sample, deciles=DECILES)

    assert lengths[0] == 5
    assert lengths[-1] == 44
    assert all(later >= earlier for earlier, later in pairwise(lengths))


def test_the_tail_length_is_the_longest_sentence_in_the_sample() -> None:
    sample = (("w",) * 3, ("w",) * 61, ("w",) * 7)

    assert tail_length(sample) == 61


def test_an_empty_sample_has_no_longest_sentence() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        tail_length(())


def test_an_empty_sample_has_no_decile_lengths() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        decile_lengths((), deciles=DECILES)


def test_a_sample_is_cut_into_at_least_one_quantile() -> None:
    with pytest.raises(ValueError, match="at least one quantile"):
        decile_lengths((("a",),), deciles=0)


def test_a_batch_holds_sample_sentences_cut_to_the_priced_length() -> None:
    sample = (("a", "b", "c"), ("d", "e"), ("f", "g", "h", "i"))

    assert cut_to(sample, 3, 2) == (("a", "b", "c"), ("f", "g", "h"))


def test_a_sentence_shorter_than_the_priced_length_is_left_out() -> None:
    sample = (("a",), ("b", "c", "d"))

    assert cut_to(sample, 3, 1) == (("b", "c", "d"),)


def test_the_long_enough_sentences_repeat_to_fill_the_batch() -> None:
    sample = (("a", "b"), ("c",))

    assert cut_to(sample, 2, 3) == (("a", "b"), ("a", "b"), ("a", "b"))


def test_a_length_no_sentence_reaches_is_refused() -> None:
    with pytest.raises(ValueError, match="reaches 9 tokens"):
        cut_to(((("a", "b"),)), 9, 1)


def test_a_length_two_quantiles_share_is_priced_once() -> None:
    assert priced_lengths((5, 5, 12, 5, 40)) == (5, 12, 40)


def test_the_priced_lengths_keep_the_order_they_were_asked_in() -> None:
    assert priced_lengths((40, 5, 12)) == (40, 5, 12)


def test_each_priced_length_reports_the_median_fastest_and_peak_of_its_runs() -> None:
    timed = {
        20: (Timing(0.5, 30), Timing(0.2, 10), Timing(0.3, 20)),
        40: (Timing(0.9, 90), Timing(0.7, 70)),
    }
    costs = summarize((("w",) * 25, ("w",) * 45), timed)

    assert costs == (
        LengthCost(20, 2, 0.3, 0.2, 30),
        LengthCost(40, 1, 0.8, 0.7, 90),
    )


def test_a_priced_length_reports_how_many_sample_sentences_reach_it() -> None:
    sample = (("w",) * 10, ("w",) * 20, ("w",) * 30)
    costs = summarize(sample, {20: (Timing(0.1, 1),)})

    assert costs[0].available == 2


def test_the_slope_of_the_priced_series_is_seconds_per_token() -> None:
    costs = (
        LengthCost(10, 100, 0.2, 0.2, 1),
        LengthCost(20, 50, 0.4, 0.4, 2),
        LengthCost(30, 10, 0.6, 0.6, 3),
    )

    assert seconds_per_token(costs) == pytest.approx(0.02)


def _gated(pytester: pytest.Pytester) -> pytest.RunResult:
    """Run one test that asks for the corpus root, under the root conftest's fixture.

    Parameters
    ----------
    pytester : pytest.Pytester
        The runner, whose directory the conftest and the test are written into.
    """
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(GATED)

    return pytester.runpytest_inprocess("-p", "no:cacheprovider", "-rs")


def test_an_unset_corpus_variable_skips_and_names_it(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LOFI_CCGBANK", raising=False)
    result = _gated(pytester)

    result.assert_outcomes(skipped=1)
    assert "set LOFI_CCGBANK" in result.stdout.str()


def test_a_corpus_variable_naming_no_directory_skips_and_names_the_path(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    absent = tmp_path / "nowhere"
    monkeypatch.setenv("LOFI_CCGBANK", str(absent))
    result = _gated(pytester)

    result.assert_outcomes(skipped=1)
    assert "is not a directory" in result.stdout.str()
    assert str(absent) in result.stdout.str()


def test_a_corpus_variable_naming_a_directory_runs_the_gated_test(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LOFI_CCGBANK", str(tmp_path))
    result = _gated(pytester)

    result.assert_outcomes(passed=1)
