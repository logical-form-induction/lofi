import math
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast, override

import pytest
import torch
from tests.conformance._sites import SiteObservation
from tests.conformance.component_plugin import plugin_context
from tests.conformance.token_encoder import (
    TokenEncoderConformance,
    TokenEncoderSuite,
)
from tests.pretrained_stand_ins import StandInLoader
from torch import nn

from lofi._artifacts import PinnedCheckpoint
from lofi.data.synthetic.training import SyntheticTrainingBatch
from lofi.nn.build import build_component
from lofi.nn.sites import ComponentId, Role, Site
from lofi.nn.tokens.llm2vec import (
    BidirectionalDecoder,
    adapt_masked_next_token,
    read_corpus,
)
from lofi.nn.tokens.window import REPORT, ContextWindow
from lofi.run import (
    ExecutionEnvironment,
    ExperimentSeed,
    ManifestIso,
    RandomStreams,
    SeedPolicy,
    installed_packages,
    open_training_manifest,
    plan_training,
    resolve_device_policy,
)
from lofi.spec import RunSpecCodec, config_search_path
from lofi.spec.artifacts import HubSourceSpec
from lofi.spec.optimizers.adamw import AdamWOptimizerSpec
from lofi.spec.tokens.llm2vec import (
    LLM2VecTokenEncoderSpec,
    MaskedNextTokenSettings,
    MaskedNextTokenSpec,
)
from lofi.spec.tokens.pretrained import ContextWindowSpec
from lofi.train import Featurized, ParserFittingPlan, ParserStep, prepare_training

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.nn.tokens.pretrained import (
        DecoderModel,
        DecoderTokenizer,
        EncoderArchitecturePlugin,
        PretrainedLoader,
        PretrainedTokenEncoder,
    )
    from lofi.spec import RunSpec
    from lofi.spec.tokens.llm2vec import MaskReplacementName
    from lofi.train import PreparedTraining

RUN: Final = (
    Path(__file__).resolve().parents[4] / "lofi/configs/run/synthetic-parser.yaml"
)
WORDS: Final = tuple(f"word-{index}" for index in range(8))
CHECKPOINT: Final = PinnedCheckpoint("an-org/a-decoder", "0" * 40)
WIDTH: Final = 6
BUDGET: Final = 3
TOKEN_ENCODER: Final = ComponentId("token_encoder")


def _loader(width: int = WIDTH) -> StandInLoader:
    return StandInLoader(WORDS, width=width, seed=3)


def _decoder(width: int = WIDTH) -> DecoderModel:
    return _loader(width).load_decoder(CHECKPOINT, dtype=torch.float32)


def _ids(*rows: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.long)


def _settings(
    *,
    mask: str = "<unk>",
    mask_probability: float = 0.5,
    replacement: MaskReplacementName = "mixed",
    steps: int = 4,
    learning_rate: float = 1e-2,
    parameterization: str = "full",
    seed: int = 0,
) -> MaskedNextTokenSettings:
    return MaskedNextTokenSettings(
        corpus="unread",
        mask=mask,
        mask_probability=mask_probability,
        replacement=replacement,
        mask_replacement_fraction=0.8,
        random_replacement_fraction=0.1,
        steps=steps,
        batch_size=2,
        optimizer=AdamWOptimizerSpec(learning_rate=learning_rate),
        parameterization=parameterization,
        rank=4,
        alpha=4.0,
        seed=seed,
    )


def _parameters(module: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _moved(before: list[torch.Tensor], module: nn.Module) -> bool:
    return any(
        not torch.equal(initial, now.detach())
        for initial, now in zip(before, module.parameters(), strict=True)
    )


def test_an_earlier_row_reads_a_later_token_once_the_causal_mask_is_lifted() -> None:
    model = _decoder()
    stack = model.get_decoder()
    first = _ids((2, 3, 4, 5))
    second = _ids((2, 3, 4, 6))
    live = torch.ones(1, 4, dtype=torch.long)
    causal = [
        stack(
            input_ids=ids,
            attention_mask=live,
            return_dict=True,
            output_hidden_states=False,
        ).last_hidden_state
        for ids in (first, second)
    ]
    decoder = BidirectionalDecoder(model)
    rows = [
        decoder(input_ids=ids, attention_mask=live, return_dict=True).last_hidden_state
        for ids in (first, second)
    ]

    assert torch.allclose(causal[0][0, 0], causal[1][0, 0])
    assert not torch.allclose(rows[0][0, 0], rows[1][0, 0])
    assert not torch.allclose(rows[0][0, :3], rows[1][0, :3])


def test_a_padded_row_moves_no_unpadded_row() -> None:
    decoder = BidirectionalDecoder(_decoder())
    padded = _ids((2, 3, 4, 0), (2, 3, 4, 7))
    live = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])

    rows = decoder(input_ids=padded, attention_mask=live, return_dict=True)

    assert torch.allclose(
        rows.last_hidden_state[0, :3], rows.last_hidden_state[1, :3], atol=1e-6
    )


def test_every_attention_module_is_told_it_masks_no_later_position() -> None:
    model = _decoder()
    flagged = [module for module in model.modules() if hasattr(module, "is_causal")]

    BidirectionalDecoder(model)

    assert flagged
    assert all(module.is_causal is False for module in flagged)


def test_the_width_and_the_logits_come_from_the_model() -> None:
    decoder = BidirectionalDecoder(_decoder(width=10))
    ids = _ids((2, 3, 4))

    logits = decoder.next_token_logits(ids, torch.ones_like(ids))

    assert decoder.config.hidden_size == 10
    assert tuple(logits.shape) == (1, 3, len(_loader().load_tokenizer(CHECKPOINT)))


def test_a_decoder_that_is_not_a_module_is_refused() -> None:
    with pytest.raises(TypeError, match="torch module"):
        BidirectionalDecoder(cast("DecoderModel", object()))


def test_a_corpus_is_read_one_item_per_nonblank_line(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "word-0   word-1\n\n  \nword-2\tword-3 \n", encoding="utf-8", newline="\n"
    )
    empty = tmp_path / "empty.txt"
    empty.write_text("\n \n", encoding="utf-8", newline="\n")

    assert read_corpus(corpus) == ("word-0 word-1", "word-2 word-3")

    with pytest.raises(ValueError, match="holds no line"):
        read_corpus(empty)


@pytest.mark.parametrize("replacement", ["mixed", "mask"])
def test_adaptation_lowers_the_masked_next_token_loss(
    replacement: MaskReplacementName,
) -> None:
    torch.manual_seed(0)
    decoder = BidirectionalDecoder(_decoder())
    lines = ("word-0 word-1 word-2 word-3", "word-4 word-5 word-6 word-7")

    losses = adapt_masked_next_token(
        decoder,
        _loader().load_tokenizer(CHECKPOINT),
        lines,
        settings=_settings(replacement=replacement, steps=60, mask_probability=0.5),
        rows=None,
    )

    assert 0 < len(losses) <= 60
    assert all(math.isfinite(loss) for loss in losses)
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10


def test_adaptation_trains_a_frozen_decoder_and_restores_its_mode() -> None:
    torch.manual_seed(0)
    decoder = BidirectionalDecoder(_decoder())
    decoder.eval()

    for parameter in decoder.parameters():
        parameter.requires_grad = False

    before = _parameters(decoder)

    adapt_masked_next_token(
        decoder,
        _loader().load_tokenizer(CHECKPOINT),
        ("word-0 word-1 word-2",),
        settings=_settings(mask_probability=1.0),
        rows=None,
    )

    assert _moved(before, decoder)
    assert decoder.training is False


@pytest.mark.parametrize(
    "lines",
    [
        ("word-0", "word-5"),
        ("unseen another", "neither nor"),
    ],
    ids=["no-predecessor", "special-tokens-only"],
)
def test_a_token_with_no_predecessor_or_a_special_token_is_never_chosen(
    lines: tuple[str, ...],
) -> None:
    decoder = BidirectionalDecoder(_decoder())
    before = _parameters(decoder)

    losses = adapt_masked_next_token(
        decoder,
        _loader().load_tokenizer(CHECKPOINT),
        lines,
        settings=_settings(mask_probability=1.0, replacement="mask"),
        rows=None,
    )

    assert losses == ()
    assert not _moved(before, decoder)


def test_adaptation_reads_no_row_past_the_declared_bound() -> None:
    decoder = BidirectionalDecoder(_decoder())
    before = _parameters(decoder)

    losses = adapt_masked_next_token(
        decoder,
        _loader().load_tokenizer(CHECKPOINT),
        ("word-0 word-1 word-2",),
        settings=_settings(mask_probability=1.0),
        rows=1,
    )

    assert losses == ()
    assert not _moved(before, decoder)


class _Recording(BidirectionalDecoder):
    def __init__(self, model: DecoderModel) -> None:
        super().__init__(model)
        self.read: list[torch.Tensor] = []

    @override
    def next_token_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        self.read.append(input_ids.clone())

        return super().next_token_logits(input_ids, attention_mask)


class _LeftPadded:
    """The stand-in tokenizer with every item's padding moved before its tokens."""

    def __init__(self) -> None:
        self._inner = _loader().load_tokenizer(CHECKPOINT)
        self.pad_token_id = self._inner.pad_token_id
        self.all_special_ids = self._inner.all_special_ids

    def get_vocab(self) -> dict[str, int]:
        return self._inner.get_vocab()

    def __len__(self) -> int:
        return len(self._inner)

    def __call__(
        self,
        texts: Sequence[str],
        *,
        padding: bool,
        return_tensors: str,
        truncation: bool,
        return_offsets_mapping: bool,
    ) -> object:
        batch = self._inner(
            texts,
            padding=padding,
            return_tensors=return_tensors,
            truncation=truncation,
            return_offsets_mapping=return_offsets_mapping,
        )

        for item, live in enumerate(batch.attention_mask.sum(dim=1).tolist()):
            shift = batch.input_ids.shape[1] - live
            batch.input_ids[item] = batch.input_ids[item].roll(shift)
            batch.attention_mask[item] = batch.attention_mask[item].roll(shift)

        return batch


def test_a_token_after_padding_is_never_chosen() -> None:
    decoder = _Recording(_decoder())
    tokenizer = cast("DecoderTokenizer", _LeftPadded())
    vocabulary = tokenizer.get_vocab()

    adapt_masked_next_token(
        decoder,
        tokenizer,
        ("word-1", "word-2 word-3 word-4"),
        settings=_settings(mask_probability=1.0, replacement="mask", steps=1),
        rows=None,
    )

    unk, pad = vocabulary["<unk>"], vocabulary["<pad>"]
    read = sorted(decoder.read[0].tolist())

    assert read == sorted(
        [[vocabulary["word-1"], pad, pad], [vocabulary["word-2"], unk, unk]]
    )


def test_a_bound_cuts_the_tail_of_a_left_padded_line() -> None:
    decoder = _Recording(_decoder())
    tokenizer = cast("DecoderTokenizer", _LeftPadded())
    vocabulary = tokenizer.get_vocab()

    adapt_masked_next_token(
        decoder,
        tokenizer,
        ("word-1 word-2", "word-3 word-4 word-5 word-6"),
        settings=_settings(mask_probability=1.0, replacement="mask", steps=1),
        rows=2,
    )

    unk = vocabulary["<unk>"]

    assert sorted(decoder.read[0].tolist()) == sorted(
        [[vocabulary["word-1"], unk], [vocabulary["word-3"], unk]]
    )


def test_the_loss_scores_each_chosen_token_from_the_row_before_it() -> None:
    model = _decoder()
    reference = BidirectionalDecoder(_decoder())
    vocabulary = _loader().load_tokenizer(CHECKPOINT).get_vocab()
    first, second = vocabulary["word-1"], vocabulary["word-2"]
    corrupted = _ids((first, vocabulary["<unk>"]))

    losses = adapt_masked_next_token(
        BidirectionalDecoder(model),
        _loader().load_tokenizer(CHECKPOINT),
        ("word-1 word-2",),
        settings=_settings(mask_probability=1.0, replacement="mask", steps=1),
        rows=None,
    )
    logits = reference.next_token_logits(corrupted, torch.ones_like(corrupted))
    expected = torch.nn.functional.cross_entropy(logits[:, 0], torch.tensor([second]))

    assert losses == pytest.approx((float(expected.detach()),), rel=1e-5)


def test_mixed_replacement_masks_four_in_five_chosen_tokens() -> None:
    torch.manual_seed(0)
    decoder = _Recording(_decoder())
    vocabulary = _loader().load_tokenizer(CHECKPOINT).get_vocab()
    line = " ".join(WORDS * 25)

    adapt_masked_next_token(
        decoder,
        _loader().load_tokenizer(CHECKPOINT),
        (line,),
        settings=_settings(mask_probability=1.0, steps=1),
        rows=None,
    )

    read = decoder.read[0][0, 1:]
    original = torch.tensor([vocabulary[word] for word in line.split(" ")])[1:]
    masked = float((read == vocabulary["<unk>"]).float().mean())
    kept = float((read == original).float().mean())

    assert 0.7 < masked < 0.9
    assert 0.05 < kept < 0.25


@pytest.mark.parametrize("learning_rate", [1e-4, 1e-3])
def test_one_adaptation_step_moves_a_weight_by_about_the_learning_rate(
    learning_rate: float,
) -> None:
    decoder = BidirectionalDecoder(_decoder())
    before = _parameters(decoder)

    adapt_masked_next_token(
        decoder,
        _loader().load_tokenizer(CHECKPOINT),
        ("word-1 word-2 word-3",),
        settings=_settings(
            mask_probability=1.0,
            replacement="mask",
            steps=1,
            learning_rate=learning_rate,
        ),
        rows=None,
    )
    largest = max(
        float((now.detach() - initial).abs().max())
        for initial, now in zip(before, decoder.parameters(), strict=True)
    )

    assert learning_rate / 2 < largest < learning_rate * 1.5


def test_the_declared_window_bounds_the_rows_adaptation_reads(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("word-0 word-1 word-2\n", encoding="utf-8", newline="\n")
    fresh = _parameters(BidirectionalDecoder(_decoder()))
    declared = LLM2VecTokenEncoderSpec(
        checkpoint=HubSourceSpec(
            repository=CHECKPOINT.repository, revision=CHECKPOINT.revision
        ),
        window=ContextWindowSpec(rows=1),
        adaptation=MaskedNextTokenSpec(
            corpus=str(corpus), mask="<unk>", mask_probability=1.0, steps=2
        ),
    )

    encoder = cast(
        "PretrainedTokenEncoder",
        build_component(declared, plugin_context(loader=_loader())),
    )

    assert not _moved(fresh, encoder)


def test_a_mask_token_the_vocabulary_lacks_or_an_empty_corpus_is_refused() -> None:
    decoder = BidirectionalDecoder(_decoder())
    tokenizer = _loader().load_tokenizer(CHECKPOINT)

    with pytest.raises(ValueError, match="no mask token '_'"):
        adapt_masked_next_token(
            decoder,
            tokenizer,
            ("word-0 word-1",),
            settings=_settings(mask="_"),
            rows=None,
        )

    with pytest.raises(ValueError, match="at least one line"):
        adapt_masked_next_token(decoder, tokenizer, (), settings=_settings(), rows=None)


def _document(directory: Path, *lines: str) -> Path:
    document = directory / "run.yaml"
    body = "".join(f"{line}\n" for line in lines)
    document.write_text(
        "defaults:\n  - synthetic-parser\n  - model.token_encoder: llm2vec\n"
        f"artifacts:\n  root: {directory / 'artifacts'}\n"
        f"model:\n  token_encoder:\n{body}",
        encoding="utf-8",
        newline="\n",
    )

    return document


def _composed(directory: Path, *lines: str) -> RunSpec:
    return RunSpecCodec(search_path=(RUN.parent, *config_search_path(directory)))(
        _document(directory, *lines), overrides=("trainer.accumulation_steps=1",)
    )


def _prepared(run_spec: RunSpec, loader: PretrainedLoader) -> PreparedTraining:
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(13, samplers=blueprint.sampler_streams))
    streams.apply()

    return prepare_training(
        blueprint,
        streams,
        resolve_device_policy(run_spec),
        pretrained_resources={"token_encoder": loader},
    )


def _adapting(directory: Path) -> tuple[str, ...]:
    corpus = directory / "corpus.txt"
    corpus.write_text(
        "word-0 word-1 word-2\nword-3 word-0 word-1\nword-4 word-5 word-6\n",
        encoding="utf-8",
        newline="\n",
    )

    return (
        "    adaptation:",
        f"      corpus: {corpus}",
        "      mask: <unk>",
        "      mask_probability: 1.0",
        "      steps: 3",
        "      batch_size: 2",
        "      parameterization: lora",
        "      rank: 2",
        "      alpha: 2.0",
        "      seed: 17",
        "      optimizer:",
        "        kind: adamw",
        "        learning_rate: 0.01",
    )


def test_a_document_seed_reproduces_lora_masks_updates_and_losses(
    tmp_path: Path,
) -> None:
    run_spec = _composed(tmp_path, *_adapting(tmp_path))
    declared = cast("LLM2VecTokenEncoderSpec", run_spec.model.token_encoder)
    adaptation = declared.settings().adaptation

    if adaptation is None:
        pytest.fail("the document declared an adaptation")

    lines = read_corpus(Path(adaptation.corpus))
    first = _Recording(_decoder())
    second = _Recording(_decoder())
    first_losses = adapt_masked_next_token(
        first,
        _loader().load_tokenizer(CHECKPOINT),
        lines,
        settings=adaptation,
        rows=None,
    )
    second_losses = adapt_masked_next_token(
        second,
        _loader().load_tokenizer(CHECKPOINT),
        lines,
        settings=adaptation,
        rows=None,
    )

    assert first_losses == second_losses
    assert len(first.read) == len(second.read) > 0
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.read, second.read, strict=True)
    )
    assert first.state_dict().keys() == second.state_dict().keys()
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )
    trained = {
        name: parameter
        for name, parameter in first.named_parameters()
        if parameter.requires_grad
    }
    assert trained
    assert all(name.endswith((".down", ".up")) for name in trained)
    assert any(
        name.endswith(".up") and bool(torch.count_nonzero(parameter))
        for name, parameter in trained.items()
    )


def test_a_document_selecting_the_fragment_trains_a_point(tmp_path: Path) -> None:
    run_spec = _composed(tmp_path, "    trainable: true", *_adapting(tmp_path))
    prepared = _prepared(run_spec, _loader())
    plan = cast("ParserFittingPlan", prepared.plan)
    encoder = prepared.model.modules["token_encoder"]
    before = _parameters(encoder)

    assert isinstance(plan.step, ParserStep)
    results = []

    for position, batch in enumerate(plan.batches):
        assert isinstance(batch, SyntheticTrainingBatch)
        results.append(plan.step.run(Featurized(batch, position)).result)

    assert isinstance(run_spec.model.token_encoder, LLM2VecTokenEncoderSpec)
    assert [result.step for result in results] == list(range(1, len(results) + 1))
    assert all(result.metrics["optimizer/loss_finite"] == 1.0 for result in results)
    assert all(
        math.isfinite(value) for result in results for value in result.metrics.values()
    )
    assert _moved(before, encoder)


def test_a_document_declaring_a_frozen_decoder_trains_a_point_around_it(
    tmp_path: Path,
) -> None:
    run_spec = _composed(tmp_path, "    trainable: false")
    prepared = _prepared(run_spec, _loader())
    plan = cast("ParserFittingPlan", prepared.plan)
    encoder = prepared.model.modules["token_encoder"]
    before = _parameters(encoder)

    assert isinstance(plan.step, ParserStep)
    results = []

    for position, batch in enumerate(plan.batches):
        assert isinstance(batch, SyntheticTrainingBatch)
        results.append(plan.step.run(Featurized(batch, position)).result)

    assert all(result.metrics["optimizer/loss_finite"] == 1.0 for result in results)
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    assert not _moved(before, encoder)
    assert prepared.model.groups.trainable


@pytest.mark.parametrize("trainable", [False, True], ids=["frozen", "fine-tuned"])
def test_the_declared_freeze_state_reaches_every_decoder_parameter(
    *, trainable: bool
) -> None:
    declared = LLM2VecTokenEncoderSpec(
        checkpoint=HubSourceSpec(
            repository=CHECKPOINT.repository, revision=CHECKPOINT.revision
        ),
        trainable=trainable,
    )

    encoder = cast(
        "PretrainedTokenEncoder",
        build_component(declared, plugin_context(loader=_loader())),
    )

    assert encoder.trainable is trainable
    assert encoder.window == ContextWindow(512, overflow=REPORT)
    assert all(
        parameter.requires_grad is trainable for parameter in encoder.parameters()
    )


def _windowed() -> PretrainedTokenEncoder:
    declared = LLM2VecTokenEncoderSpec(
        checkpoint=HubSourceSpec(
            repository=CHECKPOINT.repository, revision=CHECKPOINT.revision
        ),
        window=ContextWindowSpec(rows=BUDGET),
    )

    return cast(
        "PretrainedTokenEncoder",
        build_component(
            declared,
            plugin_context(component=str(TOKEN_ENCODER), loader=_loader()),
        ),
    )


class TestLLM2VecEncoder(TokenEncoderSuite):
    @pytest.fixture
    def conformance(self) -> TokenEncoderConformance[PretrainedTokenEncoder]:
        site = Site(TOKEN_ENCODER, Role.TOKEN_EMBEDDING)

        return TokenEncoderConformance(
            build=_windowed,
            items=(("word-0", "word-1"), ("word-2",)),
            overlong=(("word-0", "word-1", "word-2", "word-3", "word-4"),),
            budget=BUDGET,
            sites=SiteObservation(
                build=_windowed,
                observe=lambda subject: {
                    site: subject.encode_words(((WORDS[0],),)).vectors
                },
                read=lambda subject: (
                    subject.encode_words(((WORDS[0],),)).vectors.tensor,
                ),
            ),
        )


def test_the_encoder_width_is_the_models_and_its_identity_is_in_the_manifest(
    tmp_path: Path,
) -> None:
    run_spec = _composed(tmp_path, "    trainable: false")
    blueprint = plan_training(run_spec)
    encoder = cast(
        "PretrainedTokenEncoder",
        _prepared(run_spec, _loader(width=8)).model.modules["token_encoder"],
    )
    manifest = open_training_manifest(
        blueprint,
        ExperimentSeed(0).point("llm2vec", 0),
        ExecutionEnvironment(installed_packages(), "0123abcd", "cpu"),
    )
    declared = manifest.run_spec.model.token_encoder
    written = ManifestIso().forward(manifest).decode()

    assert isinstance(run_spec.model.token_encoder, LLM2VecTokenEncoderSpec)
    assert "width" not in run_spec.model.token_encoder.model_dump()
    assert encoder.output_width == 8
    assert isinstance(declared, LLM2VecTokenEncoderSpec)
    assert encoder.checkpoint == declared.settings().checkpoint
    assert declared.settings().checkpoint.repository in written
    assert declared.settings().checkpoint.revision in written
    assert (
        manifest.component_fingerprints["model.token_encoder"]
        == (blueprint.component_fingerprints["model.token_encoder"])
    )


def test_the_declared_adaptation_is_what_moves_the_loaded_weights(
    tmp_path: Path,
) -> None:
    adapted = tmp_path / "adapted"
    loaded = tmp_path / "loaded"
    adapted.mkdir()
    loaded.mkdir()
    loader = _loader()
    run_spec = _composed(adapted, *_adapting(adapted))
    declared = cast("LLM2VecTokenEncoderSpec", run_spec.model.token_encoder)
    fresh = _parameters(
        BidirectionalDecoder(
            loader.load_decoder(
                declared.settings().checkpoint,
                dtype=torch.float32,
            )
        )
    )

    with_adaptation = _prepared(run_spec, loader).model.modules["token_encoder"]
    without = _prepared(
        _composed(loaded, "    trainable: false"), loader
    ).model.modules["token_encoder"]

    assert _moved(fresh, with_adaptation)
    assert not _moved(fresh, without)


class _EncoderOnlyLoader:
    def __init__(self) -> None:
        self._inner = _loader()

    def load_encoder(
        self,
        checkpoint: PinnedCheckpoint,
        *,
        architecture: EncoderArchitecturePlugin,
        dtype: torch.dtype,
    ) -> object:
        return self._inner.load_encoder(
            checkpoint, architecture=architecture, dtype=dtype
        )

    def load_tokenizer(self, checkpoint: PinnedCheckpoint) -> object:
        return self._inner.load_tokenizer(checkpoint)


def test_a_loader_that_loads_no_decoder_is_refused(tmp_path: Path) -> None:
    run_spec = _composed(tmp_path, "    trainable: false")

    with pytest.raises(TypeError, match="loads none"):
        _prepared(run_spec, cast("PretrainedLoader", _EncoderOnlyLoader()))
