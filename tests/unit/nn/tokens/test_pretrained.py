from typing import TYPE_CHECKING, Final, Literal, NoReturn, cast, override
from urllib.parse import quote

import pytest
import torch
from tests.conformance._sites import SiteObservation
from tests.conformance.component_plugin import plugin_context
from tests.conformance.token_encoder import (
    TokenEncoderConformance,
    TokenEncoderSuite,
)

from lofi._artifacts import ArtifactFetcher, PinnedCheckpoint
from lofi.nn.build import build_component
from lofi.nn.policy import DevicePolicy
from lofi.nn.sites import ComponentId, ComponentKind, Role, Site
from lofi.nn.tokens.pool import MeanSubwordPooler
from lofi.nn.tokens.pretrained import (
    ENCODER_ONLY,
    T5_ENCODER,
    AbsentModelError,
    CheckpointAligner,
    CrossingPieceError,
    DecoderLoader,
    EncoderArchitecturePlugin,
    HuggingFaceCausalLoader,
    HuggingFacePretrainedLoader,
    LocalPretrainedLoader,
    MissingPretrainedBackendError,
    PretrainedLoader,
    PretrainedTokenEncoder,
    WordRows,
    ZeroPieceWordError,
    declared_window,
    load_aligner,
    load_pretrained,
)
from lofi.nn.tokens.window import (
    REFUSE,
    REPORT,
    ContextOverflowError,
    ContextWindow,
    OverflowPlugin,
)
from lofi.probe import ensure_publishes
from lofi.spec.artifacts import HubSourceSpec
from lofi.spec.tokens.pretrained import (
    ContextWindowSettings,
    ContextWindowSpec,
    FixedLayerReadoutSettings,
    LastLayerReadoutSettings,
    LayerReadoutSettings,
    PretrainedTokenEncoderSpec,
    ScalarMixReadoutSettings,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from torch import Tensor

TOKEN_ENCODER: Final = ComponentId("tokens")
ITEMS: Final = (("ab", "c"), ("d",))
OVERLONG: Final = (("aa", "bb", "cc", "dd"),)
BUDGET: Final = 6
LOCAL: Final = PinnedCheckpoint("an-org/a-model", "a" * 40)
GENERAL: Final = PinnedCheckpoint("an-org/an-encoder", "b" * 40)
T5: Final = PinnedCheckpoint("an-org/a-t5", "c" * 40)


class _Snapshots:
    """Write an empty snapshot directory for every requested revision."""

    def snapshot(self, repository: str, revision: str, target: Path, /) -> None:
        del repository, revision
        target.mkdir(parents=True, exist_ok=True)


class _NoArchives:
    """Refuse every archive download."""

    def download(self, url: str, target: Path, /) -> NoReturn:
        del target
        pytest.fail(f"a hub loader downloaded the archive {url}")


def _fetcher(root: Path, /) -> ArtifactFetcher:
    return ArtifactFetcher(
        root, archives=_NoArchives(), snapshots=_Snapshots(), offline=False
    )


def _snapshot(root: Path, checkpoint: PinnedCheckpoint, /) -> str:
    return str(
        root / "snapshots" / quote(checkpoint.repository, safe="") / checkpoint.revision
    )


class _Configuration:
    def __init__(self, *, hidden_size: int, d_model: int) -> None:
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.num_hidden_layers = 1
        self.num_layers = 1


class _Output:
    def __init__(
        self, last_hidden_state: Tensor, hidden_states: tuple[Tensor, ...] | None = None
    ) -> None:
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class _Encoder(torch.nn.Module):
    def __init__(self, *, hidden_size: int = 5, d_model: int = 7) -> None:
        super().__init__()
        self.config = _Configuration(hidden_size=hidden_size, d_model=d_model)
        self.scale = torch.nn.Parameter(
            torch.linspace(0.5, 1.5, max(hidden_size, d_model))
        )
        self.calls = 0

    @override
    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        return_dict: bool,
        output_hidden_states: bool,
    ) -> _Output:
        assert return_dict
        self.calls += 1
        width = self.config.d_model
        basis = torch.nn.functional.one_hot(
            input_ids.remainder(width), num_classes=width
        ).to(dtype=self.scale.dtype)
        rows = basis * self.scale[:width]
        masked = rows * attention_mask.unsqueeze(-1)
        hidden = (masked * 0.5, masked) if output_hidden_states else None
        return _Output(masked, hidden)


class _WrongWidthEncoder(_Encoder):
    @override
    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        return_dict: bool,
        output_hidden_states: bool,
    ) -> _Output:
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            output_hidden_states=output_hidden_states,
        )

        return _Output(output.last_hidden_state[..., :-1])


class _Batch:
    def __init__(
        self, input_ids: Tensor, attention_mask: Tensor, offset_mapping: Tensor
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.offset_mapping = offset_mapping


class _Tokenizer:
    pad_token_id = 0
    model_max_length = 128

    def _spans(self, text: str) -> list[tuple[int, int]]:
        return [(at, at + 1) for at, character in enumerate(text) if character != " "]

    def __call__(
        self,
        texts: Sequence[str],
        *,
        padding: bool,
        return_tensors: str,
        truncation: bool,
        return_offsets_mapping: bool,
    ) -> _Batch:
        assert padding
        assert return_tensors == "pt"
        assert not truncation
        assert return_offsets_mapping

        ids: list[list[int]] = []
        offsets: list[list[tuple[int, int]]] = []

        for text in texts:
            spans = self._spans(text)
            ids.append([ord(text[start]) % 47 + 2 for start, _ in spans] + [1])
            offsets.append([*spans, (0, 0)])

        longest = max(map(len, ids), default=0)

        for item_ids, item_offsets in zip(ids, offsets, strict=True):
            padding_rows = longest - len(item_ids)
            item_ids.extend([self.pad_token_id] * padding_rows)
            item_offsets.extend([(0, 0)] * padding_rows)

        tensor = torch.tensor(ids, dtype=torch.long)
        mask = tensor.ne(self.pad_token_id).to(dtype=torch.long)

        return _Batch(tensor, mask, torch.tensor(offsets, dtype=torch.long))


class _ZeroPieceTokenizer(_Tokenizer):
    @override
    def _spans(self, text: str) -> list[tuple[int, int]]:
        second = text.split(" ")[1] if " " in text else ""
        start = text.find(" ") + 1
        spans = super()._spans(text)

        return [span for span in spans if not start <= span[0] < start + len(second)]


class _PrefixedTokenizer(_Tokenizer):
    @override
    def _spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start = 0

        for word in text.split(" "):
            spans.append((max(start - 1, 0), start + 1))
            spans.extend((at, at + 1) for at in range(start + 1, start + len(word)))
            start += len(word) + 1

        return spans


class _GluingTokenizer(_Tokenizer):
    @override
    def _spans(self, text: str) -> list[tuple[int, int]]:
        return [(0, len(text))]


class _GappedTokenizer(_Tokenizer):
    @override
    def _spans(self, text: str) -> list[tuple[int, int]]:
        return [(0, 1), (1, 1), (1, len(text))]


class _WhitespaceRowTokenizer(_Tokenizer):
    def __init__(self) -> None:
        self.calls = 0

    @override
    def _spans(self, text: str) -> list[tuple[int, int]]:
        self.calls += 1
        spans: list[tuple[int, int]] = []
        at = 0

        while at < len(text):
            end = at + 1

            if text[at] == " ":
                while end < len(text) and text[end] == " ":
                    end += 1

                if end - at > 1:
                    spans.append((at, end))
            else:
                spans.append((at, end))

            at = end

        return spans


class _DelimiterTokenizer(_Tokenizer):
    @override
    def _spans(self, text: str) -> list[tuple[int, int]]:
        return [
            (at, at + 1)
            for at, character in enumerate(text)
            if character not in {" ", "|"}
        ]


class _Loader:
    def __init__(self, encoder: _Encoder | None = None) -> None:
        self.encoder = _Encoder() if encoder is None else encoder
        self.tokenizer = _Tokenizer()
        self.encoder_requests: list[
            tuple[PinnedCheckpoint, EncoderArchitecturePlugin, torch.dtype]
        ] = []
        self.tokenizer_requests: list[PinnedCheckpoint] = []

    def load_encoder(
        self,
        checkpoint: PinnedCheckpoint,
        *,
        architecture: EncoderArchitecturePlugin,
        dtype: torch.dtype,
    ) -> _Encoder:
        self.encoder_requests.append((checkpoint, architecture, dtype))
        return self.encoder.to(dtype=dtype)

    def load_tokenizer(self, checkpoint: PinnedCheckpoint) -> _Tokenizer:
        self.tokenizer_requests.append(checkpoint)
        return self.tokenizer


class _EncoderFactory:
    def __init__(self, encoder: _Encoder) -> None:
        self.encoder = encoder
        self.requests: list[tuple[str, bool, torch.dtype]] = []
        self.configured: list[torch.dtype] = []
        self.constructed: list[_Configuration] = []

    def __call__(self, configuration: _Configuration) -> _Encoder:
        self.constructed.append(configuration)
        self.encoder.config = configuration
        return self.encoder

    def from_pretrained(
        self,
        directory: str,
        *,
        local_files_only: bool,
        dtype: torch.dtype,
    ) -> _Encoder:
        self.requests.append((directory, local_files_only, dtype))
        return self.encoder

    def from_config(
        self, configuration: _Configuration, *, dtype: torch.dtype
    ) -> _Encoder:
        self.configured.append(dtype)
        self.encoder.config = configuration
        return self.encoder


class _ConfigurationFactory:
    def __init__(self, configuration: _Configuration) -> None:
        self.configuration = configuration
        self.requests: list[tuple[str, bool]] = []

    def from_pretrained(
        self, checkpoint: str, *, local_files_only: bool
    ) -> _Configuration:
        self.requests.append((checkpoint, local_files_only))
        return self.configuration


class _TokenizerFactory:
    def __init__(self, tokenizer: _Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.requests: list[tuple[str, bool, bool]] = []

    def from_pretrained(
        self,
        directory: str,
        *,
        local_files_only: bool,
        use_fast: bool,
    ) -> _Tokenizer:
        self.requests.append((directory, local_files_only, use_fast))
        return self.tokenizer


class _FailingFactory:
    def from_pretrained(
        self,
        directory: str,
        *,
        local_files_only: bool,
        dtype: torch.dtype,
    ) -> _Encoder:
        del dtype
        pytest.fail(f"the general factory loaded {directory}, {local_files_only}")


class _CausalModel:
    def eval(self) -> _CausalModel:
        return self


class _CausalFactory:
    def __init__(self) -> None:
        self.model = _CausalModel()
        self.requests: list[tuple[str, bool, torch.dtype]] = []

    def from_pretrained(
        self,
        directory: str,
        *,
        local_files_only: bool,
        dtype: torch.dtype,
    ) -> _CausalModel:
        self.requests.append((directory, local_files_only, dtype))
        return self.model


class _AbsentFactory:
    def from_pretrained(
        self,
        directory: str,
        *,
        local_files_only: bool,
        dtype: torch.dtype,
    ) -> NoReturn:
        del dtype
        message = f"{directory} holds no loadable model ({local_files_only})"
        raise OSError(message)


class _Backend:
    def __init__(self) -> None:
        self.mt5 = _EncoderFactory(_Encoder())
        self.general = _EncoderFactory(_HiddenEncoder())
        self.tokenizer = _TokenizerFactory(_Tokenizer())
        self.configuration = _ConfigurationFactory(
            _Configuration(hidden_size=5, d_model=7)
        )
        self.AutoConfig = self.configuration
        self.MT5EncoderModel = self.mt5
        self.AutoModel = self.general
        self.AutoTokenizer = self.tokenizer
        self.causal = _CausalFactory()
        self.AutoModelForCausalLM = self.causal


def _t5_encoder(
    stack: _Encoder | None = None,
    tokenizer: _Tokenizer | None = None,
    *,
    trainable: bool = False,
    window: ContextWindow | None = None,
) -> PretrainedTokenEncoder:
    selected_tokenizer = _Tokenizer() if tokenizer is None else tokenizer
    return PretrainedTokenEncoder(
        _Encoder() if stack is None else stack,
        selected_tokenizer,
        architecture=T5_ENCODER,
        checkpoint=T5,
        component=TOKEN_ENCODER,
        trainable=trainable,
        window=(
            ContextWindow(selected_tokenizer.model_max_length, overflow=REPORT)
            if window is None
            else window
        ),
        readout=LastLayerReadoutSettings(),
        join=" ",
    )


def _t5(*, trainable: bool = True) -> PretrainedTokenEncoder:
    return _t5_encoder(
        trainable=trainable, window=ContextWindow(BUDGET, overflow=REPORT)
    )


def _general() -> PretrainedTokenEncoder:
    return PretrainedTokenEncoder(
        _HiddenEncoder(),
        _Tokenizer(),
        architecture=ENCODER_ONLY,
        checkpoint=GENERAL,
        component=TOKEN_ENCODER,
        trainable=True,
        window=ContextWindow(BUDGET, overflow=REPORT),
        readout=LastLayerReadoutSettings(),
        join=" ",
    )


class _HiddenEncoder(_Encoder):
    @override
    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        return_dict: bool,
        output_hidden_states: bool,
    ) -> _Output:
        assert return_dict
        width = self.config.hidden_size
        basis = torch.nn.functional.one_hot(
            input_ids.remainder(width), num_classes=width
        ).to(dtype=self.scale.dtype)
        rows = basis * self.scale[:width]
        masked = rows * attention_mask.unsqueeze(-1)
        hidden = (masked * 0.5, masked) if output_hidden_states else None
        return _Output(masked, hidden)


class _LayeredEncoder(_HiddenEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.config.num_hidden_layers = 6

    @override
    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        return_dict: bool,
        output_hidden_states: bool,
    ) -> _Output:
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            output_hidden_states=False,
        )
        hidden = (
            tuple(output.last_hidden_state * scale for scale in range(1, 8))
            if output_hidden_states
            else None
        )
        return _Output(output.last_hidden_state, hidden)


@pytest.mark.parametrize(
    ("readout", "expected_scale"),
    [
        (LastLayerReadoutSettings(), 1.0),
        (FixedLayerReadoutSettings(6), 7.0),
        (ScalarMixReadoutSettings(), 4.0),
    ],
    ids=("last", "layer-6", "scalar-mix"),
)
def test_each_layer_readout_point_trains(
    readout: LayerReadoutSettings, expected_scale: float
) -> None:
    subject = PretrainedTokenEncoder(
        _LayeredEncoder(),
        _Tokenizer(),
        architecture=ENCODER_ONLY,
        checkpoint=GENERAL,
        component=TOKEN_ENCODER,
        trainable=True,
        window=ContextWindow(BUDGET, overflow=REPORT),
        readout=readout,
        join=" ",
    )
    before = tuple(parameter.detach().clone() for parameter in subject.parameters())
    encoded = subject.encode_words((("ab",),)).vectors.tensor
    baseline = _general().encode_words((("ab",),)).vectors.tensor

    assert torch.allclose(encoded, baseline * expected_scale)

    encoded.square().sum().backward()
    torch.optim.SGD(subject.parameters(), lr=1e-2).step()

    assert all(parameter.grad is not None for parameter in subject.parameters())
    assert any(
        not torch.equal(initial, parameter.detach())
        for initial, parameter in zip(before, subject.parameters(), strict=True)
    )


def test_the_declared_join_controls_both_tokenization_and_alignment() -> None:
    subject = PretrainedTokenEncoder(
        _LayeredEncoder(),
        _DelimiterTokenizer(),
        architecture=ENCODER_ONLY,
        checkpoint=GENERAL,
        component=TOKEN_ENCODER,
        trainable=False,
        window=ContextWindow(BUDGET, overflow=REPORT),
        readout=LastLayerReadoutSettings(),
        join="|",
    )

    encoded = subject.encode_words((("ab", "c"),))

    assert encoded.alignments[0].rows == ((0, 2), (2, 3))


def _t5_conformance() -> TokenEncoderConformance[PretrainedTokenEncoder]:
    site = Site(TOKEN_ENCODER, Role.TOKEN_EMBEDDING)

    return TokenEncoderConformance(
        build=_t5,
        items=ITEMS,
        overlong=OVERLONG,
        budget=BUDGET,
        sites=SiteObservation(
            build=_t5,
            observe=lambda subject: {site: subject.encode_words(ITEMS).vectors},
            read=lambda subject: (subject.encode_words(ITEMS).vectors.tensor,),
        ),
    )


def _general_conformance() -> TokenEncoderConformance[PretrainedTokenEncoder]:
    site = Site(TOKEN_ENCODER, Role.TOKEN_EMBEDDING)

    return TokenEncoderConformance(
        build=_general,
        items=ITEMS,
        overlong=OVERLONG,
        budget=BUDGET,
        sites=SiteObservation(
            build=_general,
            observe=lambda subject: {site: subject.encode_words(ITEMS).vectors},
            read=lambda subject: (subject.encode_words(ITEMS).vectors.tensor,),
        ),
    )


class TestT5Encoder(TokenEncoderSuite):
    @pytest.fixture
    def conformance(self) -> TokenEncoderConformance[PretrainedTokenEncoder]:
        return _t5_conformance()


class TestPretrainedEncoder(TokenEncoderSuite):
    @pytest.fixture
    def conformance(self) -> TokenEncoderConformance[PretrainedTokenEncoder]:
        return _general_conformance()


@pytest.mark.parametrize("build", [_t5, _general])
def test_a_pretrained_encoder_publishes_a_valid_tagged_token_map(
    build: Callable[[], PretrainedTokenEncoder],
) -> None:
    published = build().site_map()

    assert published.component == TOKEN_ENCODER
    assert published.kind is ComponentKind.TOKEN_ENCODER
    assert ensure_publishes(published) is published


def test_a_t5_encoder_reads_d_model_while_an_encoder_only_one_reads_hidden_size() -> (
    None
):
    stack = _Encoder(hidden_size=3, d_model=11)
    t5 = _t5_encoder(stack)
    general = PretrainedTokenEncoder(
        stack,
        _Tokenizer(),
        architecture=ENCODER_ONLY,
        checkpoint=GENERAL,
        component=TOKEN_ENCODER,
        trainable=True,
        window=ContextWindow(_Tokenizer.model_max_length, overflow=REPORT),
        readout=LastLayerReadoutSettings(),
        join=" ",
    )

    assert t5.output_width == 11
    assert t5.architecture is T5_ENCODER
    assert general.output_width == 3
    assert general.architecture is ENCODER_ONLY


@pytest.mark.parametrize("state", ["frozen", "fine-tuned"])
def test_a_t5_encoder_supports_both_freeze_states(state: str) -> None:
    trainable = state == "fine-tuned"
    subject = _t5(trainable=trainable)
    loss = subject.encode_words((("abc",),)).vectors.tensor.sum()

    if trainable:
        loss.backward()

    parameters = tuple(subject.parameters())

    assert parameters
    assert all(parameter.requires_grad is trainable for parameter in parameters)
    assert all((parameter.grad is not None) is trainable for parameter in parameters)


def test_a_t5_encoder_loads_only_the_encoder_and_keeps_no_decoder_state() -> None:
    loader = _Loader()
    subject = load_pretrained(
        T5,
        architecture=T5_ENCODER,
        component=TOKEN_ENCODER,
        trainable=False,
        window=ContextWindowSettings(None, "report"),
        readout=LastLayerReadoutSettings(),
        join=" ",
        loader=cast("PretrainedLoader", loader),
        dtype=torch.float32,
    )

    assert loader.encoder_requests == [(T5, T5_ENCODER, torch.float32)]
    assert loader.tokenizer_requests == [T5]
    assert subject.checkpoint == T5
    assert subject.architecture is T5_ENCODER
    assert all("decoder" not in name for name in subject.state_dict())
    assert all("decoder" not in name for name, _ in subject.named_modules())


def test_the_hugging_face_loader_loads_a_t5_encoder_without_auto_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _Backend()
    backend.AutoModel = cast("_EncoderFactory", _FailingFactory())
    monkeypatch.setattr(
        "lofi.nn.tokens.pretrained.importlib.import_module", lambda _: backend
    )
    loader = HuggingFacePretrainedLoader(_fetcher(tmp_path))

    loaded = loader.load_encoder(T5, architecture=T5_ENCODER, dtype=torch.float64)

    assert loaded is backend.mt5.encoder
    assert loader.load_tokenizer(T5) is backend.tokenizer.tokenizer
    assert backend.mt5.requests == [(_snapshot(tmp_path, T5), True, torch.float64)]
    assert backend.tokenizer.requests == [(_snapshot(tmp_path, T5), True, True)]


def test_the_hugging_face_loader_loads_an_encoder_only_stack_through_auto_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _Backend()
    backend.MT5EncoderModel = cast("_EncoderFactory", _FailingFactory())
    monkeypatch.setattr(
        "lofi.nn.tokens.pretrained.importlib.import_module", lambda _: backend
    )
    loader = HuggingFacePretrainedLoader(_fetcher(tmp_path))

    loaded = loader.load_encoder(
        GENERAL, architecture=ENCODER_ONLY, dtype=torch.float16
    )

    assert loaded is backend.general.encoder
    assert backend.general.requests == [
        (_snapshot(tmp_path, GENERAL), True, torch.float16)
    ]


def test_the_local_loader_constructs_without_a_weight_file_or_network_identifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _Backend()
    monkeypatch.setattr(
        "lofi.nn.tokens.pretrained.importlib.import_module", lambda _: backend
    )
    loader = LocalPretrainedLoader(tmp_path, LOCAL)

    loaded = loader.load_encoder(LOCAL, architecture=ENCODER_ONLY, dtype=torch.float64)

    assert loaded is backend.general.encoder
    assert loader.load_tokenizer(LOCAL) is backend.tokenizer.tokenizer
    assert backend.configuration.requests == [(str(tmp_path), True)]
    assert backend.general.requests == []
    assert backend.general.configured == [torch.float64]
    assert backend.mt5.constructed == []
    assert backend.tokenizer.requests == [(str(tmp_path), True, True)]


def test_the_local_loader_constructs_a_t5_encoder_stack_in_the_asked_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _Backend()
    monkeypatch.setattr(
        "lofi.nn.tokens.pretrained.importlib.import_module", lambda _: backend
    )
    loader = LocalPretrainedLoader(tmp_path, LOCAL)

    loaded = loader.load_encoder(LOCAL, architecture=T5_ENCODER, dtype=torch.float64)

    assert loaded is backend.mt5.encoder
    assert backend.mt5.constructed == [backend.configuration.configuration]
    assert backend.mt5.requests == []
    assert backend.general.configured == []
    assert all(p.dtype is torch.float64 for p in backend.mt5.encoder.parameters())


def test_the_local_loader_refuses_a_foreign_declared_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(_: str) -> NoReturn:
        pytest.fail("a mismatched checkpoint reached the Transformers backend")

    monkeypatch.setattr("lofi.nn.tokens.pretrained.importlib.import_module", refuse)
    loader = LocalPretrainedLoader(tmp_path, LOCAL)

    with pytest.raises(ValueError, match="resources are for checkpoint"):
        loader.load_encoder(
            PinnedCheckpoint("another/model", LOCAL.revision),
            architecture=ENCODER_ONLY,
            dtype=torch.float32,
        )

    with pytest.raises(ValueError, match="resources are for checkpoint"):
        loader.load_tokenizer(PinnedCheckpoint(LOCAL.repository, "b" * 40))


def test_a_missing_transformers_install_names_the_nn_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_: str) -> NoReturn:
        message = "no transformers"
        raise ModuleNotFoundError(message)

    monkeypatch.setattr("lofi.nn.tokens.pretrained.importlib.import_module", missing)

    with pytest.raises(MissingPretrainedBackendError, match=r"install lofi\[nn\]"):
        HuggingFacePretrainedLoader(_fetcher(tmp_path)).load_encoder(
            T5, architecture=T5_ENCODER, dtype=torch.float32
        )

    with pytest.raises(MissingPretrainedBackendError, match="pretrained models"):
        HuggingFaceCausalLoader(_fetcher(tmp_path)).load_causal_model(LOCAL)


def test_loading_reads_the_stack_and_tokenizer_of_one_checkpoint_in_one_precision() -> (
    None
):
    loader = _Loader(_HiddenEncoder())
    window = ContextWindow(BUDGET, overflow=REFUSE)
    subject = load_pretrained(
        GENERAL,
        architecture=ENCODER_ONLY,
        component=TOKEN_ENCODER,
        trainable=True,
        window=ContextWindowSettings(BUDGET, "refuse"),
        readout=LastLayerReadoutSettings(),
        join=" ",
        loader=cast("PretrainedLoader", loader),
        dtype=torch.float64,
    )

    assert loader.encoder_requests == [(GENERAL, ENCODER_ONLY, torch.float64)]
    assert loader.tokenizer_requests == [GENERAL]
    assert subject.checkpoint == GENERAL
    assert subject.window == window
    assert subject.trainable
    assert all(p.dtype is torch.float64 for p in subject.parameters())


@pytest.mark.parametrize(
    ("declared", "architecture", "width"),
    [("t5-encoder", T5_ENCODER, 7), ("encoder-only", ENCODER_ONLY, 5)],
)
def test_the_plugin_builds_the_declared_stack_in_the_policys_precision(
    declared: Literal["encoder-only", "t5-encoder"],
    architecture: EncoderArchitecturePlugin,
    width: int,
) -> None:
    loader = _Loader()
    policy = DevicePolicy(torch.device("cpu"), torch.float64)
    declaration = PretrainedTokenEncoderSpec(
        checkpoint=HubSourceSpec(repository=T5.repository, revision=T5.revision),
        architecture=declared,
        trainable=True,
        window=ContextWindowSpec(rows=3, overflow="refuse"),
    )

    built = cast(
        "PretrainedTokenEncoder",
        build_component(
            declaration,
            plugin_context(
                component="tokens",
                policy=policy,
                loader=cast("PretrainedLoader", loader),
            ),
        ),
    )

    assert loader.encoder_requests == [(T5, architecture, torch.float64)]
    assert built.architecture is architecture
    assert built.output_width == width
    assert built.checkpoint == T5
    assert built.trainable
    assert built.window == ContextWindow(3, overflow=REFUSE)
    assert all(p.dtype is torch.float64 for p in built.parameters())


@pytest.mark.parametrize(
    ("declared", "overflow"),
    [("refuse", REFUSE), ("report", REPORT)],
)
def test_a_declared_window_bounds_rows_by_its_declared_overflow(
    declared: Literal["refuse", "report"], overflow: OverflowPlugin
) -> None:
    settings = ContextWindowSpec(rows=5, overflow=declared).settings()

    tokenizer = _Tokenizer()
    assert declared_window(settings, tokenizer) == ContextWindow(5, overflow=overflow)
    assert declared_window(
        ContextWindowSettings(None, declared), tokenizer
    ) == ContextWindow(tokenizer.model_max_length, overflow=overflow)


@pytest.mark.parametrize(
    ("published", "error", "detail"),
    [
        (None, TypeError, "as an integer"),
        (True, TypeError, "as an integer"),
        (3.5, TypeError, "as an integer"),
        (0, ValueError, "finite positive"),
        (-1, ValueError, "finite positive"),
        (1_000_000_000, ValueError, "finite positive"),
    ],
)
def test_an_invalid_or_sentinel_tokenizer_window_is_refused(
    published: object, error: type[Exception], detail: str
) -> None:
    tokenizer = _Tokenizer()
    tokenizer.model_max_length = cast("int", published)

    with pytest.raises(error, match=detail):
        declared_window(ContextWindowSettings(None, "report"), tokenizer)

    assert declared_window(
        ContextWindowSettings(7, "report"), tokenizer
    ) == ContextWindow(7, overflow=REPORT)


def test_truncation_keeps_whole_words_and_preserves_the_final_special_row() -> None:
    subject = _t5_encoder(window=ContextWindow(4, overflow=REPORT))

    encoded = subject.encode_words((("ab", "cd", "e"),))

    assert encoded.lengths.tolist() == [3]
    assert encoded.alignments[0].rows == ((0, 2), None, None)
    assert encoded.truncated == ((1, 2),)
    assert encoded.truncation is not None
    assert encoded.truncation[0].positions == (1, 2)
    assert encoded.truncation[0].words == ("cd", "e")
    assert len(encoded.truncation[0]) == 2


def test_a_zero_piece_word_is_refused_before_the_encoder_runs() -> None:
    stack = _Encoder()
    subject = _t5_encoder(stack, _ZeroPieceTokenizer())

    with pytest.raises(ValueError, match=r"no row for the words at \(1,\)"):
        subject.encode_words((("ab", "cd"),))

    assert stack.calls == 0


def test_a_row_carrying_the_space_before_a_word_is_aligned_to_that_word() -> None:
    subject = _t5_encoder(tokenizer=_PrefixedTokenizer())

    encoded = subject.encode_words((("ab", "cde", "f"),))

    assert encoded.alignments[0].rows == ((0, 2), (2, 5), (5, 6))


@pytest.mark.parametrize("word", ["", " ", "  "], ids=("empty", "space", "spaces"))
@pytest.mark.parametrize(
    "tokenizer",
    [_Tokenizer, _WhitespaceRowTokenizer],
    ids=("no-whitespace-row", "whitespace-row"),
)
def test_a_blank_word_is_refused_before_tokenization_by_any_tokenizer(
    tokenizer: type[_Tokenizer], word: str
) -> None:
    stack = _Encoder()
    spy = _WhitespaceRowTokenizer()
    subject = _t5_encoder(stack, tokenizer())
    probe = _t5_encoder(tokenizer=spy)

    with pytest.raises(ValueError, match="the word at 1 of item 0"):
        subject.encode_words((("ab", word, "cd"),))

    with pytest.raises(ValueError, match="the word at 1 of item 0"):
        probe.encode_words((("ab", word, "cd"),))

    assert stack.calls == 0
    assert spy.calls == 0


def test_a_row_crossing_a_word_boundary_is_refused() -> None:
    stack = _Encoder()
    subject = _t5_encoder(stack, _GluingTokenizer())

    with pytest.raises(ValueError, match="crosses the words at 0 and 1"):
        subject.encode_words((("ab", "cd"),))

    assert stack.calls == 0


def test_subword_pooling_remains_a_separate_swappable_component() -> None:
    subject = _t5()
    encoded = subject.encode_words((("ab", "c"),))
    pooler = MeanSubwordPooler(subject.output_width, component=ComponentId("pooler"))
    pooled = pooler.pool(encoded)

    assert encoded.vectors.axes == ("item", "row", "width")
    assert pooled.axes == ("item", "word", "width")
    assert pooled.tensor.shape == (1, 2, subject.output_width)
    assert all("pool" not in name for name, _ in subject.named_modules())


def test_a_refusing_window_returns_no_partial_batch() -> None:
    subject = _t5_encoder(window=ContextWindow(3, overflow=REFUSE))

    with pytest.raises(ContextOverflowError, match="needs 5 rows"):
        subject.encode_words((("ab", "cd"),))


def test_an_empty_batch_has_the_configured_width_without_loading_rows() -> None:
    encoded = _t5().encode_words(())

    assert encoded.vectors.tensor.shape == (0, 0, 7)
    assert encoded.vectors.mask is not None
    assert encoded.vectors.mask.shape == (0, 0)
    assert encoded.alignments == ()


def test_an_encoder_output_disagreeing_with_its_configuration_is_refused() -> None:
    subject = _t5_encoder(_WrongWidthEncoder(), trainable=True)

    with pytest.raises(ValueError, match="configuration declares output shape"):
        subject.encode_words((("word",),))


@pytest.mark.parametrize("width", [0, -1])
def test_a_nonpositive_checkpoint_width_is_refused(width: int) -> None:
    stack = _Encoder(d_model=width)

    with pytest.raises(ValueError, match=r"config\.d_model"):
        _t5_encoder(stack)


def test_the_causal_loader_loads_the_model_and_tokenizer_at_the_pinned_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _Backend()
    monkeypatch.setattr(
        "lofi.nn.tokens.pretrained.importlib.import_module", lambda _: backend
    )
    loader = HuggingFaceCausalLoader(_fetcher(tmp_path))

    assert loader.load_causal_model(LOCAL) is backend.causal.model
    assert loader.load_sentence_tokenizer(LOCAL) is backend.tokenizer.tokenizer
    assert loader.dtype is torch.float32
    assert backend.causal.requests == [
        (_snapshot(tmp_path, LOCAL), True, torch.float32)
    ]
    assert backend.tokenizer.requests == [(_snapshot(tmp_path, LOCAL), True, True)]


def test_a_checkpoint_the_backend_cannot_load_is_refused_by_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _Backend()
    backend.AutoModel = cast("_EncoderFactory", _AbsentFactory())
    backend.AutoModelForCausalLM = cast("_CausalFactory", _AbsentFactory())
    monkeypatch.setattr(
        "lofi.nn.tokens.pretrained.importlib.import_module", lambda _: backend
    )
    encoders = HuggingFacePretrainedLoader(_fetcher(tmp_path))
    causal = HuggingFaceCausalLoader(_fetcher(tmp_path))

    with pytest.raises(AbsentModelError, match="an-org/a-model") as encoder:
        encoders.load_encoder(LOCAL, architecture=ENCODER_ONLY, dtype=torch.float32)

    with pytest.raises(AbsentModelError, match="an-org/a-model") as model:
        causal.load_causal_model(LOCAL)

    assert encoder.value.checkpoint == LOCAL
    assert model.value.checkpoint == LOCAL


def test_the_encoder_loader_loads_a_decoder_with_its_head_at_the_pinned_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _Backend()
    monkeypatch.setattr(
        "lofi.nn.tokens.pretrained.importlib.import_module", lambda _: backend
    )
    loader = HuggingFacePretrainedLoader(_fetcher(tmp_path))

    loaded = loader.load_decoder(LOCAL, dtype=torch.bfloat16)

    assert loaded is backend.causal.model
    assert backend.causal.requests == [
        (_snapshot(tmp_path, LOCAL), True, torch.bfloat16)
    ]
    assert isinstance(loader, DecoderLoader)


def test_a_decoder_the_backend_cannot_load_is_refused_by_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _Backend()
    backend.AutoModelForCausalLM = cast("_CausalFactory", _AbsentFactory())
    monkeypatch.setattr(
        "lofi.nn.tokens.pretrained.importlib.import_module", lambda _: backend
    )

    with pytest.raises(AbsentModelError, match="an-org/a-model") as refused:
        HuggingFacePretrainedLoader(_fetcher(tmp_path)).load_decoder(
            LOCAL, dtype=torch.float32
        )

    assert refused.value.checkpoint == LOCAL


def _decoder_assets(directory: Path) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    words = ["<unk>", "</s>", "ab", "c", "d"]
    unknown, end = words[:2]
    model = tokenizers.models.WordLevel(
        {word: index for index, word in enumerate(words)}, unk_token=unknown
    )
    backend = tokenizers.Tokenizer(model)
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token=unknown, eos_token=end
    ).save_pretrained(directory)
    transformers.LlamaConfig(
        vocab_size=len(words),
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
    ).save_pretrained(directory)


def test_the_local_loader_builds_a_decoder_from_configuration_alone(
    tmp_path: Path,
) -> None:
    _decoder_assets(tmp_path)
    loader = LocalPretrainedLoader(tmp_path, LOCAL)

    decoder = loader.load_decoder(LOCAL, dtype=torch.float64)
    tokenizer = loader.load_tokenizer(LOCAL)

    assert decoder.config.hidden_size == 4
    assert all(parameter.dtype is torch.float64 for parameter in decoder.parameters())
    assert decoder.get_output_embeddings() is not None
    assert not any(
        path.suffix in {".bin", ".safetensors"} for path in tmp_path.iterdir()
    )
    assert tokenizer.pad_token_id == tokenizer.get_vocab()["</s>"]
    assert len(tokenizer) == 5


def test_the_local_loader_refuses_a_decoder_for_a_foreign_checkpoint(
    tmp_path: Path,
) -> None:
    loader = LocalPretrainedLoader(tmp_path, LOCAL)

    with pytest.raises(ValueError, match="resources are for checkpoint"):
        loader.load_decoder(
            PinnedCheckpoint("another/model", LOCAL.revision), dtype=torch.float32
        )


def test_a_tokenizer_that_pads_keeps_its_own_padding_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _Backend()
    monkeypatch.setattr(
        "lofi.nn.tokens.pretrained.importlib.import_module", lambda _: backend
    )

    tokenizer = HuggingFacePretrainedLoader(_fetcher(tmp_path)).load_tokenizer(LOCAL)

    assert tokenizer is backend.tokenizer.tokenizer
    assert tokenizer.pad_token_id == 0


def _aligner(
    tokenizer: _Tokenizer | None = None, *, rows: int = 128
) -> CheckpointAligner:
    return CheckpointAligner(
        T5,
        _Tokenizer() if tokenizer is None else tokenizer,
        join=" ",
        window=ContextWindow(rows, overflow=REPORT),
    )


def test_an_aligner_reports_every_row_of_every_word_and_every_special_row() -> None:
    aligner = _aligner()

    rows = aligner.align(("ab", "c"))

    assert aligner.checkpoint == T5
    assert aligner.identity == T5.identity
    assert rows == WordRows(((0, 1), (2,)), (3,), 4)
    assert rows.word_rows == ((0, 1), (2,))
    assert rows.special_rows == (3,)
    assert rows.live_rows == 4


def test_a_gapped_word_keeps_its_gap_where_the_encoder_keeps_an_interval() -> None:
    rows = _aligner(_GappedTokenizer()).align(("abc",))
    encoded = _t5_encoder(tokenizer=_GappedTokenizer()).encode_words((("abc",),))

    assert rows == WordRows(((0, 2),), (1, 3), 4)
    assert encoded.alignments[0].rows == ((0, 3),)


def test_an_aligner_follows_the_declared_join() -> None:
    aligner = CheckpointAligner(
        GENERAL,
        _DelimiterTokenizer(),
        join="|",
        window=ContextWindow(BUDGET, overflow=REPORT),
    )

    assert aligner.align(("ab", "c")) == WordRows(((0, 1), (2,)), (3,), 4)


def test_a_zero_piece_word_is_a_typed_refusal_for_the_aligner_and_encoder() -> None:
    with pytest.raises(ZeroPieceWordError, match=r"no row for the words at \(1,\)") as (
        refused
    ):
        _aligner(_ZeroPieceTokenizer()).align(("ab", "cd"))

    assert refused.value.positions == (1,)

    with pytest.raises(ZeroPieceWordError):
        _t5_encoder(tokenizer=_ZeroPieceTokenizer()).encode_words((("ab", "cd"),))


def test_a_piece_crossing_two_words_is_a_typed_refusal() -> None:
    with pytest.raises(CrossingPieceError, match="crosses the words at 0 and 1") as (
        refused
    ):
        _aligner(_GluingTokenizer()).align(("ab", "cd"))

    assert refused.value.words == (0, 1)
    assert refused.value.characters == (0, 5)

    with pytest.raises(CrossingPieceError):
        _t5_encoder(tokenizer=_GluingTokenizer()).encode_words((("ab", "cd"),))


def test_an_aligner_refuses_an_overflowing_item_whatever_the_overflow_policy() -> None:
    with pytest.raises(ContextOverflowError, match="needs 5 rows") as refused:
        _aligner(rows=3).align(("ab", "cd"))

    assert refused.value.words == ("ab", "cd")
    assert refused.value.rows == 3


def test_an_aligner_refuses_a_blank_word_before_tokenizing() -> None:
    spy = _WhitespaceRowTokenizer()

    with pytest.raises(ValueError, match="the word at 1 of item 0"):
        _aligner(spy).align(("ab", " "))

    assert spy.calls == 0


def _wordpiece_assets(directory: Path) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "really", "sleep", "##s"]
    padding, unknown, start, end = words[:4]
    vocabulary = {word: index for index, word in enumerate(words)}
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordPiece(vocabulary, unk_token=unknown)
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    backend.post_processor = tokenizers.processors.TemplateProcessing(
        single=f"{start} $A {end}",
        special_tokens=[(start, vocabulary[start]), (end, vocabulary[end])],
    )
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=unknown,
        pad_token=padding,
        cls_token=start,
        sep_token=end,
        model_max_length=16,
    ).save_pretrained(directory)


def test_a_loaded_aligner_reads_a_local_fast_tokenizer_at_its_declared_window(
    tmp_path: Path,
) -> None:
    _wordpiece_assets(tmp_path)
    loader = LocalPretrainedLoader(tmp_path, LOCAL)

    aligner = load_aligner(
        LOCAL, loader=loader, join=" ", window=ContextWindowSettings(None, "refuse")
    )

    assert aligner.identity == LOCAL.identity
    assert aligner.align(("really", "sleeps")) == WordRows(((1,), (2, 3)), (0, 4), 5)

    with pytest.raises(ContextOverflowError, match="needs 17 rows"):
        aligner.align(("really",) * 15)
