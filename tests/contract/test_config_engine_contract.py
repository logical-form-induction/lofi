"""Contract tests for didactic-settings composition.

The tests cover precedence, tagged-union variants, fragment selection,
provenance, and import isolation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import pytest
from didactic.settings import (
    MissingFragmentError,
    UnknownKeyError,
    compose,
    compose_traced,
    provenance_of,
)

from lofi.spec import (
    DatasetSpec,
    FeaturizerSpec,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    RunSpec,
    TermDecoderSpec,
    TermEncoderSpec,
    TrainerSpec,
)
from lofi.spec.encoders.hrr import HrrTermEncoderSpec
from lofi.spec.encoders.transformer import TransformerTermEncoderSpec

if TYPE_CHECKING:
    from didactic.settings import ConfigValue

TREE: Final = Path(__file__).resolve().parents[2]

TIMEOUT_SECONDS = 120.0


class ProbeParser(ParserSpec):
    kind: Literal["probe-parser"] = "probe-parser"
    beam: int = 1


class ProbeDecoder(TermDecoderSpec):
    kind: Literal["probe-decoder"] = "probe-decoder"
    width: int


class ProbeTransformer(TermEncoderSpec):
    kind: Literal["probe-transformer"] = "probe-transformer"
    num_heads: int = 4


class ProbeRnn(TermEncoderSpec):
    kind: Literal["probe-rnn"] = "probe-rnn"
    hidden_size: int = 32


class TreeRnnEncoderSpec(TermEncoderSpec):
    kind: Literal["rnn"] = "rnn"
    layers: int
    norm_cap: float | None = None


class LinearRnnDecoderSpec(TermDecoderSpec):
    kind: Literal["contract-linear-rnn"] = "contract-linear-rnn"
    input_width: int
    width: int


class TreeRnnDecoderSpec(TermDecoderSpec):
    kind: Literal["contract-tree-rnn"] = "contract-tree-rnn"
    input_width: int
    width: int


class LinearHrrDecoderSpec(TermDecoderSpec):
    kind: Literal["linear-hrr"] = "linear-hrr"
    width: int


class TreeHrrDecoderSpec(TermDecoderSpec):
    kind: Literal["tree-hrr"] = "tree-hrr"
    width: int


class LinearTransformerDecoderSpec(TermDecoderSpec):
    kind: Literal["linear-transformer"] = "linear-transformer"
    input_width: int
    width: int
    heads: int
    layers: int
    feedforward_width: int


class TreeTransformerDecoderSpec(TermDecoderSpec):
    kind: Literal["tree-transformer"] = "tree-transformer"
    input_width: int
    width: int
    heads: int
    layers: int
    feedforward_width: int


class LinearDiffusionDecoderSpec(TermDecoderSpec):
    kind: Literal["linear-diffusion"] = "linear-diffusion"
    input_width: int
    width: int
    layers: int
    feedforward_width: int


class TreeDiffusionDecoderSpec(TermDecoderSpec):
    kind: Literal["tree-diffusion"] = "tree-diffusion"
    input_width: int
    width: int
    layers: int
    feedforward_width: int


class ProbeLoss(LossSpec):
    kind: Literal["probe-loss"] = "probe-loss"


class ProbeDataset(DatasetSpec):
    kind: Literal["probe-dataset"] = "probe-dataset"


class ProbeFeaturizer(FeaturizerSpec):
    kind: Literal["probe-featurizer"] = "probe-featurizer"


class ProbeTrainer(TrainerSpec):
    kind: Literal["probe-trainer"] = "probe-trainer"
    epochs: int = 1
    batch_size: int = 8


class ProbeSgd(OptimizerSpec):
    kind: Literal["probe-sgd"] = "probe-sgd"
    lr: float
    momentum: float = 0.0


RUN_DOCUMENT = """\
defaults:
  - model.encoders.terms: probe-transformer
model:
  language: {from: data}
  parser:
    kind: probe-parser
    beam: 4
  decoders:
    application:
      kind: probe-decoder
      width: 16
loss:
  terms:
    - objective:
        kind: probe-loss
data:
  datasets:
    train:
      dataset:
        kind: probe-dataset
      featurizer:
        kind: probe-featurizer
trainer:
  kind: probe-trainer
  epochs: 1
  batch_size: 8
release:
  checkpoint: validation-best
parse: {retain: forest, reachability: all}
optimizer:
  kind: probe-sgd
  lr: 0.1
  momentum: 0.0
device_policy:
  device: cpu
  compute_dtype: float32
  deterministic: true
  matmul_precision: highest
  activation_memory_bytes: 8000000000
"""

FAST_PROFILE = """\
trainer:
  epochs: 2
  batch_size: 16
optimizer:
  lr: 0.2
"""

OVERLAY = """\
trainer:
  batch_size: 32
optimizer:
  lr: 0.3
"""

TRANSFORMER_FRAGMENT = """\
kind: probe-transformer
num_heads: 8
"""

RNN_FRAGMENT = """\
kind: probe-rnn
hidden_size: 64
"""

MATRIX_DOCUMENT = """\
defaults:
  - model.encoders.terms: rnn
  - model.decoders.application: linear-rnn
model:
  language: {from: data}
  parser:
    kind: probe-parser
loss:
  terms:
    - objective:
        kind: probe-loss
data:
  datasets:
    train:
      dataset:
        kind: probe-dataset
      featurizer:
        kind: probe-featurizer
trainer:
  kind: probe-trainer
release:
  checkpoint: validation-best
parse: {retain: forest, reachability: all}
optimizer:
  kind: probe-sgd
  lr: 0.1
device_policy:
  device: cpu
  compute_dtype: float32
  deterministic: true
  matmul_precision: highest
  activation_memory_bytes: 8000000000
"""

TAG_LATER_DOCUMENT = """\
defaults:
  - optimizer: momentum
model:
  language: {from: data}
  parser:
    kind: probe-parser
loss:
  terms:
    - objective:
        kind: probe-loss
data:
  datasets:
    train:
      dataset:
        kind: probe-dataset
      featurizer:
        kind: probe-featurizer
trainer:
  kind: probe-trainer
release:
  checkpoint: validation-best
parse: {retain: forest, reachability: all}
optimizer:
  kind: probe-sgd
  lr: 0.1
device_policy:
  device: cpu
  compute_dtype: float32
  deterministic: true
  matmul_precision: highest
  activation_memory_bytes: 8000000000
"""

TAG_FIRST_DOCUMENT = """\
defaults:
  - optimizer: sgd
model:
  language: {from: data}
  parser:
    kind: probe-parser
loss:
  terms:
    - objective:
        kind: probe-loss
data:
  datasets:
    train:
      dataset:
        kind: probe-dataset
      featurizer:
        kind: probe-featurizer
trainer:
  kind: probe-trainer
release:
  checkpoint: validation-best
parse: {retain: forest, reachability: all}
optimizer:
  momentum: 0.5
device_policy:
  device: cpu
  compute_dtype: float32
  deterministic: true
  matmul_precision: highest
  activation_memory_bytes: 8000000000
"""

ENCODER_FRAGMENTS: Final[dict[str, dict[str, ConfigValue]]] = {
    "rnn": {"kind": "rnn", "layers": 2, "norm_cap": 1.0},
    "hrr": {"kind": "hrr", "width": 32, "magnitude": 1.0, "floor": 0.001},
    "transformer": {
        "kind": "transformer",
        "width": 32,
        "heads": 4,
        "layers": 2,
        "feedforward_width": 64,
        "norm_cap": 1.0,
    },
}

DECODER_FRAGMENTS: Final[dict[str, dict[str, ConfigValue]]] = {
    "linear-rnn": {
        "kind": "contract-linear-rnn",
        "input_width": 32,
        "width": 32,
    },
    "tree-rnn": {"kind": "contract-tree-rnn", "input_width": 32, "width": 32},
    "linear-hrr": {"kind": "linear-hrr", "width": 32},
    "tree-hrr": {"kind": "tree-hrr", "width": 32},
    "linear-transformer": {
        "kind": "linear-transformer",
        "input_width": 32,
        "width": 32,
        "heads": 4,
        "layers": 2,
        "feedforward_width": 64,
    },
    "tree-transformer": {
        "kind": "tree-transformer",
        "input_width": 32,
        "width": 32,
        "heads": 4,
        "layers": 2,
        "feedforward_width": 64,
    },
    "linear-diffusion": {
        "kind": "linear-diffusion",
        "input_width": 32,
        "width": 32,
        "layers": 2,
        "feedforward_width": 64,
    },
    "tree-diffusion": {
        "kind": "tree-diffusion",
        "input_width": 32,
        "width": 32,
        "layers": 2,
        "feedforward_width": 64,
    },
}

EXPECTED_ENCODERS: Final[dict[str, TermEncoderSpec]] = {
    "rnn": TreeRnnEncoderSpec(layers=2, norm_cap=1.0),
    "hrr": HrrTermEncoderSpec(width=32, magnitude=1.0, floor=0.001),
    "transformer": TransformerTermEncoderSpec(
        width=32,
        heads=4,
        layers=2,
        feedforward_width=64,
        norm_cap=1.0,
    ),
}

EXPECTED_DECODERS: Final[dict[str, TermDecoderSpec]] = {
    "linear-rnn": LinearRnnDecoderSpec(input_width=32, width=32),
    "tree-rnn": TreeRnnDecoderSpec(input_width=32, width=32),
    "linear-hrr": LinearHrrDecoderSpec(width=32),
    "tree-hrr": TreeHrrDecoderSpec(width=32),
    "linear-transformer": LinearTransformerDecoderSpec(
        input_width=32,
        width=32,
        heads=4,
        layers=2,
        feedforward_width=64,
    ),
    "tree-transformer": TreeTransformerDecoderSpec(
        input_width=32,
        width=32,
        heads=4,
        layers=2,
        feedforward_width=64,
    ),
    "linear-diffusion": LinearDiffusionDecoderSpec(
        input_width=32,
        width=32,
        layers=2,
        feedforward_width=64,
    ),
    "tree-diffusion": TreeDiffusionDecoderSpec(
        input_width=32,
        width=32,
        layers=2,
        feedforward_width=64,
    ),
}

COMPOSITION_CASES: Final[tuple[tuple[str, str], ...]] = (
    ("rnn", "linear-rnn"),
    ("rnn", "tree-rnn"),
    ("rnn", "linear-hrr"),
    ("rnn", "tree-hrr"),
    ("rnn", "linear-transformer"),
    ("rnn", "tree-transformer"),
    ("rnn", "linear-diffusion"),
    ("rnn", "tree-diffusion"),
    ("hrr", "linear-rnn"),
    ("hrr", "tree-rnn"),
    ("hrr", "linear-hrr"),
    ("hrr", "tree-hrr"),
    ("hrr", "linear-transformer"),
    ("hrr", "tree-transformer"),
    ("hrr", "linear-diffusion"),
    ("hrr", "tree-diffusion"),
    ("transformer", "linear-rnn"),
    ("transformer", "tree-rnn"),
    ("transformer", "linear-hrr"),
    ("transformer", "tree-hrr"),
    ("transformer", "linear-transformer"),
    ("transformer", "tree-transformer"),
    ("transformer", "linear-diffusion"),
    ("transformer", "tree-diffusion"),
)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")

    return path


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    write(tmp_path / "run.yaml", RUN_DOCUMENT)
    write(tmp_path / "matrix.yaml", MATRIX_DOCUMENT)
    write(tmp_path / "tag-later.yaml", TAG_LATER_DOCUMENT)
    write(tmp_path / "tag-first.yaml", TAG_FIRST_DOCUMENT)
    write(tmp_path / "profiles" / "fast.yaml", FAST_PROFILE)
    write(tmp_path / "overlay.yaml", OVERLAY)
    write(
        tmp_path / "model" / "encoders" / "terms" / "probe-transformer.yaml",
        TRANSFORMER_FRAGMENT,
    )
    write(
        tmp_path / "model" / "encoders" / "terms" / "probe-rnn.yaml",
        RNN_FRAGMENT,
    )
    write(tmp_path / "optimizer" / "momentum.json", json.dumps({"momentum": 0.5}))
    write(
        tmp_path / "optimizer" / "sgd.json",
        json.dumps({"kind": "probe-sgd", "lr": 0.1}),
    )

    for name, fragment in ENCODER_FRAGMENTS.items():
        write(
            tmp_path / "model" / "encoders" / "terms" / f"{name}.json",
            json.dumps(fragment),
        )

    for name, fragment in DECODER_FRAGMENTS.items():
        write(
            tmp_path / "model" / "decoders" / "application" / f"{name}.json",
            json.dumps(fragment),
        )

    return tmp_path


def test_layers_compose_a_run_spec_in_precedence_order(config_dir: Path) -> None:
    spec = compose(
        config_dir / "run.yaml",
        schema=RunSpec,
        profile="fast",
        overlays=[config_dir / "overlay.yaml"],
        overrides=["optimizer.lr=0.4", "trainer.epochs=3"],
    )

    assert spec.model.parser == ProbeParser(beam=4)
    assert spec.trainer == ProbeTrainer(epochs=3, batch_size=32)
    assert spec.optimizer == ProbeSgd(lr=0.4, momentum=0.0)
    assert spec.model.decoders == {"application": ProbeDecoder(width=16)}


def test_each_layer_wins_over_the_ones_below_it(config_dir: Path) -> None:
    from_file = compose(config_dir / "run.yaml", schema=RunSpec)
    with_profile = compose(config_dir / "run.yaml", schema=RunSpec, profile="fast")
    with_overlay = compose(
        config_dir / "run.yaml",
        schema=RunSpec,
        profile="fast",
        overlays=[config_dir / "overlay.yaml"],
    )

    assert from_file.trainer == ProbeTrainer(epochs=1, batch_size=8)
    assert from_file.optimizer == ProbeSgd(lr=0.1)
    assert with_profile.trainer == ProbeTrainer(epochs=2, batch_size=16)
    assert with_profile.optimizer == ProbeSgd(lr=0.2)
    assert with_overlay.trainer == ProbeTrainer(epochs=2, batch_size=32)
    assert with_overlay.optimizer == ProbeSgd(lr=0.3)


def test_a_variant_field_inside_a_tagged_union_root_is_read_from_yaml(
    config_dir: Path,
) -> None:
    spec = compose(config_dir / "run.yaml", schema=RunSpec)

    assert spec.model.encoders == {"terms": ProbeTransformer(num_heads=8)}


def test_a_variant_field_inside_a_tagged_union_root_takes_an_override(
    config_dir: Path,
) -> None:
    spec = compose(
        config_dir / "run.yaml",
        schema=RunSpec,
        overrides=["model.encoders.terms.num_heads=2", "optimizer.momentum=0.9"],
    )

    assert spec.model.encoders == {"terms": ProbeTransformer(num_heads=2)}
    assert spec.optimizer == ProbeSgd(lr=0.1, momentum=0.9)


def test_a_group_field_set_before_its_discriminator_composes_the_same(
    config_dir: Path,
) -> None:
    tag_last = compose(config_dir / "tag-later.yaml", schema=RunSpec)
    tag_first = compose(config_dir / "tag-first.yaml", schema=RunSpec)

    assert tag_last.optimizer == ProbeSgd(lr=0.1, momentum=0.5)
    assert tag_first == tag_last


def test_an_unknown_key_inside_a_selected_variant_names_the_variant(
    config_dir: Path,
) -> None:
    with pytest.raises(UnknownKeyError) as caught:
        compose(
            config_dir / "run.yaml",
            schema=RunSpec,
            overrides=["model.encoders.terms.hidden_size=9"],
        )

    error = caught.value

    assert error.path == "model.encoders.terms.hidden_size"
    assert error.allowed == ("kind", "num_heads")
    assert error.declared_by == ("probe-rnn",)
    assert "probe-transformer" in str(error)
    assert "probe-rnn" in str(error)


def test_all_twenty_four_encoder_decoder_fragments_compose(
    config_dir: Path,
) -> None:
    assert len(COMPOSITION_CASES) == 24
    assert len(set(COMPOSITION_CASES)) == 24
    assert all(
        sum(encoder == expected for encoder, _ in COMPOSITION_CASES) == 8
        for expected in ENCODER_FRAGMENTS
    )
    assert all(
        sum(decoder == expected for _, decoder in COMPOSITION_CASES) == 3
        for expected in DECODER_FRAGMENTS
    )

    composed: list[RunSpec] = []

    for encoder, decoder in COMPOSITION_CASES:
        spec = compose(
            config_dir / "matrix.yaml",
            schema=RunSpec,
            groups={
                "model.encoders.terms": encoder,
                "model.decoders.application": decoder,
            },
        )

        assert spec.model.encoders == {"terms": EXPECTED_ENCODERS[encoder]}
        assert spec.model.decoders == {"application": EXPECTED_DECODERS[decoder]}
        composed.append(spec)

    assert len(composed) == 24


def test_a_group_fragment_is_selected_from_the_defaults_list(
    config_dir: Path,
) -> None:
    spec = compose(config_dir / "run.yaml", schema=RunSpec)

    assert spec.model.encoders == {"terms": ProbeTransformer(num_heads=8)}


def test_a_group_selection_by_name_replaces_the_defaults_entry(
    config_dir: Path,
) -> None:
    by_mapping = compose(
        config_dir / "run.yaml",
        schema=RunSpec,
        groups={"model.encoders.terms": "probe-rnn"},
    )
    by_override = compose(
        config_dir / "run.yaml",
        schema=RunSpec,
        overrides=["model/encoders/terms=probe-rnn"],
    )

    assert by_mapping.model.encoders == {"terms": ProbeRnn(hidden_size=64)}
    assert by_override.model.encoders == {"terms": ProbeRnn(hidden_size=64)}


def test_a_missing_fragment_lists_the_names_available(config_dir: Path) -> None:
    with pytest.raises(MissingFragmentError) as caught:
        compose(
            config_dir / "run.yaml",
            schema=RunSpec,
            groups={"model.encoders.terms": "cnn"},
        )

    error = caught.value

    assert error.group == "model.encoders.terms"
    assert error.name == "cnn"
    assert error.available == (
        "hrr",
        "probe-rnn",
        "probe-transformer",
        "rnn",
        "transformer",
    )
    assert all(
        path.parent == config_dir / "model" / "encoders" / "terms"
        for path in error.tried
    )


def test_provenance_names_the_layer_at_every_depth(config_dir: Path) -> None:
    composed = compose_traced(
        config_dir / "run.yaml",
        schema=RunSpec,
        profile="fast",
        overlays=[config_dir / "overlay.yaml"],
        overrides=["optimizer.lr=0.4"],
    )
    provenance = composed.provenance

    assert provenance.source_of("trainer.epochs").label == "profile:fast"
    assert provenance.source_of("trainer.batch_size").label == "overlay:overlay.yaml"
    assert provenance.source_of("optimizer.lr").label == "override:optimizer.lr=0.4"
    assert provenance.source_of("optimizer.momentum").label == "file:run.yaml"
    assert provenance.source_of("model.parser.beam").label == "file:run.yaml"
    assert (
        provenance.source_of("model.encoders.terms.num_heads").label
        == "group:model.encoders.terms=probe-transformer"
    )
    assert (
        provenance.source_of("model.decoders.application.width").label
        == "file:run.yaml"
    )
    assert provenance.source_of("scheduler").label == "default"
    assert provenance_of(composed.value) == provenance


def test_provenance_covers_every_leaf_of_the_composed_model(
    config_dir: Path,
) -> None:
    composed = compose_traced(config_dir / "run.yaml", schema=RunSpec)

    by_layer = composed.provenance.by_layer()

    assert set(by_layer) == {
        "default",
        "file:run.yaml",
        "group:model.encoders.terms=probe-transformer",
    }
    assert (
        "model.encoders.terms.kind"
        in by_layer["group:model.encoders.terms=probe-transformer"]
    )
    assert "trainer.kind" in by_layer["file:run.yaml"]
    assert composed.provenance.under("optimizer").paths == (
        "optimizer.declared_output",
        "optimizer.kind",
        "optimizer.lr",
        "optimizer.momentum",
    )


COMPOSE_SCRIPT = """\
import os
import sys
from pathlib import Path
from typing import Literal

from didactic.settings import compose

from lofi.spec import (
    DatasetSpec,
    FeaturizerSpec,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    RunSpec,
    TrainerSpec,
)


class ProbeParser(ParserSpec):
    kind: Literal["probe-parser"] = "probe-parser"


class ProbeLoss(LossSpec):
    kind: Literal["probe-loss"] = "probe-loss"


class ProbeDataset(DatasetSpec):
    kind: Literal["probe-dataset"] = "probe-dataset"


class ProbeFeaturizer(FeaturizerSpec):
    kind: Literal["probe-featurizer"] = "probe-featurizer"


class ProbeTrainer(TrainerSpec):
    kind: Literal["probe-trainer"] = "probe-trainer"


class ProbeSgd(OptimizerSpec):
    kind: Literal["probe-sgd"] = "probe-sgd"
    lr: float


spec = compose(
    Path(os.environ["LOFI_PROBE_DOCUMENT"]),
    schema=RunSpec,
    overrides=["optimizer.lr=0.5"],
)
print(spec.optimizer.lr)
print(sorted(name for name in ("torch", "panproto") if name in sys.modules))
"""

MINIMAL_DOCUMENT = """\
model:
  language: {from: data}
  parser:
    kind: probe-parser
loss:
  terms:
    - objective:
        kind: probe-loss
data:
  datasets:
    train:
      dataset:
        kind: probe-dataset
      featurizer:
        kind: probe-featurizer
trainer:
  kind: probe-trainer
release:
  checkpoint: validation-best
parse: {retain: forest, reachability: all}
optimizer:
  kind: probe-sgd
  lr: 0.1
device_policy:
  device: cpu
  compute_dtype: float32
  deterministic: true
  matmul_precision: highest
  activation_memory_bytes: 8000000000
"""


def test_composing_a_run_spec_imports_neither_torch_nor_panproto(
    tmp_path: Path,
) -> None:
    document = write(tmp_path / "run.yaml", MINIMAL_DOCUMENT)

    result = subprocess.run(
        [sys.executable, "-"],
        input=COMPOSE_SCRIPT,
        env={**os.environ, "LOFI_PROBE_DOCUMENT": str(document)},
        capture_output=True,
        encoding="utf-8",
        check=False,
        cwd=TREE,
        timeout=TIMEOUT_SECONDS,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["0.5", "[]"]
