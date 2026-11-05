import importlib
import logging
import multiprocessing
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Final

from lofi.data.corpora import layers
from lofi.data.corpora.layers import _codec, _translate, reader

SAMPLE: Final = (
    Path(__file__).resolve().parents[4] / "data" / "layers_sample" / "repository"
)


def _import_data_without_effects(directory: str) -> None:
    os.chdir(directory)
    root = logging.getLogger()
    logging_state = (
        tuple(root.handlers),
        tuple(root.filters),
        root.level,
        root.disabled,
    )
    stdout = StringIO()
    stderr = StringIO()

    for name in tuple(sys.modules):
        if name in {"lofi.data", "rich"} or name.startswith(("lofi.data.", "rich.")):
            del sys.modules[name]

    with redirect_stdout(stdout), redirect_stderr(stderr):
        importlib.import_module("lofi.data")

    if stdout.getvalue() or stderr.getvalue():
        msg = "importing lofi.data wrote output"
        raise RuntimeError(msg)

    if any(name == "rich" or name.startswith("rich.") for name in sys.modules):
        msg = "importing lofi.data loaded a UI package"
        raise RuntimeError(msg)

    if (
        tuple(root.handlers),
        tuple(root.filters),
        root.level,
        root.disabled,
    ) != logging_state:
        msg = "importing lofi.data configured logging"
        raise RuntimeError(msg)


def test_the_package_re_exports_its_codec_reader_and_translation_outcomes() -> None:
    translated = (
        "LayersSkip",
        "RecordSkippedError",
        "RelationPair",
        "relation_layer_pair",
    )
    expected = [*_codec.__all__, *reader.__all__, *translated]

    assert sorted(layers.__all__) == sorted(expected)

    for name in _codec.__all__:
        assert getattr(layers, name) is getattr(_codec, name)

    for name in reader.__all__:
        assert getattr(layers, name) is getattr(reader, name)

    for name in translated:
        assert getattr(layers, name) is getattr(_translate, name)


def test_the_re_exported_names_read_a_record_through_the_package() -> None:
    codec = layers.LayersExportCodec()
    decoded = codec.decode('{"uri": "at://a/pub.layers.expression/x", "value": {}}')

    assert codec.name == layers.NAME
    assert len(decoded.records) == 1
    assert decoded.records[0].nsid == "pub.layers.expression"
    assert codec.decode(codec.encode(decoded.records)) == decoded


def test_the_re_exported_reader_reads_the_committed_sample() -> None:
    read = layers.reader_at(SAMPLE)

    assert read.name == layers.READER_NAME
    assert read.splits() == ("dev", "test", "train")
    assert read.report().emitted == sum(
        len(read.items(split)) for split in read.splits()
    )


def test_importing_the_data_package_has_no_ui_or_output_side_effect(
    tmp_path: Path,
) -> None:
    process = multiprocessing.get_context("spawn").Process(
        target=_import_data_without_effects,
        args=(str(tmp_path),),
    )
    process.start()
    process.join()

    assert process.exitcode == 0
    assert tuple(tmp_path.iterdir()) == ()
