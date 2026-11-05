from typing import TYPE_CHECKING, Final, Literal

import pytest
from didactic.settings import MissingFragmentError, compose

from lofi.spec import (
    CONFIG_DIRS_GROUP,
    ConfigDirectoryError,
    DatasetSpec,
    FeaturizerSpec,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    RunSpec,
    TermEncoderSpec,
    TrainerSpec,
    _groups,
    config_search_path,
    plugin_config_dirs,
)

if TYPE_CHECKING:
    from pathlib import Path


class GroupParser(ParserSpec):
    kind: Literal["group-parser"] = "group-parser"
    beam: int = 1


class GroupEncoder(TermEncoderSpec):
    kind: Literal["group-encoder"] = "group-encoder"
    width: int = 8


class GroupLoss(LossSpec):
    kind: Literal["group-loss"] = "group-loss"


class GroupDataset(DatasetSpec):
    kind: Literal["group-dataset"] = "group-dataset"


class GroupFeaturizer(FeaturizerSpec):
    kind: Literal["group-featurizer"] = "group-featurizer"


class GroupTrainer(TrainerSpec):
    kind: Literal["group-trainer"] = "group-trainer"
    epochs: int = 1


class GroupSgd(OptimizerSpec):
    kind: Literal["group-sgd"] = "group-sgd"
    lr: float = 0.1


RUN_DOCUMENT: Final = """\
defaults:
  - model.parser: wide
  - model.encoders.terms: packaged
  - trainer: long
  - optimizer: plugin
model:
  language: {from: data}
loss:
  terms:
    - objective:
        kind: group-loss
data:
  datasets:
    train:
      dataset:
        kind: group-dataset
      featurizer:
        kind: group-featurizer
parse:
  retain: forest
  reachability: all
release:
  checkpoint: resume-latest
device_policy:
  device: cpu
  compute_dtype: float32
  deterministic: true
  matmul_precision: highest
  activation_memory_bytes: 8000000000
"""


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")

    return path


def install_plugin(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, entry: str, value: str
) -> None:
    """Install a distribution whose config_dirs entry point names ``value``."""
    distribution = f"lofi-config-plugin-{entry}-{root.name}".replace("_", "-")
    metadata = root / f"{distribution.replace('-', '_')}-0.1.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 0.1\n",
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "entry_points.txt").write_text(
        f"[{CONFIG_DIRS_GROUP}]\n{entry} = {value}\n", encoding="utf-8", newline="\n"
    )
    monkeypatch.syspath_prepend(str(root))


def plugin_package(root: Path) -> str:
    return f"lofi_config_plugin_{root.name.replace('-', '_')}"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    write(project / "configs" / "runs" / "run.yaml", RUN_DOCUMENT)
    write(
        project / "configs" / "model" / "parser" / "wide.yaml",
        "kind: group-parser\nbeam: 16\n",
    )
    write(
        project / "configs" / "trainer" / "long.yaml",
        "kind: group-trainer\nepochs: 50\n",
    )

    return project


@pytest.fixture
def packaged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    packaged = tmp_path / "packaged"
    write(
        packaged / "model" / "encoders" / "terms" / "packaged.yaml",
        "kind: group-encoder\nwidth: 32\n",
    )
    write(packaged / "trainer" / "long.yaml", "kind: group-trainer\nepochs: 2\n")
    monkeypatch.setattr(_groups, "PACKAGED_CONFIG_ROOT", packaged)

    return packaged


@pytest.fixture
def plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    site = tmp_path / "site"
    package = site / plugin_package(tmp_path)
    write(package / "optimizer" / "plugin.yaml", "kind: group-sgd\nlr: 0.25\n")
    install_plugin(site, monkeypatch, entry="fixture", value=package.name)

    return package


def test_the_search_path_lists_project_packaged_then_plugin_roots(
    project: Path, packaged: Path, plugin: Path
) -> None:
    assert config_search_path(project) == (project / "configs", packaged, plugin)


@pytest.mark.usefixtures("packaged", "plugin")
def test_a_run_composes_from_four_group_fragments_across_the_roots(
    project: Path,
) -> None:
    spec = compose(
        project / "configs" / "runs" / "run.yaml",
        schema=RunSpec,
        search_path=config_search_path(project),
    )

    assert spec.model.parser == GroupParser(beam=16)
    assert spec.model.encoders["terms"] == GroupEncoder(width=32)
    assert spec.trainer == GroupTrainer(epochs=50)
    assert spec.optimizer == GroupSgd(lr=0.25)


@pytest.mark.usefixtures("packaged", "plugin")
def test_the_project_shadows_a_packaged_option_of_the_same_name(
    project: Path,
) -> None:
    spec = compose(
        project / "configs" / "runs" / "run.yaml",
        schema=RunSpec,
        search_path=config_search_path(project),
    )
    (project / "configs" / "trainer" / "long.yaml").unlink()
    fallen_back = compose(
        project / "configs" / "runs" / "run.yaml",
        schema=RunSpec,
        search_path=config_search_path(project),
    )

    assert spec.trainer == GroupTrainer(epochs=50)
    assert fallen_back.trainer == GroupTrainer(epochs=2)


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("long.yml", "kind: group-trainer\nepochs: 7\n"),
        ("long.toml", 'kind = "group-trainer"\nepochs = 7\n'),
        ("long.json", '{"kind": "group-trainer", "epochs": 7}'),
    ],
)
@pytest.mark.usefixtures("packaged", "plugin")
def test_an_option_resolves_under_each_declared_suffix(
    project: Path, name: str, text: str
) -> None:
    (project / "configs" / "trainer" / "long.yaml").unlink()
    write(project / "configs" / "trainer" / name, text)

    spec = compose(
        project / "configs" / "runs" / "run.yaml",
        schema=RunSpec,
        search_path=config_search_path(project),
    )

    assert spec.trainer == GroupTrainer(epochs=7)


@pytest.mark.usefixtures("plugin")
def test_an_unresolvable_option_names_every_root_tried_in_order(
    project: Path, packaged: Path
) -> None:
    with pytest.raises(MissingFragmentError) as caught:
        compose(
            project / "configs" / "runs" / "run.yaml",
            schema=RunSpec,
            search_path=config_search_path(project),
            groups={"trainer": "short"},
        )

    error = caught.value
    directories = tuple(dict.fromkeys(path.parent for path in error.tried))

    assert error.group == "trainer"
    assert error.name == "short"
    assert directories == (project / "configs" / "trainer", packaged / "trainer")
    assert [path.suffix for path in error.tried][:4] == [
        ".yaml",
        ".yml",
        ".toml",
        ".json",
    ]
    assert error.available == ("long",)


def test_an_unknown_group_names_every_root_in_search_order(
    project: Path, packaged: Path, plugin: Path
) -> None:
    with pytest.raises(MissingFragmentError) as caught:
        compose(
            project / "configs" / "runs" / "run.yaml",
            schema=RunSpec,
            search_path=config_search_path(project),
            groups={"scheduler": "cosine"},
        )

    assert caught.value.tried == (
        project / "configs" / "runs" / "scheduler",
        project / "configs" / "scheduler",
        packaged / "scheduler",
        plugin / "scheduler",
    )


def test_plugin_directories_follow_entry_point_name_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    later = tmp_path / "later"
    earlier = tmp_path / "earlier"
    later_package = later / f"a_{plugin_package(later)}"
    earlier_package = earlier / f"z_{plugin_package(earlier)}"
    write(later_package / "trainer" / "a.yaml", "")
    write(earlier_package / "trainer" / "b.yaml", "")
    install_plugin(earlier, monkeypatch, entry="alpha", value=earlier_package.name)
    install_plugin(later, monkeypatch, entry="zeta", value=later_package.name)

    assert plugin_config_dirs() == (earlier_package, later_package)


def test_a_package_spread_over_directories_contributes_each_of_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = plugin_package(tmp_path)
    first = tmp_path / "first" / name
    second = tmp_path / "second" / name
    write(first / "trainer" / "a.yaml", "")
    write(second / "optimizer" / "b.yaml", "")
    monkeypatch.syspath_prepend(str(tmp_path / "second"))
    install_plugin(tmp_path / "first", monkeypatch, entry="fixture", value=name)

    assert plugin_config_dirs() == (first, second)


def test_a_plugin_package_without_an_init_module_contributes_its_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / plugin_package(tmp_path)
    write(package / "trainer" / "a.yaml", "")
    install_plugin(tmp_path, monkeypatch, entry="fixture", value=package.name)

    assert (package / "__init__.py").exists() is False
    assert plugin_config_dirs() == (package,)


def test_a_plugin_package_with_an_init_module_contributes_its_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / plugin_package(tmp_path)
    write(package / "__init__.py", "")
    write(package / "trainer" / "a.yaml", "")
    install_plugin(tmp_path, monkeypatch, entry="fixture", value=package.name)

    assert plugin_config_dirs() == (package,)


def test_an_entry_point_naming_an_attribute_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / plugin_package(tmp_path)
    write(package / "__init__.py", "ROOT = 'x'\n")
    install_plugin(tmp_path, monkeypatch, entry="fixture", value=f"{package.name}:ROOT")

    with pytest.raises(ConfigDirectoryError) as caught:
        plugin_config_dirs()

    assert caught.value.entry == "fixture"
    assert caught.value.value == f"{package.name}:ROOT"
    assert "attribute" in str(caught.value)


def test_an_entry_point_naming_a_module_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = plugin_package(tmp_path)
    write(tmp_path / f"{module}.py", "")
    install_plugin(tmp_path, monkeypatch, entry="fixture", value=module)

    with pytest.raises(ConfigDirectoryError) as caught:
        plugin_config_dirs()

    assert "module" in str(caught.value)


def test_an_entry_point_naming_nothing_installed_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = plugin_package(tmp_path)
    install_plugin(tmp_path, monkeypatch, entry="fixture", value=missing)

    with pytest.raises(ConfigDirectoryError) as caught:
        plugin_config_dirs()

    assert caught.value.value == missing
    assert "not an installed package" in str(caught.value)


def test_an_entry_point_below_an_unimportable_parent_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = f"{plugin_package(tmp_path)}.configs"
    install_plugin(tmp_path, monkeypatch, entry="fixture", value=missing)

    with pytest.raises(ConfigDirectoryError) as caught:
        plugin_config_dirs()

    assert "cannot be imported" in str(caught.value)
