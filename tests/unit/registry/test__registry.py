import sys
from importlib.metadata import EntryPoint
from typing import TYPE_CHECKING, Final, Literal

import didactic.api as dx
import pytest
from tests.support.installed import installed

from lofi.registry import (
    ComponentPlugin,
    Registry,
    RegistryError,
    TagCollisionError,
    UnknownTagError,
    entry_point_group,
    resolve_plugin,
)
from lofi.spec.controllers import MlpControllerSpec
from lofi.spec.span_decoders.rnn import RnnSpanDecoderSpec

if TYPE_CHECKING:
    from pathlib import Path

type Component = type[dx.Model]

FAMILY: Final = "registry_fixture"
TAG: Final = "fixture"


def _write_plugin_distribution(
    root: Path,
    *,
    module: str,
    distribution: str,
) -> None:
    source = """from typing import Literal
import didactic.api as dx

class FixtureRoot(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    pass

class FixtureSpec(FixtureRoot):
    kind: Literal["fixture"] = "fixture"
"""
    (root / f"{module}.py").write_text(source, encoding="utf-8", newline="\n")

    normalized = distribution.replace("-", "_")
    metadata = root / f"{normalized}-1.2.3.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 1.2.3\n",
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "entry_points.txt").write_text(
        f"[lofi.{FAMILY}]\n{TAG} = {module}:FixtureSpec\n",
        encoding="utf-8",
        newline="\n",
    )


def _install_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    suffix = tmp_path.name.replace("-", "_")
    module = f"lofi_registry_plugin_{suffix}"
    _write_plugin_distribution(
        tmp_path, module=module, distribution=f"lofi-registry-plugin-{suffix}"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    return module


def test_an_entry_point_group_is_derived_from_its_family() -> None:
    assert entry_point_group("parser") == "lofi.parser"
    assert Registry[Component]("parser").group == "lofi.parser"

    with pytest.raises(ValueError, match="family"):
        entry_point_group("")


def test_an_installed_plugin_is_imported_only_when_its_tag_is_resolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _install_plugin(tmp_path, monkeypatch)
    registry = Registry[Component](FAMILY)

    assert registry.available() == (TAG,)
    assert module not in sys.modules

    component = registry.get(TAG)

    assert module in sys.modules
    assert component.__name__ == "FixtureSpec"
    assert component().model_dump() == {"kind": TAG}


def test_an_unknown_tag_is_refused_naming_what_is_available() -> None:
    registry = installed(FAMILY, {"chart": 1, "vector": 2})

    with pytest.raises(UnknownTagError) as caught:
        registry.get("missing")

    assert caught.value.tag == "missing"
    assert caught.value.available == ("chart", "vector")
    assert "chart, vector" in str(caught.value)


def test_a_refused_resolution_is_a_value_error_of_the_configuration() -> None:
    with pytest.raises(ValueError, match="no registry_fixture is registered"):
        installed(FAMILY, {}).get("missing")


def test_two_installed_distributions_claiming_one_tag_are_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suffix = tmp_path.name.replace("-", "_")
    first = "lofi-collision-alpha"
    second = "lofi-collision-beta"
    _write_plugin_distribution(
        tmp_path, module=f"lofi_collision_alpha_{suffix}", distribution=first
    )
    _write_plugin_distribution(
        tmp_path, module=f"lofi_collision_beta_{suffix}", distribution=second
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = Registry[Component](FAMILY)

    with pytest.raises(TagCollisionError) as caught:
        registry.get(TAG)

    assert caught.value.distributions == (first, second)

    with pytest.raises(TagCollisionError):
        registry.members()


def test_members_pair_every_tag_with_its_component_in_tag_order() -> None:
    registry = installed(FAMILY, {"zeta": "z", "alpha": "a"})

    assert registry.members() == (("alpha", "a"), ("zeta", "z"))
    assert tuple(point.name for point in registry.entries()) == ("alpha", "zeta")


def test_one_inventory_is_read_per_source_and_import_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = [0]

    def source() -> tuple[EntryPoint, ...]:
        calls[0] += 1
        return (
            EntryPoint(name=f"fixture-{calls[0]}", value="m:a", group="lofi.first"),
            EntryPoint(name="other", value="m:b", group="lofi.second"),
        )

    assert Registry[object]("first", source=source).available() == ("fixture-1",)
    assert Registry[object]("second", source=source).available() == ("other",)
    assert calls == [1]

    monkeypatch.syspath_prepend("/a/new/plugin/location")

    assert Registry[object]("first", source=source).available() == ("fixture-2",)
    assert calls == [2]


def _build(settings: object, context: object, /) -> object:
    del settings
    return context


def test_resolve_plugin_refuses_a_foreign_member_or_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = ComponentPlugin(spec=MlpControllerSpec, build=_build)
    members: dict[str, object] = {"mlp": plugin, "loose": _build}
    monkeypatch.setattr(
        Registry, "get", lambda _registry, tag: members[tag], raising=True
    )

    assert (
        resolve_plugin("controller", "mlp", ComponentPlugin, MlpControllerSpec)
        is plugin
    )

    with pytest.raises(RegistryError, match="not a ComponentPlugin"):
        resolve_plugin("controller", "loose", ComponentPlugin, MlpControllerSpec)

    with pytest.raises(RegistryError, match="builds MlpControllerSpec"):
        resolve_plugin("controller", "mlp", ComponentPlugin, RnnSpanDecoderSpec)


class _Root(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    pass


class _Variant(_Root):
    kind: Literal["variant"] = "variant"


def test_the_registry_names_its_family_and_tags() -> None:
    assert repr(installed(FAMILY, {"variant": _Variant})) == (
        "Registry(family='registry_fixture', available=('variant',))"
    )
