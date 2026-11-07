"""Composition tests for installed type-sampler choices."""

import json
import subprocess
import sys
from typing import TYPE_CHECKING, cast

from didactic.settings import load_document

from lofi.data import InventoriedTypeSampler, TypeDistribution
from lofi.data.synthetic import TypeSamplerContext
from lofi.registry import Registry, plugin_for
from lofi.run import packaged_options, section_root
from lofi.spec import TypeSamplerSpec
from lofi.spec._plugins import load_plugins
from lofi.terms import Operator, Signature, TermStore, TreeEditDistance, ball

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.spec import ConfiguredSpec


SIGNATURE = Signature(
    "configured-type-samplers",
    ("Cat",),
    (
        Operator("n", "Cat"),
        Operator("s", "Cat"),
        Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
    ),
)


def _built(spec: TypeSamplerSpec, store: TermStore) -> object:
    declaration = cast("ConfiguredSpec[object]", spec)

    return plugin_for(declaration).construct(
        declaration, TypeSamplerContext(store, "Cat")
    )


def test_packaged_type_samplers_validate_discover_construct_and_run() -> None:
    loaded = load_plugins()
    distribution = TypeDistribution(2, {}, max_depth=2)

    assert {"exhaustive", "recursive"} <= set(loaded)
    assert packaged_options("type_sampler") == ("exhaustive", "recursive")

    for name in packaged_options("type_sampler"):
        document = load_document(section_root("type_sampler") / f"{name}.yaml")
        spec = TypeSamplerSpec.model_validate_json(json.dumps(document))
        store = TermStore(SIGNATURE)
        sampler = cast("InventoriedTypeSampler", _built(spec, store))
        terms, report = sampler.sample(20, distribution, seed=73)
        center = sampler.inventory[0]
        nearby = ball(
            center,
            sampler.inventory,
            TreeEditDistance(),
            radius=1,
        )

        assert isinstance(sampler, InventoriedTypeSampler)
        assert sampler.inventory.store is store
        assert sampler.inventory.signature is store.signature
        assert len(terms) == report.draws == 20
        assert all(store.contains(term) for term in terms)
        assert nearby
        assert nearby[0] is center

    registry = Registry[object]("type_sampler")

    assert registry.available() == ("exhaustive", "recursive")


def test_type_samplers_discover_and_construct_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    probe = """
import json
from didactic.settings import load_document
from lofi.registry import Registry, plugin_for
from lofi.run import packaged_options, section_root
from lofi.data.synthetic import TypeSamplerContext
from lofi.spec import TypeSamplerSpec
from lofi.spec._plugins import load_plugins
from lofi.terms import Operator, Signature, TermStore

signature = Signature(
    "fresh-type-samplers",
    ("Cat",),
    (
        Operator("n", "Cat"),
        Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
    ),
)
loaded = load_plugins()
registry = Registry("type_sampler")
built = {}
for name in packaged_options("type_sampler"):
    document = load_document(section_root("type_sampler") / f"{name}.yaml")
    spec = TypeSamplerSpec.model_validate_json(json.dumps(document))
    store = TermStore(signature)
    sampler = plugin_for(spec).construct(spec, TypeSamplerContext(store, "Cat"))
    built[name] = {
        "candidates": len(sampler.inventory),
        "store_identity": sampler.inventory.store is store,
        "signature_identity": sampler.inventory.signature is signature,
    }
print(json.dumps({"loaded": loaded, "available": registry.available(), "built": built}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        encoding="utf-8",
        cwd=tmp_path,
    )
    observed = cast("dict[str, JsonValue]", json.loads(completed.stdout))

    assert observed["available"] == ["exhaustive", "recursive"]
    loaded = cast("list[str]", observed["loaded"])
    assert {"exhaustive", "recursive"} <= set(loaded)
    built = cast("dict[str, dict[str, JsonValue]]", observed["built"])
    assert set(built) == {"exhaustive", "recursive"}
    assert all(cast("int", result["candidates"]) > 0 for result in built.values())
    assert all(result["store_identity"] is True for result in built.values())
    assert all(result["signature_identity"] is True for result in built.values())
