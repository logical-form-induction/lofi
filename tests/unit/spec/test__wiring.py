from types import MappingProxyType
from typing import TYPE_CHECKING, Final, override

import pytest

from lofi._types import ModuleSharing, SubmoduleDeclaration
from lofi.spec import (
    Dependency,
    DependencyCycleError,
    LinearizationDisagreementError,
    UnresolvedDependencyError,
    WiringError,
)
from lofi.spec._wiring import (
    check_codec_identity,
    check_realized_sharing,
    resolve_dependencies,
    resolve_sharing,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


class Component:
    __slots__ = ("_dependencies",)

    def __init__(self, *dependencies: Dependency) -> None:
        self._dependencies = dependencies

    def dependencies(self) -> tuple[Dependency, ...]:
        return self._dependencies


class StatefulComponent(Component):
    __slots__ = ("_later", "calls")

    def __init__(
        self, dependencies: tuple[Dependency, ...], later: tuple[Dependency, ...]
    ) -> None:
        super().__init__(*dependencies)
        self._later = later
        self.calls = 0

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        self.calls += 1

        return super().dependencies() if self.calls == 1 else self._later


ENCODER = Component()
PARSER = Component(Dependency("encoder", "features"))


def test_a_dependency_resolves_to_the_role_that_produces_it() -> None:
    plan = resolve_dependencies({"parser": PARSER, "encoder": ENCODER})

    assert plan.order == ("encoder", "parser")
    assert plan.producers == {("parser", "features"): "encoder"}
    assert plan.declarations == {
        "encoder": (),
        "parser": (Dependency("encoder", "features"),),
    }


def test_resolution_uses_lexical_dependency_passes_for_topological_order() -> None:
    graph = {
        "a": Component(Dependency("b", "features")),
        "b": Component(),
        "c": Component(),
    }

    assert resolve_dependencies(graph).order == ("b", "c", "a")


def test_resolution_reads_each_components_dependencies_once() -> None:
    consumer = StatefulComponent(
        (Dependency("producer", "features"),), (Dependency("missing", "changed"),)
    )
    plan = resolve_dependencies({"consumer": consumer, "producer": Component()})

    assert plan.order == ("producer", "consumer")
    assert plan.producers == {("consumer", "features"): "producer"}
    assert consumer.calls == 1


def test_a_resolved_plan_does_not_follow_later_changes_to_the_mapping() -> None:
    graph: dict[str, Component] = {"encoder": ENCODER}
    plan = resolve_dependencies(graph)
    graph["parser"] = PARSER

    assert isinstance(plan.declarations, MappingProxyType)
    assert plan.order == ("encoder",)
    assert tuple(plan.declarations) == ("encoder",)


def test_an_unresolved_dependency_is_refused() -> None:
    with pytest.raises(UnresolvedDependencyError, match="no component fills"):
        resolve_dependencies({"parser": PARSER})


def test_two_dependencies_arriving_on_one_input_are_refused() -> None:
    graph = {
        "encoder": ENCODER,
        "terms": Component(),
        "parser": Component(
            Dependency("encoder", "features"), Dependency("terms", "features")
        ),
    }

    with pytest.raises(WiringError, match="parser's features names both") as refused:
        resolve_dependencies(graph)

    assert refused.value.left == "parser"
    assert refused.value.right == "terms"


def test_a_dependency_cycle_is_refused_with_one_cycle_named() -> None:
    graph = {
        "a": Component(Dependency("b", "x")),
        "b": Component(Dependency("c", "x")),
        "c": Component(Dependency("b", "y")),
    }

    with pytest.raises(DependencyCycleError) as refused:
        resolve_dependencies(graph)

    assert refused.value.cycle == ("b", "c")


def test_an_empty_role_is_refused() -> None:
    with pytest.raises(ValueError, match="component role"):
        resolve_dependencies({"": ENCODER})


class Sharer:
    __slots__ = ("_holds", "_lends")

    def __init__(
        self,
        *,
        holds: Mapping[str, SubmoduleDeclaration],
        lends: frozenset[str],
    ) -> None:
        self._holds = holds
        self._lends = lends

    def sharing(self) -> Mapping[str, SubmoduleDeclaration]:
        return self._holds

    def publications(self) -> frozenset[str]:
        return self._lends


TIED: Final = SubmoduleDeclaration(
    "codebook", ModuleSharing(in_parameters=True, in_state_dict=True)
)
LENDER: Final = Sharer(holds=MappingProxyType({}), lends=frozenset({"codebook"}))


def test_sharing_resolves_to_the_consumer_port_it_arrives_on() -> None:
    resolved = resolve_sharing(
        {
            "decoder": Sharer(
                holds=MappingProxyType({"encoder": TIED}), lends=frozenset()
            ),
            "encoder": LENDER,
        },
        {("decoder", "encoder"): "encoder"},
    )

    assert resolved == {("decoder", "encoder"): TIED}


def test_sharing_on_a_port_no_dependency_supplies_is_refused() -> None:
    holder = Sharer(holds=MappingProxyType({"elsewhere": TIED}), lends=frozenset())

    with pytest.raises(WiringError, match="no dependency supplies decoder's elsewhere"):
        resolve_sharing(
            {"decoder": holder, "encoder": LENDER},
            {("decoder", "encoder"): "encoder"},
        )


def test_sharing_a_submodule_the_producer_does_not_lend_is_refused() -> None:
    holder = Sharer(holds=MappingProxyType({"encoder": TIED}), lends=frozenset())
    silent = Sharer(holds=MappingProxyType({}), lends=frozenset())

    with pytest.raises(
        WiringError, match="encoder publishes no codebook submodule for decoder"
    ):
        resolve_sharing(
            {"decoder": holder, "encoder": silent},
            {("decoder", "encoder"): "encoder"},
        )


def test_realized_sharing_is_checked_on_each_edge_with_its_declaration() -> None:
    seen: list[tuple[str, str, SubmoduleDeclaration | None]] = []

    def validator(
        consumer: str, producer: str, declaration: SubmoduleDeclaration | None
    ) -> bool:
        seen.append((consumer, producer, declaration))

        return True

    check_realized_sharing(
        {
            "decoder": "built-decoder",
            "encoder": "built-encoder",
            "probe": "built-probe",
        },
        {("decoder", "encoder"): "encoder", ("probe", "source"): "encoder"},
        {("decoder", "encoder"): TIED},
        validator,
    )

    assert sorted(seen, key=lambda edge: edge[0]) == [
        ("built-decoder", "built-encoder", TIED),
        ("built-probe", "built-encoder", None),
    ]


def test_realized_sharing_that_differs_from_its_declaration_is_refused() -> None:
    def validator(
        consumer: str, producer: str, declaration: SubmoduleDeclaration | None
    ) -> bool:
        return declaration is not None and consumer != producer

    with pytest.raises(
        WiringError,
        match="sharing from encoder to probe's source differs from its declaration",
    ):
        check_realized_sharing(
            {"decoder": "built-decoder", "encoder": "built-encoder", "probe": "p"},
            {("decoder", "encoder"): "encoder", ("probe", "source"): "encoder"},
            {("decoder", "encoder"): TIED},
            validator,
        )


class View:
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, View) and other.name == self.name

    @override
    def __hash__(self) -> int:
        return hash(self.name)


def test_readers_holding_the_supplied_codec_pass_the_identity_check() -> None:
    codec = View("bf")

    check_codec_identity("model", codec, {"decoder": codec, "encoder": codec})


def test_a_reader_holding_an_equal_but_distinct_codec_is_refused_by_role() -> None:
    codec = View("bf")

    with pytest.raises(
        LinearizationDisagreementError, match="and decoder reads"
    ) as raised:
        check_codec_identity("model", codec, {"decoder": View("bf"), "encoder": codec})

    assert (raised.value.left, raised.value.right) == ("model", "decoder")
    assert raised.value.left_codec != raised.value.right_codec


def test_the_identity_check_names_the_first_divergent_role_in_lexical_order() -> None:
    codec = View("bf")

    with pytest.raises(LinearizationDisagreementError) as raised:
        check_codec_identity(
            "model", codec, {"zeta": View("df"), "alpha": View("df"), "mid": codec}
        )

    assert raised.value.right == "alpha"
