import didactic.api as dx
import pytest
from tests.support.term_reference import infix_syntax

from lofi.spec._errors import UnknownSettingError
from lofi.spec._ports import (
    ConfigSource,
    Dependency,
    LeafOrigin,
    Provenance,
)
from lofi.terms import Operator, SExpSyntax, Signature, TermSyntax

SOURCES = tuple(ConfigSource)


class Blank(dx.Model, extra="forbid"):
    steps: int = 1


def test_precedence_counts_from_the_declaration_order() -> None:
    assert [source.precedence for source in SOURCES] == list(range(len(SOURCES)))


def test_every_source_outranks_exactly_the_ones_declared_before_it() -> None:
    for position, source in enumerate(SOURCES):
        assert [source.outranks(other) for other in SOURCES] == [
            other < position for other in range(len(SOURCES))
        ]


def test_a_source_does_not_outrank_itself() -> None:
    assert not any(source.outranks(source) for source in SOURCES)


def test_the_schema_default_is_the_weakest_source() -> None:
    assert min(SOURCES, key=lambda source: source.precedence) is ConfigSource.DEFAULT


def test_an_override_is_the_strongest_source() -> None:
    assert max(SOURCES, key=lambda source: source.precedence) is ConfigSource.OVERRIDE


def test_a_provenance_reports_the_source_of_a_leaf() -> None:
    provenance = Provenance(
        origins={
            "trainer.steps": LeafOrigin(source=ConfigSource.FILE, layer="run.json"),
            "seed": LeafOrigin(source=ConfigSource.OVERRIDE, layer="seed=3"),
        }
    )

    assert provenance.source_of("trainer.steps") is ConfigSource.FILE
    assert provenance.source_of("seed") is ConfigSource.OVERRIDE


def test_a_provenance_reports_the_whole_origin_of_a_leaf() -> None:
    origin = LeafOrigin(
        source=ConfigSource.PROFILE, layer="fast", expression="${trainer.steps}"
    )
    provenance = Provenance(origins={"trainer.warmup": origin})

    assert provenance.origin_of("trainer.warmup") == origin
    assert provenance.origin_of("trainer.warmup").layer == "fast"
    assert provenance.origin_of("trainer.warmup").expression == "${trainer.steps}"


def test_an_origin_names_no_layer_and_no_expression_unless_given() -> None:
    origin = LeafOrigin(source=ConfigSource.DEFAULT)

    assert origin.layer == ""
    assert origin.expression is None


def test_a_provenance_refuses_a_leaf_it_does_not_name() -> None:
    provenance = Provenance(origins={"seed": LeafOrigin(source=ConfigSource.FILE)})

    with pytest.raises(UnknownSettingError) as refused:
        provenance.source_of("trainer.steps")

    assert refused.value.path == "trainer.steps"
    assert refused.value.declared == ("seed",)


def test_a_provenance_holds_its_paths_in_path_order() -> None:
    provenance = Provenance(
        origins={
            "seed": LeafOrigin(source=ConfigSource.FILE),
            "data.split": LeafOrigin(source=ConfigSource.PROFILE),
        }
    )

    assert provenance.paths == ("data.split", "seed")


def test_two_provenances_over_the_same_leaves_are_equal() -> None:
    one = Provenance(
        origins={
            "seed": LeafOrigin(source=ConfigSource.FILE),
            "data.split": LeafOrigin(source=ConfigSource.PROFILE),
        }
    )
    other = Provenance(
        origins={
            "data.split": LeafOrigin(source=ConfigSource.PROFILE),
            "seed": LeafOrigin(source=ConfigSource.FILE),
        }
    )

    assert one == other


def test_two_provenances_differing_in_a_layer_are_not_equal() -> None:
    one = Provenance(
        origins={"seed": LeafOrigin(source=ConfigSource.PROFILE, layer="fast")}
    )
    other = Provenance(
        origins={"seed": LeafOrigin(source=ConfigSource.PROFILE, layer="long")}
    )

    assert one != other


def test_a_provenance_reads_back_from_its_json_form() -> None:
    provenance = Provenance(
        origins={
            "seed": LeafOrigin(source=ConfigSource.FILE, layer="run.json"),
            "data.split": LeafOrigin(source=ConfigSource.DEFAULT, expression="${x}"),
        }
    )

    assert Provenance.model_validate_json(provenance.model_dump_json()) == provenance


@pytest.mark.parametrize("path", ["", "trainer..steps", ".seed", "seed."])
def test_a_provenance_refuses_a_key_that_is_not_a_dotted_path(path: str) -> None:
    with pytest.raises(dx.ValidationError):
        Provenance(origins={path: LeafOrigin(source=ConfigSource.FILE)})


def test_a_dependency_names_a_role_and_an_input() -> None:
    dependency = Dependency("encoder", "features")

    assert dependency.role == "encoder"
    assert dependency.port == "features"
    assert repr(dependency) == "Dependency('encoder', 'features')"


def test_two_dependencies_on_one_role_and_input_are_one_value() -> None:
    assert Dependency("encoder", "features") == Dependency("encoder", "features")
    assert hash(Dependency("encoder", "features")) == hash(
        Dependency("encoder", "features")
    )


@pytest.mark.parametrize(
    ("role", "port"), [("encoder", "chart"), ("controller", "features")]
)
def test_a_dependency_differing_in_either_name_is_a_different_value(
    role: str, port: str
) -> None:
    assert Dependency(role, port) != Dependency("encoder", "features")


def test_a_dependency_is_not_equal_to_something_that_is_not_one() -> None:
    assert Dependency("encoder", "features") != ("encoder", "features")


@pytest.mark.parametrize(("role", "port"), [("", "features"), ("encoder", "")])
def test_a_dependency_refuses_an_empty_name(role: str, port: str) -> None:
    with pytest.raises(ValueError, match="is empty"):
        Dependency(role, port)


def test_both_term_syntaxes_satisfy_the_syntax_port() -> None:
    signature = Signature(
        "port",
        ("Cat",),
        (
            Operator("S", "Cat"),
            Operator("/", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
            Operator("\\", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
        ),
    )

    for syntax in (
        SExpSyntax(signature),
        infix_syntax(signature, ("/", "/"), ("\\", "\\")),
    ):
        assert isinstance(syntax, TermSyntax)
        assert syntax.signature is signature
