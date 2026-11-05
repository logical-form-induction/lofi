import pytest

from lofi.spec._errors import (
    AmbiguousSettingError,
    ConfigDirectoryError,
    ConfigError,
    ConfigValueError,
    DeclarationError,
    DependencyCycleError,
    DerivationIndexError,
    IdentifierError,
    InventoryExhaustedError,
    LinearizationDisagreementError,
    MissingSettingError,
    SettingParseError,
    UndeclaredLeafError,
    UnknownSettingError,
    UnresolvedDependencyError,
    WiringError,
)

CONFIG_ERRORS = (
    UnknownSettingError(path="model.parser.width", declared=("model.parser.kind",)),
    MissingSettingError(path="model.parser.kind"),
    ConfigValueError(path="trainer.steps", reason="a step count is positive"),
    SettingParseError(path="trainer.steps", expected="int", text="many"),
    AmbiguousSettingError(path="model.parser.width", variants=("cky", "shift")),
)

WIRING_ERRORS = (
    DependencyCycleError(cycle=("encoder", "controller")),
    UnresolvedDependencyError(
        consumer="parser", producer="encoder", port="features", reason="is not filled"
    ),
    LinearizationDisagreementError(
        left="encoder", left_codec="bf", right="decoder", right_codec="df"
    ),
)


@pytest.mark.parametrize("error", CONFIG_ERRORS, ids=lambda e: type(e).__name__)
def test_every_config_error_records_a_path_and_names_it(error: ConfigError) -> None:
    assert isinstance(error, ConfigError)
    assert error.path in str(error)


@pytest.mark.parametrize("error", CONFIG_ERRORS, ids=lambda e: type(e).__name__)
def test_a_config_error_is_not_an_argument_error(error: ConfigError) -> None:
    assert not isinstance(error, ValueError | TypeError)


def test_a_config_error_refuses_an_empty_path() -> None:
    with pytest.raises(ValueError, match="path is empty"):
        ConfigError("nowhere in particular", path="")


def test_an_unknown_setting_names_what_the_schema_declares() -> None:
    refusal = UnknownSettingError(path="model.width", declared=("model.kind", "seed"))

    assert refusal.declared == ("model.kind", "seed")
    assert "model.kind, seed" in str(refusal)


def test_an_unknown_setting_names_the_variants_that_declare_it() -> None:
    refusal = UnknownSettingError(
        path="model.encoders.terms.heads",
        declared=("model.encoders.terms.kind", "model.encoders.terms.width"),
        owners=("transformer",),
    )

    assert refusal.owners == ("transformer",)
    assert "belongs to transformer" in str(refusal)
    assert "model.encoders.terms.kind, model.encoders.terms.width" in str(refusal)
    assert UnknownSettingError(path="model.width", declared=()).owners == ()


def test_an_unknown_setting_says_so_when_the_schema_declares_nothing() -> None:
    assert "nothing" in str(UnknownSettingError(path="model.width", declared=()))


def test_a_refused_value_reports_why_the_schema_refused_it() -> None:
    refusal = ConfigValueError(path="trainer.steps", reason="a step count is positive")

    assert refusal.reason == "a step count is positive"
    assert "a step count is positive" in str(refusal)


@pytest.mark.parametrize("error", WIRING_ERRORS, ids=lambda e: type(e).__name__)
def test_every_wiring_error_names_two_sides(error: WiringError) -> None:
    assert isinstance(error, WiringError)
    assert error.left
    assert error.right
    assert error.left in str(error)
    assert error.right in str(error)


@pytest.mark.parametrize("error", WIRING_ERRORS, ids=lambda e: type(e).__name__)
def test_a_wiring_error_is_not_an_argument_error(error: WiringError) -> None:
    assert not isinstance(error, ValueError | TypeError)


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_wiring_error_refuses_an_empty_side(side: str) -> None:
    named = {"left": "encoder", "right": "parser"} | {side: ""}

    with pytest.raises(ValueError, match=f"{side} is empty"):
        WiringError("they disagree", **named)


def test_a_cycle_runs_from_its_first_role_to_its_last() -> None:
    refusal = DependencyCycleError(cycle=("encoder", "controller", "parser"))

    assert refusal.cycle == ("encoder", "controller", "parser")
    assert refusal.left == "encoder"
    assert refusal.right == "parser"


def test_a_cycle_closes_back_to_where_it_started() -> None:
    message = str(DependencyCycleError(cycle=("encoder", "controller")))

    assert message.endswith("encoder -> controller -> encoder")


def test_a_self_dependency_is_a_one_role_cycle() -> None:
    refusal = DependencyCycleError(cycle=("encoder",))

    assert refusal.cycle == ("encoder",)
    assert refusal.left == refusal.right == "encoder"
    assert str(refusal).endswith("encoder -> encoder")


def test_an_empty_cycle_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one role"):
        DependencyCycleError(cycle=())


def test_a_cycle_naming_a_role_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="names a role twice"):
        DependencyCycleError(cycle=("encoder", "parser", "encoder"))


def test_an_unresolved_dependency_names_the_input_and_the_reason() -> None:
    refusal = UnresolvedDependencyError(
        consumer="parser", producer="encoder", port="features", reason="is not filled"
    )

    assert refusal.port == "features"
    assert refusal.reason == "is not filled"
    assert "features" in str(refusal)
    assert "is not filled" in str(refusal)


def test_a_linearization_disagreement_reports_both_codecs() -> None:
    refusal = LinearizationDisagreementError(
        left="encoder", left_codec="bf", right="decoder", right_codec="df"
    )

    assert refusal.left_codec == "bf"
    assert refusal.right_codec == "df"
    assert "'bf'" in str(refusal)
    assert "'df'" in str(refusal)
    assert "do not share one linearization" in str(refusal)


def test_equal_codec_contents_can_still_name_distinct_view_identities() -> None:
    refusal = LinearizationDisagreementError(
        left="encoder",
        left_codec="view[0], parents=(-1, 0)",
        right="decoder",
        right_codec="view[1], parents=(-1, 0)",
    )

    assert refusal.left_codec != refusal.right_codec
    assert "identity" in str(refusal)
    assert "differently" not in str(refusal)


def test_a_derivation_index_error_names_the_rule_and_the_fault() -> None:
    error = DerivationIndexError(rule="coordination", reason="the index is free")

    assert isinstance(error, DeclarationError)
    assert error.rule == "coordination"
    assert error.reason == "the index is free"
    assert "coordination" in str(error)
    assert "the index is free" in str(error)

    with pytest.raises(ValueError, match="rule is empty"):
        DerivationIndexError(rule="", reason="the index is free")


def test_the_declaration_errors_are_value_errors() -> None:
    assert issubclass(IdentifierError, DeclarationError)
    assert issubclass(DeclarationError, ValueError)


def test_a_setting_parse_error_names_the_type_and_the_text() -> None:
    error = SettingParseError(path="trainer.steps", expected="int", text="many")

    assert error.expected == "int"
    assert error.text == "many"
    assert "expects int" in str(error)
    assert "'many'" in str(error)


def test_an_ambiguous_setting_names_the_variants_that_declare_it() -> None:
    error = AmbiguousSettingError(path="model.parser.width", variants=("a", "b"))

    assert error.variants == ("a", "b")
    assert "a, b" in str(error)


def test_an_undeclared_leaf_is_a_type_error_naming_the_path_and_type() -> None:
    error = UndeclaredLeafError(path="trainer.started", annotation="datetime")

    assert isinstance(error, TypeError)
    assert error.path == "trainer.started"
    assert error.annotation == "datetime"
    assert "trainer.started" in str(error)
    assert "datetime" in str(error)

    with pytest.raises(ValueError, match="path is empty"):
        UndeclaredLeafError(path="", annotation="datetime")


def test_a_config_directory_error_names_the_entry_point_and_the_reason() -> None:
    error = ConfigDirectoryError(
        entry="fixture", value="plugin:ROOT", reason="names an attribute"
    )

    assert error.entry == "fixture"
    assert error.value == "plugin:ROOT"
    assert error.reason == "names an attribute"
    assert "'fixture'" in str(error)
    assert "plugin:ROOT" in str(error)
    assert "names an attribute" in str(error)
    assert not isinstance(error, ConfigError)


def test_an_exhausted_inventory_is_a_declaration_error_naming_its_shortfall() -> None:
    error = InventoryExhaustedError(requested=7, reached=6, patience=50)

    assert isinstance(error, DeclarationError)
    assert (error.requested, error.reached, error.patience) == (7, 6, 50)
    assert "reached 6 of 7 lexical categories" in str(error)
    assert "the 50 derivations" in str(error)
