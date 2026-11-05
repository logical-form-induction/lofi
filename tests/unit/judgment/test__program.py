from typing import TYPE_CHECKING, Final, cast

import pytest
import torch

from lofi.judgment import (
    Apply,
    Binary,
    Bind,
    Coordinate,
    Draw,
    Function,
    Gather,
    LinkingSteps,
    Literal,
    Name,
    ObserveBounded,
    ObserveChoice,
    ObserveOrdered,
    Operator,
    Prior,
    Table,
    TriangularTransform,
    ensure_identifier,
    evaluate,
    product,
    total,
)

if TYPE_CHECKING:
    from torch import Tensor


@pytest.mark.parametrize("text", ["eta", "_x1", "Participant", "lm_log_probability"])
def test_an_identifier_is_returned_as_given(text: str) -> None:
    assert ensure_identifier(text, "a name") is text


@pytest.mark.parametrize("text", ["", "1eta", "eta + 1", "by participant", "x[0]"])
def test_a_name_that_is_not_an_identifier_is_refused_naming_its_role(
    text: str,
) -> None:
    with pytest.raises(ValueError, match="a readout must be an identifier") as raised:
        ensure_identifier(text, "a readout")

    assert str(raised.value).endswith(repr(text))


def test_a_prior_holds_its_family_and_its_arguments_as_numbers() -> None:
    assert Prior("Normal", 0.0, 5.0).family == "Normal"
    assert Prior("Normal", 0.0, 5.0).arguments == (0.0, 5.0)
    assert Prior("Uniform").arguments == ()
    assert repr(Prior("Beta", 2.0, 3.5)) == "Prior('Beta', 2.0, 3.5)"
    assert Prior("Beta", 2, 3).arguments == (2.0, 3.0)


def test_a_prior_refuses_a_bad_family_name_or_an_infinite_argument() -> None:
    with pytest.raises(ValueError, match="distribution family must be an identifier"):
        Prior("Normal(0, 1)", 0.0)

    with pytest.raises(ValueError, match="must be finite"):
        Prior("Normal", 0.0, float("nan"))


def test_a_prior_refuses_an_argument_that_is_not_a_number() -> None:
    with pytest.raises(TypeError):
        Prior("Normal", cast("float", "0.0"), 1.0)


def test_a_draw_carries_and_validates_its_event_shape() -> None:
    draw = Draw(
        "correlation",
        ("Unit",),
        Prior("LKJCholesky", 1.0),
        event_shape=(2, 2),
    )

    assert draw.event_shape == (2, 2)
    assert repr(draw).endswith("event_shape=(2, 2))")
    assert draw != Draw("correlation", ("Unit",), Prior("LKJCholesky", 1.0))

    with pytest.raises(ValueError, match="event axes are positive"):
        Draw("bad", ("Unit",), Prior("Normal", 0.0, 1.0), event_shape=(2, 0))

    for shape in ((2, 1.5), (2, True)):
        with pytest.raises(TypeError, match="event axes are integers"):
            Draw(
                "bad",
                ("Unit",),
                Prior("Normal", 0.0, 1.0),
                event_shape=cast("tuple[int, ...]", shape),
            )


def test_a_coordinate_requires_a_nonnegative_integer() -> None:
    for index in (1.5, True):
        with pytest.raises(TypeError, match="coordinate is an integer"):
            Coordinate(Name("rows"), cast("int", index))

    with pytest.raises(ValueError, match="coordinate is nonnegative"):
        Coordinate(Name("rows"), -1)


def test_priors_compare_by_family_and_arguments() -> None:
    assert Prior("Normal", 0, 1) == Prior("Normal", 0.0, 1.0)
    assert Prior("Normal", 0.0, 1.0) != Prior("Normal", 0.0, 2.0)
    assert Prior("Normal", 1.0) != Prior("HalfNormal", 1.0)
    assert hash(Prior("Normal", 0, 1)) == hash(Prior("Normal", 0.0, 1.0))


def test_a_name_and_a_literal_hold_what_they_were_built_from() -> None:
    assert Name("eta").identifier == "eta"
    assert Literal(2.5).value == 2.5
    assert Literal(1).value == 1.0
    assert isinstance(Literal(1).value, float)
    assert Name("eta").names == frozenset({"eta"})
    assert Literal(2.5).names == frozenset()
    assert repr(Name("eta")) == "Name('eta')"
    assert repr(Literal(2.5)) == "Literal(2.5)"


def test_a_literal_must_be_finite() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        Literal(float("inf"))


def test_a_name_must_be_an_identifier() -> None:
    with pytest.raises(ValueError, match="a name must be an identifier"):
        Name("eta + 1")


def test_a_gather_reads_a_table_by_a_column() -> None:
    gathered = Gather("verb_effects", "verb")

    assert gathered.table == "verb_effects"
    assert gathered.index == "verb"
    assert gathered.names == frozenset({"verb_effects", "verb"})
    assert repr(gathered) == "Gather('verb_effects', 'verb')"


def test_a_table_reads_a_flat_draw_one_row_per_level() -> None:
    table = Table("offsets", "Participant", "Cutpoint", width=6)

    assert table.names == frozenset({"offsets"})
    assert (table.source, table.rows, table.columns, table.width) == (
        "offsets",
        "Participant",
        "Cutpoint",
        6,
    )
    assert repr(table) == "Table('offsets', 'Participant', 'Cutpoint', width=6)"


def test_a_table_needs_at_least_one_column() -> None:
    with pytest.raises(ValueError, match="at least one column"):
        Table("offsets", "Participant", "Cutpoint", width=0)


def test_a_triangular_transform_uses_the_transposed_factor_and_coordinates() -> None:
    expression = TriangularTransform(Name("factor"), Name("offsets"))
    factor = torch.tensor([[1.0, 0.0], [0.6, 0.8]], dtype=torch.float64)
    offsets = torch.tensor([[2.0, 3.0], [-1.0, 4.0]], dtype=torch.float64)
    transformed = evaluate(
        expression, {"factor": factor, "offsets": offsets}, like=offsets
    )

    assert expression.names == frozenset({"factor", "offsets"})
    assert transformed.shape == offsets.shape
    assert torch.equal(transformed, offsets @ factor.T)
    assert torch.equal(
        evaluate(
            Coordinate(expression, 1),
            {"factor": factor, "offsets": offsets},
            like=offsets,
        ),
        transformed[:, 1],
    )

    with pytest.raises(ValueError, match="nonnegative"):
        Coordinate(expression, -1)


def test_an_application_holds_its_function_and_its_argument() -> None:
    applied = Apply(Function.CUMSUM, Apply(Function.SOFTPLUS, Name("increments")))

    assert applied.argument == Apply(Function.SOFTPLUS, Name("increments"))
    assert applied.names == frozenset({"increments"})
    assert applied.function is Function.CUMSUM
    assert repr(applied) == (
        "Apply(<Function.CUMSUM: 'cumsum'>, "
        "Apply(<Function.SOFTPLUS: 'softplus'>, Name('increments')))"
    )


def test_sigmoid_is_an_expression_function() -> None:
    values = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64)

    assert torch.equal(
        evaluate(Apply(Function.SIGMOID, Name("eta")), {"eta": values}, like=values),
        torch.sigmoid(values),
    )


def test_a_bounded_observation_names_its_response_location_and_scale() -> None:
    observation = ObserveBounded(
        "unit_response",
        "Rows",
        location="bounded_location",
        scale="bounded_scale",
    )

    assert observation.response == "unit_response"
    assert observation.plate == "Rows"
    assert observation.location == "bounded_location"
    assert observation.scale == "bounded_scale"
    assert observation == ObserveBounded(
        "unit_response",
        "Rows",
        location="bounded_location",
        scale="bounded_scale",
    )


def test_a_combination_refers_to_the_names_of_both_operands() -> None:
    inner = Binary(
        Operator.ADD,
        Binary(Operator.SUB, Name("p"), Binary(Operator.MUL, Name("f"), Name("u"))),
        Name("l"),
    )
    divisor = Apply(
        Function.EXP, Binary(Operator.MUL, Name("n"), Apply(Function.LOG, Name("t")))
    )
    expression = Binary(
        Operator.DIV, Binary(Operator.MUL, Name("slope"), inner), divisor
    )

    assert expression.operator is Operator.DIV
    assert expression.right is divisor
    assert expression.names == frozenset({"slope", "p", "f", "u", "l", "n", "t"})


def test_expressions_compare_structurally() -> None:
    one = Binary(Operator.MUL, Name("a"), Literal(2.0))
    same = Binary(Operator.MUL, Name("a"), Literal(2.0))
    other = Binary(Operator.MUL, Literal(2.0), Name("a"))

    assert one == same
    assert hash(one) == hash(same)
    assert one != other
    assert Name("a") != Literal(2.0)
    assert Gather("t", "i") != Table("t", "R", "C", width=1)


def test_a_sum_and_a_product_associate_to_the_left() -> None:
    names = (Name("a"), Name("b"), Name("c"))

    assert total(names) == Binary(
        Operator.ADD, Binary(Operator.ADD, Name("a"), Name("b")), Name("c")
    )
    assert product(iter(names)) == Binary(
        Operator.MUL, Binary(Operator.MUL, Name("a"), Name("b")), Name("c")
    )
    assert product((Name("a"),)) == Name("a")


def test_a_sum_or_a_product_of_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one expression"):
        total(())

    with pytest.raises(ValueError, match="at least one expression"):
        product(())


def test_a_draw_holds_its_site_the_product_of_its_sets_and_its_prior() -> None:
    draw = Draw("offsets", ("Participant", "Cutpoint"), Prior("Normal", 0.0, 1.0))

    assert draw.site == "offsets"
    assert draw.index == ("Participant", "Cutpoint")
    assert draw.prior == Prior("Normal", 0.0, 1.0)
    assert draw == Draw("offsets", ("Participant", "Cutpoint"), Prior("Normal", 0, 1))
    assert repr(draw) == (
        "Draw('offsets', ('Participant', 'Cutpoint'), Prior('Normal', 0.0, 1.0))"
    )


def test_a_draw_needs_a_set_and_identifiers() -> None:
    with pytest.raises(ValueError, match="at least one finite set"):
        Draw("offsets", (), Prior("Normal", 0.0, 1.0))

    with pytest.raises(ValueError, match="an index set must be an identifier"):
        Draw("offsets", ("Participant * Cutpoint",), Prior("Normal", 0.0, 1.0))


def test_a_binding_holds_its_site_and_its_expression() -> None:
    bound = Bind("term", Binary(Operator.MUL, Literal(1.0), Name("p_gram")))

    assert bound.site == "term"
    assert bound.expression == Binary(Operator.MUL, Literal(1.0), Name("p_gram"))
    assert bound == Bind("term", Binary(Operator.MUL, Literal(1.0), Name("p_gram")))
    assert repr(bound) == (
        "Bind('term', Binary(<Operator.MUL: '*'>, Literal(1.0), Name('p_gram')))"
    )


def test_an_ordered_observation_names_its_location_and_its_thresholds() -> None:
    observed = ObserveOrdered(
        "response", "Resp", location="predictor", thresholds="cutpoints"
    )

    assert (observed.response, observed.plate) == ("response", "Resp")
    assert (observed.location, observed.thresholds) == ("predictor", "cutpoints")
    assert observed == ObserveOrdered(
        "response", "Resp", location="predictor", thresholds="cutpoints"
    )
    assert observed != ObserveOrdered(
        "response", "Resp", location="predictor", thresholds="scaled_cutpoints"
    )
    assert repr(observed) == (
        "ObserveOrdered('response', 'Resp', location='predictor', "
        "thresholds='cutpoints')"
    )


def test_steps_of_different_kinds_never_compare_equal() -> None:
    assert Bind("x", Name("y")) != Draw("x", ("Unit",), Prior("Normal", 0.0, 1.0))
    assert hash(Bind("x", Name("y"))) != hash(Bind("x", Name("z")))


def test_linking_steps_keep_their_draws_in_order_and_their_expression() -> None:
    expression = Binary(
        Operator.ADD,
        Binary(Operator.MUL, Name("lm_slope"), Name("lm")),
        Name("lm_shift"),
    )
    steps = LinkingSteps(
        {"lm_slope": Prior("HalfNormal", 1.0), "lm_shift": Prior("Normal", 0.0, 1.0)},
        expression,
    )

    assert tuple(steps.draws) == ("lm_slope", "lm_shift")
    assert steps.draws["lm_slope"] == Prior("HalfNormal", 1.0)
    assert steps.expression is expression
    assert repr(steps) == (
        "LinkingSteps({'lm_slope': Prior('HalfNormal', 1.0), "
        "'lm_shift': Prior('Normal', 0.0, 1.0)}, "
        "Binary(<Operator.ADD: '+'>, Binary(<Operator.MUL: '*'>, "
        "Name('lm_slope'), Name('lm')), Name('lm_shift')))"
    )


def test_linking_steps_hand_out_no_live_mapping() -> None:
    steps = LinkingSteps({"lm_slope": Prior("HalfNormal", 1.0)}, Name("lm"))

    with pytest.raises(TypeError):
        cast("dict[str, Prior]", steps.draws)["other"] = Prior("Normal", 0.0, 1.0)


def test_linking_steps_without_draws_are_a_pure_expression() -> None:
    steps = LinkingSteps({}, Name("lm"))

    assert dict(steps.draws) == {}
    assert steps.expression == Name("lm")


def test_linking_steps_refuse_a_site_that_is_not_an_identifier() -> None:
    with pytest.raises(ValueError, match="parameter site must be an identifier"):
        LinkingSteps({"lm slope": Prior("HalfNormal", 1.0)}, Name("lm"))


def test_a_choice_observation_scores_a_row_of_alternatives() -> None:
    observed = ObserveChoice("response", "Resp", logits="choice_logits")

    assert observed.response == "response"
    assert observed.plate == "Resp"
    assert observed.logits == "choice_logits"
    assert repr(observed) == (
        "ObserveChoice('response', 'Resp', logits='choice_logits')"
    )


def test_two_choice_observations_are_equal_on_their_three_names() -> None:
    observed = ObserveChoice("response", "Resp", logits="choice_logits")

    assert observed == ObserveChoice("response", "Resp", logits="choice_logits")
    assert hash(observed) == hash(
        ObserveChoice("response", "Resp", logits="choice_logits")
    )
    assert observed != ObserveChoice("response", "Resp", logits="other_logits")
    assert observed.__eq__(
        ObserveOrdered("response", "Resp", location="a", thresholds="b")
    ) is (NotImplemented)


@pytest.mark.parametrize(
    ("response", "plate", "logits"),
    [("1", "Resp", "logits"), ("response", "", "logits"), ("response", "Resp", "a b")],
)
def test_a_choice_observation_names_are_identifiers(
    response: str, plate: str, logits: str
) -> None:
    with pytest.raises(ValueError, match="must be an identifier"):
        ObserveChoice(response, plate, logits=logits)


DOUBLE: Final = torch.zeros((), dtype=torch.float64)


def environment() -> dict[str, Tensor]:
    return {
        "scale": torch.tensor([2.0], dtype=torch.float64),
        "offsets": torch.tensor([1.0, -1.0, 3.0], dtype=torch.float64),
        "index": torch.tensor([2, 0, 1, 2]),
        "column": torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64),
        "flat": torch.arange(6, dtype=torch.float64),
    }


def test_a_name_and_a_literal_evaluate_to_their_values() -> None:
    values = environment()

    assert evaluate(Name("column"), values, like=DOUBLE) is values["column"]
    assert float(evaluate(Literal(2.5), values, like=DOUBLE)) == 2.5


def test_a_literal_is_written_at_the_kind_and_on_the_device_it_is_evaluated_like() -> (
    None
):
    values = environment()
    single = torch.zeros(3, dtype=torch.float32)

    written = evaluate(Literal(2.5), values, like=single)

    assert written.dtype is torch.float32
    assert written.device == single.device
    assert written.shape == ()
    assert evaluate(Literal(2.5), values, like=DOUBLE).dtype is torch.float64


def test_a_gather_selects_the_table_rows_of_each_index() -> None:
    gathered = evaluate(Gather("offsets", "index"), environment(), like=DOUBLE)

    assert torch.equal(gathered, torch.tensor([3.0, 1.0, -1.0, 3.0]).double())


def test_a_table_reads_a_flat_site_one_row_per_level() -> None:
    table = evaluate(
        Table("flat", "Rows", "Columns", width=3), environment(), like=DOUBLE
    )

    assert torch.equal(table, torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).double())


def test_each_function_applies_along_the_last_axis() -> None:
    values = environment()
    flat = values["flat"].reshape(2, 3)

    assert torch.allclose(
        evaluate(Apply(Function.SOFTPLUS, Name("offsets")), values, like=DOUBLE),
        torch.nn.functional.softplus(values["offsets"]),
    )
    assert torch.equal(
        evaluate(
            Apply(Function.CUMSUM, Table("flat", "Rows", "Columns", width=3)),
            values,
            like=DOUBLE,
        ),
        flat.cumsum(dim=-1),
    )
    assert torch.allclose(
        evaluate(Apply(Function.EXP, Name("column")), values, like=DOUBLE),
        values["column"].exp(),
    )
    assert torch.allclose(
        evaluate(Apply(Function.LOG, Name("column")), values, like=DOUBLE),
        values["column"].log(),
    )


def test_each_operator_combines_its_operands_with_broadcasting() -> None:
    values = environment()
    column, scale = values["column"], values["scale"]
    cases = (
        (Operator.ADD, column + scale),
        (Operator.SUB, column - scale),
        (Operator.MUL, column * scale),
        (Operator.DIV, column / scale),
    )

    for operator, expected in cases:
        assert torch.equal(
            evaluate(
                Binary(operator, Name("column"), Name("scale")), values, like=DOUBLE
            ),
            expected,
        )


def test_evaluation_follows_the_tree_rather_than_the_text() -> None:
    values = environment()
    grouped = Binary(
        Operator.SUB, Name("column"), Binary(Operator.SUB, Name("scale"), Literal(1.0))
    )
    flat = Binary(
        Operator.SUB, Binary(Operator.SUB, Name("column"), Name("scale")), Literal(1.0)
    )

    assert torch.equal(evaluate(grouped, values, like=DOUBLE), values["column"] - 1.0)
    assert torch.equal(evaluate(flat, values, like=DOUBLE), values["column"] - 3.0)


def test_an_unbound_name_is_a_lookup_error() -> None:
    with pytest.raises(KeyError, match="absent"):
        evaluate(Name("absent"), environment(), like=DOUBLE)


def test_each_row_function_reduces_or_transforms_along_the_last_axis() -> None:
    values = environment()
    flat = Table("flat", "Rows", "Columns", width=3)
    rows = values["flat"].reshape(2, 3)

    assert torch.equal(
        evaluate(Apply(Function.MEAN, flat), values, like=DOUBLE), rows.mean(dim=-1)
    )
    assert torch.equal(
        evaluate(Apply(Function.SQUARE, Name("offsets")), values, like=DOUBLE),
        values["offsets"].square(),
    )
    assert torch.equal(
        evaluate(Apply(Function.SQRT, Name("column")), values, like=DOUBLE),
        values["column"].sqrt(),
    )
