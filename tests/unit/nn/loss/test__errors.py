from lofi.nn.loss._errors import (
    ImpossibleNegativeBracketError,
    ImproperMassError,
    MissingOutputError,
    ObjectiveError,
    SparseDivergenceError,
    StochasticFactoringError,
    UncomputedCellError,
    UndeclaredOutputError,
)


def test_an_undeclared_read_names_the_objective_and_the_field() -> None:
    error = UndeclaredOutputError(
        objective="coherence",
        field="cells",
        declared=frozenset({"splits", "spans"}),
    )

    assert isinstance(error, ObjectiveError)
    assert error.objective == "coherence"
    assert error.field == "cells"
    assert error.declared == frozenset({"splits", "spans"})
    assert str(error) == "coherence reads cells, having declared spans, splits"


def test_an_objective_declaring_nothing_reads_as_such() -> None:
    error = UndeclaredOutputError(
        objective="anchor",
        field="batch",
        declared=frozenset(),
    )

    assert "nothing" in str(error)


def test_a_missing_output_names_what_was_wanted_and_what_is_published() -> None:
    error = MissingOutputError(
        objective="reconstruction",
        missing=frozenset({"chart.outside"}),
        published=frozenset({"chart", "batch"}),
    )

    assert isinstance(error, ObjectiveError)
    assert error.missing == frozenset({"chart.outside"})
    assert error.published == frozenset({"chart", "batch"})
    assert str(error) == (
        "reconstruction reads chart.outside, and the graph publishes batch, chart"
    )


def test_a_graph_publishing_nothing_reads_as_such() -> None:
    error = MissingOutputError(
        objective="reconstruction",
        missing=frozenset({"cells"}),
        published=frozenset(),
    )

    assert "publishes nothing" in str(error)


def test_neither_error_is_swallowed_by_a_handler_for_argument_checking() -> None:
    errors = (
        UndeclaredOutputError(objective="a", field="cells", declared=frozenset()),
        MissingOutputError(objective="a", missing=frozenset(), published=frozenset()),
    )

    for error in errors:
        assert not isinstance(error, ValueError)
        assert not isinstance(error, TypeError)


def test_an_uncomputed_cell_names_the_item_and_the_span() -> None:
    error = UncomputedCellError(item=2, span=(1, 4))

    assert isinstance(error, ObjectiveError)
    assert not isinstance(error, ValueError)
    assert error.item == 2
    assert error.span == (1, 4)
    assert str(error) == "the readout never computed the span [1, 4) of item 2"


def test_an_improper_mass_names_the_readout_the_cell_and_the_value() -> None:
    error = ImproperMassError(readout="nouns", item=1, span=(0, 2), value=0.25)

    assert isinstance(error, ObjectiveError)
    assert not isinstance(error, ValueError)
    assert error.readout == "nouns"
    assert error.item == 1
    assert error.span == (0, 2)
    assert error.value == 0.25
    assert str(error) == (
        "the readout 'nouns' published the log probability 0.25 at the span [0, 2) "
        "of item 1, and a mass is at most one"
    )


def test_an_impossible_negative_bracket_names_the_objective_and_the_cell() -> None:
    error = ImpossibleNegativeBracketError(objective="brackets", item=1, span=(0, 2))

    assert isinstance(error, ObjectiveError)
    assert not isinstance(error, ValueError)
    assert error.objective == "brackets"
    assert error.item == 1
    assert error.span == (0, 2)
    assert str(error) == (
        "brackets excludes the span [0, 2) of item 1, which every derivation the "
        "parser assigns that item builds"
    )


def test_a_stochastic_factoring_names_the_decoder_and_its_role() -> None:
    error = StochasticFactoringError(role="model", decoder="DiffusionDecoder")

    assert isinstance(error, ObjectiveError)
    assert not isinstance(error, ValueError)
    assert error.role == "model"
    assert error.decoder == "DiffusionDecoder"
    assert str(error).startswith("the model decoder DiffusionDecoder draws a sample")


def test_a_sparse_divergence_names_the_mapping_the_comparison_and_the_finite_one() -> (
    None
):
    error = SparseDivergenceError(
        mapping="Entmax(1.5)",
        divergence="KullbackLeibler()",
        replacement="FenchelYoung(1.5)",
    )

    assert isinstance(error, ObjectiveError)
    assert not isinstance(error, ValueError)
    assert error.mapping == "Entmax(1.5)"
    assert error.divergence == "KullbackLeibler()"
    assert error.replacement == "FenchelYoung(1.5)"
    assert str(error) == (
        "Entmax(1.5) leaves symbols at exactly zero mass, and KullbackLeibler() is "
        "infinite wherever the other side has mass there. FenchelYoung(1.5) is the "
        "finite comparison built on the same entropy"
    )
