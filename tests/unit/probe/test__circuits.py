import json
from typing import TYPE_CHECKING, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi.nn.sites import ComponentId, Role, Site
from lofi.probe._circuits import (
    AblationEvidence,
    AlignmentEvidence,
    AttributionEvidence,
    Circuit,
    CircuitEdge,
    Evidence,
    StatisticalEvidence,
    Unit,
    evidence_from_json,
)
from lofi.probe._provenance import Provenance

if TYPE_CHECKING:
    from lofi._types import JsonValue

ENCODER = ComponentId("term-encoder")
DECODER = ComponentId("term-decoder")
MADE_BY = Provenance("eap-ig", "ccgbank-dev", 11)

UNITS = (
    Unit(Site(ENCODER, Role.CHART_CELL, (0, 3)), {"width": 4}),
    Unit(Site(ENCODER, Role.CHART_CELL, (0, 3)), {"width": 9}),
    Unit(Site(DECODER, Role.COMBINATOR_LOGITS), {"rule": 2}),
    Unit(Site(ENCODER, Role.RESIDUAL_STREAM, (2,)), {"row": 0, "width": 15}),
)

ATTRIBUTION = AttributionEvidence(-0.5, MADE_BY)
ABLATION = AblationEvidence(0.25, "resample", MADE_BY)
STATISTICAL = StatisticalEvidence(3.125, "span-permutation", MADE_BY)
ALIGNMENT = AlignmentEvidence(0.875, 8, MADE_BY)
VARIANTS: tuple[Evidence, ...] = (ATTRIBUTION, ABLATION, STATISTICAL, ALIGNMENT)


def test_a_unit_is_a_point_and_a_named_coordinate_rather_than_an_offset() -> None:
    unit = UNITS[3]

    assert unit.site.role is Role.RESIDUAL_STREAM
    assert dict(unit.index) == {"row": 0, "width": 15}
    assert unit.index["width"] == 15


def test_a_unit_at_a_point_holding_one_number_carries_no_coordinate() -> None:
    assert dict(Unit(Site(DECODER, Role.GATE)).index) == {}


def test_a_units_coordinate_cannot_be_written_through() -> None:
    unit = UNITS[0]

    with pytest.raises(TypeError):
        cast("dict[str, int]", unit.index)["width"] = 5

    assert unit.index["width"] == 4


def test_a_coordinate_counting_from_the_end_is_refused() -> None:
    with pytest.raises(ValueError, match="coordinate on axis 'width' is negative"):
        Unit(Site(ENCODER, Role.CHART_CELL, (0, 3)), {"width": -1})


def test_a_coordinate_on_an_unnamed_axis_is_refused() -> None:
    with pytest.raises(ValueError, match="axis name is empty"):
        Unit(Site(ENCODER, Role.CHART_CELL, (0, 3)), {"": 4})


def test_a_unit_wants_a_site_and_not_a_module_path() -> None:
    with pytest.raises(TypeError, match="site is not a Site"):
        Unit(cast("Site", "encoder.cell.0.3"))


def test_units_naming_the_same_entry_are_one_unit() -> None:
    unit = Unit(Site(ENCODER, Role.CHART_CELL, (0, 3)), {"width": 4})

    assert unit == UNITS[0]
    assert hash(unit) == hash(UNITS[0])
    assert unit != UNITS[1]
    assert unit != Unit(Site(ENCODER, Role.CHART_CELL, (0, 3)))
    assert unit != (Site(ENCODER, Role.CHART_CELL, (0, 3)), {"width": 4})


def test_a_coordinates_insertion_order_does_not_change_the_unit() -> None:
    one = Unit(Site(ENCODER, Role.RESIDUAL_STREAM, (2,)), {"row": 0, "width": 15})
    other = Unit(Site(ENCODER, Role.RESIDUAL_STREAM, (2,)), {"width": 15, "row": 0})

    assert one == other
    assert hash(one) == hash(other)


def test_a_circuit_holds_its_units_and_is_identified_by_them() -> None:
    circuit = Circuit(UNITS)

    assert circuit.units == frozenset(UNITS)
    assert len(circuit.id) == 32
    assert set(circuit.id) <= set("0123456789abcdef")


def test_two_circuits_built_in_different_orders_are_the_same_circuit() -> None:
    forward = Circuit(UNITS)
    backward = Circuit(reversed(UNITS))

    assert forward == backward
    assert hash(forward) == hash(backward)
    assert forward.id == backward.id
    assert len({forward, backward}) == 1


def test_a_repeated_unit_does_not_make_a_different_circuit() -> None:
    assert Circuit([*UNITS, UNITS[0]]) == Circuit(UNITS)
    assert Circuit([*UNITS, UNITS[0]]).id == Circuit(UNITS).id


def test_a_circuit_over_different_units_has_a_different_identifier() -> None:
    assert Circuit(UNITS).id != Circuit(UNITS[:3]).id
    assert Circuit(UNITS).id != Circuit([]).id


def test_a_search_that_found_nothing_is_the_empty_circuit() -> None:
    assert Circuit([]).units == frozenset()
    assert Circuit([]) == Circuit(())


def test_an_identifier_written_out_today_still_resolves_tomorrow() -> None:
    assert Circuit(UNITS).id == "ddece4be316115843fd3721322a3ef5d"


@settings(max_examples=50)
@given(order=st.permutations(list(range(len(UNITS)))))
def test_sorting_filtering_or_truncating_leaves_every_identifier_alone(
    order: list[int],
) -> None:
    shuffled = [UNITS[position] for position in order]

    assert Circuit(shuffled).id == Circuit(UNITS).id
    assert Circuit(shuffled[:2]).id == Circuit({shuffled[0], shuffled[1]}).id


def test_a_circuit_holds_units_and_not_the_sites_beneath_them() -> None:
    with pytest.raises(TypeError, match="holds units"):
        Circuit(cast("list[Unit]", [Site(ENCODER, Role.CHART_CELL, (0, 3))]))


def test_a_circuit_cannot_be_rebound() -> None:
    circuit = Circuit(UNITS)

    for held in ("units", "id"):
        with pytest.raises(AttributeError):
            setattr(circuit, held, None)

    assert not hasattr(circuit, "__dict__")


def test_the_repr_of_a_circuit_does_not_depend_on_the_order_it_was_built_in() -> None:
    assert repr(Circuit(UNITS)) == repr(Circuit(reversed(UNITS)))


def test_each_variant_carries_the_number_its_own_method_produces() -> None:
    assert ATTRIBUTION.effect == -0.5
    assert ABLATION.effect == 0.25
    assert ABLATION.baseline == "resample"
    assert STATISTICAL.statistic == 3.125
    assert STATISTICAL.null == "span-permutation"
    assert ALIGNMENT.accuracy == 0.875
    assert ALIGNMENT.rank == 8


def test_the_four_variants_are_told_apart_by_their_tag() -> None:
    assert [evidence.kind for evidence in VARIANTS] == [
        "attribution",
        "ablation",
        "statistical",
        "alignment",
    ]


def test_every_variant_carries_the_provenance_of_what_produced_it() -> None:
    for evidence in VARIANTS:
        assert evidence.provenance is MADE_BY


def test_every_variant_round_trips_through_json_as_the_same_evidence() -> None:
    for evidence in VARIANTS:
        written = json.loads(json.dumps(evidence.to_json()))
        read = evidence_from_json(written)

        assert read == evidence
        assert read.kind == evidence.kind
        assert read.provenance == MADE_BY


def test_a_serialized_evidence_carrying_an_unknown_tag_is_refused() -> None:
    written = dict(ATTRIBUTION.to_json())
    written["kind"] = "heuristic"

    with pytest.raises(ValueError, match="tagged 'heuristic'"):
        evidence_from_json(written)


def test_a_serialized_evidence_missing_a_field_of_its_variant_is_refused() -> None:
    written = dict(ABLATION.to_json())
    del written["baseline"]

    with pytest.raises(ValueError, match="no 'baseline' field"):
        evidence_from_json(written)


def test_a_serialized_evidence_missing_its_provenance_is_refused() -> None:
    written = dict(ATTRIBUTION.to_json())
    del written["provenance"]

    with pytest.raises(ValueError, match="no 'provenance' field"):
        evidence_from_json(written)


def test_evidence_read_from_a_bare_string_is_refused() -> None:
    with pytest.raises(TypeError, match="not a JSON object"):
        evidence_from_json(cast("JsonValue", "attribution: -0.5"))


def test_an_ablation_records_what_it_put_in_the_sources_place() -> None:
    with pytest.raises(ValueError, match="ablation baseline is empty"):
        AblationEvidence(0.25, "", MADE_BY)


def test_an_attributed_effect_that_is_not_a_number_is_refused() -> None:
    with pytest.raises(TypeError, match="attributed effect is not a number"):
        AttributionEvidence(cast("float", "large"), MADE_BY)


def test_an_effect_that_is_not_finite_is_refused() -> None:
    with pytest.raises(ValueError, match="not finite"):
        AttributionEvidence(float("nan"), MADE_BY)


def test_an_intervention_accuracy_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="not a proportion"):
        AlignmentEvidence(1.5, 8, MADE_BY)


def test_evidence_without_a_provenance_is_refused() -> None:
    with pytest.raises(TypeError, match="requires a Provenance"):
        AttributionEvidence(0.5, cast("Provenance", "eap-ig"))


def test_variants_recording_the_same_fields_are_one_piece_of_evidence() -> None:
    for evidence in VARIANTS:
        rebuilt = evidence_from_json(evidence.to_json())

        assert evidence == rebuilt
        assert hash(evidence) == hash(rebuilt)
        assert len({evidence, rebuilt}) == 1


def test_two_variants_that_happen_to_share_a_number_are_not_equal() -> None:
    attribution = AttributionEvidence(0.25, MADE_BY)

    assert attribution != AblationEvidence(0.25, "resample", MADE_BY)
    assert attribution != 0.25


def test_nothing_that_is_not_a_number_reaches_a_field_of_this_module() -> None:
    with pytest.raises(TypeError, match="not a number"):
        AttributionEvidence(cast("float", [0.5]), MADE_BY)

    with pytest.raises(TypeError, match="not an integer"):
        Unit(
            Site(ENCODER, Role.CHART_CELL, (0, 3)),
            cast("dict[str, int]", {"width": 0.5}),
        )


def test_an_edge_carries_a_source_a_target_and_what_the_claim_rests_on() -> None:
    edge = CircuitEdge(UNITS[0], UNITS[2], ATTRIBUTION)

    assert edge.source == UNITS[0]
    assert edge.target == UNITS[2]
    assert edge.evidence == ATTRIBUTION


def test_edges_recording_the_same_three_fields_are_one_edge() -> None:
    edge = CircuitEdge(UNITS[0], UNITS[2], ATTRIBUTION)
    same = CircuitEdge(UNITS[0], UNITS[2], AttributionEvidence(-0.5, MADE_BY))

    assert edge == same
    assert hash(edge) == hash(same)
    assert edge != CircuitEdge(UNITS[2], UNITS[0], ATTRIBUTION)
    assert edge != CircuitEdge(UNITS[0], UNITS[2], ABLATION)


def test_an_edge_runs_between_units_and_not_between_sites() -> None:
    with pytest.raises(TypeError, match="runs between units"):
        CircuitEdge(
            cast("Unit", Site(ENCODER, Role.CHART_CELL, (0, 3))),
            UNITS[2],
            ATTRIBUTION,
        )


def test_an_edge_wants_declared_evidence_and_not_a_bare_score() -> None:
    with pytest.raises(TypeError, match="not one of the declared kinds"):
        CircuitEdge(UNITS[0], UNITS[2], cast("Evidence", 0.5))


def test_evidence_stays_outside_the_identity_of_what_it_is_evidence_for() -> None:
    found = Circuit([UNITS[0], UNITS[2]])
    confirmed = Circuit([UNITS[0], UNITS[2]])

    assert CircuitEdge(UNITS[0], UNITS[2], ATTRIBUTION) != CircuitEdge(
        UNITS[0], UNITS[2], ABLATION
    )
    assert found == confirmed
    assert found.id == confirmed.id
