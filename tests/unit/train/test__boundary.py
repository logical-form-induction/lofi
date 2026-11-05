import json
from typing import Final, Literal

import didactic.api as dx
import pytest

from lofi.lf.provers import ProofBudget
from lofi.nn.grounding import GradientPath, GroundingMetadata, GroundingTelemetry
from lofi.run import SeedPolicy
from lofi.train._boundary import (
    CheckpointManifest,
    ComponentRecord,
    FittingRecord,
    GroundingRecord,
    ParserFittingRecord,
    PrecisionRecord,
    RngRecord,
    StageRecord,
    StateReference,
    StepFittingRecord,
    disagreements,
    grounding_record,
    read_manifest,
    staged_manifest,
    write_manifest,
)

FORWARD: Final = "forward_application"

BACKWARD: Final = "backward_application"

CONTROLLER: Final = "controller"

STEP: Final = 11


def record(**moved: str) -> ComponentRecord:
    return ComponentRecord(
        tag="hrr",
        spec_fingerprint="spec-a1",
        signature_fingerprint="sig-b2",
        codec_identity="codec-c3",
        state=StateReference(key="decoder", digest="d4"),
    ).with_(**moved)


def manifest() -> CheckpointManifest:
    return CheckpointManifest(
        run_id="run-0001",
        step=STEP,
        data_position=19,
        code_version="0.1.0",
        action_set="actions-a1",
        components={FORWARD: record(), BACKWARD: record()},
        rng=RngRecord(streams={"python": "9f3a"}),
        precision=PrecisionRecord(
            compute_dtype="bfloat16",
            autocast_device_type="cuda",
            gradient_scaling=True,
        ),
        optimizer=StateReference(key="optimizer", digest="a7"),
        scaler=StateReference(key="scaler", digest="b8"),
        scheduler=StateReference(key="scheduler", digest="c9"),
    )


def test_the_roles_of_a_manifest_are_the_keys_its_components_are_stored_under() -> None:
    assert manifest().roles == frozenset({FORWARD, BACKWARD})


def test_two_roles_of_one_kind_hold_records_that_do_not_tell_them_apart() -> None:
    held = manifest().components

    assert held[FORWARD] == held[BACKWARD]


def controller() -> ComponentRecord:
    return ComponentRecord(
        tag="mlp",
        spec_fingerprint="spec-m1",
        signature_fingerprint="sig-b2",
        codec_identity="codec-c3",
        state=StateReference(key="controller", digest="e5"),
        labels={FORWARD: 0, BACKWARD: 1},
    )


def controlled() -> CheckpointManifest:
    built = manifest()

    return built.with_(components={**built.components, CONTROLLER: controller()})


def test_a_component_record_names_no_field_holding_a_class_name() -> None:
    declared = frozenset(ComponentRecord.__field_specs__)

    assert declared == frozenset(
        {
            "tag",
            "spec_fingerprint",
            "signature_fingerprint",
            "codec_identity",
            "state",
            "labels",
            "pinned",
        }
    )


def test_a_record_without_a_label_map_holds_none() -> None:
    assert record().labels is None


def test_a_label_map_is_read_back_as_written() -> None:
    assert controller().labels == {FORWARD: 0, BACKWARD: 1}


def test_an_empty_label_map_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="names no label"):
        controller().with_(labels={})


@pytest.mark.parametrize(
    "labels",
    [{FORWARD: 1, BACKWARD: 2}, {FORWARD: 0, BACKWARD: 0}, {FORWARD: 0, BACKWARD: 2}],
)
def test_a_label_map_whose_columns_are_not_the_first_positions_is_refused(
    labels: dict[str, int],
) -> None:
    with pytest.raises(dx.ValidationError, match="fills the columns"):
        controller().with_(labels=labels)


def test_a_label_map_over_one_label_fills_column_zero() -> None:
    assert controller().with_(labels={FORWARD: 0}).labels == {FORWARD: 0}


def test_a_manifest_that_agrees_with_itself_reports_nothing() -> None:
    assert disagreements(manifest(), manifest()) == ()


def test_advancing_counters_and_random_state_are_not_grounds_for_refusal() -> None:
    moved = manifest().with_(
        step=STEP + 5,
        data_position=27,
        rng=RngRecord(streams={"python": "0000"}),
    )

    assert disagreements(manifest(), moved) == ()


def test_a_change_of_run_identity_is_grounds_for_refusal() -> None:
    (reason,) = disagreements(manifest(), manifest().with_(run_id="run-0002"))

    assert "run identity" in reason
    assert "run-0001" in reason
    assert "run-0002" in reason


def test_a_change_of_precision_is_grounds_for_refusal() -> None:
    moved = manifest().with_(
        precision=PrecisionRecord(
            compute_dtype="float32",
            autocast_device_type="cuda",
            gradient_scaling=False,
        )
    )
    (reason,) = disagreements(manifest(), moved)

    assert "precision" in reason
    assert "float32" in reason


def test_run_state_topology_compares_parts_and_keys_but_not_digests() -> None:
    built = manifest()
    redigested = built.with_(
        optimizer=StateReference(key="optimizer", digest="changed"),
        scaler=StateReference(key="scaler", digest="changed"),
        scheduler=StateReference(key="scheduler", digest="changed"),
    )
    rekeyed = built.with_(scheduler=StateReference(key="schedule", digest="changed"))

    assert disagreements(built, redigested) == ()

    (reason,) = disagreements(built, rekeyed)

    assert "run-state topology" in reason
    assert "scheduler" in reason
    assert "schedule" in reason


def test_a_role_the_checkpoint_lacks_is_reported_against_the_role_name() -> None:
    thinner = manifest().with_(components={FORWARD: record()})
    reported = disagreements(manifest(), thinner)

    assert len(reported) == 1
    assert BACKWARD in reported[0]


def test_a_role_the_wiring_lacks_is_reported_as_unwired() -> None:
    thinner = manifest().with_(components={FORWARD: record()})
    reported = disagreements(thinner, manifest())

    assert len(reported) == 1
    assert "unwired" in reported[0]


@pytest.mark.parametrize(
    "field",
    ["tag", "spec_fingerprint", "signature_fingerprint", "codec_identity"],
)
def test_each_identifying_field_of_a_record_is_grounds_for_refusal(field: str) -> None:
    apart = record(**{field: "moved"})
    moved = manifest().with_(components={FORWARD: apart, BACKWARD: record()})
    reported = disagreements(manifest(), moved)

    assert len(reported) == 1
    assert field in reported[0]
    assert FORWARD in reported[0]


def test_every_identifying_field_is_reported_rather_than_the_first() -> None:
    apart = record(tag="rnn", spec_fingerprint="other", codec_identity="other")
    moved = manifest().with_(components={FORWARD: apart, BACKWARD: record()})

    assert len(disagreements(manifest(), moved)) == 3


def test_a_record_has_exactly_the_six_fields_the_comparison_reads() -> None:
    every = record(
        tag="rnn",
        spec_fingerprint="x",
        signature_fingerprint="y",
        codec_identity="z",
    ).with_(labels={FORWARD: 0}, pinned="an-org/a-model@0123")
    moved = manifest().with_(components={FORWARD: every, BACKWARD: record()})

    assert len(disagreements(manifest(), moved)) == 6


def test_a_record_holds_no_pinned_checkpoint_unless_one_is_given() -> None:
    assert record().pinned is None


def test_a_change_of_pinned_checkpoint_is_grounds_for_refusal_naming_both() -> None:
    pinned = manifest().with_(
        components={
            FORWARD: record().with_(pinned="an-org/a-model@0123"),
            BACKWARD: record(),
        }
    )
    repinned = manifest().with_(
        components={
            FORWARD: record().with_(pinned="an-org/a-model@4567"),
            BACKWARD: record(),
        }
    )
    (reason,) = disagreements(pinned, repinned)

    assert "pinned" in reason
    assert FORWARD in reason
    assert "@0123" in reason
    assert "@4567" in reason


def test_a_change_of_action_set_is_grounds_for_refusal_naming_the_digest() -> None:
    (reason,) = disagreements(manifest(), manifest().with_(action_set="reordered"))

    assert "action set digest" in reason
    assert "actions-a1" in reason
    assert "reordered" in reason


def test_a_relabeled_controller_is_grounds_for_refusal_naming_the_role() -> None:
    swapped = controller().with_(labels={FORWARD: 1, BACKWARD: 0})
    built = controlled()
    moved = built.with_(components={**built.components, CONTROLLER: swapped})
    reported = disagreements(built, moved)

    assert len(reported) == 1
    assert CONTROLLER in reported[0]
    assert "labels" in reported[0]


def test_a_label_map_that_appears_or_vanishes_is_grounds_for_refusal() -> None:
    built = controlled()
    unlabeled = built.with_(
        components={**built.components, CONTROLLER: controller().with_(labels=None)}
    )

    assert len(disagreements(built, unlabeled)) == 1
    assert len(disagreements(unlabeled, built)) == 1


def test_a_change_of_code_version_is_grounds_for_refusal() -> None:
    reported = disagreements(manifest(), manifest().with_(code_version="0.2.0"))

    assert len(reported) == 1
    assert "code version" in reported[0]


def test_a_checkpoint_records_advancing_stream_state_and_no_seed_policy() -> None:
    built = manifest().with_(
        rng=RngRecord(streams={"loader:training": "encoded-state"}),
    )
    written = write_manifest(built)
    restored = read_manifest(written)

    assert restored.rng.streams == {"loader:training": "encoded-state"}
    assert "seed_policy" not in json.loads(written)


def test_a_checkpoint_refuses_a_seed_policy_its_run_manifest_records() -> None:
    document = json.loads(manifest().model_dump_json())
    document["seed_policy"] = dict(SeedPolicy(7).record)

    with pytest.raises(dx.ValidationError, match="seed_policy"):
        CheckpointManifest.model_validate(document)


def test_the_reasons_are_reported_in_role_order() -> None:
    moved = manifest().with_(
        components={FORWARD: record(tag="rnn"), BACKWARD: record(tag="cnn")}
    )
    reported = disagreements(manifest(), moved)

    assert [BACKWARD in reason for reason in reported] == [True, False]


def test_a_manifest_round_trips_through_its_own_wire_form() -> None:
    built = manifest()

    assert CheckpointManifest.model_validate_json(built.model_dump_json()) == built


def test_fitting_records_distinguish_best_from_resumable_checkpoints() -> None:
    best = ParserFittingRecord(
        kind="validation-best",
        completed_epochs=2,
        metric="loss/parser",
        best=0.5,
        failures=0,
        best_checkpoint=None,
    )
    latest = best.with_(
        kind="resume-latest",
        failures=1,
        best_checkpoint="000001-run-step2",
    )

    assert latest.best == best.best
    assert latest.failures == 1
    assert latest.best_checkpoint == "000001-run-step2"
    distance = FittingRecord.model_validate(
        {
            "unit": "steps",
            "kind": "resume-latest",
            "completed_steps": 3,
        }
    )

    assert distance == StepFittingRecord(completed_steps=3)

    with pytest.raises(dx.ValidationError, match="must name"):
        latest.with_(best_checkpoint=None)

    with pytest.raises(dx.ValidationError, match="cannot name"):
        best.with_(best_checkpoint="000001-run-step2")


def test_a_prevalidation_parser_checkpoint_records_only_resumable_progress() -> None:
    record = ParserFittingRecord(
        kind="resume-latest",
        completed_epochs=1,
        metric="loss/parser",
        best=None,
        failures=0,
        best_checkpoint=None,
    )

    assert FittingRecord.model_validate_json(record.model_dump_json()) == record

    with pytest.raises(dx.ValidationError, match="cannot name a best checkpoint"):
        record.with_(best_checkpoint="000001-run-step1")

    with pytest.raises(dx.ValidationError, match="cannot record validation failures"):
        record.with_(failures=1)


@pytest.mark.parametrize("stop", ["target-met", "target-not-met"])
def test_a_resumable_parser_checkpoint_records_a_target_outcome(
    stop: Literal["target-met", "target-not-met"],
) -> None:
    record = ParserFittingRecord(
        kind="resume-latest",
        completed_epochs=2,
        metric="loss/parser",
        best=0.5,
        failures=0,
        best_checkpoint="000001-run-step2",
        stop=stop,
        achieved_accuracy=0.875,
    )

    restored = FittingRecord.model_validate_json(record.model_dump_json())

    assert restored == record


def test_a_parser_target_outcome_requires_its_held_out_accuracy() -> None:
    with pytest.raises(dx.ValidationError, match="both stop and achieved_accuracy"):
        ParserFittingRecord(
            kind="resume-latest",
            completed_epochs=2,
            metric="loss/parser",
            best=0.5,
            failures=0,
            best_checkpoint="000001-run-step2",
            stop="target-met",
        )


def test_a_manifest_data_position_is_nonnegative_and_round_trips() -> None:
    built = manifest()
    restored = read_manifest(write_manifest(built))

    assert restored.data_position == 19

    for value, named in (
        (-1, "data position"),
        (True, "data_position"),
        (1.5, "data_position"),
    ):
        with pytest.raises(dx.ValidationError, match=named):
            built.with_(data_position=value)


def test_a_manifest_with_no_optimizer_state_leaves_the_three_references_empty() -> None:
    bare = CheckpointManifest(
        run_id="run-0002",
        step=0,
        data_position=0,
        code_version="0.1.0",
        action_set="actions-a1",
        components={},
        rng=RngRecord(streams={}),
        precision=PrecisionRecord(
            compute_dtype="float32",
            autocast_device_type="cpu",
            gradient_scaling=False,
        ),
    )

    assert (bare.optimizer, bare.scaler, bare.scheduler) == (None, None, None)


def test_the_three_references_are_read_back_as_they_were_written() -> None:
    built = manifest()

    assert built.optimizer == StateReference(key="optimizer", digest="a7")
    assert built.scaler == StateReference(key="scaler", digest="b8")
    assert built.scheduler == StateReference(key="scheduler", digest="c9")


def test_the_references_are_keyed_by_the_name_each_was_written_under() -> None:
    built = manifest()

    assert dict(built.references) == {
        "optimizer": built.optimizer,
        "scaler": built.scaler,
        "scheduler": built.scheduler,
    }


def test_an_absent_reference_is_absent_from_the_keyed_references() -> None:
    thinner = manifest().with_(scaler=None, scheduler=None)

    assert frozenset(thinner.references) == frozenset({"optimizer"})


def test_the_references_a_manifest_hands_out_are_no_container_a_caller_can_write() -> (
    None
):
    assert not isinstance(manifest().references, dict)


def test_two_references_under_one_key_are_refused() -> None:
    with pytest.raises(dx.ValidationError, match="optimizer"):
        manifest().with_(scaler=StateReference(key="optimizer", digest="b8"))


def test_a_reference_keyed_by_a_role_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match=FORWARD):
        manifest().with_(scaler=StateReference(key=FORWARD, digest="b8"))


def test_a_written_manifest_reads_back_equal_with_its_labels() -> None:
    built = controlled()

    assert read_manifest(write_manifest(built)) == built
    assert read_manifest(write_manifest(built)).components[CONTROLLER].labels == {
        FORWARD: 0,
        BACKWARD: 1,
    }


def test_a_written_manifest_is_one_json_object() -> None:
    assert isinstance(json.loads(write_manifest(manifest())), dict)


def test_text_that_is_not_json_is_refused_as_a_value_error() -> None:
    with pytest.raises(ValueError, match="Expecting value"):
        read_manifest("not a manifest")


def test_json_that_is_not_a_manifest_is_refused_naming_what_is_missing() -> None:
    with pytest.raises(ValueError, match="components"):
        read_manifest('{"run_id": "run-0001"}')


def test_a_manifest_with_an_unknown_field_is_refused_on_read() -> None:
    forged = {**json.loads(write_manifest(manifest())), "verified": True}

    with pytest.raises(ValueError, match="verified"):
        read_manifest(json.dumps(forged))


def test_an_unknown_field_is_refused_where_the_manifest_is_constructed() -> None:
    with pytest.raises(dx.ValidationError, match="verified"):
        CheckpointManifest.model_validate(
            {
                "run_id": "run-0003",
                "step": 0,
                "code_version": "0.1.0",
                "components": {},
                "rng": RngRecord(streams={}),
                "precision": PrecisionRecord(
                    compute_dtype="float32",
                    autocast_device_type="cpu",
                    gradient_scaling=False,
                ),
                "verified": True,
            }
        )


def grounding() -> GroundingRecord:
    return GroundingRecord(
        variant="pairwise",
        scorer="prover",
        source="truth-table",
        gradient_path="SCORE_FUNCTION",
        prover="truth-table",
        prover_version="builtin",
        budget_calls=2048,
        budget_wall_seconds=30.0,
        budget_cpu_seconds=None,
        calls=12,
        questions=16,
        questions_per_second=8.0,
        memo_hits=4,
        memo_hit_rate=0.25,
        cpu_seconds=1.0,
        wall_seconds=2.0,
        process_starts=1,
        label_agreement=0.75,
        abstentions={"timeout": 2},
        degenerate_mass=0.1,
        malformed_mass=0.0,
        reward_mean=0.75,
        variance=0.125,
        verified_share=None,
        out_of_fragment_share=0.125,
    )


def test_a_manifest_names_the_grounding_the_inference_term_trained_under() -> None:
    held = manifest().with_(grounding=grounding())

    assert held.grounding is not None
    assert held.grounding.variant == "pairwise"
    assert held.grounding.scorer == "prover"
    assert held.grounding.source == "truth-table"
    assert held.grounding.path is GradientPath.SCORE_FUNCTION
    assert held.grounding.budget_calls == 2048
    assert held.grounding.questions_per_second == 8.0
    assert held.grounding.memo_hit_rate == 0.25
    assert held.grounding.process_starts == 1
    assert manifest().grounding is None


def test_a_grounding_record_round_trips_through_the_manifest_text() -> None:
    held = manifest().with_(grounding=grounding())

    assert read_manifest(write_manifest(held)) == held
    assert read_manifest(write_manifest(manifest())).grounding is None


def test_a_change_of_grounding_is_grounds_for_refusal() -> None:
    (reason,) = disagreements(manifest(), manifest().with_(grounding=grounding()))

    assert "grounding" in reason
    assert "pairwise" in reason


def test_a_grounding_record_refuses_what_no_run_could_have_taken() -> None:
    with pytest.raises(dx.ValidationError):
        grounding().with_(variant="")

    with pytest.raises(dx.ValidationError):
        grounding().with_(gradient_path="surrogate")

    with pytest.raises(dx.ValidationError):
        grounding().with_(calls=-1)

    with pytest.raises(dx.ValidationError):
        grounding().with_(memo_hit_rate=1.1)

    with pytest.raises(dx.ValidationError):
        grounding().with_(out_of_fragment_share=1.1)


def test_a_grounding_record_names_its_derived_gradient_path() -> None:
    assert grounding().path is GradientPath.SCORE_FUNCTION


def test_grounding_identity_and_step_telemetry_lower_to_one_record() -> None:
    held = grounding_record(
        GroundingMetadata(
            "pairwise",
            scorer="prover",
            source="truth-table",
            prover="truth-table",
            prover_version=None,
            budget=ProofBudget(
                calls_per_step=4,
                wall_per_step=2.0,
                cpu_seconds=3.0,
            ),
        ),
        GroundingTelemetry(
            GradientPath.SCORE_FUNCTION,
            {"timeout": 1},
            calls=2,
            questions=3,
            questions_per_second=6.0,
            memo_hits=1,
            memo_hit_rate=1.0 / 3.0,
            cpu_seconds=0.25,
            wall_seconds=0.5,
            process_starts=2,
            label_agreement=2.0 / 3.0,
            degenerate_mass=0.125,
            malformed_mass=0.0,
            reward_mean=0.75,
            variance=0.0625,
            verified_share=0.5,
            out_of_fragment_share=1.0 / 3.0,
        ),
    )

    assert held.gradient_path == "SCORE_FUNCTION"
    assert held.budget_calls == 4
    assert held.questions == 3
    assert held.questions_per_second == 6.0
    assert held.memo_hit_rate == pytest.approx(1.0 / 3.0)
    assert held.process_starts == 2
    assert held.label_agreement == pytest.approx(2.0 / 3.0)
    assert held.out_of_fragment_share == pytest.approx(1.0 / 3.0)


def test_advancing_grounding_telemetry_does_not_refuse_a_checkpoint() -> None:
    expected = manifest().with_(grounding=grounding())
    found = expected.with_(
        grounding=grounding().with_(
            gradient_path="PATHWISE",
            calls=99,
            questions=100,
            questions_per_second=50.0,
            memo_hits=1,
            memo_hit_rate=0.01,
            wall_seconds=2.0,
        )
    )

    assert disagreements(expected, found) == ()


def staged() -> CheckpointManifest:
    return staged_manifest(
        manifest(),
        name="type-map",
        active=["chart.layers", "chart.embeddings"],
        position=1,
    )


def test_a_manifest_names_the_stage_of_the_protocol_it_was_written_at() -> None:
    held = staged()

    assert held.stage == StageRecord(
        name="type-map",
        active=("chart.embeddings", "chart.layers"),
        position=1,
    )
    assert manifest().stage is None


def test_a_stage_record_round_trips_through_the_manifest_text() -> None:
    held = staged()

    assert read_manifest(write_manifest(held)) == held
    assert read_manifest(write_manifest(manifest())).stage is None


def test_the_stage_is_not_a_reason_a_checkpoint_cannot_be_read() -> None:
    assert disagreements(manifest(), staged()) == ()
    assert disagreements(staged(), manifest()) == ()


def test_a_stage_record_refuses_what_no_protocol_could_have_written() -> None:
    with pytest.raises(dx.ValidationError):
        StageRecord(name="", active=("a",), position=0)

    with pytest.raises(dx.ValidationError):
        StageRecord(name="chart", active=("a",), position=-1)

    with pytest.raises(dx.ValidationError):
        StageRecord(name="chart", active=("",), position=0)

    with pytest.raises(dx.ValidationError):
        StageRecord(name="chart", active=("a", "a"), position=0)


def test_a_stage_that_trained_no_group_is_recorded_as_such() -> None:
    held = staged_manifest(manifest(), name="frozen", active=(), position=0)

    assert held.stage is not None
    assert held.stage.active == ()
