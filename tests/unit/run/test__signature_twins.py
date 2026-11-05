import json
from typing import TYPE_CHECKING, cast

from lofi.run import (
    MeasuredRole,
    PointManifest,
    RandomStreams,
    SignatureTwinSettings,
    SweepPoint,
    measure_point,
    resolve_device_policy,
)
from lofi.run._seed import ExperimentSeed
from lofi.run._signature_twins import signature_twin_augmenter
from lofi.run._training import plan_training
from lofi.spec import (
    PACKAGED_CONFIG_ROOT,
    RunSpec,
    RunSpecCodec,
    SourceComparisonPointSpec,
)
from lofi.train import prepare_training

if TYPE_CHECKING:
    from lofi._types import JsonValue


PACKAGED_CODEC = RunSpecCodec(
    search_path=(PACKAGED_CONFIG_ROOT / "run", PACKAGED_CONFIG_ROOT)
)


def _measured(point: SweepPoint, experiment_seed: ExperimentSeed) -> PointManifest:
    """Build one generated point and measure its production model graph."""
    declared = RunSpec.model_validate(point.config)
    derivation = experiment_seed.point(point.point_id, point.seed)
    blueprint = plan_training(declared, point=derivation)
    streams = RandomStreams(
        derivation.policy(accelerator=None, samplers=blueprint.sampler_streams)
    )
    streams.apply()
    prepared = prepare_training(
        blueprint, streams, resolve_device_policy(blueprint.run_spec)
    )
    declarations = blueprint.run_spec.model.checkpointed_components()
    owned = prepared.model.role_parameters()
    return measure_point(
        point,
        {
            role: MeasuredRole(cast("str", declarations[role].kind), owned[role])
            for role in sorted(owned)
        },
        corpus_fingerprint=blueprint.corpus_revision,
    )


def test_one_candidate_expands_to_observed_plus_three_trainable_twins() -> None:
    base = PACKAGED_CODEC(
        PACKAGED_CONFIG_ROOT / "run" / "synthetic-parser.yaml",
        extra={"model.language.from": "data"},
    )
    train = base.data.datasets["train"]
    base = base.with_(
        data=base.data.with_(
            datasets={
                **base.data.datasets,
                "train": train.with_(dataset=train.dataset.with_(seed="derived")),
            }
        )
    )
    declared = base.with_(
        comparison=SourceComparisonPointSpec(
            source="source",
            condition="primary",
            direction="bidirectional",
            twin="observed",
        )
    )
    config = cast("dict[str, JsonValue]", json.loads(declared.model_dump_json()))
    point = SweepPoint(config, seed=0)

    settings = SignatureTwinSettings(3, 17, "primary", "observed")
    expanded = signature_twin_augmenter(settings, ExperimentSeed(29))(point)

    assignments = tuple(
        cast("dict[str, JsonValue]", member.config["signature_assignment"])
        for member in expanded
    )
    assert len(expanded) == 4
    assert [
        cast("dict[str, JsonValue]", member.config["comparison"])["twin"]
        for member in expanded
    ] == ["observed", "twin-001", "twin-002", "twin-003"]
    assert len({json.dumps(value, sort_keys=True) for value in assignments}) == 4
    assert all(
        member.config["data"] == expanded[0].config["data"] for member in expanded
    )
    datasets = tuple(
        plan_training(RunSpec.model_validate(member.config)).checked_dataset("train")
        for member in expanded
    )
    assert len({dataset.declaration for dataset in datasets}) == 1
    assert len({dataset.source_grammar.declaration_digest for dataset in datasets}) == 1

    manifests = tuple(_measured(member, ExperimentSeed(29)) for member in expanded[:2])
    assert manifests[0].parameter_count == manifests[1].parameter_count > 0
    assert manifests[0].trainable_count == manifests[1].trainable_count > 0
    assert manifests[0].corpus_fingerprint == manifests[1].corpus_fingerprint
    assert (
        tuple(
            PointManifest.model_validate_json(manifest.model_dump_json())
            for manifest in manifests
        )
        == manifests
    )

    control = declared.with_(
        comparison=SourceComparisonPointSpec(
            source="source",
            condition="reversed-ground-truth",
            direction="bidirectional",
            twin="unexpanded",
        )
    )
    control_config = cast("dict[str, JsonValue]", json.loads(control.model_dump_json()))
    [observed_control] = signature_twin_augmenter(settings, ExperimentSeed(29))(
        SweepPoint(control_config, seed=0)
    )
    assert (
        cast("dict[str, JsonValue]", observed_control.config["comparison"])["twin"]
        == "observed"
    )
    assert observed_control.config["signature_assignment"] is not None
