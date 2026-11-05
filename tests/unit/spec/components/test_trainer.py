from tests.unit.spec.components.conftest import FiniteSampler, StepTrainer, declaration

from lofi.spec.components.run import RunSpec


def test_a_distance_shape_needs_no_parser_loss_or_featurizer_placeholder() -> None:
    run = RunSpec.model_validate(
        {
            "model": {
                "language": {"from": "data"},
                "encoders": {"terms": {"kind": "sized", "width": 8}},
            },
            "data": {"datasets": {"train": {"dataset": {"kind": "sample-dataset"}}}},
            "trainer": {"kind": "step-trainer", "steps": 4},
            "release": {"checkpoint": "resume-latest"},
            "parse": {"retain": "forest", "reachability": "all"},
            "type_sampler": {"kind": "finite-sampler"},
            "optimizer": {"kind": "adam-optimizer"},
            "device_policy": declaration()["device_policy"],
        }
    )

    assert run.model.parser is None
    assert run.loss is None
    assert run.data.datasets["train"].featurizer is None
    assert run.type_sampler == FiniteSampler()


def test_a_trainer_variant_that_declares_nothing_requires_and_refuses_nothing() -> None:
    trainer = StepTrainer(steps=4)

    assert trainer.required_roles() == frozenset()
    assert trainer.forbidden_roles() == frozenset()
    assert trainer.dataset_requirements() == {}
