"""Tests for the acquisition-declaration package surface."""

from tests.unit.spec.acquisition.test_active import a_document

from lofi.spec import acquisition
from lofi.spec.acquisition import active, collectors, constraints


def test_the_package_publishes_the_current_acquisition_declarations() -> None:
    assert set(acquisition.__all__) == (
        set(active.__all__) | set(collectors.__all__) | set(constraints.__all__)
    )

    for module in (active, collectors, constraints):
        for name in module.__all__:
            assert vars(acquisition)[name] is vars(module)[name]


def test_the_public_decoder_returns_immutable_acquisition_settings() -> None:
    settings = acquisition.decode_active_learning(a_document()).settings()

    resolved = settings.acquisition
    assert (resolved.kind, resolved.batch_size, resolved.allocate_by) == (
        "rc2.1",
        500,
        "item['structure']",
    )
    assert resolved.share_tolerance == 0.02
    assert resolved.constraints == (
        acquisition.UniquenessSettings("uniqueness", "item['verb']"),
    )
    assert settings.stopping == acquisition.AgreementStopSettings(0.8, "lower-bound")
    assert settings.collector == acquisition.BeadCollectorSettings("bead")
