from lofi.spec import readouts
from lofi.spec.readouts import (
    DecodedMeaningSettings,
    DecodedMeaningSpec,
    MeaningInductionSettings,
    MeaningInductionSpec,
    ProofPushforwardSettings,
    ProofTermMeaningSettings,
    ProofTermMeaningSpec,
    ReadoutMass,
    ReadoutSource,
    TargetSource,
    TermsReadoutSettings,
    TermsReadoutSpec,
)


def test_the_package_publishes_its_declarations() -> None:
    assert readouts.__all__ == [
        "DecodedMeaningSettings",
        "DecodedMeaningSpec",
        "MeaningInductionSettings",
        "MeaningInductionSpec",
        "ProofPushforwardSettings",
        "ProofTermMeaningSettings",
        "ProofTermMeaningSpec",
        "ReadoutMass",
        "ReadoutSource",
        "TargetSource",
        "TermsReadoutSettings",
        "TermsReadoutSpec",
    ]
    assert readouts.DecodedMeaningSpec is DecodedMeaningSpec
    assert readouts.DecodedMeaningSettings is DecodedMeaningSettings
    assert readouts.MeaningInductionSettings is MeaningInductionSettings
    assert readouts.MeaningInductionSpec is MeaningInductionSpec
    assert readouts.ProofTermMeaningSpec is ProofTermMeaningSpec
    assert readouts.ProofPushforwardSettings is ProofPushforwardSettings
    assert readouts.ProofTermMeaningSettings is ProofTermMeaningSettings
    assert readouts.TermsReadoutSettings is TermsReadoutSettings
    assert readouts.ReadoutMass is ReadoutMass
    assert readouts.ReadoutSource is ReadoutSource
    assert readouts.TargetSource is TargetSource
    assert readouts.TermsReadoutSpec is TermsReadoutSpec
