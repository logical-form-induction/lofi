from types import MappingProxyType

import torch

from lofi._types import ReadoutId
from lofi.lf import THREE_WAY, ArgumentRenaming, InferenceLabel
from lofi.nn._values import LogDistribution
from lofi.nn.grounding import (
    CellMass,
    EvidenceLedger,
    GradientPath,
    GroundedBatch,
    GroundingInput,
    GroundingMetadata,
    GroundingQuery,
    InferenceGrounding,
    Phase,
)
from lofi.nn.sites import ComponentId, SiteMap


class Grounding:
    grounding_metadata = GroundingMetadata(
        "fixture",
        scorer="fixture",
        source="fixture",
        prover=None,
        prover_version=None,
        budget=None,
    )

    @property
    def labels(self) -> tuple[InferenceLabel, ...]:
        return THREE_WAY.labels

    @property
    def requires(self) -> frozenset[str]:
        return frozenset({"forms"})

    def ground(self, batch: GroundingInput) -> GroundedBatch:
        del batch
        values = torch.tensor([[0.75, 0.125, 0.125]]).log()
        return GroundedBatch(
            LogDistribution(values, ("pair", "label"), normalized_over="label"),
            labels=THREE_WAY.labels,
            complete=True,
            residual=torch.zeros(1),
            decided=torch.ones(1, dtype=torch.bool),
            cells=CellMass(torch.tensor([[0.75, 0.0, 0.25, 0.0]])),
            gradient_path=GradientPath.PATHWISE,
            baseline=None,
            auxiliary=None,
            ledger=EvidenceLedger.empty(),
            variance=None,
            reward=None,
        )

    def site_map(self) -> SiteMap:
        return SiteMap.custom(ComponentId("grounding"), {})

    def open_step(self) -> None:
        pass


class Pairs:
    premises = (ReadoutId("premise"),)
    hypotheses = (ReadoutId("hypothesis"),)
    alignments = ((ArgumentRenaming.identity(),),)
    labels = (InferenceLabel.ENTAILS,)


def test_the_grounding_port_has_one_batched_prediction_route() -> None:
    grounding = Grounding()

    assert isinstance(grounding, InferenceGrounding)
    assert grounding.labels == THREE_WAY.labels
    assert grounding.requires == frozenset({"forms"})
    assert dict(grounding.site_map()) == {}


def test_grounding_input_freezes_the_published_forms() -> None:
    batch = GroundingInput(
        MappingProxyType({}),
        choice_rows=MappingProxyType({}),
        pairs=Pairs(),
        query=GroundingQuery.FULL,
        sentences=None,
        phase=Phase.EVALUATE,
    )

    assert dict(batch.forms) == {}
    assert batch.phase is Phase.EVALUATE
