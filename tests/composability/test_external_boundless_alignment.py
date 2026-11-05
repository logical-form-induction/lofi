"""Boundless alignment composes with model-ready external causal pairs."""

from typing import Final, override

import torch
from torch import nn

from lofi import probe
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._das import AlignmentObjective
from lofi.stats import LabelPermutation

VARIABLE: Final = probe.TypeAt((1, 2))
POLICY: Final = probe.AlignmentPolicy("all", "exclude", contiguous=True)
TOKENIZER: Final = "boundless-tokenizer"
MODEL: Final = ComponentId("boundless-external-model")
SITE: Final = Site(MODEL, Role.RESIDUAL_STREAM, (0,))


class TinyModel(nn.Module):
    """Score a binary decision from one published token representation."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 2, bias=False)

        with torch.no_grad():
            self.hidden.weight.copy_(torch.eye(2))

    @override
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden(vectors)
        signal = hidden[:, 1, 0]

        return torch.stack((-signal, signal), dim=-1)


def score(
    output: probe.Output,
    variable: probe.HighLevelVariable,
    decisions: tuple[str | int, ...],
) -> torch.Tensor:
    """Return the model's scores in its declared decision order."""
    assert variable == VARIABLE
    assert decisions == ("A", "B")
    assert isinstance(output, torch.Tensor)

    return output


def pair(name: str, nuisance: float) -> probe.ModelCausalPair:
    """Return one confirmed pair carrying exact distinct model inputs."""
    record = probe.GeneratedPair(
        (name, "runs", "today"),
        ("D", "A", "X"),
        ("D", "B", "X"),
        {VARIABLE: "A"},
        {VARIABLE: "B"},
        VARIABLE,
        "tiny-grammar:boundless",
        probe.Provenance("grammar-counterfactual", "unit", 9),
    )
    alignment = probe.TokenizerAlignment(TOKENIZER, ((0,), (1,), (2,)), (), 3)
    aligned = probe.AlignedInput(record.tokens, (alignment,))
    generation = probe.CounterfactualGeneration(
        (record,),
        {record.id: aligned},
        (),
        probe.AlignmentManifest(POLICY, aligned.tokenizers, 1, 1),
    )
    base = torch.tensor([[[nuisance, 0.1], [-2.0, 0.5], [0.2, -0.1]]])
    variant = base.clone()
    variant[:, 1] = torch.tensor([2.0, -0.5])

    return probe.ModelCausalPair(
        record.id,
        probe.confirm_pair(record),
        generation.aligned_input(record.id).tokens,
        generation.manifest.policy,
        base_inputs={"vectors": base},
        variant_inputs={"vectors": variant},
        base_alignment=alignment,
        variant_alignment=alignment,
    )


def subject() -> tuple[TinyModel, probe.ExternalSubject]:
    """Return the external model under its public causal-subject adapter."""
    declared = probe.ExternalSiteMap(
        MODEL,
        (
            probe.ExternalSite(
                "hidden",
                Role.RESIDUAL_STREAM,
                0,
                SiteSpec(
                    ("batch", "position", "feature"),
                    DTypeClass.REAL,
                    differentiable=True,
                    access=Access.OBSERVE | Access.PATCH,
                ),
            ),
        ),
    )
    module = TinyModel()
    external = probe.ExternalModel(module, declared)

    return (
        module,
        probe.ExternalSubject(
            external,
            POLICY,
            TOKENIZER,
            {VARIABLE: ("A", "B")},
            score,
            batch_axis="batch",
            row_axis="position",
        ),
    )


def test_boundless_search_fits_through_the_external_model_ready_subject() -> None:
    module, causal = subject()
    pairs = tuple(
        pair(name, nuisance)
        for name, nuisance in zip(
            ("kim", "lee", "max", "sam"), (0.0, 0.1, -0.1, 0.2), strict=True
        )
    )
    before = tuple(
        causal.run(pairs, side).detach().clone()
        for side in (probe.Side.BASE, probe.Side.VARIANT)
    )
    search = probe.BoundlessSearch(
        boundaries=(0.0, 1.0),
        margin=1e-6,
        annealing=probe.Annealing(initial=1.0, final=0.05),
        schedule=probe.Schedule(epochs=6, learning_rate=0.1, batch_size=4, seed=2),
        null=LabelPermutation(replicates=1, seed=3),
        objective=AlignmentObjective.CROSS_ENTROPY,
    )
    alignment = probe.DistributedAlignment[probe.Observation, probe.ModelCausalPair](
        causal, SITE, VARIABLE, search=search
    ).fit(pairs)

    assert alignment.fitted is not None
    assert alignment.fitted.pairs == pairs
    assert alignment.fitted.boundaries[0] < alignment.fitted.boundaries[1]
    assert alignment.iia(pairs).rank == alignment.fitted.discrete_width
    assert all(
        torch.equal(causal.run(pairs, side), expected)
        for side, expected in zip(
            (probe.Side.BASE, probe.Side.VARIANT), before, strict=True
        )
    )
    assert module.training
    assert all(parameter.grad is None for parameter in module.parameters())
