import json
import math
from hashlib import sha256
from typing import TYPE_CHECKING, Final

import pytest
import torch
from tests.support.installed import install

from lofi.judgment import PARTICIPANT, DesignCodec, ReadDesign, log_likelihood
from lofi.judgment.synthetic_designs import (
    KNOWN_RANDOM_EFFECTS,
    KNOWN_RANDOM_EFFECTS_TAG,
    PRIOR_PREDICTIVE_REFERENCE,
    PRIOR_PREDICTIVE_REFERENCE_TAG,
    SCORE,
    KnownEffectsDesign,
    generating_spec,
)
from lofi.registry import UnknownTagError
from lofi.run.fits import RegisteredDesigns

if TYPE_CHECKING:
    from lofi.judgment import JudgmentDesign

PARTICIPANT_EFFECTS: Final = frozenset(
    {"participant_intercept_offsets", "participant_cutpoint_offsets"}
)


def columns_of(design: JudgmentDesign) -> tuple[object, ...]:
    codec = DesignCodec(
        design.response_levels,
        factors=tuple(design.factors),
        readouts=tuple(design.readouts),
    )

    return codec.decode(design)


@pytest.mark.parametrize(
    ("tag", "declared"),
    [
        (KNOWN_RANDOM_EFFECTS_TAG, KNOWN_RANDOM_EFFECTS),
        (PRIOR_PREDICTIVE_REFERENCE_TAG, PRIOR_PREDICTIVE_REFERENCE),
    ],
)
def test_a_registered_synthetic_design_resolves_to_its_declaration(
    tag: str, declared: KnownEffectsDesign
) -> None:
    read = RegisteredDesigns().read(tag)
    assert isinstance(read, ReadDesign)
    design = read.design

    assert design.n_rows == declared.participants * declared.items
    assert len(design.participant_levels) == declared.participants
    assert len(design.item_levels) == declared.items
    assert design.response_levels == declared.levels
    assert tuple(design.readouts) == (SCORE,)
    assert columns_of(design) == columns_of(declared())


@pytest.mark.parametrize(
    ("tag", "declared"),
    [
        (KNOWN_RANDOM_EFFECTS_TAG, KNOWN_RANDOM_EFFECTS),
        (PRIOR_PREDICTIVE_REFERENCE_TAG, PRIOR_PREDICTIVE_REFERENCE),
    ],
)
def test_a_registered_synthetic_design_records_what_it_was_drawn_at(
    tag: str, declared: KnownEffectsDesign
) -> None:
    provenance = RegisteredDesigns().read(tag).provenance
    sites = declared.program().sites(declared.generating())
    population = {
        site: float(values.reshape(-1)[0])
        for site, values in sites.items()
        if site not in PARTICIPANT_EFFECTS
    }
    digest = sha256(
        json.dumps(population, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()

    assert provenance.tag == tag
    assert provenance.seed == declared.seed
    assert (provenance.participants, provenance.items, provenance.levels) == (
        declared.participants,
        declared.items,
        declared.levels,
    )
    assert provenance.population_digest == digest == declared.population_digest
    assert provenance.checkpoint is None
    assert provenance.checkpoint_dtype is None
    assert provenance.unigram_digest is None
    assert provenance.corpus_digests == {}


def test_a_synthetic_design_reproduces_exactly_from_its_seed() -> None:
    first = KnownEffectsDesign(
        "draw", participants=6, items=8, levels=5, seed=11, latent_range=(-2.5, 2.5)
    )
    again = KnownEffectsDesign(
        "draw", participants=6, items=8, levels=5, seed=11, latent_range=(-2.5, 2.5)
    )
    other = KnownEffectsDesign(
        "draw", participants=6, items=8, levels=5, seed=12, latent_range=(-2.5, 2.5)
    )

    assert columns_of(first()) == columns_of(again())
    assert torch.equal(first.generating(), again.generating())
    assert not torch.equal(first().responses, other().responses)
    assert not torch.equal(first.generating(), other.generating())


def test_the_generating_vector_is_laid_out_as_the_generating_program() -> None:
    declared = KnownEffectsDesign(
        "draw", participants=6, items=8, levels=5, seed=3, latent_range=(-2.5, 2.5)
    )
    program = declared.program()
    sites = program.sites(declared.generating())

    assert declared.generating().shape == (program.size,)
    assert sites["participant_intercept_offsets"].shape == (6,)
    assert sites["participant_cutpoint_offsets"].numel() == 6 * 4
    assert declared.spec == generating_spec(5)
    assert declared.spec.groups == (PARTICIPANT,)


def test_the_responses_follow_the_known_participant_effects() -> None:
    declared = KNOWN_RANDOM_EFFECTS
    design = declared()
    program = declared.program()
    generating = declared.generating()
    sites = program.sites(generating)
    offsets = sites["participant_intercept_offsets"]
    responses = design.responses.to(torch.float64)
    means = torch.stack(
        [
            responses[design.participants == participant].mean()
            for participant in range(declared.participants)
        ]
    )
    correlation = torch.corrcoef(torch.stack((offsets, means)))[0, 1]
    start = 0

    for slot in program.layout:
        if slot.site == "participant_intercept_offsets":
            break

        start += slot.size

    flattened = generating.clone()
    flattened[start : start + declared.participants] = 0.0

    assert correlation > 0.5
    assert log_likelihood(program, generating).sum() > (
        log_likelihood(program, flattened).sum()
    )


def test_an_unregistered_synthetic_design_is_refused() -> None:
    with pytest.raises(UnknownTagError, match="known-random-effectz"):
        RegisteredDesigns().read("known-random-effectz")


def test_the_simulator_compiles_its_linking_through_the_installed_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declared = KnownEffectsDesign(
        "draw", participants=6, items=8, levels=5, seed=3, latent_range=(-2.5, 2.5)
    )
    assert declared().responses.numel() == 6 * 8

    install(monkeypatch, "linking", ())

    with pytest.raises(ValueError, match="this compiler compiles \\(\\)"):
        declared()


def test_a_design_at_declared_scores_rates_every_item_at_its_score() -> None:
    scores = (-2.0, -1.0, 0.0, 1.0, 2.0)
    declared = KnownEffectsDesign(
        "at-scores",
        participants=4,
        items=5,
        levels=3,
        seed=7,
        latent_range=(-1.5, 1.5),
        scores=scores,
    )
    design = declared()
    read = design.readouts[SCORE]

    assert design.n_rows == 4 * 5
    assert {
        design.item_levels[int(item)]: float(value)
        for item, value in zip(design.items, read, strict=True)
    } == {f"item-{index}": score for index, score in enumerate(scores)}
    assert columns_of(design) == columns_of(
        KnownEffectsDesign(
            "at-scores",
            participants=4,
            items=5,
            levels=3,
            seed=7,
            latent_range=(-1.5, 1.5),
            scores=scores,
        )()
    )


def test_declared_scores_leave_the_participant_draws_to_the_seed() -> None:
    drawn = KnownEffectsDesign(
        "draw", participants=6, items=8, levels=5, seed=3, latent_range=(-2.5, 2.5)
    )
    scores = tuple(float(value) for value in drawn().readouts[SCORE][:8])
    declared = KnownEffectsDesign(
        "draw",
        participants=6,
        items=8,
        levels=5,
        seed=3,
        latent_range=(-2.5, 2.5),
        scores=scores,
    )
    sites = declared.program().sites(declared.generating())
    other = KnownEffectsDesign(
        "draw",
        participants=6,
        items=8,
        levels=5,
        seed=4,
        latent_range=(-2.5, 2.5),
        scores=scores,
    )

    assert sites["participant_intercept_offsets"].shape == (6,)
    assert not torch.equal(declared.generating(), other.generating())


def test_declared_scores_name_every_item_once() -> None:
    with pytest.raises(ValueError, match="3 scores for 4 items"):
        KnownEffectsDesign(
            "draw",
            participants=2,
            items=4,
            levels=3,
            seed=0,
            latent_range=(-1.5, 1.5),
            scores=(0.0, 1.0, 2.0),
        )


def test_the_registered_designs_space_their_cutpoints_one_unit_apart() -> None:
    for declared in (KNOWN_RANDOM_EFFECTS, PRIOR_PREDICTIVE_REFERENCE):
        sites = declared.program().sites(declared.generating())

        assert declared.latent_range == (-2.5, 4.5)
        assert float(sites["cutpoint_base"].reshape(-1)[0]) == pytest.approx(-2.5)
        assert torch.nn.functional.softplus(
            sites["cutpoint_increments"]
        ).tolist() == pytest.approx([1.0] * (declared.levels - 1))


@pytest.mark.parametrize(
    ("levels", "latent_range", "cutpoints"),
    [
        (3, (-3.0, 3.0), (-1.0, 1.0)),
        (4, (-2.0, 6.0), (0.0, 2.0, 4.0)),
        (2, (-1.0, 3.0), (1.0,)),
    ],
)
def test_the_cutpoints_divide_the_declared_range_into_equal_intervals(
    levels: int, latent_range: tuple[float, float], cutpoints: tuple[float, ...]
) -> None:
    declared = KnownEffectsDesign(
        "spaced",
        participants=3,
        items=4,
        levels=levels,
        seed=5,
        latent_range=latent_range,
    )
    sites = declared.program().sites(declared.generating())
    base = float(sites["cutpoint_base"].reshape(-1)[0])
    rises = torch.nn.functional.softplus(sites["cutpoint_increments"]).tolist()
    placed: list[float] = []
    for rise in rises:
        placed.append((placed[-1] if placed else base) + rise)

    assert placed == pytest.approx(cutpoints)


def test_a_three_level_design_answers_on_every_level() -> None:
    declared = KnownEffectsDesign(
        "three", participants=6, items=20, levels=3, seed=2, latent_range=(-3.0, 3.0)
    )

    assert set(declared().responses.tolist()) == {0, 1, 2}


def test_the_population_digest_identifies_the_declared_range() -> None:
    first = KnownEffectsDesign(
        "draw", participants=2, items=3, levels=3, seed=0, latent_range=(-3.0, 3.0)
    )
    other = KnownEffectsDesign(
        "draw", participants=2, items=3, levels=3, seed=0, latent_range=(-2.0, 2.0)
    )

    assert first.population_digest != other.population_digest
    assert first.read().provenance.population_digest == first.population_digest


@pytest.mark.parametrize("latent_range", [(1.0, 1.0), (2.0, -2.0), (0.0, math.inf)])
def test_a_range_without_positive_finite_width_is_refused(
    latent_range: tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="latent range"):
        KnownEffectsDesign(
            "draw",
            participants=2,
            items=3,
            levels=3,
            seed=0,
            latent_range=latent_range,
        )
